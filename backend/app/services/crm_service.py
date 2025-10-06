"""
Zoho CRM Integration Service

Provides authentication, data synchronization, and CRUD operations
for Zoho CRM integration in the Voice AI Agent system.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlencode
import httpx
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from ..core.config import settings
from ..models.crm import CRMAccount, CRMSyncJob, CRMFieldMapping, Lead, LeadActivity
from ..core.database import get_db

logger = logging.getLogger(__name__)


class ZohoCRMError(Exception):
    """Base exception for Zoho CRM operations."""
    pass


class ZohoAuthError(ZohoCRMError):
    """Authentication related errors."""
    pass


class ZohoAPIError(ZohoCRMError):
    """API operation errors."""
    def __init__(self, message: str, status_code: int = None, response_data: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class ZohoCRMService:
    """
    Service class for Zoho CRM integration.
    
    Handles OAuth authentication, token management, and all CRM operations.
    """
    
    def __init__(self):
        self.client_id = settings.ZOHO_CLIENT_ID
        self.client_secret = settings.ZOHO_CLIENT_SECRET
        self.redirect_uri = settings.ZOHO_REDIRECT_URI
        self.api_domain = settings.ZOHO_API_DOMAIN
        self.accounts_url = settings.ZOHO_ACCOUNTS_URL
        self.sandbox = settings.ZOHO_SANDBOX_MODE
        
        self._http_client: Optional[httpx.AsyncClient] = None
        self._current_account: Optional[CRMAccount] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.CRM_TIMEOUT),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._http_client:
            await self._http_client.aclose()
    
    def get_auth_url(self, state: str = None) -> str:
        """
        Generate Zoho OAuth authorization URL.
        
        Args:
            state: Optional state parameter for security
            
        Returns:
            Authorization URL for Zoho OAuth flow
        """
        params = {
            "scope": "ZohoCRM.modules.ALL,ZohoCRM.settings.ALL,ZohoCRM.users.READ",
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "access_type": "offline",
        }
        
        if state:
            params["state"] = state
            
        return f"{self.accounts_url}/oauth/v2/auth?{urlencode(params)}"
    
    async def exchange_code_for_tokens(
        self, 
        code: str, 
        db: AsyncSession
    ) -> CRMAccount:
        """
        Exchange authorization code for access and refresh tokens.
        
        Args:
            code: Authorization code from OAuth callback
            db: Database session
            
        Returns:
            CRMAccount instance with tokens
            
        Raises:
            ZohoAuthError: If token exchange fails
        """
        if not self._http_client:
            raise RuntimeError("Service not initialized. Use as async context manager.")
        
        token_data = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
            "code": code,
        }
        
        try:
            response = await self._http_client.post(
                f"{self.accounts_url}/oauth/v2/token",
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            
            tokens = response.json()
            
            if "error" in tokens:
                raise ZohoAuthError(f"Token exchange failed: {tokens['error']}")
            
            # Get organization details
            org_info = await self._get_organization_info(tokens["access_token"])
            
            # Create or update CRM account
            account = await self._save_crm_account(db, tokens, org_info)
            
            logger.info(f"Successfully authenticated with Zoho CRM org: {org_info['id']}")
            return account
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during token exchange: {e}")
            raise ZohoAuthError(f"Token exchange failed: {e.response.text}")
        except Exception as e:
            logger.error(f"Unexpected error during token exchange: {e}")
            raise ZohoAuthError(f"Token exchange failed: {str(e)}")
    
    async def refresh_access_token(self, account: CRMAccount, db: AsyncSession) -> str:
        """
        Refresh access token using refresh token.
        
        Args:
            account: CRMAccount with refresh token
            db: Database session
            
        Returns:
            New access token
            
        Raises:
            ZohoAuthError: If token refresh fails
        """
        if not account.refresh_token:
            raise ZohoAuthError("No refresh token available")
        
        if not self._http_client:
            raise RuntimeError("Service not initialized. Use as async context manager.")
        
        refresh_data = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": account.refresh_token,
        }
        
        try:
            response = await self._http_client.post(
                f"{self.accounts_url}/oauth/v2/token",
                data=refresh_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            
            tokens = response.json()
            
            if "error" in tokens:
                raise ZohoAuthError(f"Token refresh failed: {tokens['error']}")
            
            # Update account with new tokens
            account.access_token = tokens["access_token"]
            account.token_expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=tokens.get("expires_in", 3600)
            )
            
            await db.commit()
            
            logger.info(f"Successfully refreshed access token for org: {account.zoho_org_id}")
            return tokens["access_token"]
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during token refresh: {e}")
            raise ZohoAuthError(f"Token refresh failed: {e.response.text}")
        except Exception as e:
            logger.error(f"Unexpected error during token refresh: {e}")
            raise ZohoAuthError(f"Token refresh failed: {str(e)}")
    
    async def ensure_valid_token(self, account: CRMAccount, db: AsyncSession) -> str:
        """
        Ensure account has valid access token, refreshing if necessary.
        
        Args:
            account: CRMAccount instance
            db: Database session
            
        Returns:
            Valid access token
        """
        now = datetime.now(timezone.utc)
        
        # Check if token needs refresh (5 minute buffer)
        if (not account.access_token or 
            not account.token_expires_at or 
            account.token_expires_at <= now + timedelta(minutes=5)):
            
            return await self.refresh_access_token(account, db)
        
        return account.access_token
    
    async def sync_lead_to_zoho(
        self, 
        lead: Lead, 
        account: CRMAccount, 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Synchronize a lead to Zoho CRM.
        
        Args:
            lead: Lead instance to sync
            account: CRMAccount for authentication
            db: Database session
            
        Returns:
            Sync result with Zoho lead ID
            
        Raises:
            ZohoAPIError: If sync operation fails
        """
        access_token = await self.ensure_valid_token(account, db)
        
        # Transform lead data to Zoho format
        zoho_data = await self._transform_lead_to_zoho(lead, account, db)
        
        try:
            if lead.zoho_lead_id:
                # Update existing lead
                result = await self._update_zoho_lead(
                    lead.zoho_lead_id, zoho_data, access_token
                )
            else:
                # Create new lead
                result = await self._create_zoho_lead(zoho_data, access_token)
                
                if result.get("data") and len(result["data"]) > 0:
                    lead.zoho_lead_id = result["data"][0]["details"]["id"]
            
            # Update sync status
            lead.last_synced_at = datetime.now(timezone.utc)
            lead.sync_status = "synced"
            lead.sync_error = None
            
            await db.commit()
            
            logger.info(f"Successfully synced lead {lead.id} to Zoho CRM")
            return result
            
        except Exception as e:
            # Update error status
            lead.sync_status = "error"
            lead.sync_error = str(e)
            await db.commit()
            
            logger.error(f"Failed to sync lead {lead.id} to Zoho: {e}")
            raise
    
    async def sync_leads_from_zoho(
        self, 
        account: CRMAccount, 
        db: AsyncSession,
        since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Sync leads from Zoho CRM to local database.
        
        Args:
            account: CRMAccount for authentication
            db: Database session
            since: Only sync leads modified since this datetime
            
        Returns:
            Sync summary with counts and results
        """
        access_token = await self.ensure_valid_token(account, db)
        
        # Create sync job
        sync_job = CRMSyncJob(
            account_id=account.id,
            job_type="sync_leads",
            status="running",
            started_at=datetime.now(timezone.utc)
        )
        db.add(sync_job)
        await db.commit()
        
        try:
            # Get leads from Zoho
            zoho_leads = await self._get_zoho_leads(access_token, since)
            
            processed = 0
            successful = 0
            failed = 0
            
            for zoho_lead in zoho_leads:
                processed += 1
                try:
                    await self._upsert_lead_from_zoho(zoho_lead, account, db)
                    successful += 1
                except Exception as e:
                    failed += 1
                    logger.error(f"Failed to process Zoho lead {zoho_lead.get('id')}: {e}")
            
            # Update sync job
            sync_job.status = "completed"
            sync_job.completed_at = datetime.now(timezone.utc)
            sync_job.records_processed = processed
            sync_job.records_successful = successful
            sync_job.records_failed = failed
            
            await db.commit()
            
            result = {
                "processed": processed,
                "successful": successful,
                "failed": failed,
                "job_id": str(sync_job.id)
            }
            
            logger.info(f"Completed lead sync from Zoho: {result}")
            return result
            
        except Exception as e:
            # Update sync job with error
            sync_job.status = "failed"
            sync_job.completed_at = datetime.now(timezone.utc)
            sync_job.error_message = str(e)
            await db.commit()
            
            logger.error(f"Lead sync from Zoho failed: {e}")
            raise ZohoAPIError(f"Lead sync failed: {str(e)}")
    
    async def create_appointment_in_zoho(
        self,
        lead: Lead,
        appointment_data: Dict[str, Any],
        account: CRMAccount,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Create an appointment/event in Zoho CRM.
        
        Args:
            lead: Lead associated with appointment
            appointment_data: Appointment details
            account: CRMAccount for authentication
            db: Database session
            
        Returns:
            Created event details from Zoho
        """
        access_token = await self.ensure_valid_token(account, db)
        
        # Transform appointment data to Zoho event format
        zoho_event = {
            "Event_Title": appointment_data.get("title", "Consultation"),
            "Start_DateTime": appointment_data["start_time"].isoformat(),
            "End_DateTime": appointment_data["end_time"].isoformat(),
            "Description": appointment_data.get("description", ""),
            "Location": appointment_data.get("location", ""),
        }
        
        # Link to lead if available
        if lead.zoho_lead_id:
            zoho_event["What_Id"] = lead.zoho_lead_id
        
        return await self._create_zoho_event(zoho_event, access_token)
    
    # Private helper methods
    
    async def _get_organization_info(self, access_token: str) -> Dict[str, Any]:
        """Get organization information from Zoho."""
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
        
        response = await self._http_client.get(
            f"{self.api_domain}/crm/v2/org",
            headers=headers
        )
        response.raise_for_status()
        
        result = response.json()
        if "org" in result and len(result["org"]) > 0:
            return result["org"][0]
        
        raise ZohoAuthError("Could not retrieve organization information")
    
    async def _save_crm_account(
        self,
        db: AsyncSession,
        tokens: Dict[str, Any],
        org_info: Dict[str, Any]
    ) -> CRMAccount:
        """Save or update CRM account with tokens."""
        
        # Check if account already exists
        stmt = select(CRMAccount).where(CRMAccount.zoho_org_id == org_info["id"])
        result = await db.execute(stmt)
        account = result.scalar_one_or_none()
        
        expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=tokens.get("expires_in", 3600)
        )
        
        if account:
            # Update existing account
            account.access_token = tokens["access_token"]
            account.refresh_token = tokens.get("refresh_token", account.refresh_token)
            account.token_expires_at = expires_at
            account.is_active = True
        else:
            # Create new account
            account = CRMAccount(
                name=org_info.get("company_name", "Unknown"),
                zoho_org_id=org_info["id"],
                access_token=tokens["access_token"],
                refresh_token=tokens.get("refresh_token"),
                token_expires_at=expires_at,
                api_domain=self.api_domain,
                is_active=True,
                sandbox_mode=self.sandbox
            )
            db.add(account)
        
        await db.commit()
        return account
    
    async def _transform_lead_to_zoho(
        self,
        lead: Lead,
        account: CRMAccount,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Transform internal lead to Zoho CRM format."""
        
        # Get field mappings for the account
        stmt = select(CRMFieldMapping).where(
            CRMFieldMapping.account_id == account.id,
            CRMFieldMapping.module_name == "Leads"
        )
        result = await db.execute(stmt)
        mappings = {m.internal_field: m.crm_field for m in result.scalars()}
        
        # Base lead data
        zoho_data = {
            mappings.get("first_name", "First_Name"): lead.first_name,
            mappings.get("last_name", "Last_Name"): lead.last_name,
            mappings.get("email", "Email"): lead.email,
            mappings.get("phone", "Phone"): lead.phone,
            mappings.get("company", "Company"): lead.company,
            mappings.get("title", "Designation"): lead.title,
            mappings.get("lead_source", "Lead_Source"): lead.source,
            mappings.get("lead_status", "Lead_Status"): lead.status.title().replace("_", " "),
        }
        
        # Add custom fields
        if lead.practice_areas:
            zoho_data[mappings.get("practice_areas", "Practice_Areas")] = ", ".join(lead.practice_areas)
        
        if lead.legal_issue:
            zoho_data[mappings.get("legal_issue", "Legal_Issue")] = lead.legal_issue
        
        if lead.conversation_summary:
            zoho_data[mappings.get("description", "Description")] = lead.conversation_summary
        
        # Remove None values
        return {k: v for k, v in zoho_data.items() if v is not None}
    
    async def _create_zoho_lead(
        self,
        lead_data: Dict[str, Any],
        access_token: str
    ) -> Dict[str, Any]:
        """Create lead in Zoho CRM."""
        headers = {
            "Authorization": f"Zoho-oauthtoken {access_token}",
            "Content-Type": "application/json"
        }
        
        payload = {"data": [lead_data]}
        
        response = await self._http_client.post(
            f"{self.api_domain}/crm/v2/Leads",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 201:
            raise ZohoAPIError(
                f"Failed to create lead in Zoho",
                response.status_code,
                response.json()
            )
        
        return response.json()
    
    async def _update_zoho_lead(
        self,
        lead_id: str,
        lead_data: Dict[str, Any],
        access_token: str
    ) -> Dict[str, Any]:
        """Update lead in Zoho CRM."""
        headers = {
            "Authorization": f"Zoho-oauthtoken {access_token}",
            "Content-Type": "application/json"
        }
        
        payload = {"data": [lead_data]}
        
        response = await self._http_client.put(
            f"{self.api_domain}/crm/v2/Leads/{lead_id}",
            headers=headers,
            json=payload
        )
        
        if response.status_code not in [200, 202]:
            raise ZohoAPIError(
                f"Failed to update lead in Zoho",
                response.status_code,
                response.json()
            )
        
        return response.json()
    
    async def _get_zoho_leads(
        self,
        access_token: str,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get leads from Zoho CRM."""
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
        
        params = {"per_page": 200}
        if since:
            params["modified_since"] = since.isoformat()
        
        all_leads = []
        page = 1
        
        while True:
            params["page"] = page
            
            response = await self._http_client.get(
                f"{self.api_domain}/crm/v2/Leads",
                headers=headers,
                params=params
            )
            
            if response.status_code != 200:
                raise ZohoAPIError(
                    f"Failed to fetch leads from Zoho",
                    response.status_code,
                    response.json()
                )
            
            result = response.json()
            leads = result.get("data", [])
            
            if not leads:
                break
            
            all_leads.extend(leads)
            
            # Check if there are more pages
            info = result.get("info", {})
            if not info.get("more_records", False):
                break
            
            page += 1
        
        return all_leads
    
    async def _upsert_lead_from_zoho(
        self,
        zoho_lead: Dict[str, Any],
        account: CRMAccount,
        db: AsyncSession
    ) -> None:
        """Create or update local lead from Zoho data."""
        
        zoho_id = zoho_lead["id"]
        
        # Check if lead already exists
        stmt = select(Lead).where(Lead.zoho_lead_id == zoho_id)
        result = await db.execute(stmt)
        lead = result.scalar_one_or_none()
        
        # Transform Zoho data to internal format
        lead_data = {
            "first_name": zoho_lead.get("First_Name", ""),
            "last_name": zoho_lead.get("Last_Name", ""),
            "email": zoho_lead.get("Email"),
            "phone": zoho_lead.get("Phone"),
            "company": zoho_lead.get("Company"),
            "title": zoho_lead.get("Designation"),
            "source": zoho_lead.get("Lead_Source", "crm_import"),
            "status": zoho_lead.get("Lead_Status", "new").lower().replace(" ", "_"),
            "zoho_lead_id": zoho_id,
            "last_synced_at": datetime.now(timezone.utc),
            "sync_status": "synced"
        }
        
        if lead:
            # Update existing lead
            for key, value in lead_data.items():
                if value is not None:
                    setattr(lead, key, value)
        else:
            # Create new lead
            lead = Lead(**lead_data)
            db.add(lead)
        
        await db.commit()
    
    async def _create_zoho_event(
        self,
        event_data: Dict[str, Any],
        access_token: str
    ) -> Dict[str, Any]:
        """Create event in Zoho CRM."""
        headers = {
            "Authorization": f"Zoho-oauthtoken {access_token}",
            "Content-Type": "application/json"
        }
        
        payload = {"data": [event_data]}
        
        response = await self._http_client.post(
            f"{self.api_domain}/crm/v2/Events",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 201:
            raise ZohoAPIError(
                f"Failed to create event in Zoho",
                response.status_code,
                response.json()
            )
        
        return response.json()


# Singleton instance
crm_service = ZohoCRMService()