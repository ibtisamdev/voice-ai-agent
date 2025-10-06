"""
Tests for CRM Integration (Phase 3)

Test suite for Zoho CRM integration, lead management,
and related functionality.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from httpx import AsyncClient

from backend.app.models.crm import CRMAccount, Lead, LeadActivity, Appointment
from backend.app.services.crm_service import ZohoCRMService, ZohoAuthError, ZohoAPIError
from backend.app.services.lead_service import LeadService


@pytest.fixture
def mock_crm_account():
    """Mock CRM account for testing."""
    return CRMAccount(
        id=uuid4(),
        name="Test Law Firm",
        zoho_org_id="test_org_123",
        access_token="test_access_token",
        refresh_token="test_refresh_token",
        token_expires_at=datetime.now(timezone.utc),
        api_domain="https://www.zohoapis.com",
        is_active=True,
        sandbox_mode=True
    )


@pytest.fixture
def mock_lead():
    """Mock lead for testing."""
    return Lead(
        id=uuid4(),
        first_name="John",
        last_name="Doe",
        email="john.doe@example.com",
        phone="+1234567890",
        company="Acme Corp",
        source="phone_call",
        status="new",
        lead_score=75,
        priority="high",
        practice_areas=["personal_injury"],
        legal_issue="Car accident claim",
        urgency_level="high"
    )


class TestZohoCRMService:
    """Test suite for Zoho CRM Service."""
    
    @pytest.mark.asyncio
    async def test_get_auth_url(self):
        """Test generation of OAuth authorization URL."""
        service = ZohoCRMService()
        
        auth_url = service.get_auth_url("test_state")
        
        assert "oauth/v2/auth" in auth_url
        assert "client_id" in auth_url
        assert "response_type=code" in auth_url
        assert "state=test_state" in auth_url
    
    @pytest.mark.asyncio
    async def test_exchange_code_for_tokens_success(self, mock_db: AsyncSession):
        """Test successful token exchange."""
        service = ZohoCRMService()
        
        # Mock HTTP responses
        mock_token_response = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
            "expires_in": 3600
        }
        
        mock_org_response = {
            "org": [{
                "id": "org_123",
                "company_name": "Test Firm"
            }]
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.return_value.json.return_value = mock_token_response
            mock_client.return_value.__aenter__.return_value.get.return_value.json.return_value = mock_org_response
            mock_client.return_value.__aenter__.return_value.post.return_value.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.get.return_value.raise_for_status = MagicMock()
            
            async with service:
                account = await service.exchange_code_for_tokens("test_code", mock_db)
            
            assert account.access_token == "new_access_token"
            assert account.refresh_token == "new_refresh_token"
            assert account.zoho_org_id == "org_123"
    
    @pytest.mark.asyncio
    async def test_exchange_code_for_tokens_failure(self, mock_db: AsyncSession):
        """Test failed token exchange."""
        service = ZohoCRMService()
        
        mock_error_response = {
            "error": "invalid_code",
            "error_description": "Invalid authorization code"
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.return_value.json.return_value = mock_error_response
            mock_client.return_value.__aenter__.return_value.post.return_value.raise_for_status = MagicMock()
            
            async with service:
                with pytest.raises(ZohoAuthError):
                    await service.exchange_code_for_tokens("invalid_code", mock_db)
    
    @pytest.mark.asyncio
    async def test_refresh_access_token(self, mock_db: AsyncSession, mock_crm_account):
        """Test access token refresh."""
        service = ZohoCRMService()
        
        mock_token_response = {
            "access_token": "refreshed_access_token",
            "expires_in": 3600
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.return_value.json.return_value = mock_token_response
            mock_client.return_value.__aenter__.return_value.post.return_value.raise_for_status = MagicMock()
            
            async with service:
                new_token = await service.refresh_access_token(mock_crm_account, mock_db)
            
            assert new_token == "refreshed_access_token"
            assert mock_crm_account.access_token == "refreshed_access_token"
    
    @pytest.mark.asyncio
    async def test_sync_lead_to_zoho_create(self, mock_db: AsyncSession, mock_crm_account, mock_lead):
        """Test syncing new lead to Zoho CRM."""
        service = ZohoCRMService()
        
        mock_create_response = {
            "data": [{
                "details": {
                    "id": "zoho_lead_123"
                }
            }]
        }
        
        with patch.object(service, 'ensure_valid_token', return_value="valid_token"), \
             patch.object(service, '_transform_lead_to_zoho', return_value={"First_Name": "John"}), \
             patch.object(service, '_create_zoho_lead', return_value=mock_create_response):
            
            async with service:
                result = await service.sync_lead_to_zoho(mock_lead, mock_crm_account, mock_db)
            
            assert mock_lead.zoho_lead_id == "zoho_lead_123"
            assert mock_lead.sync_status == "synced"
            assert mock_lead.sync_error is None
    
    @pytest.mark.asyncio
    async def test_sync_lead_to_zoho_update(self, mock_db: AsyncSession, mock_crm_account, mock_lead):
        """Test updating existing lead in Zoho CRM."""
        service = ZohoCRMService()
        mock_lead.zoho_lead_id = "existing_zoho_id"
        
        mock_update_response = {
            "data": [{
                "details": {
                    "id": "existing_zoho_id"
                }
            }]
        }
        
        with patch.object(service, 'ensure_valid_token', return_value="valid_token"), \
             patch.object(service, '_transform_lead_to_zoho', return_value={"First_Name": "John"}), \
             patch.object(service, '_update_zoho_lead', return_value=mock_update_response):
            
            async with service:
                result = await service.sync_lead_to_zoho(mock_lead, mock_crm_account, mock_db)
            
            assert mock_lead.sync_status == "synced"
            assert mock_lead.sync_error is None


class TestLeadService:
    """Test suite for Lead Service."""
    
    @pytest.mark.asyncio
    async def test_create_lead_from_conversation(self, mock_db: AsyncSession):
        """Test creating lead from conversation data."""
        service = LeadService()
        
        conversation_data = {
            "first_name": "Jane",
            "last_name": "Smith",
            "email": "jane.smith@example.com",
            "phone": "+1987654321",
            "company": "Smith & Associates",
            "legal_issue": "Divorce proceedings",
            "practice_areas": ["family_law"],
            "urgency_level": "normal",
            "call_duration": 180,
            "call_quality_score": 0.9,
            "summary": "Client needs help with divorce case"
        }
        
        with patch.object(service, 'auto_assign_lead') as mock_assign, \
             patch.object(service, 'add_lead_activity') as mock_activity:
            
            lead = await service.create_lead_from_conversation(
                conversation_data, 
                "session_123", 
                mock_db
            )
            
            assert lead.first_name == "Jane"
            assert lead.last_name == "Smith"
            assert lead.email == "jane.smith@example.com"
            assert lead.source == "phone_call"
            assert lead.channel == "inbound"
            assert lead.lead_score > 0
            assert lead.priority in ["low", "medium", "high", "urgent"]
    
    @pytest.mark.asyncio
    async def test_calculate_lead_score(self):
        """Test lead score calculation."""
        service = LeadService()
        
        # High-value lead
        high_value_lead = Lead(
            first_name="John",
            last_name="Doe",
            email="john@example.com",
            phone="+1234567890",
            company="Big Corp",
            practice_areas=["personal_injury"],
            urgency_level="high",
            call_duration=300,
            call_quality_score=0.9
        )
        
        score = service.scoring_service.calculate_lead_score(high_value_lead)
        assert score >= 70  # Should be high score
        
        # Low-value lead
        low_value_lead = Lead(
            first_name="Jane",
            last_name="Doe",
            urgency_level="low",
            call_duration=30
        )
        
        score = service.scoring_service.calculate_lead_score(low_value_lead)
        assert score <= 40  # Should be low score
    
    @pytest.mark.asyncio
    async def test_update_lead_score(self, mock_db: AsyncSession, mock_lead):
        """Test lead score update."""
        service = LeadService()
        mock_lead.lead_score = 50
        
        with patch.object(service.scoring_service, 'calculate_lead_score', return_value=80):
            new_score = await service.update_lead_score(mock_lead, mock_db)
            
            assert new_score == 80
            assert mock_lead.lead_score == 80
    
    @pytest.mark.asyncio
    async def test_add_lead_activity(self, mock_db: AsyncSession):
        """Test adding activity to lead."""
        service = LeadService()
        lead_id = uuid4()
        
        activity = await service.add_lead_activity(
            lead_id=lead_id,
            activity_type="call",
            direction="inbound",
            status="completed",
            description="Initial consultation call",
            duration=15,
            call_outcome="answered",
            db=mock_db
        )
        
        assert activity.lead_id == lead_id
        assert activity.activity_type == "call"
        assert activity.direction == "inbound"
        assert activity.status == "completed"
        assert activity.duration == 15
    
    @pytest.mark.asyncio
    async def test_get_leads_for_follow_up(self, mock_db: AsyncSession):
        """Test getting leads that need follow-up."""
        service = LeadService()
        
        # Mock leads that need follow-up
        past_time = datetime.now(timezone.utc) - timezone.timedelta(hours=1)
        
        mock_leads = [
            Lead(
                id=uuid4(),
                first_name="Lead1",
                last_name="Test",
                status="new",
                priority="high",
                next_follow_up=past_time
            ),
            Lead(
                id=uuid4(),
                first_name="Lead2", 
                last_name="Test",
                status="contacted",
                priority="medium",
                next_follow_up=past_time
            )
        ]
        
        with patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute:
            mock_execute.return_value.scalars.return_value.all.return_value = mock_leads
            
            leads = await service.get_leads_for_follow_up(10, mock_db)
            
            assert len(leads) == 2
            assert all(lead.next_follow_up <= datetime.now(timezone.utc) for lead in leads)


class TestCRMAPIEndpoints:
    """Test suite for CRM API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_auth_url_endpoint(self, client: AsyncClient):
        """Test CRM auth URL endpoint."""
        response = await client.get("/api/v1/crm/auth")
        
        assert response.status_code == 200
        data = response.json()
        assert "auth_url" in data
        assert "state" in data
        assert "oauth/v2/auth" in data["auth_url"]
    
    @pytest.mark.asyncio
    async def test_create_lead_endpoint(self, client: AsyncClient):
        """Test create lead endpoint."""
        lead_data = {
            "first_name": "Test",
            "last_name": "Lead",
            "email": "test@example.com",
            "phone": "+1234567890",
            "legal_issue": "Test legal issue",
            "urgency_level": "normal"
        }
        
        response = await client.post("/api/v1/crm/leads", json=lead_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["first_name"] == "Test"
        assert data["last_name"] == "Lead"
        assert data["email"] == "test@example.com"
        assert "id" in data
    
    @pytest.mark.asyncio
    async def test_list_leads_endpoint(self, client: AsyncClient):
        """Test list leads endpoint."""
        response = await client.get("/api/v1/crm/leads")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_create_appointment_endpoint(self, client: AsyncClient, mock_lead):
        """Test create appointment endpoint."""
        appointment_data = {
            "lead_id": str(mock_lead.id),
            "title": "Initial Consultation",
            "appointment_type": "consultation",
            "scheduled_start": "2024-12-01T10:00:00Z",
            "scheduled_end": "2024-12-01T11:00:00Z",
            "meeting_type": "in_person"
        }
        
        with patch('backend.app.api.v1.crm.select') as mock_select:
            mock_select.return_value.scalar_one_or_none.return_value = mock_lead
            
            response = await client.post("/api/v1/crm/appointments", json=appointment_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["title"] == "Initial Consultation"
            assert data["appointment_type"] == "consultation"
    
    @pytest.mark.asyncio
    async def test_sync_lead_endpoint(self, client: AsyncClient, mock_lead):
        """Test manual lead sync endpoint."""
        with patch('backend.app.services.lead_service.lead_service.sync_lead_to_crm', return_value=True):
            response = await client.post(f"/api/v1/crm/leads/{mock_lead.id}/sync")
            
            assert response.status_code == 200
            data = response.json()
            assert data["sync_status"] == "synced"


@pytest.mark.integration
class TestCRMIntegration:
    """Integration tests for CRM functionality."""
    
    @pytest.mark.asyncio
    async def test_full_lead_lifecycle(self, mock_db: AsyncSession):
        """Test complete lead lifecycle from creation to CRM sync."""
        # This would test the full flow:
        # 1. Create lead from conversation
        # 2. Score and prioritize lead
        # 3. Assign to attorney
        # 4. Schedule follow-up
        # 5. Sync to CRM
        # 6. Create appointment
        
        # Implementation would use real database and mock external services
        pass
    
    @pytest.mark.asyncio
    async def test_crm_webhook_processing(self, mock_db: AsyncSession):
        """Test processing of incoming CRM webhooks."""
        # This would test webhook handling:
        # 1. Receive webhook from Zoho
        # 2. Validate webhook signature
        # 3. Process lead updates
        # 4. Update local database
        
        pass
    
    @pytest.mark.asyncio  
    async def test_bulk_lead_sync(self, mock_db: AsyncSession):
        """Test bulk synchronization of leads from CRM."""
        # This would test:
        # 1. Fetch leads from Zoho in batches
        # 2. Transform CRM data to internal format
        # 3. Handle conflicts and duplicates
        # 4. Update sync status and errors
        
        pass


# Mock fixtures for testing
@pytest.fixture
async def mock_db():
    """Mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
async def client():
    """Mock HTTP client for API testing."""
    return AsyncMock(spec=AsyncClient)