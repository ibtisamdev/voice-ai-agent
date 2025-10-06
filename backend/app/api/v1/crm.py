"""
CRM API Endpoints

FastAPI endpoints for Zoho CRM integration, lead management,
and appointment scheduling.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, Field

from ...core.database import get_db
from ...core.config import settings
from ...models.crm import CRMAccount, Lead, LeadActivity, Appointment, CRMSyncJob
from ...services.crm_service import crm_service, ZohoCRMError, ZohoAuthError
from ...services.lead_service import lead_service

router = APIRouter(prefix="/crm", tags=["CRM Integration"])


# Pydantic schemas
class CRMAuthResponse(BaseModel):
    auth_url: str
    state: str


class CRMAccountResponse(BaseModel):
    id: str
    name: str
    zoho_org_id: str
    is_active: bool
    sandbox_mode: bool
    last_sync: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class LeadCreateRequest(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    email: Optional[str] = Field(None, max_length=255)
    phone: Optional[str] = Field(None, max_length=50)
    company: Optional[str] = Field(None, max_length=255)
    title: Optional[str] = Field(None, max_length=100)
    source: str = Field(default="api", max_length=50)
    legal_issue: Optional[str] = None
    practice_areas: Optional[List[str]] = None
    urgency_level: str = Field(default="normal", pattern="^(low|normal|high|emergency)$")
    notes: Optional[str] = None


class LeadUpdateRequest(BaseModel):
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    email: Optional[str] = Field(None, max_length=255)
    phone: Optional[str] = Field(None, max_length=50)
    company: Optional[str] = Field(None, max_length=255)
    title: Optional[str] = Field(None, max_length=100)
    status: Optional[str] = Field(None, pattern="^(new|contacted|qualified|converted|lost)$")
    legal_issue: Optional[str] = None
    practice_areas: Optional[List[str]] = None
    urgency_level: Optional[str] = Field(None, pattern="^(low|normal|high|emergency)$")
    notes: Optional[str] = None


class LeadResponse(BaseModel):
    id: str
    first_name: str
    last_name: str
    email: Optional[str]
    phone: Optional[str]
    company: Optional[str]
    title: Optional[str]
    source: str
    status: str
    lead_score: int
    priority: str
    practice_areas: Optional[List[str]]
    legal_issue: Optional[str]
    urgency_level: str
    created_at: datetime
    updated_at: datetime
    next_follow_up: Optional[datetime]
    zoho_lead_id: Optional[str]
    sync_status: str
    
    class Config:
        from_attributes = True


class AppointmentCreateRequest(BaseModel):
    lead_id: str
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    appointment_type: str = Field(..., pattern="^(consultation|follow_up|meeting)$")
    scheduled_start: datetime
    scheduled_end: datetime
    attorney_name: Optional[str] = Field(None, max_length=255)
    meeting_type: str = Field(default="in_person", pattern="^(in_person|phone|video)$")
    location: Optional[str] = Field(None, max_length=500)
    notes: Optional[str] = None


class AppointmentResponse(BaseModel):
    id: str
    lead_id: str
    title: str
    description: Optional[str]
    appointment_type: str
    scheduled_start: datetime
    scheduled_end: datetime
    status: str
    attorney_name: Optional[str]
    meeting_type: str
    location: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class SyncJobResponse(BaseModel):
    id: str
    job_type: str
    status: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    records_processed: int
    records_successful: int
    records_failed: int
    error_message: Optional[str]
    
    class Config:
        from_attributes = True


# Authentication endpoints
@router.get("/auth", response_model=CRMAuthResponse)
async def get_auth_url():
    """
    Get Zoho OAuth authorization URL.
    
    Returns authorization URL for Zoho CRM integration.
    """
    try:
        state = f"auth_{datetime.now().timestamp()}"
        auth_url = crm_service.get_auth_url(state)
        
        return CRMAuthResponse(auth_url=auth_url, state=state)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate auth URL: {str(e)}")


@router.get("/auth/callback")
async def auth_callback(
    code: str = Query(..., description="Authorization code from Zoho"),
    state: Optional[str] = Query(None, description="State parameter"),
    error: Optional[str] = Query(None, description="Error from Zoho"),
    db: AsyncSession = Depends(get_db)
):
    """
    Handle OAuth callback from Zoho.
    
    Exchanges authorization code for access tokens and creates CRM account.
    """
    if error:
        raise HTTPException(status_code=400, detail=f"Authorization failed: {error}")
    
    if not code:
        raise HTTPException(status_code=400, detail="Authorization code is required")
    
    try:
        async with crm_service as crm:
            account = await crm.exchange_code_for_tokens(code, db)
        
        # Redirect to success page or return account info
        return {
            "message": "CRM integration successful",
            "account": CRMAccountResponse.from_orm(account)
        }
    
    except ZohoAuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")


# Account management endpoints
@router.get("/accounts", response_model=List[CRMAccountResponse])
async def list_crm_accounts(db: AsyncSession = Depends(get_db)):
    """List all CRM accounts."""
    stmt = select(CRMAccount).order_by(desc(CRMAccount.created_at))
    result = await db.execute(stmt)
    accounts = result.scalars().all()
    
    return [CRMAccountResponse.from_orm(account) for account in accounts]


@router.get("/accounts/{account_id}", response_model=CRMAccountResponse)
async def get_crm_account(
    account_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get specific CRM account."""
    stmt = select(CRMAccount).where(CRMAccount.id == account_id)
    result = await db.execute(stmt)
    account = result.scalar_one_or_none()
    
    if not account:
        raise HTTPException(status_code=404, detail="CRM account not found")
    
    return CRMAccountResponse.from_orm(account)


@router.delete("/accounts/{account_id}")
async def delete_crm_account(
    account_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Delete CRM account."""
    stmt = select(CRMAccount).where(CRMAccount.id == account_id)
    result = await db.execute(stmt)
    account = result.scalar_one_or_none()
    
    if not account:
        raise HTTPException(status_code=404, detail="CRM account not found")
    
    account.is_active = False
    await db.commit()
    
    return {"message": "CRM account deactivated"}


# Lead management endpoints
@router.post("/leads", response_model=LeadResponse)
async def create_lead(
    lead_data: LeadCreateRequest,
    sync_to_crm: bool = Query(False, description="Automatically sync to CRM"),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Create new lead."""
    try:
        # Create lead
        lead = Lead(
            **lead_data.dict(exclude_unset=True)
        )
        
        # Calculate lead score
        lead.lead_score = lead_service.scoring_service.calculate_lead_score(lead)
        lead.priority = lead_service.scoring_service.determine_priority(
            lead.lead_score, 
            lead.urgency_level
        )
        
        db.add(lead)
        await db.commit()
        await db.refresh(lead)
        
        # Sync to CRM if requested
        if sync_to_crm:
            background_tasks.add_task(lead_service.sync_lead_to_crm, lead, None, db)
        
        return LeadResponse.from_orm(lead)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create lead: {str(e)}")


@router.get("/leads", response_model=List[LeadResponse])
async def list_leads(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    status: Optional[str] = Query(None, description="Filter by status"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    source: Optional[str] = Query(None, description="Filter by source"),
    search: Optional[str] = Query(None, description="Search in name, email, phone"),
    db: AsyncSession = Depends(get_db)
):
    """List leads with filtering and pagination."""
    stmt = select(Lead).where(Lead.is_deleted == False)
    
    # Apply filters
    if status:
        stmt = stmt.where(Lead.status == status)
    if priority:
        stmt = stmt.where(Lead.priority == priority)
    if source:
        stmt = stmt.where(Lead.source == source)
    if search:
        search_term = f"%{search}%"
        stmt = stmt.where(
            or_(
                Lead.first_name.ilike(search_term),
                Lead.last_name.ilike(search_term),
                Lead.email.ilike(search_term),
                Lead.phone.ilike(search_term)
            )
        )
    
    # Apply pagination and ordering
    stmt = stmt.order_by(desc(Lead.created_at)).offset(skip).limit(limit)
    
    result = await db.execute(stmt)
    leads = result.scalars().all()
    
    return [LeadResponse.from_orm(lead) for lead in leads]


@router.get("/leads/{lead_id}", response_model=LeadResponse)
async def get_lead(
    lead_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get specific lead."""
    stmt = select(Lead).where(and_(Lead.id == lead_id, Lead.is_deleted == False))
    result = await db.execute(stmt)
    lead = result.scalar_one_or_none()
    
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    return LeadResponse.from_orm(lead)


@router.put("/leads/{lead_id}", response_model=LeadResponse)
async def update_lead(
    lead_id: UUID,
    lead_data: LeadUpdateRequest,
    sync_to_crm: bool = Query(False, description="Sync changes to CRM"),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Update lead."""
    stmt = select(Lead).where(and_(Lead.id == lead_id, Lead.is_deleted == False))
    result = await db.execute(stmt)
    lead = result.scalar_one_or_none()
    
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    # Update fields
    update_data = lead_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(lead, field, value)
    
    # Recalculate score if relevant fields changed
    if any(field in update_data for field in ['urgency_level', 'practice_areas']):
        lead.lead_score = lead_service.scoring_service.calculate_lead_score(lead)
        lead.priority = lead_service.scoring_service.determine_priority(
            lead.lead_score, 
            lead.urgency_level
        )
    
    await db.commit()
    await db.refresh(lead)
    
    # Sync to CRM if requested
    if sync_to_crm:
        background_tasks.add_task(lead_service.sync_lead_to_crm, lead, None, db)
    
    return LeadResponse.from_orm(lead)


@router.delete("/leads/{lead_id}")
async def delete_lead(
    lead_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Soft delete lead."""
    stmt = select(Lead).where(and_(Lead.id == lead_id, Lead.is_deleted == False))
    result = await db.execute(stmt)
    lead = result.scalar_one_or_none()
    
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    lead.is_deleted = True
    lead.deleted_at = datetime.now(timezone.utc)
    await db.commit()
    
    return {"message": "Lead deleted"}


@router.post("/leads/{lead_id}/sync")
async def sync_lead_to_crm(
    lead_id: UUID,
    account_id: Optional[UUID] = Query(None, description="Specific CRM account ID"),
    db: AsyncSession = Depends(get_db)
):
    """Manually sync lead to CRM."""
    stmt = select(Lead).where(and_(Lead.id == lead_id, Lead.is_deleted == False))
    result = await db.execute(stmt)
    lead = result.scalar_one_or_none()
    
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    try:
        success = await lead_service.sync_lead_to_crm(lead, account_id, db)
        
        if success:
            return {"message": "Lead synced successfully", "sync_status": "synced"}
        else:
            return {"message": "Lead sync failed", "sync_status": "error"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


# Appointment management endpoints
@router.post("/appointments", response_model=AppointmentResponse)
async def create_appointment(
    appointment_data: AppointmentCreateRequest,
    sync_to_crm: bool = Query(False, description="Sync to CRM calendar"),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Create new appointment."""
    # Verify lead exists
    stmt = select(Lead).where(Lead.id == UUID(appointment_data.lead_id))
    result = await db.execute(stmt)
    lead = result.scalar_one_or_none()
    
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    try:
        appointment = Appointment(**appointment_data.dict())
        db.add(appointment)
        await db.commit()
        await db.refresh(appointment)
        
        # Add activity to lead
        await lead_service.add_lead_activity(
            lead_id=lead.id,
            activity_type="meeting",
            status="scheduled",
            subject=appointment.title,
            description=appointment.description,
            scheduled_at=appointment.scheduled_start,
            db=db
        )
        
        # Sync to CRM if requested
        if sync_to_crm:
            background_tasks.add_task(_sync_appointment_to_crm, appointment, lead, db)
        
        return AppointmentResponse.from_orm(appointment)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create appointment: {str(e)}")


@router.get("/appointments", response_model=List[AppointmentResponse])
async def list_appointments(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    lead_id: Optional[UUID] = Query(None, description="Filter by lead"),
    status: Optional[str] = Query(None, description="Filter by status"),
    start_date: Optional[datetime] = Query(None, description="Filter from date"),
    end_date: Optional[datetime] = Query(None, description="Filter to date"),
    db: AsyncSession = Depends(get_db)
):
    """List appointments with filtering."""
    stmt = select(Appointment).where(Appointment.is_deleted == False)
    
    if lead_id:
        stmt = stmt.where(Appointment.lead_id == lead_id)
    if status:
        stmt = stmt.where(Appointment.status == status)
    if start_date:
        stmt = stmt.where(Appointment.scheduled_start >= start_date)
    if end_date:
        stmt = stmt.where(Appointment.scheduled_end <= end_date)
    
    stmt = stmt.order_by(Appointment.scheduled_start).offset(skip).limit(limit)
    
    result = await db.execute(stmt)
    appointments = result.scalars().all()
    
    return [AppointmentResponse.from_orm(apt) for apt in appointments]


# Sync management endpoints
@router.post("/sync")
async def trigger_sync(
    sync_type: str = Query("leads", description="Type of sync (leads, contacts, events)"),
    account_id: Optional[UUID] = Query(None, description="Specific account ID"),
    since: Optional[datetime] = Query(None, description="Sync since this datetime"),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Trigger manual CRM sync."""
    # Get CRM account
    if account_id:
        stmt = select(CRMAccount).where(CRMAccount.id == account_id)
    else:
        stmt = select(CRMAccount).where(CRMAccount.is_active == True).limit(1)
    
    result = await db.execute(stmt)
    account = result.scalar_one_or_none()
    
    if not account:
        raise HTTPException(status_code=404, detail="No active CRM account found")
    
    # Trigger background sync
    if sync_type == "leads":
        background_tasks.add_task(_sync_leads_from_crm, account, since, db)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported sync type: {sync_type}")
    
    return {"message": f"Sync triggered for {sync_type}", "account_id": str(account.id)}


@router.get("/sync/jobs", response_model=List[SyncJobResponse])
async def list_sync_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: AsyncSession = Depends(get_db)
):
    """List sync jobs."""
    stmt = select(CRMSyncJob)
    
    if status:
        stmt = stmt.where(CRMSyncJob.status == status)
    
    stmt = stmt.order_by(desc(CRMSyncJob.created_at)).offset(skip).limit(limit)
    
    result = await db.execute(stmt)
    jobs = result.scalars().all()
    
    return [SyncJobResponse.from_orm(job) for job in jobs]


@router.get("/analytics")
async def get_crm_analytics(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Get CRM analytics."""
    try:
        analytics = await lead_service.get_lead_analytics(start_date, end_date, db)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


# Background task functions
async def _sync_appointment_to_crm(appointment: Appointment, lead: Lead, db: AsyncSession):
    """Background task to sync appointment to CRM."""
    try:
        # Get active CRM account
        stmt = select(CRMAccount).where(CRMAccount.is_active == True).limit(1)
        result = await db.execute(stmt)
        account = result.scalar_one_or_none()
        
        if account:
            appointment_data = {
                "title": appointment.title,
                "description": appointment.description,
                "start_time": appointment.scheduled_start,
                "end_time": appointment.scheduled_end,
                "location": appointment.location
            }
            
            async with crm_service as crm:
                result = await crm.create_appointment_in_zoho(lead, appointment_data, account, db)
                
                # Update appointment with Zoho event ID
                if result.get("data"):
                    appointment.zoho_event_id = result["data"][0]["details"]["id"]
                    await db.commit()
    
    except Exception as e:
        logger.error(f"Failed to sync appointment to CRM: {e}")


async def _sync_leads_from_crm(account: CRMAccount, since: Optional[datetime], db: AsyncSession):
    """Background task to sync leads from CRM."""
    try:
        async with crm_service as crm:
            await crm.sync_leads_from_zoho(account, db, since)
    except Exception as e:
        logger.error(f"Failed to sync leads from CRM: {e}")


# Helper imports
import logging
from sqlalchemy import or_

logger = logging.getLogger(__name__)