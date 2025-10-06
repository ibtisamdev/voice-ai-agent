"""
Campaign API Endpoints

FastAPI endpoints for campaign management, contact lists,
and outbound calling campaigns.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID
import csv
import io

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, Field

from ...core.database import get_db
from ...core.config import settings
from ...models.campaign import (
    Campaign, CampaignList, CampaignContact, CampaignSchedule, 
    DoNotCallList, CampaignAnalytics
)
from ...models.telephony import CallRecord
from ...services.call_service import call_service

router = APIRouter(prefix="/campaigns", tags=["Campaign Management"])


# Pydantic schemas
class CampaignCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    campaign_type: str = Field(..., regex="^(outbound_sales|follow_up|survey|appointment_reminder)$")
    dialer_mode: str = Field(default="progressive", regex="^(progressive|predictive|preview|manual)$")
    max_concurrent_calls: int = Field(default=10, ge=1, le=100)
    call_timeout: int = Field(default=30, ge=10, le=120)
    retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_delay: int = Field(default=3600, ge=300, le=86400)
    timezone: str = Field(default="America/New_York")
    calling_hours_start: str = Field(default="09:00", regex="^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    calling_hours_end: str = Field(default="18:00", regex="^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    calling_days: str = Field(default="1,2,3,4,5")
    respect_dnc: bool = Field(default=True)
    practice_area: Optional[str] = Field(None, max_length=100)
    script_template: Optional[str] = None


class CampaignUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    dialer_mode: Optional[str] = Field(None, regex="^(progressive|predictive|preview|manual)$")
    max_concurrent_calls: Optional[int] = Field(None, ge=1, le=100)
    status: Optional[str] = Field(None, regex="^(draft|scheduled|running|paused|completed|cancelled)$")
    script_template: Optional[str] = None


class CampaignResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    campaign_type: str
    status: str
    dialer_mode: str
    max_concurrent_calls: int
    scheduled_start: Optional[datetime]
    scheduled_end: Optional[datetime]
    actual_start: Optional[datetime]
    actual_end: Optional[datetime]
    total_leads: int
    calls_attempted: int
    calls_connected: int
    conversations_completed: int
    appointments_scheduled: int
    conversion_rate: float
    created_at: datetime
    created_by: str
    
    class Config:
        from_attributes = True


class CampaignListCreateRequest(BaseModel):
    campaign_id: str
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class CampaignListResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    total_contacts: int
    valid_contacts: int
    invalid_contacts: int
    dnc_contacts: int
    processing_status: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class CampaignContactCreateRequest(BaseModel):
    campaign_list_id: str
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    phone_number: str = Field(..., min_length=10, max_length=50)
    email: Optional[str] = Field(None, max_length=255)
    company: Optional[str] = Field(None, max_length=255)
    title: Optional[str] = Field(None, max_length=100)
    priority: int = Field(default=1, ge=1, le=5)
    custom_data: Optional[Dict[str, Any]] = None


class CampaignContactResponse(BaseModel):
    id: str
    first_name: Optional[str]
    last_name: Optional[str]
    phone_number: str
    email: Optional[str]
    company: Optional[str]
    call_status: str
    call_attempts: int
    last_call_attempt: Optional[datetime]
    contact_outcome: Optional[str]
    priority: int
    phone_validated: bool
    
    class Config:
        from_attributes = True


class CampaignScheduleCreateRequest(BaseModel):
    campaign_id: str
    schedule_type: str = Field(..., regex="^(daily|weekly|monthly|custom)$")
    start_date: datetime
    end_date: Optional[datetime] = None
    daily_start_time: str = Field(..., regex="^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    daily_end_time: str = Field(..., regex="^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    daily_max_calls: Optional[int] = Field(None, ge=1, le=10000)
    monday_enabled: bool = Field(default=True)
    tuesday_enabled: bool = Field(default=True)
    wednesday_enabled: bool = Field(default=True)
    thursday_enabled: bool = Field(default=True)
    friday_enabled: bool = Field(default=True)
    saturday_enabled: bool = Field(default=False)
    sunday_enabled: bool = Field(default=False)
    timezone: str = Field(default="America/New_York")


class CampaignScheduleResponse(BaseModel):
    id: str
    schedule_type: str
    start_date: datetime
    end_date: Optional[datetime]
    daily_start_time: str
    daily_end_time: str
    is_active: bool
    last_execution: Optional[datetime]
    next_execution: Optional[datetime]
    
    class Config:
        from_attributes = True


class DoNotCallCreateRequest(BaseModel):
    phone_number: str = Field(..., min_length=10, max_length=50)
    source: str = Field(..., max_length=50)
    source_details: Optional[str] = Field(None, max_length=255)
    expiry_date: Optional[datetime] = None
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    email: Optional[str] = Field(None, max_length=255)
    notes: Optional[str] = None


class DoNotCallResponse(BaseModel):
    id: str
    phone_number: str
    source: str
    registered_date: datetime
    expiry_date: Optional[datetime]
    status: str
    first_name: Optional[str]
    last_name: Optional[str]
    
    class Config:
        from_attributes = True


# Campaign Management
@router.post("/", response_model=CampaignResponse)
async def create_campaign(
    campaign_data: CampaignCreateRequest,
    created_by: str = Query(..., description="User ID creating the campaign"),
    db: AsyncSession = Depends(get_db)
):
    """Create new campaign."""
    try:
        campaign = Campaign(
            **campaign_data.dict(),
            created_by=created_by,
            status="draft"
        )
        
        db.add(campaign)
        await db.commit()
        await db.refresh(campaign)
        
        return CampaignResponse.from_orm(campaign)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create campaign: {str(e)}")


@router.get("/", response_model=List[CampaignResponse])
async def list_campaigns(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None, description="Filter by status"),
    campaign_type: Optional[str] = Query(None, description="Filter by type"),
    created_by: Optional[str] = Query(None, description="Filter by creator"),
    db: AsyncSession = Depends(get_db)
):
    """List campaigns with filtering."""
    stmt = select(Campaign).where(Campaign.is_deleted == False)
    
    if status:
        stmt = stmt.where(Campaign.status == status)
    if campaign_type:
        stmt = stmt.where(Campaign.campaign_type == campaign_type)
    if created_by:
        stmt = stmt.where(Campaign.created_by == created_by)
    
    stmt = stmt.order_by(desc(Campaign.created_at)).offset(skip).limit(limit)
    
    result = await db.execute(stmt)
    campaigns = result.scalars().all()
    
    return [CampaignResponse.from_orm(campaign) for campaign in campaigns]


@router.get("/{campaign_id}", response_model=CampaignResponse)
async def get_campaign(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get specific campaign."""
    stmt = select(Campaign).where(and_(Campaign.id == campaign_id, Campaign.is_deleted == False))
    result = await db.execute(stmt)
    campaign = result.scalar_one_or_none()
    
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    return CampaignResponse.from_orm(campaign)


@router.put("/{campaign_id}", response_model=CampaignResponse)
async def update_campaign(
    campaign_id: UUID,
    campaign_data: CampaignUpdateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Update campaign."""
    stmt = select(Campaign).where(and_(Campaign.id == campaign_id, Campaign.is_deleted == False))
    result = await db.execute(stmt)
    campaign = result.scalar_one_or_none()
    
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    # Don't allow updates to running campaigns except status changes
    if campaign.status == "running" and campaign_data.status != "paused":
        raise HTTPException(status_code=400, detail="Cannot modify running campaign")
    
    # Update fields
    update_data = campaign_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(campaign, field, value)
    
    await db.commit()
    await db.refresh(campaign)
    
    return CampaignResponse.from_orm(campaign)


@router.delete("/{campaign_id}")
async def delete_campaign(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Soft delete campaign."""
    stmt = select(Campaign).where(and_(Campaign.id == campaign_id, Campaign.is_deleted == False))
    result = await db.execute(stmt)
    campaign = result.scalar_one_or_none()
    
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    if campaign.status == "running":
        raise HTTPException(status_code=400, detail="Cannot delete running campaign")
    
    campaign.is_deleted = True
    campaign.deleted_at = datetime.now(timezone.utc)
    await db.commit()
    
    return {"message": "Campaign deleted"}


# Campaign Control
@router.post("/{campaign_id}/start")
async def start_campaign(
    campaign_id: UUID,
    schedule_start: Optional[datetime] = Query(None, description="When to start the campaign"),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Start campaign."""
    stmt = select(Campaign).where(and_(Campaign.id == campaign_id, Campaign.is_deleted == False))
    result = await db.execute(stmt)
    campaign = result.scalar_one_or_none()
    
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    if campaign.status not in ["draft", "scheduled", "paused"]:
        raise HTTPException(status_code=400, detail=f"Cannot start campaign in {campaign.status} status")
    
    # Check if campaign has contacts
    contact_count_stmt = (
        select(func.count(CampaignContact.id))
        .select_from(CampaignContact)
        .join(CampaignList)
        .where(CampaignList.campaign_id == campaign_id)
    )
    contact_count = await db.scalar(contact_count_stmt)
    
    if contact_count == 0:
        raise HTTPException(status_code=400, detail="Campaign has no contacts")
    
    try:
        if schedule_start and schedule_start > datetime.now(timezone.utc):
            # Schedule for later
            campaign.status = "scheduled"
            campaign.scheduled_start = schedule_start
        else:
            # Start immediately
            campaign.status = "running"
            campaign.actual_start = datetime.now(timezone.utc)
            
            # Start campaign execution in background
            background_tasks.add_task(_execute_campaign, campaign_id, db)
        
        await db.commit()
        
        return {"message": "Campaign started", "status": campaign.status}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start campaign: {str(e)}")


@router.post("/{campaign_id}/pause")
async def pause_campaign(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Pause running campaign."""
    stmt = select(Campaign).where(and_(Campaign.id == campaign_id, Campaign.is_deleted == False))
    result = await db.execute(stmt)
    campaign = result.scalar_one_or_none()
    
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    if campaign.status != "running":
        raise HTTPException(status_code=400, detail="Campaign is not running")
    
    campaign.status = "paused"
    await db.commit()
    
    return {"message": "Campaign paused"}


@router.post("/{campaign_id}/stop")
async def stop_campaign(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Stop campaign completely."""
    stmt = select(Campaign).where(and_(Campaign.id == campaign_id, Campaign.is_deleted == False))
    result = await db.execute(stmt)
    campaign = result.scalar_one_or_none()
    
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    if campaign.status in ["completed", "cancelled"]:
        raise HTTPException(status_code=400, detail="Campaign already stopped")
    
    campaign.status = "cancelled"
    campaign.actual_end = datetime.now(timezone.utc)
    await db.commit()
    
    return {"message": "Campaign stopped"}


# Campaign Lists
@router.post("/{campaign_id}/lists", response_model=CampaignListResponse)
async def create_campaign_list(
    campaign_id: UUID,
    list_data: CampaignListCreateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Create campaign list."""
    # Verify campaign exists
    campaign_stmt = select(Campaign).where(Campaign.id == campaign_id)
    campaign_result = await db.execute(campaign_stmt)
    campaign = campaign_result.scalar_one_or_none()
    
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    try:
        campaign_list = CampaignList(
            campaign_id=campaign_id,
            **list_data.dict(exclude={'campaign_id'})
        )
        
        db.add(campaign_list)
        await db.commit()
        await db.refresh(campaign_list)
        
        return CampaignListResponse.from_orm(campaign_list)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create list: {str(e)}")


@router.get("/{campaign_id}/lists", response_model=List[CampaignListResponse])
async def list_campaign_lists(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """List campaign lists."""
    # Verify campaign exists
    campaign_stmt = select(Campaign).where(Campaign.id == campaign_id)
    campaign_result = await db.execute(campaign_stmt)
    campaign = campaign_result.scalar_one_or_none()
    
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    stmt = (
        select(CampaignList)
        .where(CampaignList.campaign_id == campaign_id)
        .order_by(CampaignList.created_at)
    )
    
    result = await db.execute(stmt)
    lists = result.scalars().all()
    
    return [CampaignListResponse.from_orm(lst) for lst in lists]


@router.post("/lists/{list_id}/upload")
async def upload_contact_list(
    list_id: UUID,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload contacts from CSV file."""
    # Verify list exists
    list_stmt = select(CampaignList).where(CampaignList.id == list_id)
    list_result = await db.execute(list_stmt)
    campaign_list = list_result.scalar_one_or_none()
    
    if not campaign_list:
        raise HTTPException(status_code=404, detail="Campaign list not found")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read CSV content
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Process CSV in background
        background_tasks.add_task(
            _process_csv_upload,
            list_id,
            csv_content,
            file.filename,
            db
        )
        
        # Update list status
        campaign_list.processing_status = "processing"
        campaign_list.source_type = "upload"
        campaign_list.source_file = file.filename
        await db.commit()
        
        return {"message": "File uploaded successfully, processing in background"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# Campaign Contacts
@router.post("/lists/{list_id}/contacts", response_model=CampaignContactResponse)
async def add_campaign_contact(
    list_id: UUID,
    contact_data: CampaignContactCreateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Add individual contact to campaign list."""
    # Verify list exists
    list_stmt = select(CampaignList).where(CampaignList.id == list_id)
    list_result = await db.execute(list_stmt)
    campaign_list = list_result.scalar_one_or_none()
    
    if not campaign_list:
        raise HTTPException(status_code=404, detail="Campaign list not found")
    
    try:
        contact = CampaignContact(
            campaign_list_id=list_id,
            **contact_data.dict(exclude={'campaign_list_id'})
        )
        
        db.add(contact)
        
        # Update list statistics
        campaign_list.total_contacts += 1
        campaign_list.valid_contacts += 1
        
        await db.commit()
        await db.refresh(contact)
        
        return CampaignContactResponse.from_orm(contact)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add contact: {str(e)}")


@router.get("/lists/{list_id}/contacts", response_model=List[CampaignContactResponse])
async def list_campaign_contacts(
    list_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=10000),
    status: Optional[str] = Query(None, description="Filter by call status"),
    db: AsyncSession = Depends(get_db)
):
    """List contacts in campaign list."""
    # Verify list exists
    list_stmt = select(CampaignList).where(CampaignList.id == list_id)
    list_result = await db.execute(list_stmt)
    campaign_list = list_result.scalar_one_or_none()
    
    if not campaign_list:
        raise HTTPException(status_code=404, detail="Campaign list not found")
    
    stmt = select(CampaignContact).where(CampaignContact.campaign_list_id == list_id)
    
    if status:
        stmt = stmt.where(CampaignContact.call_status == status)
    
    stmt = stmt.order_by(CampaignContact.priority.desc(), CampaignContact.created_at).offset(skip).limit(limit)
    
    result = await db.execute(stmt)
    contacts = result.scalars().all()
    
    return [CampaignContactResponse.from_orm(contact) for contact in contacts]


# Do Not Call Management
@router.post("/dnc", response_model=DoNotCallResponse)
async def add_dnc_number(
    dnc_data: DoNotCallCreateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Add number to Do Not Call list."""
    try:
        dnc_record = DoNotCallList(
            **dnc_data.dict(),
            registered_date=datetime.now(timezone.utc),
            status="active"
        )
        
        db.add(dnc_record)
        await db.commit()
        await db.refresh(dnc_record)
        
        return DoNotCallResponse.from_orm(dnc_record)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add DNC record: {str(e)}")


@router.get("/dnc", response_model=List[DoNotCallResponse])
async def list_dnc_numbers(
    skip: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=10000),
    search: Optional[str] = Query(None, description="Search phone numbers"),
    db: AsyncSession = Depends(get_db)
):
    """List Do Not Call numbers."""
    stmt = select(DoNotCallList).where(DoNotCallList.status == "active")
    
    if search:
        stmt = stmt.where(DoNotCallList.phone_number.contains(search))
    
    stmt = stmt.order_by(desc(DoNotCallList.registered_date)).offset(skip).limit(limit)
    
    result = await db.execute(stmt)
    dnc_records = result.scalars().all()
    
    return [DoNotCallResponse.from_orm(record) for record in dnc_records]


@router.get("/dnc/check/{phone_number}")
async def check_dnc_status(
    phone_number: str,
    db: AsyncSession = Depends(get_db)
):
    """Check if phone number is on DNC list."""
    stmt = (
        select(DoNotCallList)
        .where(and_(
            DoNotCallList.phone_number == phone_number,
            DoNotCallList.status == "active"
        ))
    )
    
    result = await db.execute(stmt)
    dnc_record = result.scalar_one_or_none()
    
    return {
        "phone_number": phone_number,
        "is_dnc": dnc_record is not None,
        "source": dnc_record.source if dnc_record else None,
        "registered_date": dnc_record.registered_date if dnc_record else None
    }


# Campaign Analytics
@router.get("/{campaign_id}/analytics")
async def get_campaign_analytics(
    campaign_id: UUID,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed campaign analytics."""
    # Verify campaign exists
    campaign_stmt = select(Campaign).where(Campaign.id == campaign_id)
    campaign_result = await db.execute(campaign_stmt)
    campaign = campaign_result.scalar_one_or_none()
    
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    try:
        # Get call analytics for this campaign
        call_analytics = await call_service.get_call_analytics(
            start_date=start_date,
            end_date=end_date,
            campaign_id=campaign_id,
            db=db
        )
        
        # Get contact statistics
        total_contacts_stmt = (
            select(func.count(CampaignContact.id))
            .select_from(CampaignContact)
            .join(CampaignList)
            .where(CampaignList.campaign_id == campaign_id)
        )
        total_contacts = await db.scalar(total_contacts_stmt)
        
        contacted_stmt = (
            select(func.count(CampaignContact.id))
            .select_from(CampaignContact)
            .join(CampaignList)
            .where(and_(
                CampaignList.campaign_id == campaign_id,
                CampaignContact.call_status == "contacted"
            ))
        )
        contacted = await db.scalar(contacted_stmt)
        
        return {
            "campaign_id": str(campaign_id),
            "campaign_name": campaign.name,
            "campaign_status": campaign.status,
            "total_contacts": total_contacts,
            "contacted": contacted,
            "contact_rate": (contacted / total_contacts * 100) if total_contacts > 0 else 0,
            "call_analytics": call_analytics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


# Background task functions
async def _execute_campaign(campaign_id: UUID, db: AsyncSession):
    """Execute campaign by processing contacts and making calls."""
    # This would implement the actual campaign execution logic
    # For now, it's a placeholder that would:
    # 1. Get pending contacts from campaign lists
    # 2. Check DNC status
    # 3. Respect calling hours and pacing rules
    # 4. Make calls via call_service
    # 5. Update contact status based on call outcomes
    pass


async def _process_csv_upload(list_id: UUID, csv_content: str, filename: str, db: AsyncSession):
    """Process uploaded CSV file and create contacts."""
    try:
        # Get the campaign list
        list_stmt = select(CampaignList).where(CampaignList.id == list_id)
        list_result = await db.execute(list_stmt)
        campaign_list = list_result.scalar_one_or_none()
        
        if not campaign_list:
            return
        
        # Parse CSV
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        
        total_processed = 0
        valid_contacts = 0
        invalid_contacts = 0
        
        for row in csv_reader:
            total_processed += 1
            
            try:
                # Extract contact data (adjust field mapping as needed)
                contact_data = {
                    "campaign_list_id": list_id,
                    "first_name": row.get("first_name", "").strip(),
                    "last_name": row.get("last_name", "").strip(),
                    "phone_number": row.get("phone", "").strip(),
                    "email": row.get("email", "").strip(),
                    "company": row.get("company", "").strip(),
                    "title": row.get("title", "").strip(),
                }
                
                # Validate required fields
                if not contact_data["phone_number"]:
                    invalid_contacts += 1
                    continue
                
                # Create contact
                contact = CampaignContact(**contact_data)
                db.add(contact)
                valid_contacts += 1
                
            except Exception as e:
                invalid_contacts += 1
                continue
        
        # Update campaign list statistics
        campaign_list.total_contacts = total_processed
        campaign_list.valid_contacts = valid_contacts
        campaign_list.invalid_contacts = invalid_contacts
        campaign_list.processing_status = "completed"
        campaign_list.imported_at = datetime.now(timezone.utc)
        
        await db.commit()
        
    except Exception as e:
        # Mark as failed
        if campaign_list:
            campaign_list.processing_status = "error"
            campaign_list.processing_error = str(e)
            await db.commit()