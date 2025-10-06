"""
Telephony API Endpoints

FastAPI endpoints for call management, SIP configuration,
and telephony operations.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, Field

from ...core.database import get_db
from ...core.config import settings
from ...models.telephony import CallRecord, CallEvent, SIPAccount, CallQueue, QueueMember
from ...models.crm import Lead
from ...services.call_service import call_service
from telephony.sip_gateway import sip_gateway

router = APIRouter(prefix="/telephony", tags=["Telephony"])


# Pydantic schemas
class SIPAccountCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=255)
    domain: str = Field(..., min_length=1, max_length=255)
    server: str = Field(..., min_length=1, max_length=255)
    port: int = Field(default=5060, ge=1, le=65535)
    transport: str = Field(default="UDP", pattern="^(UDP|TCP|TLS)$")
    max_concurrent_calls: int = Field(default=10, ge=1, le=100)


class SIPAccountResponse(BaseModel):
    id: str
    name: str
    username: str
    domain: str
    server: str
    port: int
    transport: str
    registration_status: str
    last_registration: Optional[datetime]
    is_active: bool
    max_concurrent_calls: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class CallCreateRequest(BaseModel):
    destination: str = Field(..., min_length=1, max_length=50)
    caller_id: Optional[str] = Field(None, max_length=50)
    campaign_id: Optional[str] = None
    lead_id: Optional[str] = None


class CallResponse(BaseModel):
    id: str
    call_id: str
    direction: str
    caller_number: Optional[str]
    called_number: Optional[str]
    status: str
    disposition: Optional[str]
    initiated_at: datetime
    answered_at: Optional[datetime]
    ended_at: Optional[datetime]
    duration: Optional[int]
    lead_id: Optional[str]
    campaign_id: Optional[str]
    
    class Config:
        from_attributes = True


class CallEventResponse(BaseModel):
    id: str
    event_type: str
    event_time: datetime
    event_data: Optional[Dict[str, Any]]
    sip_method: Optional[str]
    sip_response_code: Optional[int]
    
    class Config:
        from_attributes = True


class CallQueueCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    max_wait_time: int = Field(default=300, ge=30, le=3600)
    max_queue_size: int = Field(default=50, ge=1, le=500)
    routing_strategy: str = Field(default="round_robin", pattern="^(round_robin|longest_idle|random)$")
    overflow_destination: Optional[str] = Field(None, max_length=255)


class CallQueueResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    max_wait_time: int
    max_queue_size: int
    routing_strategy: str
    is_active: bool
    current_queue_size: int
    total_calls_received: int
    average_wait_time: float
    
    class Config:
        from_attributes = True


class QueueMemberCreateRequest(BaseModel):
    queue_id: str
    member_name: str = Field(..., min_length=1, max_length=255)
    member_extension: Optional[str] = Field(None, max_length=20)
    member_phone: Optional[str] = Field(None, max_length=50)
    priority: int = Field(default=1, ge=1, le=10)
    max_concurrent_calls: int = Field(default=1, ge=1, le=10)


class QueueMemberResponse(BaseModel):
    id: str
    member_name: str
    member_extension: Optional[str]
    member_phone: Optional[str]
    priority: int
    status: str
    calls_taken: int
    last_call_time: Optional[datetime]
    
    class Config:
        from_attributes = True


# SIP Account Management
@router.post("/sip-accounts", response_model=SIPAccountResponse)
async def create_sip_account(
    account_data: SIPAccountCreateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Create new SIP account."""
    try:
        sip_account = SIPAccount(**account_data.dict())
        db.add(sip_account)
        await db.commit()
        await db.refresh(sip_account)
        
        # Register account with SIP gateway
        sip_config = {
            "username": sip_account.username,
            "password": sip_account.password,
            "domain": sip_account.domain,
            "server": sip_account.server,
            "port": sip_account.port
        }
        
        try:
            await sip_gateway.register_account(sip_config)
            sip_account.registration_status = "registered"
            sip_account.last_registration = datetime.now(timezone.utc)
        except Exception as e:
            sip_account.registration_status = "failed"
            sip_account.registration_error = str(e)
        
        await db.commit()
        
        return SIPAccountResponse.from_orm(sip_account)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create SIP account: {str(e)}")


@router.get("/sip-accounts", response_model=List[SIPAccountResponse])
async def list_sip_accounts(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    active_only: bool = Query(False, description="Show only active accounts"),
    db: AsyncSession = Depends(get_db)
):
    """List SIP accounts."""
    stmt = select(SIPAccount)
    
    if active_only:
        stmt = stmt.where(SIPAccount.is_active == True)
    
    stmt = stmt.order_by(desc(SIPAccount.created_at)).offset(skip).limit(limit)
    
    result = await db.execute(stmt)
    accounts = result.scalars().all()
    
    return [SIPAccountResponse.from_orm(account) for account in accounts]


@router.get("/sip-accounts/{account_id}", response_model=SIPAccountResponse)
async def get_sip_account(
    account_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get specific SIP account."""
    stmt = select(SIPAccount).where(SIPAccount.id == account_id)
    result = await db.execute(stmt)
    account = result.scalar_one_or_none()
    
    if not account:
        raise HTTPException(status_code=404, detail="SIP account not found")
    
    return SIPAccountResponse.from_orm(account)


@router.delete("/sip-accounts/{account_id}")
async def delete_sip_account(
    account_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Delete SIP account."""
    stmt = select(SIPAccount).where(SIPAccount.id == account_id)
    result = await db.execute(stmt)
    account = result.scalar_one_or_none()
    
    if not account:
        raise HTTPException(status_code=404, detail="SIP account not found")
    
    account.is_active = False
    await db.commit()
    
    return {"message": "SIP account deactivated"}


# Call Management
@router.post("/calls", response_model=CallResponse)
async def make_call(
    call_data: CallCreateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Initiate outbound call."""
    try:
        campaign_id = UUID(call_data.campaign_id) if call_data.campaign_id else None
        lead_id = UUID(call_data.lead_id) if call_data.lead_id else None
        
        call_record = await call_service.make_outbound_call(
            destination=call_data.destination,
            campaign_id=campaign_id,
            lead_id=lead_id,
            caller_id=call_data.caller_id,
            db=db
        )
        
        if not call_record:
            raise HTTPException(status_code=500, detail="Failed to initiate call")
        
        return CallResponse.from_orm(call_record)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Call failed: {str(e)}")


@router.get("/calls", response_model=List[CallResponse])
async def list_calls(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    direction: Optional[str] = Query(None, description="Filter by direction"),
    status: Optional[str] = Query(None, description="Filter by status"),
    start_date: Optional[datetime] = Query(None, description="Filter from date"),
    end_date: Optional[datetime] = Query(None, description="Filter to date"),
    campaign_id: Optional[UUID] = Query(None, description="Filter by campaign"),
    db: AsyncSession = Depends(get_db)
):
    """List calls with filtering."""
    stmt = select(CallRecord)
    
    conditions = []
    if direction:
        conditions.append(CallRecord.direction == direction)
    if status:
        conditions.append(CallRecord.status == status)
    if start_date:
        conditions.append(CallRecord.initiated_at >= start_date)
    if end_date:
        conditions.append(CallRecord.initiated_at <= end_date)
    if campaign_id:
        conditions.append(CallRecord.campaign_id == campaign_id)
    
    if conditions:
        stmt = stmt.where(and_(*conditions))
    
    stmt = stmt.order_by(desc(CallRecord.initiated_at)).offset(skip).limit(limit)
    
    result = await db.execute(stmt)
    calls = result.scalars().all()
    
    return [CallResponse.from_orm(call) for call in calls]


@router.get("/calls/{call_id}", response_model=CallResponse)
async def get_call(
    call_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get specific call record."""
    stmt = select(CallRecord).where(CallRecord.id == call_id)
    result = await db.execute(stmt)
    call = result.scalar_one_or_none()
    
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    return CallResponse.from_orm(call)


@router.post("/calls/{call_id}/hangup")
async def hangup_call(
    call_id: UUID,
    reason: Optional[str] = Query(None, description="Hangup reason"),
    db: AsyncSession = Depends(get_db)
):
    """Hangup active call."""
    # Get call record
    stmt = select(CallRecord).where(CallRecord.id == call_id)
    result = await db.execute(stmt)
    call_record = result.scalar_one_or_none()
    
    if not call_record:
        raise HTTPException(status_code=404, detail="Call not found")
    
    if call_record.status in ["ended", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail="Call already ended")
    
    try:
        await call_service.hangup_call(call_record.call_id, reason, db)
        return {"message": "Call hung up successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to hangup call: {str(e)}")


@router.post("/calls/{call_id}/transfer")
async def transfer_call(
    call_id: UUID,
    destination: str = Query(..., description="Transfer destination"),
    transfer_type: str = Query("blind", description="Transfer type"),
    db: AsyncSession = Depends(get_db)
):
    """Transfer call to another destination."""
    # Get call record
    stmt = select(CallRecord).where(CallRecord.id == call_id)
    result = await db.execute(stmt)
    call_record = result.scalar_one_or_none()
    
    if not call_record:
        raise HTTPException(status_code=404, detail="Call not found")
    
    if call_record.status != "answered":
        raise HTTPException(status_code=400, detail="Call must be answered to transfer")
    
    try:
        success = await call_service.transfer_call(
            call_record.call_id,
            destination,
            transfer_type,
            db
        )
        
        if success:
            return {"message": f"Call transferred to {destination}"}
        else:
            raise HTTPException(status_code=500, detail="Transfer failed")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transfer failed: {str(e)}")


@router.post("/calls/{call_id}/dtmf")
async def send_dtmf(
    call_id: UUID,
    digits: str = Query(..., description="DTMF digits to send"),
    db: AsyncSession = Depends(get_db)
):
    """Send DTMF tones to call."""
    # Get call record
    stmt = select(CallRecord).where(CallRecord.id == call_id)
    result = await db.execute(stmt)
    call_record = result.scalar_one_or_none()
    
    if not call_record:
        raise HTTPException(status_code=404, detail="Call not found")
    
    if call_record.status != "answered":
        raise HTTPException(status_code=400, detail="Call must be answered to send DTMF")
    
    try:
        await call_service.send_dtmf(call_record.call_id, digits, db)
        return {"message": f"DTMF sent: {digits}"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DTMF failed: {str(e)}")


@router.get("/calls/{call_id}/events", response_model=List[CallEventResponse])
async def get_call_events(
    call_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    """Get call events."""
    # First get the call record
    call_stmt = select(CallRecord).where(CallRecord.id == call_id)
    call_result = await db.execute(call_stmt)
    call_record = call_result.scalar_one_or_none()
    
    if not call_record:
        raise HTTPException(status_code=404, detail="Call not found")
    
    # Get events
    stmt = (
        select(CallEvent)
        .where(CallEvent.call_record_id == call_id)
        .order_by(CallEvent.event_time)
        .offset(skip)
        .limit(limit)
    )
    
    result = await db.execute(stmt)
    events = result.scalars().all()
    
    return [CallEventResponse.from_orm(event) for event in events]


# Queue Management
@router.post("/queues", response_model=CallQueueResponse)
async def create_call_queue(
    queue_data: CallQueueCreateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Create new call queue."""
    try:
        queue = CallQueue(**queue_data.dict())
        db.add(queue)
        await db.commit()
        await db.refresh(queue)
        
        return CallQueueResponse.from_orm(queue)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create queue: {str(e)}")


@router.get("/queues", response_model=List[CallQueueResponse])
async def list_call_queues(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    active_only: bool = Query(False),
    db: AsyncSession = Depends(get_db)
):
    """List call queues."""
    stmt = select(CallQueue)
    
    if active_only:
        stmt = stmt.where(CallQueue.is_active == True)
    
    stmt = stmt.order_by(CallQueue.name).offset(skip).limit(limit)
    
    result = await db.execute(stmt)
    queues = result.scalars().all()
    
    return [CallQueueResponse.from_orm(queue) for queue in queues]


@router.post("/queues/{queue_id}/members", response_model=QueueMemberResponse)
async def add_queue_member(
    queue_id: UUID,
    member_data: QueueMemberCreateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Add member to call queue."""
    # Verify queue exists
    queue_stmt = select(CallQueue).where(CallQueue.id == queue_id)
    queue_result = await db.execute(queue_stmt)
    queue = queue_result.scalar_one_or_none()
    
    if not queue:
        raise HTTPException(status_code=404, detail="Queue not found")
    
    try:
        member = QueueMember(
            queue_id=queue_id,
            **member_data.dict(exclude={'queue_id'})
        )
        
        db.add(member)
        await db.commit()
        await db.refresh(member)
        
        return QueueMemberResponse.from_orm(member)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add queue member: {str(e)}")


@router.get("/queues/{queue_id}/members", response_model=List[QueueMemberResponse])
async def list_queue_members(
    queue_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """List queue members."""
    # Verify queue exists
    queue_stmt = select(CallQueue).where(CallQueue.id == queue_id)
    queue_result = await db.execute(queue_stmt)
    queue = queue_result.scalar_one_or_none()
    
    if not queue:
        raise HTTPException(status_code=404, detail="Queue not found")
    
    stmt = (
        select(QueueMember)
        .where(QueueMember.queue_id == queue_id)
        .order_by(QueueMember.priority.desc(), QueueMember.member_name)
    )
    
    result = await db.execute(stmt)
    members = result.scalars().all()
    
    return [QueueMemberResponse.from_orm(member) for member in members]


# Analytics
@router.get("/analytics")
async def get_telephony_analytics(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    campaign_id: Optional[UUID] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Get telephony analytics."""
    try:
        analytics = await call_service.get_call_analytics(
            start_date=start_date,
            end_date=end_date,
            campaign_id=campaign_id,
            db=db
        )
        return analytics
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


# WebSocket for real-time call monitoring
@router.websocket("/ws/monitor")
async def websocket_call_monitor(websocket: WebSocket):
    """WebSocket endpoint for real-time call monitoring."""
    await websocket.accept()
    
    # Store connection for call events
    call_service.add_call_callback(
        lambda event_type, call_info: _send_call_update(websocket, event_type, call_info)
    )
    
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            # Could handle commands like "subscribe_to_queue", "filter_calls", etc.
            
    except WebSocketDisconnect:
        # Remove callback when client disconnects
        pass
    except Exception as e:
        await websocket.close(code=1000)


async def _send_call_update(websocket: WebSocket, event_type: str, call_info):
    """Send call update via WebSocket."""
    try:
        update = {
            "event_type": event_type,
            "call_id": call_info.call_id,
            "state": call_info.state.value,
            "caller_number": call_info.caller_number,
            "called_number": call_info.called_number,
            "direction": call_info.direction,
            "duration": call_info.duration,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await websocket.send_json(update)
    
    except Exception:
        # WebSocket connection closed
        pass


# Status endpoints
@router.get("/status")
async def get_telephony_status():
    """Get telephony system status."""
    try:
        # Get active calls from SIP gateway
        active_calls = sip_gateway.get_active_calls()
        
        return {
            "sip_gateway_running": sip_gateway.is_running,
            "active_calls": len(active_calls),
            "max_concurrent_calls": settings.MAX_CONCURRENT_CALLS,
            "system_status": "healthy" if sip_gateway.is_running else "offline"
        }
    
    except Exception as e:
        return {
            "sip_gateway_running": False,
            "active_calls": 0,
            "max_concurrent_calls": settings.MAX_CONCURRENT_CALLS,
            "system_status": "error",
            "error": str(e)
        }