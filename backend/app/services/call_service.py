"""
Call Management Service

Handles call orchestration, routing, recording, and integration
with the Voice AI Agent system.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from uuid import UUID, uuid4
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from ..core.config import settings
from ..models.telephony import CallRecord, CallEvent, SIPAccount, CallQueue, QueueCall
from ..models.crm import Lead, LeadActivity
from ..models.campaign import Campaign, CampaignContact
from .lead_service import lead_service
from telephony.sip_gateway import sip_gateway, CallInfo, CallState

logger = logging.getLogger(__name__)


class CallRoutingService:
    """Service for intelligent call routing."""
    
    @staticmethod
    async def route_inbound_call(
        caller_number: str,
        called_number: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Route inbound call based on business rules.
        
        Args:
            caller_number: Caller's phone number
            called_number: Number that was called
            db: Database session
            
        Returns:
            Routing decision with destination and metadata
        """
        # Check if caller is an existing lead
        stmt = select(Lead).where(Lead.phone == caller_number).order_by(desc(Lead.created_at))
        result = await db.execute(stmt)
        existing_lead = result.scalar_one_or_none()
        
        # Default routing to AI agent
        routing = {
            "destination": "ai_agent",
            "queue_id": None,
            "priority": "normal",
            "metadata": {
                "existing_lead": existing_lead.id if existing_lead else None,
                "caller_history": bool(existing_lead)
            }
        }
        
        # High priority routing for existing high-value leads
        if existing_lead and existing_lead.priority in ["urgent", "high"]:
            routing["priority"] = "high"
            routing["metadata"]["lead_priority"] = existing_lead.priority
        
        # Route to specific queue based on called number
        if called_number.endswith("1"):  # Personal injury line
            routing["queue_id"] = "personal_injury"
        elif called_number.endswith("2"):  # Criminal defense line
            routing["queue_id"] = "criminal_defense"
        
        logger.info(f"Routed call from {caller_number} to {routing['destination']}")
        return routing
    
    @staticmethod
    async def find_available_agent(
        queue_id: str = None,
        practice_area: str = None,
        db: AsyncSession = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find available agent/attorney for call transfer.
        
        Args:
            queue_id: Specific queue to search
            practice_area: Required practice area
            db: Database session
            
        Returns:
            Agent info if available
        """
        # This would implement actual agent/attorney lookup logic
        # For now, return mock data
        return {
            "agent_id": "attorney_001",
            "name": "John Smith",
            "extension": "101",
            "specializations": ["personal_injury", "medical_malpractice"]
        }


class CallService:
    """
    Service class for call management operations.
    
    Handles call creation, tracking, routing, recording, and AI integration.
    """
    
    def __init__(self):
        self.routing_service = CallRoutingService()
        self.active_ai_sessions: Dict[str, str] = {}  # call_id -> session_id
        self.call_callbacks: List[Callable] = []
    
    async def initialize(self):
        """Initialize call service and SIP gateway."""
        await sip_gateway.initialize()
        
        # Register SIP account if configured
        if all([settings.SIP_SERVER, settings.SIP_USERNAME, settings.SIP_PASSWORD]):
            sip_config = {
                "server": settings.SIP_SERVER,
                "username": settings.SIP_USERNAME,
                "password": settings.SIP_PASSWORD,
                "domain": settings.SIP_DOMAIN or settings.SIP_SERVER,
                "port": settings.SIP_PORT
            }
            
            await sip_gateway.register_account(sip_config)
            logger.info("SIP account registered successfully")
        
        # Setup call event callbacks
        sip_gateway.add_call_callback(self._handle_call_event)
    
    async def shutdown(self):
        """Shutdown call service."""
        await sip_gateway.shutdown()
    
    def add_call_callback(self, callback: Callable):
        """Add callback for call events."""
        self.call_callbacks.append(callback)
    
    async def handle_inbound_call(
        self,
        call_info: CallInfo,
        db: AsyncSession
    ) -> CallRecord:
        """
        Handle incoming call and create call record.
        
        Args:
            call_info: Call information from SIP gateway
            db: Database session
            
        Returns:
            Created CallRecord
        """
        # Route the call
        routing = await self.routing_service.route_inbound_call(
            call_info.caller_number,
            call_info.called_number,
            db
        )
        
        # Create call record
        call_record = CallRecord(
            call_id=call_info.call_id,
            direction="inbound",
            caller_number=call_info.caller_number,
            called_number=call_info.called_number,
            initiated_at=call_info.start_time,
            status="ringing",
            sip_call_id=call_info.sip_call_id
        )
        
        # Link to existing lead if found
        if routing["metadata"].get("existing_lead"):
            call_record.lead_id = routing["metadata"]["existing_lead"]
        
        db.add(call_record)
        await db.commit()
        await db.refresh(call_record)
        
        # Log call event
        await self._log_call_event(
            call_record.id,
            "CALL_INITIATED",
            {"routing": routing},
            db
        )
        
        # Start AI session for the call
        await self._start_ai_session(call_record, db)
        
        logger.info(f"Inbound call handled: {call_record.call_id}")
        return call_record
    
    async def make_outbound_call(
        self,
        destination: str,
        campaign_id: UUID = None,
        lead_id: UUID = None,
        caller_id: str = None,
        db: AsyncSession = None
    ) -> Optional[CallRecord]:
        """
        Initiate outbound call.
        
        Args:
            destination: Phone number to call
            campaign_id: Associated campaign ID
            lead_id: Associated lead ID
            caller_id: Caller ID to present
            db: Database session
            
        Returns:
            CallRecord if call initiated successfully
        """
        try:
            # Make the call via SIP gateway
            call_data = {
                "campaign_id": str(campaign_id) if campaign_id else None,
                "lead_id": str(lead_id) if lead_id else None
            }
            
            call_id = await sip_gateway.make_call(
                account_id=settings.SIP_USERNAME,
                destination=destination,
                call_data=call_data
            )
            
            if not call_id:
                logger.error(f"Failed to initiate outbound call to {destination}")
                return None
            
            # Create call record
            call_record = CallRecord(
                call_id=call_id,
                direction="outbound",
                caller_number=caller_id or settings.SIP_USERNAME,
                called_number=destination,
                initiated_at=datetime.now(timezone.utc),
                status="calling",
                campaign_id=campaign_id,
                lead_id=lead_id
            )
            
            db.add(call_record)
            await db.commit()
            await db.refresh(call_record)
            
            # Log call event
            await self._log_call_event(
                call_record.id,
                "OUTBOUND_INITIATED",
                {"destination": destination},
                db
            )
            
            logger.info(f"Outbound call initiated: {call_id} to {destination}")
            return call_record
            
        except Exception as e:
            logger.error(f"Error making outbound call to {destination}: {e}")
            return None
    
    async def update_call_status(
        self,
        call_id: str,
        status: str,
        disposition: str = None,
        hangup_cause: str = None,
        db: AsyncSession = None
    ):
        """
        Update call status and log event.
        
        Args:
            call_id: Call ID
            status: New status
            disposition: Call disposition
            hangup_cause: Reason for hangup
            db: Database session
        """
        stmt = select(CallRecord).where(CallRecord.call_id == call_id)
        result = await db.execute(stmt)
        call_record = result.scalar_one_or_none()
        
        if not call_record:
            logger.warning(f"Call record not found: {call_id}")
            return
        
        old_status = call_record.status
        call_record.status = status
        
        if disposition:
            call_record.disposition = disposition
        
        if hangup_cause:
            call_record.hangup_cause = hangup_cause
        
        # Update timing
        now = datetime.now(timezone.utc)
        if status == "answered" and not call_record.answered_at:
            call_record.answered_at = now
        elif status in ["ended", "failed", "cancelled"] and not call_record.ended_at:
            call_record.ended_at = now
            
            # Calculate duration
            if call_record.answered_at:
                duration = call_record.ended_at - call_record.answered_at
                call_record.duration = int(duration.total_seconds())
        
        await db.commit()
        
        # Log status change event
        await self._log_call_event(
            call_record.id,
            "STATUS_CHANGE",
            {
                "old_status": old_status,
                "new_status": status,
                "disposition": disposition,
                "hangup_cause": hangup_cause
            },
            db
        )
        
        # Handle call completion
        if status in ["ended", "failed", "cancelled"]:
            await self._handle_call_completion(call_record, db)
    
    async def transfer_call(
        self,
        call_id: str,
        destination: str,
        transfer_type: str = "blind",
        db: AsyncSession = None
    ) -> bool:
        """
        Transfer call to another destination.
        
        Args:
            call_id: Call ID to transfer
            destination: Transfer destination
            transfer_type: Type of transfer (blind, attended)
            db: Database session
            
        Returns:
            True if transfer successful
        """
        try:
            # Execute transfer via SIP gateway
            await sip_gateway.transfer_call(call_id, destination)
            
            # Update call record
            stmt = select(CallRecord).where(CallRecord.call_id == call_id)
            result = await db.execute(stmt)
            call_record = result.scalar_one_or_none()
            
            if call_record:
                call_record.transfer_count += 1
                call_record.transferred_to = destination
                await db.commit()
                
                # Log transfer event
                await self._log_call_event(
                    call_record.id,
                    "CALL_TRANSFERRED",
                    {
                        "destination": destination,
                        "transfer_type": transfer_type
                    },
                    db
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error transferring call {call_id}: {e}")
            return False
    
    async def hangup_call(self, call_id: str, reason: str = None, db: AsyncSession = None):
        """
        Hangup call.
        
        Args:
            call_id: Call ID to hangup
            reason: Reason for hangup
            db: Database session
        """
        try:
            await sip_gateway.hangup_call(call_id)
            await self.update_call_status(
                call_id,
                "ended",
                hangup_cause=reason or "NORMAL_CLEARING",
                db=db
            )
        except Exception as e:
            logger.error(f"Error hanging up call {call_id}: {e}")
    
    async def send_dtmf(self, call_id: str, digits: str, db: AsyncSession = None):
        """
        Send DTMF tones to call.
        
        Args:
            call_id: Call ID
            digits: DTMF digits to send
            db: Database session
        """
        try:
            await sip_gateway.send_dtmf(call_id, digits)
            
            # Log DTMF event
            stmt = select(CallRecord).where(CallRecord.call_id == call_id)
            result = await db.execute(stmt)
            call_record = result.scalar_one_or_none()
            
            if call_record:
                await self._log_call_event(
                    call_record.id,
                    "DTMF_SENT",
                    {"digits": digits},
                    db
                )
            
        except Exception as e:
            logger.error(f"Error sending DTMF to call {call_id}: {e}")
    
    async def get_call_analytics(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        campaign_id: UUID = None,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """
        Get call analytics for specified period.
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            campaign_id: Filter by campaign
            db: Database session
            
        Returns:
            Dictionary with analytics data
        """
        if not start_date:
            start_date = datetime.now(timezone.utc) - timedelta(days=30)
        if not end_date:
            end_date = datetime.now(timezone.utc)
        
        # Build base query
        base_conditions = [
            CallRecord.initiated_at >= start_date,
            CallRecord.initiated_at <= end_date
        ]
        
        if campaign_id:
            base_conditions.append(CallRecord.campaign_id == campaign_id)
        
        # Total calls
        total_calls_stmt = select(func.count(CallRecord.id)).where(and_(*base_conditions))
        total_calls = await db.scalar(total_calls_stmt)
        
        # Answered calls
        answered_calls_stmt = select(func.count(CallRecord.id)).where(
            and_(*base_conditions, CallRecord.status == "answered")
        )
        answered_calls = await db.scalar(answered_calls_stmt)
        
        # Average duration
        avg_duration_stmt = select(func.avg(CallRecord.duration)).where(
            and_(*base_conditions, CallRecord.duration.isnot(None))
        )
        avg_duration = await db.scalar(avg_duration_stmt) or 0
        
        # Call outcomes
        outcome_stmt = select(
            CallRecord.disposition,
            func.count(CallRecord.id)
        ).where(and_(*base_conditions)).group_by(CallRecord.disposition)
        
        outcome_result = await db.execute(outcome_stmt)
        call_outcomes = dict(outcome_result.all())
        
        # Calculate rates
        answer_rate = (answered_calls / total_calls * 100) if total_calls > 0 else 0
        
        return {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "total_calls": total_calls,
            "answered_calls": answered_calls,
            "answer_rate": round(answer_rate, 2),
            "average_duration": round(avg_duration, 1),
            "call_outcomes": call_outcomes
        }
    
    # Private helper methods
    
    async def _handle_call_event(self, event_type: str, call_info: CallInfo):
        """Handle call events from SIP gateway."""
        try:
            # This method processes events from the SIP gateway
            # and updates the database accordingly
            
            # Get database session (would need proper session management)
            # For now, this is a placeholder for the event handling logic
            
            logger.info(f"Call event: {event_type} for call {call_info.call_id}")
            
            # Notify registered callbacks
            for callback in self.call_callbacks:
                try:
                    await callback(event_type, call_info)
                except Exception as e:
                    logger.error(f"Call callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling call event: {e}")
    
    async def _start_ai_session(self, call_record: CallRecord, db: AsyncSession):
        """Start AI session for call."""
        try:
            # Generate AI session ID
            ai_session_id = str(uuid4())
            self.active_ai_sessions[call_record.call_id] = ai_session_id
            
            # This would integrate with the existing voice AI WebSocket system
            # The AI session would handle:
            # - Audio processing (STT/TTS)
            # - Conversation management
            # - Intent classification
            # - Response generation
            
            logger.info(f"Started AI session {ai_session_id} for call {call_record.call_id}")
            
        except Exception as e:
            logger.error(f"Error starting AI session for call {call_record.call_id}: {e}")
    
    async def _handle_call_completion(self, call_record: CallRecord, db: AsyncSession):
        """Handle call completion tasks."""
        try:
            # End AI session
            if call_record.call_id in self.active_ai_sessions:
                ai_session_id = self.active_ai_sessions[call_record.call_id]
                del self.active_ai_sessions[call_record.call_id]
                logger.info(f"Ended AI session {ai_session_id}")
            
            # Create or update lead if this was an inbound call
            if call_record.direction == "inbound" and not call_record.lead_id:
                # Extract conversation data (would come from AI session)
                conversation_data = {
                    "phone": call_record.caller_number,
                    "call_duration": call_record.duration,
                    "call_quality_score": call_record.audio_quality_score
                }
                
                # Create lead
                lead = await lead_service.create_lead_from_conversation(
                    conversation_data,
                    call_record.call_id,
                    db
                )
                
                # Link call to lead
                call_record.lead_id = lead.id
                await db.commit()
            
            # Update campaign contact if applicable
            if call_record.campaign_id and call_record.direction == "outbound":
                await self._update_campaign_contact_status(call_record, db)
            
            logger.info(f"Call completion handling finished for {call_record.call_id}")
            
        except Exception as e:
            logger.error(f"Error handling call completion: {e}")
    
    async def _update_campaign_contact_status(self, call_record: CallRecord, db: AsyncSession):
        """Update campaign contact based on call outcome."""
        if not call_record.campaign_id:
            return
        
        # Find campaign contact
        stmt = select(CampaignContact).where(
            and_(
                CampaignContact.phone_number == call_record.called_number,
                CampaignContact.campaign_list_id.in_(
                    select(CampaignList.id).where(
                        CampaignList.campaign_id == call_record.campaign_id
                    )
                )
            )
        )
        result = await db.execute(stmt)
        contact = result.scalar_one_or_none()
        
        if contact:
            contact.call_attempts += 1
            contact.last_call_attempt = call_record.initiated_at
            contact.contact_outcome = call_record.disposition
            
            # Update status based on outcome
            if call_record.disposition == "answered":
                contact.call_status = "contacted"
            elif call_record.disposition in ["busy", "no_answer"]:
                contact.call_status = "pending"  # Can retry
            else:
                contact.call_status = "failed"
            
            await db.commit()
    
    async def _log_call_event(
        self,
        call_record_id: UUID,
        event_type: str,
        event_data: Dict[str, Any],
        db: AsyncSession
    ):
        """Log call event to database."""
        try:
            event = CallEvent(
                call_record_id=call_record_id,
                event_type=event_type,
                event_time=datetime.now(timezone.utc),
                event_data=event_data
            )
            
            db.add(event)
            await db.commit()
            
        except Exception as e:
            logger.error(f"Error logging call event: {e}")


# Singleton instance
call_service = CallService()