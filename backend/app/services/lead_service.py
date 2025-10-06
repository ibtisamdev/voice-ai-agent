"""
Lead Management Service

Handles lead lifecycle management, scoring, assignment, and follow-up
for the Voice AI Agent system.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from ..core.config import settings
from ..models.crm import Lead, LeadActivity, Appointment, CRMAccount
from ..models.campaign import Campaign
from .crm_service import crm_service, ZohoCRMError

logger = logging.getLogger(__name__)


class LeadScoringService:
    """Service for lead scoring and qualification."""
    
    @staticmethod
    def calculate_lead_score(lead: Lead, conversation_data: Dict[str, Any] = None) -> int:
        """
        Calculate lead score based on various factors.
        
        Args:
            lead: Lead instance
            conversation_data: Optional conversation analysis data
            
        Returns:
            Lead score (0-100)
        """
        score = 0
        
        # Basic contact information completeness (20 points max)
        if lead.email:
            score += 5
        if lead.phone:
            score += 5
        if lead.company:
            score += 5
        if lead.title:
            score += 5
        
        # Legal issue urgency (25 points max)
        if lead.urgency_level == "emergency":
            score += 25
        elif lead.urgency_level == "high":
            score += 20
        elif lead.urgency_level == "normal":
            score += 10
        elif lead.urgency_level == "low":
            score += 5
        
        # Practice area value (20 points max)
        high_value_areas = ["personal_injury", "medical_malpractice", "class_action"]
        medium_value_areas = ["criminal_defense", "family_law", "employment"]
        
        if lead.practice_areas:
            for area in lead.practice_areas:
                if area in high_value_areas:
                    score += 20
                    break
                elif area in medium_value_areas:
                    score += 15
                    break
            else:
                score += 10  # Other practice areas
        
        # Call quality and engagement (20 points max)
        if lead.call_quality_score:
            score += int(lead.call_quality_score * 20)
        
        if lead.call_duration:
            # Longer calls typically indicate higher engagement
            if lead.call_duration > 300:  # 5+ minutes
                score += 10
            elif lead.call_duration > 180:  # 3+ minutes
                score += 7
            elif lead.call_duration > 60:   # 1+ minute
                score += 5
        
        # Conversation analysis (15 points max)
        if conversation_data:
            sentiment = conversation_data.get("sentiment", {})
            if sentiment.get("compound", 0) > 0.3:  # Positive sentiment
                score += 10
            
            intent_confidence = conversation_data.get("intent_confidence", 0)
            if intent_confidence > 0.8:
                score += 5
        
        return min(score, 100)  # Cap at 100
    
    @staticmethod
    def determine_priority(score: int, urgency_level: str) -> str:
        """
        Determine lead priority based on score and urgency.
        
        Args:
            score: Lead score (0-100)
            urgency_level: Urgency level string
            
        Returns:
            Priority level: urgent, high, medium, low
        """
        if urgency_level == "emergency" or score >= settings.HIGH_PRIORITY_SCORE_THRESHOLD:
            return "urgent"
        elif score >= 70:
            return "high"
        elif score >= 40:
            return "medium"
        else:
            return "low"


class LeadService:
    """
    Service class for lead management operations.
    
    Handles lead creation, scoring, assignment, follow-up scheduling,
    and CRM synchronization.
    """
    
    def __init__(self):
        self.scoring_service = LeadScoringService()
    
    async def create_lead_from_conversation(
        self,
        conversation_data: Dict[str, Any],
        conversation_session_id: str,
        db: AsyncSession
    ) -> Lead:
        """
        Create a new lead from conversation data.
        
        Args:
            conversation_data: Extracted conversation data
            conversation_session_id: Session ID for tracking
            db: Database session
            
        Returns:
            Created Lead instance
        """
        # Extract basic lead information
        lead_data = {
            "first_name": conversation_data.get("first_name", ""),
            "last_name": conversation_data.get("last_name", ""),
            "email": conversation_data.get("email"),
            "phone": conversation_data.get("phone"),
            "company": conversation_data.get("company"),
            "source": "phone_call",
            "channel": "inbound",
            "conversation_session_id": conversation_session_id,
            "call_duration": conversation_data.get("call_duration"),
            "call_quality_score": conversation_data.get("call_quality_score"),
            "conversation_summary": conversation_data.get("summary"),
            "legal_issue": conversation_data.get("legal_issue"),
            "practice_areas": conversation_data.get("practice_areas", []),
            "urgency_level": conversation_data.get("urgency_level", "normal"),
            "status": "new",
            "custom_fields": conversation_data.get("custom_fields", {})
        }
        
        # Create lead
        lead = Lead(**lead_data)
        
        # Calculate lead score
        lead.lead_score = self.scoring_service.calculate_lead_score(lead, conversation_data)
        lead.priority = self.scoring_service.determine_priority(lead.lead_score, lead.urgency_level)
        
        # Set follow-up time
        if lead.priority in ["urgent", "high"]:
            lead.next_follow_up = datetime.now(timezone.utc) + timedelta(hours=1)
        elif lead.priority == "medium":
            lead.next_follow_up = datetime.now(timezone.utc) + timedelta(hours=4)
        else:
            lead.next_follow_up = datetime.now(timezone.utc) + timedelta(hours=24)
        
        db.add(lead)
        await db.commit()
        await db.refresh(lead)
        
        # Auto-assign if enabled
        if settings.AUTO_ASSIGN_LEADS:
            await self.auto_assign_lead(lead, db)
        
        # Create initial activity record
        await self.add_lead_activity(
            lead_id=lead.id,
            activity_type="call",
            direction="inbound",
            status="completed",
            description="Initial conversation",
            call_outcome="answered",
            duration=conversation_data.get("call_duration"),
            db=db
        )
        
        logger.info(f"Created new lead {lead.id} from conversation {conversation_session_id}")
        return lead
    
    async def auto_assign_lead(self, lead: Lead, db: AsyncSession) -> Optional[str]:
        """
        Automatically assign lead to available attorney.
        
        Args:
            lead: Lead to assign
            db: Database session
            
        Returns:
            Assigned attorney ID if successful
        """
        # This is a placeholder for attorney assignment logic
        # In a real implementation, this would:
        # 1. Query available attorneys
        # 2. Check their specializations vs practice areas
        # 3. Check current workload
        # 4. Assign based on round-robin or other logic
        
        # For now, we'll just mark the assignment timestamp
        lead.assigned_at = datetime.now(timezone.utc)
        await db.commit()
        
        return None  # Would return actual attorney ID
    
    async def update_lead_score(self, lead: Lead, db: AsyncSession) -> int:
        """
        Recalculate and update lead score.
        
        Args:
            lead: Lead to update
            db: Database session
            
        Returns:
            New lead score
        """
        old_score = lead.lead_score
        lead.lead_score = self.scoring_service.calculate_lead_score(lead)
        lead.priority = self.scoring_service.determine_priority(lead.lead_score, lead.urgency_level)
        
        await db.commit()
        
        if lead.lead_score != old_score:
            logger.info(f"Updated lead {lead.id} score from {old_score} to {lead.lead_score}")
        
        return lead.lead_score
    
    async def add_lead_activity(
        self,
        lead_id: UUID,
        activity_type: str,
        direction: str = None,
        status: str = "completed",
        subject: str = None,
        description: str = None,
        duration: int = None,
        call_outcome: str = None,
        call_id: str = None,
        phone_number: str = None,
        recording_url: str = None,
        metadata: Dict[str, Any] = None,
        db: AsyncSession = None
    ) -> LeadActivity:
        """
        Add activity record to lead.
        
        Args:
            lead_id: Lead UUID
            activity_type: Type of activity (call, email, meeting, note)
            direction: inbound/outbound (for calls)
            status: Activity status
            subject: Activity subject/title
            description: Detailed description
            duration: Duration in minutes
            call_outcome: Call outcome if applicable
            call_id: Call system ID
            phone_number: Phone number used
            recording_url: URL to call recording
            metadata: Additional metadata
            db: Database session
            
        Returns:
            Created LeadActivity instance
        """
        activity = LeadActivity(
            lead_id=lead_id,
            activity_type=activity_type,
            direction=direction,
            status=status,
            subject=subject,
            description=description,
            duration=duration,
            call_id=call_id,
            phone_number=phone_number,
            call_outcome=call_outcome,
            recording_url=recording_url,
            completed_at=datetime.now(timezone.utc) if status == "completed" else None,
            created_by="system",
            metadata=metadata or {}
        )
        
        db.add(activity)
        await db.commit()
        await db.refresh(activity)
        
        logger.info(f"Added {activity_type} activity to lead {lead_id}")
        return activity
    
    async def schedule_follow_up(
        self,
        lead: Lead,
        follow_up_time: datetime,
        activity_type: str = "call",
        notes: str = None,
        db: AsyncSession = None
    ) -> LeadActivity:
        """
        Schedule follow-up activity for lead.
        
        Args:
            lead: Lead instance
            follow_up_time: When to follow up
            activity_type: Type of follow-up activity
            notes: Additional notes
            db: Database session
            
        Returns:
            Scheduled LeadActivity
        """
        activity = LeadActivity(
            lead_id=lead.id,
            activity_type=activity_type,
            status="scheduled",
            subject=f"Follow-up {activity_type}",
            description=notes,
            scheduled_at=follow_up_time,
            created_by="system"
        )
        
        db.add(activity)
        
        # Update lead's next follow-up time
        lead.next_follow_up = follow_up_time
        lead.follow_up_count += 1
        
        await db.commit()
        await db.refresh(activity)
        
        logger.info(f"Scheduled {activity_type} follow-up for lead {lead.id} at {follow_up_time}")
        return activity
    
    async def get_leads_for_follow_up(
        self,
        limit: int = 50,
        db: AsyncSession = None
    ) -> List[Lead]:
        """
        Get leads that need follow-up.
        
        Args:
            limit: Maximum number of leads to return
            db: Database session
            
        Returns:
            List of leads needing follow-up
        """
        now = datetime.now(timezone.utc)
        
        stmt = (
            select(Lead)
            .where(
                and_(
                    Lead.next_follow_up <= now,
                    Lead.status.in_(["new", "contacted", "qualified"]),
                    Lead.is_deleted == False
                )
            )
            .order_by(Lead.priority.desc(), Lead.next_follow_up)
            .limit(limit)
        )
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def sync_lead_to_crm(
        self,
        lead: Lead,
        account_id: UUID = None,
        db: AsyncSession = None
    ) -> bool:
        """
        Synchronize lead to CRM system.
        
        Args:
            lead: Lead to synchronize
            account_id: CRM account ID (optional)
            db: Database session
            
        Returns:
            True if sync successful
        """
        try:
            # Get CRM account
            if account_id:
                stmt = select(CRMAccount).where(CRMAccount.id == account_id)
            else:
                stmt = select(CRMAccount).where(CRMAccount.is_active == True).limit(1)
            
            result = await db.execute(stmt)
            account = result.scalar_one_or_none()
            
            if not account:
                logger.warning(f"No active CRM account found for lead {lead.id}")
                return False
            
            # Sync using CRM service
            async with crm_service as crm:
                await crm.sync_lead_to_zoho(lead, account, db)
            
            return True
            
        except ZohoCRMError as e:
            logger.error(f"CRM sync failed for lead {lead.id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error syncing lead {lead.id}: {e}")
            return False
    
    async def get_lead_analytics(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """
        Get lead analytics for specified period.
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            db: Database session
            
        Returns:
            Dictionary with analytics data
        """
        if not start_date:
            start_date = datetime.now(timezone.utc) - timedelta(days=30)
        if not end_date:
            end_date = datetime.now(timezone.utc)
        
        # Total leads created
        total_leads_stmt = (
            select(func.count(Lead.id))
            .where(
                and_(
                    Lead.created_at >= start_date,
                    Lead.created_at <= end_date,
                    Lead.is_deleted == False
                )
            )
        )
        total_leads = await db.scalar(total_leads_stmt)
        
        # Leads by source
        source_stmt = (
            select(Lead.source, func.count(Lead.id))
            .where(
                and_(
                    Lead.created_at >= start_date,
                    Lead.created_at <= end_date,
                    Lead.is_deleted == False
                )
            )
            .group_by(Lead.source)
        )
        source_result = await db.execute(source_stmt)
        leads_by_source = dict(source_result.all())
        
        # Leads by status
        status_stmt = (
            select(Lead.status, func.count(Lead.id))
            .where(
                and_(
                    Lead.created_at >= start_date,
                    Lead.created_at <= end_date,
                    Lead.is_deleted == False
                )
            )
            .group_by(Lead.status)
        )
        status_result = await db.execute(status_stmt)
        leads_by_status = dict(status_result.all())
        
        # Average lead score
        avg_score_stmt = (
            select(func.avg(Lead.lead_score))
            .where(
                and_(
                    Lead.created_at >= start_date,
                    Lead.created_at <= end_date,
                    Lead.is_deleted == False
                )
            )
        )
        avg_score = await db.scalar(avg_score_stmt) or 0
        
        # Conversion metrics
        converted_leads_stmt = (
            select(func.count(Lead.id))
            .where(
                and_(
                    Lead.created_at >= start_date,
                    Lead.created_at <= end_date,
                    Lead.status == "converted",
                    Lead.is_deleted == False
                )
            )
        )
        converted_leads = await db.scalar(converted_leads_stmt)
        
        conversion_rate = (converted_leads / total_leads * 100) if total_leads > 0 else 0
        
        return {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "total_leads": total_leads,
            "converted_leads": converted_leads,
            "conversion_rate": round(conversion_rate, 2),
            "average_score": round(avg_score, 1),
            "leads_by_source": leads_by_source,
            "leads_by_status": leads_by_status
        }
    
    async def update_lead_status(
        self,
        lead: Lead,
        new_status: str,
        notes: str = None,
        db: AsyncSession = None
    ) -> Lead:
        """
        Update lead status and log activity.
        
        Args:
            lead: Lead to update
            new_status: New status value
            notes: Optional notes about status change
            db: Database session
            
        Returns:
            Updated Lead instance
        """
        old_status = lead.status
        lead.status = new_status
        
        # Add status change activity
        await self.add_lead_activity(
            lead_id=lead.id,
            activity_type="note",
            status="completed",
            subject=f"Status changed from {old_status} to {new_status}",
            description=notes,
            db=db
        )
        
        await db.commit()
        
        logger.info(f"Updated lead {lead.id} status from {old_status} to {new_status}")
        return lead


# Singleton instance
lead_service = LeadService()