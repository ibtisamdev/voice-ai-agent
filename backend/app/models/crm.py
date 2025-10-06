from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timezone
import uuid

from .base import Base, TimestampMixin, SoftDeleteMixin


class CRMAccount(Base, TimestampMixin):
    """Zoho CRM account configuration and authentication tokens."""
    __tablename__ = "crm_accounts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    zoho_org_id = Column(String(100), unique=True, nullable=False)
    access_token = Column(Text, nullable=True)
    refresh_token = Column(Text, nullable=True)
    token_expires_at = Column(DateTime(timezone=True), nullable=True)
    api_domain = Column(String(255), default="https://www.zohoapis.com")
    is_active = Column(Boolean, default=True)
    sandbox_mode = Column(Boolean, default=True)
    
    # Relationships
    sync_jobs = relationship("CRMSyncJob", back_populates="account")
    field_mappings = relationship("CRMFieldMapping", back_populates="account")


class CRMSyncJob(Base, TimestampMixin):
    """Track CRM synchronization jobs and their status."""
    __tablename__ = "crm_sync_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    account_id = Column(UUID(as_uuid=True), ForeignKey("crm_accounts.id"), nullable=False)
    job_type = Column(String(50), nullable=False)  # sync_leads, sync_contacts, etc.
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    records_processed = Column(Integer, default=0)
    records_successful = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    account = relationship("CRMAccount", back_populates="sync_jobs")


class CRMFieldMapping(Base, TimestampMixin):
    """Map internal fields to CRM fields for data synchronization."""
    __tablename__ = "crm_field_mappings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    account_id = Column(UUID(as_uuid=True), ForeignKey("crm_accounts.id"), nullable=False)
    module_name = Column(String(50), nullable=False)  # Leads, Contacts, Deals, etc.
    internal_field = Column(String(100), nullable=False)
    crm_field = Column(String(100), nullable=False)
    field_type = Column(String(20), nullable=False)  # string, integer, boolean, datetime
    is_required = Column(Boolean, default=False)
    default_value = Column(String(255), nullable=True)
    transformation_rule = Column(Text, nullable=True)  # JSON transformation rules
    
    # Relationships
    account = relationship("CRMAccount", back_populates="field_mappings")


class Lead(Base, TimestampMixin, SoftDeleteMixin):
    """Internal lead management with CRM synchronization."""
    __tablename__ = "leads"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Lead Information
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    company = Column(String(255), nullable=True)
    title = Column(String(100), nullable=True)
    
    # Lead Source and Channel
    source = Column(String(50), nullable=False)  # phone_call, web_form, referral, etc.
    channel = Column(String(50), nullable=True)  # inbound, outbound, campaign_id
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=True)
    
    # Lead Status and Scoring
    status = Column(String(50), default="new")  # new, contacted, qualified, converted, lost
    lead_score = Column(Integer, default=0)
    priority = Column(String(20), default="medium")  # low, medium, high, urgent
    
    # Legal Practice Areas
    practice_areas = Column(JSON, nullable=True)  # ["personal_injury", "criminal_defense"]
    legal_issue = Column(Text, nullable=True)
    urgency_level = Column(String(20), default="normal")  # low, normal, high, emergency
    
    # Assignment and Follow-up
    assigned_attorney_id = Column(UUID(as_uuid=True), nullable=True)
    assigned_at = Column(DateTime(timezone=True), nullable=True)
    next_follow_up = Column(DateTime(timezone=True), nullable=True)
    follow_up_count = Column(Integer, default=0)
    
    # CRM Integration
    zoho_lead_id = Column(String(100), nullable=True, unique=True)
    last_synced_at = Column(DateTime(timezone=True), nullable=True)
    sync_status = Column(String(20), default="pending")  # pending, synced, error
    sync_error = Column(Text, nullable=True)
    
    # Conversation and Call Data
    conversation_session_id = Column(String(255), nullable=True)
    call_duration = Column(Integer, nullable=True)  # seconds
    call_quality_score = Column(Float, nullable=True)  # 0.0 - 1.0
    conversation_summary = Column(Text, nullable=True)
    
    # Additional Metadata
    notes = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)  # ["consultation", "urgent", "personal_injury"]
    custom_fields = Column(JSON, nullable=True)
    
    # Relationships
    campaign = relationship("Campaign", back_populates="leads")
    appointments = relationship("Appointment", back_populates="lead")
    activities = relationship("LeadActivity", back_populates="lead")


class LeadActivity(Base, TimestampMixin):
    """Track all activities and interactions with leads."""
    __tablename__ = "lead_activities"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lead_id = Column(UUID(as_uuid=True), ForeignKey("leads.id"), nullable=False)
    
    activity_type = Column(String(50), nullable=False)  # call, email, meeting, note, etc.
    direction = Column(String(20), nullable=True)  # inbound, outbound
    status = Column(String(20), nullable=False)  # completed, scheduled, cancelled
    
    subject = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    duration = Column(Integer, nullable=True)  # minutes
    
    scheduled_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Call-specific fields
    call_id = Column(String(255), nullable=True)
    phone_number = Column(String(50), nullable=True)
    call_outcome = Column(String(50), nullable=True)  # answered, voicemail, busy, no_answer
    recording_url = Column(String(500), nullable=True)
    
    # CRM Integration
    zoho_activity_id = Column(String(100), nullable=True)
    last_synced_at = Column(DateTime(timezone=True), nullable=True)
    
    created_by = Column(String(100), nullable=True)  # user_id or "system"
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    lead = relationship("Lead", back_populates="activities")


class Appointment(Base, TimestampMixin, SoftDeleteMixin):
    """Appointment scheduling and management."""
    __tablename__ = "appointments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lead_id = Column(UUID(as_uuid=True), ForeignKey("leads.id"), nullable=False)
    
    # Appointment Details
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    appointment_type = Column(String(50), nullable=False)  # consultation, follow_up, meeting
    
    # Scheduling
    scheduled_start = Column(DateTime(timezone=True), nullable=False)
    scheduled_end = Column(DateTime(timezone=True), nullable=False)
    timezone = Column(String(50), default="America/New_York")
    
    # Status and Confirmation
    status = Column(String(20), default="scheduled")  # scheduled, confirmed, completed, cancelled, no_show
    confirmation_status = Column(String(20), default="pending")  # pending, confirmed, declined
    confirmed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Attorney Assignment
    attorney_id = Column(UUID(as_uuid=True), nullable=True)
    attorney_name = Column(String(255), nullable=True)
    
    # Meeting Details
    meeting_type = Column(String(20), default="in_person")  # in_person, phone, video
    meeting_url = Column(String(500), nullable=True)
    meeting_phone = Column(String(50), nullable=True)
    location = Column(String(500), nullable=True)
    
    # Reminders
    reminder_sent = Column(Boolean, default=False)
    reminder_sent_at = Column(DateTime(timezone=True), nullable=True)
    
    # CRM Integration
    zoho_event_id = Column(String(100), nullable=True)
    calendar_event_id = Column(String(255), nullable=True)
    last_synced_at = Column(DateTime(timezone=True), nullable=True)
    
    # Additional Info
    notes = Column(Text, nullable=True)
    custom_fields = Column(JSON, nullable=True)
    
    # Relationships
    lead = relationship("Lead", back_populates="appointments")


class CRMWebhook(Base, TimestampMixin):
    """Track incoming webhooks from CRM system."""
    __tablename__ = "crm_webhooks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    webhook_id = Column(String(100), nullable=False)  # Zoho webhook ID
    event_type = Column(String(50), nullable=False)  # module.create, module.update, etc.
    module_name = Column(String(50), nullable=False)  # Leads, Contacts, Deals
    record_id = Column(String(100), nullable=False)  # CRM record ID
    
    payload = Column(JSON, nullable=False)
    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    processing_error = Column(Text, nullable=True)
    
    source_ip = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)


# Import Campaign model (will be created in campaign.py)
from sqlalchemy import Table

# This will be properly imported once campaign.py is created
# For now, we'll create a simple placeholder
class Campaign(Base, TimestampMixin):
    """Placeholder for Campaign model - will be implemented in campaign.py"""
    __tablename__ = "campaigns"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    status = Column(String(50), default="draft")
    
    # Relationships
    leads = relationship("Lead", back_populates="campaign")