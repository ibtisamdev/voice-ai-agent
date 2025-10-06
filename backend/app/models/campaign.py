from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timezone
import uuid

from .base import Base, TimestampMixin, SoftDeleteMixin


class Campaign(Base, TimestampMixin, SoftDeleteMixin):
    """Marketing and outbound call campaigns."""
    __tablename__ = "campaigns"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Campaign Basic Info
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    campaign_type = Column(String(50), nullable=False)  # outbound_sales, follow_up, survey, etc.
    
    # Status and Lifecycle
    status = Column(String(20), default="draft")  # draft, scheduled, running, paused, completed, cancelled
    created_by = Column(String(100), nullable=False)  # user ID
    
    # Scheduling
    scheduled_start = Column(DateTime(timezone=True), nullable=True)
    scheduled_end = Column(DateTime(timezone=True), nullable=True)
    actual_start = Column(DateTime(timezone=True), nullable=True)
    actual_end = Column(DateTime(timezone=True), nullable=True)
    
    # Calling Configuration
    dialer_mode = Column(String(20), default="progressive")  # progressive, predictive, preview, manual
    max_concurrent_calls = Column(Integer, default=10)
    call_timeout = Column(Integer, default=30)  # seconds
    retry_attempts = Column(Integer, default=3)
    retry_delay = Column(Integer, default=3600)  # seconds between retries
    
    # Time Zone and Schedule Settings
    timezone = Column(String(50), default="America/New_York")
    calling_hours_start = Column(String(5), default="09:00")  # HH:MM
    calling_hours_end = Column(String(5), default="18:00")  # HH:MM
    calling_days = Column(String(20), default="1,2,3,4,5")  # Monday=1, Sunday=7
    
    # Lead Management
    auto_create_leads = Column(Boolean, default=True)
    lead_source = Column(String(50), default="campaign")
    practice_area = Column(String(100), nullable=True)
    
    # Script and Flow
    script_template = Column(Text, nullable=True)
    conversation_flow_id = Column(String(100), nullable=True)
    fallback_message = Column(Text, nullable=True)
    
    # Compliance and DNC
    respect_dnc = Column(Boolean, default=True)
    max_abandon_rate = Column(Float, default=0.03)  # 3% max abandon rate
    compliance_notes = Column(Text, nullable=True)
    
    # Performance Targets
    target_contacts = Column(Integer, nullable=True)
    target_conversations = Column(Integer, nullable=True)
    target_appointments = Column(Integer, nullable=True)
    target_conversion_rate = Column(Float, nullable=True)
    
    # Analytics and Results
    total_leads = Column(Integer, default=0)
    calls_attempted = Column(Integer, default=0)
    calls_connected = Column(Integer, default=0)
    conversations_completed = Column(Integer, default=0)
    appointments_scheduled = Column(Integer, default=0)
    conversion_rate = Column(Float, default=0.0)
    average_call_duration = Column(Float, default=0.0)  # seconds
    
    # Cost Tracking
    estimated_cost = Column(Float, nullable=True)
    actual_cost = Column(Float, default=0.0)
    cost_per_lead = Column(Float, nullable=True)
    cost_per_appointment = Column(Float, nullable=True)
    currency = Column(String(3), default="USD")
    
    # Custom Fields and Metadata
    tags = Column(JSON, nullable=True)
    custom_fields = Column(JSON, nullable=True)
    campaign_metadata = Column(JSON, nullable=True)
    
    # Relationships
    leads = relationship("Lead", back_populates="campaign")
    call_records = relationship("CallRecord", back_populates="campaign")
    campaign_lists = relationship("CampaignList", back_populates="campaign")
    campaign_schedules = relationship("CampaignSchedule", back_populates="campaign")


class CampaignList(Base, TimestampMixin):
    """Lists of contacts/numbers to call for campaigns."""
    __tablename__ = "campaign_lists"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False)
    
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # List Source
    source_type = Column(String(50), nullable=False)  # upload, crm_import, manual, api
    source_file = Column(String(500), nullable=True)  # original file path
    imported_at = Column(DateTime(timezone=True), nullable=True)
    imported_by = Column(String(100), nullable=True)
    
    # List Statistics
    total_contacts = Column(Integer, default=0)
    valid_contacts = Column(Integer, default=0)
    invalid_contacts = Column(Integer, default=0)
    dnc_contacts = Column(Integer, default=0)
    
    # Processing Status
    processing_status = Column(String(20), default="pending")  # pending, processing, completed, error
    processing_error = Column(Text, nullable=True)
    
    # Validation Results
    validation_completed = Column(Boolean, default=False)
    validation_summary = Column(JSON, nullable=True)
    
    # Custom Fields
    custom_fields = Column(JSON, nullable=True)
    
    # Relationships
    campaign = relationship("Campaign", back_populates="campaign_lists")
    contacts = relationship("CampaignContact", back_populates="campaign_list")


class CampaignContact(Base, TimestampMixin):
    """Individual contacts within campaign lists."""
    __tablename__ = "campaign_contacts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_list_id = Column(UUID(as_uuid=True), ForeignKey("campaign_lists.id"), nullable=False)
    lead_id = Column(UUID(as_uuid=True), ForeignKey("leads.id"), nullable=True)
    
    # Contact Information
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    phone_number = Column(String(50), nullable=False)
    email = Column(String(255), nullable=True)
    company = Column(String(255), nullable=True)
    title = Column(String(100), nullable=True)
    
    # Call Status
    call_status = Column(String(20), default="pending")  # pending, calling, contacted, failed, dnc
    call_attempts = Column(Integer, default=0)
    last_call_attempt = Column(DateTime(timezone=True), nullable=True)
    next_call_scheduled = Column(DateTime(timezone=True), nullable=True)
    
    # Call Results
    contact_outcome = Column(String(50), nullable=True)  # answered, voicemail, busy, no_answer, invalid
    conversation_completed = Column(Boolean, default=False)
    appointment_scheduled = Column(Boolean, default=False)
    follow_up_required = Column(Boolean, default=False)
    
    # DNC and Compliance
    dnc_status = Column(String(20), default="unknown")  # clear, listed, unknown, error
    dnc_checked_at = Column(DateTime(timezone=True), nullable=True)
    opt_out_requested = Column(Boolean, default=False)
    opt_out_date = Column(DateTime(timezone=True), nullable=True)
    
    # Priority and Scoring
    priority = Column(Integer, default=1)  # 1=low, 5=high
    lead_score = Column(Integer, nullable=True)
    contact_quality = Column(Float, nullable=True)  # 0.0 - 1.0
    
    # Timing Preferences
    preferred_call_time = Column(String(50), nullable=True)  # "morning", "afternoon", "evening"
    timezone = Column(String(50), nullable=True)
    do_not_call_before = Column(String(5), nullable=True)  # HH:MM
    do_not_call_after = Column(String(5), nullable=True)   # HH:MM
    
    # Additional Data
    notes = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)
    custom_data = Column(JSON, nullable=True)  # Additional imported fields
    
    # Validation Status
    phone_validated = Column(Boolean, default=False)
    phone_valid = Column(Boolean, nullable=True)
    phone_type = Column(String(20), nullable=True)  # mobile, landline, voip
    carrier = Column(String(100), nullable=True)
    
    # Relationships
    campaign_list = relationship("CampaignList", back_populates="contacts")
    lead = relationship("Lead")


class CampaignSchedule(Base, TimestampMixin):
    """Advanced scheduling rules for campaigns."""
    __tablename__ = "campaign_schedules"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False)
    
    # Schedule Type
    schedule_type = Column(String(20), nullable=False)  # daily, weekly, monthly, custom
    
    # Recurrence Rules
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=True)
    recurrence_pattern = Column(String(100), nullable=True)  # RRULE format
    
    # Daily Schedule
    daily_start_time = Column(String(5), nullable=True)  # HH:MM
    daily_end_time = Column(String(5), nullable=True)    # HH:MM
    daily_max_calls = Column(Integer, nullable=True)
    
    # Weekly Schedule  
    monday_enabled = Column(Boolean, default=True)
    tuesday_enabled = Column(Boolean, default=True)
    wednesday_enabled = Column(Boolean, default=True)
    thursday_enabled = Column(Boolean, default=True)
    friday_enabled = Column(Boolean, default=True)
    saturday_enabled = Column(Boolean, default=False)
    sunday_enabled = Column(Boolean, default=False)
    
    # Time Zone Settings
    timezone = Column(String(50), default="America/New_York")
    respect_contact_timezone = Column(Boolean, default=True)
    
    # Exclusions
    exclude_holidays = Column(Boolean, default=True)
    holiday_calendar = Column(String(50), default="US")  # US, CA, UK, etc.
    custom_exclusions = Column(JSON, nullable=True)  # Array of date strings
    
    # Pacing and Throttling
    calls_per_hour = Column(Integer, nullable=True)
    calls_per_day = Column(Integer, nullable=True)
    calls_per_week = Column(Integer, nullable=True)
    concurrent_call_limit = Column(Integer, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    last_execution = Column(DateTime(timezone=True), nullable=True)
    next_execution = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    campaign = relationship("Campaign", back_populates="campaign_schedules")


class DoNotCallList(Base, TimestampMixin):
    """Do Not Call (DNC) registry management."""
    __tablename__ = "dnc_list"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Phone Number (normalized)
    phone_number = Column(String(20), nullable=False, unique=True)
    original_format = Column(String(50), nullable=True)
    country_code = Column(String(5), default="+1")
    
    # DNC Source
    source = Column(String(50), nullable=False)  # federal, state, internal, opt_out
    source_details = Column(String(255), nullable=True)
    
    # Registration Details
    registered_date = Column(DateTime(timezone=True), nullable=False)
    expiry_date = Column(DateTime(timezone=True), nullable=True)
    last_verified = Column(DateTime(timezone=True), nullable=True)
    
    # Status
    status = Column(String(20), default="active")  # active, expired, removed
    
    # Opt-out Details (for internal DNC)
    opt_out_campaign_id = Column(UUID(as_uuid=True), nullable=True)
    opt_out_reason = Column(String(100), nullable=True)
    opt_out_method = Column(String(20), nullable=True)  # voice, email, web, sms
    
    # Contact Information (if available)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    email = Column(String(255), nullable=True)
    
    # Additional Metadata
    notes = Column(Text, nullable=True)
    custom_fields = Column(JSON, nullable=True)


class CampaignAnalytics(Base, TimestampMixin):
    """Daily analytics snapshots for campaigns."""
    __tablename__ = "campaign_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False)
    
    # Analytics Date
    analytics_date = Column(DateTime(timezone=True), nullable=False)
    
    # Call Volume Metrics
    calls_attempted = Column(Integer, default=0)
    calls_connected = Column(Integer, default=0)
    calls_answered = Column(Integer, default=0)
    calls_abandoned = Column(Integer, default=0)
    calls_failed = Column(Integer, default=0)
    
    # Timing Metrics
    average_call_duration = Column(Float, default=0.0)  # seconds
    average_wait_time = Column(Float, default=0.0)      # seconds
    average_talk_time = Column(Float, default=0.0)      # seconds
    
    # Conversion Metrics
    conversations_completed = Column(Integer, default=0)
    appointments_scheduled = Column(Integer, default=0)
    leads_generated = Column(Integer, default=0)
    conversion_rate = Column(Float, default=0.0)
    
    # Quality Metrics
    abandon_rate = Column(Float, default=0.0)
    contact_rate = Column(Float, default=0.0)  # answered / attempted
    completion_rate = Column(Float, default=0.0)  # completed / answered
    
    # Cost Metrics
    daily_cost = Column(Float, default=0.0)
    cost_per_call = Column(Float, default=0.0)
    cost_per_lead = Column(Float, default=0.0)
    cost_per_appointment = Column(Float, default=0.0)
    
    # Agent/System Performance
    agent_utilization = Column(Float, default=0.0)     # 0.0 - 1.0
    system_utilization = Column(Float, default=0.0)    # 0.0 - 1.0
    
    # Additional Metrics
    voicemail_left = Column(Integer, default=0)
    callbacks_requested = Column(Integer, default=0)
    dnc_hits = Column(Integer, default=0)
    complaints_received = Column(Integer, default=0)
    
    # Relationships
    campaign = relationship("Campaign")


# Import dependencies
from .crm import Lead
from .telephony import CallRecord