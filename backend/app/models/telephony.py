from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timezone
import uuid

from .base import Base, TimestampMixin, SoftDeleteMixin


class SIPAccount(Base, TimestampMixin):
    """SIP account configuration for telephony integration."""
    __tablename__ = "sip_accounts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    username = Column(String(100), nullable=False, unique=True)
    password = Column(String(255), nullable=False)
    domain = Column(String(255), nullable=False)
    server = Column(String(255), nullable=False)
    port = Column(Integer, default=5060)
    transport = Column(String(10), default="UDP")  # UDP, TCP, TLS
    
    # Registration Status
    registration_status = Column(String(20), default="unregistered")  # registered, unregistered, failed
    last_registration = Column(DateTime(timezone=True), nullable=True)
    registration_expires = Column(DateTime(timezone=True), nullable=True)
    registration_error = Column(Text, nullable=True)
    
    # Account Settings
    is_active = Column(Boolean, default=True)
    max_concurrent_calls = Column(Integer, default=10)
    codec_preferences = Column(JSON, nullable=True)  # ["PCMU", "PCMA", "G722"]
    
    # Quality Settings
    dtmf_mode = Column(String(20), default="RFC2833")  # RFC2833, SIP_INFO, INBAND
    rtp_timeout = Column(Integer, default=60)
    
    # Relationships
    call_records = relationship("CallRecord", back_populates="sip_account")


class CallRecord(Base, TimestampMixin):
    """Comprehensive call logging and analytics."""
    __tablename__ = "call_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id = Column(String(255), unique=True, nullable=False)
    
    # SIP and Campaign Relations
    sip_account_id = Column(UUID(as_uuid=True), ForeignKey("sip_accounts.id"), nullable=True)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=True)
    lead_id = Column(UUID(as_uuid=True), ForeignKey("leads.id"), nullable=True)
    
    # Call Direction and Type
    direction = Column(String(20), nullable=False)  # inbound, outbound
    call_type = Column(String(50), default="voice")  # voice, conference, transfer
    
    # Phone Numbers
    caller_number = Column(String(50), nullable=True)
    called_number = Column(String(50), nullable=True)
    original_called_number = Column(String(50), nullable=True)  # before transfers
    
    # Call Timing
    initiated_at = Column(DateTime(timezone=True), nullable=False)
    answered_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    duration = Column(Integer, nullable=True)  # seconds
    billable_duration = Column(Integer, nullable=True)  # seconds
    ring_duration = Column(Integer, nullable=True)  # seconds
    
    # Call Status and Outcome
    status = Column(String(20), nullable=False)  # ringing, answered, busy, failed, cancelled
    hangup_cause = Column(String(50), nullable=True)  # NORMAL_CLEARING, BUSY, NO_ANSWER, etc.
    disposition = Column(String(50), nullable=True)  # answered, voicemail, busy, no_answer, failed
    
    # Quality Metrics
    audio_quality_score = Column(Float, nullable=True)  # 1.0 - 5.0 (MOS score)
    jitter_avg = Column(Float, nullable=True)  # ms
    packet_loss_percent = Column(Float, nullable=True)
    latency_avg = Column(Float, nullable=True)  # ms
    
    # Recording and Transcription
    recording_enabled = Column(Boolean, default=False)
    recording_path = Column(String(500), nullable=True)
    recording_duration = Column(Integer, nullable=True)  # seconds
    recording_size = Column(Integer, nullable=True)  # bytes
    
    # AI Processing Results
    transcription = Column(Text, nullable=True)
    transcription_confidence = Column(Float, nullable=True)
    sentiment_analysis = Column(JSON, nullable=True)
    conversation_summary = Column(Text, nullable=True)
    intent_classification = Column(JSON, nullable=True)
    
    # Transfer Information
    transfer_count = Column(Integer, default=0)
    transferred_to = Column(String(50), nullable=True)
    transfer_reason = Column(String(100), nullable=True)
    
    # Cost and Billing
    cost_per_minute = Column(Float, nullable=True)
    total_cost = Column(Float, nullable=True)
    currency = Column(String(3), default="USD")
    
    # Technical Details
    user_agent = Column(String(255), nullable=True)
    codec_used = Column(String(20), nullable=True)
    sip_call_id = Column(String(255), nullable=True)
    remote_ip = Column(String(45), nullable=True)
    
    # Custom Fields and Metadata
    tags = Column(JSON, nullable=True)
    custom_fields = Column(JSON, nullable=True)
    call_metadata = Column(JSON, nullable=True)
    
    # Relationships
    sip_account = relationship("SIPAccount", back_populates="call_records")
    campaign = relationship("Campaign", back_populates="call_records")
    lead = relationship("Lead")  # Back reference defined in crm.py
    call_events = relationship("CallEvent", back_populates="call_record")


class CallEvent(Base, TimestampMixin):
    """Detailed call event logging for debugging and analytics."""
    __tablename__ = "call_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_record_id = Column(UUID(as_uuid=True), ForeignKey("call_records.id"), nullable=False)
    
    event_type = Column(String(50), nullable=False)  # INVITE, RINGING, ANSWER, HANGUP, etc.
    event_time = Column(DateTime(timezone=True), nullable=False)
    event_data = Column(JSON, nullable=True)
    
    # SIP Specific
    sip_method = Column(String(20), nullable=True)  # INVITE, BYE, CANCEL, etc.
    sip_response_code = Column(Integer, nullable=True)  # 200, 404, 486, etc.
    sip_reason_phrase = Column(String(100), nullable=True)
    
    # Audio/RTP Events
    rtp_event = Column(String(50), nullable=True)  # START, STOP, DTMF, etc.
    dtmf_digit = Column(String(1), nullable=True)
    
    # Call Flow Events
    flow_event = Column(String(50), nullable=True)  # MENU_SELECTION, TRANSFER_START, etc.
    flow_data = Column(JSON, nullable=True)
    
    # Error and Debug Info
    error_code = Column(String(20), nullable=True)
    error_message = Column(Text, nullable=True)
    debug_info = Column(JSON, nullable=True)
    
    # Relationships
    call_record = relationship("CallRecord", back_populates="call_events")


class CallQueue(Base, TimestampMixin):
    """Call queue management for inbound calls."""
    __tablename__ = "call_queues"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    
    # Queue Configuration
    max_wait_time = Column(Integer, default=300)  # seconds
    max_queue_size = Column(Integer, default=50)
    
    # Routing Strategy
    routing_strategy = Column(String(50), default="round_robin")  # round_robin, longest_idle, random
    overflow_destination = Column(String(255), nullable=True)  # phone number or voicemail
    
    # Queue Music and Messages
    music_on_hold = Column(String(500), nullable=True)  # file path
    welcome_message = Column(String(500), nullable=True)  # file path
    position_announcements = Column(Boolean, default=True)
    estimated_wait_announcements = Column(Boolean, default=True)
    
    # Operating Hours
    business_hours_enabled = Column(Boolean, default=True)
    business_hours_start = Column(String(5), default="09:00")  # HH:MM
    business_hours_end = Column(String(5), default="17:00")  # HH:MM
    business_days = Column(String(50), default="1,2,3,4,5")  # Monday=1, Sunday=7
    timezone = Column(String(50), default="America/New_York")
    
    # After Hours Handling
    after_hours_message = Column(String(500), nullable=True)
    after_hours_destination = Column(String(255), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    current_queue_size = Column(Integer, default=0)
    
    # Analytics
    total_calls_received = Column(Integer, default=0)
    total_calls_answered = Column(Integer, default=0)
    total_calls_abandoned = Column(Integer, default=0)
    average_wait_time = Column(Float, default=0.0)  # seconds
    
    # Relationships
    queue_members = relationship("QueueMember", back_populates="queue")
    queue_calls = relationship("QueueCall", back_populates="queue")


class QueueMember(Base, TimestampMixin):
    """Queue member configuration (agents/attorneys)."""
    __tablename__ = "queue_members"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    queue_id = Column(UUID(as_uuid=True), ForeignKey("call_queues.id"), nullable=False)
    
    member_id = Column(String(100), nullable=False)  # attorney/agent ID
    member_name = Column(String(255), nullable=False)
    member_extension = Column(String(20), nullable=True)
    member_phone = Column(String(50), nullable=True)
    
    # Member Settings
    priority = Column(Integer, default=1)  # higher number = higher priority
    penalty = Column(Integer, default=0)  # call routing penalty
    max_concurrent_calls = Column(Integer, default=1)
    
    # Status and Availability
    status = Column(String(20), default="available")  # available, busy, unavailable, paused
    paused = Column(Boolean, default=False)
    pause_reason = Column(String(100), nullable=True)
    
    # Statistics
    calls_taken = Column(Integer, default=0)
    calls_missed = Column(Integer, default=0)
    total_talk_time = Column(Integer, default=0)  # seconds
    last_call_time = Column(DateTime(timezone=True), nullable=True)
    
    # Schedule
    schedule_enabled = Column(Boolean, default=False)
    schedule_data = Column(JSON, nullable=True)  # Weekly schedule
    
    # Relationships
    queue = relationship("CallQueue", back_populates="queue_members")


class QueueCall(Base, TimestampMixin):
    """Individual call tracking within queues."""
    __tablename__ = "queue_calls"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    queue_id = Column(UUID(as_uuid=True), ForeignKey("call_queues.id"), nullable=False)
    call_record_id = Column(UUID(as_uuid=True), ForeignKey("call_records.id"), nullable=False)
    
    # Queue Entry
    entered_at = Column(DateTime(timezone=True), nullable=False)
    position_in_queue = Column(Integer, nullable=False)
    
    # Queue Exit
    exited_at = Column(DateTime(timezone=True), nullable=True)
    exit_reason = Column(String(50), nullable=True)  # answered, abandoned, timeout, transferred
    
    # Assignment
    assigned_member_id = Column(String(100), nullable=True)
    assigned_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timing
    wait_time = Column(Integer, nullable=True)  # seconds
    queue_time = Column(Integer, nullable=True)  # total time in queue
    
    # Call Back
    callback_requested = Column(Boolean, default=False)
    callback_number = Column(String(50), nullable=True)
    callback_scheduled = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    queue = relationship("CallQueue", back_populates="queue_calls")
    call_record = relationship("CallRecord")


# Import models that this depends on
from .crm import Lead
from .campaign import Campaign