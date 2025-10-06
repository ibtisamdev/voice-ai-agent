from typing import Optional
from pydantic import validator
from pydantic_settings import BaseSettings
import secrets


class Settings(BaseSettings):
    # Application
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Voice AI Agent"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Database
    DATABASE_URL: str
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 40
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_MAX_CONNECTIONS: int = 50
    
    # Celery Configuration  
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # LLM Configuration
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2:7b-chat"
    OLLAMA_TEMPERATURE: float = 0.7
    OLLAMA_MAX_TOKENS: int = 2048
    OLLAMA_TIMEOUT: int = 30
    OLLAMA_KEEP_ALIVE: str = "24h"
    
    # Vector Database Configuration
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma"
    CHROMA_COLLECTION_NAME: str = "legal_documents"
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_CACHE_SIZE: int = 1000
    
    # RAG Configuration
    MAX_CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_RETRIEVAL_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    RERANK_TOP_K: int = 3
    
    # Document Processing
    UPLOAD_MAX_SIZE: str = "50MB"
    SUPPORTED_FORMATS: str = "pdf,docx,txt"
    PROCESSING_BATCH_SIZE: int = 10
    
    # Voice Processing Configuration
    WHISPER_MODEL_SIZE: str = "base"  # tiny, base, small, medium, large
    WHISPER_LANGUAGE: Optional[str] = None  # Auto-detect if None
    WHISPER_DEVICE: str = "auto"  # auto, cpu, cuda
    
    # Audio Processing
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHUNK_DURATION_MS: int = 1000
    AUDIO_BUFFER_SIZE: int = 4096
    VAD_AGGRESSIVENESS: int = 2  # 0-3, higher = more aggressive
    NOISE_REDUCTION_ENABLED: bool = True
    
    # TTS Configuration
    TTS_DEFAULT_ENGINE: str = "coqui"  # coqui, elevenlabs, azure, system
    TTS_DEFAULT_VOICE: Optional[str] = None
    TTS_CACHE_ENABLED: bool = True
    TTS_CACHE_SIZE: int = 100
    TTS_RATE: float = 1.0
    TTS_VOLUME: float = 1.0
    
    # External API Keys (optional)
    ELEVENLABS_API_KEY: Optional[str] = None
    AZURE_SPEECH_KEY: Optional[str] = None
    AZURE_SPEECH_REGION: str = "eastus"
    
    # Conversation Management
    CONVERSATION_SESSION_TTL: int = 86400  # 24 hours
    MAX_CONVERSATION_TURNS: int = 1000
    MAX_SESSION_DURATION: int = 3600  # 1 hour
    
    # Intent Classification
    INTENT_CONFIDENCE_THRESHOLD: float = 0.5
    INTENT_USE_BERT: bool = True
    INTENT_MODEL_NAME: str = "microsoft/DialoGPT-medium"
    
    # WebSocket Configuration
    WS_MAX_CONNECTIONS: int = 100
    WS_PING_INTERVAL: int = 30
    WS_PING_TIMEOUT: int = 10
    
    # Voice Service Paths
    VOICE_MODEL_PATH: str = "/app/models"
    VOICE_CACHE_PATH: str = "/app/cache"
    CONVERSATION_FLOWS_PATH: str = "/app/ai/conversation/flows"
    
    # =============================================================================
    # Phase 3: Integration Layer Configuration
    # =============================================================================
    
    # Zoho CRM Configuration
    ZOHO_CLIENT_ID: Optional[str] = None
    ZOHO_CLIENT_SECRET: Optional[str] = None
    ZOHO_REDIRECT_URI: str = "http://localhost:8000/api/v1/crm/auth/callback"
    ZOHO_REFRESH_TOKEN: Optional[str] = None
    ZOHO_API_DOMAIN: str = "https://www.zohoapis.com"
    ZOHO_ACCOUNTS_URL: str = "https://accounts.zoho.com"
    ZOHO_SANDBOX_MODE: bool = True
    
    # CRM Sync Configuration
    CRM_SYNC_INTERVAL: int = 300  # 5 minutes
    CRM_BATCH_SIZE: int = 100
    CRM_RETRY_ATTEMPTS: int = 3
    CRM_TIMEOUT: int = 30
    
    # Telephony Configuration
    SIP_SERVER: Optional[str] = None
    SIP_USERNAME: Optional[str] = None
    SIP_PASSWORD: Optional[str] = None
    SIP_DOMAIN: Optional[str] = None
    SIP_PORT: int = 5060
    SIP_TRANSPORT: str = "UDP"  # UDP, TCP, TLS
    
    # Call Management
    MAX_CONCURRENT_CALLS: int = 50
    CALL_TIMEOUT: int = 30  # seconds
    CALL_RETRY_ATTEMPTS: int = 3
    CALL_RETRY_DELAY: int = 5  # seconds
    ENABLE_CALL_RECORDING: bool = False
    CALL_RECORDING_PATH: str = "/app/recordings"
    
    # Campaign Configuration
    DIALER_MODE: str = "progressive"  # progressive, predictive, preview
    CAMPAIGN_MAX_ATTEMPTS: int = 3
    CAMPAIGN_RETRY_DELAY: int = 3600  # 1 hour
    CALL_ABANDON_RATE_LIMIT: float = 0.03  # 3% max abandon rate
    
    # DNC (Do Not Call) Configuration
    DNC_ENABLED: bool = True
    DNC_CHECK_TIMEOUT: int = 5
    
    # Appointment Scheduling
    SCHEDULING_TIMEZONE: str = "America/New_York"
    BUSINESS_HOURS_START: str = "09:00"
    BUSINESS_HOURS_END: str = "17:00"
    BUSINESS_DAYS: str = "monday,tuesday,wednesday,thursday,friday"
    APPOINTMENT_BUFFER_MINUTES: int = 15
    
    # Lead Management
    LEAD_SCORING_ENABLED: bool = True
    HIGH_PRIORITY_SCORE_THRESHOLD: int = 80
    AUTO_ASSIGN_LEADS: bool = True
    LEAD_FOLLOW_UP_HOURS: int = 24
    
    # Twilio Fallback Configuration (optional)
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_PHONE_NUMBER: Optional[str] = None
    
    # Background Task Configuration
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: list = ["json"]
    CELERY_TIMEZONE: str = "UTC"
    
    # Webhook Configuration
    WEBHOOK_SECRET_KEY: str = secrets.token_urlsafe(32)
    WEBHOOK_TIMEOUT: int = 10
    WEBHOOK_RETRY_ATTEMPTS: int = 3
    
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str]) -> str:
        if not v:
            raise ValueError("DATABASE_URL is required")
        return v
    
    @validator("SECRET_KEY", pre=True)
    def validate_secret_key(cls, v: Optional[str]) -> str:
        if not v or len(v) < 32:
            return secrets.token_urlsafe(32)
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()