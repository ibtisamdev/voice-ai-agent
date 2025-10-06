"""
Phase 3 Test Configuration

Configuration and utilities for Phase 3 integration tests.
"""

import pytest
import asyncio
import os
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient
from fastapi.testclient import TestClient

# Import application components
from backend.app.main import app
from backend.app.core.database import get_db
from backend.app.core.config import Settings
from backend.app.models.base import Base


# Test settings
class TestSettings(Settings):
    """Test-specific settings."""
    
    # Use test database
    DATABASE_URL: str = "sqlite+aiosqlite:///./test.db"
    
    # Mock external services
    MOCK_CRM_RESPONSES: bool = True
    MOCK_SIP_GATEWAY: bool = True
    CAMPAIGN_TEST_MODE: bool = True
    
    # Disable external API calls
    ZOHO_CLIENT_ID: str = "test_client_id"
    ZOHO_CLIENT_SECRET: str = "test_client_secret"
    SIP_SERVER: str = "test.sip.server"
    SIP_USERNAME: str = "test_user"
    SIP_PASSWORD: str = "test_pass"
    
    # Use test Redis instances
    REDIS_URL: str = "redis://localhost:6379/15"  # Test database
    CELERY_BROKER_URL: str = "redis://localhost:6379/14"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/13"


# Override settings for testing
test_settings = TestSettings()


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        test_settings.DATABASE_URL,
        echo=False,
        future=True
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    await engine.dispose()


@pytest.fixture
async def test_db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def client(test_db_session) -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client."""
    
    # Override database dependency
    async def override_get_db():
        yield test_db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    # Clean up
    app.dependency_overrides.clear()


@pytest.fixture
def sync_client() -> Generator[TestClient, None, None]:
    """Create synchronous test client."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_zoho_api():
    """Mock Zoho CRM API responses."""
    
    class MockZohoAPI:
        def __init__(self):
            self.responses = {}
        
        def mock_token_response(self):
            return {
                "access_token": "mock_access_token",
                "refresh_token": "mock_refresh_token",
                "expires_in": 3600
            }
        
        def mock_org_response(self):
            return {
                "org": [{
                    "id": "mock_org_123",
                    "company_name": "Mock Law Firm"
                }]
            }
        
        def mock_lead_create_response(self):
            return {
                "data": [{
                    "details": {
                        "id": "mock_lead_123"
                    }
                }]
            }
        
        def mock_leads_list_response(self):
            return {
                "data": [
                    {
                        "id": "lead_1",
                        "First_Name": "John",
                        "Last_Name": "Doe",
                        "Email": "john@example.com",
                        "Phone": "+1234567890"
                    },
                    {
                        "id": "lead_2", 
                        "First_Name": "Jane",
                        "Last_Name": "Smith",
                        "Email": "jane@example.com",
                        "Phone": "+1987654321"
                    }
                ]
            }
    
    return MockZohoAPI()


@pytest.fixture
def mock_sip_gateway():
    """Mock SIP gateway for testing."""
    
    class MockSIPGateway:
        def __init__(self):
            self.is_running = False
            self.accounts = {}
            self.active_calls = {}
            self.call_callbacks = []
        
        async def initialize(self):
            self.is_running = True
        
        async def shutdown(self):
            self.is_running = False
        
        async def register_account(self, sip_config):
            account_id = sip_config["username"]
            self.accounts[account_id] = sip_config
            return account_id
        
        async def make_call(self, account_id, destination, call_data=None):
            call_id = f"mock_call_{len(self.active_calls) + 1}"
            self.active_calls[call_id] = {
                "account_id": account_id,
                "destination": destination,
                "call_data": call_data
            }
            return call_id
        
        async def hangup_call(self, call_id):
            if call_id in self.active_calls:
                del self.active_calls[call_id]
        
        async def transfer_call(self, call_id, destination):
            if call_id in self.active_calls:
                self.active_calls[call_id]["transferred_to"] = destination
        
        def get_active_calls(self):
            return list(self.active_calls.keys())
        
        def add_call_callback(self, callback):
            self.call_callbacks.append(callback)
    
    return MockSIPGateway()


@pytest.fixture
def mock_celery():
    """Mock Celery for background task testing."""
    
    class MockCelery:
        def __init__(self):
            self.tasks = []
        
        def delay(self, *args, **kwargs):
            task_id = f"task_{len(self.tasks) + 1}"
            self.tasks.append({
                "id": task_id,
                "args": args,
                "kwargs": kwargs
            })
            return MockAsyncResult(task_id)
        
        def apply_async(self, *args, **kwargs):
            return self.delay(*args, **kwargs)
    
    class MockAsyncResult:
        def __init__(self, task_id):
            self.id = task_id
            self.state = "PENDING"
            self.result = None
        
        def get(self, timeout=None):
            return self.result
        
        def ready(self):
            return self.state in ["SUCCESS", "FAILURE"]
    
    return MockCelery()


@pytest.fixture
def sample_conversation_data():
    """Sample conversation data for testing."""
    return {
        "first_name": "John",
        "last_name": "Doe", 
        "email": "john.doe@example.com",
        "phone": "+1234567890",
        "company": "Acme Corporation",
        "legal_issue": "Personal injury claim from car accident",
        "practice_areas": ["personal_injury"],
        "urgency_level": "high",
        "call_duration": 180,
        "call_quality_score": 0.9,
        "sentiment": {
            "compound": 0.2,
            "positive": 0.6,
            "neutral": 0.3,
            "negative": 0.1
        },
        "intent_confidence": 0.85,
        "summary": "Client was in a car accident and needs legal representation for personal injury claim.",
        "custom_fields": {
            "accident_date": "2024-11-15",
            "insurance_company": "State Farm",
            "injuries": ["back pain", "whiplash"]
        }
    }


@pytest.fixture
def sample_campaign_data():
    """Sample campaign data for testing."""
    return {
        "name": "Personal Injury Outreach Q4",
        "description": "Outbound campaign for personal injury leads",
        "campaign_type": "outbound_sales",
        "dialer_mode": "progressive",
        "max_concurrent_calls": 5,
        "call_timeout": 30,
        "retry_attempts": 2,
        "retry_delay": 3600,
        "timezone": "America/New_York",
        "calling_hours_start": "09:00",
        "calling_hours_end": "17:00",
        "calling_days": "1,2,3,4,5",
        "respect_dnc": True,
        "practice_area": "personal_injury",
        "script_template": "Hello, this is calling from [FIRM_NAME] regarding your recent inquiry about legal services..."
    }


@pytest.fixture
def sample_contact_list():
    """Sample contact list for campaign testing."""
    return [
        {
            "first_name": "Alice",
            "last_name": "Johnson",
            "phone": "+1234567891",
            "email": "alice@example.com",
            "company": "Tech Corp",
            "priority": 1
        },
        {
            "first_name": "Bob", 
            "last_name": "Williams",
            "phone": "+1234567892",
            "email": "bob@example.com",
            "company": "Consulting LLC",
            "priority": 2
        },
        {
            "first_name": "Carol",
            "last_name": "Brown",
            "phone": "+1234567893", 
            "email": "carol@example.com",
            "company": "Marketing Inc",
            "priority": 3
        }
    ]


# Test utilities
class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def create_csv_content(contacts):
        """Create CSV content from contact list."""
        import csv
        import io
        
        output = io.StringIO()
        fieldnames = ["first_name", "last_name", "phone", "email", "company", "priority"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        writer.writeheader()
        for contact in contacts:
            writer.writerow(contact)
        
        return output.getvalue()
    
    @staticmethod
    def assert_lead_data_matches(lead, conversation_data):
        """Assert that lead data matches conversation data."""
        assert lead.first_name == conversation_data["first_name"]
        assert lead.last_name == conversation_data["last_name"]
        assert lead.email == conversation_data["email"]
        assert lead.phone == conversation_data["phone"]
        assert lead.company == conversation_data["company"]
        assert lead.legal_issue == conversation_data["legal_issue"]
        assert lead.practice_areas == conversation_data["practice_areas"]
        assert lead.urgency_level == conversation_data["urgency_level"]
    
    @staticmethod
    def assert_call_record_valid(call_record):
        """Assert that call record has valid data."""
        assert call_record.call_id is not None
        assert call_record.direction in ["inbound", "outbound"]
        assert call_record.status in ["ringing", "answered", "busy", "failed", "ended"]
        assert call_record.initiated_at is not None
    
    @staticmethod
    def assert_campaign_valid(campaign):
        """Assert that campaign has valid configuration."""
        assert campaign.name is not None
        assert campaign.campaign_type in ["outbound_sales", "follow_up", "survey", "appointment_reminder"]
        assert campaign.dialer_mode in ["progressive", "predictive", "preview", "manual"]
        assert campaign.status in ["draft", "scheduled", "running", "paused", "completed", "cancelled"]
        assert campaign.max_concurrent_calls > 0
        assert campaign.respect_dnc is not None


# Pytest configuration
def pytest_configure(config):
    """Configure pytest for Phase 3 tests."""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["MOCK_CRM_RESPONSES"] = "true"
    os.environ["MOCK_SIP_GATEWAY"] = "true"
    os.environ["CAMPAIGN_TEST_MODE"] = "true"


def pytest_collection_modifyitems(config, items):
    """Modify test collection for Phase 3."""
    # Add markers for integration tests
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "crm" in item.nodeid:
            item.add_marker(pytest.mark.crm)
        if "telephony" in item.nodeid:
            item.add_marker(pytest.mark.telephony)
        if "campaign" in item.nodeid:
            item.add_marker(pytest.mark.campaign)


# Custom pytest markers
pytest_plugins = []

# Register custom markers
def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "crm: mark test as CRM-related")
    config.addinivalue_line("markers", "telephony: mark test as telephony-related")
    config.addinivalue_line("markers", "campaign: mark test as campaign-related")
    config.addinivalue_line("markers", "slow: mark test as slow running")


# Export test utilities
__all__ = [
    "TestSettings",
    "TestUtils",
    "test_settings"
]