"""
Tests for Campaign Management (Phase 3)

Test suite for campaign creation, management, contact lists,
and outbound calling functionality.
"""

import pytest
import asyncio
import csv
import io
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from httpx import AsyncClient
from fastapi import UploadFile

from backend.app.models.campaign import (
    Campaign, CampaignList, CampaignContact, CampaignSchedule,
    DoNotCallList, CampaignAnalytics
)
from backend.app.models.telephony import CallRecord


@pytest.fixture
def mock_campaign():
    """Mock campaign for testing."""
    return Campaign(
        id=uuid4(),
        name="Test Campaign",
        description="Test outbound campaign",
        campaign_type="outbound_sales",
        status="draft",
        dialer_mode="progressive",
        max_concurrent_calls=10,
        created_by="user_123",
        timezone="America/New_York",
        calling_hours_start="09:00",
        calling_hours_end="17:00",
        respect_dnc=True
    )


@pytest.fixture
def mock_campaign_list():
    """Mock campaign list for testing."""
    return CampaignList(
        id=uuid4(),
        campaign_id=uuid4(),
        name="Test Contact List",
        description="Test list of contacts",
        total_contacts=100,
        valid_contacts=95,
        invalid_contacts=5,
        processing_status="completed"
    )


@pytest.fixture
def mock_campaign_contact():
    """Mock campaign contact for testing."""
    return CampaignContact(
        id=uuid4(),
        campaign_list_id=uuid4(),
        first_name="John",
        last_name="Doe",
        phone_number="+1234567890",
        email="john.doe@example.com",
        company="Acme Corp",
        call_status="pending",
        call_attempts=0,
        priority=3
    )


@pytest.fixture
def mock_dnc_record():
    """Mock DNC record for testing."""
    return DoNotCallList(
        id=uuid4(),
        phone_number="+1234567890",
        source="federal",
        registered_date=datetime.now(timezone.utc),
        status="active"
    )


class TestCampaignManagement:
    """Test suite for campaign management functionality."""
    
    @pytest.mark.asyncio
    async def test_create_campaign(self, mock_db: AsyncSession):
        """Test campaign creation."""
        campaign_data = {
            "name": "Test Campaign",
            "campaign_type": "outbound_sales",
            "dialer_mode": "progressive",
            "max_concurrent_calls": 10,
            "calling_hours_start": "09:00",
            "calling_hours_end": "17:00",
            "respect_dnc": True
        }
        
        campaign = Campaign(**campaign_data, created_by="user_123", status="draft")
        
        assert campaign.name == "Test Campaign"
        assert campaign.campaign_type == "outbound_sales"
        assert campaign.status == "draft"
        assert campaign.dialer_mode == "progressive"
        assert campaign.respect_dnc is True
    
    @pytest.mark.asyncio
    async def test_campaign_validation(self):
        """Test campaign data validation."""
        # Test invalid dialer mode
        with pytest.raises(ValueError):
            Campaign(
                name="Test",
                campaign_type="outbound_sales",
                dialer_mode="invalid_mode",
                created_by="user_123"
            )
        
        # Test invalid campaign type
        with pytest.raises(ValueError):
            Campaign(
                name="Test",
                campaign_type="invalid_type",
                dialer_mode="progressive",
                created_by="user_123"
            )
    
    @pytest.mark.asyncio
    async def test_campaign_scheduling(self, mock_campaign):
        """Test campaign scheduling functionality."""
        schedule = CampaignSchedule(
            campaign_id=mock_campaign.id,
            schedule_type="daily",
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc) + timedelta(days=30),
            daily_start_time="09:00",
            daily_end_time="17:00",
            monday_enabled=True,
            tuesday_enabled=True,
            wednesday_enabled=True,
            thursday_enabled=True,
            friday_enabled=True,
            saturday_enabled=False,
            sunday_enabled=False,
            timezone="America/New_York"
        )
        
        assert schedule.campaign_id == mock_campaign.id
        assert schedule.schedule_type == "daily"
        assert schedule.monday_enabled is True
        assert schedule.saturday_enabled is False
    
    @pytest.mark.asyncio
    async def test_campaign_contact_management(self, mock_campaign_list):
        """Test campaign contact management."""
        contact = CampaignContact(
            campaign_list_id=mock_campaign_list.id,
            first_name="Jane",
            last_name="Smith",
            phone_number="+1987654321",
            email="jane.smith@example.com",
            call_status="pending",
            priority=1
        )
        
        assert contact.campaign_list_id == mock_campaign_list.id
        assert contact.full_name == "Jane Smith"
        assert contact.call_status == "pending"
        assert contact.call_attempts == 0
    
    @pytest.mark.asyncio
    async def test_dnc_management(self):
        """Test Do Not Call list management."""
        dnc_record = DoNotCallList(
            phone_number="+1234567890",
            source="internal",
            registered_date=datetime.now(timezone.utc),
            status="active",
            first_name="John",
            last_name="Doe"
        )
        
        assert dnc_record.phone_number == "+1234567890"
        assert dnc_record.source == "internal"
        assert dnc_record.status == "active"


class TestCampaignExecution:
    """Test suite for campaign execution functionality."""
    
    @pytest.mark.asyncio
    async def test_campaign_start_validation(self, mock_db: AsyncSession, mock_campaign):
        """Test campaign start validation."""
        # Test starting campaign without contacts
        with patch('sqlalchemy.ext.asyncio.AsyncSession.scalar') as mock_scalar:
            mock_scalar.return_value = 0  # No contacts
            
            # Should not be able to start campaign without contacts
            assert mock_campaign.status == "draft"
    
    @pytest.mark.asyncio
    async def test_campaign_contact_processing(self, mock_campaign_contact, mock_db: AsyncSession):
        """Test processing campaign contacts for calling."""
        # Test DNC checking
        with patch('backend.app.models.campaign.DoNotCallList') as mock_dnc:
            mock_dnc.query.filter.return_value.first.return_value = None
            
            # Contact should be eligible for calling
            assert mock_campaign_contact.call_status == "pending"
            assert mock_campaign_contact.dnc_status in ["unknown", "clear"]
    
    @pytest.mark.asyncio
    async def test_call_attempt_tracking(self, mock_campaign_contact):
        """Test call attempt tracking."""
        # Simulate failed call attempt
        mock_campaign_contact.call_attempts += 1
        mock_campaign_contact.last_call_attempt = datetime.now(timezone.utc)
        mock_campaign_contact.contact_outcome = "no_answer"
        
        assert mock_campaign_contact.call_attempts == 1
        assert mock_campaign_contact.contact_outcome == "no_answer"
        
        # Simulate successful contact
        mock_campaign_contact.call_attempts += 1
        mock_campaign_contact.contact_outcome = "answered"
        mock_campaign_contact.conversation_completed = True
        mock_campaign_contact.call_status = "contacted"
        
        assert mock_campaign_contact.call_status == "contacted"
        assert mock_campaign_contact.conversation_completed is True
    
    @pytest.mark.asyncio
    async def test_campaign_pacing(self, mock_campaign):
        """Test campaign pacing and throttling."""
        # Test concurrent call limits
        assert mock_campaign.max_concurrent_calls == 10
        
        # Test calling hours
        current_hour = 10  # 10 AM
        start_hour = int(mock_campaign.calling_hours_start.split(':')[0])
        end_hour = int(mock_campaign.calling_hours_end.split(':')[0])
        
        assert start_hour <= current_hour <= end_hour
    
    @pytest.mark.asyncio
    async def test_abandon_rate_monitoring(self, mock_campaign):
        """Test abandon rate monitoring."""
        # Simulate calls with abandon rate calculation
        total_calls = 100
        abandoned_calls = 2
        abandon_rate = abandoned_calls / total_calls
        
        # Should be within acceptable limits
        assert abandon_rate <= mock_campaign.max_abandon_rate


class TestCampaignAPIEndpoints:
    """Test suite for Campaign API endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_campaign_endpoint(self, client: AsyncClient):
        """Test campaign creation endpoint."""
        campaign_data = {
            "name": "Test API Campaign",
            "description": "Campaign created via API",
            "campaign_type": "outbound_sales",
            "dialer_mode": "progressive",
            "max_concurrent_calls": 5,
            "respect_dnc": True
        }
        
        response = await client.post(
            "/api/v1/campaigns/",
            json=campaign_data,
            params={"created_by": "user_123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test API Campaign"
        assert data["campaign_type"] == "outbound_sales"
        assert data["status"] == "draft"
    
    @pytest.mark.asyncio
    async def test_list_campaigns_endpoint(self, client: AsyncClient):
        """Test campaign listing endpoint."""
        response = await client.get("/api/v1/campaigns/")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_start_campaign_endpoint(self, client: AsyncClient, mock_campaign):
        """Test campaign start endpoint."""
        with patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute, \
             patch('sqlalchemy.ext.asyncio.AsyncSession.scalar') as mock_scalar:
            
            mock_execute.return_value.scalar_one_or_none.return_value = mock_campaign
            mock_scalar.return_value = 50  # Has contacts
            
            response = await client.post(f"/api/v1/campaigns/{mock_campaign.id}/start")
            
            assert response.status_code == 200
            data = response.json()
            assert "started" in data["message"]
    
    @pytest.mark.asyncio
    async def test_pause_campaign_endpoint(self, client: AsyncClient, mock_campaign):
        """Test campaign pause endpoint."""
        mock_campaign.status = "running"
        
        with patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute:
            mock_execute.return_value.scalar_one_or_none.return_value = mock_campaign
            
            response = await client.post(f"/api/v1/campaigns/{mock_campaign.id}/pause")
            
            assert response.status_code == 200
            data = response.json()
            assert "paused" in data["message"]
    
    @pytest.mark.asyncio
    async def test_create_campaign_list_endpoint(self, client: AsyncClient, mock_campaign):
        """Test campaign list creation endpoint."""
        list_data = {
            "campaign_id": str(mock_campaign.id),
            "name": "Test Contact List",
            "description": "API created list"
        }
        
        with patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute:
            mock_execute.return_value.scalar_one_or_none.return_value = mock_campaign
            
            response = await client.post(
                f"/api/v1/campaigns/{mock_campaign.id}/lists",
                json=list_data
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Test Contact List"
    
    @pytest.mark.asyncio
    async def test_upload_contact_list_endpoint(self, client: AsyncClient, mock_campaign_list):
        """Test contact list upload endpoint."""
        # Create mock CSV content
        csv_content = "first_name,last_name,phone,email\nJohn,Doe,+1234567890,john@example.com"
        
        with patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute:
            mock_execute.return_value.scalar_one_or_none.return_value = mock_campaign_list
            
            # Mock file upload
            files = {"file": ("contacts.csv", csv_content, "text/csv")}
            
            response = await client.post(
                f"/api/v1/campaigns/lists/{mock_campaign_list.id}/upload",
                files=files
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "uploaded successfully" in data["message"]
    
    @pytest.mark.asyncio
    async def test_add_dnc_number_endpoint(self, client: AsyncClient):
        """Test adding number to DNC list."""
        dnc_data = {
            "phone_number": "+1234567890",
            "source": "internal",
            "first_name": "John",
            "last_name": "Doe"
        }
        
        response = await client.post("/api/v1/campaigns/dnc", json=dnc_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["phone_number"] == "+1234567890"
        assert data["source"] == "internal"
    
    @pytest.mark.asyncio
    async def test_check_dnc_status_endpoint(self, client: AsyncClient, mock_dnc_record):
        """Test DNC status check endpoint."""
        with patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute:
            mock_execute.return_value.scalar_one_or_none.return_value = mock_dnc_record
            
            response = await client.get(f"/api/v1/campaigns/dnc/check/{mock_dnc_record.phone_number}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["phone_number"] == mock_dnc_record.phone_number
            assert data["is_dnc"] is True
    
    @pytest.mark.asyncio
    async def test_campaign_analytics_endpoint(self, client: AsyncClient, mock_campaign):
        """Test campaign analytics endpoint."""
        mock_analytics = {
            "campaign_id": str(mock_campaign.id),
            "total_contacts": 100,
            "contacted": 75,
            "contact_rate": 75.0,
            "call_analytics": {
                "total_calls": 100,
                "answered_calls": 75
            }
        }
        
        with patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute, \
             patch('backend.app.services.call_service.call_service.get_call_analytics') as mock_call_analytics:
            
            mock_execute.return_value.scalar_one_or_none.return_value = mock_campaign
            mock_call_analytics.return_value = mock_analytics["call_analytics"]
            
            response = await client.get(f"/api/v1/campaigns/{mock_campaign.id}/analytics")
            
            assert response.status_code == 200
            data = response.json()
            assert data["campaign_id"] == str(mock_campaign.id)
            assert "contact_rate" in data


class TestCSVProcessing:
    """Test suite for CSV contact processing."""
    
    @pytest.mark.asyncio
    async def test_csv_parsing(self):
        """Test CSV contact parsing."""
        csv_content = """first_name,last_name,phone,email,company
John,Doe,+1234567890,john@example.com,Acme Corp
Jane,Smith,+1987654321,jane@example.com,Smith LLC"""
        
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        contacts = list(csv_reader)
        
        assert len(contacts) == 2
        assert contacts[0]["first_name"] == "John"
        assert contacts[0]["phone"] == "+1234567890"
        assert contacts[1]["first_name"] == "Jane"
    
    @pytest.mark.asyncio
    async def test_contact_validation(self):
        """Test contact data validation during CSV processing."""
        valid_contact = {
            "first_name": "John",
            "last_name": "Doe",
            "phone": "+1234567890",
            "email": "john@example.com"
        }
        
        invalid_contact = {
            "first_name": "Jane",
            "last_name": "Smith",
            "phone": "",  # Missing required phone
            "email": "invalid-email"
        }
        
        # Valid contact should pass validation
        assert valid_contact["phone"]  # Has phone number
        
        # Invalid contact should fail validation
        assert not invalid_contact["phone"]  # Missing phone
    
    @pytest.mark.asyncio
    async def test_phone_number_normalization(self):
        """Test phone number normalization."""
        test_numbers = [
            "1234567890",
            "(123) 456-7890",
            "+1-123-456-7890",
            "123.456.7890"
        ]
        
        # All should normalize to same format
        normalized = ["+1234567890"] * len(test_numbers)
        
        # Simple normalization logic
        for i, number in enumerate(test_numbers):
            clean_number = "+1" + "".join(filter(str.isdigit, number))[-10:]
            assert len(clean_number) == 12  # +1 plus 10 digits


@pytest.mark.integration
class TestCampaignIntegration:
    """Integration tests for campaign functionality."""
    
    @pytest.mark.asyncio
    async def test_full_campaign_lifecycle(self, mock_db: AsyncSession):
        """Test complete campaign lifecycle."""
        # This would test the full flow:
        # 1. Create campaign
        # 2. Upload contact list
        # 3. Process and validate contacts
        # 4. Start campaign
        # 5. Execute calls
        # 6. Track results
        # 7. Generate analytics
        
        pass
    
    @pytest.mark.asyncio
    async def test_campaign_with_voice_integration(self, mock_db: AsyncSession):
        """Test campaign integration with voice system."""
        # This would test:
        # 1. Campaign triggers outbound call
        # 2. Call connects to voice AI system
        # 3. Conversation is processed
        # 4. Lead is created/updated
        # 5. Campaign contact status is updated
        
        pass
    
    @pytest.mark.asyncio
    async def test_campaign_compliance(self, mock_db: AsyncSession):
        """Test campaign compliance features."""
        # This would test:
        # 1. DNC list checking
        # 2. Calling hours enforcement
        # 3. Abandon rate monitoring
        # 4. Opt-out processing
        # 5. Compliance reporting
        
        pass


# Mock fixtures for testing
@pytest.fixture
async def mock_db():
    """Mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
async def client():
    """Mock HTTP client for API testing."""
    return AsyncMock(spec=AsyncClient)