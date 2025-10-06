"""
Tests for Telephony Integration (Phase 3)

Test suite for SIP gateway, call management, and telephony operations.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from httpx import AsyncClient

from backend.app.models.telephony import CallRecord, CallEvent, SIPAccount, CallQueue
from backend.app.services.call_service import CallService
from telephony.sip_gateway import SIPGateway, CallInfo, CallState


@pytest.fixture
def mock_sip_account():
    """Mock SIP account for testing."""
    return SIPAccount(
        id=uuid4(),
        name="Test SIP Account",
        username="test_user",
        password="test_pass",
        domain="sip.example.com",
        server="sip.example.com",
        port=5060,
        transport="UDP",
        registration_status="registered",
        is_active=True,
        max_concurrent_calls=10
    )


@pytest.fixture
def mock_call_record():
    """Mock call record for testing."""
    return CallRecord(
        id=uuid4(),
        call_id="call_123",
        direction="inbound",
        caller_number="+1234567890",
        called_number="+1987654321", 
        status="ringing",
        initiated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def mock_call_info():
    """Mock call info from SIP gateway."""
    return CallInfo(
        call_id="call_123",
        sip_call_id="sip_call_456",
        caller_number="+1234567890",
        called_number="+1987654321",
        direction="inbound",
        state=CallState.RINGING,
        start_time=datetime.now(timezone.utc)
    )


class TestSIPGateway:
    """Test suite for SIP Gateway."""
    
    @pytest.mark.asyncio
    async def test_initialize_gateway(self):
        """Test SIP gateway initialization."""
        gateway = SIPGateway()
        
        with patch('pjsua2.Endpoint') as mock_endpoint:
            mock_endpoint.return_value.libCreate = MagicMock()
            mock_endpoint.return_value.libInit = MagicMock()
            mock_endpoint.return_value.libStart = MagicMock()
            
            await gateway.initialize()
            
            assert gateway.is_running is True
            mock_endpoint.return_value.libCreate.assert_called_once()
            mock_endpoint.return_value.libInit.assert_called_once()
            mock_endpoint.return_value.libStart.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_register_sip_account(self):
        """Test SIP account registration."""
        gateway = SIPGateway()
        gateway.is_running = True
        
        sip_config = {
            "username": "test_user",
            "password": "test_pass",
            "domain": "sip.example.com",
            "server": "sip.example.com"
        }
        
        with patch('telephony.sip_gateway.VoiceAIAccount') as mock_account:
            mock_account.return_value.create = MagicMock()
            
            account_id = await gateway.register_account(sip_config)
            
            assert account_id == "test_user"
            assert account_id in gateway.accounts
            mock_account.return_value.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_outbound_call(self):
        """Test making outbound call."""
        gateway = SIPGateway()
        gateway.is_running = True
        
        mock_account = MagicMock()
        mock_call = MagicMock()
        mock_call.call_info.call_id = "call_123"
        mock_account.make_outbound_call.return_value = mock_call
        
        gateway.accounts["test_account"] = mock_account
        
        call_id = await gateway.make_call("test_account", "+1234567890")
        
        assert call_id == "call_123"
        assert call_id in gateway.active_calls
        mock_account.make_outbound_call.assert_called_once_with("+1234567890")
    
    @pytest.mark.asyncio
    async def test_hangup_call(self):
        """Test hanging up call."""
        gateway = SIPGateway()
        
        mock_call = MagicMock()
        gateway.active_calls["call_123"] = mock_call
        
        await gateway.hangup_call("call_123")
        
        mock_call.hangup.assert_called_once()
        assert "call_123" not in gateway.active_calls
    
    @pytest.mark.asyncio
    async def test_transfer_call(self):
        """Test call transfer."""
        gateway = SIPGateway()
        
        mock_call = MagicMock()
        gateway.active_calls["call_123"] = mock_call
        
        await gateway.transfer_call("call_123", "+1555123456")
        
        mock_call.xfer.assert_called_once_with("+1555123456", unittest.mock.ANY)
    
    @pytest.mark.asyncio
    async def test_send_dtmf(self):
        """Test sending DTMF tones."""
        gateway = SIPGateway()
        
        mock_call = MagicMock()
        gateway.active_calls["call_123"] = mock_call
        
        await gateway.send_dtmf("call_123", "123")
        
        mock_call.dialDtmf.assert_called_once_with("123")


class TestCallService:
    """Test suite for Call Service."""
    
    @pytest.mark.asyncio
    async def test_initialize_call_service(self):
        """Test call service initialization."""
        service = CallService()
        
        with patch('telephony.sip_gateway.sip_gateway.initialize') as mock_init, \
             patch('telephony.sip_gateway.sip_gateway.register_account') as mock_register, \
             patch('backend.app.core.config.settings') as mock_settings:
            
            mock_settings.SIP_SERVER = "sip.example.com"
            mock_settings.SIP_USERNAME = "test_user"
            mock_settings.SIP_PASSWORD = "test_pass"
            mock_settings.SIP_DOMAIN = "sip.example.com"
            mock_settings.SIP_PORT = 5060
            
            await service.initialize()
            
            mock_init.assert_called_once()
            mock_register.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_inbound_call(self, mock_db: AsyncSession, mock_call_info):
        """Test handling inbound call."""
        service = CallService()
        
        with patch.object(service.routing_service, 'route_inbound_call') as mock_route, \
             patch.object(service, '_start_ai_session') as mock_ai:
            
            mock_route.return_value = {
                "destination": "ai_agent",
                "priority": "normal",
                "metadata": {"existing_lead": None}
            }
            
            call_record = await service.handle_inbound_call(mock_call_info, mock_db)
            
            assert call_record.call_id == mock_call_info.call_id
            assert call_record.direction == "inbound"
            assert call_record.status == "ringing"
            mock_ai.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_outbound_call(self, mock_db: AsyncSession):
        """Test making outbound call."""
        service = CallService()
        
        with patch('telephony.sip_gateway.sip_gateway.make_call') as mock_make_call:
            mock_make_call.return_value = "call_123"
            
            call_record = await service.make_outbound_call(
                destination="+1234567890",
                caller_id="+1987654321",
                db=mock_db
            )
            
            assert call_record is not None
            assert call_record.direction == "outbound"
            assert call_record.called_number == "+1234567890"
            assert call_record.caller_number == "+1987654321"
    
    @pytest.mark.asyncio
    async def test_update_call_status(self, mock_db: AsyncSession, mock_call_record):
        """Test updating call status."""
        service = CallService()
        
        with patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute, \
             patch.object(service, '_log_call_event') as mock_log:
            
            mock_execute.return_value.scalar_one_or_none.return_value = mock_call_record
            
            await service.update_call_status(
                call_id=mock_call_record.call_id,
                status="answered",
                db=mock_db
            )
            
            assert mock_call_record.status == "answered"
            mock_log.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_transfer_call(self, mock_db: AsyncSession, mock_call_record):
        """Test call transfer."""
        service = CallService()
        mock_call_record.status = "answered"
        
        with patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute, \
             patch('telephony.sip_gateway.sip_gateway.transfer_call') as mock_transfer:
            
            mock_execute.return_value.scalar_one_or_none.return_value = mock_call_record
            
            success = await service.transfer_call(
                call_id=str(mock_call_record.id),
                destination="+1555123456",
                db=mock_db
            )
            
            assert success is True
            assert mock_call_record.transferred_to == "+1555123456"
            assert mock_call_record.transfer_count == 1
    
    @pytest.mark.asyncio
    async def test_call_analytics(self, mock_db: AsyncSession):
        """Test call analytics generation."""
        service = CallService()
        
        # Mock analytics data
        with patch('sqlalchemy.ext.asyncio.AsyncSession.scalar') as mock_scalar, \
             patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute:
            
            mock_scalar.side_effect = [100, 75, 180.5]  # total, answered, avg_duration
            mock_execute.return_value.all.return_value = [
                ("answered", 75),
                ("no_answer", 15),
                ("busy", 10)
            ]
            
            analytics = await service.get_call_analytics(db=mock_db)
            
            assert analytics["total_calls"] == 100
            assert analytics["answered_calls"] == 75
            assert analytics["answer_rate"] == 75.0
            assert analytics["average_duration"] == 180.5
            assert "call_outcomes" in analytics


class TestCallRouting:
    """Test suite for call routing functionality."""
    
    @pytest.mark.asyncio
    async def test_route_inbound_call_new_caller(self, mock_db: AsyncSession):
        """Test routing for new caller."""
        from backend.app.services.call_service import CallRoutingService
        
        with patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute:
            mock_execute.return_value.scalar_one_or_none.return_value = None  # No existing lead
            
            routing = await CallRoutingService.route_inbound_call(
                "+1234567890",
                "+1987654321",
                mock_db
            )
            
            assert routing["destination"] == "ai_agent"
            assert routing["priority"] == "normal"
            assert routing["metadata"]["existing_lead"] is None
    
    @pytest.mark.asyncio
    async def test_route_inbound_call_existing_high_priority_lead(self, mock_db: AsyncSession):
        """Test routing for existing high-priority lead."""
        from backend.app.services.call_service import CallRoutingService
        from backend.app.models.crm import Lead
        
        high_priority_lead = Lead(
            id=uuid4(),
            phone="+1234567890",
            priority="urgent"
        )
        
        with patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute:
            mock_execute.return_value.scalar_one_or_none.return_value = high_priority_lead
            
            routing = await CallRoutingService.route_inbound_call(
                "+1234567890",
                "+1987654321",
                mock_db
            )
            
            assert routing["destination"] == "ai_agent"
            assert routing["priority"] == "high"
            assert routing["metadata"]["existing_lead"] == high_priority_lead.id


class TestTelephonyAPIEndpoints:
    """Test suite for Telephony API endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_sip_account_endpoint(self, client: AsyncClient):
        """Test SIP account creation endpoint."""
        account_data = {
            "name": "Test Account",
            "username": "test_user",
            "password": "test_pass",
            "domain": "sip.example.com",
            "server": "sip.example.com",
            "port": 5060,
            "transport": "UDP"
        }
        
        response = await client.post("/api/v1/telephony/sip-accounts", json=account_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Account"
        assert data["username"] == "test_user"
        assert data["domain"] == "sip.example.com"
    
    @pytest.mark.asyncio
    async def test_make_call_endpoint(self, client: AsyncClient):
        """Test outbound call endpoint."""
        call_data = {
            "destination": "+1234567890",
            "caller_id": "+1987654321"
        }
        
        with patch('backend.app.services.call_service.call_service.make_outbound_call') as mock_call:
            mock_call_record = CallRecord(
                id=uuid4(),
                call_id="call_123",
                direction="outbound",
                called_number="+1234567890",
                status="calling"
            )
            mock_call.return_value = mock_call_record
            
            response = await client.post("/api/v1/telephony/calls", json=call_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["direction"] == "outbound"
            assert data["called_number"] == "+1234567890"
    
    @pytest.mark.asyncio
    async def test_hangup_call_endpoint(self, client: AsyncClient, mock_call_record):
        """Test call hangup endpoint."""
        with patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute, \
             patch('backend.app.services.call_service.call_service.hangup_call') as mock_hangup:
            
            mock_execute.return_value.scalar_one_or_none.return_value = mock_call_record
            
            response = await client.post(f"/api/v1/telephony/calls/{mock_call_record.id}/hangup")
            
            assert response.status_code == 200
            data = response.json()
            assert "hung up successfully" in data["message"]
    
    @pytest.mark.asyncio
    async def test_transfer_call_endpoint(self, client: AsyncClient, mock_call_record):
        """Test call transfer endpoint."""
        mock_call_record.status = "answered"
        
        with patch('sqlalchemy.ext.asyncio.AsyncSession.execute') as mock_execute, \
             patch('backend.app.services.call_service.call_service.transfer_call') as mock_transfer:
            
            mock_execute.return_value.scalar_one_or_none.return_value = mock_call_record
            mock_transfer.return_value = True
            
            response = await client.post(
                f"/api/v1/telephony/calls/{mock_call_record.id}/transfer",
                params={"destination": "+1555123456"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "transferred" in data["message"]
    
    @pytest.mark.asyncio
    async def test_telephony_status_endpoint(self, client: AsyncClient):
        """Test telephony system status endpoint."""
        with patch('telephony.sip_gateway.sip_gateway.is_running', True), \
             patch('telephony.sip_gateway.sip_gateway.get_active_calls', return_value=[]):
            
            response = await client.get("/api/v1/telephony/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["sip_gateway_running"] is True
            assert data["system_status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_telephony_analytics_endpoint(self, client: AsyncClient):
        """Test telephony analytics endpoint."""
        mock_analytics = {
            "total_calls": 100,
            "answered_calls": 75,
            "answer_rate": 75.0,
            "average_duration": 180.5
        }
        
        with patch('backend.app.services.call_service.call_service.get_call_analytics') as mock_analytics_func:
            mock_analytics_func.return_value = mock_analytics
            
            response = await client.get("/api/v1/telephony/analytics")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_calls"] == 100
            assert data["answer_rate"] == 75.0


@pytest.mark.integration
class TestTelephonyIntegration:
    """Integration tests for telephony functionality."""
    
    @pytest.mark.asyncio
    async def test_full_call_lifecycle(self, mock_db: AsyncSession):
        """Test complete call lifecycle from initiation to completion."""
        # This would test the full flow:
        # 1. Register SIP account
        # 2. Receive inbound call
        # 3. Route call to AI agent
        # 4. Process conversation
        # 5. Create lead if needed
        # 6. End call and update records
        
        pass
    
    @pytest.mark.asyncio
    async def test_concurrent_call_handling(self, mock_db: AsyncSession):
        """Test handling multiple concurrent calls."""
        # This would test:
        # 1. Multiple simultaneous inbound calls
        # 2. Resource allocation and limits
        # 3. Call quality maintenance
        # 4. Proper cleanup on completion
        
        pass
    
    @pytest.mark.asyncio
    async def test_call_queue_management(self, mock_db: AsyncSession):
        """Test call queue functionality."""
        # This would test:
        # 1. Adding calls to queue
        # 2. Queue member assignment
        # 3. Wait time tracking
        # 4. Overflow handling
        
        pass


# Import required modules for mocking
import unittest.mock

# Mock fixtures for testing
@pytest.fixture
async def mock_db():
    """Mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
async def client():
    """Mock HTTP client for API testing."""
    return AsyncMock(spec=AsyncClient)