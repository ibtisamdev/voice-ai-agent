"""
SIP Gateway Implementation

Handles SIP protocol communication for inbound and outbound calls
in the Voice AI Agent system.
"""

import asyncio
import logging
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from uuid import uuid4
import socket
import threading
from dataclasses import dataclass
from enum import Enum

try:
    import pjsua2 as pj
    HAS_PJSUA = True
except ImportError:
    HAS_PJSUA = False
    logging.warning("PJSUA2 not available. SIP functionality will be limited.")

from ..backend.app.core.config import settings
from ..backend.app.models.telephony import CallRecord, CallEvent, SIPAccount

logger = logging.getLogger(__name__)


class CallState(Enum):
    """Call state enumeration."""
    IDLE = "idle"
    CALLING = "calling"
    INCOMING = "incoming"
    EARLY = "early"
    CONNECTING = "connecting"
    CONFIRMED = "confirmed"
    DISCONNECTED = "disconnected"


@dataclass
class CallInfo:
    """Call information container."""
    call_id: str
    sip_call_id: str
    caller_number: str
    called_number: str
    direction: str  # inbound, outbound
    state: CallState
    start_time: datetime
    answer_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: int = 0
    audio_active: bool = False
    recording_active: bool = False


class AudioBridge:
    """Handles audio streaming between SIP and AI processing."""
    
    def __init__(self, call_info: CallInfo):
        self.call_info = call_info
        self.audio_callbacks: List[Callable] = []
        self.recording_buffer: List[bytes] = []
        self.is_recording = False
        
    def add_audio_callback(self, callback: Callable[[bytes], None]):
        """Add callback for audio data."""
        self.audio_callbacks.append(callback)
    
    def remove_audio_callback(self, callback: Callable):
        """Remove audio callback."""
        if callback in self.audio_callbacks:
            self.audio_callbacks.remove(callback)
    
    def on_audio_received(self, audio_data: bytes):
        """Handle incoming audio data from SIP."""
        # Forward to AI processing
        for callback in self.audio_callbacks:
            try:
                callback(audio_data)
            except Exception as e:
                logger.error(f"Audio callback error: {e}")
        
        # Record if enabled
        if self.is_recording:
            self.recording_buffer.append(audio_data)
    
    def send_audio(self, audio_data: bytes):
        """Send audio data to SIP call."""
        # This would be implemented to send audio back to the call
        pass
    
    def start_recording(self):
        """Start recording call audio."""
        self.is_recording = True
        self.recording_buffer.clear()
    
    def stop_recording(self) -> bytes:
        """Stop recording and return audio data."""
        self.is_recording = False
        return b''.join(self.recording_buffer)


if HAS_PJSUA:
    class VoiceAICall(pj.Call):
        """Custom call class with AI integration."""
        
        def __init__(self, account, call_id: str = None):
            pj.Call.__init__(self, account, call_id)
            self.call_info = None
            self.audio_bridge = None
            self.ai_session_id = None
            self.call_callbacks: List[Callable] = []
        
        def add_call_callback(self, callback: Callable):
            """Add callback for call events."""
            self.call_callbacks.append(callback)
        
        def on_call_state(self):
            """Handle call state changes."""
            try:
                ci = self.getInfo()
                self._update_call_info(ci)
                
                # Notify callbacks
                for callback in self.call_callbacks:
                    try:
                        callback("state_change", self.call_info)
                    except Exception as e:
                        logger.error(f"Call callback error: {e}")
                
                logger.info(f"Call {self.call_info.call_id} state: {self.call_info.state.value}")
                
                if ci.state == pj.PJSIP_INV_STATE_CONFIRMED:
                    # Call answered, setup audio
                    self._setup_audio()
                elif ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
                    # Call ended
                    self._cleanup_call()
                    
            except Exception as e:
                logger.error(f"Error handling call state change: {e}")
        
        def on_call_media_state(self):
            """Handle media state changes."""
            try:
                ci = self.getInfo()
                
                for mi in ci.media:
                    if mi.type == pj.PJMEDIA_TYPE_AUDIO and mi.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                        # Audio is active
                        if self.audio_bridge:
                            self.audio_bridge.call_info.audio_active = True
                        
                        # Start AI session if not already started
                        if not self.ai_session_id:
                            self._start_ai_session()
                        
                        logger.info(f"Audio active for call {self.call_info.call_id}")
                        
            except Exception as e:
                logger.error(f"Error handling media state change: {e}")
        
        def _update_call_info(self, ci):
            """Update call information from PJSUA call info."""
            if not self.call_info:
                self.call_info = CallInfo(
                    call_id=str(uuid4()),
                    sip_call_id=ci.callIdString,
                    caller_number=ci.remoteUri,
                    called_number=ci.localUri,
                    direction="inbound" if ci.role == pj.PJSIP_ROLE_UAS else "outbound",
                    state=CallState.IDLE,
                    start_time=datetime.now(timezone.utc)
                )
            
            # Update state
            if ci.state == pj.PJSIP_INV_STATE_CALLING:
                self.call_info.state = CallState.CALLING
            elif ci.state == pj.PJSIP_INV_STATE_INCOMING:
                self.call_info.state = CallState.INCOMING
            elif ci.state == pj.PJSIP_INV_STATE_EARLY:
                self.call_info.state = CallState.EARLY
            elif ci.state == pj.PJSIP_INV_STATE_CONNECTING:
                self.call_info.state = CallState.CONNECTING
            elif ci.state == pj.PJSIP_INV_STATE_CONFIRMED:
                self.call_info.state = CallState.CONFIRMED
                if not self.call_info.answer_time:
                    self.call_info.answer_time = datetime.now(timezone.utc)
            elif ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
                self.call_info.state = CallState.DISCONNECTED
                self.call_info.end_time = datetime.now(timezone.utc)
                
                # Calculate duration
                if self.call_info.answer_time:
                    duration = self.call_info.end_time - self.call_info.answer_time
                    self.call_info.duration = int(duration.total_seconds())
        
        def _setup_audio(self):
            """Setup audio bridge for AI processing."""
            if not self.audio_bridge:
                self.audio_bridge = AudioBridge(self.call_info)
                
                # Start recording if enabled
                if settings.ENABLE_CALL_RECORDING:
                    self.audio_bridge.start_recording()
        
        def _start_ai_session(self):
            """Start AI session for this call."""
            self.ai_session_id = str(uuid4())
            # This would integrate with the existing voice AI system
            logger.info(f"Started AI session {self.ai_session_id} for call {self.call_info.call_id}")
        
        def _cleanup_call(self):
            """Cleanup call resources."""
            if self.audio_bridge and self.audio_bridge.is_recording:
                audio_data = self.audio_bridge.stop_recording()
                # Save recording if needed
                
            # End AI session
            if self.ai_session_id:
                logger.info(f"Ended AI session {self.ai_session_id}")


    class VoiceAIAccount(pj.Account):
        """Custom account class for SIP registration."""
        
        def __init__(self, sip_config: Dict[str, Any]):
            pj.Account.__init__(self)
            self.sip_config = sip_config
            self.active_calls: Dict[str, VoiceAICall] = {}
            self.call_callbacks: List[Callable] = []
        
        def add_call_callback(self, callback: Callable):
            """Add callback for new calls."""
            self.call_callbacks.append(callback)
        
        def on_incoming_call(self, prm):
            """Handle incoming calls."""
            try:
                # Create new call
                call = VoiceAICall(self)
                
                # Add to active calls
                call_id = str(uuid4())
                self.active_calls[call_id] = call
                
                # Setup call callbacks
                for callback in self.call_callbacks:
                    call.add_call_callback(callback)
                
                # Answer the call
                call_prm = pj.CallOpParam()
                call_prm.statusCode = 200
                call.answer(call_prm)
                
                logger.info(f"Incoming call answered: {call_id}")
                
            except Exception as e:
                logger.error(f"Error handling incoming call: {e}")
        
        def on_reg_state(self):
            """Handle registration state changes."""
            try:
                ai = self.getInfo()
                if ai.regIsActive:
                    logger.info(f"SIP registration active for {self.sip_config['username']}")
                else:
                    logger.warning(f"SIP registration lost for {self.sip_config['username']}")
            except Exception as e:
                logger.error(f"Error handling registration state: {e}")
        
        def make_outbound_call(self, destination: str) -> Optional[VoiceAICall]:
            """Make outbound call."""
            try:
                call = VoiceAICall(self)
                
                # Add to active calls
                call_id = str(uuid4())
                self.active_calls[call_id] = call
                
                # Setup call callbacks
                for callback in self.call_callbacks:
                    call.add_call_callback(callback)
                
                # Make the call
                call_prm = pj.CallOpParam()
                call.makeCall(destination, call_prm)
                
                logger.info(f"Outbound call initiated to {destination}: {call_id}")
                return call
                
            except Exception as e:
                logger.error(f"Error making outbound call: {e}")
                return None


class SIPGateway:
    """
    Main SIP Gateway class.
    
    Handles SIP registration, call management, and integration
    with the Voice AI system.
    """
    
    def __init__(self):
        self.endpoint: Optional[pj.Endpoint] = None
        self.transport: Optional[pj.Transport] = None
        self.accounts: Dict[str, VoiceAIAccount] = {}
        self.active_calls: Dict[str, VoiceAICall] = {}
        self.call_callbacks: List[Callable] = []
        self.is_running = False
        
        if not HAS_PJSUA:
            logger.error("PJSUA2 not available. SIP functionality disabled.")
    
    async def initialize(self):
        """Initialize SIP gateway."""
        if not HAS_PJSUA:
            raise RuntimeError("PJSUA2 not available")
        
        try:
            # Create endpoint
            self.endpoint = pj.Endpoint()
            self.endpoint.libCreate()
            
            # Initialize endpoint
            ep_cfg = pj.EpConfig()
            ep_cfg.logConfig.level = 3  # Info level
            ep_cfg.logConfig.consoleLevel = 3
            
            self.endpoint.libInit(ep_cfg)
            
            # Create transport
            transport_cfg = pj.TransportConfig()
            transport_cfg.port = settings.SIP_PORT
            
            if settings.SIP_TRANSPORT.upper() == "UDP":
                self.transport = self.endpoint.transportCreate(pj.PJSIP_TRANSPORT_UDP, transport_cfg)
            elif settings.SIP_TRANSPORT.upper() == "TCP":
                self.transport = self.endpoint.transportCreate(pj.PJSIP_TRANSPORT_TCP, transport_cfg)
            else:
                raise ValueError(f"Unsupported transport: {settings.SIP_TRANSPORT}")
            
            # Start endpoint
            self.endpoint.libStart()
            self.is_running = True
            
            logger.info(f"SIP Gateway initialized on port {settings.SIP_PORT}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SIP gateway: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown SIP gateway."""
        if not self.is_running:
            return
        
        try:
            # Hangup all active calls
            for call in self.active_calls.values():
                try:
                    call.hangup(pj.CallOpParam())
                except:
                    pass
            
            # Destroy accounts
            for account in self.accounts.values():
                try:
                    account.delete()
                except:
                    pass
            
            # Destroy endpoint
            if self.endpoint:
                self.endpoint.libDestroy()
            
            self.is_running = False
            logger.info("SIP Gateway shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during SIP gateway shutdown: {e}")
    
    async def register_account(self, sip_config: Dict[str, Any]) -> str:
        """
        Register SIP account.
        
        Args:
            sip_config: SIP configuration dictionary
            
        Returns:
            Account ID
        """
        if not self.is_running:
            raise RuntimeError("SIP Gateway not initialized")
        
        try:
            # Create account
            account = VoiceAIAccount(sip_config)
            
            # Configure account
            acc_cfg = pj.AccountConfig()
            acc_cfg.idUri = f"sip:{sip_config['username']}@{sip_config['domain']}"
            acc_cfg.regConfig.registrarUri = f"sip:{sip_config['server']}"
            
            # Authentication
            cred = pj.AuthCredInfo("digest", "*", sip_config['username'], 0, sip_config['password'])
            acc_cfg.sipConfig.authCreds.append(cred)
            
            # Create and register
            account.create(acc_cfg)
            
            # Add callbacks
            for callback in self.call_callbacks:
                account.add_call_callback(callback)
            
            # Store account
            account_id = sip_config['username']
            self.accounts[account_id] = account
            
            logger.info(f"SIP account registered: {account_id}")
            return account_id
            
        except Exception as e:
            logger.error(f"Failed to register SIP account: {e}")
            raise
    
    def add_call_callback(self, callback: Callable):
        """Add callback for call events."""
        self.call_callbacks.append(callback)
        
        # Add to existing accounts
        for account in self.accounts.values():
            account.add_call_callback(callback)
    
    async def make_call(
        self, 
        account_id: str, 
        destination: str,
        call_data: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Make outbound call.
        
        Args:
            account_id: SIP account ID
            destination: Destination number/URI
            call_data: Additional call metadata
            
        Returns:
            Call ID if successful
        """
        if account_id not in self.accounts:
            raise ValueError(f"Account not found: {account_id}")
        
        account = self.accounts[account_id]
        call = account.make_outbound_call(destination)
        
        if call and call.call_info:
            call_id = call.call_info.call_id
            self.active_calls[call_id] = call
            
            # Store additional metadata
            if call_data:
                call.call_info.__dict__.update(call_data)
            
            return call_id
        
        return None
    
    async def hangup_call(self, call_id: str):
        """Hangup call by ID."""
        if call_id in self.active_calls:
            call = self.active_calls[call_id]
            try:
                call.hangup(pj.CallOpParam())
                del self.active_calls[call_id]
                logger.info(f"Call {call_id} hung up")
            except Exception as e:
                logger.error(f"Error hanging up call {call_id}: {e}")
    
    async def transfer_call(self, call_id: str, destination: str):
        """Transfer call to another destination."""
        if call_id in self.active_calls:
            call = self.active_calls[call_id]
            try:
                # Unattended transfer
                prm = pj.CallOpParam()
                call.xfer(destination, prm)
                logger.info(f"Call {call_id} transferred to {destination}")
            except Exception as e:
                logger.error(f"Error transferring call {call_id}: {e}")
    
    def get_call_info(self, call_id: str) -> Optional[CallInfo]:
        """Get call information."""
        if call_id in self.active_calls:
            return self.active_calls[call_id].call_info
        return None
    
    def get_active_calls(self) -> List[CallInfo]:
        """Get all active call information."""
        return [call.call_info for call in self.active_calls.values() if call.call_info]
    
    async def send_dtmf(self, call_id: str, digits: str):
        """Send DTMF tones to call."""
        if call_id in self.active_calls:
            call = self.active_calls[call_id]
            try:
                call.dialDtmf(digits)
                logger.info(f"DTMF sent to call {call_id}: {digits}")
            except Exception as e:
                logger.error(f"Error sending DTMF to call {call_id}: {e}")


# Fallback implementation when PJSUA is not available
class MockSIPGateway:
    """Mock SIP Gateway for development/testing when PJSUA is not available."""
    
    def __init__(self):
        self.is_running = False
        self.active_calls = {}
        self.call_callbacks = []
    
    async def initialize(self):
        self.is_running = True
        logger.warning("Using Mock SIP Gateway - no real SIP functionality")
    
    async def shutdown(self):
        self.is_running = False
    
    async def register_account(self, sip_config: Dict[str, Any]) -> str:
        return sip_config['username']
    
    def add_call_callback(self, callback: Callable):
        self.call_callbacks.append(callback)
    
    async def make_call(self, account_id: str, destination: str, call_data: Dict[str, Any] = None) -> Optional[str]:
        call_id = str(uuid4())
        logger.info(f"Mock call initiated: {call_id} to {destination}")
        return call_id
    
    async def hangup_call(self, call_id: str):
        logger.info(f"Mock call hung up: {call_id}")
    
    async def transfer_call(self, call_id: str, destination: str):
        logger.info(f"Mock call transferred: {call_id} to {destination}")
    
    def get_call_info(self, call_id: str) -> Optional[CallInfo]:
        return None
    
    def get_active_calls(self) -> List[CallInfo]:
        return []
    
    async def send_dtmf(self, call_id: str, digits: str):
        logger.info(f"Mock DTMF sent: {digits} to call {call_id}")


# Create gateway instance
if HAS_PJSUA:
    sip_gateway = SIPGateway()
else:
    sip_gateway = MockSIPGateway()