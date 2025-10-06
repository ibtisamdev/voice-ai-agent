"""
WebSocket endpoints for real-time voice processing.
Handles audio streaming, transcription, and synthesis over WebSocket connections.
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Dict, Optional, Any
from datetime import datetime
import base64

from fastapi import WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.routing import APIRouter
import numpy as np

from ai.voice import audio_processor, stt_service, tts_service
from ai.conversation import conversation_state_manager, dialog_flow_engine
from ai.decision_engine import intent_classifier
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for voice sessions."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept a WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_metadata[session_id] = {
            "connected_at": time.time(),
            "last_activity": time.time(),
            "audio_chunks_received": 0,
            "transcriptions_sent": 0,
            "synthesis_requests": 0
        }
        logger.info(f"WebSocket connected for session {session_id}")
    
    def disconnect(self, session_id: str):
        """Disconnect a WebSocket."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_metadata:
            del self.session_metadata[session_id]
        logger.info(f"WebSocket disconnected for session {session_id}")
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send a message to a specific session."""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_text(json.dumps(message))
                self._update_activity(session_id)
            except Exception as e:
                logger.error(f"Error sending message to {session_id}: {e}")
                self.disconnect(session_id)
    
    async def send_audio(self, session_id: str, audio_data: bytes, audio_format: str = "wav"):
        """Send audio data to a specific session."""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                # Encode audio as base64
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                message = {
                    "type": "audio",
                    "data": audio_b64,
                    "format": audio_format,
                    "timestamp": time.time()
                }
                await websocket.send_text(json.dumps(message))
                self._update_activity(session_id)
            except Exception as e:
                logger.error(f"Error sending audio to {session_id}: {e}")
                self.disconnect(session_id)
    
    def _update_activity(self, session_id: str):
        """Update last activity timestamp."""
        if session_id in self.session_metadata:
            self.session_metadata[session_id]["last_activity"] = time.time()
    
    def get_connection_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get connection metadata."""
        return self.session_metadata.get(session_id)
    
    def get_active_sessions(self) -> list:
        """Get list of active session IDs."""
        return list(self.active_connections.keys())


# Global connection manager
manager = ConnectionManager()


class VoiceSessionHandler:
    """Handles voice session logic and coordination."""
    
    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.conversation_session = None
        self.active_flow = None
        
        # Audio processing settings
        self.audio_buffer = bytearray()
        self.buffer_size_ms = 1000  # Process 1-second chunks
        self.sample_rate = 16000
        self.bytes_per_sample = 2
        self.chunk_size = int(self.sample_rate * self.buffer_size_ms / 1000 * self.bytes_per_sample)
        
        # Session state
        self.is_processing = False
        self.last_transcription = ""
        self.conversation_context = {}
    
    async def initialize(self, call_direction: str = "inbound", phone_number: str = None):
        """Initialize the voice session."""
        try:
            # Create conversation session
            from ai.conversation.state_manager import CallDirection, ParticipantInfo
            
            direction = CallDirection.INBOUND if call_direction == "inbound" else CallDirection.OUTBOUND
            
            participant = ParticipantInfo(
                id=str(uuid.uuid4()),
                phone_number=phone_number,
                role="caller"
            )
            
            self.conversation_session = await conversation_state_manager.create_session(
                direction=direction,
                phone_number=phone_number,
                participant_info=participant,
                metadata={"websocket_session": self.session_id}
            )
            
            # Start default conversation flow
            await self.start_conversation_flow("legal_consultation")
            
            logger.info(f"Voice session initialized for {self.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing voice session: {e}")
            return False
    
    async def start_conversation_flow(self, flow_id: str):
        """Start a conversation flow."""
        try:
            if not self.conversation_session:
                raise Exception("Conversation session not initialized")
            
            # Start flow
            response = await dialog_flow_engine.start_flow(self.conversation_session.session_id, flow_id)
            
            if response:
                # Send initial message via TTS
                await self.synthesize_and_send(response)
                self.active_flow = flow_id
            
        except Exception as e:
            logger.error(f"Error starting conversation flow: {e}")
    
    async def process_audio_chunk(self, audio_data: bytes):
        """Process incoming audio data."""
        try:
            # Add to buffer
            self.audio_buffer.extend(audio_data)
            
            # Process if buffer is large enough
            if len(self.audio_buffer) >= self.chunk_size:
                # Extract chunk
                chunk_data = bytes(self.audio_buffer[:self.chunk_size])
                self.audio_buffer = self.audio_buffer[self.chunk_size:]
                
                # Process chunk
                await self._process_audio_chunk(chunk_data)
                
                # Update metadata
                if self.session_id in manager.session_metadata:
                    manager.session_metadata[self.session_id]["audio_chunks_received"] += 1
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    async def _process_audio_chunk(self, chunk_data: bytes):
        """Process a single audio chunk."""
        try:
            # Process audio (VAD, noise reduction)
            from ai.voice.audio_processor import AudioChunk
            
            audio_chunk = AudioChunk(
                data=chunk_data,
                timestamp=time.time(),
                is_speech=True,  # Will be determined by VAD
                confidence=1.0
            )
            
            processed_chunk = audio_processor.process_audio_chunk(
                chunk_data,
                apply_noise_reduction=True,
                apply_vad=True
            )
            
            # Only transcribe if speech is detected
            if processed_chunk.is_speech and processed_chunk.confidence > 0.5:
                await self._transcribe_chunk(processed_chunk)
            
        except Exception as e:
            logger.error(f"Error in audio chunk processing: {e}")
    
    async def _transcribe_chunk(self, audio_chunk):
        """Transcribe an audio chunk."""
        try:
            if self.is_processing:
                return  # Skip if already processing
            
            self.is_processing = True
            
            # Transcribe
            result = await stt_service.transcribe_audio_chunk(audio_chunk)
            
            if result.text.strip():
                # Send transcription to client
                await manager.send_message(self.session_id, {
                    "type": "transcription",
                    "text": result.text,
                    "confidence": result.confidence,
                    "is_final": True,
                    "timestamp": result.timestamp
                })
                
                # Process with conversation flow
                await self._process_user_input(result.text)
                
                # Update metadata
                if self.session_id in manager.session_metadata:
                    manager.session_metadata[self.session_id]["transcriptions_sent"] += 1
            
        except Exception as e:
            logger.error(f"Error transcribing chunk: {e}")
        finally:
            self.is_processing = False
    
    async def _process_user_input(self, text: str):
        """Process user input through conversation system."""
        try:
            # Classify intent
            intent_result = await intent_classifier.classify_intent(
                text, 
                context=self.conversation_context
            )
            
            # Add conversation turn
            if self.conversation_session:
                await conversation_state_manager.add_conversation_turn(
                    session_id=self.conversation_session.session_id,
                    speaker_id="user",
                    speaker_role="caller",
                    input_text=text,
                    intent=intent_result.primary_intent.name,
                    intent_confidence=intent_result.primary_intent.confidence,
                    entities=intent_result.primary_intent.entities
                )
            
            # Process through dialog flow
            if self.active_flow and self.conversation_session:
                response = await dialog_flow_engine.process_user_input(
                    self.conversation_session.session_id,
                    text
                )
                
                if response:
                    await self.synthesize_and_send(response)
                    
                    # Add assistant turn
                    await conversation_state_manager.add_conversation_turn(
                        session_id=self.conversation_session.session_id,
                        speaker_id="ai_agent",
                        speaker_role="bot",
                        response_text=response
                    )
            
            # Update context
            self.conversation_context.update({
                "last_user_input": text,
                "last_intent": intent_result.primary_intent.name,
                "intent_confidence": intent_result.primary_intent.confidence
            })
            
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
    
    async def synthesize_and_send(self, text: str):
        """Synthesize text to speech and send audio."""
        try:
            # Synthesize speech
            tts_result = await tts_service.synthesize(
                text=text,
                voice_id=None,  # Use default voice
                speed=1.0,
                use_cache=True
            )
            
            # Send audio to client
            await manager.send_audio(
                self.session_id,
                tts_result.audio_data,
                tts_result.audio_format
            )
            
            # Send text as well for accessibility
            await manager.send_message(self.session_id, {
                "type": "synthesis",
                "text": text,
                "voice_id": tts_result.voice_id,
                "duration_ms": tts_result.duration_ms,
                "timestamp": tts_result.timestamp
            })
            
            # Update metadata
            if self.session_id in manager.session_metadata:
                manager.session_metadata[self.session_id]["synthesis_requests"] += 1
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            # Send text-only fallback
            await manager.send_message(self.session_id, {
                "type": "synthesis_error",
                "text": text,
                "error": "TTS synthesis failed"
            })
    
    async def handle_message(self, message: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        try:
            message_type = message.get("type")
            
            if message_type == "audio":
                # Audio data
                audio_b64 = message.get("data")
                if audio_b64:
                    audio_data = base64.b64decode(audio_b64)
                    await self.process_audio_chunk(audio_data)
            
            elif message_type == "text":
                # Text input
                text = message.get("text", "").strip()
                if text:
                    await self._process_user_input(text)
            
            elif message_type == "control":
                # Control commands
                command = message.get("command")
                await self._handle_control_command(command, message.get("params", {}))
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_control_command(self, command: str, params: Dict[str, Any]):
        """Handle control commands."""
        try:
            if command == "start_flow":
                flow_id = params.get("flow_id", "legal_consultation")
                await self.start_conversation_flow(flow_id)
            
            elif command == "end_session":
                if self.conversation_session:
                    await conversation_state_manager.end_session(
                        self.conversation_session.session_id,
                        resolution_status=params.get("resolution", "completed")
                    )
            
            elif command == "mute":
                # Handle mute command
                await manager.send_message(self.session_id, {
                    "type": "control_ack",
                    "command": "mute",
                    "status": "acknowledged"
                })
            
            elif command == "unmute":
                # Handle unmute command
                await manager.send_message(self.session_id, {
                    "type": "control_ack", 
                    "command": "unmute",
                    "status": "acknowledged"
                })
            
            else:
                logger.warning(f"Unknown control command: {command}")
                
        except Exception as e:
            logger.error(f"Error handling control command: {e}")
    
    async def cleanup(self):
        """Clean up session resources."""
        try:
            if self.conversation_session:
                await conversation_state_manager.end_session(
                    self.conversation_session.session_id,
                    resolution_status="disconnected"
                )
            
            logger.info(f"Voice session {self.session_id} cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")


# Active voice sessions
voice_sessions: Dict[str, VoiceSessionHandler] = {}


@router.websocket("/voice/stream/{session_id}")
async def voice_stream_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time voice streaming.
    
    Handles bidirectional audio streaming, transcription, and synthesis.
    """
    try:
        # Connect WebSocket
        await manager.connect(websocket, session_id)
        
        # Create voice session handler
        voice_handler = VoiceSessionHandler(session_id, websocket)
        voice_sessions[session_id] = voice_handler
        
        # Initialize session
        await voice_handler.initialize()
        
        # Send welcome message
        await manager.send_message(session_id, {
            "type": "connected",
            "session_id": session_id,
            "message": "Voice session connected",
            "timestamp": time.time()
        })
        
        # Message loop
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle message
                await voice_handler.handle_message(message)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for session {session_id}")
                break
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from session {session_id}")
                await manager.send_message(session_id, {
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                logger.error(f"Error in voice stream: {e}")
                await manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                })
                
    except Exception as e:
        logger.error(f"Error in voice stream endpoint: {e}")
    finally:
        # Cleanup
        if session_id in voice_sessions:
            await voice_sessions[session_id].cleanup()
            del voice_sessions[session_id]
        
        manager.disconnect(session_id)


@router.websocket("/voice/transcription/{session_id}")
async def transcription_stream_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming transcription only.
    
    Provides real-time transcription without conversation flow.
    """
    try:
        await manager.connect(websocket, session_id)
        
        # Send welcome
        await manager.send_message(session_id, {
            "type": "transcription_ready",
            "session_id": session_id,
            "timestamp": time.time()
        })
        
        # Start STT streaming session
        await stt_service.start_streaming_session(session_id)
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "audio":
                    # Process audio for transcription only
                    audio_b64 = message.get("data")
                    if audio_b64:
                        audio_data = base64.b64decode(audio_b64)
                        
                        # Create audio chunk
                        from ai.voice.audio_processor import AudioChunk
                        chunk = AudioChunk(
                            data=audio_data,
                            timestamp=time.time(),
                            is_speech=True,
                            confidence=1.0
                        )
                        
                        # Process with STT streaming
                        result = await stt_service.process_streaming_audio(session_id, chunk)
                        
                        if result:
                            await manager.send_message(session_id, {
                                "type": "transcription",
                                "text": result.text,
                                "confidence": result.confidence,
                                "is_partial": result.is_partial,
                                "timestamp": result.timestamp
                            })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in transcription stream: {e}")
                
    except Exception as e:
        logger.error(f"Error in transcription endpoint: {e}")
    finally:
        # End STT session
        await stt_service.end_streaming_session(session_id)
        manager.disconnect(session_id)


@router.get("/voice/sessions")
async def get_active_sessions():
    """Get list of active voice sessions."""
    try:
        sessions = []
        for session_id in manager.get_active_sessions():
            info = manager.get_connection_info(session_id)
            if info:
                sessions.append({
                    "session_id": session_id,
                    "connected_at": info["connected_at"],
                    "last_activity": info["last_activity"],
                    "stats": {
                        "audio_chunks_received": info["audio_chunks_received"],
                        "transcriptions_sent": info["transcriptions_sent"],
                        "synthesis_requests": info["synthesis_requests"]
                    }
                })
        
        return {
            "active_sessions": len(sessions),
            "sessions": sessions,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting active sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/voice/sessions/{session_id}")
async def disconnect_session(session_id: str):
    """Disconnect a specific voice session."""
    try:
        if session_id in voice_sessions:
            await voice_sessions[session_id].cleanup()
            del voice_sessions[session_id]
        
        manager.disconnect(session_id)
        
        return {
            "session_id": session_id,
            "status": "disconnected",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error disconnecting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voice/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Get status of a specific voice session."""
    try:
        if session_id not in manager.get_active_sessions():
            raise HTTPException(status_code=404, detail="Session not found")
        
        info = manager.get_connection_info(session_id)
        session_handler = voice_sessions.get(session_id)
        
        status = {
            "session_id": session_id,
            "connected": True,
            "connection_info": info,
            "timestamp": time.time()
        }
        
        if session_handler and session_handler.conversation_session:
            conversation_session = await conversation_state_manager.get_session(
                session_handler.conversation_session.session_id
            )
            if conversation_session:
                status["conversation"] = {
                    "session_id": conversation_session.session_id,
                    "state": conversation_session.state.value,
                    "total_turns": conversation_session.total_turns,
                    "active_flow": session_handler.active_flow
                }
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail=str(e))