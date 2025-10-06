"""
REST API endpoints for voice services.
Handles batch transcription, synthesis, and voice session management.
"""

import asyncio
import logging
import tempfile
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import base64

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel, Field, validator

from ai.voice import audio_processor, stt_service, tts_service
from ai.conversation import conversation_state_manager, dialog_flow_engine
from ai.decision_engine import intent_classifier
from backend.app.core.config import settings
from backend.app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# Request/Response Models
class TranscriptionRequest(BaseModel):
    """Request model for audio transcription."""
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    language: Optional[str] = Field("en", description="Language code for transcription")
    enable_diarization: bool = Field(False, description="Enable speaker diarization")
    model_size: Optional[str] = Field("base", description="Whisper model size to use")


class TranscriptionResponse(BaseModel):
    """Response model for transcription."""
    success: bool
    text: str
    confidence: float
    language: Optional[str] = None
    language_confidence: Optional[float] = None
    segments: Optional[List[Dict[str, Any]]] = None
    word_timestamps: Optional[List[Dict[str, Any]]] = None
    speaker_id: Optional[str] = None
    processing_time_ms: float
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class SynthesisRequest(BaseModel):
    """Request model for text-to-speech synthesis."""
    text: str = Field(..., description="Text to synthesize", max_length=5000)
    voice_id: Optional[str] = Field(None, description="Voice ID to use")
    engine: Optional[str] = Field(None, description="TTS engine to use")
    language: str = Field("en", description="Language code")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech rate")
    pitch: float = Field(1.0, ge=0.5, le=2.0, description="Pitch adjustment")
    volume: float = Field(1.0, ge=0.1, le=1.0, description="Volume level")
    emotion: Optional[str] = Field(None, description="Emotion style")
    style: Optional[str] = Field(None, description="Speaking style")
    use_ssml: bool = Field(False, description="Use SSML formatting")
    output_format: str = Field("wav", description="Audio output format")


class SynthesisResponse(BaseModel):
    """Response model for synthesis."""
    success: bool
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    audio_format: str
    sample_rate: int
    duration_ms: float
    voice_id: str
    engine: str
    processing_time_ms: float
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class VoiceSessionRequest(BaseModel):
    """Request model for creating voice session."""
    call_direction: str = Field("inbound", description="Call direction: inbound/outbound")
    phone_number: Optional[str] = Field(None, description="Caller phone number")
    caller_id: Optional[str] = Field(None, description="Caller ID")
    flow_id: str = Field("legal_consultation", description="Initial conversation flow")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @validator('call_direction')
    def validate_direction(cls, v):
        if v not in ['inbound', 'outbound']:
            raise ValueError('call_direction must be "inbound" or "outbound"')
        return v


class VoiceSessionResponse(BaseModel):
    """Response model for voice session."""
    success: bool
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    state: Optional[str] = None
    flow_id: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class IntentClassificationRequest(BaseModel):
    """Request model for intent classification."""
    text: str = Field(..., description="Text to classify", max_length=1000)
    context: Optional[Dict[str, Any]] = Field(None, description="Conversation context")
    use_bert: Optional[bool] = Field(None, description="Force BERT usage")


class IntentClassificationResponse(BaseModel):
    """Response model for intent classification."""
    success: bool
    primary_intent: Optional[Dict[str, Any]] = None
    secondary_intents: Optional[List[Dict[str, Any]]] = None
    confidence_scores: Optional[Dict[str, float]] = None
    processing_time_ms: float
    model_used: str
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class VoiceHealthResponse(BaseModel):
    """Response model for voice service health check."""
    healthy: bool
    services: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# Endpoints
@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio_file: Optional[UploadFile] = File(None),
    request: Optional[TranscriptionRequest] = None
):
    """
    Transcribe audio to text using Whisper STT.
    
    Can accept either an uploaded file or base64 encoded audio data.
    """
    try:
        # Get audio data
        audio_data = None
        
        if audio_file:
            # Read uploaded file
            audio_data = await audio_file.read()
            
        elif request and request.audio_data:
            # Decode base64 audio
            try:
                audio_data = base64.b64decode(request.audio_data)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid base64 audio data: {e}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Either audio_file or audio_data must be provided"
            )
        
        # Create temporary file for audio processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        try:
            # Create audio chunk
            from ai.voice.audio_processor import AudioChunk
            
            audio_chunk = AudioChunk(
                data=audio_data,
                timestamp=datetime.utcnow().timestamp(),
                is_speech=True,
                confidence=1.0
            )
            
            # Transcribe
            language = request.language if request else "en"
            enable_diarization = request.enable_diarization if request else False
            
            result = await stt_service.transcribe_audio_chunk(
                audio_chunk,
                language=language,
                enable_diarization=enable_diarization
            )
            
            return TranscriptionResponse(
                success=True,
                text=result.text,
                confidence=result.confidence,
                language=result.language,
                language_confidence=result.language_confidence,
                segments=result.segments,
                word_timestamps=result.word_timestamps,
                speaker_id=result.speaker_id,
                processing_time_ms=result.processing_time_ms
            )
            
        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        return TranscriptionResponse(
            success=False,
            text="",
            confidence=0.0,
            processing_time_ms=0.0
        )


@router.post("/synthesize", response_model=SynthesisResponse)
async def synthesize_speech(request: SynthesisRequest):
    """
    Synthesize speech from text using TTS.
    """
    try:
        # Map engine name to enum if provided
        engine = None
        if request.engine:
            from ai.voice.tts_service import TTSEngine
            engine_map = {
                "coqui": TTSEngine.COQUI_TTS,
                "elevenlabs": TTSEngine.ELEVENLABS,
                "azure": TTSEngine.AZURE,
                "system": TTSEngine.PYTTSX3
            }
            engine = engine_map.get(request.engine.lower())
        
        # Synthesize
        result = await tts_service.synthesize(
            text=request.text,
            voice_id=request.voice_id,
            engine=engine,
            language=request.language,
            speed=request.speed,
            pitch=request.pitch,
            volume=request.volume,
            emotion=request.emotion,
            style=request.style,
            use_ssml=request.use_ssml
        )
        
        # Encode audio as base64
        audio_b64 = base64.b64encode(result.audio_data).decode('utf-8')
        
        return SynthesisResponse(
            success=True,
            audio_data=audio_b64,
            audio_format=result.audio_format,
            sample_rate=result.sample_rate,
            duration_ms=result.duration_ms,
            voice_id=result.voice_id,
            engine=result.engine.value,
            processing_time_ms=result.processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error in synthesis: {e}")
        return SynthesisResponse(
            success=False,
            audio_data=None,
            audio_format="error",
            sample_rate=0,
            duration_ms=0.0,
            voice_id="",
            engine="error",
            processing_time_ms=0.0,
            error=str(e)
        )


@router.get("/voices")
async def get_available_voices():
    """Get list of available TTS voices."""
    try:
        voices = tts_service.get_available_voices()
        
        voices_data = []
        for voice in voices:
            voices_data.append({
                "id": voice.id,
                "name": voice.name,
                "language": voice.language,
                "gender": voice.gender.value,
                "engine": voice.engine.value,
                "sample_rate": voice.sample_rate,
                "metadata": voice.metadata
            })
        
        return {
            "voices": voices_data,
            "total_voices": len(voices_data),
            "engines": tts_service.get_available_engines(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions", response_model=VoiceSessionResponse)
async def create_voice_session(request: VoiceSessionRequest):
    """Create a new voice session for conversation."""
    try:
        # Create conversation session
        from ai.conversation.state_manager import CallDirection, ParticipantInfo
        
        direction = CallDirection.INBOUND if request.call_direction == "inbound" else CallDirection.OUTBOUND
        
        participant = ParticipantInfo(
            id=str(uuid.uuid4()),
            phone_number=request.phone_number,
            role="caller"
        )
        
        conversation_session = await conversation_state_manager.create_session(
            direction=direction,
            phone_number=request.phone_number,
            caller_id=request.caller_id,
            participant_info=participant,
            metadata=request.metadata
        )
        
        # Start conversation flow
        await dialog_flow_engine.start_flow(conversation_session.session_id, request.flow_id)
        
        return VoiceSessionResponse(
            success=True,
            session_id=conversation_session.session_id,
            conversation_id=conversation_session.call_id,
            state=conversation_session.state.value,
            flow_id=request.flow_id
        )
        
    except Exception as e:
        logger.error(f"Error creating voice session: {e}")
        return VoiceSessionResponse(
            success=False,
            error=str(e)
        )


@router.get("/sessions/{session_id}")
async def get_voice_session(session_id: str):
    """Get voice session details."""
    try:
        session = await conversation_state_manager.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get flow status
        flow_status = dialog_flow_engine.get_flow_status(session_id)
        
        return {
            "session_id": session.session_id,
            "call_id": session.call_id,
            "direction": session.direction.value,
            "state": session.state.value,
            "participants": [p.to_dict() for p in session.participants],
            "context": session.context.to_dict(),
            "total_turns": session.total_turns,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "flow_status": flow_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def end_voice_session(session_id: str, resolution_status: str = "completed"):
    """End a voice session."""
    try:
        success = await conversation_state_manager.end_session(
            session_id,
            resolution_status=resolution_status
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "status": "ended",
            "resolution": resolution_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str, limit: Optional[int] = None):
    """Get conversation history for a session."""
    try:
        history = await conversation_state_manager.get_session_history(session_id, limit)
        
        history_data = []
        for turn in history:
            history_data.append({
                "turn_id": turn.turn_id,
                "timestamp": turn.timestamp,
                "speaker_id": turn.speaker_id,
                "speaker_role": turn.speaker_role,
                "input_text": turn.input_text,
                "response_text": turn.response_text,
                "intent": turn.intent,
                "intent_confidence": turn.intent_confidence,
                "entities": turn.entities,
                "processing_time_ms": turn.processing_time_ms
            })
        
        return {
            "session_id": session_id,
            "history": history_data,
            "total_turns": len(history_data),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting session history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify-intent", response_model=IntentClassificationResponse)
async def classify_intent(request: IntentClassificationRequest):
    """Classify intent from text."""
    try:
        result = await intent_classifier.classify_intent(
            text=request.text,
            context=request.context,
            use_bert=request.use_bert
        )
        
        # Convert to serializable format
        primary_intent = {
            "name": result.primary_intent.name,
            "category": result.primary_intent.category.value,
            "confidence": result.primary_intent.confidence,
            "entities": result.primary_intent.entities,
            "context": result.primary_intent.context,
            "description": result.primary_intent.description
        }
        
        secondary_intents = []
        for intent in result.secondary_intents:
            secondary_intents.append({
                "name": intent.name,
                "category": intent.category.value,
                "confidence": intent.confidence,
                "entities": intent.entities,
                "context": intent.context,
                "description": intent.description
            })
        
        return IntentClassificationResponse(
            success=True,
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence_scores=result.confidence_scores,
            processing_time_ms=result.processing_time_ms,
            model_used=result.model_used
        )
        
    except Exception as e:
        logger.error(f"Error classifying intent: {e}")
        return IntentClassificationResponse(
            success=False,
            processing_time_ms=0.0,
            model_used="error",
            error=str(e)
        )


@router.get("/intents")
async def get_available_intents():
    """Get list of available intents."""
    try:
        intents = intent_classifier.get_available_intents()
        
        intent_data = []
        for intent_name in intents:
            description = intent_classifier.get_intent_description(intent_name)
            intent_data.append({
                "name": intent_name,
                "description": description
            })
        
        return {
            "intents": intent_data,
            "total_intents": len(intent_data),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting intents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flows")
async def get_available_flows():
    """Get list of available conversation flows."""
    try:
        flows = []
        for flow_id, flow in dialog_flow_engine.flows.items():
            flows.append({
                "id": flow.id,
                "name": flow.name,
                "description": flow.description,
                "start_node": flow.start_node,
                "total_nodes": len(flow.nodes),
                "metadata": flow.metadata
            })
        
        return {
            "flows": flows,
            "active_flows": dialog_flow_engine.get_active_flows(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting flows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_voice_stats():
    """Get voice service statistics."""
    try:
        # Get stats from all services
        audio_stats = audio_processor.get_stats()
        stt_stats = stt_service.get_stats()
        tts_stats = tts_service.get_stats()
        intent_stats = intent_classifier.get_stats()
        conversation_stats = await conversation_state_manager.get_statistics()
        
        return {
            "audio_processing": audio_stats,
            "speech_to_text": stt_stats,
            "text_to_speech": tts_stats,
            "intent_classification": intent_stats,
            "conversations": conversation_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=VoiceHealthResponse)
async def health_check():
    """Comprehensive health check for voice services."""
    try:
        # Check all services
        audio_health = await audio_processor.health_check()
        stt_health = await stt_service.health_check()
        tts_health = await tts_service.health_check()
        intent_health = await intent_classifier.health_check()
        conversation_health = await conversation_state_manager.health_check()
        dialog_health = await dialog_flow_engine.health_check()
        
        # Overall health
        all_services_healthy = all([
            audio_health.get("healthy", False),
            stt_health.get("healthy", False),
            tts_health.get("healthy", False),
            intent_health.get("healthy", False),
            conversation_health.get("healthy", False),
            dialog_health.get("healthy", False)
        ])
        
        return VoiceHealthResponse(
            healthy=all_services_healthy,
            services={
                "audio_processor": audio_health,
                "speech_to_text": stt_health,
                "text_to_speech": tts_health,
                "intent_classifier": intent_health,
                "conversation_manager": conversation_health,
                "dialog_flow": dialog_health
            }
        )
        
    except Exception as e:
        logger.error(f"Voice health check failed: {e}")
        return VoiceHealthResponse(
            healthy=False,
            services={"error": str(e)}
        )