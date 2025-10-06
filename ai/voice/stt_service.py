"""
Speech-to-Text Service using OpenAI Whisper for real-time transcription.
Supports streaming transcription, multiple languages, and speaker diarization.
"""

import asyncio
import logging
import whisper
import numpy as np
import torch
import io
import wave
import tempfile
from typing import Dict, List, Optional, AsyncGenerator, Any, Tuple
from datetime import datetime
import json
import time
from pathlib import Path
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from enum import Enum

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logging.warning("faster-whisper not available - using standard whisper")

try:
    import pyannote.audio
    from pyannote.audio import Pipeline
    SPEAKER_DIARIZATION_AVAILABLE = True
except ImportError:
    SPEAKER_DIARIZATION_AVAILABLE = False
    logging.warning("pyannote.audio not available - speaker diarization disabled")

from .audio_processor import AudioChunk, audio_processor
from app.core.config import settings

logger = logging.getLogger(__name__)


class WhisperModelSize(Enum):
    """Whisper model sizes."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription."""
    text: str
    confidence: float
    language: Optional[str] = None
    language_confidence: Optional[float] = None
    segments: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: float = 0.0
    timestamp: float = 0.0
    is_final: bool = True
    speaker_id: Optional[str] = None
    word_timestamps: Optional[List[Dict[str, Any]]] = None


@dataclass
class StreamingTranscriptionChunk:
    """Chunk of streaming transcription."""
    text: str
    confidence: float
    is_partial: bool
    chunk_index: int
    timestamp: float
    audio_duration_ms: float


class WhisperSTTEngine:
    """Whisper-based STT engine."""
    
    def __init__(self, model_size: WhisperModelSize = WhisperModelSize.BASE, device: str = "auto"):
        self.model_size = model_size
        self.device = self._get_device(device)
        self.model = None
        self.faster_model = None
        self.use_faster_whisper = FASTER_WHISPER_AVAILABLE
        self.sample_rate = 16000
        
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    async def initialize(self) -> bool:
        """Initialize the Whisper model."""
        try:
            logger.info(f"Loading Whisper model {self.model_size.value} on {self.device}")
            
            if self.use_faster_whisper:
                # Use faster-whisper for better performance
                self.faster_model = WhisperModel(
                    self.model_size.value,
                    device=self.device,
                    compute_type="float16" if self.device == "cuda" else "int8"
                )
                logger.info("Using faster-whisper for optimized performance")
            else:
                # Use standard whisper
                self.model = whisper.load_model(self.model_size.value, device=self.device)
                logger.info("Using standard whisper")
            
            # Test transcription
            test_audio = np.zeros(self.sample_rate, dtype=np.float32)  # 1 second of silence
            await self._transcribe_audio(test_audio)
            
            logger.info("Whisper STT engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper STT engine: {e}")
            return False
    
    async def _transcribe_audio(self, audio: np.ndarray, language: Optional[str] = None) -> TranscriptionResult:
        """Transcribe audio using Whisper."""
        start_time = time.time()
        
        try:
            if self.use_faster_whisper and self.faster_model:
                # Use faster-whisper
                segments, info = self.faster_model.transcribe(
                    audio,
                    language=language,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                # Convert segments to list
                segments_list = list(segments)
                
                # Extract text and confidence
                text = " ".join([segment.text.strip() for segment in segments_list])
                confidence = np.mean([segment.avg_logprob for segment in segments_list]) if segments_list else 0.0
                confidence = max(0.0, min(1.0, (confidence + 1.0) / 2.0))  # Normalize to 0-1
                
                # Build detailed segments
                detailed_segments = []
                word_timestamps = []
                
                for segment in segments_list:
                    seg_dict = {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "confidence": max(0.0, min(1.0, (segment.avg_logprob + 1.0) / 2.0))
                    }
                    detailed_segments.append(seg_dict)
                    
                    # Add word timestamps if available
                    if hasattr(segment, 'words') and segment.words:
                        for word in segment.words:
                            word_timestamps.append({
                                "word": word.word,
                                "start": word.start,
                                "end": word.end,
                                "confidence": max(0.0, min(1.0, (word.probability + 1.0) / 2.0)) if hasattr(word, 'probability') else confidence
                            })
                
                result = TranscriptionResult(
                    text=text,
                    confidence=confidence,
                    language=info.language if info else None,
                    language_confidence=info.language_probability if info else None,
                    segments=detailed_segments,
                    word_timestamps=word_timestamps,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    timestamp=time.time()
                )
                
            else:
                # Use standard whisper
                result_dict = self.model.transcribe(
                    audio,
                    language=language,
                    word_timestamps=True,
                    fp16=self.device == "cuda"
                )
                
                text = result_dict["text"].strip()
                
                # Calculate average confidence from segments
                segments = result_dict.get("segments", [])
                if segments:
                    # Whisper doesn't directly provide confidence, estimate from log probabilities
                    confidences = []
                    for segment in segments:
                        if "avg_logprob" in segment:
                            # Convert log probability to confidence score
                            conf = max(0.0, min(1.0, (segment["avg_logprob"] + 1.0) / 2.0))
                            confidences.append(conf)
                    confidence = np.mean(confidences) if confidences else 0.5
                else:
                    confidence = 0.5
                
                # Extract word timestamps
                word_timestamps = []
                for segment in segments:
                    if "words" in segment:
                        for word in segment["words"]:
                            word_timestamps.append({
                                "word": word.get("word", ""),
                                "start": word.get("start", 0.0),
                                "end": word.get("end", 0.0),
                                "confidence": confidence  # Use segment confidence
                            })
                
                result = TranscriptionResult(
                    text=text,
                    confidence=confidence,
                    language=result_dict.get("language"),
                    segments=segments,
                    word_timestamps=word_timestamps,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    timestamp=time.time()
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                timestamp=time.time()
            )
    
    async def transcribe_chunk(self, audio_chunk: AudioChunk, language: Optional[str] = None) -> TranscriptionResult:
        """Transcribe a single audio chunk."""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_chunk.data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32767.0
            
            # Ensure minimum length for transcription
            min_samples = self.sample_rate // 2  # 0.5 seconds
            if len(audio_float) < min_samples:
                # Pad with zeros
                padding = np.zeros(min_samples - len(audio_float))
                audio_float = np.concatenate([audio_float, padding])
            
            # Transcribe
            result = await self._transcribe_audio(audio_float, language)
            
            # Add chunk metadata
            if result.segments is None:
                result.segments = []
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio chunk: {e}")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                timestamp=time.time()
            )


class SpeakerDiarization:
    """Speaker diarization using pyannote.audio."""
    
    def __init__(self):
        self.enabled = SPEAKER_DIARIZATION_AVAILABLE
        self.pipeline = None
        
    async def initialize(self) -> bool:
        """Initialize speaker diarization pipeline."""
        if not self.enabled:
            logger.warning("Speaker diarization not available")
            return False
        
        try:
            # Load pre-trained pipeline
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1")
            logger.info("Speaker diarization initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize speaker diarization: {e}")
            self.enabled = False
            return False
    
    def diarize_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """Perform speaker diarization on audio file."""
        if not self.enabled or not self.pipeline:
            return {"speakers": [], "error": "Speaker diarization not available"}
        
        try:
            # Apply diarization
            diarization = self.pipeline(audio_file_path)
            
            # Convert to serializable format
            speakers = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end,
                    "duration": turn.duration
                })
            
            return {"speakers": speakers}
            
        except Exception as e:
            logger.error(f"Error in speaker diarization: {e}")
            return {"speakers": [], "error": str(e)}


class STTService:
    """Main Speech-to-Text service."""
    
    def __init__(self):
        self.whisper_engine = WhisperSTTEngine()
        self.diarization = SpeakerDiarization()
        self.initialized = False
        self.processing_queue = Queue()
        self.streaming_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.chunk_duration_ms = 1000  # Process 1-second chunks
        self.overlap_ms = 200  # 200ms overlap between chunks
        self.min_speech_confidence = 0.5  # Minimum VAD confidence for transcription
        
        # Statistics
        self.stats = {
            "transcriptions_completed": 0,
            "total_audio_duration_seconds": 0.0,
            "total_processing_time_ms": 0.0,
            "average_confidence": 0.0,
            "languages_detected": set()
        }
    
    async def initialize(self) -> bool:
        """Initialize the STT service."""
        try:
            logger.info("Initializing STT service...")
            
            # Initialize Whisper engine
            if not await self.whisper_engine.initialize():
                raise Exception("Failed to initialize Whisper engine")
            
            # Initialize speaker diarization (optional)
            await self.diarization.initialize()
            
            self.initialized = True
            logger.info("STT service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize STT service: {e}")
            return False
    
    async def transcribe_audio_chunk(
        self, 
        audio_chunk: AudioChunk,
        language: Optional[str] = None,
        enable_diarization: bool = False
    ) -> TranscriptionResult:
        """Transcribe a single audio chunk."""
        if not self.initialized:
            raise Exception("STT service not initialized")
        
        # Only process if there's speech detected
        if audio_chunk.confidence < self.min_speech_confidence:
            return TranscriptionResult(
                text="",
                confidence=0.0,
                timestamp=time.time()
            )
        
        try:
            # Transcribe using Whisper
            result = await self.whisper_engine.transcribe_chunk(audio_chunk, language)
            
            # Add speaker diarization if enabled
            if enable_diarization and result.text and self.diarization.enabled:
                # Save audio chunk to temporary file for diarization
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = temp_file.name
                    
                # Save audio data to file
                with wave.open(temp_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(16000)  # 16kHz
                    wav_file.writeframes(audio_chunk.data)
                
                try:
                    # Perform diarization
                    diarization_result = self.diarization.diarize_audio(temp_path)
                    if diarization_result["speakers"]:
                        # Use the first speaker for simplicity
                        result.speaker_id = diarization_result["speakers"][0]["speaker"]
                finally:
                    # Clean up temp file
                    Path(temp_path).unlink(missing_ok=True)
            
            # Update statistics
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio chunk: {e}")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                timestamp=time.time()
            )
    
    async def start_streaming_session(
        self,
        session_id: str,
        language: Optional[str] = None,
        enable_diarization: bool = False
    ) -> bool:
        """Start a streaming transcription session."""
        try:
            if session_id in self.streaming_sessions:
                logger.warning(f"Streaming session {session_id} already exists")
                return False
            
            session_data = {
                "language": language,
                "enable_diarization": enable_diarization,
                "audio_buffer": b"",
                "last_transcription": None,
                "partial_transcriptions": [],
                "start_time": time.time(),
                "chunk_count": 0
            }
            
            self.streaming_sessions[session_id] = session_data
            logger.info(f"Started streaming STT session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting streaming session: {e}")
            return False
    
    async def process_streaming_audio(
        self,
        session_id: str,
        audio_chunk: AudioChunk
    ) -> Optional[StreamingTranscriptionChunk]:
        """Process audio for streaming transcription."""
        if session_id not in self.streaming_sessions:
            logger.error(f"Streaming session {session_id} not found")
            return None
        
        session = self.streaming_sessions[session_id]
        
        try:
            # Only process speech chunks
            if audio_chunk.confidence < self.min_speech_confidence:
                return None
            
            # Add to buffer
            session["audio_buffer"] += audio_chunk.data
            session["chunk_count"] += 1
            
            # Check if we have enough audio to transcribe
            buffer_duration_ms = len(session["audio_buffer"]) / (16000 * 2) * 1000
            
            if buffer_duration_ms >= self.chunk_duration_ms:
                # Create audio chunk from buffer
                buffer_chunk = AudioChunk(
                    data=session["audio_buffer"],
                    timestamp=time.time(),
                    is_speech=True,
                    confidence=audio_chunk.confidence
                )
                
                # Transcribe
                result = await self.transcribe_audio_chunk(
                    buffer_chunk,
                    session["language"],
                    session["enable_diarization"]
                )
                
                # Create streaming chunk
                streaming_chunk = StreamingTranscriptionChunk(
                    text=result.text,
                    confidence=result.confidence,
                    is_partial=False,
                    chunk_index=session["chunk_count"],
                    timestamp=result.timestamp,
                    audio_duration_ms=buffer_duration_ms
                )
                
                # Update session
                session["last_transcription"] = result
                session["partial_transcriptions"].append(streaming_chunk)
                
                # Keep overlap for next chunk
                overlap_bytes = int(self.overlap_ms * 16000 * 2 / 1000)
                session["audio_buffer"] = session["audio_buffer"][-overlap_bytes:]
                
                return streaming_chunk
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing streaming audio: {e}")
            return None
    
    async def end_streaming_session(self, session_id: str) -> Optional[List[StreamingTranscriptionChunk]]:
        """End a streaming transcription session."""
        if session_id not in self.streaming_sessions:
            logger.error(f"Streaming session {session_id} not found")
            return None
        
        try:
            session = self.streaming_sessions[session_id]
            
            # Process any remaining audio in buffer
            if session["audio_buffer"]:
                buffer_chunk = AudioChunk(
                    data=session["audio_buffer"],
                    timestamp=time.time(),
                    is_speech=True,
                    confidence=1.0
                )
                
                result = await self.transcribe_audio_chunk(
                    buffer_chunk,
                    session["language"],
                    session["enable_diarization"]
                )
                
                if result.text:
                    final_chunk = StreamingTranscriptionChunk(
                        text=result.text,
                        confidence=result.confidence,
                        is_partial=False,
                        chunk_index=session["chunk_count"] + 1,
                        timestamp=result.timestamp,
                        audio_duration_ms=len(session["audio_buffer"]) / (16000 * 2) * 1000
                    )
                    session["partial_transcriptions"].append(final_chunk)
            
            # Get all transcriptions
            all_transcriptions = session["partial_transcriptions"].copy()
            
            # Clean up session
            del self.streaming_sessions[session_id]
            
            logger.info(f"Ended streaming STT session: {session_id}")
            return all_transcriptions
            
        except Exception as e:
            logger.error(f"Error ending streaming session: {e}")
            return None
    
    def _update_stats(self, result: TranscriptionResult) -> None:
        """Update processing statistics."""
        self.stats["transcriptions_completed"] += 1
        self.stats["total_processing_time_ms"] += result.processing_time_ms
        
        if result.confidence > 0:
            # Update average confidence
            total_confidence = self.stats["average_confidence"] * (self.stats["transcriptions_completed"] - 1)
            self.stats["average_confidence"] = (total_confidence + result.confidence) / self.stats["transcriptions_completed"]
        
        if result.language:
            self.stats["languages_detected"].add(result.language)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        stats["languages_detected"] = list(stats["languages_detected"])
        stats["active_streaming_sessions"] = len(self.streaming_sessions)
        
        if stats["transcriptions_completed"] > 0:
            stats["average_processing_time_ms"] = stats["total_processing_time_ms"] / stats["transcriptions_completed"]
        else:
            stats["average_processing_time_ms"] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            "transcriptions_completed": 0,
            "total_audio_duration_seconds": 0.0,
            "total_processing_time_ms": 0.0,
            "average_confidence": 0.0,
            "languages_detected": set()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of STT service."""
        try:
            return {
                "healthy": self.initialized,
                "whisper_model": self.whisper_engine.model_size.value,
                "device": self.whisper_engine.device,
                "faster_whisper_available": FASTER_WHISPER_AVAILABLE,
                "speaker_diarization_available": SPEAKER_DIARIZATION_AVAILABLE,
                "active_sessions": len(self.streaming_sessions),
                "stats": self.get_stats(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"STT health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global STT service instance
stt_service = STTService()