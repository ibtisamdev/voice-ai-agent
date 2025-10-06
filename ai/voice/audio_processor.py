"""
Audio Processing Service for real-time voice processing.
Handles WebSocket audio streams, voice activity detection, and noise reduction.
"""

import asyncio
import logging
import numpy as np
import webrtcvad
import audioop
import wave
import io
from typing import Dict, List, Optional, AsyncGenerator, Callable, Any
from datetime import datetime
import json
import time
from pathlib import Path
import tempfile
import base64

try:
    import noisereduce as nr
    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    NOISE_REDUCE_AVAILABLE = False
    logging.warning("noisereduce not available - noise reduction will be disabled")

try:
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available - advanced audio processing will be limited")

from app.core.config import settings

logger = logging.getLogger(__name__)


class AudioFormat:
    """Audio format specifications."""
    def __init__(self, sample_rate: int = 16000, channels: int = 1, bit_depth: int = 16):
        self.sample_rate = sample_rate
        self.channels = channels
        self.bit_depth = bit_depth
        self.frame_duration_ms = 30  # WebRTC VAD compatible
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        self.bytes_per_sample = bit_depth // 8


class AudioChunk:
    """Represents a processed audio chunk."""
    def __init__(self, data: bytes, timestamp: float, is_speech: bool = False, 
                 confidence: float = 0.0, metadata: Optional[Dict[str, Any]] = None):
        self.data = data
        self.timestamp = timestamp
        self.is_speech = is_speech
        self.confidence = confidence
        self.metadata = metadata or {}
        self.duration_ms = len(data) / (16000 * 2) * 1000  # Assume 16kHz, 16-bit


class VoiceActivityDetector:
    """Voice Activity Detection using WebRTC VAD."""
    
    def __init__(self, aggressiveness: int = 2):
        """
        Initialize VAD.
        
        Args:
            aggressiveness: VAD aggressiveness (0-3, higher = more aggressive)
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_duration_ms = 30  # Supported: 10, 20, 30ms
        self.sample_rate = 16000  # WebRTC VAD requires 16kHz
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        self.speech_frames = []
        self.speech_threshold = 0.5  # Ratio of speech frames to consider as speech
        
    def is_speech(self, frame: bytes) -> bool:
        """Check if audio frame contains speech."""
        try:
            if len(frame) != self.frame_size * 2:  # 16-bit = 2 bytes per sample
                return False
            return self.vad.is_speech(frame, self.sample_rate)
        except Exception as e:
            logger.warning(f"VAD error: {e}")
            return False
    
    def process_chunk(self, audio_data: bytes) -> float:
        """
        Process audio chunk and return speech confidence.
        
        Returns:
            float: Speech confidence (0.0 - 1.0)
        """
        try:
            # Split into frames
            frame_length = self.frame_size * 2  # 16-bit = 2 bytes per sample
            frames = []
            
            for i in range(0, len(audio_data), frame_length):
                frame = audio_data[i:i + frame_length]
                if len(frame) == frame_length:
                    frames.append(frame)
            
            if not frames:
                return 0.0
            
            # Check speech in each frame
            speech_frames = sum(1 for frame in frames if self.is_speech(frame))
            confidence = speech_frames / len(frames)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error processing VAD chunk: {e}")
            return 0.0


class NoiseReducer:
    """Noise reduction for audio streams."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and NOISE_REDUCE_AVAILABLE
        self.sample_rate = 16000
        self.noise_profile = None
        self.adaptation_rate = 0.1
        
        if not self.enabled:
            logger.warning("Noise reduction disabled - missing dependencies")
    
    def reduce_noise(self, audio_data: bytes, learn_noise: bool = False) -> bytes:
        """
        Reduce noise in audio data.
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            learn_noise: Whether to learn noise profile from this audio
            
        Returns:
            bytes: Noise-reduced audio data
        """
        if not self.enabled:
            return audio_data
        
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float for processing
            audio_float = audio_array.astype(np.float32) / 32767.0
            
            # Apply noise reduction
            if learn_noise:
                # Use this audio to learn noise characteristics
                reduced_audio = nr.reduce_noise(
                    y=audio_float, 
                    sr=self.sample_rate,
                    stationary=False
                )
            else:
                # Apply noise reduction
                reduced_audio = nr.reduce_noise(
                    y=audio_float, 
                    sr=self.sample_rate,
                    stationary=True
                )
            
            # Convert back to int16
            reduced_int16 = (reduced_audio * 32767.0).astype(np.int16)
            
            return reduced_int16.tobytes()
            
        except Exception as e:
            logger.error(f"Noise reduction error: {e}")
            return audio_data
    
    def apply_high_pass_filter(self, audio_data: bytes, cutoff_freq: int = 300) -> bytes:
        """Apply high-pass filter to remove low-frequency noise."""
        if not SCIPY_AVAILABLE:
            return audio_data
        
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            # Design high-pass filter
            nyquist = self.sample_rate / 2
            normal_cutoff = cutoff_freq / nyquist
            b, a = scipy.signal.butter(4, normal_cutoff, btype='high', analog=False)
            
            # Apply filter
            filtered = scipy.signal.filtfilt(b, a, audio_array)
            
            # Convert back to int16
            filtered_int16 = np.clip(filtered, -32767, 32767).astype(np.int16)
            
            return filtered_int16.tobytes()
            
        except Exception as e:
            logger.error(f"High-pass filter error: {e}")
            return audio_data


class AudioProcessor:
    """Main audio processing service for voice AI agent."""
    
    def __init__(self):
        self.audio_format = AudioFormat()
        self.vad = VoiceActivityDetector(aggressiveness=2)
        self.noise_reducer = NoiseReducer(enabled=True)
        self.processing_enabled = True
        self.chunk_handlers: List[Callable] = []
        self.stats = {
            "chunks_processed": 0,
            "speech_chunks": 0,
            "total_audio_seconds": 0.0,
            "processing_time_ms": 0.0
        }
        
    async def initialize(self) -> bool:
        """Initialize the audio processor."""
        try:
            logger.info("Initializing audio processor...")
            
            # Test VAD
            test_frame = b'\x00' * (self.audio_format.frame_size * 2)
            self.vad.is_speech(test_frame)
            
            logger.info("Audio processor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize audio processor: {e}")
            return False
    
    def add_chunk_handler(self, handler: Callable[[AudioChunk], None]) -> None:
        """Add a handler for processed audio chunks."""
        self.chunk_handlers.append(handler)
    
    def remove_chunk_handler(self, handler: Callable[[AudioChunk], None]) -> None:
        """Remove a chunk handler."""
        if handler in self.chunk_handlers:
            self.chunk_handlers.remove(handler)
    
    async def process_audio_stream(
        self, 
        audio_stream: AsyncGenerator[bytes, None],
        session_id: str,
        apply_noise_reduction: bool = True,
        apply_vad: bool = True
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        Process streaming audio data.
        
        Args:
            audio_stream: Async generator yielding audio bytes
            session_id: Unique session identifier
            apply_noise_reduction: Whether to apply noise reduction
            apply_vad: Whether to apply voice activity detection
        
        Yields:
            AudioChunk: Processed audio chunks
        """
        try:
            logger.info(f"Starting audio stream processing for session {session_id}")
            
            chunk_count = 0
            async for raw_audio in audio_stream:
                start_time = time.time()
                
                # Convert audio format if needed
                processed_audio = self._convert_audio_format(raw_audio)
                
                # Apply noise reduction
                if apply_noise_reduction:
                    processed_audio = self.noise_reducer.reduce_noise(processed_audio)
                    processed_audio = self.noise_reducer.apply_high_pass_filter(processed_audio)
                
                # Apply voice activity detection
                is_speech = False
                speech_confidence = 0.0
                if apply_vad:
                    speech_confidence = self.vad.process_chunk(processed_audio)
                    is_speech = speech_confidence > self.vad.speech_threshold
                
                # Create audio chunk
                timestamp = time.time()
                chunk = AudioChunk(
                    data=processed_audio,
                    timestamp=timestamp,
                    is_speech=is_speech,
                    confidence=speech_confidence,
                    metadata={
                        "session_id": session_id,
                        "chunk_index": chunk_count,
                        "processing_time_ms": (time.time() - start_time) * 1000,
                        "noise_reduction_applied": apply_noise_reduction,
                        "vad_applied": apply_vad
                    }
                )
                
                # Update statistics
                self._update_stats(chunk, time.time() - start_time)
                
                # Notify handlers
                for handler in self.chunk_handlers:
                    try:
                        handler(chunk)
                    except Exception as e:
                        logger.error(f"Error in chunk handler: {e}")
                
                chunk_count += 1
                yield chunk
                
        except Exception as e:
            logger.error(f"Error processing audio stream: {e}")
            raise
    
    def process_audio_chunk(
        self, 
        audio_data: bytes,
        apply_noise_reduction: bool = True,
        apply_vad: bool = True
    ) -> AudioChunk:
        """
        Process a single audio chunk.
        
        Args:
            audio_data: Raw audio bytes
            apply_noise_reduction: Whether to apply noise reduction
            apply_vad: Whether to apply voice activity detection
        
        Returns:
            AudioChunk: Processed audio chunk
        """
        start_time = time.time()
        
        try:
            # Convert audio format if needed
            processed_audio = self._convert_audio_format(audio_data)
            
            # Apply noise reduction
            if apply_noise_reduction:
                processed_audio = self.noise_reducer.reduce_noise(processed_audio)
                processed_audio = self.noise_reducer.apply_high_pass_filter(processed_audio)
            
            # Apply voice activity detection
            is_speech = False
            speech_confidence = 0.0
            if apply_vad:
                speech_confidence = self.vad.process_chunk(processed_audio)
                is_speech = speech_confidence > self.vad.speech_threshold
            
            # Create audio chunk
            timestamp = time.time()
            chunk = AudioChunk(
                data=processed_audio,
                timestamp=timestamp,
                is_speech=is_speech,
                confidence=speech_confidence,
                metadata={
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "noise_reduction_applied": apply_noise_reduction,
                    "vad_applied": apply_vad
                }
            )
            
            # Update statistics
            self._update_stats(chunk, time.time() - start_time)
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            # Return unprocessed chunk on error
            return AudioChunk(
                data=audio_data,
                timestamp=time.time(),
                is_speech=False,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def _convert_audio_format(self, audio_data: bytes) -> bytes:
        """Convert audio to the standard format (16kHz, 16-bit, mono)."""
        # For now, assume input is already in correct format
        # In production, you'd implement proper audio conversion here
        return audio_data
    
    def _update_stats(self, chunk: AudioChunk, processing_time: float) -> None:
        """Update processing statistics."""
        self.stats["chunks_processed"] += 1
        if chunk.is_speech:
            self.stats["speech_chunks"] += 1
        self.stats["total_audio_seconds"] += chunk.duration_ms / 1000.0
        self.stats["processing_time_ms"] += processing_time * 1000
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        if stats["chunks_processed"] > 0:
            stats["speech_ratio"] = stats["speech_chunks"] / stats["chunks_processed"]
            stats["avg_processing_time_ms"] = stats["processing_time_ms"] / stats["chunks_processed"]
        else:
            stats["speech_ratio"] = 0.0
            stats["avg_processing_time_ms"] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            "chunks_processed": 0,
            "speech_chunks": 0,
            "total_audio_seconds": 0.0,
            "processing_time_ms": 0.0
        }
    
    def save_audio_chunk(self, chunk: AudioChunk, filepath: str) -> bool:
        """Save an audio chunk to file for debugging."""
        try:
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(self.audio_format.channels)
                wav_file.setsampwidth(self.audio_format.bytes_per_sample)
                wav_file.setframerate(self.audio_format.sample_rate)
                wav_file.writeframes(chunk.data)
            
            logger.info(f"Audio chunk saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving audio chunk: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of audio processor."""
        try:
            # Test VAD
            test_frame = b'\x00' * (self.audio_format.frame_size * 2)
            vad_working = self.vad.is_speech(test_frame) is not None
            
            # Test noise reduction
            noise_reduction_available = NOISE_REDUCE_AVAILABLE
            
            return {
                "healthy": True,
                "vad_working": vad_working,
                "noise_reduction_available": noise_reduction_available,
                "scipy_available": SCIPY_AVAILABLE,
                "processing_enabled": self.processing_enabled,
                "audio_format": {
                    "sample_rate": self.audio_format.sample_rate,
                    "channels": self.audio_format.channels,
                    "bit_depth": self.audio_format.bit_depth
                },
                "stats": self.get_stats(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Audio processor health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global audio processor instance
audio_processor = AudioProcessor()