"""
Text-to-Speech Service with multiple TTS engine support.
Supports Coqui TTS, ElevenLabs, Azure Cognitive Services, and local fallbacks.
"""

import asyncio
import logging
import io
import wave
import tempfile
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import base64
import aiohttp
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available")

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    logging.warning("TorchAudio not available")

try:
    from TTS.api import TTS
    COQUI_TTS_AVAILABLE = TORCH_AVAILABLE  # Only if torch works
except ImportError:
    COQUI_TTS_AVAILABLE = False
    logging.warning("Coqui TTS not available")

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_TTS_AVAILABLE = True
except ImportError:
    AZURE_TTS_AVAILABLE = False
    logging.warning("Azure Cognitive Services Speech SDK not available")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logging.warning("pyttsx3 not available")

from app.core.config import settings

logger = logging.getLogger(__name__)


class TTSEngine(Enum):
    """Available TTS engines."""
    COQUI_TTS = "coqui_tts"
    ELEVENLABS = "elevenlabs"
    AZURE = "azure"
    PYTTSX3 = "pyttsx3"
    SYSTEM = "system"


class VoiceGender(Enum):
    """Voice gender options."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


@dataclass
class Voice:
    """Voice configuration."""
    id: str
    name: str
    language: str
    gender: VoiceGender
    engine: TTSEngine
    sample_rate: int = 22050
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TTSRequest:
    """TTS request configuration."""
    text: str
    voice_id: Optional[str] = None
    language: Optional[str] = "en"
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    emotion: Optional[str] = None
    style: Optional[str] = None
    use_ssml: bool = False


@dataclass
class TTSResult:
    """Result of TTS synthesis."""
    audio_data: bytes
    audio_format: str
    sample_rate: int
    duration_ms: float
    text: str
    voice_id: str
    engine: TTSEngine
    processing_time_ms: float
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


class CoquiTTSEngine:
    """Coqui TTS engine for local text-to-speech."""
    
    def __init__(self):
        self.available = COQUI_TTS_AVAILABLE
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.sample_rate = 22050
        
    async def initialize(self) -> bool:
        """Initialize Coqui TTS engine."""
        if not self.available:
            logger.warning("Coqui TTS not available")
            return False

        try:
            # Set environment variable to accept Coqui license non-interactively
            import os
            os.environ["COQUI_TOS_AGREED"] = "1"

            # Fix for PyTorch 2.6+ weights_only=True security changes
            # Add XTTS and other TTS config classes to safe globals
            try:
                from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
                torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig])
                logger.info("Added XTTS classes to PyTorch safe globals")
            except ImportError:
                logger.debug("Could not import XTTS config classes (not needed for basic models)")

            # Use simplest working Coqui model
            model_name = "tts_models/en/ljspeech/speedy-speech"
            self.model = TTS(model_name=model_name).to(self.device)
            self.models["en"] = self.model

            logger.info(f"Coqui TTS initialized with model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Coqui TTS: {e}")
            self.available = False
            return False
    
    def get_voices(self) -> List[Voice]:
        """Get available voices."""
        if not self.available:
            return []
        
        return [
            Voice(
                id="coqui_en_female",
                name="Coqui English Female",
                language="en",
                gender=VoiceGender.FEMALE,
                engine=TTSEngine.COQUI_TTS,
                sample_rate=self.sample_rate
            )
        ]
    
    async def synthesize(self, request: TTSRequest) -> TTSResult:
        """Synthesize speech using Coqui TTS."""
        if not self.available or not self.model:
            raise Exception("Coqui TTS not available")
        
        start_time = time.time()
        
        try:
            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Synthesize speech with Tacotron2 (no speaker conditioning needed)
            self.model.tts_to_file(
                text=request.text,
                file_path=temp_path
            )
            
            # Read audio data
            with open(temp_path, "rb") as f:
                audio_data = f.read()
            
            # Calculate duration
            with wave.open(temp_path, "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration_ms = (frames / rate) * 1000
            
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
            return TTSResult(
                audio_data=audio_data,
                audio_format="wav",
                sample_rate=self.sample_rate,
                duration_ms=duration_ms,
                text=request.text,
                voice_id=request.voice_id or "coqui_en_female",
                engine=TTSEngine.COQUI_TTS,
                processing_time_ms=(time.time() - start_time) * 1000,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Coqui TTS synthesis error: {e}")
            raise


class ElevenLabsTTSEngine:
    """ElevenLabs TTS engine for high-quality speech synthesis."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(settings, 'ELEVENLABS_API_KEY', None)
        self.available = bool(self.api_key)
        self.base_url = "https://api.elevenlabs.io/v1"
        self.voices_cache = {}
        self.cache_expiry = timedelta(hours=1)
        
    async def initialize(self) -> bool:
        """Initialize ElevenLabs TTS engine."""
        if not self.available:
            logger.warning("ElevenLabs API key not available")
            return False
        
        try:
            # Test API connection by fetching voices
            voices = await self.get_voices()
            if voices:
                logger.info(f"ElevenLabs TTS initialized with {len(voices)} voices")
                return True
            else:
                logger.error("No voices available from ElevenLabs")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize ElevenLabs TTS: {e}")
            self.available = False
            return False
    
    async def get_voices(self) -> List[Voice]:
        """Get available voices from ElevenLabs."""
        if not self.available:
            return []
        
        try:
            # Check cache
            if "voices" in self.voices_cache:
                cache_time, voices = self.voices_cache["voices"]
                if datetime.now() - cache_time < self.cache_expiry:
                    return voices
            
            headers = {"xi-api-key": self.api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/voices", headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        voices = []
                        
                        for voice_data in data.get("voices", []):
                            voice = Voice(
                                id=voice_data["voice_id"],
                                name=voice_data["name"],
                                language=voice_data.get("language", "en"),
                                gender=VoiceGender.NEUTRAL,  # ElevenLabs doesn't specify gender
                                engine=TTSEngine.ELEVENLABS,
                                sample_rate=22050,
                                metadata={
                                    "category": voice_data.get("category"),
                                    "description": voice_data.get("description"),
                                    "preview_url": voice_data.get("preview_url")
                                }
                            )
                            voices.append(voice)
                        
                        # Cache voices
                        self.voices_cache["voices"] = (datetime.now(), voices)
                        return voices
                    else:
                        logger.error(f"Failed to fetch ElevenLabs voices: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching ElevenLabs voices: {e}")
            return []
    
    async def synthesize(self, request: TTSRequest) -> TTSResult:
        """Synthesize speech using ElevenLabs API."""
        if not self.available:
            raise Exception("ElevenLabs API not available")
        
        start_time = time.time()
        
        try:
            headers = {
                "xi-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            # Use default voice if none specified
            voice_id = request.voice_id or "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
            
            # Prepare request data
            data = {
                "text": request.text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5,
                    "style": min(1.0, max(0.0, request.pitch)),  # Use pitch as style
                    "use_speaker_boost": True
                }
            }
            
            # Add emotion/style if specified
            if request.emotion or request.style:
                data["voice_settings"]["style"] = 0.8
            
            url = f"{self.base_url}/text-to-speech/{voice_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        audio_data = await response.read()
                        
                        # ElevenLabs returns MP3, convert to WAV if needed
                        # For now, keep as MP3
                        
                        # Estimate duration (rough calculation for MP3)
                        duration_ms = len(request.text) * 50  # Rough estimate: 50ms per character
                        
                        return TTSResult(
                            audio_data=audio_data,
                            audio_format="mp3",
                            sample_rate=22050,
                            duration_ms=duration_ms,
                            text=request.text,
                            voice_id=voice_id,
                            engine=TTSEngine.ELEVENLABS,
                            processing_time_ms=(time.time() - start_time) * 1000,
                            timestamp=time.time()
                        )
                    else:
                        error_text = await response.text()
                        raise Exception(f"ElevenLabs API error {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"ElevenLabs synthesis error: {e}")
            raise


class AzureTTSEngine:
    """Azure Cognitive Services TTS engine."""
    
    def __init__(self, api_key: Optional[str] = None, region: Optional[str] = None):
        self.api_key = api_key or getattr(settings, 'AZURE_SPEECH_KEY', None)
        self.region = region or getattr(settings, 'AZURE_SPEECH_REGION', 'eastus')
        self.available = AZURE_TTS_AVAILABLE and bool(self.api_key)
        self.speech_config = None
        
    async def initialize(self) -> bool:
        """Initialize Azure TTS engine."""
        if not self.available:
            logger.warning("Azure TTS not available (missing SDK or API key)")
            return False
        
        try:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.api_key,
                region=self.region
            )
            
            # Test synthesis
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
            result = synthesizer.speak_text_async("Test").get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info("Azure TTS initialized successfully")
                return True
            else:
                logger.error(f"Azure TTS test failed: {result.reason}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Azure TTS: {e}")
            self.available = False
            return False
    
    def get_voices(self) -> List[Voice]:
        """Get available Azure voices."""
        if not self.available:
            return []
        
        # Return a subset of popular Azure voices
        return [
            Voice(
                id="en-US-JennyNeural",
                name="Jenny (Neural)",
                language="en-US",
                gender=VoiceGender.FEMALE,
                engine=TTSEngine.AZURE,
                sample_rate=24000
            ),
            Voice(
                id="en-US-GuyNeural",
                name="Guy (Neural)",
                language="en-US",
                gender=VoiceGender.MALE,
                engine=TTSEngine.AZURE,
                sample_rate=24000
            ),
            Voice(
                id="en-US-AriaNeural",
                name="Aria (Neural)",
                language="en-US",
                gender=VoiceGender.FEMALE,
                engine=TTSEngine.AZURE,
                sample_rate=24000
            )
        ]
    
    async def synthesize(self, request: TTSRequest) -> TTSResult:
        """Synthesize speech using Azure TTS."""
        if not self.available or not self.speech_config:
            raise Exception("Azure TTS not available")
        
        start_time = time.time()
        
        try:
            # Configure voice
            voice_name = request.voice_id or "en-US-JennyNeural"
            self.speech_config.speech_synthesis_voice_name = voice_name
            
            # Create synthesizer
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=None
            )
            
            # Prepare text (with SSML if requested)
            if request.use_ssml:
                ssml_text = f"""
                <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
                    <voice name='{voice_name}'>
                        <prosody rate='{request.speed:.1f}' pitch='{request.pitch:+.0%}' volume='{request.volume:.1f}'>
                            {request.text}
                        </prosody>
                    </voice>
                </speak>
                """
                result = synthesizer.speak_ssml_async(ssml_text).get()
            else:
                result = synthesizer.speak_text_async(request.text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio_data = result.audio_data
                
                # Calculate duration
                duration_ms = len(audio_data) / (24000 * 2) * 1000  # Assume 24kHz, 16-bit
                
                return TTSResult(
                    audio_data=audio_data,
                    audio_format="wav",
                    sample_rate=24000,
                    duration_ms=duration_ms,
                    text=request.text,
                    voice_id=voice_name,
                    engine=TTSEngine.AZURE,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    timestamp=time.time()
                )
            else:
                raise Exception(f"Azure TTS synthesis failed: {result.reason}")
                
        except Exception as e:
            logger.error(f"Azure TTS synthesis error: {e}")
            raise


class SystemTTSEngine:
    """System/fallback TTS engine using pyttsx3."""
    
    def __init__(self):
        self.available = PYTTSX3_AVAILABLE
        self.engine = None
        
    async def initialize(self) -> bool:
        """Initialize system TTS engine."""
        if not self.available:
            logger.warning("pyttsx3 not available")
            return False
        
        try:
            self.engine = pyttsx3.init()
            
            # Test synthesis
            self.engine.say("Test")
            self.engine.runAndWait()
            
            logger.info("System TTS (pyttsx3) initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system TTS: {e}")
            self.available = False
            return False
    
    def get_voices(self) -> List[Voice]:
        """Get available system voices."""
        if not self.available or not self.engine:
            return []
        
        try:
            voices = self.engine.getProperty('voices')
            result = []
            
            for i, voice in enumerate(voices):
                # Determine gender from voice name/ID
                name = voice.name.lower()
                if any(word in name for word in ['female', 'woman', 'girl', 'zira', 'hazel']):
                    gender = VoiceGender.FEMALE
                elif any(word in name for word in ['male', 'man', 'boy', 'david', 'mark']):
                    gender = VoiceGender.MALE
                else:
                    gender = VoiceGender.NEUTRAL
                
                result.append(Voice(
                    id=voice.id,
                    name=voice.name,
                    language=voice.languages[0] if voice.languages else "en",
                    gender=gender,
                    engine=TTSEngine.PYTTSX3,
                    sample_rate=22050
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting system voices: {e}")
            return []
    
    async def synthesize(self, request: TTSRequest) -> TTSResult:
        """Synthesize speech using system TTS."""
        if not self.available or not self.engine:
            raise Exception("System TTS not available")
        
        start_time = time.time()
        
        try:
            # Configure voice settings
            if request.voice_id:
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    if voice.id == request.voice_id:
                        self.engine.setProperty('voice', voice.id)
                        break
            
            # Set rate and volume
            self.engine.setProperty('rate', int(200 * request.speed))
            self.engine.setProperty('volume', request.volume)
            
            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Synthesize to file
            self.engine.save_to_file(request.text, temp_path)
            self.engine.runAndWait()
            
            # Read audio data
            with open(temp_path, "rb") as f:
                audio_data = f.read()
            
            # Calculate duration
            duration_ms = len(request.text) * 60  # Rough estimate
            
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
            return TTSResult(
                audio_data=audio_data,
                audio_format="wav",
                sample_rate=22050,
                duration_ms=duration_ms,
                text=request.text,
                voice_id=request.voice_id or "system_default",
                engine=TTSEngine.PYTTSX3,
                processing_time_ms=(time.time() - start_time) * 1000,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"System TTS synthesis error: {e}")
            raise


class TTSService:
    """Main Text-to-Speech service with multiple engine support."""
    
    def __init__(self):
        self.engines = {
            TTSEngine.COQUI_TTS: CoquiTTSEngine(),
            TTSEngine.ELEVENLABS: ElevenLabsTTSEngine(),
            TTSEngine.AZURE: AzureTTSEngine(),
            TTSEngine.PYTTSX3: SystemTTSEngine()
        }
        
        self.preferred_engine_order = [
            TTSEngine.ELEVENLABS,
            TTSEngine.AZURE,
            TTSEngine.COQUI_TTS,
            TTSEngine.PYTTSX3
        ]
        
        self.initialized_engines = set()
        self.audio_cache = {}
        self.cache_max_size = 100
        self.cache_expiry = timedelta(hours=1)
        
        # Statistics
        self.stats = {
            "syntheses_completed": 0,
            "total_characters_synthesized": 0,
            "total_processing_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "engine_usage": {engine.value: 0 for engine in TTSEngine}
        }
    
    async def initialize(self) -> bool:
        """Initialize TTS service and available engines."""
        try:
            logger.info("Initializing TTS service...")
            
            # Initialize engines in order of preference
            for engine_type in self.preferred_engine_order:
                engine = self.engines[engine_type]
                try:
                    if await engine.initialize():
                        self.initialized_engines.add(engine_type)
                        logger.info(f"Initialized {engine_type.value} TTS engine")
                except Exception as e:
                    logger.warning(f"Failed to initialize {engine_type.value}: {e}")
            
            if not self.initialized_engines:
                logger.error("No TTS engines available")
                return False
            
            logger.info(f"TTS service initialized with {len(self.initialized_engines)} engines")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS service: {e}")
            return False
    
    def get_available_engines(self) -> List[TTSEngine]:
        """Get list of available TTS engines."""
        return list(self.initialized_engines)
    
    def get_available_voices(self, engine: Optional[TTSEngine] = None) -> List[Voice]:
        """Get available voices from all or specific engine."""
        voices = []
        
        if engine:
            if engine in self.initialized_engines:
                engine_obj = self.engines[engine]
                voices.extend(engine_obj.get_voices())
        else:
            # Get voices from all initialized engines
            for engine_type in self.initialized_engines:
                engine_obj = self.engines[engine_type]
                voices.extend(engine_obj.get_voices())
        
        return voices
    
    def _get_cache_key(self, request: TTSRequest) -> str:
        """Generate cache key for TTS request."""
        key_data = {
            "text": request.text,
            "voice_id": request.voice_id,
            "language": request.language,
            "speed": round(request.speed, 2),
            "pitch": round(request.pitch, 2),
            "volume": round(request.volume, 2),
            "emotion": request.emotion,
            "style": request.style,
            "use_ssml": request.use_ssml
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[TTSResult]:
        """Get cached TTS result if available and not expired."""
        if cache_key in self.audio_cache:
            cache_time, result = self.audio_cache[cache_key]
            if datetime.now() - cache_time < self.cache_expiry:
                self.stats["cache_hits"] += 1
                return result
            else:
                # Remove expired entry
                del self.audio_cache[cache_key]
        
        self.stats["cache_misses"] += 1
        return None
    
    def _cache_result(self, cache_key: str, result: TTSResult) -> None:
        """Cache TTS result."""
        # Clean old entries if cache is full
        if len(self.audio_cache) >= self.cache_max_size:
            # Remove oldest entries
            sorted_cache = sorted(
                self.audio_cache.items(),
                key=lambda x: x[1][0]  # Sort by timestamp
            )
            for old_key, _ in sorted_cache[:self.cache_max_size // 4]:
                del self.audio_cache[old_key]
        
        self.audio_cache[cache_key] = (datetime.now(), result)
    
    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        engine: Optional[TTSEngine] = None,
        language: Optional[str] = "en",
        speed: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
        emotion: Optional[str] = None,
        style: Optional[str] = None,
        use_ssml: bool = False,
        use_cache: bool = True
    ) -> TTSResult:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice_id: Specific voice ID to use
            engine: Preferred TTS engine
            language: Language code
            speed: Speech rate (0.5 - 2.0)
            pitch: Pitch adjustment (-1.0 - 1.0)
            volume: Volume level (0.0 - 1.0)
            emotion: Emotion style (if supported)
            style: Speaking style (if supported)
            use_ssml: Whether to use SSML formatting
            use_cache: Whether to use audio caching
        
        Returns:
            TTSResult: Synthesized audio result
        """
        # Create request object
        request = TTSRequest(
            text=text,
            voice_id=voice_id,
            language=language,
            speed=speed,
            pitch=pitch,
            volume=volume,
            emotion=emotion,
            style=style,
            use_ssml=use_ssml
        )
        
        # Check cache first
        cache_key = self._get_cache_key(request)
        if use_cache:
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.debug("Returning cached TTS result")
                return cached_result
        
        # Determine which engine to use
        if engine and engine in self.initialized_engines:
            engines_to_try = [engine]
        else:
            # Try engines in order of preference
            engines_to_try = [e for e in self.preferred_engine_order if e in self.initialized_engines]
        
        # Try synthesizing with available engines
        last_error = None
        for engine_type in engines_to_try:
            try:
                engine_obj = self.engines[engine_type]
                result = await engine_obj.synthesize(request)
                
                # Update statistics
                self._update_stats(result)
                
                # Cache result
                if use_cache:
                    self._cache_result(cache_key, result)
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"TTS synthesis failed with {engine_type.value}: {e}")
                continue
        
        # All engines failed
        raise Exception(f"All TTS engines failed. Last error: {last_error}")
    
    async def synthesize_streaming(
        self,
        text_stream: AsyncGenerator[str, None],
        voice_id: Optional[str] = None,
        engine: Optional[TTSEngine] = None,
        **kwargs
    ) -> AsyncGenerator[TTSResult, None]:
        """
        Synthesize speech from streaming text.
        
        Args:
            text_stream: Async generator yielding text chunks
            voice_id: Voice ID to use
            engine: Preferred TTS engine
            **kwargs: Additional synthesis parameters
        
        Yields:
            TTSResult: Synthesized audio chunks
        """
        try:
            async for text_chunk in text_stream:
                if text_chunk.strip():  # Only process non-empty chunks
                    result = await self.synthesize(
                        text=text_chunk,
                        voice_id=voice_id,
                        engine=engine,
                        **kwargs
                    )
                    yield result
                    
        except Exception as e:
            logger.error(f"Error in streaming TTS synthesis: {e}")
            raise
    
    def _update_stats(self, result: TTSResult) -> None:
        """Update synthesis statistics."""
        self.stats["syntheses_completed"] += 1
        self.stats["total_characters_synthesized"] += len(result.text)
        self.stats["total_processing_time_ms"] += result.processing_time_ms
        self.stats["engine_usage"][result.engine.value] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get synthesis statistics."""
        stats = self.stats.copy()
        
        if stats["syntheses_completed"] > 0:
            stats["average_processing_time_ms"] = stats["total_processing_time_ms"] / stats["syntheses_completed"]
            stats["average_characters_per_synthesis"] = stats["total_characters_synthesized"] / stats["syntheses_completed"]
        else:
            stats["average_processing_time_ms"] = 0.0
            stats["average_characters_per_synthesis"] = 0.0
        
        # Cache statistics
        stats["cache_size"] = len(self.audio_cache)
        total_cache_requests = stats["cache_hits"] + stats["cache_misses"]
        if total_cache_requests > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / total_cache_requests
        else:
            stats["cache_hit_rate"] = 0.0
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear audio cache."""
        self.audio_cache.clear()
        logger.info("TTS audio cache cleared")
    
    def reset_stats(self) -> None:
        """Reset synthesis statistics."""
        self.stats = {
            "syntheses_completed": 0,
            "total_characters_synthesized": 0,
            "total_processing_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "engine_usage": {engine.value: 0 for engine in TTSEngine}
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of TTS service."""
        try:
            engine_status = {}
            
            # Check each initialized engine
            for engine_type in self.initialized_engines:
                try:
                    # Test synthesis with each engine
                    test_result = await self.synthesize(
                        text="Hello, this is a test.",
                        engine=engine_type,
                        use_cache=False
                    )
                    engine_status[engine_type.value] = {
                        "healthy": True,
                        "test_synthesis_time_ms": test_result.processing_time_ms
                    }
                except Exception as e:
                    engine_status[engine_type.value] = {
                        "healthy": False,
                        "error": str(e)
                    }
            
            # Overall health
            healthy_engines = [status["healthy"] for status in engine_status.values()]
            overall_healthy = any(healthy_engines)
            
            return {
                "healthy": overall_healthy,
                "initialized_engines": [e.value for e in self.initialized_engines],
                "engine_status": engine_status,
                "total_voices": len(self.get_available_voices()),
                "stats": self.get_stats(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"TTS health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global TTS service instance
tts_service = TTSService()