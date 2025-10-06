"""Tests for Speech-to-Text (STT) voice service."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import asyncio
from pathlib import Path

from ai.voice.stt_service import STTService, WhisperSTTEngine
from ai.voice.audio_processor import AudioProcessor


class TestWhisperSTTEngine:
    """Test Whisper STT engine."""
    
    @pytest.fixture
    def mock_whisper_model(self):
        """Mock Whisper model."""
        model = Mock()
        model.transcribe.return_value = {
            'text': 'Hello, this is a test transcription.',
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.5,
                    'text': 'Hello, this is a test transcription.',
                    'words': [
                        {'start': 0.0, 'end': 0.5, 'word': 'Hello'},
                        {'start': 0.6, 'end': 0.8, 'word': 'this'},
                        {'start': 0.9, 'end': 1.0, 'word': 'is'},
                        {'start': 1.1, 'end': 1.2, 'word': 'a'},
                        {'start': 1.3, 'end': 1.6, 'word': 'test'},
                        {'start': 1.7, 'end': 2.5, 'word': 'transcription'}
                    ]
                }
            ],
            'language': 'en'
        }
        return model
    
    @pytest.fixture
    def stt_engine(self, mock_whisper_model):
        """Create STT engine with mocked Whisper model."""
        with patch('ai.voice.stt_service.whisper.load_model', return_value=mock_whisper_model):
            engine = WhisperSTTEngine(model_size='base')
            return engine
    
    def test_engine_initialization(self, stt_engine):
        """Test STT engine initialization."""
        assert stt_engine.model_size == 'base'
        assert stt_engine.model is not None
        assert stt_engine.sample_rate == 16000
    
    def test_transcribe_audio_file(self, stt_engine, tmp_path):
        """Test audio file transcription."""
        # Create dummy audio file path
        audio_file = tmp_path / "test_audio.wav"
        audio_file.write_bytes(b"dummy audio data")
        
        result = stt_engine.transcribe_file(str(audio_file))
        
        assert result['text'] == 'Hello, this is a test transcription.'
        assert result['language'] == 'en'
        assert len(result['segments']) == 1
        assert result['segments'][0]['start'] == 0.0
        assert result['segments'][0]['end'] == 2.5
    
    def test_transcribe_audio_array(self, stt_engine):
        """Test audio array transcription."""
        # Create dummy audio array
        audio_array = np.random.random(16000).astype(np.float32)  # 1 second of audio
        
        result = stt_engine.transcribe_array(audio_array)
        
        assert result['text'] == 'Hello, this is a test transcription.'
        assert result['language'] == 'en'
        assert 'segments' in result
    
    def test_streaming_transcription(self, stt_engine):
        """Test streaming transcription session."""
        session_id = "test_session_123"
        
        # Start streaming session
        session = stt_engine.start_streaming_session(session_id)
        assert session['session_id'] == session_id
        assert session['status'] == 'active'
        
        # Add audio chunk
        audio_chunk = np.random.random(1600).astype(np.float32)  # 0.1 second
        stt_engine.add_audio_chunk(session_id, audio_chunk)
        
        # Get partial result
        result = stt_engine.get_partial_result(session_id)
        assert 'partial_text' in result
        
        # End session
        final_result = stt_engine.end_streaming_session(session_id)
        assert final_result['text'] == 'Hello, this is a test transcription.'


class TestSTTService:
    """Test STT service."""
    
    @pytest.fixture
    def mock_engine(self):
        """Mock STT engine."""
        engine = Mock()
        engine.transcribe_file.return_value = {
            'text': 'Test transcription',
            'language': 'en',
            'confidence': 0.95,
            'segments': []
        }
        engine.transcribe_array.return_value = {
            'text': 'Test transcription',
            'language': 'en',
            'confidence': 0.95,
            'segments': []
        }
        return engine
    
    @pytest.fixture
    def stt_service(self, mock_engine):
        """Create STT service with mocked engine."""
        with patch('ai.voice.stt_service.WhisperSTTEngine', return_value=mock_engine):
            service = STTService()
            return service
    
    def test_service_initialization(self, stt_service):
        """Test STT service initialization."""
        assert stt_service.primary_engine is not None
        assert hasattr(stt_service, 'active_sessions')
    
    @pytest.mark.asyncio
    async def test_transcribe_file_async(self, stt_service, tmp_path):
        """Test async file transcription."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"dummy audio")
        
        result = await stt_service.transcribe_file_async(str(audio_file))
        
        assert result['text'] == 'Test transcription'
        assert result['language'] == 'en'
        assert result['confidence'] == 0.95
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_async(self, stt_service):
        """Test async audio array transcription."""
        audio_data = np.random.random(16000).astype(np.float32)
        
        result = await stt_service.transcribe_audio_async(audio_data)
        
        assert result['text'] == 'Test transcription'
        assert result['language'] == 'en'
    
    def test_create_session(self, stt_service):
        """Test session creation."""
        session_id = "test_session_456"
        
        session = stt_service.create_session(session_id)
        
        assert session['session_id'] == session_id
        assert session_id in stt_service.active_sessions
    
    def test_session_management(self, stt_service):
        """Test session lifecycle management."""
        session_id = "test_session_789"
        
        # Create session
        stt_service.create_session(session_id)
        assert stt_service.is_session_active(session_id)
        
        # End session
        stt_service.end_session(session_id)
        assert not stt_service.is_session_active(session_id)
    
    def test_audio_chunk_processing(self, stt_service):
        """Test audio chunk processing in streaming mode."""
        session_id = "streaming_test"
        stt_service.create_session(session_id)
        
        # Process audio chunk
        audio_chunk = np.random.random(1600).astype(np.float32)
        result = stt_service.process_audio_chunk(session_id, audio_chunk)
        
        assert 'status' in result
        assert result['session_id'] == session_id


class TestAudioProcessor:
    """Test audio processing utilities."""
    
    @pytest.fixture
    def audio_processor(self):
        """Create audio processor."""
        return AudioProcessor()
    
    def test_processor_initialization(self, audio_processor):
        """Test audio processor initialization."""
        assert audio_processor.sample_rate == 16000
        assert audio_processor.vad is not None
    
    def test_audio_validation(self, audio_processor):
        """Test audio format validation."""
        # Valid audio
        valid_audio = np.random.random(16000).astype(np.float32)
        assert audio_processor.validate_audio(valid_audio)
        
        # Invalid audio (wrong type)
        invalid_audio = np.random.random(16000).astype(np.int32)
        assert not audio_processor.validate_audio(invalid_audio)
    
    def test_noise_reduction(self, audio_processor):
        """Test noise reduction processing."""
        # Create noisy audio
        clean_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))  # 440Hz tone
        noise = np.random.random(16000) * 0.1  # Low level noise
        noisy_audio = (clean_signal + noise).astype(np.float32)
        
        # Apply noise reduction
        cleaned_audio = audio_processor.reduce_noise(noisy_audio)
        
        assert cleaned_audio.shape == noisy_audio.shape
        assert cleaned_audio.dtype == np.float32
    
    def test_voice_activity_detection(self, audio_processor):
        """Test voice activity detection."""
        # Silent audio
        silent_audio = np.zeros(16000, dtype=np.float32)
        assert not audio_processor.detect_voice_activity(silent_audio)
        
        # Audio with speech-like content
        speech_audio = np.random.random(16000).astype(np.float32) * 0.5
        # Note: This is a simple test - real VAD would need actual speech patterns
        result = audio_processor.detect_voice_activity(speech_audio)
        assert isinstance(result, bool)
    
    def test_audio_chunking(self, audio_processor):
        """Test audio chunking for streaming."""
        long_audio = np.random.random(48000).astype(np.float32)  # 3 seconds
        chunk_duration_ms = 1000  # 1 second chunks
        
        chunks = audio_processor.chunk_audio(long_audio, chunk_duration_ms)
        
        assert len(chunks) == 3
        for chunk in chunks:
            assert len(chunk) == 16000  # 1 second at 16kHz
    
    def test_format_conversion(self, audio_processor):
        """Test audio format conversion."""
        # Test float32 to int16 conversion
        float_audio = np.random.random(1000).astype(np.float32)
        int_audio = audio_processor.float32_to_int16(float_audio)
        
        assert int_audio.dtype == np.int16
        assert int_audio.shape == float_audio.shape
        
        # Test back conversion
        converted_back = audio_processor.int16_to_float32(int_audio)
        assert converted_back.dtype == np.float32
        np.testing.assert_allclose(converted_back, float_audio, atol=1e-4)


@pytest.mark.integration
class TestSTTIntegration:
    """Integration tests for STT service."""
    
    @pytest.mark.skipif(
        not Path("/app/models/whisper").exists(),
        reason="Whisper models not available"
    )
    def test_real_whisper_model(self):
        """Test with real Whisper model (requires model download)."""
        engine = WhisperSTTEngine(model_size='tiny')  # Use smallest model for testing
        
        # Create simple test audio (sine wave)
        duration = 2.0
        sample_rate = 16000
        frequency = 440
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        result = engine.transcribe_array(audio)
        
        # Should at least return some structure even for synthetic audio
        assert 'text' in result
        assert 'language' in result
        assert isinstance(result['text'], str)
    
    @pytest.mark.asyncio
    async def test_concurrent_transcription(self):
        """Test concurrent transcription requests."""
        service = STTService()
        
        # Create multiple audio samples
        audio_samples = [
            np.random.random(16000).astype(np.float32) for _ in range(3)
        ]
        
        # Process concurrently
        with patch.object(service.primary_engine, 'transcribe_array') as mock_transcribe:
            mock_transcribe.return_value = {
                'text': 'Concurrent test',
                'language': 'en',
                'confidence': 0.9
            }
            
            tasks = [
                service.transcribe_audio_async(audio) 
                for audio in audio_samples
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            for result in results:
                assert result['text'] == 'Concurrent test'
                assert result['language'] == 'en'