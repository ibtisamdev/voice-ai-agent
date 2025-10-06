"""Tests for Text-to-Speech (TTS) voice service."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import asyncio
from pathlib import Path
import tempfile
import io

from ai.voice.tts_service import TTSService, CoquiTTSEngine, ElevenLabsTTSEngine, AzureTTSEngine


class TestCoquiTTSEngine:
    """Test Coqui TTS engine."""
    
    @pytest.fixture
    def mock_coqui_tts(self):
        """Mock Coqui TTS model."""
        tts = Mock()
        tts.tts.return_value = np.random.random(22050).astype(np.float32)  # 1 second at 22kHz
        tts.list_speakers.return_value = ['speaker1', 'speaker2']
        return tts
    
    @pytest.fixture
    def coqui_engine(self, mock_coqui_tts):
        """Create Coqui TTS engine with mocked model."""
        with patch('ai.voice.tts_service.TTS', return_value=mock_coqui_tts):
            engine = CoquiTTSEngine()
            return engine
    
    def test_engine_initialization(self, coqui_engine):
        """Test Coqui TTS engine initialization."""
        assert coqui_engine.model is not None
        assert coqui_engine.sample_rate == 22050
        assert hasattr(coqui_engine, 'speakers')
    
    def test_text_synthesis(self, coqui_engine):
        """Test text synthesis."""
        text = "Hello, this is a test of text-to-speech synthesis."
        
        result = coqui_engine.synthesize(text)
        
        assert 'audio' in result
        assert 'metadata' in result
        assert isinstance(result['audio'], np.ndarray)
        assert result['audio'].dtype == np.float32
        assert result['metadata']['sample_rate'] == 22050
        assert result['metadata']['duration'] > 0
    
    def test_synthesis_with_voice(self, coqui_engine):
        """Test synthesis with specific voice."""
        text = "Testing voice selection."
        voice_id = "speaker1"
        
        result = coqui_engine.synthesize(text, voice_id=voice_id)
        
        assert result['metadata']['voice_id'] == voice_id
        assert isinstance(result['audio'], np.ndarray)
    
    def test_available_voices(self, coqui_engine):
        """Test getting available voices."""
        voices = coqui_engine.get_available_voices()
        
        assert isinstance(voices, list)
        assert len(voices) > 0
        for voice in voices:
            assert 'id' in voice
            assert 'name' in voice
    
    def test_audio_file_output(self, coqui_engine, tmp_path):
        """Test saving audio to file."""
        text = "Testing file output."
        output_file = tmp_path / "test_output.wav"
        
        result = coqui_engine.synthesize_to_file(text, str(output_file))
        
        assert output_file.exists()
        assert result['file_path'] == str(output_file)
        assert result['metadata']['format'] == 'wav'


class TestElevenLabsTTSEngine:
    """Test ElevenLabs TTS engine."""
    
    @pytest.fixture
    def mock_elevenlabs_client(self):
        """Mock ElevenLabs client."""
        client = Mock()
        
        # Mock audio response
        audio_response = Mock()
        audio_response.content = b"fake_audio_data_" * 1000  # Simulate audio bytes
        
        client.generate.return_value = audio_response
        client.get_voices.return_value = Mock(voices=[
            Mock(voice_id="voice1", name="Alice"),
            Mock(voice_id="voice2", name="Bob")
        ])
        
        return client
    
    @pytest.fixture
    def elevenlabs_engine(self, mock_elevenlabs_client):
        """Create ElevenLabs TTS engine with mocked client."""
        with patch('ai.voice.tts_service.ElevenLabs', return_value=mock_elevenlabs_client):
            engine = ElevenLabsTTSEngine(api_key="test_key")
            return engine
    
    def test_engine_initialization(self, elevenlabs_engine):
        """Test ElevenLabs TTS engine initialization."""
        assert elevenlabs_engine.client is not None
        assert elevenlabs_engine.api_key == "test_key"
    
    def test_text_synthesis(self, elevenlabs_engine):
        """Test text synthesis with ElevenLabs."""
        text = "Hello from ElevenLabs TTS."
        
        result = elevenlabs_engine.synthesize(text)
        
        assert 'audio' in result
        assert 'metadata' in result
        assert isinstance(result['audio'], bytes)
        assert result['metadata']['engine'] == 'elevenlabs'
    
    def test_synthesis_with_voice(self, elevenlabs_engine):
        """Test synthesis with specific ElevenLabs voice."""
        text = "Testing ElevenLabs voice."
        voice_id = "voice1"
        
        result = elevenlabs_engine.synthesize(text, voice_id=voice_id)
        
        assert result['metadata']['voice_id'] == voice_id
        elevenlabs_engine.client.generate.assert_called_once()
    
    def test_available_voices(self, elevenlabs_engine):
        """Test getting ElevenLabs voices."""
        voices = elevenlabs_engine.get_available_voices()
        
        assert isinstance(voices, list)
        assert len(voices) == 2
        assert voices[0]['id'] == 'voice1'
        assert voices[0]['name'] == 'Alice'
    
    def test_api_error_handling(self, elevenlabs_engine):
        """Test API error handling."""
        elevenlabs_engine.client.generate.side_effect = Exception("API Error")
        
        with pytest.raises(Exception):
            elevenlabs_engine.synthesize("Test text")


class TestAzureTTSEngine:
    """Test Azure TTS engine."""
    
    @pytest.fixture
    def mock_azure_client(self):
        """Mock Azure Speech client."""
        client = Mock()
        
        # Mock synthesis result
        result = Mock()
        result.audio_data = b"azure_audio_data_" * 500
        result.reason = Mock()
        result.reason.name = "SynthesisCompleted"
        
        client.speak_text_async.return_value.get.return_value = result
        
        return client
    
    @pytest.fixture
    def azure_engine(self, mock_azure_client):
        """Create Azure TTS engine with mocked client."""
        with patch('ai.voice.tts_service.SpeechSynthesizer', return_value=mock_azure_client):
            engine = AzureTTSEngine(subscription_key="test_key", region="eastus")
            return engine
    
    def test_engine_initialization(self, azure_engine):
        """Test Azure TTS engine initialization."""
        assert azure_engine.client is not None
        assert azure_engine.subscription_key == "test_key"
        assert azure_engine.region == "eastus"
    
    def test_text_synthesis(self, azure_engine):
        """Test text synthesis with Azure."""
        text = "Hello from Azure TTS."
        
        result = azure_engine.synthesize(text)
        
        assert 'audio' in result
        assert 'metadata' in result
        assert isinstance(result['audio'], bytes)
        assert result['metadata']['engine'] == 'azure'
    
    def test_synthesis_with_voice(self, azure_engine):
        """Test synthesis with Azure voice."""
        text = "Testing Azure voice."
        voice_id = "en-US-JennyNeural"
        
        result = azure_engine.synthesize(text, voice_id=voice_id)
        
        assert result['metadata']['voice_id'] == voice_id
    
    def test_ssml_synthesis(self, azure_engine):
        """Test SSML synthesis."""
        ssml = """
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
            <voice name="en-US-JennyNeural">
                <prosody rate="slow" pitch="low">Hello, this is SSML test.</prosody>
            </voice>
        </speak>
        """
        
        result = azure_engine.synthesize_ssml(ssml)
        
        assert 'audio' in result
        assert result['metadata']['format'] == 'ssml'


class TestTTSService:
    """Test TTS service orchestration."""
    
    @pytest.fixture
    def mock_engines(self):
        """Mock TTS engines."""
        coqui_engine = Mock()
        coqui_engine.synthesize.return_value = {
            'audio': np.random.random(22050).astype(np.float32),
            'metadata': {'engine': 'coqui', 'sample_rate': 22050, 'duration': 1.0}
        }
        coqui_engine.get_available_voices.return_value = [
            {'id': 'coqui_voice1', 'name': 'Coqui Voice 1'}
        ]
        
        elevenlabs_engine = Mock()
        elevenlabs_engine.synthesize.return_value = {
            'audio': b"elevenlabs_audio_data",
            'metadata': {'engine': 'elevenlabs', 'duration': 1.0}
        }
        elevenlabs_engine.get_available_voices.return_value = [
            {'id': 'el_voice1', 'name': 'ElevenLabs Voice 1'}
        ]
        
        return {'coqui': coqui_engine, 'elevenlabs': elevenlabs_engine}
    
    @pytest.fixture
    def tts_service(self, mock_engines):
        """Create TTS service with mocked engines."""
        with patch('ai.voice.tts_service.CoquiTTSEngine', return_value=mock_engines['coqui']), \
             patch('ai.voice.tts_service.ElevenLabsTTSEngine', return_value=mock_engines['elevenlabs']):
            service = TTSService(default_engine='coqui')
            return service
    
    def test_service_initialization(self, tts_service):
        """Test TTS service initialization."""
        assert tts_service.default_engine == 'coqui'
        assert 'coqui' in tts_service.engines
        assert len(tts_service.engines) > 0
    
    @pytest.mark.asyncio
    async def test_text_synthesis_async(self, tts_service):
        """Test async text synthesis."""
        text = "Hello, this is an async TTS test."
        
        result = await tts_service.synthesize_async(text)
        
        assert 'audio' in result
        assert 'metadata' in result
        assert result['metadata']['engine'] == 'coqui'
    
    def test_engine_selection(self, tts_service):
        """Test engine selection for synthesis."""
        text = "Testing engine selection."
        
        # Test with specific engine
        result = tts_service.synthesize(text, engine='elevenlabs')
        assert result['metadata']['engine'] == 'elevenlabs'
        
        # Test with default engine
        result = tts_service.synthesize(text)
        assert result['metadata']['engine'] == 'coqui'
    
    def test_engine_fallback(self, tts_service, mock_engines):
        """Test engine fallback on failure."""
        # Make primary engine fail
        mock_engines['coqui'].synthesize.side_effect = Exception("Engine failed")
        
        text = "Testing fallback mechanism."
        
        # Should fallback to another engine
        result = tts_service.synthesize(text)
        
        # Should succeed with fallback engine
        assert 'audio' in result
        assert 'metadata' in result
    
    def test_voice_listing(self, tts_service):
        """Test getting all available voices."""
        voices = tts_service.get_all_voices()
        
        assert isinstance(voices, dict)
        assert 'coqui' in voices
        assert 'elevenlabs' in voices
        assert len(voices['coqui']) > 0
        assert len(voices['elevenlabs']) > 0
    
    def test_caching(self, tts_service):
        """Test synthesis result caching."""
        text = "This text should be cached."
        
        # First synthesis
        result1 = tts_service.synthesize(text, use_cache=True)
        
        # Second synthesis (should use cache)
        result2 = tts_service.synthesize(text, use_cache=True)
        
        # Should be identical results
        assert result1['metadata']['engine'] == result2['metadata']['engine']
        
        # Mock should only be called once due to caching
        tts_service.engines['coqui'].synthesize.assert_called_once()
    
    def test_audio_format_conversion(self, tts_service):
        """Test audio format conversion."""
        text = "Testing format conversion."
        
        # Test different output formats
        wav_result = tts_service.synthesize(text, output_format='wav')
        mp3_result = tts_service.synthesize(text, output_format='mp3')
        
        assert wav_result['metadata']['format'] == 'wav'
        assert mp3_result['metadata']['format'] == 'mp3'
    
    @pytest.mark.asyncio
    async def test_concurrent_synthesis(self, tts_service):
        """Test concurrent synthesis requests."""
        texts = [
            "First concurrent text.",
            "Second concurrent text.",
            "Third concurrent text."
        ]
        
        # Process concurrently
        tasks = [tts_service.synthesize_async(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert 'audio' in result
            assert 'metadata' in result
    
    def test_synthesis_parameters(self, tts_service):
        """Test synthesis with various parameters."""
        text = "Testing synthesis parameters."
        
        result = tts_service.synthesize(
            text,
            voice_id="test_voice",
            speed=1.2,
            pitch=1.1,
            volume=0.8
        )
        
        assert 'audio' in result
        metadata = result['metadata']
        assert metadata.get('voice_id') == "test_voice" or 'voice_id' in metadata
    
    def test_synthesis_validation(self, tts_service):
        """Test input validation for synthesis."""
        # Test empty text
        with pytest.raises(ValueError):
            tts_service.synthesize("")
        
        # Test invalid engine
        with pytest.raises(ValueError):
            tts_service.synthesize("Test", engine="nonexistent_engine")
        
        # Test invalid parameters
        with pytest.raises(ValueError):
            tts_service.synthesize("Test", speed=5.0)  # Too fast


@pytest.mark.integration
class TestTTSIntegration:
    """Integration tests for TTS service."""
    
    @pytest.mark.skipif(
        not Path("/app/models/tts").exists(),
        reason="TTS models not available"
    )
    def test_real_coqui_model(self):
        """Test with real Coqui TTS model (requires model download)."""
        engine = CoquiTTSEngine()
        
        text = "This is a real TTS integration test."
        result = engine.synthesize(text)
        
        assert 'audio' in result
        assert isinstance(result['audio'], np.ndarray)
        assert result['audio'].dtype == np.float32
        assert len(result['audio']) > 0
    
    def test_audio_quality_metrics(self):
        """Test audio quality metrics."""
        service = TTSService()
        
        text = "Testing audio quality metrics and validation."
        result = service.synthesize(text)
        
        audio = result['audio']
        metadata = result['metadata']
        
        # Check basic audio properties
        assert metadata['sample_rate'] > 0
        assert metadata['duration'] > 0
        
        if isinstance(audio, np.ndarray):
            # Check for silence (audio should have variance)
            assert np.var(audio) > 1e-6
            
            # Check for clipping (values should be in reasonable range)
            assert np.max(np.abs(audio)) <= 1.0
    
    @pytest.mark.asyncio
    async def test_stress_test(self):
        """Stress test with multiple concurrent requests."""
        service = TTSService()
        
        # Generate many synthesis requests
        texts = [f"Stress test number {i}" for i in range(10)]
        
        # Process all concurrently
        tasks = [service.synthesize_async(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that most requests succeeded
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 8  # Allow some failures under stress
        
        for result in successful_results:
            assert 'audio' in result
            assert 'metadata' in result