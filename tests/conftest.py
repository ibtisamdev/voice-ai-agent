import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import redis
import numpy as np
from unittest.mock import MagicMock

from app.main import app
from app.core.database import get_db, get_redis_client, Base
from app.core.config import settings

# Test database URL (SQLite in memory)
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


def override_get_redis():
    """Override Redis dependency for testing."""
    # Return a mock Redis client for testing
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    return mock_redis


@pytest.fixture(scope="session")
def client():
    """Create test client."""
    # Create test database tables
    Base.metadata.create_all(bind=engine)
    
    # Override dependencies
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_redis_client] = override_get_redis
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session():
    """Create database session for testing."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    mock = MagicMock()
    mock.ping.return_value = True
    return mock


@pytest.fixture(autouse=True)
def clear_lru_cache():
    """Clear any LRU caches between tests."""
    yield
    # Add any cache clearing logic here if needed


# Voice service fixtures
@pytest.fixture
def mock_stt_service():
    """Mock STT service for testing."""
    stt_service = MagicMock()
    stt_service.transcribe_file_async.return_value = {
        'text': 'Mock transcription result',
        'confidence': 0.95,
        'language': 'en'
    }
    stt_service.transcribe_audio_async.return_value = {
        'text': 'Mock transcription result',
        'confidence': 0.95,
        'language': 'en'
    }
    stt_service.create_session.return_value = {
        'session_id': 'mock_session',
        'status': 'active'
    }
    return stt_service


@pytest.fixture
def mock_tts_service():
    """Mock TTS service for testing."""
    tts_service = MagicMock()
    tts_service.synthesize_async.return_value = {
        'audio': b'mock_audio_data',
        'metadata': {
            'duration': 2.5,
            'sample_rate': 22050,
            'format': 'wav'
        }
    }
    tts_service.get_available_voices.return_value = [
        {'id': 'voice1', 'name': 'Test Voice 1'},
        {'id': 'voice2', 'name': 'Test Voice 2'}
    ]
    return tts_service


@pytest.fixture
def mock_conversation_state():
    """Mock conversation state manager for testing."""
    state_manager = MagicMock()
    
    mock_session = MagicMock()
    mock_session.session_id = 'test_session'
    mock_session.add_turn = MagicMock()
    mock_session.to_dict.return_value = {'session_id': 'test_session'}
    
    state_manager.create_session.return_value = mock_session
    state_manager.get_session.return_value = mock_session
    state_manager.update_session.return_value = True
    
    return state_manager


@pytest.fixture
def mock_dialog_engine():
    """Mock dialog engine for testing."""
    dialog_engine = MagicMock()
    dialog_engine.start_flow.return_value = {
        'response': 'Hello! How can I help you?',
        'next_action': 'wait_for_input',
        'current_node': 'greeting'
    }
    dialog_engine.process_input.return_value = {
        'response': 'I understand you need assistance.',
        'next_action': 'wait_for_input',
        'current_node': 'assistance'
    }
    return dialog_engine


@pytest.fixture
def mock_intent_classifier():
    """Mock intent classifier for testing."""
    classifier = MagicMock()
    classifier.classify.return_value = {
        'intent': 'general_inquiry',
        'confidence': 0.8
    }
    classifier.classify_batch.return_value = [
        {'intent': 'general_inquiry', 'confidence': 0.8}
    ]
    return classifier


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    import numpy as np
    # Generate 1 second of sine wave at 440Hz
    sample_rate = 16000
    duration = 1.0
    frequency = 440
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


@pytest.fixture
def temp_audio_file(tmp_path, sample_audio_data):
    """Create temporary audio file for testing."""
    import wave
    
    audio_file = tmp_path / "test_audio.wav"
    
    # Convert float32 to int16 for WAV format
    audio_int16 = (sample_audio_data * 32767).astype(np.int16)
    
    with wave.open(str(audio_file), 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(16000)  # 16kHz
        wav_file.writeframes(audio_int16.tobytes())
    
    return str(audio_file)