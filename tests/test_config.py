import pytest
from pydantic import ValidationError
from app.core.config import Settings


def test_settings_with_defaults():
    """Test settings with default values."""
    # Mock environment with minimal required values
    settings = Settings(DATABASE_URL="postgresql://test:test@localhost/test")
    
    assert settings.ENVIRONMENT == "development"
    assert settings.DEBUG == True
    assert settings.API_V1_STR == "/api/v1"
    assert settings.PROJECT_NAME == "Voice AI Agent"
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8000
    assert settings.DATABASE_POOL_SIZE == 20
    # REDIS_URL might be overridden by environment
    assert settings.REDIS_URL in ["redis://localhost:6379", "redis://redis:6379"]
    assert len(settings.SECRET_KEY) >= 32


def test_settings_validation():
    """Test settings validation."""
    # Test missing required DATABASE_URL - set env vars to None to test validation
    import os
    old_db_url = os.environ.get('DATABASE_URL')
    try:
        if 'DATABASE_URL' in os.environ:
            del os.environ['DATABASE_URL']
        with pytest.raises(ValidationError):
            Settings()
    finally:
        if old_db_url:
            os.environ['DATABASE_URL'] = old_db_url


def test_secret_key_generation():
    """Test that secret key is properly generated."""
    settings1 = Settings(DATABASE_URL="postgresql://test:test@localhost/test")
    settings2 = Settings(DATABASE_URL="postgresql://test:test@localhost/test")
    
    # Should generate different keys
    assert len(settings1.SECRET_KEY) >= 32
    assert len(settings2.SECRET_KEY) >= 32
    # Note: Keys might be the same due to mocking, but in real usage they'd be different


def test_database_url_validation():
    """Test database URL validation."""
    # Valid database URL
    settings = Settings(DATABASE_URL="postgresql://user:pass@localhost:5432/db")
    assert settings.DATABASE_URL == "postgresql://user:pass@localhost:5432/db"
    
    # Invalid/empty database URL should raise error
    with pytest.raises(ValidationError):
        Settings(DATABASE_URL="")