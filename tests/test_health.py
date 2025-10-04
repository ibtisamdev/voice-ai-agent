import pytest
from fastapi.testclient import TestClient
from app.core.config import settings


def test_health_endpoint(client: TestClient):
    """Test basic health check endpoint."""
    response = client.get("/api/v1/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "healthy"
    assert data["service"] == settings.PROJECT_NAME
    assert data["version"] == "0.1.0"
    assert data["environment"] == settings.ENVIRONMENT
    assert "timestamp" in data


def test_readiness_endpoint(client: TestClient):
    """Test readiness check endpoint."""
    response = client.get("/api/v1/ready")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "ready"
    assert "checks" in data
    assert data["checks"]["database"] == True
    assert data["checks"]["redis"] == True
    assert data["checks"]["overall"] == True
    assert "timestamp" in data


def test_version_endpoint(client: TestClient):
    """Test version information endpoint."""
    response = client.get("/api/v1/version")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["service"] == settings.PROJECT_NAME
    assert data["version"] == "0.1.0"
    assert data["api_version"] == settings.API_V1_STR
    assert data["environment"] == settings.ENVIRONMENT
    assert "debug" in data
    assert "timestamp" in data


def test_root_endpoint(client: TestClient):
    """Test root endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "message" in data
    assert "docs" in data
    assert "health" in data
    assert settings.PROJECT_NAME in data["message"]


def test_health_endpoint_headers(client: TestClient):
    """Test that response includes proper headers."""
    response = client.get("/api/v1/health")
    
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time" in response.headers


def test_invalid_endpoint(client: TestClient):
    """Test invalid endpoint returns 404."""
    response = client.get("/api/v1/nonexistent")
    
    assert response.status_code == 404