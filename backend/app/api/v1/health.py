from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any
import redis
import asyncio
from datetime import datetime

from ...core.config import settings
from ...core.database import get_db, get_redis_client
from ...core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", summary="Health Check")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": settings.PROJECT_NAME,
        "version": "0.1.0",
        "environment": settings.ENVIRONMENT
    }


@router.get("/ready", summary="Readiness Check")
async def readiness_check(
    db: Session = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis_client)
) -> Dict[str, Any]:
    """Comprehensive readiness check including dependencies."""
    
    checks = {
        "database": False,
        "redis": False,
        "overall": False
    }
    
    # Database check
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        checks["database"] = True
        logger.info("Database connection: OK")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        checks["database"] = False
    
    # Redis check
    try:
        redis_client.ping()
        checks["redis"] = True
        logger.info("Redis connection: OK")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        checks["redis"] = False
    
    # Overall status
    checks["overall"] = all([checks["database"], checks["redis"]])
    
    response_data = {
        "status": "ready" if checks["overall"] else "not_ready",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
        "service": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT
    }
    
    if not checks["overall"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response_data
        )
    
    return response_data


@router.get("/version", summary="Version Information")
async def version_info() -> Dict[str, Any]:
    """Get version and build information."""
    return {
        "service": settings.PROJECT_NAME,
        "version": "0.1.0",
        "api_version": settings.API_V1_STR,
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
        "timestamp": datetime.utcnow().isoformat()
    }