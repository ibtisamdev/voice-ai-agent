from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time
import uuid

from .core.config import settings
from .core.logging import setup_logging, get_logger, RequestContextFilter
from .api.v1 import health

# Setup logging
setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="0.1.0",
    description="AI Law Firm Voice Agent API",
    openapi_url=f"{settings.API_V1_STR}/openapi.json" if settings.DEBUG else None,
    docs_url=f"{settings.API_V1_STR}/docs" if settings.DEBUG else None,
    redoc_url=f"{settings.API_V1_STR}/redoc" if settings.DEBUG else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Add request logging and timing."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    
    # Add request context to logs
    logger.info(
        f"Request started: {request.method} {request.url}",
        extra={"request_id": request_id}
    )
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"Request completed: {request.method} {request.url} - "
        f"Status: {response.status_code} - Time: {process_time:.3f}s",
        extra={"request_id": request_id}
    )
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": exc.errors(),
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions."""
    request_id = getattr(request.state, "request_id", None)
    logger.error(f"Unhandled exception: {exc}", exc_info=True, extra={"request_id": request_id})
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "request_id": request_id
        }
    )


# Include routers
app.include_router(
    health.router,
    prefix=settings.API_V1_STR,
    tags=["health"]
)

# Import and include new API routers
from .api.v1 import llm, rag, documents, voice, voice_ws

app.include_router(
    llm.router,
    prefix=f"{settings.API_V1_STR}/llm",
    tags=["llm"]
)

app.include_router(
    rag.router,
    prefix=f"{settings.API_V1_STR}/rag",
    tags=["rag"]
)

app.include_router(
    documents.router,
    prefix=f"{settings.API_V1_STR}/documents",
    tags=["documents"]
)

app.include_router(
    voice.router,
    prefix=f"{settings.API_V1_STR}/voice",
    tags=["voice"]
)

app.include_router(
    voice_ws.router,
    prefix=f"{settings.API_V1_STR}/ws",
    tags=["websockets"]
)


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info(f"Starting {settings.PROJECT_NAME}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Initialize voice services
    try:
        from ai.voice import audio_processor, stt_service, tts_service
        from ai.conversation import conversation_state_manager, dialog_flow_engine
        from ai.decision_engine import intent_classifier
        
        logger.info("Initializing voice services...")
        
        # Initialize in dependency order
        await audio_processor.initialize()
        await stt_service.initialize()
        await tts_service.initialize()
        await conversation_state_manager.initialize()
        await dialog_flow_engine.initialize()
        await intent_classifier.initialize()
        
        logger.info("Voice services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize voice services: {e}")
        # Continue startup even if voice services fail


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info(f"Shutting down {settings.PROJECT_NAME}")


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs": f"{settings.API_V1_STR}/docs",
        "health": f"{settings.API_V1_STR}/health"
    }