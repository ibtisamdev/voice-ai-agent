from fastapi import APIRouter

from . import health, voice, voice_ws, llm, rag, documents
from . import crm, telephony, campaigns  # Phase 3 endpoints

api_router = APIRouter()

# Phase 1 & 2 endpoints
api_router.include_router(health.router)
api_router.include_router(voice.router)
api_router.include_router(voice_ws.router)
api_router.include_router(llm.router)
api_router.include_router(rag.router)
api_router.include_router(documents.router)

# Phase 3 endpoints
api_router.include_router(crm.router)
api_router.include_router(telephony.router)
api_router.include_router(campaigns.router)