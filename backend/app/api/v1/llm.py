"""
LLM API endpoints for direct language model interactions.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import uuid

from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
import json

from ai.llm.llm_service import llm_service
from ai.llm.prompt_templates import LegalPromptTemplates
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# Request/Response Models
class LLMGenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="The prompt to generate text from", min_length=1)
    system_prompt: Optional[str] = Field(None, description="System prompt to guide the response")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(2048, ge=1, le=4096, description="Maximum tokens to generate")
    stream: bool = Field(False, description="Whether to stream the response")


class LLMChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['system', 'user', 'assistant']:
            raise ValueError('Role must be one of: system, user, assistant')
        return v


class LLMChatRequest(BaseModel):
    """Request model for chat completion."""
    messages: List[LLMChatMessage] = Field(..., description="List of chat messages")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(2048, ge=1, le=4096, description="Maximum tokens to generate")


class LLMDocumentAnalysisRequest(BaseModel):
    """Request model for document analysis."""
    document_text: str = Field(..., description="Document text to analyze", min_length=1)
    analysis_type: str = Field("general", description="Type of analysis to perform")
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        valid_types = ['general', 'contract', 'brief', 'estate', 'litigation']
        if v not in valid_types:
            raise ValueError(f'Analysis type must be one of: {", ".join(valid_types)}')
        return v


class LLMConversationRequest(BaseModel):
    """Request model for starting a conversation."""
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")
    system_prompt_type: str = Field("legal_assistant", description="Type of system prompt")
    
    @validator('system_prompt_type')
    def validate_system_prompt_type(cls, v):
        valid_types = ['legal_assistant', 'document_analysis', 'conversation']
        if v not in valid_types:
            raise ValueError(f'System prompt type must be one of: {", ".join(valid_types)}')
        return v


class LLMResponse(BaseModel):
    """Response model for LLM operations."""
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class LLMHealthResponse(BaseModel):
    """Response model for health check."""
    healthy: bool
    current_model: Optional[str] = None
    available_models: List[str] = []
    inference_time_seconds: Optional[float] = None
    active_conversations: int = 0
    timestamp: str


# API Endpoints
@router.post("/generate", response_model=LLMResponse)
async def generate_text(request: LLMGenerateRequest, http_request: Request):
    """Generate text using the LLM."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        logger.info(f"Generate request: {request.prompt[:100]}...", extra={"request_id": request_id})
        
        if request.stream:
            # For streaming responses
            async def generate_stream():
                try:
                    yield "data: " + json.dumps({"type": "start"}) + "\n\n"
                    
                    async for chunk in llm_service.generate_response(
                        prompt=request.prompt,
                        system_prompt=request.system_prompt,
                        conversation_id=request.conversation_id,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        stream=True
                    ):
                        if chunk:
                            data = {
                                "type": "content",
                                "content": chunk
                            }
                            yield "data: " + json.dumps(data) + "\n\n"
                    
                    yield "data: " + json.dumps({"type": "end"}) + "\n\n"
                    
                except Exception as e:
                    error_data = {
                        "type": "error",
                        "error": str(e)
                    }
                    yield "data: " + json.dumps(error_data) + "\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # Non-streaming response
            response_text = ""
            async for chunk in llm_service.generate_response(
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                conversation_id=request.conversation_id,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False
            ):
                response_text += chunk
            
            return LLMResponse(
                success=True,
                response=response_text,
                metadata={
                    "conversation_id": request.conversation_id,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens
                }
            )
    
    except Exception as e:
        logger.error(f"Error in generate endpoint: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text generation failed: {str(e)}"
        )


@router.post("/chat", response_model=LLMResponse)
async def chat_completion(request: LLMChatRequest, http_request: Request):
    """Complete a chat conversation."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        logger.info(f"Chat request with {len(request.messages)} messages", extra={"request_id": request_id})
        
        # Convert to internal format
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        response_text = await llm_service.chat_completion(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return LLMResponse(
            success=True,
            response=response_text,
            metadata={
                "message_count": len(messages),
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }
        )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat completion failed: {str(e)}"
        )


@router.post("/analyze-document", response_model=LLMResponse)
async def analyze_document(request: LLMDocumentAnalysisRequest, http_request: Request):
    """Analyze a legal document."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        logger.info(f"Document analysis request: {request.analysis_type}", extra={"request_id": request_id})
        
        analysis_result = await llm_service.analyze_document(
            document_text=request.document_text,
            analysis_type=request.analysis_type
        )
        
        return LLMResponse(
            success=True,
            response=analysis_result,
            metadata={
                "analysis_type": request.analysis_type,
                "document_length": len(request.document_text)
            }
        )
    
    except Exception as e:
        logger.error(f"Error in analyze-document endpoint: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document analysis failed: {str(e)}"
        )


@router.post("/conversation/start", response_model=LLMResponse)
async def start_conversation(request: LLMConversationRequest, http_request: Request):
    """Start a new conversation context."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        conversation_id = request.conversation_id or str(uuid.uuid4())
        system_prompt = LegalPromptTemplates.get_system_prompt(request.system_prompt_type)
        
        llm_service.start_conversation(conversation_id, system_prompt)
        
        logger.info(f"Started conversation: {conversation_id}", extra={"request_id": request_id})
        
        return LLMResponse(
            success=True,
            response=f"Conversation {conversation_id} started",
            metadata={
                "conversation_id": conversation_id,
                "system_prompt_type": request.system_prompt_type
            }
        )
    
    except Exception as e:
        logger.error(f"Error starting conversation: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start conversation: {str(e)}"
        )


@router.delete("/conversation/{conversation_id}")
async def end_conversation(conversation_id: str, http_request: Request):
    """End a conversation and clean up context."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        llm_service.end_conversation(conversation_id)
        
        logger.info(f"Ended conversation: {conversation_id}", extra={"request_id": request_id})
        
        return LLMResponse(
            success=True,
            response=f"Conversation {conversation_id} ended",
            metadata={"conversation_id": conversation_id}
        )
    
    except Exception as e:
        logger.error(f"Error ending conversation: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to end conversation: {str(e)}"
        )


@router.get("/conversation/{conversation_id}/history")
async def get_conversation_history(conversation_id: str, http_request: Request):
    """Get conversation history."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        history = llm_service.get_conversation_history(conversation_id)
        
        return LLMResponse(
            success=True,
            response="Conversation history retrieved",
            metadata={
                "conversation_id": conversation_id,
                "message_count": len(history),
                "history": history
            }
        )
    
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation history: {str(e)}"
        )


@router.get("/models")
async def get_available_models(http_request: Request):
    """Get list of available models."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        models = await llm_service.get_available_models()
        current_model = llm_service.current_model
        
        return LLMResponse(
            success=True,
            response="Available models retrieved",
            metadata={
                "current_model": current_model,
                "available_models": models,
                "model_count": len(models)
            }
        )
    
    except Exception as e:
        logger.error(f"Error getting models: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get models: {str(e)}"
        )


@router.post("/models/{model_name}/switch")
async def switch_model(model_name: str, http_request: Request):
    """Switch to a different model."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        success = await llm_service.switch_model(model_name)
        
        if success:
            logger.info(f"Switched to model: {model_name}", extra={"request_id": request_id})
            return LLMResponse(
                success=True,
                response=f"Switched to model: {model_name}",
                metadata={"current_model": model_name}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {model_name} not available"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching model: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to switch model: {str(e)}"
        )


@router.get("/health", response_model=LLMHealthResponse)
async def health_check():
    """Check LLM service health."""
    try:
        health_info = await llm_service.health_check()
        
        return LLMHealthResponse(
            healthy=health_info.get("healthy", False),
            current_model=health_info.get("current_model"),
            available_models=health_info.get("available_models", []),
            inference_time_seconds=health_info.get("inference_time_seconds"),
            active_conversations=health_info.get("active_conversations", 0),
            timestamp=health_info.get("timestamp", datetime.utcnow().isoformat())
        )
    
    except Exception as e:
        logger.error(f"Error in LLM health check: {e}")
        return LLMHealthResponse(
            healthy=False,
            current_model=None,
            available_models=[],
            inference_time_seconds=None,
            active_conversations=0,
            timestamp=datetime.utcnow().isoformat()
        )


@router.get("/info")
async def get_model_info(http_request: Request):
    """Get detailed information about the current model."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        model_info = await llm_service.get_model_info()
        
        return LLMResponse(
            success=True,
            response="Model information retrieved",
            metadata=model_info
        )
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )