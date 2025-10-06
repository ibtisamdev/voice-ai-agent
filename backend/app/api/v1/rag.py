"""
RAG API endpoints for document-based question answering.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from fastapi import APIRouter, HTTPException, status, Depends, Request, Query
from pydantic import BaseModel, Field, validator
import json

from ai.rag.rag_service import rag_service, RAGServiceError
from ai.rag.chunking_strategies import ChunkingStrategy
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# Request/Response Models
class RAGQueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., description="Question to ask", min_length=1)
    context_filters: Optional[Dict[str, Any]] = Field(None, description="Filters for document context")
    max_results: Optional[int] = Field(5, ge=1, le=20, description="Maximum number of results to retrieve")
    similarity_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold for retrieval")
    include_sources: bool = Field(True, description="Whether to include source documents in response")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    system_prompt_type: str = Field("legal_assistant", description="Type of system prompt to use")
    
    @validator('system_prompt_type')
    def validate_system_prompt_type(cls, v):
        valid_types = ['legal_assistant', 'document_analysis', 'conversation']
        if v not in valid_types:
            raise ValueError(f'System prompt type must be one of: {", ".join(valid_types)}')
        return v


class DocumentSearchRequest(BaseModel):
    """Request model for document search."""
    query: Optional[str] = Field(None, description="Search query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of results")


class RAGQueryResponse(BaseModel):
    """Response model for RAG queries."""
    success: bool
    query: Optional[str] = None
    answer: Optional[str] = None
    sources: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class DocumentSearchResponse(BaseModel):
    """Response model for document search."""
    success: bool
    results: List[Dict[str, Any]] = []
    total_results: int = 0
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class RAGStatsResponse(BaseModel):
    """Response model for RAG statistics."""
    success: bool
    stats: Dict[str, Any] = {}
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class RAGHealthResponse(BaseModel):
    """Response model for RAG health check."""
    healthy: bool
    initialized: bool
    components: Dict[str, Any] = {}
    end_to_end_test: Dict[str, Any] = {}
    cache_stats: Dict[str, Any] = {}
    total_check_time_seconds: Optional[float] = None
    timestamp: str


# API Endpoints
@router.post("/query", response_model=RAGQueryResponse)
async def query_documents(request: RAGQueryRequest, http_request: Request):
    """Query documents using RAG (Retrieval-Augmented Generation)."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        logger.info(f"RAG query: {request.question[:100]}...", extra={"request_id": request_id})
        
        # Ensure RAG service is initialized
        if not rag_service.initialized:
            logger.info("RAG service not initialized, initializing now...")
            if not await rag_service.initialize():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="RAG service initialization failed"
                )
        
        # Execute query
        result = await rag_service.query(
            question=request.question,
            context_filters=request.context_filters,
            max_results=request.max_results,
            similarity_threshold=request.similarity_threshold,
            include_sources=request.include_sources,
            conversation_id=request.conversation_id,
            system_prompt_type=request.system_prompt_type
        )
        
        logger.info(f"RAG query completed in {result.total_time:.2f}s", extra={"request_id": request_id})
        
        return RAGQueryResponse(
            success=True,
            query=result.query,
            answer=result.answer,
            sources=result.sources,
            metadata={
                "retrieval_time_seconds": result.retrieval_time,
                "generation_time_seconds": result.generation_time,
                "total_time_seconds": result.total_time,
                "sources_count": len(result.sources),
                "conversation_id": request.conversation_id,
                "filters_applied": request.context_filters or {}
            }
        )
    
    except RAGServiceError as e:
        logger.error(f"RAG service error: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"RAG query failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in RAG query endpoint: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query failed: {str(e)}"
        )


@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(request: DocumentSearchRequest, http_request: Request):
    """Search documents by query or metadata filters."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        logger.info(f"Document search: {request.query or 'metadata only'}", extra={"request_id": request_id})
        
        # Ensure RAG service is initialized
        if not rag_service.initialized:
            if not await rag_service.initialize():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="RAG service initialization failed"
                )
        
        # Execute search
        results = await rag_service.search_documents(
            query=request.query,
            filters=request.filters,
            limit=request.limit
        )
        
        return DocumentSearchResponse(
            success=True,
            results=results,
            total_results=len(results),
            metadata={
                "query": request.query,
                "filters": request.filters or {},
                "limit": request.limit
            }
        )
    
    except Exception as e:
        logger.error(f"Error in document search endpoint: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document search failed: {str(e)}"
        )


@router.get("/stats", response_model=RAGStatsResponse)
async def get_rag_stats(http_request: Request):
    """Get RAG system statistics."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        stats = await rag_service.get_document_stats()
        
        return RAGStatsResponse(
            success=True,
            stats=stats
        )
    
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get RAG stats: {str(e)}"
        )


@router.delete("/cache")
async def clear_cache(http_request: Request):
    """Clear RAG query cache."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        rag_service.clear_cache()
        
        logger.info("RAG cache cleared", extra={"request_id": request_id})
        
        return {
            "success": True,
            "message": "Cache cleared successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str, http_request: Request):
    """Delete a document and all its chunks."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        # Ensure RAG service is initialized
        if not rag_service.initialized:
            if not await rag_service.initialize():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="RAG service initialization failed"
                )
        
        success = await rag_service.delete_document(document_id)
        
        if success:
            logger.info(f"Document deleted: {document_id}", extra={"request_id": request_id})
            return {
                "success": True,
                "message": f"Document {document_id} deleted successfully",
                "document_id": document_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found or could not be deleted"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.get("/health", response_model=RAGHealthResponse)
async def health_check():
    """Check RAG service health."""
    try:
        health_info = await rag_service.health_check()
        
        return RAGHealthResponse(
            healthy=health_info.get("healthy", False),
            initialized=health_info.get("initialized", False),
            components=health_info.get("components", {}),
            end_to_end_test=health_info.get("end_to_end_test", {}),
            cache_stats=health_info.get("cache_stats", {}),
            total_check_time_seconds=health_info.get("total_check_time_seconds"),
            timestamp=health_info.get("timestamp", datetime.utcnow().isoformat())
        )
    
    except Exception as e:
        logger.error(f"Error in RAG health check: {e}")
        return RAGHealthResponse(
            healthy=False,
            initialized=False,
            components={},
            end_to_end_test={},
            cache_stats={},
            total_check_time_seconds=None,
            timestamp=datetime.utcnow().isoformat()
        )


@router.post("/initialize")
async def initialize_rag_service(http_request: Request):
    """Initialize or reinitialize the RAG service."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        logger.info("Initializing RAG service...", extra={"request_id": request_id})
        
        success = await rag_service.initialize()
        
        if success:
            logger.info("RAG service initialized successfully", extra={"request_id": request_id})
            return {
                "success": True,
                "message": "RAG service initialized successfully",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="RAG service initialization failed"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initializing RAG service: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize RAG service: {str(e)}"
        )


# Utility endpoints for debugging and monitoring
@router.get("/collections/info")
async def get_collection_info(http_request: Request):
    """Get information about the vector store collection."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        from ai.rag.vector_store import vector_store
        
        if not vector_store.collection:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector store not initialized"
            )
        
        stats = vector_store.get_collection_stats()
        
        return {
            "success": True,
            "collection_info": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collection info: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get collection info: {str(e)}"
        )


@router.get("/embeddings/info")
async def get_embedding_info(http_request: Request):
    """Get information about the embedding service."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        from ai.rag.embedding_service import embedding_service
        
        model_info = embedding_service.get_model_info()
        cache_stats = embedding_service.get_cache_stats()
        
        return {
            "success": True,
            "embedding_info": {
                "model_info": model_info,
                "cache_stats": cache_stats
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting embedding info: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get embedding info: {str(e)}"
        )


@router.post("/embeddings/cache/clear")
async def clear_embedding_cache(http_request: Request):
    """Clear the embedding cache."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        from ai.rag.embedding_service import embedding_service
        
        embedding_service.clear_cache()
        
        logger.info("Embedding cache cleared", extra={"request_id": request_id})
        
        return {
            "success": True,
            "message": "Embedding cache cleared successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error clearing embedding cache: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear embedding cache: {str(e)}"
        )