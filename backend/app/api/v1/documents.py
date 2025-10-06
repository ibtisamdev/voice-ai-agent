"""
Document management API endpoints for file upload, processing, and ingestion.
"""

import logging
import tempfile
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import asyncio
import uuid

from fastapi import APIRouter, HTTPException, status, Depends, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import shutil

from ai.rag.rag_service import rag_service, DocumentIngestionResult, RAGServiceError
from ai.rag.document_processor import document_processor, DocumentProcessingError
from ai.rag.chunking_strategies import ChunkingStrategy
from backend.app.core.config import settings
from backend.app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# Request/Response Models
class DocumentProcessRequest(BaseModel):
    """Request model for document processing."""
    file_path: str = Field(..., description="Path to the document file")
    document_type: Optional[str] = Field(None, description="Type of document")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    chunking_strategy: str = Field("hybrid", description="Chunking strategy to use")
    max_chunk_size: Optional[int] = Field(None, description="Maximum chunk size")
    chunk_overlap: Optional[int] = Field(None, description="Chunk overlap size")
    
    @validator('chunking_strategy')
    def validate_chunking_strategy(cls, v):
        valid_strategies = [e.value for e in ChunkingStrategy]
        if v not in valid_strategies:
            raise ValueError(f'Chunking strategy must be one of: {", ".join(valid_strategies)}')
        return v


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    success: bool
    file_id: Optional[str] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class DocumentProcessResponse(BaseModel):
    """Response model for document processing."""
    success: bool
    document_id: Optional[str] = None
    chunks_created: int = 0
    processing_time_seconds: Optional[float] = None
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class DocumentIngestionResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool
    document_id: Optional[str] = None
    chunks_created: int = 0
    processing_time_seconds: Optional[float] = None
    metadata: Dict[str, Any] = {}
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class BatchProcessRequest(BaseModel):
    """Request model for batch processing."""
    file_paths: List[str] = Field(..., description="List of file paths to process")
    document_type: Optional[str] = Field(None, description="Default document type")
    chunking_strategy: str = Field("hybrid", description="Chunking strategy to use")
    batch_size: int = Field(5, ge=1, le=20, description="Batch size for processing")
    
    @validator('chunking_strategy')
    def validate_chunking_strategy(cls, v):
        valid_strategies = [e.value for e in ChunkingStrategy]
        if v not in valid_strategies:
            raise ValueError(f'Chunking strategy must be one of: {", ".join(valid_strategies)}')
        return v


class BatchProcessResponse(BaseModel):
    """Response model for batch processing."""
    success: bool
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    results: List[Dict[str, Any]] = []
    total_processing_time_seconds: Optional[float] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# Helper functions
def validate_file_type(filename: str) -> bool:
    """Validate if file type is supported."""
    supported_extensions = settings.SUPPORTED_FORMATS.split(",")
    file_extension = Path(filename).suffix.lower().lstrip('.')
    return file_extension in supported_extensions


def get_file_size_mb(file_size: int) -> float:
    """Convert file size to MB."""
    return file_size / (1024 * 1024)


def parse_max_size(max_size_str: str) -> int:
    """Parse max size string (e.g., '50MB') to bytes."""
    if max_size_str.upper().endswith('MB'):
        return int(max_size_str[:-2]) * 1024 * 1024
    elif max_size_str.upper().endswith('KB'):
        return int(max_size_str[:-2]) * 1024
    else:
        return int(max_size_str)


# API Endpoints
@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_type: Optional[str] = Form(None),
    http_request: Request = None
):
    """Upload a document file for processing."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        logger.info(f"Document upload: {file.filename}", extra={"request_id": request_id})
        
        # Validate file type
        if not validate_file_type(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Supported formats: {settings.SUPPORTED_FORMATS}"
            )
        
        # Check file size
        max_size_bytes = parse_max_size(settings.UPLOAD_MAX_SIZE)
        file_size = 0
        
        # Create a temporary file
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        temp_file_path = f"/tmp/voiceai_upload_{file_id}{file_extension}"
        
        try:
            with open(temp_file_path, "wb") as temp_file:
                while chunk := await file.read(8192):  # Read in 8KB chunks
                    file_size += len(chunk)
                    
                    # Check size limit
                    if file_size > max_size_bytes:
                        temp_file.close()
                        os.unlink(temp_file_path)
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"File too large. Maximum size: {settings.UPLOAD_MAX_SIZE}"
                        )
                    
                    temp_file.write(chunk)
        
        except HTTPException:
            raise
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"File upload failed: {str(e)}"
            )
        
        logger.info(f"File uploaded successfully: {file_size} bytes", extra={"request_id": request_id})
        
        return DocumentUploadResponse(
            success=True,
            file_id=file_id,
            file_path=temp_file_path,
            file_size=file_size,
            file_type=file_extension.lstrip('.'),
            timestamp=datetime.utcnow().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.post("/process", response_model=DocumentProcessResponse)
async def process_document(request: DocumentProcessRequest, http_request: Request):
    """Process a document file without ingesting into RAG system."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        logger.info(f"Document processing: {request.file_path}", extra={"request_id": request_id})
        
        # Check if file exists
        if not os.path.exists(request.file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {request.file_path}"
            )
        
        start_time = datetime.utcnow()
        
        # Process the document
        processed_doc = await document_processor.process_file(
            file_path=request.file_path,
            document_type=request.document_type,
            metadata=request.metadata
        )
        
        # Chunk the document
        from ai.rag.chunking_strategies import document_chunker
        
        chunking_strategy = ChunkingStrategy(request.chunking_strategy)
        chunks = document_chunker.chunk_document(
            text=processed_doc["text"],
            strategy=chunking_strategy,
            max_chunk_size=request.max_chunk_size,
            overlap=request.chunk_overlap,
            preserve_structure=True,
            source_document=request.file_path
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        result_metadata = {
            **processed_doc["metadata"],
            "document_id": document_id,
            "chunks_created": len(chunks),
            "chunking_strategy": request.chunking_strategy,
            "processing_time_seconds": processing_time
        }
        
        logger.info(f"Document processed: {len(chunks)} chunks in {processing_time:.2f}s", 
                   extra={"request_id": request_id})
        
        return DocumentProcessResponse(
            success=True,
            document_id=document_id,
            chunks_created=len(chunks),
            processing_time_seconds=processing_time,
            metadata=result_metadata
        )
    
    except DocumentProcessingError as e:
        logger.error(f"Document processing error: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document processing failed: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in process endpoint: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )


@router.post("/ingest", response_model=DocumentIngestionResponse)
async def ingest_document(request: DocumentProcessRequest, http_request: Request):
    """Process and ingest a document into the RAG system."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        logger.info(f"Document ingestion: {request.file_path}", extra={"request_id": request_id})
        
        # Ensure RAG service is initialized
        if not rag_service.initialized:
            logger.info("RAG service not initialized, initializing now...")
            if not await rag_service.initialize():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="RAG service initialization failed"
                )
        
        # Check if file exists
        if not os.path.exists(request.file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {request.file_path}"
            )
        
        # Ingest document
        chunking_strategy = ChunkingStrategy(request.chunking_strategy)
        result = await rag_service.ingest_document(
            file_path=request.file_path,
            document_type=request.document_type,
            metadata=request.metadata,
            chunking_strategy=chunking_strategy,
            max_chunk_size=request.max_chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        if result.success:
            logger.info(f"Document ingested successfully: {result.chunks_created} chunks", 
                       extra={"request_id": request_id})
            
            return DocumentIngestionResponse(
                success=True,
                document_id=result.document_id,
                chunks_created=result.chunks_created,
                processing_time_seconds=result.metadata.get("processing_time_seconds"),
                metadata=result.metadata
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document ingestion failed: {result.error}"
            )
    
    except RAGServiceError as e:
        logger.error(f"RAG service error: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document ingestion failed: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ingest endpoint: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}"
        )


@router.post("/upload-and-ingest", response_model=DocumentIngestionResponse)
async def upload_and_ingest_document(
    file: UploadFile = File(...),
    document_type: Optional[str] = Form(None),
    chunking_strategy: str = Form("hybrid"),
    max_chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    http_request: Request = None
):
    """Upload and immediately ingest a document into the RAG system."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        logger.info(f"Upload and ingest: {file.filename}", extra={"request_id": request_id})
        
        # First upload the file
        upload_response = await upload_document(file, document_type, http_request)
        
        if not upload_response.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File upload failed"
            )
        
        try:
            # Then ingest it
            ingest_request = DocumentProcessRequest(
                file_path=upload_response.file_path,
                document_type=document_type,
                chunking_strategy=chunking_strategy,
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            result = await ingest_document(ingest_request, http_request)
            
            # Clean up uploaded file
            if os.path.exists(upload_response.file_path):
                os.unlink(upload_response.file_path)
            
            return result
            
        except Exception as e:
            # Clean up uploaded file on error
            if upload_response.file_path and os.path.exists(upload_response.file_path):
                os.unlink(upload_response.file_path)
            raise e
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload-and-ingest endpoint: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload and ingest failed: {str(e)}"
        )


@router.post("/batch-process", response_model=BatchProcessResponse)
async def batch_process_documents(request: BatchProcessRequest, http_request: Request):
    """Process multiple documents in batch."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        logger.info(f"Batch processing: {len(request.file_paths)} files", extra={"request_id": request_id})
        
        # Ensure RAG service is initialized for ingestion
        if not rag_service.initialized:
            if not await rag_service.initialize():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="RAG service initialization failed"
                )
        
        start_time = datetime.utcnow()
        results = []
        successful_files = 0
        failed_files = 0
        
        # Process files in batches
        for i in range(0, len(request.file_paths), request.batch_size):
            batch_files = request.file_paths[i:i + request.batch_size]
            batch_tasks = []
            
            for file_path in batch_files:
                if not os.path.exists(file_path):
                    results.append({
                        "file_path": file_path,
                        "success": False,
                        "error": "File not found"
                    })
                    failed_files += 1
                    continue
                
                # Create ingestion task
                chunking_strategy = ChunkingStrategy(request.chunking_strategy)
                task = rag_service.ingest_document(
                    file_path=file_path,
                    document_type=request.document_type,
                    chunking_strategy=chunking_strategy
                )
                batch_tasks.append((file_path, task))
            
            # Execute batch
            if batch_tasks:
                batch_results = await asyncio.gather(
                    *[task for _, task in batch_tasks],
                    return_exceptions=True
                )
                
                for (file_path, _), result in zip(batch_tasks, batch_results):
                    if isinstance(result, Exception):
                        results.append({
                            "file_path": file_path,
                            "success": False,
                            "error": str(result)
                        })
                        failed_files += 1
                    else:
                        results.append({
                            "file_path": file_path,
                            "success": result.success,
                            "document_id": result.document_id if result.success else None,
                            "chunks_created": result.chunks_created,
                            "error": result.error if not result.success else None
                        })
                        if result.success:
                            successful_files += 1
                        else:
                            failed_files += 1
        
        total_processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Batch processing completed: {successful_files} successful, {failed_files} failed", 
                   extra={"request_id": request_id})
        
        return BatchProcessResponse(
            success=True,
            total_files=len(request.file_paths),
            successful_files=successful_files,
            failed_files=failed_files,
            results=results,
            total_processing_time_seconds=total_processing_time
        )
    
    except Exception as e:
        logger.error(f"Error in batch processing endpoint: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )


@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats."""
    return {
        "supported_formats": settings.SUPPORTED_FORMATS.split(","),
        "max_file_size": settings.UPLOAD_MAX_SIZE,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/processing-estimate")
async def estimate_processing_time(file_path: str):
    """Estimate processing time for a file."""
    try:
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        estimated_time = document_processor.estimate_processing_time(file_path)
        
        return {
            "file_path": file_path,
            "estimated_time_seconds": estimated_time,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to estimate processing time: {str(e)}"
        )


@router.delete("/cleanup-temp")
async def cleanup_temp_files(http_request: Request):
    """Clean up temporary uploaded files."""
    try:
        request_id = getattr(http_request.state, 'request_id', 'unknown')
        
        temp_dir = "/tmp"
        cleaned_files = 0
        
        for file_path in Path(temp_dir).glob("voiceai_upload_*"):
            try:
                if file_path.is_file():
                    file_path.unlink()
                    cleaned_files += 1
            except Exception as e:
                logger.warning(f"Failed to delete temp file {file_path}: {e}")
        
        logger.info(f"Cleaned up {cleaned_files} temporary files", extra={"request_id": request_id})
        
        return {
            "success": True,
            "files_cleaned": cleaned_files,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {e}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cleanup failed: {str(e)}"
        )