"""
Unit tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import tempfile
import os

from backend.app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_file():
    """Create a sample file for upload testing."""
    content = "Sample legal document content for testing."
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert "health" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data


class TestLLMEndpoints:
    """Test LLM API endpoints."""
    
    @patch('ai.llm.llm_service.llm_service')
    def test_generate_text(self, mock_llm_service, client):
        """Test text generation endpoint."""
        # Mock LLM response
        async def mock_generate():
            yield "This is a test response."
        
        mock_llm_service.generate_response = AsyncMock(return_value=mock_generate())
        
        request_data = {
            "prompt": "What is contract law?",
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        response = client.post("/api/v1/llm/generate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "response" in data
    
    @patch('ai.llm.llm_service.llm_service')
    def test_chat_completion(self, mock_llm_service, client):
        """Test chat completion endpoint."""
        mock_llm_service.chat_completion = AsyncMock(return_value="This is a chat response.")
        
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        response = client.post("/api/v1/llm/chat", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "response" in data
    
    @patch('ai.llm.llm_service.llm_service')
    def test_analyze_document(self, mock_llm_service, client):
        """Test document analysis endpoint."""
        mock_llm_service.analyze_document = AsyncMock(return_value="Document analysis result.")
        
        request_data = {
            "document_text": "Sample contract text",
            "analysis_type": "contract"
        }
        
        response = client.post("/api/v1/llm/analyze-document", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "response" in data
    
    def test_get_models(self, client):
        """Test get models endpoint."""
        with patch('ai.llm.llm_service.llm_service') as mock_llm_service:
            mock_llm_service.get_available_models = AsyncMock(return_value=["llama2:7b-chat"])
            mock_llm_service.current_model = "llama2:7b-chat"
            
            response = client.get("/api/v1/llm/models")
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert "metadata" in data
    
    def test_invalid_chat_message_role(self, client):
        """Test chat endpoint with invalid role."""
        request_data = {
            "messages": [
                {"role": "invalid_role", "content": "Hello"}
            ]
        }
        
        response = client.post("/api/v1/llm/chat", json=request_data)
        assert response.status_code == 422  # Validation error


class TestRAGEndpoints:
    """Test RAG API endpoints."""
    
    @patch('ai.rag.rag_service.rag_service')
    def test_query_documents(self, mock_rag_service, client):
        """Test RAG query endpoint."""
        # Mock RAG response
        mock_result = MagicMock()
        mock_result.query = "What is contract law?"
        mock_result.answer = "Contract law is..."
        mock_result.sources = []
        mock_result.retrieval_time = 0.1
        mock_result.generation_time = 0.5
        mock_result.total_time = 0.6
        
        mock_rag_service.initialized = True
        mock_rag_service.query = AsyncMock(return_value=mock_result)
        
        request_data = {
            "question": "What is contract law?",
            "max_results": 5
        }
        
        response = client.post("/api/v1/rag/query", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["query"] == "What is contract law?"
        assert "answer" in data
    
    @patch('ai.rag.rag_service.rag_service')
    def test_search_documents(self, mock_rag_service, client):
        """Test document search endpoint."""
        mock_rag_service.initialized = True
        mock_rag_service.search_documents = AsyncMock(return_value=[
            {"id": "1", "document": "Sample document", "metadata": {}}
        ])
        
        request_data = {
            "query": "contract",
            "limit": 10
        }
        
        response = client.post("/api/v1/rag/search", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 1
    
    @patch('ai.rag.rag_service.rag_service')
    def test_delete_document(self, mock_rag_service, client):
        """Test document deletion endpoint."""
        mock_rag_service.initialized = True
        mock_rag_service.delete_document = AsyncMock(return_value=True)
        
        response = client.delete("/api/v1/rag/documents/test-doc-id")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
    
    @patch('ai.rag.rag_service.rag_service')
    def test_clear_cache(self, mock_rag_service, client):
        """Test cache clearing endpoint."""
        mock_rag_service.clear_cache = MagicMock()
        
        response = client.delete("/api/v1/rag/cache")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True


class TestDocumentEndpoints:
    """Test document management endpoints."""
    
    def test_upload_document(self, client, sample_file):
        """Test document upload endpoint."""
        with open(sample_file, 'rb') as f:
            files = {"file": ("test.txt", f, "text/plain")}
            response = client.post("/api/v1/documents/upload", files=files)
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "file_id" in data
        assert "file_path" in data
    
    def test_upload_unsupported_format(self, client):
        """Test uploading unsupported file format."""
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as f:
                files = {"file": ("test.xyz", f, "application/octet-stream")}
                response = client.post("/api/v1/documents/upload", files=files)
            
            assert response.status_code == 400
        finally:
            os.unlink(temp_path)
    
    @patch('ai.rag.document_processor.document_processor')
    @patch('ai.rag.chunking_strategies.document_chunker')
    def test_process_document(self, mock_chunker, mock_processor, client):
        """Test document processing endpoint."""
        # Mock processor response
        mock_processor.process_file = AsyncMock(return_value={
            "text": "Sample document text",
            "metadata": {"document_type": "contract"},
            "document_type": "contract",
            "source": "/tmp/test.txt"
        })
        
        # Mock chunker response
        mock_chunk = MagicMock()
        mock_chunk.text = "Sample chunk"
        mock_chunk.metadata = MagicMock()
        mock_chunk.metadata.chunk_index = 0
        mock_chunk.metadata.strategy = "hybrid"
        mock_chunk.metadata.token_count = 10
        mock_chunk.metadata.start_char = 0
        mock_chunk.metadata.end_char = 12
        mock_chunk.metadata.section_type = None
        mock_chunk.metadata.legal_citations = None
        
        mock_chunker.chunk_document = MagicMock(return_value=[mock_chunk])
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            request_data = {
                "file_path": temp_path,
                "document_type": "contract",
                "chunking_strategy": "hybrid"
            }
            
            response = client.post("/api/v1/documents/process", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert data["chunks_created"] == 1
        finally:
            os.unlink(temp_path)
    
    def test_process_nonexistent_file(self, client):
        """Test processing non-existent file."""
        request_data = {
            "file_path": "/nonexistent/file.txt",
            "chunking_strategy": "hybrid"
        }
        
        response = client.post("/api/v1/documents/process", json=request_data)
        assert response.status_code == 404
    
    def test_get_supported_formats(self, client):
        """Test get supported formats endpoint."""
        response = client.get("/api/v1/documents/supported-formats")
        assert response.status_code == 200
        
        data = response.json()
        assert "supported_formats" in data
        assert "max_file_size" in data
    
    def test_invalid_chunking_strategy(self, client):
        """Test processing with invalid chunking strategy."""
        request_data = {
            "file_path": "/tmp/test.txt",
            "chunking_strategy": "invalid_strategy"
        }
        
        response = client.post("/api/v1/documents/process", json=request_data)
        assert response.status_code == 422  # Validation error


class TestErrorHandling:
    """Test error handling across endpoints."""
    
    def test_llm_service_error(self, client):
        """Test LLM service error handling."""
        with patch('ai.llm.llm_service.llm_service') as mock_llm_service:
            mock_llm_service.generate_response = AsyncMock(side_effect=Exception("LLM error"))
            
            request_data = {
                "prompt": "Test prompt"
            }
            
            response = client.post("/api/v1/llm/generate", json=request_data)
            assert response.status_code == 500
    
    def test_rag_service_error(self, client):
        """Test RAG service error handling."""
        with patch('ai.rag.rag_service.rag_service') as mock_rag_service:
            mock_rag_service.initialized = True
            mock_rag_service.query = AsyncMock(side_effect=Exception("RAG error"))
            
            request_data = {
                "question": "Test question"
            }
            
            response = client.post("/api/v1/rag/query", json=request_data)
            assert response.status_code == 500
    
    def test_missing_required_fields(self, client):
        """Test validation of required fields."""
        # Empty prompt
        response = client.post("/api/v1/llm/generate", json={})
        assert response.status_code == 422
        
        # Empty question
        response = client.post("/api/v1/rag/query", json={})
        assert response.status_code == 422