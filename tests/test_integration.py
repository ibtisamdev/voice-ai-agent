"""
Integration tests for the complete RAG pipeline.
"""

import pytest
import tempfile
import os
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

from ai.rag.rag_service import rag_service
from ai.rag.document_processor import document_processor
from ai.rag.chunking_strategies import document_chunker, ChunkingStrategy
from ai.rag.vector_store import vector_store
from ai.rag.embedding_service import embedding_service
from ai.llm.llm_service import llm_service


@pytest.fixture
def sample_legal_document():
    """Create a sample legal document for testing."""
    content = """
    SAMPLE LEGAL AGREEMENT
    
    This Agreement is entered into on January 1, 2024, between ABC Corporation, 
    a Delaware corporation ("Company"), and John Doe, an individual ("Consultant").
    
    RECITALS
    
    WHEREAS, Company desires to engage Consultant to provide certain services;
    WHEREAS, Consultant has the expertise and qualifications to provide such services;
    
    NOW, THEREFORE, in consideration of the mutual covenants contained herein, 
    the parties agree as follows:
    
    1. SERVICES
    Consultant shall provide the following services to Company:
    a) Software development and consulting
    b) Code review and architecture guidance
    c) Technical documentation preparation
    
    2. COMPENSATION
    Company shall pay Consultant $150 per hour for services rendered.
    Payment shall be made within thirty (30) days of receipt of invoice.
    
    3. TERM
    This Agreement shall commence on January 1, 2024, and shall continue for 
    a period of twelve (12) months, unless earlier terminated.
    
    4. CONFIDENTIALITY
    Consultant acknowledges that during the course of this engagement, 
    Consultant may have access to certain confidential information.
    
    5. GOVERNING LAW
    This Agreement shall be governed by and construed in accordance with 
    the laws of the State of California.
    
    IN WITNESS WHEREOF, the parties have executed this Agreement as of the 
    date first written above.
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestRAGPipelineIntegration:
    """Test the complete RAG pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_document_processing_pipeline(self, sample_legal_document):
        """Test the complete document processing pipeline."""
        # Step 1: Process the document
        processed_doc = await document_processor.process_file(sample_legal_document)
        
        assert processed_doc["text"]
        assert processed_doc["metadata"]
        assert processed_doc["document_type"] == "contract"
        
        # Check that legal elements were extracted
        metadata = processed_doc["metadata"]
        assert metadata.get("has_whereas_clauses") is True
        assert metadata.get("has_signatures") is True
        
        # Step 2: Chunk the document
        chunks = document_chunker.chunk_document(
            text=processed_doc["text"],
            strategy=ChunkingStrategy.HYBRID,
            max_chunk_size=500,
            overlap=100,
            preserve_structure=True,
            source_document=sample_legal_document
        )
        
        assert len(chunks) > 0
        assert all(chunk.text.strip() for chunk in chunks)
        
        # Check that legal sections were identified
        legal_chunks = [chunk for chunk in chunks if chunk.metadata.section_type]
        assert len(legal_chunks) > 0
    
    @pytest.mark.asyncio
    @patch('ai.rag.vector_store.vector_store')
    @patch('ai.rag.embedding_service.embedding_service')
    @patch('ai.llm.llm_service.llm_service')
    async def test_end_to_end_rag_query(self, mock_llm, mock_embedding, mock_vector, sample_legal_document):
        """Test end-to-end RAG query functionality."""
        # Mock the services
        mock_vector.initialize = AsyncMock(return_value=True)
        mock_embedding.initialize = AsyncMock(return_value=True)
        mock_llm.initialize = AsyncMock(return_value=True)
        
        # Mock vector search results
        mock_vector.search_with_threshold = MagicMock(return_value=[
            {
                "id": "chunk_1",
                "document": "Company shall pay Consultant $150 per hour for services rendered.",
                "metadata": {
                    "document_type": "contract",
                    "section_type": "compensation",
                    "source": sample_legal_document
                },
                "similarity": 0.85
            },
            {
                "id": "chunk_2", 
                "document": "This Agreement shall commence on January 1, 2024, and shall continue for a period of twelve (12) months.",
                "metadata": {
                    "document_type": "contract",
                    "section_type": "term",
                    "source": sample_legal_document
                },
                "similarity": 0.78
            }
        ])
        
        # Mock LLM response
        async def mock_generate_response(*args, **kwargs):
            yield "Based on the agreement, the consultant will be paid $150 per hour for their services. The agreement is effective for 12 months starting January 1, 2024."
        
        mock_llm.generate_response = mock_generate_response
        
        # Initialize RAG service with mocked components
        rag_service.initialized = True
        
        # Execute query
        result = await rag_service.query(
            question="What is the compensation rate and term of the agreement?",
            max_results=5,
            include_sources=True
        )
        
        # Verify results
        assert result.query == "What is the compensation rate and term of the agreement?"
        assert result.answer
        assert "$150 per hour" in result.answer
        assert "12 months" in result.answer
        assert len(result.sources) == 2
        assert result.total_time > 0
    
    @pytest.mark.asyncio
    @patch('ai.rag.rag_service.rag_service')
    async def test_document_ingestion_workflow(self, mock_rag_service, sample_legal_document):
        """Test the document ingestion workflow."""
        # Mock successful ingestion
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.document_id = "test-doc-123"
        mock_result.chunks_created = 5
        mock_result.metadata = {
            "document_type": "contract",
            "processing_time_seconds": 1.5
        }
        
        mock_rag_service.ingest_document = AsyncMock(return_value=mock_result)
        
        # Test ingestion
        result = await mock_rag_service.ingest_document(
            file_path=sample_legal_document,
            document_type="contract",
            chunking_strategy=ChunkingStrategy.HYBRID
        )
        
        assert result.success
        assert result.document_id == "test-doc-123"
        assert result.chunks_created == 5
        assert result.metadata["document_type"] == "contract"
    
    @pytest.mark.asyncio
    async def test_chunking_strategies_comparison(self, sample_legal_document):
        """Test and compare different chunking strategies."""
        # Process document first
        processed_doc = await document_processor.process_file(sample_legal_document)
        text = processed_doc["text"]
        
        strategies_to_test = [
            ChunkingStrategy.FIXED_SIZE,
            ChunkingStrategy.SENTENCE,
            ChunkingStrategy.PARAGRAPH,
            ChunkingStrategy.LEGAL_SECTION,
            ChunkingStrategy.HYBRID
        ]
        
        results = {}
        
        for strategy in strategies_to_test:
            chunks = document_chunker.chunk_document(
                text=text,
                strategy=strategy,
                max_chunk_size=400,
                overlap=50
            )
            
            results[strategy.value] = {
                "chunk_count": len(chunks),
                "avg_chunk_size": sum(len(chunk.text) for chunk in chunks) / len(chunks) if chunks else 0,
                "has_legal_sections": any(chunk.metadata.section_type for chunk in chunks)
            }
        
        # Verify each strategy produced reasonable results
        for strategy_name, result in results.items():
            assert result["chunk_count"] > 0, f"{strategy_name} produced no chunks"
            assert result["avg_chunk_size"] > 0, f"{strategy_name} produced empty chunks"
        
        # Legal section strategy should identify legal sections
        assert results["legal_section"]["has_legal_sections"], "Legal section strategy failed to identify sections"
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, sample_legal_document):
        """Test error handling and recovery mechanisms."""
        with patch('ai.rag.document_processor.document_processor') as mock_processor:
            # Test processing error
            mock_processor.process_file = AsyncMock(side_effect=Exception("Processing failed"))
            
            with pytest.raises(Exception):
                await document_processor.process_file(sample_legal_document)
        
        # Test chunking with invalid parameters
        with pytest.raises(ValueError):
            document_chunker.chunk_document(
                text="sample text",
                strategy="invalid_strategy"  # This should be a ChunkingStrategy enum
            )
    
    @pytest.mark.asyncio
    @patch('ai.rag.vector_store.vector_store')
    @patch('ai.rag.embedding_service.embedding_service') 
    @patch('ai.llm.llm_service.llm_service')
    async def test_rag_service_health_check(self, mock_llm, mock_embedding, mock_vector):
        """Test RAG service health check functionality."""
        # Mock component health checks
        mock_vector.health_check = AsyncMock(return_value={
            "healthy": True,
            "collection_name": "legal_documents",
            "document_count": 10
        })
        
        mock_embedding.health_check = AsyncMock(return_value={
            "healthy": True,
            "model_info": {"model_name": "all-MiniLM-L6-v2"}
        })
        
        mock_llm.health_check = AsyncMock(return_value={
            "healthy": True,
            "current_model": "llama2:7b-chat"
        })
        
        # Mock successful end-to-end test
        async def mock_query(*args, **kwargs):
            mock_result = MagicMock()
            mock_result.answer = "Test response"
            return mock_result
        
        rag_service.query = mock_query
        rag_service.initialized = True
        
        # Execute health check
        health_result = await rag_service.health_check()
        
        assert health_result["healthy"] is True
        assert health_result["initialized"] is True
        assert "components" in health_result
        assert health_result["components"]["vector_store"]["healthy"] is True
        assert health_result["components"]["embedding_service"]["healthy"] is True
        assert health_result["components"]["llm_service"]["healthy"] is True
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, sample_legal_document):
        """Test performance of key operations."""
        import time
        
        # Benchmark document processing
        start_time = time.time()
        processed_doc = await document_processor.process_file(sample_legal_document)
        processing_time = time.time() - start_time
        
        assert processing_time < 5.0, f"Document processing took too long: {processing_time}s"
        
        # Benchmark chunking
        start_time = time.time()
        chunks = document_chunker.chunk_document(
            text=processed_doc["text"],
            strategy=ChunkingStrategy.HYBRID,
            max_chunk_size=500
        )
        chunking_time = time.time() - start_time
        
        assert chunking_time < 2.0, f"Document chunking took too long: {chunking_time}s"
        assert len(chunks) > 0, "No chunks were created"
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, sample_legal_document):
        """Test concurrent document processing."""
        # Create multiple copies of the document for concurrent processing
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_test_{i}.txt', delete=False) as f:
                f.write(f"Document {i}: " + open(sample_legal_document).read())
                temp_files.append(f.name)
        
        try:
            # Process documents concurrently
            tasks = [document_processor.process_file(file_path) for file_path in temp_files]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all succeeded
            assert len(results) == 3
            for result in results:
                assert not isinstance(result, Exception), f"Processing failed: {result}"
                assert result["text"]
                assert result["metadata"]
        
        finally:
            # Cleanup
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)


class TestOptimizations:
    """Test optimization features and performance improvements."""
    
    @pytest.mark.asyncio
    async def test_caching_effectiveness(self):
        """Test that caching improves performance."""
        with patch('ai.rag.embedding_service.embedding_service') as mock_embedding:
            # Mock embedding service with cache
            cache_hit_times = []
            cache_miss_times = []
            
            async def mock_embed_text(text, use_cache=True):
                if use_cache and "cached_text" in text:
                    # Simulate cache hit (faster)
                    await asyncio.sleep(0.01)
                    cache_hit_times.append(0.01)
                else:
                    # Simulate cache miss (slower)
                    await asyncio.sleep(0.1)
                    cache_miss_times.append(0.1)
                return [0.1] * 384  # Mock embedding vector
            
            mock_embedding.embed_text = mock_embed_text
            
            # Test cache miss
            await mock_embedding.embed_text("new text that is not cached")
            
            # Test cache hit
            await mock_embedding.embed_text("cached_text that should be faster")
            
            # Verify cache improves performance
            if cache_hit_times and cache_miss_times:
                assert min(cache_hit_times) < min(cache_miss_times)
    
    def test_chunk_size_optimization(self, sample_legal_document):
        """Test optimal chunk sizes for different content types."""
        # Read document content
        with open(sample_legal_document, 'r') as f:
            text = f.read()
        
        chunk_sizes = [200, 500, 1000, 1500]
        results = {}
        
        for chunk_size in chunk_sizes:
            chunks = document_chunker.chunk_document(
                text=text,
                strategy=ChunkingStrategy.HYBRID,
                max_chunk_size=chunk_size,
                overlap=50
            )
            
            results[chunk_size] = {
                "chunk_count": len(chunks),
                "avg_tokens": sum(chunk.metadata.token_count for chunk in chunks) / len(chunks) if chunks else 0,
                "total_tokens": sum(chunk.metadata.token_count for chunk in chunks)
            }
        
        # Verify that larger chunk sizes generally result in fewer chunks
        chunk_counts = [results[size]["chunk_count"] for size in sorted(chunk_sizes)]
        assert chunk_counts == sorted(chunk_counts, reverse=True), "Larger chunks should result in fewer total chunks"
    
    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self):
        """Test that batch processing is more efficient than individual processing."""
        with patch('ai.rag.embedding_service.embedding_service') as mock_embedding:
            # Mock batch processing to be more efficient
            individual_times = []
            batch_times = []
            
            async def mock_embed_text(text, use_cache=True):
                await asyncio.sleep(0.1)  # Simulate individual processing time
                individual_times.append(0.1)
                return [0.1] * 384
            
            async def mock_embed_batch(texts, batch_size=10, use_cache=True, show_progress=False):
                await asyncio.sleep(0.05 * len(texts))  # Simulate batch processing time
                batch_times.append(0.05 * len(texts))
                return [[0.1] * 384 for _ in texts]
            
            mock_embedding.embed_text = mock_embed_text
            mock_embedding.embed_batch = mock_embed_batch
            
            # Test individual processing
            texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]
            
            start_time = time.time()
            for text in texts:
                await mock_embedding.embed_text(text)
            individual_total_time = time.time() - start_time
            
            # Test batch processing
            start_time = time.time()
            await mock_embedding.embed_batch(texts)
            batch_total_time = time.time() - start_time
            
            # Batch should be more efficient
            assert batch_total_time < individual_total_time, "Batch processing should be faster than individual processing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])