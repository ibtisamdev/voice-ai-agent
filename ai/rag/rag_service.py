"""
RAG (Retrieval-Augmented Generation) service.
Combines document retrieval with LLM generation for legal query answering.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import time
from pathlib import Path
import uuid

from .vector_store import vector_store, VectorStoreError
from .embedding_service import embedding_service, EmbeddingServiceError
from .document_processor import document_processor, DocumentProcessingError
from .chunking_strategies import document_chunker, ChunkingStrategy, TextChunk
from ai.llm.llm_service import llm_service
from ai.llm.prompt_templates import LegalPromptTemplates
from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGServiceError(Exception):
    """Raised when RAG operations fail."""
    pass


class DocumentIngestionResult:
    """Result of document ingestion."""
    def __init__(self, success: bool, document_id: str, chunks_created: int, 
                 metadata: Dict[str, Any], error: Optional[str] = None):
        self.success = success
        self.document_id = document_id
        self.chunks_created = chunks_created
        self.metadata = metadata
        self.error = error


class RAGQueryResult:
    """Result of a RAG query."""
    def __init__(self, query: str, answer: str, sources: List[Dict[str, Any]], 
                 retrieval_time: float, generation_time: float, total_time: float):
        self.query = query
        self.answer = answer
        self.sources = sources
        self.retrieval_time = retrieval_time
        self.generation_time = generation_time
        self.total_time = total_time


class RAGService:
    """RAG service for legal document question answering."""
    
    def __init__(self):
        self.initialized = False
        self.query_cache: Dict[str, RAGQueryResult] = {}
        self.cache_ttl = 3600  # 1 hour cache TTL
    
    async def initialize(self) -> bool:
        """Initialize all RAG components."""
        try:
            logger.info("Initializing RAG service...")
            
            # Initialize vector store
            if not await vector_store.initialize():
                raise RAGServiceError("Failed to initialize vector store")
            
            # Initialize embedding service
            if not await embedding_service.initialize():
                raise RAGServiceError("Failed to initialize embedding service")
            
            # Initialize LLM service
            if not await llm_service.initialize():
                raise RAGServiceError("Failed to initialize LLM service")
            
            self.initialized = True
            logger.info("RAG service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            return False
    
    async def ingest_document(
        self,
        file_path: str,
        document_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
        max_chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> DocumentIngestionResult:
        """Ingest a document into the RAG system."""
        if not self.initialized:
            raise RAGServiceError("RAG service not initialized")
        
        document_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"Starting document ingestion: {file_path}")
            
            # Process the document
            processed_doc = await document_processor.process_file(
                file_path=file_path,
                document_type=document_type,
                metadata=metadata
            )
            
            # Chunk the document
            chunks = document_chunker.chunk_document(
                text=processed_doc["text"],
                strategy=chunking_strategy,
                max_chunk_size=max_chunk_size or settings.MAX_CHUNK_SIZE,
                overlap=chunk_overlap or settings.CHUNK_OVERLAP,
                preserve_structure=True,
                source_document=file_path
            )
            
            # Prepare chunk data for vector store
            chunk_texts = []
            chunk_metadatas = []
            chunk_ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk.text)
                
                # Combine document metadata with chunk metadata
                chunk_metadata = {
                    **processed_doc["metadata"],
                    "chunk_id": chunk_id,
                    "chunk_index": chunk.metadata.chunk_index,
                    "total_chunks": len(chunks),
                    "chunking_strategy": chunk.metadata.strategy,
                    "document_id": document_id,
                    "token_count": chunk.metadata.token_count,
                    "start_char": chunk.metadata.start_char,
                    "end_char": chunk.metadata.end_char
                }
                
                if chunk.metadata.section_type:
                    chunk_metadata["section_type"] = chunk.metadata.section_type
                
                if chunk.metadata.legal_citations:
                    chunk_metadata["legal_citations"] = chunk.metadata.legal_citations
                
                chunk_metadatas.append(chunk_metadata)
            
            # Store chunks in vector database
            stored_ids = vector_store.add_documents(
                documents=chunk_texts,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
            
            processing_time = time.time() - start_time
            
            result_metadata = {
                **processed_doc["metadata"],
                "document_id": document_id,
                "processing_time_seconds": processing_time,
                "ingestion_timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Document ingested successfully: {len(stored_ids)} chunks in {processing_time:.2f}s")
            
            return DocumentIngestionResult(
                success=True,
                document_id=document_id,
                chunks_created=len(stored_ids),
                metadata=result_metadata
            )
            
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {e}")
            return DocumentIngestionResult(
                success=False,
                document_id=document_id,
                chunks_created=0,
                metadata={},
                error=str(e)
            )
    
    async def query(
        self,
        question: str,
        context_filters: Optional[Dict[str, Any]] = None,
        max_results: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        include_sources: bool = True,
        conversation_id: Optional[str] = None,
        system_prompt_type: str = "legal_assistant"
    ) -> RAGQueryResult:
        """Query the RAG system with a question."""
        if not self.initialized:
            raise RAGServiceError("RAG service not initialized")
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(question, context_filters, max_results, similarity_threshold)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.info("Returning cached result")
                return cached_result
            
            # Retrieve relevant documents
            retrieval_start = time.time()
            relevant_docs = await self._retrieve_documents(
                query=question,
                filters=context_filters,
                max_results=max_results or settings.MAX_RETRIEVAL_RESULTS,
                similarity_threshold=similarity_threshold or settings.SIMILARITY_THRESHOLD
            )
            retrieval_time = time.time() - retrieval_start
            
            # Generate response using LLM
            generation_start = time.time()
            answer = await self._generate_answer(
                question=question,
                relevant_docs=relevant_docs,
                conversation_id=conversation_id,
                system_prompt_type=system_prompt_type
            )
            generation_time = time.time() - generation_start
            
            total_time = time.time() - start_time
            
            # Prepare sources for response
            sources = []
            if include_sources:
                sources = self._format_sources(relevant_docs)
            
            result = RAGQueryResult(
                query=question,
                answer=answer,
                sources=sources,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time
            )
            
            # Cache the result
            self._cache_result(cache_key, result)
            
            logger.info(f"Query completed in {total_time:.2f}s (retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query '{question}': {e}")
            raise RAGServiceError(f"Query failed: {e}")
    
    async def _retrieve_documents(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        max_results: int,
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from vector store."""
        try:
            # Search with threshold filtering
            results = vector_store.search_with_threshold(
                query=query,
                similarity_threshold=similarity_threshold,
                n_results=max_results,
                where=filters
            )
            
            # Re-rank results if we have more than target
            if len(results) > settings.RERANK_TOP_K:
                results = await self._rerank_results(query, results[:settings.RERANK_TOP_K * 2])
                results = results[:settings.RERANK_TOP_K]
            
            logger.info(f"Retrieved {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    async def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-rank search results for better relevance."""
        try:
            # Simple re-ranking based on legal relevance indicators
            scored_results = []
            
            for result in results:
                score = result["similarity"]
                text = result["document"]
                metadata = result["metadata"]
                
                # Boost score for legal citations
                if metadata.get("legal_citations"):
                    score += 0.1
                
                # Boost score for specific document types
                doc_type = metadata.get("document_type", "")
                if doc_type in ["contract", "brief", "statute"]:
                    score += 0.05
                
                # Boost score for section matches
                section_type = metadata.get("section_type", "")
                if section_type in ["definitions", "representations", "governing_law"]:
                    score += 0.05
                
                # Penalty for very short chunks (likely incomplete)
                if len(text) < 200:
                    score -= 0.1
                
                scored_results.append((score, result))
            
            # Sort by adjusted score
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            return [result for score, result in scored_results]
            
        except Exception as e:
            logger.error(f"Error re-ranking results: {e}")
            return results
    
    async def _generate_answer(
        self,
        question: str,
        relevant_docs: List[Dict[str, Any]],
        conversation_id: Optional[str],
        system_prompt_type: str
    ) -> str:
        """Generate answer using LLM with retrieved context."""
        try:
            # Prepare context from relevant documents
            context_parts = []
            for i, doc in enumerate(relevant_docs):
                metadata = doc["metadata"]
                source_info = f"Source {i+1}"
                
                if metadata.get("source"):
                    source_file = Path(metadata["source"]).name
                    source_info += f" ({source_file})"
                
                if metadata.get("section_type"):
                    source_info += f" - {metadata['section_type']}"
                
                context_parts.append(f"{source_info}:\n{doc['document']}\n")
            
            context = "\n".join(context_parts)
            
            # Get system prompt
            system_prompt = LegalPromptTemplates.get_system_prompt(system_prompt_type)
            
            # Format the query with context
            if context:
                prompt = LegalPromptTemplates.format_rag_query(context, question)
            else:
                prompt = f"Please answer the following legal question:\n\n{question}\n\nNote: No specific document context was found. Please provide general legal information with appropriate disclaimers."
            
            # Generate response
            response = ""
            async for chunk in llm_service.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                conversation_id=conversation_id,
                stream=False
            ):
                response += chunk
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating a response to your question. Please try again or contact support."
    
    def _format_sources(self, relevant_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source documents for response."""
        sources = []
        
        for i, doc in enumerate(relevant_docs):
            metadata = doc["metadata"]
            
            source = {
                "index": i + 1,
                "similarity": round(doc["similarity"], 3),
                "text_preview": doc["document"][:200] + "..." if len(doc["document"]) > 200 else doc["document"],
                "metadata": {
                    "document_type": metadata.get("document_type", "unknown"),
                    "source_file": Path(metadata.get("source", "")).name if metadata.get("source") else "unknown",
                    "chunk_index": metadata.get("chunk_index", 0),
                    "section_type": metadata.get("section_type"),
                    "legal_citations": metadata.get("legal_citations")
                }
            }
            
            # Remove None values
            source["metadata"] = {k: v for k, v in source["metadata"].items() if v is not None}
            
            sources.append(source)
        
        return sources
    
    def _get_cache_key(self, question: str, filters: Optional[Dict[str, Any]], 
                      max_results: Optional[int], similarity_threshold: Optional[float]) -> str:
        """Generate cache key for query."""
        import hashlib
        key_data = {
            "question": question.lower().strip(),
            "filters": filters or {},
            "max_results": max_results,
            "similarity_threshold": similarity_threshold
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[RAGQueryResult]:
        """Get cached result if available and not expired."""
        if cache_key in self.query_cache:
            # Simple TTL check (in production, use proper cache with expiration)
            return self.query_cache.get(cache_key)
        return None
    
    def _cache_result(self, cache_key: str, result: RAGQueryResult) -> None:
        """Cache query result."""
        # Simple in-memory cache (in production, use Redis or similar)
        if len(self.query_cache) > 100:  # Limit cache size
            # Remove oldest entries
            keys_to_remove = list(self.query_cache.keys())[:20]
            for key in keys_to_remove:
                del self.query_cache[key]
        
        self.query_cache[cache_key] = result
    
    def clear_cache(self) -> None:
        """Clear query cache."""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks."""
        try:
            # Delete all chunks for this document
            deleted_count = vector_store.delete_documents({"document_id": document_id})
            
            if deleted_count > 0:
                logger.info(f"Deleted document {document_id} ({deleted_count} chunks)")
                return True
            else:
                logger.warning(f"No chunks found for document {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    async def search_documents(
        self,
        query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search documents by query or metadata filters."""
        try:
            if query:
                # Semantic search
                results = vector_store.search_with_threshold(
                    query=query,
                    n_results=limit,
                    where=filters
                )
            else:
                # Metadata-only search
                results = vector_store.search_by_metadata(
                    where=filters or {},
                    include=["documents", "metadatas", "ids"]
                )[:limit]
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    async def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about ingested documents."""
        try:
            stats = vector_store.get_collection_stats()
            
            # Add RAG-specific stats
            stats["query_cache_size"] = len(self.query_cache)
            stats["initialized"] = self.initialized
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for RAG service."""
        try:
            start_time = time.time()
            
            # Check individual components
            vector_health = await vector_store.health_check()
            embedding_health = await embedding_service.health_check()
            llm_health = await llm_service.health_check()
            
            # Test end-to-end functionality if all components are healthy
            e2e_test_passed = False
            e2e_time = None
            
            if (vector_health.get("healthy") and 
                embedding_health.get("healthy") and 
                llm_health.get("healthy")):
                
                try:
                    e2e_start = time.time()
                    
                    # Simple end-to-end test
                    test_result = await self.query(
                        question="What is contract law?",
                        max_results=1,
                        include_sources=False
                    )
                    
                    e2e_test_passed = bool(test_result.answer)
                    e2e_time = time.time() - e2e_start
                    
                except Exception as e:
                    logger.warning(f"End-to-end test failed: {e}")
            
            total_time = time.time() - start_time
            
            return {
                "healthy": all([
                    vector_health.get("healthy", False),
                    embedding_health.get("healthy", False),
                    llm_health.get("healthy", False),
                    self.initialized
                ]),
                "initialized": self.initialized,
                "components": {
                    "vector_store": vector_health,
                    "embedding_service": embedding_health,
                    "llm_service": llm_health
                },
                "end_to_end_test": {
                    "passed": e2e_test_passed,
                    "time_seconds": e2e_time
                },
                "cache_stats": {
                    "query_cache_size": len(self.query_cache)
                },
                "total_check_time_seconds": total_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"RAG health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global RAG service instance
rag_service = RAGService()