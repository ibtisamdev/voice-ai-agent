"""
Vector store implementation using ChromaDB.
Handles document storage, embedding, and similarity search.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""
    pass


class ChromaVectorStore:
    """ChromaDB-based vector store for document storage and retrieval."""
    
    def __init__(self, persist_directory: str = None, collection_name: str = None):
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.client = None
        self.collection = None
        self.embedding_function = None
        
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> bool:
        """Initialize ChromaDB client and collection."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Set up embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.EMBEDDING_MODEL
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Connected to existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
            logger.info("Vector store initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            return False
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the vector store."""
        try:
            if not self.collection:
                raise VectorStoreError("Vector store not initialized")
            
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]
            
            # Add default metadata if not provided
            if metadatas is None:
                metadatas = [{"timestamp": datetime.utcnow().isoformat()} for _ in documents]
            else:
                # Ensure all metadata has timestamp
                for metadata in metadatas:
                    if "timestamp" not in metadata:
                        metadata["timestamp"] = datetime.utcnow().isoformat()
            
            # Add documents to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to collection")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise VectorStoreError(f"Failed to add documents: {e}")
    
    def add_document(
        self,
        document: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """Add a single document to the vector store."""
        doc_id = doc_id or str(uuid.uuid4())
        metadata = metadata or {}
        
        ids = self.add_documents([document], [metadata], [doc_id])
        return ids[0]
    
    def search(
        self,
        query: str,
        n_results: int = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Search for similar documents."""
        try:
            if not self.collection:
                raise VectorStoreError("Vector store not initialized")
            
            n_results = n_results or settings.MAX_RETRIEVAL_RESULTS
            include = include or ["documents", "metadatas", "distances"]
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise VectorStoreError(f"Search failed: {e}")
    
    def search_with_threshold(
        self,
        query: str,
        similarity_threshold: float = None,
        n_results: int = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search documents with similarity threshold filtering."""
        try:
            similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
            n_results = n_results or settings.MAX_RETRIEVAL_RESULTS
            
            results = self.search(
                query=query,
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances", "ids"]
            )
            
            # Filter by similarity threshold
            filtered_results = []
            if results["ids"] and len(results["ids"]) > 0:
                for i, distance in enumerate(results["distances"][0]):
                    # Convert distance to similarity (cosine distance -> cosine similarity)
                    similarity = 1 - distance
                    
                    if similarity >= similarity_threshold:
                        filtered_results.append({
                            "id": results["ids"][0][i],
                            "document": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "similarity": similarity,
                            "distance": distance
                        })
            
            # Sort by similarity (highest first)
            filtered_results.sort(key=lambda x: x["similarity"], reverse=True)
            
            logger.info(f"Found {len(filtered_results)} documents above threshold {similarity_threshold}")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in threshold search: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        try:
            if not self.collection:
                raise VectorStoreError("Vector store not initialized")
            
            results = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )
            
            if results["ids"] and len(results["ids"]) > 0:
                return {
                    "id": results["ids"][0],
                    "document": results["documents"][0],
                    "metadata": results["metadatas"][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None
    
    def update_document(
        self,
        doc_id: str,
        document: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a document and/or its metadata."""
        try:
            if not self.collection:
                raise VectorStoreError("Vector store not initialized")
            
            # Get current document
            current = self.get_document(doc_id)
            if not current:
                logger.warning(f"Document {doc_id} not found for update")
                return False
            
            # Prepare update data
            update_data = {"ids": [doc_id]}
            
            if document is not None:
                update_data["documents"] = [document]
            
            if metadata is not None:
                # Merge with existing metadata
                updated_metadata = current["metadata"].copy()
                updated_metadata.update(metadata)
                updated_metadata["updated_at"] = datetime.utcnow().isoformat()
                update_data["metadatas"] = [updated_metadata]
            
            self.collection.update(**update_data)
            logger.info(f"Updated document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        try:
            if not self.collection:
                raise VectorStoreError("Vector store not initialized")
            
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def delete_documents(self, where: Dict[str, Any]) -> int:
        """Delete documents matching criteria."""
        try:
            if not self.collection:
                raise VectorStoreError("Vector store not initialized")
            
            # Get documents to delete
            results = self.collection.get(where=where, include=["ids"])
            doc_ids = results["ids"]
            
            if doc_ids:
                self.collection.delete(ids=doc_ids)
                logger.info(f"Deleted {len(doc_ids)} documents")
                return len(doc_ids)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            if not self.collection:
                return {"error": "Vector store not initialized"}
            
            # Get collection info
            count = self.collection.count()
            
            # Get sample documents for metadata analysis
            sample_results = self.collection.get(
                limit=min(100, count),
                include=["metadatas"]
            )
            
            # Analyze metadata
            document_types = {}
            sources = {}
            
            for metadata in sample_results.get("metadatas", []):
                doc_type = metadata.get("document_type", "unknown")
                document_types[doc_type] = document_types.get(doc_type, 0) + 1
                
                source = metadata.get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1
            
            return {
                "collection_name": self.collection_name,
                "total_documents": count,
                "document_types": document_types,
                "sources": sources,
                "sample_size": len(sample_results.get("metadatas", [])),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def search_by_metadata(
        self,
        where: Dict[str, Any],
        include: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search documents by metadata criteria."""
        try:
            if not self.collection:
                raise VectorStoreError("Vector store not initialized")
            
            include = include or ["documents", "metadatas", "ids"]
            
            results = self.collection.get(
                where=where,
                include=include
            )
            
            # Format results
            formatted_results = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    result = {"id": doc_id}
                    
                    if "documents" in include and results.get("documents"):
                        result["document"] = results["documents"][i]
                    
                    if "metadatas" in include and results.get("metadatas"):
                        result["metadata"] = results["metadatas"][i]
                    
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []
    
    def reset_collection(self) -> bool:
        """Reset the collection (delete all documents)."""
        try:
            if not self.collection:
                raise VectorStoreError("Vector store not initialized")
            
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Reset collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check vector store health."""
        try:
            if not self.collection:
                return {"healthy": False, "error": "Not initialized"}
            
            # Test basic operations
            start_time = datetime.utcnow()
            
            # Test count
            count = self.collection.count()
            
            # Test search if there are documents
            search_time = None
            if count > 0:
                search_start = datetime.utcnow()
                results = self.search("test query", n_results=1)
                search_time = (datetime.utcnow() - search_start).total_seconds()
            
            total_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "healthy": True,
                "collection_name": self.collection_name,
                "document_count": count,
                "search_time_seconds": search_time,
                "total_check_time_seconds": total_time,
                "embedding_model": settings.EMBEDDING_MODEL,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global vector store instance
vector_store = ChromaVectorStore()