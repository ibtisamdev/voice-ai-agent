"""
Embedding service using sentence-transformers.
Handles text embedding generation with caching and batch processing.
"""

import asyncio
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingServiceError(Exception):
    """Raised when embedding operations fail."""
    pass


class EmbeddingCache:
    """Simple in-memory cache for embeddings."""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_hours = ttl_hours
    
    def _get_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model."""
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get embedding from cache if available and not expired."""
        key = self._get_key(text, model_name)
        
        if key in self.cache:
            entry = self.cache[key]
            timestamp = datetime.fromisoformat(entry["timestamp"])
            
            # Check if expired
            if datetime.utcnow() - timestamp < timedelta(hours=self.ttl_hours):
                return entry["embedding"]
            else:
                # Remove expired entry
                del self.cache[key]
        
        return None
    
    def set(self, text: str, model_name: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            # Remove 10% of oldest entries
            entries = list(self.cache.items())
            entries.sort(key=lambda x: x[1]["timestamp"])
            num_to_remove = max(1, self.max_size // 10)
            
            for i in range(num_to_remove):
                del self.cache[entries[i][0]]
        
        key = self._get_key(text, model_name)
        self.cache[key] = {
            "embedding": embedding,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.utcnow()
        valid_entries = 0
        
        for entry in self.cache.values():
            timestamp = datetime.fromisoformat(entry["timestamp"])
            if now - timestamp < timedelta(hours=self.ttl_hours):
                valid_entries += 1
        
        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "max_size": self.max_size,
            "ttl_hours": self.ttl_hours
        }


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self, model_name: str = None, cache_size: int = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model: Optional[SentenceTransformer] = None
        self.cache = EmbeddingCache(max_size=cache_size or settings.EMBEDDING_CACHE_SIZE)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_loaded = False
    
    async def initialize(self) -> bool:
        """Initialize the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Load model in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(self.model_name, device=self.device)
            )
            
            self._model_loaded = True
            logger.info(f"Embedding model loaded on device: {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False
    
    def _validate_model(self) -> None:
        """Ensure model is loaded."""
        if not self._model_loaded or self.model is None:
            raise EmbeddingServiceError("Embedding model not initialized")
    
    async def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """Generate embedding for a single text."""
        self._validate_model()
        
        if not text or not text.strip():
            raise EmbeddingServiceError("Empty text provided for embedding")
        
        # Check cache first
        if use_cache:
            cached_embedding = self.cache.get(text, self.model_name)
            if cached_embedding is not None:
                return cached_embedding
        
        try:
            # Generate embedding in thread
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self.model.encode(text, convert_to_tensor=False, normalize_embeddings=True)
            )
            
            # Convert to list
            embedding_list = embedding.tolist()
            
            # Cache the result
            if use_cache:
                self.cache.set(text, self.model_name, embedding_list)
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise EmbeddingServiceError(f"Failed to generate embedding: {e}")
    
    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = None,
        use_cache: bool = True,
        show_progress: bool = False
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        self._validate_model()
        
        if not texts:
            return []
        
        batch_size = batch_size or settings.PROCESSING_BATCH_SIZE
        results = []
        cached_count = 0
        generated_count = 0
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            texts_to_generate = []
            indices_to_generate = []
            
            # Check cache for each text in batch
            for j, text in enumerate(batch):
                if not text or not text.strip():
                    batch_results.append([])
                    continue
                
                if use_cache:
                    cached_embedding = self.cache.get(text, self.model_name)
                    if cached_embedding is not None:
                        batch_results.append(cached_embedding)
                        cached_count += 1
                        continue
                
                # Need to generate this embedding
                texts_to_generate.append(text)
                indices_to_generate.append(len(batch_results))
                batch_results.append(None)  # Placeholder
            
            # Generate embeddings for texts not in cache
            if texts_to_generate:
                try:
                    loop = asyncio.get_event_loop()
                    embeddings = await loop.run_in_executor(
                        None,
                        lambda: self.model.encode(
                            texts_to_generate,
                            convert_to_tensor=False,
                            normalize_embeddings=True,
                            show_progress_bar=show_progress and len(texts_to_generate) > 10
                        )
                    )
                    
                    # Store results and cache
                    for idx, embedding, text in zip(indices_to_generate, embeddings, texts_to_generate):
                        embedding_list = embedding.tolist()
                        batch_results[idx] = embedding_list
                        
                        if use_cache:
                            self.cache.set(text, self.model_name, embedding_list)
                    
                    generated_count += len(texts_to_generate)
                    
                except Exception as e:
                    logger.error(f"Error generating batch embeddings: {e}")
                    # Fill with empty embeddings for failed batch
                    for idx in indices_to_generate:
                        batch_results[idx] = []
            
            results.extend(batch_results)
        
        logger.info(f"Generated embeddings: {generated_count} new, {cached_count} from cache")
        return results
    
    async def compute_similarity(
        self,
        text1: str,
        text2: str,
        use_cache: bool = True
    ) -> float:
        """Compute cosine similarity between two texts."""
        try:
            # Get embeddings
            embedding1, embedding2 = await asyncio.gather(
                self.embed_text(text1, use_cache),
                self.embed_text(text2, use_cache)
            )
            
            # Compute cosine similarity
            embedding1 = np.array(embedding1)
            embedding2 = np.array(embedding2)
            
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    async def find_most_similar(
        self,
        query_text: str,
        candidate_texts: List[str],
        top_k: int = 5,
        use_cache: bool = True
    ) -> List[Tuple[str, float]]:
        """Find most similar texts to query."""
        try:
            # Get query embedding
            query_embedding = await self.embed_text(query_text, use_cache)
            
            # Get candidate embeddings
            candidate_embeddings = await self.embed_batch(candidate_texts, use_cache=use_cache)
            
            # Compute similarities
            similarities = []
            query_array = np.array(query_embedding)
            
            for text, embedding in zip(candidate_texts, candidate_embeddings):
                if embedding:  # Skip empty embeddings
                    candidate_array = np.array(embedding)
                    similarity = np.dot(query_array, candidate_array)
                    similarities.append((text, float(similarity)))
                else:
                    similarities.append((text, 0.0))
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding most similar texts: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "model_loaded": self._model_loaded
        }
        
        if self._model_loaded and self.model:
            try:
                # Get model dimensions
                test_embedding = self.model.encode("test", convert_to_tensor=False)
                info["embedding_dimension"] = len(test_embedding)
                info["max_sequence_length"] = self.model.max_seq_length
            except Exception as e:
                info["error"] = str(e)
        
        return info
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        try:
            start_time = time.time()
            
            # Check if model is loaded
            if not self._model_loaded:
                return {
                    "healthy": False,
                    "error": "Model not loaded",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Test embedding generation
            test_text = "This is a test sentence for health check."
            embedding = await self.embed_text(test_text, use_cache=False)
            
            # Verify embedding
            if not embedding or len(embedding) == 0:
                raise EmbeddingServiceError("Generated empty embedding")
            
            end_time = time.time()
            
            return {
                "healthy": True,
                "model_info": self.get_model_info(),
                "cache_stats": self.get_cache_stats(),
                "test_embedding_dimension": len(embedding),
                "test_generation_time_seconds": end_time - start_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global embedding service instance
embedding_service = EmbeddingService()