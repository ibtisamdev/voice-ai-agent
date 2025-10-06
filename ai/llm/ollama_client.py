"""
Ollama client for local LLM inference.
Handles connection management, model loading, and inference requests.
"""

import asyncio
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any
from datetime import datetime
import aiohttp
import json

from app.core.config import settings

logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when connection to Ollama server fails."""
    pass


class OllamaModelError(Exception):
    """Raised when model-related operations fail."""
    pass


class OllamaClient:
    """Async client for Ollama LLM inference."""
    
    def __init__(self, host: str = None, timeout: int = None):
        self.host = host or settings.OLLAMA_HOST
        self.timeout = timeout or settings.OLLAMA_TIMEOUT
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        
    async def connect(self):
        """Initialize HTTP session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
    async def close(self):
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            
    async def health_check(self) -> bool:
        """Check if Ollama server is running and healthy."""
        try:
            await self.connect()
            async with self.session.get(f"{self.host}/") as response:
                if response.status == 200:
                    text = await response.text()
                    return "Ollama is running" in text
                return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
            
    async def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        try:
            await self.connect()
            async with self.session.get(f"{self.host}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("models", [])
                else:
                    raise OllamaConnectionError(f"Failed to list models: {response.status}")
        except aiohttp.ClientError as e:
            raise OllamaConnectionError(f"Connection error: {e}")
            
    async def pull_model(self, model_name: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Pull a model from Ollama registry."""
        try:
            await self.connect()
            payload = {"name": model_name}
            
            async with self.session.post(
                f"{self.host}/api/pull",
                json=payload
            ) as response:
                if response.status != 200:
                    raise OllamaModelError(f"Failed to pull model: {response.status}")
                    
                async for line in response.content:
                    if line:
                        try:
                            yield json.loads(line.decode('utf-8'))
                        except json.JSONDecodeError:
                            continue
        except aiohttp.ClientError as e:
            raise OllamaConnectionError(f"Connection error during model pull: {e}")
            
    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[List[int]] = None,
        stream: bool = False,
        raw: bool = False,
        format: Optional[str] = None,
        keep_alive: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate text using Ollama model."""
        try:
            await self.connect()
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream
            }
            
            # Add optional parameters
            if system:
                payload["system"] = system
            if template:
                payload["template"] = template
            if context:
                payload["context"] = context
            if raw:
                payload["raw"] = raw
            if format:
                payload["format"] = format
            if keep_alive:
                payload["keep_alive"] = keep_alive
            if options:
                payload["options"] = options
                
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise OllamaModelError(f"Generation failed: {response.status} - {error_text}")
                
                if stream:
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                yield data
                            except json.JSONDecodeError:
                                continue
                else:
                    data = await response.json()
                    yield data
                    
        except aiohttp.ClientError as e:
            raise OllamaConnectionError(f"Connection error during generation: {e}")
            
    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        format: Optional[str] = None,
        keep_alive: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Chat with Ollama model using conversation format."""
        try:
            await self.connect()
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream
            }
            
            # Add optional parameters
            if format:
                payload["format"] = format
            if keep_alive:
                payload["keep_alive"] = keep_alive
            if options:
                payload["options"] = options
                
            async with self.session.post(
                f"{self.host}/api/chat",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise OllamaModelError(f"Chat failed: {response.status} - {error_text}")
                
                if stream:
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                yield data
                            except json.JSONDecodeError:
                                continue
                else:
                    data = await response.json()
                    yield data
                    
        except aiohttp.ClientError as e:
            raise OllamaConnectionError(f"Connection error during chat: {e}")
            
    async def embeddings(
        self,
        model: str,
        prompt: str,
        keep_alive: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """Generate embeddings for text."""
        try:
            await self.connect()
            
            payload = {
                "model": model,
                "prompt": prompt
            }
            
            if keep_alive:
                payload["keep_alive"] = keep_alive
            if options:
                payload["options"] = options
                
            async with self.session.post(
                f"{self.host}/api/embeddings",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise OllamaModelError(f"Embeddings failed: {response.status} - {error_text}")
                
                data = await response.json()
                return data.get("embedding", [])
                
        except aiohttp.ClientError as e:
            raise OllamaConnectionError(f"Connection error during embeddings: {e}")
            
    async def delete_model(self, model_name: str) -> bool:
        """Delete a model from local storage."""
        try:
            await self.connect()
            payload = {"name": model_name}
            
            async with self.session.delete(
                f"{self.host}/api/delete",
                json=payload
            ) as response:
                return response.status == 200
                
        except aiohttp.ClientError as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            return False
            
    async def copy_model(self, source: str, destination: str) -> bool:
        """Copy a model to a new name."""
        try:
            await self.connect()
            payload = {"source": source, "destination": destination}
            
            async with self.session.post(
                f"{self.host}/api/copy",
                json=payload
            ) as response:
                return response.status == 200
                
        except aiohttp.ClientError as e:
            logger.error(f"Error copying model {source} to {destination}: {e}")
            return False