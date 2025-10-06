"""
LLM service for managing language model interactions.
Handles model loading, context management, and response generation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any, Tuple
from datetime import datetime
import json
import time
from contextlib import asynccontextmanager

from .ollama_client import OllamaClient, OllamaConnectionError, OllamaModelError
from .prompt_templates import LegalPromptTemplates, ConversationFlowTemplates
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """High-level service for LLM operations."""
    
    def __init__(self):
        self.client = OllamaClient()
        self.current_model = settings.OLLAMA_MODEL
        self.conversation_contexts: Dict[str, List[Dict[str, str]]] = {}
        self.model_cache: Dict[str, Any] = {}
        
    async def initialize(self) -> bool:
        """Initialize the LLM service and check model availability."""
        try:
            # Check if Ollama is running
            if not await self.client.health_check():
                logger.error("Ollama server is not running")
                return False
            
            # Check if current model is available
            models = await self.client.list_models()
            model_names = [model.get("name", "") for model in models]
            
            if not any(self.current_model in name for name in model_names):
                logger.warning(f"Model {self.current_model} not found. Available models: {model_names}")
                # Attempt to pull the model
                logger.info(f"Attempting to pull model: {self.current_model}")
                await self._pull_model_with_progress(self.current_model)
            
            logger.info(f"LLM service initialized with model: {self.current_model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            return False
    
    async def _pull_model_with_progress(self, model_name: str) -> bool:
        """Pull a model and log progress."""
        try:
            logger.info(f"Starting download of model: {model_name}")
            async for progress in self.client.pull_model(model_name):
                if "status" in progress:
                    if progress.get("status") == "downloading":
                        completed = progress.get("completed", 0)
                        total = progress.get("total", 1)
                        percentage = (completed / total) * 100 if total > 0 else 0
                        logger.info(f"Download progress: {percentage:.1f}%")
                    elif progress.get("status") == "success":
                        logger.info(f"Successfully downloaded model: {model_name}")
                        return True
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> AsyncGenerator[str, None]:
        """Generate a response using the LLM."""
        try:
            # Set up options
            options = {
                "temperature": temperature or settings.OLLAMA_TEMPERATURE,
                "num_predict": max_tokens or settings.OLLAMA_MAX_TOKENS,
            }
            
            # Use conversation context if provided
            if conversation_id and conversation_id in self.conversation_contexts:
                messages = self.conversation_contexts[conversation_id].copy()
                messages.append({"role": "user", "content": prompt})
                
                # Add system message if provided
                if system_prompt:
                    messages.insert(0, {"role": "system", "content": system_prompt})
                
                response_content = ""
                async for chunk in self.client.chat(
                    model=self.current_model,
                    messages=messages,
                    stream=stream,
                    keep_alive=settings.OLLAMA_KEEP_ALIVE,
                    options=options
                ):
                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        response_content += content
                        if stream:
                            yield content
                
                # Update conversation context
                messages.append({"role": "assistant", "content": response_content})
                self.conversation_contexts[conversation_id] = messages
                
                if not stream:
                    yield response_content
            else:
                # Single prompt generation
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                
                response_content = ""
                async for chunk in self.client.generate(
                    model=self.current_model,
                    prompt=full_prompt,
                    stream=stream,
                    keep_alive=settings.OLLAMA_KEEP_ALIVE,
                    options=options
                ):
                    if "response" in chunk:
                        content = chunk["response"]
                        response_content += content
                        if stream:
                            yield content
                
                if not stream:
                    yield response_content
                    
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield f"I apologize, but I'm experiencing technical difficulties. Please try again or contact support."
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Complete a chat conversation."""
        try:
            options = {
                "temperature": temperature or settings.OLLAMA_TEMPERATURE,
                "num_predict": max_tokens or settings.OLLAMA_MAX_TOKENS,
            }
            
            response_content = ""
            async for chunk in self.client.chat(
                model=self.current_model,
                messages=messages,
                stream=False,
                keep_alive=settings.OLLAMA_KEEP_ALIVE,
                options=options
            ):
                if "message" in chunk and "content" in chunk["message"]:
                    response_content += chunk["message"]["content"]
            
            return response_content
            
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again."
    
    async def analyze_document(self, document_text: str, analysis_type: str = "general") -> str:
        """Analyze a legal document."""
        try:
            system_prompt = LegalPromptTemplates.get_system_prompt("document_analysis")
            
            if analysis_type == "contract":
                prompt = LegalPromptTemplates.format_contract_review(document_text)
            else:
                prompt = LegalPromptTemplates.format_document_summary(document_text)
            
            response = ""
            async for chunk in self.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                stream=False
            ):
                response += chunk
            
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return "I was unable to analyze this document. Please try again or contact support."
    
    async def answer_legal_question(self, question: str, context: str = "") -> str:
        """Answer a legal question with optional context."""
        try:
            system_prompt = LegalPromptTemplates.get_system_prompt("legal_assistant")
            
            if context:
                prompt = LegalPromptTemplates.format_rag_query(context, question)
            else:
                prompt = question
            
            response = ""
            async for chunk in self.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                stream=False
            ):
                response += chunk
            
            return response
            
        except Exception as e:
            logger.error(f"Error answering legal question: {e}")
            return "I was unable to answer your question. Please try again or speak with an attorney."
    
    def start_conversation(self, conversation_id: str, system_prompt: Optional[str] = None) -> None:
        """Start a new conversation context."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        self.conversation_contexts[conversation_id] = messages
        logger.info(f"Started conversation: {conversation_id}")
    
    def end_conversation(self, conversation_id: str) -> None:
        """End a conversation and clean up context."""
        if conversation_id in self.conversation_contexts:
            del self.conversation_contexts[conversation_id]
            logger.info(f"Ended conversation: {conversation_id}")
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_contexts.get(conversation_id, [])
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            models = await self.client.list_models()
            return [model.get("name", "") for model in models]
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different model."""
        try:
            models = await self.get_available_models()
            if not any(model_name in model for model in models):
                logger.warning(f"Model {model_name} not available")
                return False
            
            self.current_model = model_name
            logger.info(f"Switched to model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            return False
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            models = await self.client.list_models()
            for model in models:
                if self.current_model in model.get("name", ""):
                    return model
            return {}
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        try:
            start_time = time.time()
            
            # Check Ollama connection
            ollama_healthy = await self.client.health_check()
            
            # Check model availability
            models = await self.get_available_models()
            model_available = any(self.current_model in model for model in models)
            
            # Test inference (simple prompt)
            inference_time = None
            if ollama_healthy and model_available:
                test_start = time.time()
                test_response = ""
                async for chunk in self.generate_response("Hello", stream=False):
                    test_response += chunk
                inference_time = time.time() - test_start
            
            total_time = time.time() - start_time
            
            return {
                "healthy": ollama_healthy and model_available,
                "ollama_connected": ollama_healthy,
                "model_available": model_available,
                "current_model": self.current_model,
                "available_models": models,
                "inference_time_seconds": inference_time,
                "total_check_time_seconds": total_time,
                "active_conversations": len(self.conversation_contexts),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            await self.client.close()
            self.conversation_contexts.clear()
            logger.info("LLM service cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global service instance
llm_service = LLMService()