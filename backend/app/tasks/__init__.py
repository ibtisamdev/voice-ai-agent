"""
Celery tasks for Voice AI Agent.
This module contains asynchronous tasks for:
- Voice processing (STT, TTS)
- Document analysis and indexing
- AI processing and conversation management
- Background maintenance tasks
"""

from .voice_tasks import *
from .document_tasks import *
from .ai_tasks import *

__all__ = [
    # Voice tasks
    'process_audio_async',
    'synthesize_speech_async',
    
    # Document tasks  
    'process_document_async',
    'update_document_index',
    
    # AI tasks
    'process_conversation_async',
    'analyze_intent_async',
]