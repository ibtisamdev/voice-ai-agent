"""
Conversation State Manager for managing call sessions and conversation context.
Uses Redis for distributed session storage and state persistence.
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
from contextlib import asynccontextmanager

from app.core.config import settings

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Conversation state enumeration."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    WAITING_FOR_INPUT = "waiting_for_input"
    PROCESSING = "processing"
    TRANSFERRING = "transferring"
    ON_HOLD = "on_hold"
    ENDING = "ending"
    ENDED = "ended"
    ERROR = "error"


class CallDirection(Enum):
    """Call direction enumeration."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"


@dataclass
class ParticipantInfo:
    """Information about a call participant."""
    id: str
    name: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None
    role: str = "caller"  # caller, agent, bot
    language: Optional[str] = "en"
    timezone: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParticipantInfo':
        return cls(**data)


@dataclass
class ConversationContext:
    """Conversation context data."""
    intent: Optional[str] = None
    entities: Optional[Dict[str, Any]] = None
    current_topic: Optional[str] = None
    user_preferences: Optional[Dict[str, Any]] = None
    business_context: Optional[Dict[str, Any]] = None
    collected_information: Optional[Dict[str, Any]] = None
    next_steps: Optional[List[str]] = None
    confidence_scores: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationContext':
        return cls(**data)


@dataclass
class ConversationTurn:
    """Single turn in conversation."""
    turn_id: str
    timestamp: float
    speaker_id: str
    speaker_role: str
    input_text: Optional[str] = None
    input_audio_duration_ms: Optional[float] = None
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    entities: Optional[Dict[str, Any]] = None
    response_text: Optional[str] = None
    response_audio_duration_ms: Optional[float] = None
    processing_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        return cls(**data)


@dataclass
class ConversationSession:
    """Complete conversation session data."""
    session_id: str
    call_id: Optional[str]
    direction: CallDirection
    state: ConversationState
    participants: List[ParticipantInfo]
    context: ConversationContext
    conversation_history: List[ConversationTurn]
    
    # Timestamps
    created_at: float
    updated_at: float
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    
    # Call details
    phone_number: Optional[str] = None
    caller_id: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    # System details
    ai_model_used: Optional[str] = None
    voice_id: Optional[str] = None
    language: str = "en"
    
    # Metrics
    total_turns: int = 0
    user_satisfaction_score: Optional[float] = None
    resolution_status: Optional[str] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert participants and turns to dicts
        data['participants'] = [p.to_dict() for p in self.participants]
        data['conversation_history'] = [t.to_dict() for t in self.conversation_history]
        data['context'] = self.context.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSession':
        # Convert nested objects
        participants = [ParticipantInfo.from_dict(p) for p in data.get('participants', [])]
        conversation_history = [ConversationTurn.from_dict(t) for t in data.get('conversation_history', [])]
        context = ConversationContext.from_dict(data.get('context', {}))
        
        # Create session
        session_data = data.copy()
        session_data['participants'] = participants
        session_data['conversation_history'] = conversation_history
        session_data['context'] = context
        session_data['direction'] = CallDirection(data['direction'])
        session_data['state'] = ConversationState(data['state'])
        
        return cls(**session_data)


class RedisConnectionManager:
    """Manages Redis connections with connection pooling."""
    
    def __init__(self, redis_url: str = None, max_connections: int = 10):
        self.redis_url = redis_url or settings.REDIS_URL
        self.max_connections = max_connections
        self.pool = None
        self.connected = False
    
    async def initialize(self) -> bool:
        """Initialize Redis connection pool."""
        try:
            self.pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            
            # Test connection
            redis_client = redis.Redis(connection_pool=self.pool)
            await redis_client.ping()
            await redis_client.close()
            
            self.connected = True
            logger.info("Redis connection pool initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            return False
    
    @asynccontextmanager
    async def get_connection(self):
        """Get Redis connection from pool."""
        if not self.connected:
            raise Exception("Redis connection not initialized")
        
        redis_client = redis.Redis(connection_pool=self.pool)
        try:
            yield redis_client
        finally:
            await redis_client.close()
    
    async def cleanup(self):
        """Cleanup Redis connections."""
        if self.pool:
            await self.pool.disconnect()
        self.connected = False


class ConversationStateManager:
    """Manages conversation states and session data using Redis."""
    
    def __init__(self):
        self.redis_manager = RedisConnectionManager()
        self.session_ttl = 86400  # 24 hours default TTL
        self.active_sessions: Set[str] = set()
        
        # Redis key prefixes
        self.session_prefix = "voice_ai:session:"
        self.active_sessions_key = "voice_ai:active_sessions"
        self.stats_key = "voice_ai:stats"
        
        # Conversation limits
        self.max_turns_per_session = 1000
        self.max_session_duration = 3600  # 1 hour max
        
    async def initialize(self) -> bool:
        """Initialize the conversation state manager."""
        try:
            if not await self.redis_manager.initialize():
                return False
            
            # Load active sessions from Redis
            await self._load_active_sessions()
            
            logger.info("Conversation state manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize conversation state manager: {e}")
            return False
    
    async def _load_active_sessions(self):
        """Load active session IDs from Redis."""
        try:
            async with self.redis_manager.get_connection() as redis_client:
                session_ids = await redis_client.smembers(self.active_sessions_key)
                self.active_sessions = {sid.decode() for sid in session_ids}
                logger.info(f"Loaded {len(self.active_sessions)} active sessions")
        except Exception as e:
            logger.error(f"Error loading active sessions: {e}")
    
    async def create_session(
        self,
        direction: CallDirection,
        phone_number: Optional[str] = None,
        caller_id: Optional[str] = None,
        participant_info: Optional[ParticipantInfo] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationSession:
        """Create a new conversation session."""
        try:
            session_id = str(uuid.uuid4())
            call_id = str(uuid.uuid4())
            current_time = time.time()
            
            # Create participant info if not provided
            if participant_info is None:
                participant_info = ParticipantInfo(
                    id=str(uuid.uuid4()),
                    phone_number=phone_number,
                    role="caller"
                )
            
            # Create bot participant
            bot_participant = ParticipantInfo(
                id="ai_agent",
                name="AI Legal Assistant",
                role="bot"
            )
            
            # Create session
            session = ConversationSession(
                session_id=session_id,
                call_id=call_id,
                direction=direction,
                state=ConversationState.INITIALIZING,
                participants=[participant_info, bot_participant],
                context=ConversationContext(),
                conversation_history=[],
                created_at=current_time,
                updated_at=current_time,
                phone_number=phone_number,
                caller_id=caller_id,
                metadata=metadata or {}
            )
            
            # Save to Redis
            await self._save_session(session)
            
            # Add to active sessions
            self.active_sessions.add(session_id)
            async with self.redis_manager.get_connection() as redis_client:
                await redis_client.sadd(self.active_sessions_key, session_id)
            
            logger.info(f"Created conversation session: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error creating conversation session: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Retrieve a conversation session by ID."""
        try:
            async with self.redis_manager.get_connection() as redis_client:
                session_key = f"{self.session_prefix}{session_id}"
                session_data = await redis_client.get(session_key)
                
                if session_data:
                    session_dict = json.loads(session_data)
                    return ConversationSession.from_dict(session_dict)
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            return None
    
    async def update_session(self, session: ConversationSession) -> bool:
        """Update a conversation session."""
        try:
            session.updated_at = time.time()
            await self._save_session(session)
            return True
            
        except Exception as e:
            logger.error(f"Error updating session {session.session_id}: {e}")
            return False
    
    async def _save_session(self, session: ConversationSession):
        """Save session to Redis."""
        async with self.redis_manager.get_connection() as redis_client:
            session_key = f"{self.session_prefix}{session.session_id}"
            session_data = json.dumps(session.to_dict(), default=str)
            await redis_client.setex(session_key, self.session_ttl, session_data)
    
    async def add_conversation_turn(
        self,
        session_id: str,
        speaker_id: str,
        speaker_role: str,
        input_text: Optional[str] = None,
        input_audio_duration_ms: Optional[float] = None,
        intent: Optional[str] = None,
        intent_confidence: Optional[float] = None,
        entities: Optional[Dict[str, Any]] = None,
        response_text: Optional[str] = None,
        response_audio_duration_ms: Optional[float] = None,
        processing_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a conversation turn to the session."""
        try:
            session = await self.get_session(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return False
            
            # Check limits
            if len(session.conversation_history) >= self.max_turns_per_session:
                logger.warning(f"Session {session_id} has reached maximum turns")
                return False
            
            # Create turn
            turn = ConversationTurn(
                turn_id=str(uuid.uuid4()),
                timestamp=time.time(),
                speaker_id=speaker_id,
                speaker_role=speaker_role,
                input_text=input_text,
                input_audio_duration_ms=input_audio_duration_ms,
                intent=intent,
                intent_confidence=intent_confidence,
                entities=entities,
                response_text=response_text,
                response_audio_duration_ms=response_audio_duration_ms,
                processing_time_ms=processing_time_ms,
                metadata=metadata
            )
            
            # Add to session
            session.conversation_history.append(turn)
            session.total_turns = len(session.conversation_history)
            
            # Update context if new intent detected
            if intent and intent_confidence and intent_confidence > 0.7:
                session.context.intent = intent
                if not session.context.confidence_scores:
                    session.context.confidence_scores = {}
                session.context.confidence_scores['intent'] = intent_confidence
            
            # Update entities
            if entities:
                if not session.context.entities:
                    session.context.entities = {}
                session.context.entities.update(entities)
            
            # Save updated session
            await self.update_session(session)
            
            logger.debug(f"Added turn to session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding conversation turn: {e}")
            return False
    
    async def update_session_state(self, session_id: str, new_state: ConversationState) -> bool:
        """Update the state of a conversation session."""
        try:
            session = await self.get_session(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return False
            
            old_state = session.state
            session.state = new_state
            
            # Update timestamps
            if new_state == ConversationState.ACTIVE and old_state == ConversationState.INITIALIZING:
                session.started_at = time.time()
            elif new_state == ConversationState.ENDED:
                session.ended_at = time.time()
                if session.started_at:
                    session.duration_seconds = session.ended_at - session.started_at
            
            await self.update_session(session)
            
            logger.info(f"Session {session_id} state changed: {old_state.value} -> {new_state.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating session state: {e}")
            return False
    
    async def update_context(
        self,
        session_id: str,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
        current_topic: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        business_context: Optional[Dict[str, Any]] = None,
        collected_information: Optional[Dict[str, Any]] = None,
        next_steps: Optional[List[str]] = None
    ) -> bool:
        """Update conversation context."""
        try:
            session = await self.get_session(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return False
            
            # Update context fields
            if intent:
                session.context.intent = intent
            if entities:
                if not session.context.entities:
                    session.context.entities = {}
                session.context.entities.update(entities)
            if current_topic:
                session.context.current_topic = current_topic
            if user_preferences:
                if not session.context.user_preferences:
                    session.context.user_preferences = {}
                session.context.user_preferences.update(user_preferences)
            if business_context:
                if not session.context.business_context:
                    session.context.business_context = {}
                session.context.business_context.update(business_context)
            if collected_information:
                if not session.context.collected_information:
                    session.context.collected_information = {}
                session.context.collected_information.update(collected_information)
            if next_steps:
                session.context.next_steps = next_steps
            
            await self.update_session(session)
            return True
            
        except Exception as e:
            logger.error(f"Error updating context: {e}")
            return False
    
    async def end_session(
        self,
        session_id: str,
        resolution_status: Optional[str] = None,
        user_satisfaction_score: Optional[float] = None
    ) -> bool:
        """End a conversation session."""
        try:
            session = await self.get_session(session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return False
            
            # Update session
            await self.update_session_state(session_id, ConversationState.ENDED)
            
            # Update final metrics
            if resolution_status:
                session.resolution_status = resolution_status
            if user_satisfaction_score is not None:
                session.user_satisfaction_score = user_satisfaction_score
            
            await self.update_session(session)
            
            # Remove from active sessions
            self.active_sessions.discard(session_id)
            async with self.redis_manager.get_connection() as redis_client:
                await redis_client.srem(self.active_sessions_key, session_id)
            
            logger.info(f"Ended conversation session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            return False
    
    async def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.active_sessions)
    
    async def get_session_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[ConversationTurn]:
        """Get conversation history for a session."""
        try:
            session = await self.get_session(session_id)
            if not session:
                return []
            
            history = session.conversation_history
            if limit:
                history = history[-limit:]  # Get last N turns
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting session history: {e}")
            return []
    
    async def search_sessions(
        self,
        phone_number: Optional[str] = None,
        caller_id: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        state: Optional[ConversationState] = None,
        limit: int = 100
    ) -> List[ConversationSession]:
        """Search for sessions based on criteria."""
        try:
            sessions = []
            
            # Get all active session keys
            async with self.redis_manager.get_connection() as redis_client:
                pattern = f"{self.session_prefix}*"
                async for key in redis_client.scan_iter(pattern):
                    session_data = await redis_client.get(key)
                    if session_data:
                        try:
                            session_dict = json.loads(session_data)
                            session = ConversationSession.from_dict(session_dict)
                            
                            # Apply filters
                            if phone_number and session.phone_number != phone_number:
                                continue
                            if caller_id and session.caller_id != caller_id:
                                continue
                            if state and session.state != state:
                                continue
                            if date_from and session.created_at < date_from.timestamp():
                                continue
                            if date_to and session.created_at > date_to.timestamp():
                                continue
                            
                            sessions.append(session)
                            
                            if len(sessions) >= limit:
                                break
                                
                        except Exception as e:
                            logger.warning(f"Error parsing session data: {e}")
                            continue
            
            # Sort by creation time (newest first)
            sessions.sort(key=lambda s: s.created_at, reverse=True)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error searching sessions: {e}")
            return []
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count of cleaned sessions."""
        try:
            cleaned_count = 0
            current_time = time.time()
            expired_sessions = []
            
            # Check active sessions for expiration
            for session_id in list(self.active_sessions):
                session = await self.get_session(session_id)
                if not session:
                    # Session not found in Redis, remove from active set
                    expired_sessions.append(session_id)
                    continue
                
                # Check if session is expired
                if (current_time - session.updated_at) > self.max_session_duration:
                    expired_sessions.append(session_id)
                    # End the session
                    await self.end_session(session_id, resolution_status="timeout")
            
            # Remove expired sessions from active set
            for session_id in expired_sessions:
                self.active_sessions.discard(session_id)
                cleaned_count += 1
            
            if cleaned_count > 0:
                # Update Redis active sessions set
                async with self.redis_manager.get_connection() as redis_client:
                    if expired_sessions:
                        await redis_client.srem(self.active_sessions_key, *expired_sessions)
                
                logger.info(f"Cleaned up {cleaned_count} expired sessions")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
            return 0
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        try:
            stats = {
                "active_sessions": len(self.active_sessions),
                "total_sessions_today": 0,
                "average_session_duration": 0.0,
                "total_conversation_turns": 0,
                "most_common_intents": {},
                "average_user_satisfaction": 0.0
            }
            
            # Calculate statistics from recent sessions
            date_from = datetime.now() - timedelta(days=1)
            recent_sessions = await self.search_sessions(date_from=date_from, limit=1000)
            
            if recent_sessions:
                stats["total_sessions_today"] = len(recent_sessions)
                
                # Calculate averages
                total_duration = 0
                total_turns = 0
                total_satisfaction = 0
                satisfaction_count = 0
                intent_counts = {}
                
                for session in recent_sessions:
                    if session.duration_seconds:
                        total_duration += session.duration_seconds
                    total_turns += session.total_turns
                    
                    if session.user_satisfaction_score is not None:
                        total_satisfaction += session.user_satisfaction_score
                        satisfaction_count += 1
                    
                    if session.context.intent:
                        intent = session.context.intent
                        intent_counts[intent] = intent_counts.get(intent, 0) + 1
                
                if len(recent_sessions) > 0:
                    stats["average_session_duration"] = total_duration / len(recent_sessions)
                
                stats["total_conversation_turns"] = total_turns
                
                if satisfaction_count > 0:
                    stats["average_user_satisfaction"] = total_satisfaction / satisfaction_count
                
                # Get top 5 intents
                sorted_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)
                stats["most_common_intents"] = dict(sorted_intents[:5])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of conversation state manager."""
        try:
            # Test Redis connection
            async with self.redis_manager.get_connection() as redis_client:
                await redis_client.ping()
                redis_healthy = True
        except Exception as e:
            redis_healthy = False
            redis_error = str(e)
        
        return {
            "healthy": redis_healthy,
            "redis_connected": redis_healthy,
            "redis_error": redis_error if not redis_healthy else None,
            "active_sessions": len(self.active_sessions),
            "session_ttl_seconds": self.session_ttl,
            "max_turns_per_session": self.max_turns_per_session,
            "max_session_duration_seconds": self.max_session_duration,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.redis_manager.cleanup()
            logger.info("Conversation state manager cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global conversation state manager instance
conversation_state_manager = ConversationStateManager()