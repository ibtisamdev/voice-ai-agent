# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Voice AI Agent is a sophisticated AI-powered voice agent for law firms with real-time conversation capabilities, intelligent document processing, and CRM integration. Built for private, self-hosted deployment using Docker.

**Current Status**: Phase 2 completed (Voice & Conversation Engine). Phase 3 (Integration Layer) and Phase 4 (Production Deployment) are planned.

## Essential Commands

### Development Environment
```bash
make setup           # Initial project setup with model downloads
make dev             # Start development environment (main services)
make dev-tools       # Start with PgAdmin, Redis Commander
make clean           # Clean up containers and volumes
```

### Testing & Quality
```bash
make test            # Run all tests
make test-cov        # Run tests with coverage report
make lint            # Run flake8 and mypy linting
make format          # Format code with black
./scripts/test_voice.sh -v  # Run voice service tests with verbose output
```

### Voice Services (Specialized Commands)
```bash
make voice-test      # Test STT, TTS, and conversation endpoints
make voice-models    # Download/manage Whisper and Ollama models
make voice-debug     # Enable voice debugging with audio file saves
make voice-flows     # Manage conversation flow YAML files
make voice-cache     # View/manage voice processing cache
make voice-db        # Access voice-specific database tables
```

### Database & Debugging
```bash
make db-shell        # PostgreSQL shell (user: voiceai, db: voiceai_db)
make redis-cli       # Redis CLI access
make shell           # API container bash shell
make logs            # Show API logs
```

### Single Test Execution
```bash
# Run specific test files
docker-compose -f docker/docker-compose.yml run --rm api pytest tests/test_voice_stt.py -v
docker-compose -f docker/docker-compose.yml run --rm api pytest tests/test_conversation.py::TestDialogFlowEngine -v
```

## Architecture Overview

The system follows a microservices architecture with clear separation of concerns:

### Core Structure
```
backend/app/          # FastAPI application
├── api/v1/          # REST & WebSocket endpoints
├── core/            # Configuration, database, logging
├── models/          # SQLAlchemy database models
└── services/        # Business logic services

ai/                  # AI processing modules
├── voice/           # Audio processing, STT, TTS
├── conversation/    # Dialog flows, state management
├── decision_engine/ # Intent classification
├── llm/            # Ollama integration
└── rag/            # Document processing, vector store
```

### Key Components

**Voice Processing Pipeline**:
- `ai/voice/audio_processor.py`: WebRTC VAD, noise reduction, audio format handling
- `ai/voice/stt_service.py`: Whisper integration with streaming support
- `ai/voice/tts_service.py`: Multi-engine TTS (Coqui, ElevenLabs, Azure)

**Conversation Management**:
- `ai/conversation/state_manager.py`: Redis-based session persistence with turn tracking
- `ai/conversation/dialog_flow.py`: YAML-based conversation flows with slot filling
- `ai/decision_engine/intent_classifier.py`: BERT + rule-based intent classification

**Real-time Communication**:
- `backend/app/api/v1/voice_ws.py`: WebSocket handler for bidirectional audio streaming
- Connection management with session metadata and error handling

### Data Flow

1. **Audio Input** → Audio Processor (VAD, noise reduction) → STT Service → Text
2. **Text** → Intent Classifier → Dialog Flow Engine → Response Text
3. **Response** → TTS Service → Audio Output via WebSocket
4. **Session State** → Redis (conversation turns, context, metadata)
5. **Persistent Data** → PostgreSQL (conversations, voice_data, analytics schemas)

### Configuration System

Settings managed via `backend/app/core/config.py` with environment variable override:
- Voice processing: Whisper model size, TTS engines, audio parameters
- AI services: Ollama models, embedding configuration, RAG parameters
- Infrastructure: Database URLs, Redis connections, logging levels

### Database Schema

Three main schemas in PostgreSQL:
- `conversations.*`: Sessions, turns, conversation metadata
- `voice_data.*`: Transcriptions, syntheses, audio processing metrics
- `analytics.*`: Daily stats, session summaries, performance metrics

## Development Patterns

### Voice Service Integration
When adding new voice features:
1. Implement core logic in `ai/voice/`
2. Add WebSocket handlers in `backend/app/api/v1/voice_ws.py`
3. Create tests in `tests/test_voice_*.py` with mocked dependencies
4. Update configuration in `backend/app/core/config.py`
5. Add Makefile commands for testing/debugging

### Conversation Flow Development
Dialog flows are YAML-based in `ai/conversation/flows/`:
- Node types: response, intent, slot_filling, conditional, end
- Support for dynamic branching and context-aware responses
- Test flows with WebSocket connections to `/ws/voice/stream/{session_id}`

### Database Changes
1. Update models in `backend/app/models/`
2. Create Alembic migration: `alembic revision --autogenerate -m "description"`
3. Test migration: `alembic upgrade head`
4. For development, schemas are auto-created from `docker/init.sql`

### Testing Strategy
- Unit tests with mocked dependencies for each AI module
- Integration tests for WebSocket voice streaming
- Voice service tests using `scripts/test_voice.sh`
- Use `pytest -m integration` for integration tests only

## Voice Services Configuration

### Model Downloads
- Whisper models: Auto-downloaded to `/app/models/whisper/`
- Ollama models: Managed via `make voice-models`
- TTS models: Downloaded on first use per engine

### Audio Processing
- Default: 16kHz, mono, float32 format
- VAD: WebRTC with configurable aggressiveness (0-3)
- Noise reduction: Optional via noisereduce library
- Chunking: 1-second chunks for real-time processing

### WebSocket Protocol
Audio streaming via `/ws/voice/stream/{session_id}`:
```json
// Audio input
{"type": "audio", "data": "hex_encoded_bytes", "format": "raw_float32", "sample_rate": 16000}

// Text input  
{"type": "text", "text": "user message"}

// Response format
{"type": "transcription", "text": "...", "is_final": true}
{"type": "response", "text": "...", "audio": "base64_encoded"}
```

## Environment Requirements

### Development
- Docker Desktop 4.0+, Docker Compose 2.0+
- 16GB+ RAM for voice models
- 50GB+ storage for models and data

### Voice Model Storage
- Whisper base: ~150MB
- Llama 2 7B: ~4GB  
- Coqui TTS: ~100MB per voice
- Total: ~5GB for full setup