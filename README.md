# Voice AI Agent 🤖📞

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-teal.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-orange.svg)](LICENSE)

A sophisticated AI-powered voice agent designed for law firms, featuring real-time conversation capabilities, intelligent document processing, and seamless CRM integration. Built for private, self-hosted deployment with enterprise-grade security and performance.

## 🌟 Key Features

### 🎙️ Voice Processing
- **Real-time Speech-to-Text** using OpenAI Whisper
- **Multi-engine Text-to-Speech** (Coqui TTS, ElevenLabs, Azure)
- **Voice Activity Detection** with noise reduction
- **Speaker Diarization** for multi-party calls
- **WebSocket streaming** for low-latency interactions

### 🧠 AI Intelligence
- **Local LLM Integration** via Ollama (Llama 2, Mistral)
- **RAG-powered Knowledge Base** with ChromaDB
- **Intent Classification** using BERT and rule-based systems
- **Conversation Flow Management** with YAML-based definitions
- **Context-aware Responses** with persistent session state

### 📄 Document Processing
- **Multi-format Support** (PDF, DOCX, TXT)
- **Intelligent Chunking** strategies for legal documents
- **Semantic Search** with embedding-based retrieval
- **Legal Citation** extraction and processing

### 🏢 Enterprise Ready
- **Redis Session Management** for scalability
- **PostgreSQL Database** for persistent storage
- **Docker Containerization** for easy deployment
- **API-first Architecture** with comprehensive endpoints
- **Health Monitoring** and performance metrics

## 🚀 Quick Start

### Prerequisites
- Docker Desktop 4.0+
- Docker Compose 2.0+
- 16GB+ RAM (recommended)
- 50GB+ free disk space

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd voice-ai-agent
   ```

2. **Initial setup:**
   ```bash
   make setup
   ```

3. **Start development environment:**
   ```bash
   make dev
   ```

4. **Verify installation:**
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

## 📊 Current Implementation Status

### ✅ Completed Phases

**Phase 1: Core AI Infrastructure**
- ✅ Ollama LLM integration with Llama 2/Mistral models
- ✅ ChromaDB vector database for document storage
- ✅ RAG pipeline with semantic search
- ✅ Document processing (PDF, DOCX, TXT)
- ✅ FastAPI REST endpoints

**Phase 2: Voice & Conversation Engine**
- ✅ Whisper STT with real-time transcription
- ✅ Multi-engine TTS (Coqui, ElevenLabs, Azure)
- ✅ Voice activity detection and noise reduction
- ✅ Conversation state management with Redis
- ✅ Dialog flow engine with YAML configurations
- ✅ BERT-based intent classification
- ✅ WebSocket endpoints for real-time voice streaming

### 🚧 Upcoming Phases

**Phase 3: Integration Layer** (Planned)
- 📞 Telephony integration (SIP/Twilio)
- 🔗 Zoho CRM connectivity
- 📅 Appointment scheduling system
- 📊 Campaign management tools

**Phase 4: Production Deployment** (Planned)
- 🐳 Production Docker optimization
- 📈 Monitoring and alerting (Prometheus/Grafana)
- 🔐 Security hardening
- 📚 Comprehensive documentation

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Voice Client  │◄──►│  WebSocket API  │◄──►│ Conversation    │
│   (Browser/App) │    │   (FastAPI)     │    │ State Manager   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Audio Processor │◄──►│ Speech Services │◄──►│ Dialog Flow     │
│ (VAD, Noise     │    │ (STT/TTS)       │    │ Engine          │
│ Reduction)      │    └─────────────────┘    └─────────────────┘
└─────────────────┘                                    │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ChromaDB      │◄──►│   RAG Service   │◄──►│ Intent          │
│ (Vector Store)  │    │ (Retrieval)     │    │ Classifier      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │◄──►│   LLM Service   │◄──►│     Redis       │
│  (Database)     │    │   (Ollama)      │    │   (Cache)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **SQLAlchemy** - Database ORM
- **Alembic** - Database migrations
- **Redis** - Session and cache management
- **PostgreSQL** - Primary database

### AI & Machine Learning
- **Ollama** - Local LLM inference
- **OpenAI Whisper** - Speech-to-text
- **Transformers** - BERT-based classification
- **ChromaDB** - Vector database
- **Sentence Transformers** - Text embeddings

### Voice Processing
- **Coqui TTS** - Local text-to-speech
- **ElevenLabs API** - Premium voice synthesis
- **Azure Speech** - Enterprise TTS option
- **WebRTC VAD** - Voice activity detection
- **Librosa** - Audio processing

### Infrastructure
- **Docker & Docker Compose** - Containerization
- **NGINX** - Reverse proxy (production)
- **Prometheus & Grafana** - Monitoring (planned)

## 📝 API Documentation

### Core Endpoints

#### Health & Status
```bash
GET /api/v1/health              # Service health check
GET /api/v1/ready               # Readiness probe
```

#### Voice Services
```bash
POST /api/v1/voice/transcribe   # Audio transcription
POST /api/v1/voice/synthesize   # Text-to-speech
GET  /api/v1/voice/voices       # Available voices
WS   /ws/voice/stream/{id}      # Real-time voice
```

#### Document Processing
```bash
POST /api/v1/documents/upload   # Upload documents
GET  /api/v1/documents/         # List documents
POST /api/v1/rag/query          # RAG queries
```

#### LLM Integration
```bash
POST /api/v1/llm/generate       # Text generation
POST /api/v1/llm/chat           # Chat completion
GET  /api/v1/llm/models         # Available models
```

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc

## 🧪 Development

### Available Commands

```bash
make help          # Show all commands
make setup         # Initial project setup
make dev           # Start development environment
make dev-tools     # Start with admin tools (PgAdmin, etc.)
make test          # Run test suite
make logs          # View application logs
make shell         # Access API container
make clean         # Clean up containers
```

### Development Workflow

1. **Code Changes**: Hot reloading enabled for API
2. **Database**: Use Alembic for schema migrations
3. **Testing**: Comprehensive test suite with pytest
4. **Debugging**: Container logs and shell access

### Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Test specific components
pytest tests/test_voice_stt.py -v
pytest tests/test_conversation.py -v
```

## 🐳 Docker Services

The development environment includes:

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | FastAPI application |
| PostgreSQL | 5432 | Primary database |
| Redis | 6379 | Session store |
| Ollama | 11434 | LLM inference |
| ChromaDB | 8001 | Vector database |
| PgAdmin | 5050 | Database admin (optional) |

## ⚙️ Configuration

### Environment Variables

Key configuration options:

```env
# Database
DATABASE_URL=postgresql://user:pass@host:port/db

# LLM
OLLAMA_MODEL=llama2:7b-chat
OLLAMA_TEMPERATURE=0.7

# Voice Services
WHISPER_MODEL_SIZE=base
TTS_DEFAULT_ENGINE=coqui
ELEVENLABS_API_KEY=your-key  # Optional

# Redis
REDIS_URL=redis://localhost:6379
```

See `.env.example` for complete configuration options.

## 📋 Requirements

### Hardware Requirements

**Minimum (Development):**
- 8GB RAM
- 4 CPU cores
- 20GB storage
- Docker support

**Recommended (Production):**
- 16GB+ RAM
- 8+ CPU cores
- 100GB+ SSD storage
- GPU for faster inference (optional)

### Model Requirements

**LLM Models** (auto-downloaded):
- Llama 2 7B: ~4GB
- Mistral 7B: ~4GB
- Whisper Base: ~150MB

## 🔧 Troubleshooting

### Common Issues

1. **Memory Issues**: Increase Docker memory limit to 8GB+
2. **Port Conflicts**: Check and update ports in docker-compose.yml
3. **Model Downloads**: Ensure sufficient disk space and internet
4. **Performance**: Close unnecessary applications, use SSD storage

### Getting Help

1. Check logs: `make logs`
2. Verify health: `curl http://localhost:8000/api/v1/health`
3. Review setup: `docs/setup.md`
4. Docker status: `docker-compose ps`

## 📚 Documentation

- **[Setup Guide](docs/setup.md)** - Detailed installation
- **[LLM Configuration](docs/llm-setup.md)** - Model setup
- **[ROADMAP](claude-docs/ROADMAP.md)** - Development phases
- **[API Reference](http://localhost:8000/api/v1/docs)** - Interactive docs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Use Cases

### Legal Industry
- **Client Intake**: Automated initial consultations
- **Appointment Scheduling**: AI-powered booking system
- **Document Review**: Intelligent document analysis
- **Case Management**: Context-aware client interactions

### Features in Development
- **Multi-language Support**: International client base
- **Advanced Analytics**: Call performance metrics
- **CRM Integration**: Seamless data synchronization
- **Mobile App**: On-the-go client management

## 🔮 Future Roadmap

### Short Term (Next 2 Months)
- Telephony integration (Twilio/SIP)
- CRM connectivity (Zoho, Salesforce)
- Production deployment guides
- Performance optimization

### Long Term (6+ Months)
- Multi-tenant architecture
- Advanced conversation analytics
- Mobile applications
- Enterprise SSO integration

---

**Built with ❤️ for the legal industry**

For questions, issues, or contributions, please visit our [documentation](docs/) or create an issue.