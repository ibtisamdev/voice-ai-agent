# Voice AI Agent - Setup Guide

## Prerequisites

- Docker Desktop 4.0+
- Docker Compose 2.0+
- Git
- 8GB+ RAM available for Docker
- 10GB+ free disk space

### Optional (for local development)
- Python 3.11+
- Poetry or pip

## Quick Start

1. **Clone and setup:**
   ```bash
   cd voice-ai-agent
   make setup
   ```

2. **Start development environment:**
   ```bash
   make dev
   ```

3. **Verify installation:**
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

## Detailed Setup

### 1. Environment Configuration

The setup script creates `.env` files from templates. Review and update these files:

**Main `.env` file:**
```env
# Update these values for your environment
DATABASE_URL=postgresql://voiceai:voiceai_dev@localhost:5432/voiceai_db
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secure-secret-key-here
```

**Docker `.env` file:**
```env
# Docker-specific configuration
POSTGRES_USER=voiceai
POSTGRES_PASSWORD=voiceai_dev
POSTGRES_DB=voiceai_db
```

### 2. Development Environment

The development environment includes:

- **FastAPI application** (port 8000)
- **PostgreSQL database** (port 5432)
- **Redis cache** (port 6379)
- **PgAdmin** (port 5050, optional)
- **Redis Commander** (port 8081, optional)

### 3. Available Commands

```bash
make help          # Show all commands
make setup         # Initial setup
make dev           # Start development
make dev-tools     # Start with admin tools
make test          # Run tests
make logs          # View logs
make clean         # Clean up
```

### 4. Access Points

- **API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/api/v1/docs
- **Health Check:** http://localhost:8000/api/v1/health
- **PgAdmin:** http://localhost:5050 (admin@voiceai.com / admin)
- **Redis Commander:** http://localhost:8081

## Project Structure

```
voice-ai-agent/
├── backend/           # FastAPI application
│   ├── app/
│   │   ├── api/      # API routes
│   │   ├── core/     # Core configuration
│   │   ├── models/   # Database models
│   │   └── services/ # Business logic
│   └── requirements.txt
├── ai/               # AI components (Phase 1+)
├── telephony/        # Voice/SIP integration (Phase 3+)
├── docker/           # Docker configuration
├── scripts/          # Setup and utility scripts
├── tests/            # Test suite
└── docs/             # Documentation
```

## Development Workflow

### 1. Code Changes

The development environment supports hot reloading:
- Backend code changes automatically restart the API
- No need to rebuild containers for code changes

### 2. Database Changes

For database schema changes:
```bash
# Generate migration
make shell
alembic revision --autogenerate -m "description"

# Apply migration
alembic upgrade head
```

### 3. Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test
docker-compose -f docker/docker-compose.yml run --rm api pytest tests/test_specific.py -v
```

### 4. Debugging

```bash
# View logs
make logs

# Get shell access
make shell

# Database shell
make db-shell

# Redis CLI
make redis-cli
```

## Troubleshooting

### Common Issues

1. **Port conflicts:**
   - Change ports in `docker-compose.yml`
   - Stop conflicting services

2. **Permission errors:**
   ```bash
   chmod +x scripts/*.sh
   ```

3. **Docker issues:**
   ```bash
   make clean
   docker system prune -f
   make dev
   ```

4. **Database connection errors:**
   - Check if PostgreSQL container is healthy
   - Verify DATABASE_URL in .env

### Performance Issues

- Increase Docker memory allocation (8GB+ recommended)
- Use Docker Desktop with WSL2 backend (Windows)
- Close unnecessary applications

### Getting Help

1. Check logs: `make logs`
2. Verify health: `curl http://localhost:8000/api/v1/ready`
3. Review Docker status: `docker-compose ps`

## Next Steps

After successful setup:

1. **Phase 1:** Implement AI infrastructure (LLM + RAG)
2. **Phase 2:** Add voice processing capabilities
3. **Phase 3:** Integrate telephony and CRM

See `ROADMAP.md` for detailed implementation phases.