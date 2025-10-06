.PHONY: help setup dev dev-tools test clean logs shell db-migrate db-upgrade lint format voice-test voice-models voice-debug prod-build prod-deploy prod-backup prod-security

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Initial project setup
	@echo "Setting up Voice AI Agent..."
	chmod +x scripts/setup.sh
	./scripts/setup.sh

dev: ## Start development environment
	@echo "Starting development environment..."
	docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up --build

dev-tools: ## Start development environment with tools (pgadmin, redis-commander)
	@echo "Starting development environment with tools..."
	docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml --profile tools up --build

test: ## Run tests
	@echo "Running tests..."
	docker-compose -f docker/docker-compose.yml run --rm api pytest tests/ -v

test-cov: ## Run tests with coverage
	@echo "Running tests with coverage..."
	docker-compose -f docker/docker-compose.yml run --rm api pytest tests/ -v --cov=app --cov-report=html

lint: ## Run linting
	@echo "Running linting..."
	docker-compose -f docker/docker-compose.yml run --rm api flake8 app/
	docker-compose -f docker/docker-compose.yml run --rm api mypy app/

format: ## Format code
	@echo "Formatting code..."
	docker-compose -f docker/docker-compose.yml run --rm api black app/ tests/

logs: ## Show application logs
	docker-compose -f docker/docker-compose.yml logs -f api

shell: ## Get shell access to API container
	docker-compose -f docker/docker-compose.yml exec api bash

db-shell: ## Get database shell access
	docker-compose -f docker/docker-compose.yml exec postgres psql -U voiceai -d voiceai_db

redis-cli: ## Get Redis CLI access
	docker-compose -f docker/docker-compose.yml exec redis redis-cli

clean: ## Clean up containers and volumes
	@echo "Cleaning up..."
	docker-compose -f docker/docker-compose.yml down -v
	docker system prune -f

clean-all: ## Clean up everything including images
	@echo "Cleaning up everything..."
	docker-compose -f docker/docker-compose.yml down -v --rmi all
	docker system prune -af

restart: ## Restart development environment
	@echo "Restarting..."
	docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml restart

stop: ## Stop development environment
	@echo "Stopping..."
	docker-compose -f docker/docker-compose.yml down

# =============================================================================
# Voice-specific commands
# =============================================================================

voice-test: ## Test voice services (STT, TTS, conversation)
	@echo "Testing voice services..."
	@echo "1. Testing text-to-speech..."
	curl -X POST "http://localhost:8000/api/v1/voice/synthesize" \
		-H "Content-Type: application/json" \
		-d '{"text": "Hello, this is a test of the voice synthesis system.", "voice_id": "default"}' \
		--output /tmp/test_tts.wav
	@echo "TTS output saved to /tmp/test_tts.wav"
	
	@echo "2. Testing available voices..."
	curl -X GET "http://localhost:8000/api/v1/voice/voices"
	
	@echo "3. Testing conversation session..."
	curl -X POST "http://localhost:8000/api/v1/conversation/sessions" \
		-H "Content-Type: application/json" \
		-d '{"session_type": "voice_call", "direction": "inbound"}'

voice-models: ## Download and manage voice models
	@echo "Managing voice models..."
	@echo "Downloading Whisper models..."
	docker-compose -f docker/docker-compose.yml exec api python -c "import whisper; whisper.load_model('base')"
	
	@echo "Checking Ollama models..."
	docker-compose -f docker/docker-compose.yml exec ollama ollama list
	
	@echo "Available model commands:"
	@echo "  Pull new model: docker-compose -f docker/docker-compose.yml exec ollama ollama pull MODEL_NAME"
	@echo "  Remove model: docker-compose -f docker/docker-compose.yml exec ollama ollama rm MODEL_NAME"

voice-debug: ## Enable voice debugging and show logs
	@echo "Enabling voice debugging..."
	docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d
	@echo "Voice debug mode enabled. Audio debug files will be saved to data/cache/audio/"
	@echo "Showing voice-related logs..."
	docker-compose -f docker/docker-compose.yml logs -f api | grep -E "(voice|audio|conversation|whisper|tts)"

voice-flows: ## Manage conversation flows
	@echo "Conversation flow management..."
	@echo "Available flows:"
	@ls -la ai/conversation/flows/ 2>/dev/null || echo "No flows directory found"
	@echo ""
	@echo "To create a new flow:"
	@echo "  1. Create YAML file in ai/conversation/flows/"
	@echo "  2. Follow the structure in ai/conversation/flows/default.yaml"
	@echo "  3. Restart the service to load new flows"

voice-cache: ## Manage voice processing cache
	@echo "Voice cache management..."
	@echo "Cache directories:"
	@echo "  Audio cache: $(shell du -sh data/cache/audio 2>/dev/null || echo 'Not found')"
	@echo "  Conversation cache: $(shell du -sh data/cache/conversation 2>/dev/null || echo 'Not found')"
	@echo "  Model cache: $(shell du -sh data/models 2>/dev/null || echo 'Not found')"
	@echo ""
	@echo "To clear cache:"
	@echo "  make voice-cache-clear"

voice-cache-clear: ## Clear voice processing cache
	@echo "Clearing voice cache..."
	rm -rf data/cache/audio/*
	rm -rf data/cache/conversation/*
	@echo "Voice cache cleared"

voice-websocket-test: ## Test WebSocket voice streaming
	@echo "Testing WebSocket voice streaming..."
	@echo "Use this test command in another terminal:"
	@echo "  wscat -c ws://localhost:8000/ws/voice/stream/test-session"
	@echo "Or test with browser console:"
	@echo "  const ws = new WebSocket('ws://localhost:8000/ws/voice/stream/test-session');"

# Database commands for voice data
voice-db: ## Access voice-specific database tables
	@echo "Voice database tables:"
	docker-compose -f docker/docker-compose.yml exec postgres psql -U voiceai -d voiceai_db -c "\dt conversations.*"
	docker-compose -f docker/docker-compose.yml exec postgres psql -U voiceai -d voiceai_db -c "\dt voice_data.*"
	docker-compose -f docker/docker-compose.yml exec postgres psql -U voiceai -d voiceai_db -c "\dt analytics.*"

voice-db-stats: ## Show voice processing statistics
	@echo "Voice processing statistics:"
	docker-compose -f docker/docker-compose.yml exec postgres psql -U voiceai -d voiceai_db -c "SELECT * FROM analytics.session_summary LIMIT 10;"

# Development helpers
voice-dev: ## Start development with voice debugging enabled
	@echo "Starting development environment with voice debugging..."
	docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up --build

voice-logs: ## Show voice-specific logs
	@echo "Voice service logs:"
	docker-compose -f docker/docker-compose.yml logs -f api | grep -E "(voice|audio|conversation|whisper|tts|VAD|STT|TTS)"

# =============================================================================
# Production deployment commands
# =============================================================================

prod-build: ## Build production Docker image
	@echo "Building production Docker image..."
	docker build -f backend/Dockerfile.prod.hetzner -t voiceai:latest backend/

prod-deploy: ## Deploy to production
	@echo "Deploying to production..."
	@echo "Environment: $(ENV)"
	@echo "Image tag: $(TAG)"
	./scripts/deploy/deploy.sh $(ENV) $(TAG)

prod-deploy-staging: ## Deploy to staging
	@echo "Deploying to staging environment..."
	./scripts/deploy/deploy.sh staging latest

prod-deploy-production: ## Deploy to production
	@echo "Deploying to production environment..."
	./scripts/deploy/deploy.sh production latest

prod-smoke-test: ## Run production smoke tests
	@echo "Running production smoke tests..."
	./scripts/deploy/smoke-tests.sh

prod-backup: ## Create production backup
	@echo "Creating production backup..."
	./scripts/backup/backup.sh

prod-backup-restore: ## Restore from production backup
	@echo "Restoring from production backup..."
	@echo "Usage: make prod-backup-restore BACKUP=backup_name"
	./scripts/backup/restore.sh $(BACKUP)

prod-security: ## Run security hardening
	@echo "Running security hardening..."
	sudo ./scripts/security/harden-server.sh

prod-optimize-db: ## Optimize production database
	@echo "Optimizing production database..."
	./scripts/db/optimize-db.sh

# =============================================================================
# Phase 4 production commands
# =============================================================================

prod-status: ## Check production deployment status
	@echo "Production Deployment Status"
	@echo "==========================="
	@echo "Date: $(shell date)"
	@echo ""
	@echo "Docker Containers:"
	@docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep voiceai || echo "No Voice AI containers running"
	@echo ""
	@echo "System Resources:"
	@echo "Memory: $(shell free -h | grep Mem | awk '{print $$3 "/" $$2}')"
	@echo "Disk: $(shell df -h / | awk 'NR==2 {print $$3 "/" $$2 " (" $$5 " used)"}')"
	@echo "Load: $(shell uptime | awk -F'load average:' '{ print $$2 }')"
	@echo ""
	@echo "Services Health:"
	@curl -f -s http://localhost/api/v1/health 2>/dev/null && echo "✓ API: Healthy" || echo "✗ API: Unhealthy"
	@echo ""

prod-logs: ## Show production logs
	@echo "Production logs (last 100 lines):"
	@docker-compose -f docker/docker-compose.prod.yml logs --tail=100

prod-monitor: ## Open monitoring dashboard URLs
	@echo "Monitoring Dashboards:"
	@echo "====================="
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"
	@echo "API Health: http://localhost/api/v1/health"
	@echo ""
	@echo "To open in browser:"
	@echo "xdg-open http://localhost:3000  # Grafana"
	@echo "xdg-open http://localhost:9090  # Prometheus"

prod-backup-status: ## Check backup status
	@echo "Backup Status:"
	@echo "=============="
	@./scripts/backup/backup-status.sh 2>/dev/null || echo "Backup status script not found"

prod-ssl-renew: ## Renew SSL certificates
	@echo "Renewing SSL certificates..."
	@sudo certbot renew --quiet
	@docker-compose -f docker/docker-compose.prod.yml restart nginx
	@echo "SSL certificates renewed and nginx restarted"

prod-scale: ## Scale production services
	@echo "Scaling production services..."
	@echo "Usage: make prod-scale SERVICE=api REPLICAS=4"
	@docker-compose -f docker/docker-compose.prod.yml up -d --scale $(SERVICE)=$(REPLICAS)

prod-update: ## Update production deployment
	@echo "Updating production deployment..."
	@git pull origin main
	@./scripts/deploy/deploy.sh production $(shell git rev-parse --short HEAD)
	@./scripts/deploy/smoke-tests.sh

# =============================================================================
# Maintenance commands
# =============================================================================

maintenance-start: ## Start maintenance mode
	@echo "Starting maintenance mode..."
	@docker-compose -f docker/docker-compose.prod.yml stop nginx
	@echo "Maintenance mode started - services are not accessible"

maintenance-stop: ## Stop maintenance mode
	@echo "Stopping maintenance mode..."
	@docker-compose -f docker/docker-compose.prod.yml start nginx
	@echo "Maintenance mode stopped - services are accessible"

maintenance-status: ## Check if in maintenance mode
	@echo "Maintenance Status:"
	@docker-compose -f docker/docker-compose.prod.yml ps nginx | grep -q "Up" && echo "✓ Services are accessible" || echo "⚠ Services are in maintenance mode"

# =============================================================================
# Documentation commands
# =============================================================================

docs-build: ## Build documentation
	@echo "Building documentation..."
	@echo "Documentation available in docs/ directory"

docs-deploy: ## Deploy documentation
	@echo "Deploying documentation..."
	@echo "See docs/deployment/HETZNER_DEPLOYMENT.md for deployment guide"

# =============================================================================
# Development Phase 4 commands
# =============================================================================

phase4-init: ## Initialize Phase 4 production environment
	@echo "Initializing Phase 4 Production Environment..."
	@echo "1. Creating production configuration..."
	@cp .env.production .env.prod
	@echo "2. Building production images..."
	@make prod-build
	@echo "3. Running security hardening..."
	@echo "   Note: Run 'sudo make prod-security' on production server"
	@echo "4. Setting up monitoring..."
	@echo "   Monitoring stack will be deployed with production compose"
	@echo ""
	@echo "Phase 4 initialization complete!"
	@echo "Next steps:"
	@echo "  1. Deploy to staging: make prod-deploy-staging"
	@echo "  2. Run smoke tests: make prod-smoke-test"
	@echo "  3. Deploy to production: make prod-deploy-production"

phase4-test: ## Run Phase 4 integration tests
	@echo "Running Phase 4 integration tests..."
	@pytest tests/test_phase3_config.py -v
	@pytest tests/test_crm_integration.py -v
	@pytest tests/test_telephony_integration.py -v
	@pytest tests/test_campaign_integration.py -v
	@echo "Phase 4 tests completed!"

# Help for Phase 4
phase4-help: ## Show Phase 4 specific help
	@echo "Voice AI Agent - Phase 4 Production Deployment Commands"
	@echo "======================================================"
	@echo ""
	@echo "Production Deployment:"
	@echo "  prod-build              Build production Docker image"
	@echo "  prod-deploy-staging     Deploy to staging environment"
	@echo "  prod-deploy-production  Deploy to production environment"
	@echo "  prod-smoke-test         Run production smoke tests"
	@echo ""
	@echo "Maintenance:"
	@echo "  prod-backup             Create production backup"
	@echo "  prod-backup-restore     Restore from backup"
	@echo "  prod-security           Run security hardening"
	@echo "  prod-optimize-db        Optimize database performance"
	@echo ""
	@echo "Monitoring:"
	@echo "  prod-status             Check deployment status"
	@echo "  prod-logs               Show production logs"
	@echo "  prod-monitor            Show monitoring URLs"
	@echo ""
	@echo "Utilities:"
	@echo "  maintenance-start       Enter maintenance mode"
	@echo "  maintenance-stop        Exit maintenance mode"
	@echo "  prod-ssl-renew          Renew SSL certificates"
	@echo "  prod-scale              Scale services"
	@echo ""
	@echo "Phase 4 Setup:"
	@echo "  phase4-init             Initialize Phase 4 environment"
	@echo "  phase4-test             Run Phase 4 integration tests"