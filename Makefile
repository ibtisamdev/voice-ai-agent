.PHONY: help setup dev dev-tools test clean logs shell db-migrate db-upgrade lint format

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