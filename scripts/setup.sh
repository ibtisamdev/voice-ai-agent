#!/bin/bash

# Voice AI Agent Setup Script
set -e

echo "🚀 Setting up Voice AI Agent development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose are available${NC}"

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}📝 Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✅ Created .env file${NC}"
    echo -e "${YELLOW}⚠️  Please review and update .env file with your settings${NC}"
else
    echo -e "${BLUE}ℹ️  .env file already exists${NC}"
fi

# Create Docker environment file if it doesn't exist
if [ ! -f docker/.env ]; then
    echo -e "${YELLOW}📝 Creating docker/.env file from template...${NC}"
    cp docker/.env.example docker/.env
    echo -e "${GREEN}✅ Created docker/.env file${NC}"
else
    echo -e "${BLUE}ℹ️  docker/.env file already exists${NC}"
fi

# Initialize Git repository if not already initialized
if [ ! -d .git ]; then
    echo -e "${YELLOW}📁 Initializing Git repository...${NC}"
    git init
    git add .
    git commit -m "Initial commit: Voice AI Agent foundation setup"
    echo -e "${GREEN}✅ Git repository initialized${NC}"
else
    echo -e "${BLUE}ℹ️  Git repository already exists${NC}"
fi

# Create necessary directories
echo -e "${YELLOW}📁 Creating necessary directories...${NC}"
mkdir -p data/documents
mkdir -p data/chroma
mkdir -p logs
echo -e "${GREEN}✅ Directories created${NC}"

# Set up Python virtual environment (optional for local development)
if command -v python3 &> /dev/null; then
    echo -e "${YELLOW}🐍 Setting up Python virtual environment...${NC}"
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo -e "${GREEN}✅ Virtual environment created${NC}"
        echo -e "${BLUE}ℹ️  To activate: source venv/bin/activate${NC}"
    else
        echo -e "${BLUE}ℹ️  Virtual environment already exists${NC}"
    fi
fi

# Build and start services
echo -e "${YELLOW}🐳 Building Docker containers...${NC}"
docker-compose -f docker/docker-compose.yml build

echo -e "${YELLOW}🚀 Starting services...${NC}"
docker-compose -f docker/docker-compose.yml up -d

# Wait for services to be healthy
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
sleep 10

# Check service health
echo -e "${YELLOW}🔍 Checking service health...${NC}"

# Check if API is responding
if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✅ API service is healthy${NC}"
else
    echo -e "${RED}❌ API service is not responding${NC}"
    echo -e "${YELLOW}📋 Check logs: make logs${NC}"
fi

# Check if PostgreSQL is responding
if docker-compose -f docker/docker-compose.yml exec -T postgres pg_isready -U voiceai > /dev/null 2>&1; then
    echo -e "${GREEN}✅ PostgreSQL is healthy${NC}"
else
    echo -e "${RED}❌ PostgreSQL is not responding${NC}"
fi

# Check if Redis is responding
if docker-compose -f docker/docker-compose.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis is healthy${NC}"
else
    echo -e "${RED}❌ Redis is not responding${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo ""
echo -e "${BLUE}📚 Quick start commands:${NC}"
echo -e "  ${YELLOW}make help${NC}          - Show all available commands"
echo -e "  ${YELLOW}make dev${NC}           - Start development environment"
echo -e "  ${YELLOW}make logs${NC}          - View application logs"
echo -e "  ${YELLOW}make test${NC}          - Run tests"
echo -e "  ${YELLOW}make clean${NC}         - Clean up containers"
echo ""
echo -e "${BLUE}🌐 Access points:${NC}"
echo -e "  API: http://localhost:8000"
echo -e "  API Docs: http://localhost:8000/api/v1/docs"
echo -e "  Health Check: http://localhost:8000/api/v1/health"
echo ""
echo -e "${BLUE}🛠  Development tools (optional):${NC}"
echo -e "  ${YELLOW}make dev-tools${NC}     - Start with PgAdmin and Redis Commander"
echo -e "  PgAdmin: http://localhost:5050 (admin@voiceai.com / admin)"
echo -e "  Redis Commander: http://localhost:8081"
echo ""
echo -e "${YELLOW}⚠️  Next steps:${NC}"
echo -e "  1. Review and update .env file"
echo -e "  2. Run 'make test' to verify everything is working"
echo -e "  3. Start implementing Phase 1 features"