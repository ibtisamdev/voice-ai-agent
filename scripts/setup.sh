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
mkdir -p data/models/whisper
mkdir -p data/models/tts
mkdir -p data/cache/audio
mkdir -p data/cache/conversation
mkdir -p logs
mkdir -p ai/conversation/flows
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

# Download voice models
echo -e "${YELLOW}🎙️  Setting up voice models...${NC}"

# Create default conversation flow if it doesn't exist
if [ ! -f ai/conversation/flows/default.yaml ]; then
    echo -e "${YELLOW}📝 Creating default conversation flow...${NC}"
    cat > ai/conversation/flows/default.yaml << 'EOF'
name: "Legal Consultation Default Flow"
version: "1.0"
description: "Basic flow for legal consultation intake"

nodes:
  - id: "greeting"
    type: "response"
    content: "Hello! I'm your AI legal assistant. How can I help you today?"
    next: "intent_classification"
    
  - id: "intent_classification"
    type: "intent"
    intents:
      - "legal_consultation"
      - "appointment_booking"
      - "general_inquiry"
    default: "general_inquiry"
    
  - id: "legal_consultation"
    type: "response"
    content: "I understand you need legal consultation. Can you briefly describe your legal matter?"
    next: "collect_details"
    
  - id: "appointment_booking"
    type: "response"
    content: "I'd be happy to help you schedule an appointment. What type of legal service do you need?"
    next: "collect_appointment_details"
    
  - id: "general_inquiry"
    type: "response"
    content: "I'm here to help with legal matters. Could you please be more specific about what you need assistance with?"
    next: "intent_classification"
    
  - id: "collect_details"
    type: "slot_filling"
    slots:
      - name: "case_type"
        prompt: "What type of legal case is this?"
        required: true
      - name: "urgency"
        prompt: "How urgent is this matter?"
        required: false
    next: "summary"
    
  - id: "collect_appointment_details"
    type: "slot_filling"
    slots:
      - name: "preferred_date"
        prompt: "What date would work best for you?"
        required: true
      - name: "preferred_time"
        prompt: "What time of day would you prefer?"
        required: true
    next: "appointment_confirmation"
    
  - id: "summary"
    type: "response"
    content: "Thank you for the information. I'll connect you with the appropriate legal specialist. Is there anything else I can help you with?"
    next: "end"
    
  - id: "appointment_confirmation"
    type: "response"
    content: "Perfect! I'll schedule your appointment and send you a confirmation. Is there anything else I can help you with today?"
    next: "end"
    
  - id: "end"
    type: "end"
    content: "Thank you for contacting us. Have a great day!"
EOF
    echo -e "${GREEN}✅ Default conversation flow created${NC}"
fi

# Function to check if Ollama is running and download models
setup_ollama_models() {
    echo -e "${YELLOW}🧠 Setting up LLM models via Ollama...${NC}"
    
    # Wait for Ollama service to be ready
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose -f docker/docker-compose.yml exec -T ollama curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Ollama service is ready${NC}"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            echo -e "${RED}❌ Ollama service failed to start${NC}"
            return 1
        fi
        
        echo -e "${YELLOW}⏳ Waiting for Ollama to start (attempt $attempt/$max_attempts)...${NC}"
        sleep 5
        ((attempt++))
    done
    
    # Download required models
    echo -e "${YELLOW}📥 Downloading Llama 2 7B model (this may take several minutes)...${NC}"
    docker-compose -f docker/docker-compose.yml exec -T ollama ollama pull llama2:7b-chat
    
    echo -e "${YELLOW}📥 Downloading embedding model...${NC}"
    docker-compose -f docker/docker-compose.yml exec -T ollama ollama pull nomic-embed-text
    
    echo -e "${GREEN}✅ LLM models downloaded successfully${NC}"
}

# Function to setup Whisper models
setup_whisper_models() {
    echo -e "${YELLOW}🎤 Setting up Whisper models...${NC}"
    
    # The Whisper models will be downloaded automatically when first used
    # But we can pre-download the base model to speed up first use
    echo -e "${YELLOW}📥 Pre-downloading Whisper base model...${NC}"
    
    # Create a temporary Python script to download Whisper model
    cat > /tmp/download_whisper.py << 'EOF'
import whisper
import os

# Set cache directory
os.environ['WHISPER_CACHE_DIR'] = '/app/models/whisper'

try:
    # Download base model (most commonly used)
    model = whisper.load_model("base")
    print("✅ Whisper base model downloaded successfully")
except Exception as e:
    print(f"❌ Failed to download Whisper model: {e}")
EOF

    # Run the download script in the API container
    if docker-compose -f docker/docker-compose.yml exec -T api python /tmp/download_whisper.py; then
        echo -e "${GREEN}✅ Whisper models ready${NC}"
    else
        echo -e "${YELLOW}⚠️  Whisper models will be downloaded on first use${NC}"
    fi
}

# Build and start services
echo -e "${YELLOW}🐳 Building Docker containers...${NC}"
docker-compose -f docker/docker-compose.yml build

echo -e "${YELLOW}🚀 Starting services...${NC}"
docker-compose -f docker/docker-compose.yml up -d

# Wait for services to be healthy
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
sleep 10

# Setup models after services are running
setup_ollama_models
setup_whisper_models

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
echo -e "  1. Review and update .env file (especially voice API keys)"
echo -e "  2. Run 'make test' to verify everything is working"
echo -e "  3. Test voice services: curl -X POST http://localhost:8000/api/v1/voice/synthesize"
echo -e "  4. Upload documents via: http://localhost:8000/api/v1/docs"
echo -e "  5. Try conversation flows with WebSocket at: ws://localhost:8000/ws/voice/stream"
echo ""
echo -e "${BLUE}🎙️  Voice Service Notes:${NC}"
echo -e "  - Whisper models: Auto-downloaded on first use"
echo -e "  - TTS Engines: Coqui (local), ElevenLabs (API key required)"
echo -e "  - Conversation flows: Located in ai/conversation/flows/"
echo -e "  - Audio cache: data/cache/audio/"