#!/bin/bash

# Start development environment script
set -e

echo "🚀 Starting Voice AI Agent development environment..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Run 'make setup' first."
    exit 1
fi

# Start development environment
echo "🐳 Starting Docker containers..."
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up --build

echo "✅ Development environment started!"