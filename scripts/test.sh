#!/bin/bash

# Test script
set -e

echo "🧪 Running tests for Voice AI Agent..."

# Run tests in Docker container
docker-compose -f docker/docker-compose.yml run --rm api pytest tests/ -v

echo "✅ Tests completed!"