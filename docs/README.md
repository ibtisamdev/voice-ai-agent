# Voice AI Agent Documentation

This directory contains comprehensive documentation for the Voice AI Agent project.

## Quick Start

- **[Setup Guide](setup.md)** - Complete installation and configuration
- **[LLM Setup](llm-setup.md)** - Hardware requirements and model selection

## Documentation Index

### Getting Started
- [Setup Guide](setup.md) - Initial project setup and Docker environment
- [LLM Setup](llm-setup.md) - Language model installation and hardware requirements

### Architecture
- [Project Structure](../README.md) - Overview of the project
- [ROADMAP](../claude-docs/ROADMAP.md) - Implementation phases and timeline

### Development
- [API Documentation](http://localhost:8000/api/v1/docs) - Interactive API docs (when running)
- [Testing Guide](setup.md#testing) - How to run tests and validate setup

## Hardware Requirements Summary

### Minimum (Basic Development)
- 8GB RAM
- 10GB storage
- Docker support

### Recommended (Full Feature Development)
- 16GB+ RAM
- 50GB+ storage  
- Modern multi-core CPU
- Optional: GPU for faster inference

### Your Current Setup (16GB Mac)
✅ **Perfect for:**
- Llama 2 7B models
- Full development environment
- Testing and prototyping

⚠️ **Possible but tight:**
- Llama 2 13B models
- May need to close other apps

## Quick Reference

### Common Commands
```bash
make setup        # Initial setup
make dev          # Start development
make test         # Run tests
make logs         # View logs
make clean        # Clean up
```

### API Endpoints
- Health: http://localhost:8000/api/v1/health
- Docs: http://localhost:8000/api/v1/docs
- Ready: http://localhost:8000/api/v1/ready

### Recommended Models for 16GB
- `llama2:7b-chat` (4GB) - **Best choice**
- `mistral:7b` (4GB) - Good alternative
- `llama2:7b-q4_0` (2.5GB) - If memory is tight

## Support

If you encounter issues:
1. Check the [Setup Guide](setup.md#troubleshooting)
2. Review [LLM Setup](llm-setup.md#troubleshooting)
3. Run `make logs` to view application logs
4. Verify system requirements are met