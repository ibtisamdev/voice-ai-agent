# LLM Setup Guide - Hardware Requirements & Recommendations

## System Requirements

### Minimum Requirements
- **RAM**: 8GB (for 7B models)
- **Storage**: 10GB+ free space
- **CPU**: Modern multi-core processor
- **OS**: macOS 10.15+, Linux, or Windows

### Recommended Requirements
- **RAM**: 16GB+ (optimal for 7B-13B models)
- **Storage**: 50GB+ free space (for multiple models)
- **CPU**: Apple Silicon (M1/M2/M3) or modern Intel/AMD
- **GPU**: Optional but improves performance

## Model Compatibility by RAM

### 16GB RAM (Your Current Setup) ✅

| Model | Size | RAM Usage | Performance | Recommended |
|-------|------|-----------|-------------|-------------|
| Llama 2 7B | ~4GB | 4-6GB | Excellent | ✅ **Best Choice** |
| Llama 2 7B-Chat | ~4GB | 4-6GB | Excellent | ✅ **Best Choice** |
| Mistral 7B | ~4GB | 4-6GB | Excellent | ✅ **Great Alternative** |
| Code Llama 7B | ~4GB | 4-6GB | Good | ✅ **For Code Tasks** |
| Llama 2 13B | ~7GB | 8-10GB | Good* | ⚠️ **Possible but tight** |
| Llama 2 13B-Chat | ~7GB | 8-10GB | Good* | ⚠️ **Possible but tight** |

*\*May require closing other applications*

### Quantized Models (Reduced Memory) 🎯

| Model | Original | Quantized | RAM Savings | Quality Loss |
|-------|----------|-----------|-------------|--------------|
| Llama 2 7B-Q4 | 4GB | ~2.5GB | 37% | Minimal |
| Llama 2 7B-Q8 | 4GB | ~3.5GB | 12% | Negligible |
| Llama 2 13B-Q4 | 7GB | ~4GB | 43% | Minimal |
| Llama 2 13B-Q8 | 7GB | ~6GB | 14% | Negligible |

## Installation & Setup

### 1. Install Ollama

**macOS:**
```bash
# Download from https://ollama.ai
curl -fsSL https://ollama.ai/install.sh | sh

# Or via Homebrew
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull Recommended Models

**Start with Llama 2 7B (Recommended for 16GB):**
```bash
# Standard model
ollama pull llama2:7b

# Chat-optimized version
ollama pull llama2:7b-chat

# Quantized version (if memory is tight)
ollama pull llama2:7b-q4_0
```

**Alternative - Mistral 7B:**
```bash
ollama pull mistral:7b
```

**For More Memory (Optional):**
```bash
# Only if you have memory available
ollama pull llama2:13b-chat
```

### 3. Test Installation

```bash
# Check available models
ollama list

# Test a model
ollama run llama2:7b-chat "What is contract law?"

# Check running processes
ollama ps
```

## Performance Optimization

### For 16GB Mac Systems

1. **Memory Management:**
   ```bash
   # Monitor system memory
   htop  # or Activity Monitor on macOS
   
   # Check Ollama memory usage
   ollama ps
   ```

2. **Model Selection Strategy:**
   ```bash
   # Start with smallest effective model
   ollama pull llama2:7b-chat
   
   # Test performance
   time ollama run llama2:7b-chat "Explain legal contracts briefly"
   
   # Upgrade if needed and memory allows
   ollama pull llama2:13b-chat
   ```

3. **System Optimization:**
   - Close unnecessary applications
   - Use Activity Monitor to free up memory
   - Consider restarting before heavy LLM usage

### Model Switching

```bash
# Stop current model
ollama stop llama2:7b

# Start different model
ollama run llama2:13b-chat

# Or use API to switch models programmatically
curl -X POST http://localhost:11434/api/chat \
  -d '{"model": "llama2:7b-chat", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Legal Domain Optimization

### Fine-tuning Considerations

1. **Base Model Selection:**
   - **Llama 2 7B-Chat**: Best balance of performance and memory
   - **Mistral 7B**: Good alternative with strong reasoning
   - **Code Llama 7B**: If contract parsing/generation needed

2. **Legal-Specific Prompting:**
   ```python
   legal_prompt_template = """
   You are a legal AI assistant for a law firm. 
   Provide accurate, professional legal information.
   Always include appropriate disclaimers.
   
   Context: {context}
   Question: {question}
   
   Answer:
   """
   ```

### Model Benchmarking

```bash
# Create benchmark script
cat > benchmark_models.sh << 'EOF'
#!/bin/bash
echo "Benchmarking LLM Models..."

models=("llama2:7b-chat" "mistral:7b")

for model in "${models[@]}"; do
    echo "Testing $model..."
    start_time=$(date +%s.%N)
    
    ollama run $model "What are the key elements of a valid contract?" > /dev/null
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc -l)
    
    echo "$model: ${duration}s"
    echo "---"
done
EOF

chmod +x benchmark_models.sh
./benchmark_models.sh
```

## Integration with Voice AI Agent

### API Configuration

```python
# backend/app/core/config.py
class Settings(BaseSettings):
    # LLM Configuration
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2:7b-chat"
    OLLAMA_TEMPERATURE: float = 0.7
    OLLAMA_MAX_TOKENS: int = 2048
    OLLAMA_TIMEOUT: int = 30
```

### Docker Integration

```yaml
# docker/docker-compose.yml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: voiceai_ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
      - ./models:/models
    environment:
      - OLLAMA_KEEP_ALIVE=24h
    networks:
      - voiceai_network
    # GPU support (if available)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ollama_data:
```

## Troubleshooting

### Common Issues

1. **Out of Memory:**
   ```bash
   # Switch to smaller model
   ollama pull llama2:7b-q4_0
   
   # Or increase swap space (Linux)
   sudo swapon --show
   ```

2. **Slow Performance:**
   ```bash
   # Check system resources
   top -p $(pgrep ollama)
   
   # Reduce concurrent requests
   # Monitor temperature throttling on Mac
   ```

3. **Model Loading Errors:**
   ```bash
   # Clear Ollama cache
   rm -rf ~/.ollama/models/blobs/*
   
   # Re-pull model
   ollama pull llama2:7b-chat
   ```

### Performance Monitoring

```bash
# Monitor Ollama performance
curl http://localhost:11434/api/ps

# System monitoring
iostat -x 1
vmstat 1

# Memory usage
free -h  # Linux
vm_stat  # macOS
```

## Recommendations for Your 16GB Setup

### Phase 1 Implementation

1. **Start Simple:**
   ```bash
   ollama pull llama2:7b-chat
   ```

2. **Test Performance:**
   - Monitor memory usage during operation
   - Benchmark response times for legal queries
   - Test concurrent request handling

3. **Scale Up If Needed:**
   ```bash
   # If performance is good, try larger model
   ollama pull llama2:13b-chat
   ```

### Production Considerations

- **Memory Buffer**: Keep 2-4GB free for other services
- **Model Persistence**: Configure `OLLAMA_KEEP_ALIVE` for faster responses
- **Backup Strategy**: Store trained/fine-tuned models separately
- **Monitoring**: Implement memory and performance alerts

## Next Steps

1. Install Ollama using the commands above
2. Pull `llama2:7b-chat` model
3. Test basic functionality
4. Integrate with Phase 1 RAG pipeline
5. Benchmark performance with legal documents
6. Optimize based on real-world usage patterns

The 7B models should work excellently on your 16GB Mac while leaving plenty of room for the rest of the application stack!