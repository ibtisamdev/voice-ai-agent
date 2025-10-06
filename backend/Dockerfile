FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies including voice processing libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-client \
        build-essential \
        libpq-dev \
        # Voice processing dependencies
        ffmpeg \
        libsndfile1 \
        libasound2-dev \
        portaudio19-dev \
        libportaudio2 \
        # TTS engine
        espeak-ng \
        # Audio codec support
        libavcodec-extra \
        libavformat-dev \
        libavutil-dev \
        # System tools for audio processing
        alsa-utils \
        pulseaudio \
        # Additional libraries for ML models
        libgomp1 \
        libopenblas-dev \
        # Network tools for service communication
        curl \
        netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain for building packages like sudachipy
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . ~/.cargo/env

# Install uv for faster Python package installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
COPY requirements-dev.txt .

# Install PyTorch first (for better dependency resolution)
RUN uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install main dependencies
RUN uv pip install --system -r requirements.txt \
    && uv pip install --system -r requirements-dev.txt

# Install additional voice processing packages
RUN uv pip install --system \
    openai-whisper \
    faster-whisper \
    librosa \
    soundfile \
    pyaudio \
    webrtcvad \
    scipy \
    transformers \
    sentence-transformers

# Create directories for voice processing
RUN mkdir -p /app/models/whisper \
    && mkdir -p /app/models/tts \
    && mkdir -p /app/cache/audio \
    && mkdir -p /app/cache/conversation \
    && mkdir -p /app/debug/audio

# Copy project
COPY . .

# Create non-root user and set permissions
RUN adduser --disabled-password --gecos '' appuser \
    && chown -R appuser:appuser /app \
    && chmod +x /app/scripts/*.sh 2>/dev/null || true

# Switch to non-root user
USER appuser

# Pre-download some models during build (optional, for faster startup)
# This can be commented out to reduce image size
ENV WHISPER_CACHE_DIR=/app/models/whisper
ENV TRANSFORMERS_CACHE=/app/models/transformers

# Set voice processing environment variables
ENV VOICE_MODEL_PATH=/app/models
ENV VOICE_CACHE_PATH=/app/cache
ENV AUDIO_DEBUG_PATH=/app/debug/audio

# Expose port
EXPOSE 8000

# Health check for voice services
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]