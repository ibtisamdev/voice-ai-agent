# TTS Testing and Diagnosis Guide

This guide contains scripts and instructions for testing TTS (Text-to-Speech) functionality independently before integrating into the Voice AI Agent application.

## 🚀 Quick Start

Run these scripts in order to diagnose and test TTS functionality:

### 1. System Diagnosis
```bash
python diagnose_tts.py
```
This comprehensive diagnostic script checks:
- System information and audio capabilities
- Python package installations
- File permissions
- Individual TTS engine tests
- Generates a detailed report with recommendations

### 2. Standalone Testing
```bash
python test_tts_standalone.py
```
Tests individual TTS engines:
- Checks available dependencies (PyTorch, Coqui TTS, pyttsx3)
- Tests each engine independently
- Creates audio files for quality comparison
- Provides clear success/failure feedback

### 3. Simple TTS Implementation
```bash
python simple_tts.py
```
A simplified, reliable TTS implementation:
- Automatic engine selection
- Fallback mechanisms
- Easy-to-use API
- Example usage patterns

### 4. Docker Environment Testing
```bash
# Inside Docker container
docker-compose exec api python test_tts_docker.py
```
Tests TTS within the Docker environment:
- Verifies Docker-specific dependencies
- Tests application TTS service integration
- Checks model paths and permissions
- Generates container-specific reports

## 📋 Scripts Overview

### `diagnose_tts.py`
**Purpose**: Comprehensive TTS system diagnosis
**Features**:
- System and audio system checks
- Python package verification
- File permission testing
- Individual engine testing
- Detailed analysis and recommendations
- JSON report generation

**Usage**:
```bash
python diagnose_tts.py
```

**Output**: 
- Console summary
- JSON report file (`tts_diagnostic_report_YYYYMMDD_HHMMSS.json`)
- Test audio files

### `test_tts_standalone.py`
**Purpose**: Test TTS engines independently
**Features**:
- Dependency checking
- Engine-specific testing
- Audio file generation
- Performance metrics
- Clear pass/fail results

**Usage**:
```bash
python test_tts_standalone.py
```

**Output**:
- Test results for each engine
- Audio files for quality testing
- Recommendations for improvements

### `simple_tts.py`
**Purpose**: Simplified TTS implementation for easy testing
**Features**:
- Auto engine selection
- Fallback mechanisms
- Easy API
- Quality metrics

**Usage as script**:
```bash
python simple_tts.py
```

**Usage as module**:
```python
from simple_tts import SimpleTTS

tts = SimpleTTS()
result = tts.synthesize("Hello world!")
if result.success:
    print(f"Audio saved to: {result.audio_file}")
```

### `test_tts_docker.py`
**Purpose**: Test TTS in Docker environment
**Features**:
- Docker environment verification
- Application integration testing
- Container-specific diagnostics
- Service health checks

**Usage**:
```bash
# From host
docker-compose exec api python test_tts_docker.py

# Copy report from container
docker cp <container_id>:/tmp/tts_test_report.json ./
```

## 🔧 Common Issues and Solutions

### Issue: "Coqui TTS not working"
**Symptoms**: Falling back to system TTS (pyttsx3)
**Solutions**:
1. Check PyTorch installation:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```
2. Install Coqui TTS:
   ```bash
   pip install TTS torch torchaudio
   ```
3. Set environment variable:
   ```bash
   export COQUI_TOS_AGREED=1
   ```

### Issue: "No audio output"
**Symptoms**: Audio files created but no sound
**Solutions**:
1. Check audio system:
   ```bash
   # Test system audio
   speaker-test -t wav -c 2
   
   # Check PulseAudio
   pulseaudio --check -v
   ```
2. Verify audio file:
   ```bash
   ffmpeg -i test_audio.wav
   ```

### Issue: "Permission denied"
**Symptoms**: Cannot write audio files
**Solutions**:
1. Check directory permissions:
   ```bash
   ls -la /tmp
   mkdir -p ./audio_output
   ```
2. Run with appropriate permissions

### Issue: "Model download fails"
**Symptoms**: Coqui TTS models won't download
**Solutions**:
1. Check internet connection
2. Set model cache directory:
   ```bash
   export TTS_HOME=/app/models/tts
   mkdir -p $TTS_HOME
   ```
3. Use faster model:
   ```python
   model_name = "tts_models/en/ljspeech/fast_pitch"
   ```

## 🎯 Expected Results

### Successful Diagnosis
When everything is working correctly, you should see:
- ✅ All dependencies available
- ✅ Multiple TTS engines working
- 🎵 Audio files generated
- 📊 Quality metrics reported

### Typical Working Configuration
- **pyttsx3**: Always works (system fallback)
- **Coqui TTS**: Works with PyTorch + TTS package
- **ElevenLabs**: Works with API key
- **Azure**: Works with SDK + API key

## 📁 Output Files

The scripts generate various output files:

### Audio Files
- `*_test.wav` - Test audio files from each engine
- Located in `/tmp/` or current directory
- Can be played to compare quality

### Report Files
- `tts_diagnostic_report_*.json` - Detailed diagnostic data
- Contains system info, test results, recommendations
- Useful for debugging and documentation

## 🚀 Integration Steps

Once you have working TTS engines:

1. **Update Configuration**:
   ```bash
   # Copy improved settings
   cp .env.example .env
   # Edit TTS_DEFAULT_ENGINE and other settings
   ```

2. **Test in Application**:
   ```bash
   # Test voice endpoints
   make voice-test
   
   # Check application logs
   make logs | grep -i tts
   ```

3. **Verify WebSocket Integration**:
   ```bash
   # Test real-time voice processing
   docker-compose exec api python -c "
   import asyncio
   from ai.voice.tts_service import tts_service
   async def test():
       await tts_service.initialize()
       result = await tts_service.synthesize('Hello from Voice AI Agent!')
       print(f'Success: {result.audio_format}, {result.duration_ms}ms')
   asyncio.run(test())
   "
   ```

## 🔍 Debugging Tips

### Enable Debug Logging
```bash
export TTS_DEBUG_ENABLED=true
export TTS_SAVE_DEBUG_AUDIO=true
```

### Check Application Integration
```python
# Test the actual TTS service
from ai.voice.tts_service import TTSService
import asyncio

async def test_integration():
    service = TTSService()
    await service.initialize()
    result = await service.synthesize("Test message")
    print(f"Success: {result.engine}, Duration: {result.duration_ms}ms")

asyncio.run(test_integration())
```

### Monitor Performance
```bash
# Check synthesis timing
time python simple_tts.py

# Monitor resource usage
top -p $(pgrep -f tts)
```

## 📞 Getting Help

If you encounter issues:

1. **Run full diagnosis**: `python diagnose_tts.py`
2. **Check the report**: Review the JSON output
3. **Test individually**: Use `simple_tts.py` to isolate issues
4. **Verify Docker environment**: Use `test_tts_docker.py`
5. **Check logs**: Look for error messages in application logs

The diagnostic scripts provide detailed error messages and recommendations to help resolve most common TTS issues.