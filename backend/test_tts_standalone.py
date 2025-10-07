#!/usr/bin/env python3
"""
Standalone TTS Test Script
Tests different TTS engines independently to diagnose issues.
"""

import os
import sys
import logging
import tempfile
import wave
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check which TTS dependencies are available."""
    dependencies = {}
    
    # Check PyTorch
    try:
        import torch
        dependencies['torch'] = {
            'available': True,
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
    except ImportError as e:
        dependencies['torch'] = {'available': False, 'error': str(e)}
    
    # Check TorchAudio
    try:
        import torchaudio
        dependencies['torchaudio'] = {
            'available': True,
            'version': torchaudio.__version__
        }
    except ImportError as e:
        dependencies['torchaudio'] = {'available': False, 'error': str(e)}
    
    # Check Coqui TTS
    try:
        from TTS.api import TTS
        dependencies['coqui_tts'] = {
            'available': True,
            'models': TTS.list_models()[:5]  # Show first 5 models
        }
    except ImportError as e:
        dependencies['coqui_tts'] = {'available': False, 'error': str(e)}
    
    # Check pyttsx3
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        dependencies['pyttsx3'] = {
            'available': True,
            'voice_count': len(voices) if voices else 0,
            'voices': [{'id': v.id, 'name': v.name} for v in (voices[:3] if voices else [])]
        }
        engine.stop()
    except ImportError as e:
        dependencies['pyttsx3'] = {'available': False, 'error': str(e)}
    except Exception as e:
        dependencies['pyttsx3'] = {'available': False, 'error': f"Init error: {str(e)}"}
    
    # Check Azure Speech SDK
    try:
        import azure.cognitiveservices.speech as speechsdk
        dependencies['azure_speech'] = {
            'available': True,
            'version': speechsdk.__version__ if hasattr(speechsdk, '__version__') else 'unknown'
        }
    except ImportError as e:
        dependencies['azure_speech'] = {'available': False, 'error': str(e)}
    
    return dependencies

def test_pyttsx3_tts(text: str = "Hello, this is a test of pyttsx3 text to speech.") -> Dict[str, Any]:
    """Test pyttsx3 TTS engine."""
    logger.info("Testing pyttsx3 TTS...")
    
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        
        # Get available voices
        voices = engine.getProperty('voices')
        logger.info(f"Found {len(voices) if voices else 0} pyttsx3 voices")
        
        if voices:
            for i, voice in enumerate(voices[:3]):  # Show first 3 voices
                logger.info(f"  Voice {i}: {voice.name} ({voice.id})")
        
        # Test synthesis
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        start_time = time.time()
        
        # Configure engine
        engine.setProperty('rate', 150)  # Speed
        engine.setProperty('volume', 1.0)  # Volume
        
        # Save to file
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        
        synthesis_time = time.time() - start_time
        
        # Check if file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            
            # Try to get audio info
            try:
                with wave.open(output_path, 'rb') as wav_file:
                    duration = wav_file.getnframes() / wav_file.getframerate()
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
            except Exception as e:
                duration = None
                sample_rate = None
                channels = None
                logger.warning(f"Could not read WAV file info: {e}")
            
            logger.info(f"✅ pyttsx3 synthesis successful!")
            logger.info(f"   Output: {output_path}")
            logger.info(f"   File size: {file_size} bytes")
            logger.info(f"   Duration: {duration:.2f}s" if duration else "   Duration: unknown")
            logger.info(f"   Sample rate: {sample_rate}Hz" if sample_rate else "   Sample rate: unknown")
            logger.info(f"   Synthesis time: {synthesis_time:.2f}s")
            
            return {
                'success': True,
                'output_file': output_path,
                'file_size': file_size,
                'duration': duration,
                'sample_rate': sample_rate,
                'channels': channels,
                'synthesis_time': synthesis_time,
                'engine': 'pyttsx3'
            }
        else:
            return {'success': False, 'error': 'Output file was not created', 'engine': 'pyttsx3'}
            
    except Exception as e:
        logger.error(f"❌ pyttsx3 TTS failed: {e}")
        return {'success': False, 'error': str(e), 'engine': 'pyttsx3'}

def test_coqui_tts(text: str = "Hello, this is a test of Coqui TTS text to speech.") -> Dict[str, Any]:
    """Test Coqui TTS engine."""
    logger.info("Testing Coqui TTS...")
    
    try:
        # Set environment variable to accept Coqui license
        os.environ["COQUI_TOS_AGREED"] = "1"
        
        from TTS.api import TTS
        
        # Try to use a simple, fast model
        model_name = "tts_models/en/ljspeech/tacotron2-DDC"
        logger.info(f"Initializing Coqui TTS with model: {model_name}")
        
        start_init = time.time()
        tts = TTS(model_name=model_name)
        init_time = time.time() - start_init
        
        logger.info(f"Model initialized in {init_time:.2f}s")
        
        # Test synthesis
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        start_time = time.time()
        tts.tts_to_file(text=text, file_path=output_path)
        synthesis_time = time.time() - start_time
        
        # Check if file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            
            # Try to get audio info
            try:
                with wave.open(output_path, 'rb') as wav_file:
                    duration = wav_file.getnframes() / wav_file.getframerate()
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
            except Exception as e:
                duration = None
                sample_rate = None
                channels = None
                logger.warning(f"Could not read WAV file info: {e}")
            
            logger.info(f"✅ Coqui TTS synthesis successful!")
            logger.info(f"   Model: {model_name}")
            logger.info(f"   Output: {output_path}")
            logger.info(f"   File size: {file_size} bytes")
            logger.info(f"   Duration: {duration:.2f}s" if duration else "   Duration: unknown")
            logger.info(f"   Sample rate: {sample_rate}Hz" if sample_rate else "   Sample rate: unknown")
            logger.info(f"   Init time: {init_time:.2f}s")
            logger.info(f"   Synthesis time: {synthesis_time:.2f}s")
            
            return {
                'success': True,
                'output_file': output_path,
                'file_size': file_size,
                'duration': duration,
                'sample_rate': sample_rate,
                'channels': channels,
                'init_time': init_time,
                'synthesis_time': synthesis_time,
                'model': model_name,
                'engine': 'coqui'
            }
        else:
            return {'success': False, 'error': 'Output file was not created', 'engine': 'coqui'}
            
    except Exception as e:
        logger.error(f"❌ Coqui TTS failed: {e}")
        return {'success': False, 'error': str(e), 'engine': 'coqui'}

def test_simple_coqui_tts(text: str = "Hello, this is a simple Coqui TTS test.") -> Dict[str, Any]:
    """Test Coqui TTS with the simplest available model."""
    logger.info("Testing Coqui TTS with simple model...")
    
    try:
        os.environ["COQUI_TOS_AGREED"] = "1"
        from TTS.api import TTS
        
        # Try the fastest, simplest model
        model_name = "tts_models/en/ljspeech/fast_pitch"
        logger.info(f"Trying simple model: {model_name}")
        
        start_init = time.time()
        tts = TTS(model_name=model_name)
        init_time = time.time() - start_init
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        start_time = time.time()
        tts.tts_to_file(text=text, file_path=output_path)
        synthesis_time = time.time() - start_time
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"✅ Simple Coqui TTS successful!")
            logger.info(f"   Output: {output_path}")
            logger.info(f"   File size: {file_size} bytes")
            
            return {
                'success': True,
                'output_file': output_path,
                'file_size': file_size,
                'init_time': init_time,
                'synthesis_time': synthesis_time,
                'model': model_name,
                'engine': 'coqui_simple'
            }
        else:
            return {'success': False, 'error': 'Output file was not created', 'engine': 'coqui_simple'}
            
    except Exception as e:
        logger.error(f"❌ Simple Coqui TTS failed: {e}")
        return {'success': False, 'error': str(e), 'engine': 'coqui_simple'}

def main():
    """Main test function."""
    print("🎙️  TTS Standalone Test Script")
    print("=" * 50)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    deps = check_dependencies()
    
    for dep_name, dep_info in deps.items():
        if dep_info['available']:
            print(f"   ✅ {dep_name}: Available")
            if 'version' in dep_info:
                print(f"      Version: {dep_info['version']}")
            if dep_name == 'torch' and 'cuda_available' in dep_info:
                print(f"      CUDA: {'Yes' if dep_info['cuda_available'] else 'No'}")
            if dep_name == 'pyttsx3' and 'voice_count' in dep_info:
                print(f"      Voices: {dep_info['voice_count']}")
        else:
            print(f"   ❌ {dep_name}: Not available")
            print(f"      Error: {dep_info['error']}")
    
    # Test TTS engines
    print("\n2. Testing TTS engines...")
    
    test_text = "Hello! This is a test of text to speech synthesis. The quick brown fox jumps over the lazy dog."
    results = []
    
    # Test pyttsx3 (should always work)
    if deps['pyttsx3']['available']:
        print("\n   Testing pyttsx3...")
        result = test_pyttsx3_tts(test_text)
        results.append(result)
    else:
        print("\n   ⏭️  Skipping pyttsx3 (not available)")
    
    # Test Coqui TTS
    if deps['coqui_tts']['available'] and deps['torch']['available']:
        print("\n   Testing Coqui TTS...")
        result = test_coqui_tts(test_text)
        results.append(result)
        
        # If standard model failed, try simple model
        if not result['success']:
            print("\n   Trying simple Coqui model...")
            simple_result = test_simple_coqui_tts(test_text)
            results.append(simple_result)
    else:
        print("\n   ⏭️  Skipping Coqui TTS (dependencies not available)")
    
    # Summary
    print("\n3. Test Summary")
    print("=" * 30)
    
    successful_engines = []
    failed_engines = []
    
    for result in results:
        if result['success']:
            successful_engines.append(result)
            print(f"   ✅ {result['engine']}: SUCCESS")
            print(f"      Output: {result['output_file']}")
            if 'synthesis_time' in result:
                print(f"      Synthesis time: {result['synthesis_time']:.2f}s")
            if 'file_size' in result:
                print(f"      File size: {result['file_size']} bytes")
        else:
            failed_engines.append(result)
            print(f"   ❌ {result['engine']}: FAILED")
            print(f"      Error: {result['error']}")
    
    print(f"\n📊 Results: {len(successful_engines)} successful, {len(failed_engines)} failed")
    
    if successful_engines:
        print("\n🎉 Working TTS engines found!")
        print("   You can use these audio files to test quality:")
        for result in successful_engines:
            print(f"   • {result['engine']}: {result['output_file']}")
        
        print("\n💡 Recommendations:")
        if any(r['engine'].startswith('coqui') for r in successful_engines):
            print("   • Coqui TTS is working - use this for high quality synthesis")
        elif any(r['engine'] == 'pyttsx3' for r in successful_engines):
            print("   • Only pyttsx3 is working - consider installing Coqui TTS dependencies")
            print("   • Run: pip install TTS torch torchaudio")
        
    else:
        print("\n❌ No TTS engines are working!")
        print("   Please check the errors above and install missing dependencies.")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    main()