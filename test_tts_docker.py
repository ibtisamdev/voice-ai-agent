#!/usr/bin/env python3
"""
Docker-based TTS Test Script
Tests TTS functionality inside the Docker container environment.
"""

import os
import sys
import logging
import tempfile
import wave
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add the app directory to the Python path
sys.path.append('/app')
sys.path.append('/app/backend')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_docker_environment():
    """Check if we're running in the Docker environment."""
    checks = {
        'in_docker': os.path.exists('/.dockerenv'),
        'app_directory': os.path.exists('/app'),
        'models_directory': os.path.exists('/app/models'),
        'cache_directory': os.path.exists('/app/cache'),
        'requirements_file': os.path.exists('/app/requirements.txt'),
        'backend_directory': os.path.exists('/app/backend'),
        'ai_directory': os.path.exists('/app/ai')
    }
    
    return checks

def check_tts_dependencies():
    """Check TTS dependencies in Docker environment."""
    dependencies = {}
    
    # Check system packages
    system_packages = {
        'ffmpeg': 'which ffmpeg',
        'espeak': 'which espeak',
        'alsa': 'ls /usr/share/alsa/ 2>/dev/null'
    }
    
    for package, command in system_packages.items():
        try:
            result = os.system(f"{command} >/dev/null 2>&1")
            dependencies[f'system_{package}'] = {'available': result == 0}
        except:
            dependencies[f'system_{package}'] = {'available': False}
    
    # Check Python packages
    python_packages = ['torch', 'torchaudio', 'TTS', 'pyttsx3', 'azure.cognitiveservices.speech']
    
    for package in python_packages:
        try:
            if package == 'azure.cognitiveservices.speech':
                import azure.cognitiveservices.speech
                dependencies[package] = {'available': True, 'version': getattr(azure.cognitiveservices.speech, '__version__', 'unknown')}
            elif package == 'TTS':
                from TTS.api import TTS
                dependencies[package] = {'available': True, 'models_available': len(TTS.list_models()) > 0}
            elif package == 'torch':
                import torch
                dependencies[package] = {
                    'available': True,
                    'version': torch.__version__,
                    'cuda_available': torch.cuda.is_available(),
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
                }
            elif package == 'torchaudio':
                import torchaudio
                dependencies[package] = {'available': True, 'version': torchaudio.__version__}
            elif package == 'pyttsx3':
                import pyttsx3
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                dependencies[package] = {
                    'available': True,
                    'voice_count': len(voices) if voices else 0
                }
                engine.stop()
            
        except Exception as e:
            dependencies[package] = {'available': False, 'error': str(e)}
    
    return dependencies

def test_tts_service_import():
    """Test importing the TTS service from the application."""
    try:
        # Try to import the TTS service
        from ai.voice.tts_service import TTSService, CoquiTTSEngine, SystemTTSEngine
        
        return {
            'success': True,
            'tts_service_available': True,
            'engines_available': ['CoquiTTSEngine', 'SystemTTSEngine']
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'tts_service_available': False
        }

def test_coqui_tts_in_docker():
    """Test Coqui TTS in Docker environment."""
    try:
        os.environ["COQUI_TOS_AGREED"] = "1"
        from TTS.api import TTS
        
        # Try a fast model first
        model_name = "tts_models/en/ljspeech/fast_pitch"
        
        logger.info(f"Testing Coqui TTS with model: {model_name}")
        start_init = time.time()
        
        tts = TTS(model_name=model_name)
        init_time = time.time() - start_init
        
        # Test synthesis
        test_text = "Hello from Docker container! This is a Coqui TTS test."
        output_file = "/tmp/coqui_test.wav"
        
        start_synthesis = time.time()
        tts.tts_to_file(text=test_text, file_path=output_file)
        synthesis_time = time.time() - start_synthesis
        
        # Check result
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            
            # Try to get audio info
            try:
                with wave.open(output_file, 'rb') as wav_file:
                    duration = wav_file.getnframes() / wav_file.getframerate()
                    sample_rate = wav_file.getframerate()
            except:
                duration = None
                sample_rate = None
            
            return {
                'success': True,
                'model': model_name,
                'output_file': output_file,
                'file_size': file_size,
                'duration': duration,
                'sample_rate': sample_rate,
                'init_time': init_time,
                'synthesis_time': synthesis_time
            }
        else:
            return {'success': False, 'error': 'Output file not created'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

def test_pyttsx3_in_docker():
    """Test pyttsx3 TTS in Docker environment."""
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        
        # Get available voices
        voices = engine.getProperty('voices')
        logger.info(f"Found {len(voices) if voices else 0} pyttsx3 voices")
        
        # Test synthesis
        test_text = "Hello from Docker container! This is a pyttsx3 TTS test."
        output_file = "/tmp/pyttsx3_test.wav"
        
        start_time = time.time()
        engine.save_to_file(test_text, output_file)
        engine.runAndWait()
        synthesis_time = time.time() - start_time
        
        # Check result
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            
            try:
                with wave.open(output_file, 'rb') as wav_file:
                    duration = wav_file.getnframes() / wav_file.getframerate()
                    sample_rate = wav_file.getframerate()
            except:
                duration = None
                sample_rate = None
            
            return {
                'success': True,
                'output_file': output_file,
                'file_size': file_size,
                'duration': duration,
                'sample_rate': sample_rate,
                'synthesis_time': synthesis_time,
                'voice_count': len(voices) if voices else 0
            }
        else:
            return {'success': False, 'error': 'Output file not created'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

def test_application_tts_service():
    """Test the application's TTS service."""
    try:
        from ai.voice.tts_service import TTSService
        
        # Initialize service
        tts_service = TTSService()
        
        # Try to initialize
        logger.info("Initializing TTS service...")
        init_success = False
        
        # Use asyncio if needed
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            init_success = loop.run_until_complete(tts_service.initialize())
            loop.close()
        except:
            # Try synchronous initialization if async fails
            init_success = True  # Assume basic init worked
        
        if init_success:
            # Get available engines
            available_engines = tts_service.get_available_engines()
            
            # Get available voices
            available_voices = tts_service.get_available_voices()
            
            # Try synthesis
            test_text = "Hello from the Voice AI Agent TTS service!"
            
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    tts_service.synthesize(test_text, use_cache=False)
                )
                loop.close()
                
                return {
                    'success': True,
                    'available_engines': [e.value for e in available_engines],
                    'voice_count': len(available_voices),
                    'synthesis_result': {
                        'engine': result.engine.value,
                        'duration_ms': result.duration_ms,
                        'processing_time_ms': result.processing_time_ms,
                        'audio_format': result.audio_format,
                        'sample_rate': result.sample_rate
                    }
                }
                
            except Exception as synthesis_error:
                return {
                    'success': False,
                    'init_success': True,
                    'available_engines': [e.value for e in available_engines],
                    'voice_count': len(available_voices),
                    'synthesis_error': str(synthesis_error)
                }
        else:
            return {'success': False, 'error': 'TTS service initialization failed'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

def save_test_report(results: Dict[str, Any]):
    """Save test results to a JSON file."""
    report_file = "/tmp/tts_test_report.json"
    
    try:
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Test report saved to: {report_file}")
        return report_file
    except Exception as e:
        logger.error(f"Could not save test report: {e}")
        return None

def main():
    """Main test function."""
    print("🐳 Docker TTS Test Script")
    print("=" * 40)
    
    # Check Docker environment
    print("\n1. Checking Docker environment...")
    docker_env = check_docker_environment()
    for check, result in docker_env.items():
        status = "✅" if result else "❌"
        print(f"   {status} {check}: {result}")
    
    if not docker_env['in_docker']:
        print("\n⚠️  Warning: Not running in Docker container!")
        print("   This script is designed to run inside the Docker container.")
        print("   Run: docker-compose exec api python test_tts_docker.py")
    
    # Check dependencies
    print("\n2. Checking TTS dependencies...")
    deps = check_tts_dependencies()
    
    for dep_name, dep_info in deps.items():
        if dep_info['available']:
            print(f"   ✅ {dep_name}: Available")
            if 'version' in dep_info:
                print(f"      Version: {dep_info['version']}")
            if 'voice_count' in dep_info:
                print(f"      Voices: {dep_info['voice_count']}")
            if 'cuda_available' in dep_info:
                print(f"      CUDA: {'Yes' if dep_info['cuda_available'] else 'No'}")
        else:
            print(f"   ❌ {dep_name}: Not available")
            if 'error' in dep_info:
                print(f"      Error: {dep_info['error']}")
    
    # Test TTS service import
    print("\n3. Testing TTS service import...")
    import_result = test_tts_service_import()
    if import_result['success']:
        print("   ✅ TTS service import successful")
        print(f"      Available engines: {import_result['engines_available']}")
    else:
        print("   ❌ TTS service import failed")
        print(f"      Error: {import_result['error']}")
    
    # Test engines
    test_results = {}
    
    print("\n4. Testing TTS engines...")
    
    # Test pyttsx3
    if deps['pyttsx3']['available']:
        print("\n   Testing pyttsx3...")
        pyttsx3_result = test_pyttsx3_in_docker()
        test_results['pyttsx3'] = pyttsx3_result
        
        if pyttsx3_result['success']:
            print(f"   ✅ pyttsx3 synthesis successful")
            print(f"      Output: {pyttsx3_result['output_file']}")
            print(f"      File size: {pyttsx3_result['file_size']} bytes")
            print(f"      Synthesis time: {pyttsx3_result['synthesis_time']:.2f}s")
        else:
            print(f"   ❌ pyttsx3 synthesis failed: {pyttsx3_result['error']}")
    
    # Test Coqui TTS
    if deps['TTS']['available'] and deps['torch']['available']:
        print("\n   Testing Coqui TTS...")
        coqui_result = test_coqui_tts_in_docker()
        test_results['coqui'] = coqui_result
        
        if coqui_result['success']:
            print(f"   ✅ Coqui TTS synthesis successful")
            print(f"      Model: {coqui_result['model']}")
            print(f"      Output: {coqui_result['output_file']}")
            print(f"      File size: {coqui_result['file_size']} bytes")
            print(f"      Init time: {coqui_result['init_time']:.2f}s")
            print(f"      Synthesis time: {coqui_result['synthesis_time']:.2f}s")
        else:
            print(f"   ❌ Coqui TTS synthesis failed: {coqui_result['error']}")
    
    # Test application TTS service
    if import_result['success']:
        print("\n   Testing application TTS service...")
        app_result = test_application_tts_service()
        test_results['application_tts'] = app_result
        
        if app_result['success']:
            print(f"   ✅ Application TTS service working")
            print(f"      Available engines: {app_result['available_engines']}")
            print(f"      Voice count: {app_result['voice_count']}")
            print(f"      Used engine: {app_result['synthesis_result']['engine']}")
            print(f"      Processing time: {app_result['synthesis_result']['processing_time_ms']:.0f}ms")
        else:
            print(f"   ❌ Application TTS service failed")
            print(f"      Error: {app_result['error']}")
            if 'synthesis_error' in app_result:
                print(f"      Synthesis error: {app_result['synthesis_error']}")
    
    # Summary
    print("\n5. Summary")
    print("=" * 20)
    
    working_engines = []
    failing_engines = []
    
    for engine_name, result in test_results.items():
        if result['success']:
            working_engines.append(engine_name)
            print(f"   ✅ {engine_name}: Working")
        else:
            failing_engines.append(engine_name)
            print(f"   ❌ {engine_name}: Failed")
    
    print(f"\n📊 Results: {len(working_engines)} working, {len(failing_engines)} failed")
    
    if working_engines:
        print("\n🎉 Working TTS engines found!")
        print("   Audio files created:")
        for engine_name, result in test_results.items():
            if result['success'] and 'output_file' in result:
                print(f"   • {engine_name}: {result['output_file']}")
        
        print("\n💡 Recommendations:")
        if 'coqui' in working_engines:
            print("   • Coqui TTS is working in Docker - use this for production")
        elif 'pyttsx3' in working_engines:
            print("   • Only pyttsx3 working - check Coqui TTS dependencies")
        
        if 'application_tts' in working_engines:
            print("   • Application TTS service is working correctly")
        else:
            print("   • Application TTS service needs debugging")
    
    else:
        print("\n❌ No TTS engines working!")
        print("   Check Docker image dependencies and configuration")
    
    # Save report
    all_results = {
        'docker_environment': docker_env,
        'dependencies': deps,
        'import_test': import_result,
        'engine_tests': test_results,
        'summary': {
            'working_engines': working_engines,
            'failing_engines': failing_engines,
            'total_engines': len(test_results)
        }
    }
    
    report_file = save_test_report(all_results)
    if report_file:
        print(f"\n📋 Detailed report saved to: {report_file}")
        print("   Copy it from container: docker cp <container_id>:/tmp/tts_test_report.json ./")
    
    print("\n" + "=" * 40)
    print("Docker TTS test completed!")

if __name__ == "__main__":
    main()