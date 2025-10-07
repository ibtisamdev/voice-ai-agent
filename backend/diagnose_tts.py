#!/usr/bin/env python3
"""
TTS Diagnostic Script
Comprehensive diagnosis of TTS system configuration and issues.
"""

import os
import sys
import logging
import subprocess
import platform
import tempfile
import json
import wave
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTSDiagnostic:
    """Comprehensive TTS diagnostic tool."""
    
    def __init__(self):
        self.results = {}
        self.audio_files = []
        
    def check_system_info(self) -> Dict[str, Any]:
        """Check system information."""
        system_info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'machine': platform.machine(),
            'system': platform.system(),
            'python_executable': sys.executable,
            'working_directory': os.getcwd(),
            'environment_variables': {}
        }
        
        # Check relevant environment variables
        env_vars = [
            'PYTHONPATH', 'PATH', 'CUDA_VISIBLE_DEVICES', 'TORCH_HOME',
            'HF_HOME', 'TRANSFORMERS_CACHE', 'TTS_HOME', 'COQUI_TOS_AGREED',
            'AZURE_SPEECH_KEY', 'ELEVENLABS_API_KEY'
        ]
        
        for var in env_vars:
            system_info['environment_variables'][var] = os.environ.get(var, 'Not set')
        
        return system_info
    
    def check_audio_system(self) -> Dict[str, Any]:
        """Check audio system capabilities."""
        audio_info = {
            'audio_devices': [],
            'audio_libraries': {},
            'pulseaudio_running': False,
            'alsa_available': False
        }
        
        # Check for audio system commands
        commands_to_check = [
            ('aplay', 'ALSA playback'),
            ('pactl', 'PulseAudio control'),
            ('ffmpeg', 'FFmpeg'),
            ('espeak', 'eSpeak TTS'),
            ('festival', 'Festival TTS'),
            ('sox', 'SoX audio processing')
        ]
        
        for command, description in commands_to_check:
            try:
                result = subprocess.run(['which', command], capture_output=True, text=True)
                available = result.returncode == 0
                audio_info['audio_libraries'][command] = {
                    'available': available,
                    'path': result.stdout.strip() if available else None,
                    'description': description
                }
            except:
                audio_info['audio_libraries'][command] = {
                    'available': False,
                    'description': description
                }
        
        # Check PulseAudio
        try:
            result = subprocess.run(['pgrep', 'pulseaudio'], capture_output=True)
            audio_info['pulseaudio_running'] = result.returncode == 0
        except:
            pass
        
        # Check ALSA
        audio_info['alsa_available'] = os.path.exists('/proc/asound/cards')
        
        # Try to list audio devices
        try:
            if audio_info['audio_libraries']['aplay']['available']:
                result = subprocess.run(['aplay', '-l'], capture_output=True, text=True)
                if result.returncode == 0:
                    audio_info['alsa_devices'] = result.stdout
        except:
            pass
        
        return audio_info
    
    def check_python_packages(self) -> Dict[str, Any]:
        """Check Python package installations."""
        packages_info = {}
        
        # Core packages
        packages_to_check = [
            'torch', 'torchaudio', 'numpy', 'scipy', 'librosa', 'soundfile',
            'TTS', 'pyttsx3', 'transformers', 'sentence-transformers',
            'azure.cognitiveservices.speech', 'requests', 'aiohttp'
        ]
        
        for package_name in packages_to_check:
            try:
                if package_name == 'azure.cognitiveservices.speech':
                    import azure.cognitiveservices.speech as speech_sdk
                    packages_info[package_name] = {
                        'available': True,
                        'version': getattr(speech_sdk, '__version__', 'unknown'),
                        'location': speech_sdk.__file__
                    }
                else:
                    package = __import__(package_name)
                    packages_info[package_name] = {
                        'available': True,
                        'version': getattr(package, '__version__', 'unknown'),
                        'location': getattr(package, '__file__', 'unknown')
                    }
                    
                    # Special checks for specific packages
                    if package_name == 'torch':
                        packages_info[package_name].update({
                            'cuda_available': package.cuda.is_available(),
                            'cuda_version': package.version.cuda if package.cuda.is_available() else None,
                            'device_count': package.cuda.device_count() if package.cuda.is_available() else 0
                        })
                    
                    elif package_name == 'TTS':
                        from TTS.api import TTS
                        packages_info[package_name].update({
                            'models_available': len(TTS.list_models()),
                            'sample_models': TTS.list_models()[:5]
                        })
                    
                    elif package_name == 'pyttsx3':
                        engine = package.init()
                        voices = engine.getProperty('voices')
                        packages_info[package_name].update({
                            'voice_count': len(voices) if voices else 0,
                            'sample_voices': [
                                {'id': v.id, 'name': v.name, 'languages': v.languages}
                                for v in (voices[:3] if voices else [])
                            ]
                        })
                        engine.stop()
                        
            except Exception as e:
                packages_info[package_name] = {
                    'available': False,
                    'error': str(e)
                }
        
        return packages_info
    
    def check_file_permissions(self) -> Dict[str, Any]:
        """Check file and directory permissions."""
        permissions_info = {}
        
        # Directories to check
        dirs_to_check = [
            '/tmp',
            '/app/models' if os.path.exists('/app') else './models',
            '/app/cache' if os.path.exists('/app') else './cache',
            '.'
        ]
        
        for directory in dirs_to_check:
            if os.path.exists(directory):
                try:
                    # Test write permissions
                    test_file = os.path.join(directory, 'tts_test_write.tmp')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    
                    permissions_info[directory] = {
                        'exists': True,
                        'readable': os.access(directory, os.R_OK),
                        'writable': True,
                        'executable': os.access(directory, os.X_OK)
                    }
                except:
                    permissions_info[directory] = {
                        'exists': True,
                        'readable': os.access(directory, os.R_OK),
                        'writable': False,
                        'executable': os.access(directory, os.X_OK)
                    }
            else:
                permissions_info[directory] = {'exists': False}
        
        return permissions_info
    
    def test_tts_engines(self) -> Dict[str, Any]:
        """Test individual TTS engines."""
        engine_results = {}
        
        test_text = "This is a diagnostic test of text to speech synthesis."
        
        # Test pyttsx3
        engine_results['pyttsx3'] = self._test_pyttsx3(test_text)
        
        # Test Coqui TTS
        engine_results['coqui'] = self._test_coqui_tts(test_text)
        
        # Test system espeak if available
        engine_results['espeak'] = self._test_espeak(test_text)
        
        return engine_results
    
    def _test_pyttsx3(self, text: str) -> Dict[str, Any]:
        """Test pyttsx3 engine."""
        try:
            import pyttsx3
            
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            
            # Create test file
            output_file = tempfile.mktemp(suffix='_pyttsx3_test.wav')
            
            start_time = time.time()
            engine.save_to_file(text, output_file)
            engine.runAndWait()
            synthesis_time = time.time() - start_time
            
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                self.audio_files.append(output_file)
                
                # Try to get audio info
                try:
                    with wave.open(output_file, 'rb') as wav_file:
                        duration = wav_file.getnframes() / wav_file.getframerate()
                        sample_rate = wav_file.getframerate()
                        channels = wav_file.getnchannels()
                except:
                    duration = sample_rate = channels = None
                
                return {
                    'success': True,
                    'output_file': output_file,
                    'file_size': file_size,
                    'synthesis_time': synthesis_time,
                    'duration': duration,
                    'sample_rate': sample_rate,
                    'channels': channels,
                    'voice_count': len(voices) if voices else 0
                }
            else:
                return {'success': False, 'error': 'Output file not created'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_coqui_tts(self, text: str) -> Dict[str, Any]:
        """Test Coqui TTS engine."""
        try:
            os.environ["COQUI_TOS_AGREED"] = "1"
            from TTS.api import TTS
            
            # Try a fast model
            models_to_try = [
                "tts_models/en/ljspeech/fast_pitch",
                "tts_models/en/ljspeech/tacotron2-DDC",
                "tts_models/en/ljspeech/glow-tts"
            ]
            
            for model_name in models_to_try:
                try:
                    start_init = time.time()
                    tts = TTS(model_name=model_name)
                    init_time = time.time() - start_init
                    
                    output_file = tempfile.mktemp(suffix=f'_coqui_{model_name.split("/")[-1]}_test.wav')
                    
                    start_synthesis = time.time()
                    tts.tts_to_file(text=text, file_path=output_file)
                    synthesis_time = time.time() - start_synthesis
                    
                    if os.path.exists(output_file):
                        file_size = os.path.getsize(output_file)
                        self.audio_files.append(output_file)
                        
                        # Try to get audio info
                        try:
                            with wave.open(output_file, 'rb') as wav_file:
                                duration = wav_file.getnframes() / wav_file.getframerate()
                                sample_rate = wav_file.getframerate()
                                channels = wav_file.getnchannels()
                        except:
                            duration = sample_rate = channels = None
                        
                        return {
                            'success': True,
                            'model': model_name,
                            'output_file': output_file,
                            'file_size': file_size,
                            'init_time': init_time,
                            'synthesis_time': synthesis_time,
                            'duration': duration,
                            'sample_rate': sample_rate,
                            'channels': channels
                        }
                except Exception as model_error:
                    logger.warning(f"Coqui model {model_name} failed: {model_error}")
                    continue
            
            return {'success': False, 'error': 'All Coqui models failed'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _test_espeak(self, text: str) -> Dict[str, Any]:
        """Test espeak system TTS."""
        try:
            output_file = tempfile.mktemp(suffix='_espeak_test.wav')
            
            # Run espeak command
            cmd = ['espeak', '-w', output_file, text]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                self.audio_files.append(output_file)
                
                return {
                    'success': True,
                    'output_file': output_file,
                    'file_size': file_size,
                    'command': ' '.join(cmd)
                }
            else:
                return {
                    'success': False,
                    'error': f"espeak failed: {result.stderr}",
                    'return_code': result.returncode
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run complete TTS diagnostic."""
        logger.info("Starting TTS diagnostic...")
        
        self.results = {
            'timestamp': time.time(),
            'system_info': self.check_system_info(),
            'audio_system': self.check_audio_system(),
            'python_packages': self.check_python_packages(),
            'file_permissions': self.check_file_permissions(),
            'tts_engines': self.test_tts_engines()
        }
        
        # Analysis
        self.results['analysis'] = self._analyze_results()
        
        return self.results
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze diagnostic results and provide recommendations."""
        analysis = {
            'working_engines': [],
            'failing_engines': [],
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check TTS engines
        for engine_name, result in self.results['tts_engines'].items():
            if result['success']:
                analysis['working_engines'].append(engine_name)
            else:
                analysis['failing_engines'].append({
                    'engine': engine_name,
                    'error': result['error']
                })
        
        # Check for critical issues
        if not self.results['python_packages']['torch']['available']:
            analysis['critical_issues'].append("PyTorch not available - required for Coqui TTS")
        
        if not self.results['python_packages']['TTS']['available']:
            analysis['critical_issues'].append("Coqui TTS package not available")
        
        if not self.results['python_packages']['pyttsx3']['available']:
            analysis['critical_issues'].append("pyttsx3 not available - no fallback TTS")
        
        # Check for warnings
        if not self.results['audio_system']['pulseaudio_running']:
            analysis['warnings'].append("PulseAudio not running - may affect audio playback")
        
        if not self.results['file_permissions'].get('/tmp', {}).get('writable', False):
            analysis['warnings'].append("Cannot write to /tmp - may affect temporary audio files")
        
        # Generate recommendations
        if not analysis['working_engines']:
            analysis['recommendations'].append("Install TTS dependencies: pip install pyttsx3 TTS torch torchaudio")
        
        if 'coqui' not in analysis['working_engines'] and self.results['python_packages']['TTS']['available']:
            analysis['recommendations'].append("Coqui TTS available but not working - check CUDA/model downloads")
        
        if len(analysis['working_engines']) == 1 and 'pyttsx3' in analysis['working_engines']:
            analysis['recommendations'].append("Only pyttsx3 working - install Coqui TTS for better quality")
        
        return analysis
    
    def save_report(self, filename: str = None) -> str:
        """Save diagnostic report to file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"tts_diagnostic_report_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            return filename
        except Exception as e:
            logger.error(f"Could not save report: {e}")
            return None
    
    def print_summary(self):
        """Print diagnostic summary."""
        print("\n🔍 TTS Diagnostic Summary")
        print("=" * 40)
        
        # System info
        system = self.results['system_info']
        print(f"\n📋 System: {system['system']} {system['architecture'][0]}")
        print(f"🐍 Python: {system['python_version'].split()[0]}")
        
        # Package status
        packages = self.results['python_packages']
        print(f"\n📦 Key Packages:")
        for pkg in ['torch', 'TTS', 'pyttsx3']:
            if pkg in packages:
                status = "✅" if packages[pkg]['available'] else "❌"
                version = packages[pkg].get('version', 'unknown') if packages[pkg]['available'] else ''
                print(f"   {status} {pkg}: {version}")
        
        # TTS engines
        engines = self.results['tts_engines']
        print(f"\n🎙️  TTS Engines:")
        for engine_name, result in engines.items():
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {engine_name}")
            if result['success'] and 'output_file' in result:
                print(f"      Audio: {result['output_file']}")
        
        # Analysis
        analysis = self.results['analysis']
        print(f"\n📊 Results:")
        print(f"   Working engines: {len(analysis['working_engines'])}")
        print(f"   Failed engines: {len(analysis['failing_engines'])}")
        print(f"   Critical issues: {len(analysis['critical_issues'])}")
        print(f"   Warnings: {len(analysis['warnings'])}")
        
        # Issues
        if analysis['critical_issues']:
            print(f"\n🚨 Critical Issues:")
            for issue in analysis['critical_issues']:
                print(f"   • {issue}")
        
        if analysis['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in analysis['warnings']:
                print(f"   • {warning}")
        
        # Recommendations
        if analysis['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"   • {rec}")
        
        # Audio files
        if self.audio_files:
            print(f"\n🎵 Test Audio Files:")
            for audio_file in self.audio_files:
                print(f"   • {audio_file}")

def main():
    """Main diagnostic function."""
    print("🔍 TTS Diagnostic Tool")
    print("=" * 30)
    
    diagnostic = TTSDiagnostic()
    
    try:
        # Run full diagnostic
        results = diagnostic.run_full_diagnostic()
        
        # Print summary
        diagnostic.print_summary()
        
        # Save report
        report_file = diagnostic.save_report()
        if report_file:
            print(f"\n📋 Detailed report saved to: {report_file}")
        
        print("\n" + "=" * 30)
        print("Diagnostic completed!")
        
        # Return exit code based on results
        if diagnostic.results['analysis']['working_engines']:
            return 0  # Success
        else:
            return 1  # No working engines
            
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return 2  # Diagnostic error

if __name__ == "__main__":
    sys.exit(main())