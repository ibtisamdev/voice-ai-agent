#!/usr/bin/env python3
"""
Simple TTS Implementation
A simplified, reliable text-to-speech implementation focusing on working engines.
"""

import os
import sys
import logging
import tempfile
import wave
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTSEngine(Enum):
    """Available TTS engines."""
    PYTTSX3 = "pyttsx3"
    COQUI = "coqui"
    AUTO = "auto"

@dataclass
class TTSResult:
    """Result of TTS synthesis."""
    success: bool
    audio_file: Optional[str] = None
    error: Optional[str] = None
    engine_used: Optional[str] = None
    synthesis_time: Optional[float] = None
    duration: Optional[float] = None
    file_size: Optional[int] = None
    sample_rate: Optional[int] = None

class SimpleTTS:
    """Simple, reliable TTS implementation."""
    
    def __init__(self, preferred_engine: TTSEngine = TTSEngine.AUTO):
        self.preferred_engine = preferred_engine
        self.available_engines = []
        self._check_available_engines()
        
        # Choose default engine
        if preferred_engine == TTSEngine.AUTO:
            self.default_engine = self._choose_best_engine()
        else:
            self.default_engine = preferred_engine
            
        logger.info(f"SimpleTTS initialized with default engine: {self.default_engine.value}")
        logger.info(f"Available engines: {[e.value for e in self.available_engines]}")
    
    def _check_available_engines(self):
        """Check which TTS engines are available."""
        # Check pyttsx3
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.stop()
            self.available_engines.append(TTSEngine.PYTTSX3)
            logger.info("✅ pyttsx3 is available")
        except Exception as e:
            logger.warning(f"❌ pyttsx3 not available: {e}")
        
        # Check Coqui TTS
        try:
            import torch
            from TTS.api import TTS
            self.available_engines.append(TTSEngine.COQUI)
            logger.info("✅ Coqui TTS is available")
        except Exception as e:
            logger.warning(f"❌ Coqui TTS not available: {e}")
    
    def _choose_best_engine(self) -> TTSEngine:
        """Choose the best available engine."""
        if TTSEngine.COQUI in self.available_engines:
            return TTSEngine.COQUI
        elif TTSEngine.PYTTSX3 in self.available_engines:
            return TTSEngine.PYTTSX3
        else:
            raise RuntimeError("No TTS engines available!")
    
    def synthesize_pyttsx3(self, text: str, output_file: str, 
                          rate: int = 150, volume: float = 1.0) -> TTSResult:
        """Synthesize speech using pyttsx3."""
        try:
            import pyttsx3
            
            start_time = time.time()
            
            engine = pyttsx3.init()
            engine.setProperty('rate', rate)
            engine.setProperty('volume', volume)
            
            # Use a specific voice if available
            voices = engine.getProperty('voices')
            if voices and len(voices) > 0:
                # Try to find a good quality voice
                for voice in voices:
                    if 'zira' in voice.name.lower() or 'david' in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
                else:
                    engine.setProperty('voice', voices[0].id)
            
            engine.save_to_file(text, output_file)
            engine.runAndWait()
            
            synthesis_time = time.time() - start_time
            
            # Check if file was created and get info
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                
                # Try to get audio duration
                try:
                    with wave.open(output_file, 'rb') as wav_file:
                        duration = wav_file.getnframes() / wav_file.getframerate()
                        sample_rate = wav_file.getframerate()
                except:
                    duration = None
                    sample_rate = None
                
                return TTSResult(
                    success=True,
                    audio_file=output_file,
                    engine_used="pyttsx3",
                    synthesis_time=synthesis_time,
                    duration=duration,
                    file_size=file_size,
                    sample_rate=sample_rate
                )
            else:
                return TTSResult(success=False, error="Output file not created", engine_used="pyttsx3")
                
        except Exception as e:
            return TTSResult(success=False, error=str(e), engine_used="pyttsx3")
    
    def synthesize_coqui(self, text: str, output_file: str, model_name: str = None) -> TTSResult:
        """Synthesize speech using Coqui TTS."""
        try:
            os.environ["COQUI_TOS_AGREED"] = "1"
            from TTS.api import TTS
            
            # Choose model
            if model_name is None:
                # Try different models in order of preference (fast to slow)
                models_to_try = [
                    "tts_models/en/ljspeech/fast_pitch",
                    "tts_models/en/ljspeech/tacotron2-DDC",
                    "tts_models/en/ljspeech/glow-tts",
                ]
            else:
                models_to_try = [model_name]
            
            for model in models_to_try:
                try:
                    logger.info(f"Trying Coqui model: {model}")
                    start_time = time.time()
                    
                    tts = TTS(model_name=model)
                    tts.tts_to_file(text=text, file_path=output_file)
                    
                    synthesis_time = time.time() - start_time
                    
                    # Check if file was created
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
                        
                        logger.info(f"✅ Coqui synthesis successful with model: {model}")
                        return TTSResult(
                            success=True,
                            audio_file=output_file,
                            engine_used=f"coqui ({model})",
                            synthesis_time=synthesis_time,
                            duration=duration,
                            file_size=file_size,
                            sample_rate=sample_rate
                        )
                    
                except Exception as e:
                    logger.warning(f"Model {model} failed: {e}")
                    continue
            
            return TTSResult(success=False, error="All Coqui models failed", engine_used="coqui")
            
        except Exception as e:
            return TTSResult(success=False, error=str(e), engine_used="coqui")
    
    def synthesize(self, text: str, output_file: str = None, 
                  engine: TTSEngine = None, **kwargs) -> TTSResult:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_file: Output file path (optional, will create temp file if None)
            engine: Specific engine to use (optional, uses default if None)
            **kwargs: Engine-specific parameters
        
        Returns:
            TTSResult with synthesis results
        """
        if not text.strip():
            return TTSResult(success=False, error="Empty text provided")
        
        # Create output file if not provided
        if output_file is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_file = temp_file.name
            temp_file.close()
        
        # Choose engine
        if engine is None:
            engine = self.default_engine
        
        # Check if requested engine is available
        if engine not in self.available_engines:
            logger.warning(f"Requested engine {engine.value} not available, falling back to default")
            engine = self.default_engine
        
        logger.info(f"Synthesizing with {engine.value}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Synthesize with chosen engine
        if engine == TTSEngine.PYTTSX3:
            return self.synthesize_pyttsx3(text, output_file, **kwargs)
        elif engine == TTSEngine.COQUI:
            return self.synthesize_coqui(text, output_file, **kwargs)
        else:
            return TTSResult(success=False, error=f"Unknown engine: {engine}")
    
    def test_all_engines(self, test_text: str = "Hello, this is a test of text to speech.") -> Dict[str, TTSResult]:
        """Test all available engines."""
        results = {}
        
        for engine in self.available_engines:
            logger.info(f"Testing engine: {engine.value}")
            
            # Create temp file for this engine
            with tempfile.NamedTemporaryFile(suffix=f'_{engine.value}.wav', delete=False) as tmp:
                output_file = tmp.name
            
            result = self.synthesize(test_text, output_file, engine)
            results[engine.value] = result
            
            if result.success:
                logger.info(f"✅ {engine.value} test successful: {result.audio_file}")
            else:
                logger.error(f"❌ {engine.value} test failed: {result.error}")
        
        return results

def main():
    """Main function for testing SimpleTTS."""
    print("🎙️  Simple TTS Test")
    print("=" * 30)
    
    try:
        # Initialize SimpleTTS
        tts = SimpleTTS()
        
        if not tts.available_engines:
            print("❌ No TTS engines available!")
            print("Please install at least one TTS engine:")
            print("  pip install pyttsx3")
            print("  pip install TTS torch torchaudio")
            return
        
        # Test text
        test_text = "Hello! This is a simple text to speech test. The quick brown fox jumps over the lazy dog."
        
        print(f"\n📝 Test text: {test_text}")
        print(f"🎯 Default engine: {tts.default_engine.value}")
        
        # Test default engine
        print(f"\n1. Testing default engine ({tts.default_engine.value})...")
        result = tts.synthesize(test_text)
        
        if result.success:
            print(f"✅ Success!")
            print(f"   Output: {result.audio_file}")
            print(f"   Engine: {result.engine_used}")
            print(f"   Time: {result.synthesis_time:.2f}s")
            if result.duration:
                print(f"   Duration: {result.duration:.2f}s")
            if result.file_size:
                print(f"   File size: {result.file_size} bytes")
        else:
            print(f"❌ Failed: {result.error}")
        
        # Test all available engines
        print(f"\n2. Testing all available engines...")
        all_results = tts.test_all_engines(test_text)
        
        print(f"\n📊 Summary:")
        for engine_name, result in all_results.items():
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            print(f"   {engine_name}: {status}")
            if result.success:
                print(f"      File: {result.audio_file}")
            else:
                print(f"      Error: {result.error}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        successful_engines = [name for name, result in all_results.items() if result.success]
        
        if "coqui" in [name.split()[0] for name in successful_engines]:
            print("   • Coqui TTS is working - excellent quality!")
            print("   • You can use Coqui TTS in your application")
        elif "pyttsx3" in successful_engines:
            print("   • pyttsx3 is working - basic quality")
            print("   • Consider installing Coqui TTS for better quality:")
            print("     pip install TTS torch torchaudio")
        
        print(f"\n🎵 Audio files created:")
        for name, result in all_results.items():
            if result.success:
                print(f"   • {name}: {result.audio_file}")
        
        print(f"\n💻 Usage example:")
        print(f"   from simple_tts import SimpleTTS")
        print(f"   tts = SimpleTTS()")
        print(f"   result = tts.synthesize('Hello world!')")
        print(f"   if result.success:")
        print(f"       print(f'Audio saved to: {{result.audio_file}}')")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()