import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict
import logging
import asyncio
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_TTS_AVAILABLE = True
except ImportError:
    AZURE_TTS_AVAILABLE = False


class TTSService:
    def __init__(self):
        if not AZURE_TTS_AVAILABLE:
            raise RuntimeError("Azure TTS not available - install azure-cognitiveservices-speech")
        
        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION")
        
        if not self.speech_key or not self.speech_region:
            raise RuntimeError("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables not set")

        # Azure voice mapping for different languages
        self.voice_mapping: Dict[str, str] = {
            "en": "en-IN-KavyaNeural",
            "hi": "hi-IN-SwaraNeural", 
            "kn": "kn-IN-SapnaNeural",
        }
        
        logger.info("[TTS] Azure Speech TTS service initialized")

    def generate_audio(self, text: str, language: str) -> str:
        """Generate audio using Azure Speech TTS"""
        logger.info(f"[TTS] Starting Azure TTS generation for {language}: '{text[:50]}...'")
        
        try:
            voice_name = self.voice_mapping.get(language, self.voice_mapping["en"])
            logger.info(f"[TTS] Using voice: {voice_name} for language: {language}")
            
            # Create temp file
            temp_dir = "/tmp" if os.path.exists("/tmp") else "/app/tmp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = Path(temp_dir) / f"audio_{language}_{uuid.uuid4().hex[:8]}.mp3"
            
            # Configure Azure Speech
            speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
            speech_config.speech_synthesis_voice_name = voice_name
            
            audio_config = speechsdk.audio.AudioOutputConfig(filename=str(temp_path))
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config, 
                audio_config=audio_config
            )

            # Synthesize text to speech
            result = synthesizer.speak_text_async(text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                file_size = os.path.getsize(temp_path)
                logger.info(f"[TTS] Azure TTS audio file created: {temp_path} (size: {file_size} bytes)")
                return str(temp_path)
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = speechsdk.CancellationDetails(result)
                logger.error(f"[TTS] Azure TTS cancelled for {language}: {cancellation_details.reason}")
                raise RuntimeError(f"Azure TTS cancelled for {language}: {cancellation_details.reason}")
            else:
                logger.error(f"[TTS] Azure TTS failed for {language}: {result.reason}")
                raise RuntimeError(f"Azure TTS failed for {language}: {result.reason}")
                
        except Exception as e:
            logger.error(f"[TTS] Azure TTS generation failed for {language}: {str(e)}", exc_info=True)
            raise RuntimeError(f"TTS generation failed for {language}: {str(e)}")

    def get_audio_duration(self, file_path: str) -> float:
        """Get audio duration in seconds (estimated for MP3)"""
        try:
            if file_path and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                estimated_duration = file_size / (16 * 1024)  
                return max(1.0, estimated_duration) 
            return 3.5  
        except Exception as e:
            logger.warning(f"[TTS] Could not estimate duration for {file_path}: {e}")
            return 3.0

    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        try:
            if file_path and os.path.exists(file_path):
                return os.path.getsize(file_path)
            return 50000  
        except Exception as e:
            logger.warning(f"[TTS] Could not get file size for {file_path}: {e}")
            return 50000