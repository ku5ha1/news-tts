import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict
import logging
import asyncio
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import httpx

load_dotenv()

logger = logging.getLogger(__name__)


class TTSService:
    ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
    _elevenlabs_client = None

    def __init__(self):
        voice_id = os.getenv("ELEVENLABS_VOICE_ID")
        if not voice_id:
            raise RuntimeError("ELEVENLABS_VOICE_ID environment variable not set")
        
        if TTSService._elevenlabs_client is None:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                raise RuntimeError("ELEVENLABS_API_KEY environment variable not set")
            
            import httpx

            TTSService._elevenlabs_client = ElevenLabs(
                api_key=api_key,
                http_client=httpx.Client(timeout=120.0)  
            )
            logger.info("[TTS] ElevenLabs client initialized")

        self.voice_mapping: Dict[str, str] = {
            "en": voice_id, 
            "hi": voice_id,  
            "kn": voice_id,  
        }

    def generate_audio(self, text: str, language: str) -> str:
        """Generate audio using ElevenLabs API"""
        logger.info(f"[TTS] Starting ElevenLabs audio generation for {language}: '{text[:50]}...'")
        
        try:
            voice_id = self.voice_mapping.get(language, self.voice_mapping["en"])
            logger.info(f"[TTS] Using voice ID: {voice_id} for language: {language}")
            
            try:
                audio = TTSService._elevenlabs_client.text_to_speech.convert(
                    text=text,
                    voice_id=voice_id,
                    model_id="eleven_v3", 
                    output_format="mp3_44100_128",
                )
            except Exception as e:
                if "timeout" in str(e).lower() or "read timeout" in str(e).lower():
                    logger.error(f"[TTS] ElevenLabs timeout for {language}: {e}")
                    raise RuntimeError(f"TTS generation timeout for {language}: {str(e)}")
                else:
                    raise
                
            temp_dir = "/tmp" if os.path.exists("/tmp") else "/app/tmp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = Path(temp_dir) / f"audio_{language}_{uuid.uuid4().hex[:8]}.mp3"

            with open(temp_path, "wb") as audio_file:
                for chunk in audio:
                    audio_file.write(chunk)
            
            file_size = os.path.getsize(temp_path)
            logger.info(f"[TTS] ElevenLabs audio file created: {temp_path} (size: {file_size} bytes)")
            
            return str(temp_path)
            
        except ImportError as e:
            logger.error(f"[TTS] ElevenLabs SDK import error: {str(e)}", exc_info=True)
            raise RuntimeError(f"ElevenLabs SDK not available: {str(e)}")
        except ConnectionError as e:
            logger.error(f"[TTS] ElevenLabs API connection error: {str(e)}", exc_info=True)
            raise RuntimeError(f"ElevenLabs API connection failed: {str(e)}")
        except Exception as e:
            logger.error(f"[TTS] ElevenLabs generation failed for {language}: {str(e)}", exc_info=True)
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