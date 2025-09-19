import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict
import logging
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class TTSService:
    _elevenlabs_client = None

    def __init__(self):
        # Initialize ElevenLabs client
        if TTSService._elevenlabs_client is None:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                raise RuntimeError("ELEVENLABS_API_KEY environment variable not set")
            
            TTSService._elevenlabs_client = ElevenLabs(api_key=api_key)
            logger.info("[TTS] ElevenLabs client initialized")

        # Voice mapping per language (using appropriate voices for each language)
        self.voice_mapping: Dict[str, str] = {
            "en": "JBFqnCBsd6RMkjVDRZzb",  # Default English voice
            "hi": "JBFqnCBsd6RMkjVDRZzb",  # Use same voice for Hindi (multilingual model)
            "kn": "JBFqnCBsd6RMkjVDRZzb",  # Use same voice for Kannada (multilingual model)
        }

    def generate_audio(self, text: str, language: str) -> str:
        """Generate audio using ElevenLabs API"""
        logger.info(f"[TTS] Starting ElevenLabs audio generation for {language}: '{text[:50]}...'")
        
        try:
            # Get voice ID for language
            voice_id = self.voice_mapping.get(language, self.voice_mapping["en"])
            logger.info(f"[TTS] Using voice ID: {voice_id} for language: {language}")
            
            # Generate audio using ElevenLabs
            audio = TTSService._elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",  # Supports multiple languages
                output_format="mp3_44100_128",
            )
            
            # Create temporary file
            temp_dir = "/tmp" if os.path.exists("/tmp") else "/app/tmp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = Path(temp_dir) / f"audio_{language}_{uuid.uuid4().hex[:8]}.mp3"
            
            # Write audio data to file
            with open(temp_path, "wb") as audio_file:
                for chunk in audio:
                    audio_file.write(chunk)
            
            file_size = os.path.getsize(temp_path)
            logger.info(f"[TTS] ElevenLabs audio file created: {temp_path} (size: {file_size} bytes)")
            
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"[TTS] ElevenLabs generation failed for {language}: {str(e)}", exc_info=True)
            raise RuntimeError(f"TTS generation failed for {language}: {str(e)}")

    def get_audio_duration(self, file_path: str) -> float:
        """Get audio duration in seconds (estimated for MP3)"""
        try:
            if file_path and os.path.exists(file_path):
                # For MP3, estimate duration based on file size
                # Rough estimate: 128kbps MP3 ≈ 16KB per second
                file_size = os.path.getsize(file_path)
                estimated_duration = file_size / (16 * 1024)  # 16KB per second
                return max(1.0, estimated_duration)  # Minimum 1 second
            return 3.5  # Default duration
        except Exception as e:
            logger.warning(f"[TTS] Could not estimate duration for {file_path}: {e}")
            return 3.0

    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        try:
            if file_path and os.path.exists(file_path):
                return os.path.getsize(file_path)
            return 50000  # Default MP3 file size estimate
        except Exception as e:
            logger.warning(f"[TTS] Could not get file size for {file_path}: {e}")
            return 50000