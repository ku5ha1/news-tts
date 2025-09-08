import os
import tempfile
import uuid
from elevenlabs import generate, save, set_api_key
from app.config.settings import Settings

class TTSService:
    def __init__(self):
        self.settings = Settings()
        set_api_key(self.settings.ELEVENLABS_API_KEY)
        
        # Voice mappings for different languages
        self.voice_mappings = {
            "en": "0s2MqkqwzPYZVFGZpMXE",  
            "hi": "0s2MqkqwzPYZVFGZpMXE",  
            "kn": "0s2MqkqwzPYZVFGZpMXE"   
        }

    def generate_audio(self, text: str, language: str) -> str:
        """Generate audio file and return the file path"""
        try:
            # Select appropriate voice for the language
            voice_id = self.voice_mappings.get(language, self.voice_mappings["en"])
            
            # Generate audio using ElevenLabs
            audio = generate(
                text=text,
                voice=voice_id,
                model="eleven_multilingual_v2"
            )
            
            # Create temporary file with explicit directory
            temp_dir = "/tmp" if os.path.exists("/tmp") else "/app/tmp"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir, exist_ok=True)
            
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=".mp3",
                prefix=f"audio_{language}_{uuid.uuid4().hex[:8]}_",
                dir=temp_dir
            )
            
            # Save audio to file
            save(audio, temp_file.name)
            
            return temp_file.name
            
        except Exception as e:
            print(f"Error generating audio for {language}: {str(e)}")
            # Return a placeholder file path for now
            return None

    def get_audio_duration(self, file_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            # This is a placeholder - in production you'd use a library like pydub
            return 3.5  # Default duration
        except Exception:
            return 3.0

    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        try:
            return os.path.getsize(file_path)
        except Exception:
            return 687168  # Default file size