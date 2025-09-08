import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict

import torch
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from app.config.settings import Settings

class TTSService:
    _model = None
    _prompt_tokenizer = None
    _desc_tokenizer = None
    _device = None

    def __init__(self):
        self.settings = Settings()

        # Voice/style presets via caption prompts (female newsreader per language)
        self.female_news_caption_by_lang: Dict[str, str] = {
            "en": "A female Indian English newsreader, neutral, clear diction, medium pace, studio recording, no background noise.",
            "hi": "एक महिला हिंदी समाचार वाचक, स्पष्ट उच्चारण, मध्यम गति, न्यूट्रल टोन, स्टूडियो क्वालिटी रिकॉर्डिंग।",
            "kn": "ಮಹಿಳಾ ಕನ್ನಡ ಸುದ್ದಿವಾಚಕಿ, ಸ್ಪಷ್ಟ ಉಚ್ಚಾರಣೆ, ಮಧ್ಯಮ ವೇಗ, ನ್ಯೂಟ್ರಲ್ ಶೈಲಿ, ಸ್ಟುಡಿಯೋ ಗುಣಮಟ್ಟದ ಧ್ವನಿ.",
        }

        # Lazy-load model/tokenizers on first use to reduce startup time
        if TTSService._device is None:
            TTSService._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_model_loaded(self):
        if TTSService._model is not None:
            return

        model_name = os.getenv("INDIC_TTS_MODEL", "ai4bharat/indic-parler-tts")
        # Load model and two tokenizers (prompt + description)
        TTSService._model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(TTSService._device).eval()
        TTSService._prompt_tokenizer = AutoTokenizer.from_pretrained(model_name)
        TTSService._desc_tokenizer = AutoTokenizer.from_pretrained(TTSService._model.config.text_encoder._name_or_path)

    def generate_audio(self, text: str, language: str) -> str:
        """Generate audio file using Indic Parler-TTS and return the file path (WAV)."""
        try:
            self._ensure_model_loaded()

            caption = self.female_news_caption_by_lang.get(language, self.female_news_caption_by_lang["en"])

            # Tokenize description (caption) and prompt (text)
            desc_inputs = TTSService._desc_tokenizer(caption, return_tensors="pt").to(TTSService._device)
            prompt_inputs = TTSService._prompt_tokenizer(text, return_tensors="pt").to(TTSService._device)

            with torch.no_grad():
                generation = TTSService._model.generate(
                    input_ids=desc_inputs.input_ids,
                    attention_mask=desc_inputs.attention_mask,
                    prompt_input_ids=prompt_inputs.input_ids,
                    prompt_attention_mask=prompt_inputs.attention_mask,
                )

            audio_arr = generation.detach().cpu().numpy().squeeze()
            sampling_rate = int(TTSService._model.config.sampling_rate)

            # Create temporary file with explicit directory
            temp_dir = "/tmp" if os.path.exists("/tmp") else "/app/tmp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = Path(temp_dir) / f"audio_{language}_{uuid.uuid4().hex[:8]}.wav"

            # Write WAV file
            sf.write(str(temp_path), audio_arr, sampling_rate)

            return str(temp_path)

        except Exception as e:
            print(f"Error generating audio for {language}: {str(e)}")
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