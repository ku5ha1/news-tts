import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict

import torch
import logging
import soundfile as sf
from transformers import AutoTokenizer, AutoModel
from parler_tts import ParlerTTSForConditionalGeneration
from app.config.settings import Settings

class TTSService:
    _model = None
    _prompt_tokenizer = None
    _desc_tokenizer = None
    _device = None
    _indicf5 = None

    def __init__(self):
        self.settings = Settings()

        # Voice/style presets via caption prompts (female newsreader per language)
        self.female_news_caption_by_lang: Dict[str, str] = {
            "en": "Adult female Indian English news anchor, neutral news style, clear diction, slightly higher pitch, medium pace, studio-quality recording, no background noise.",
            "hi": "प्रौढ़ महिला हिंदी समाचार वाचक, न्यूट्रल न्यूज़ शैली, स्पष्ट उच्चारण, थोड़ी ऊँची पिच, मध्यम गति, स्टूडियो-क्वालिटी रिकॉर्डिंग, बिना पृष्ठभूमि शोर।",
            "kn": "ಪ್ರೌಢ ಮಹಿಳಾ ಕನ್ನಡ ಸುದ್ದಿವಾಚಕಿ, ನ್ಯೂಟ್ರಲ್ ನ್ಯೂಸ್ ಶೈಲಿ, ಸ್ಪಷ್ಟ ಉಚ್ಚಾರಣೆ, ಸ್ವಲ್ಪ ಹೆಚ್ಚಿನ ಪಿಚ್, ಮಧ್ಯಮ ವೇಗ, ಸ್ಟುಡಿಯೋ-ಗುಣಮಟ್ಟದ ಧ್ವನಿ, ಹಿನ್ನಲೆ ಶಬ್ದವಿಲ್ಲದೆ.",
        }

        # Lazy-load model/tokenizers on first use to reduce startup time
        if TTSService._device is None:
            TTSService._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger = logging.getLogger(__name__)
            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            logger.info(f"Torch device (TTS): {device_name}")

    def _ensure_model_loaded(self):
        if TTSService._model is not None:
            return

        model_name = os.getenv("INDIC_TTS_MODEL", "ai4bharat/indic-parler-tts")
        # Load model and two tokenizers (prompt + description)
        TTSService._model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(
            TTSService._device,
            dtype=torch.float16 if TTSService._device.type == "cuda" else torch.float32,
        ).eval()
        TTSService._prompt_tokenizer = AutoTokenizer.from_pretrained(model_name)
        TTSService._desc_tokenizer = AutoTokenizer.from_pretrained(TTSService._model.config.text_encoder._name_or_path)

    def _ensure_indicf5_loaded(self):
        if TTSService._indicf5 is not None:
            return
        model_name = os.getenv("INDICF5_MODEL", "ai4bharat/IndicF5")
        TTSService._indicf5 = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(
            TTSService._device,
            dtype=torch.float16 if TTSService._device.type == "cuda" else torch.float32,
        ).eval()

    def _get_indicf5_refs(self, language: str) -> tuple[str | None, str | None]:
        # Expect env like INDICF5_REF_EN_AUDIO, INDICF5_REF_EN_TEXT, etc.
        key = {"en": "EN", "hi": "HI", "kn": "KN"}.get(language, None)
        if not key:
            return None, None
        audio_path = os.getenv(f"INDICF5_REF_{key}_AUDIO")
        ref_text = os.getenv(f"INDICF5_REF_{key}_TEXT")
        return audio_path, ref_text

    def generate_audio(self, text: str, language: str) -> str:
        """Generate audio file. Prefer IndicF5 with reference voice; fallback to Indic Parler-TTS."""
        try:
            # 1) Try IndicF5 if refs are configured
            ref_audio, ref_text = self._get_indicf5_refs(language)
            if ref_audio and ref_text and os.path.exists(ref_audio):
                try:
                    self._ensure_indicf5_loaded()
                    with torch.no_grad():
                        audio = TTSService._indicf5(
                            text,
                            ref_audio_path=ref_audio,
                            ref_text=ref_text,
                        )
                    # Normalize to float32 and save at 24 kHz
                    import numpy as np
                    if hasattr(audio, "dtype") and audio.dtype == np.int16:
                        audio = audio.astype(np.float32) / 32768.0
                    audio_arr = np.array(audio, dtype=np.float32)
                    sampling_rate = 24000
                except Exception as e:
                    print(f"IndicF5 generation failed for {language}: {e}. Falling back to Parler-TTS.")
                    audio_arr = None
            else:
                audio_arr = None

            # 2) Fallback to Indic Parler-TTS
            if audio_arr is None:
                self._ensure_model_loaded()
                caption = self.female_news_caption_by_lang.get(language, self.female_news_caption_by_lang["en"])
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

            # Write WAV file (PCM_16)
            sf.write(str(temp_path), audio_arr, sampling_rate, subtype='PCM_16')

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