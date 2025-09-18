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

logger = logging.getLogger(__name__)

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

        # Initialize device with better detection
        if TTSService._device is None:
            if torch.cuda.is_available():
                TTSService._device = torch.device("cuda")
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"[TTS] GPU detected: {device_name}")
            else:
                TTSService._device = torch.device("cpu")
                logger.warning("[TTS] No GPU detected, using CPU")

    def _ensure_model_loaded(self):
        """Load TTS model with comprehensive error handling"""
        if TTSService._model is not None:
            return

        logger.info("[TTS] Loading Indic Parler-TTS model...")
        
        try:
            # Use local model path if available, otherwise Hugging Face
            local_model_path = "/app/models/indic-parler-tts"
            if os.path.exists(local_model_path):
                model_name_or_path = local_model_path
                logger.info(f"[TTS] Using local model: {model_name_or_path}")
            else:
                model_name_or_path = os.getenv("INDIC_TTS_MODEL", "ai4bharat/indic-parler-tts")
                logger.info(f"[TTS] Using remote model: {model_name_or_path}")

            cache_dir = os.getenv("TRANSFORMERS_CACHE", "/app/.cache/huggingface/transformers")
            
            # Load main TTS model
            logger.info("[TTS] Loading ParlerTTS model...")
            TTSService._model = ParlerTTSForConditionalGeneration.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                trust_remote_code=True,
                local_files_only=os.path.exists(local_model_path)
            ).to(
                TTSService._device,
                dtype=torch.float16 if TTSService._device.type == "cuda" else torch.float32,
            ).eval()
            
            # Load prompt tokenizer
            logger.info("[TTS] Loading prompt tokenizer...")
            TTSService._prompt_tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                trust_remote_code=True,
                local_files_only=os.path.exists(local_model_path)
            )
            
            # Load description tokenizer
            logger.info("[TTS] Loading description tokenizer...")
            TTSService._desc_tokenizer = AutoTokenizer.from_pretrained(
                TTSService._model.config.text_encoder._name_or_path,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            logger.info("[TTS] Indic Parler-TTS model loaded successfully")
            
        except Exception as e:
            logger.error(f"[TTS] Failed to load TTS model: {str(e)}", exc_info=True)
            raise RuntimeError(f"TTS model loading failed: {str(e)}")

    def _ensure_indicf5_loaded(self):
        """Load IndicF5 model with error handling"""
        if TTSService._indicf5 is not None:
            return
            
        logger.info("[TTS] Loading IndicF5 model...")
        try:
            model_name = os.getenv("INDICF5_MODEL", "ai4bharat/IndicF5")
            cache_dir = os.getenv("TRANSFORMERS_CACHE", "/app/.cache/huggingface/transformers")
            
            TTSService._indicf5 = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir
            ).to(
                TTSService._device,
                dtype=torch.float16 if TTSService._device.type == "cuda" else torch.float32,
            ).eval()
            
            logger.info("[TTS] IndicF5 model loaded successfully")
            
        except Exception as e:
            logger.error(f"[TTS] Failed to load IndicF5 model: {str(e)}", exc_info=True)
            raise RuntimeError(f"IndicF5 model loading failed: {str(e)}")

    def _get_indicf5_refs(self, language: str) -> tuple[str | None, str | None]:
        """Get IndicF5 reference audio and text for language"""
        key = {"en": "EN", "hi": "HI", "kn": "KN"}.get(language, None)
        if not key:
            return None, None
        audio_path = os.getenv(f"INDICF5_REF_{key}_AUDIO")
        ref_text = os.getenv(f"INDICF5_REF_{key}_TEXT")
        return audio_path, ref_text

    def generate_audio(self, text: str, language: str) -> str:
        """Generate audio file. Prefer IndicF5 with reference voice; fallback to Indic Parler-TTS."""
        logger.info(f"[TTS] Starting audio generation for {language}: '{text[:50]}...'")
        
        try:
            audio_arr = None
            sampling_rate = 24000
            
            # 1) Try IndicF5 if refs are configured
            ref_audio, ref_text = self._get_indicf5_refs(language)
            if ref_audio and ref_text and os.path.exists(ref_audio):
                try:
                    logger.info(f"[TTS] Attempting IndicF5 for {language}")
                    self._ensure_indicf5_loaded()
                    
                    with torch.no_grad():
                        audio = TTSService._indicf5(
                            text,
                            ref_audio_path=ref_audio,
                            ref_text=ref_text,
                        )
                    
                    # Normalize to float32
                    import numpy as np
                    if hasattr(audio, "dtype") and audio.dtype == np.int16:
                        audio = audio.astype(np.float32) / 32768.0
                    audio_arr = np.array(audio, dtype=np.float32)
                    logger.info(f"[TTS] IndicF5 generation successful for {language}")
                    
                except Exception as e:
                    logger.warning(f"[TTS] IndicF5 generation failed for {language}: {str(e)}. Falling back to Parler-TTS.")
                    audio_arr = None

            # 2) Fallback to Indic Parler-TTS
            if audio_arr is None:
                logger.info(f"[TTS] Using Parler-TTS for {language}")
                self._ensure_model_loaded()
                
                caption = self.female_news_caption_by_lang.get(language, self.female_news_caption_by_lang["en"])
                logger.info(f"[TTS] Using caption: {caption[:50]}...")
                
                desc_inputs = TTSService._desc_tokenizer(caption, return_tensors="pt").to(TTSService._device)
                prompt_inputs = TTSService._prompt_tokenizer(text, return_tensors="pt").to(TTSService._device)
                
                logger.info("[TTS] Starting model generation...")
                with torch.no_grad():
                    generation = TTSService._model.generate(
                        input_ids=desc_inputs.input_ids,
                        attention_mask=desc_inputs.attention_mask,
                        prompt_input_ids=prompt_inputs.input_ids,
                        prompt_attention_mask=prompt_inputs.attention_mask,
                    )
                
                audio_arr = generation.detach().cpu().numpy().squeeze()
                sampling_rate = int(TTSService._model.config.sampling_rate)
                logger.info(f"[TTS] Parler-TTS generation successful for {language}")

            # Create temporary file
            temp_dir = "/tmp" if os.path.exists("/tmp") else "/app/tmp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = Path(temp_dir) / f"audio_{language}_{uuid.uuid4().hex[:8]}.wav"

            logger.info(f"[TTS] Writing audio file: {temp_path}")
            # Write WAV file (PCM_16)
            sf.write(str(temp_path), audio_arr, sampling_rate, subtype='PCM_16')
            
            file_size = os.path.getsize(temp_path)
            logger.info(f"[TTS] Audio file created: {temp_path} (size: {file_size} bytes)")
            
            return str(temp_path)

        except Exception as e:
            logger.error(f"[TTS] Error generating audio for {language}: {str(e)}", exc_info=True)
            raise RuntimeError(f"TTS generation failed for {language}: {str(e)}")  # DON'T RETURN None!

    def get_audio_duration(self, file_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            if file_path and os.path.exists(file_path):
                # Use soundfile to get actual duration
                info = sf.info(file_path)
                return float(info.duration)
            return 3.5  # Default duration
        except Exception as e:
            logger.warning(f"[TTS] Could not get duration for {file_path}: {e}")
            return 3.0

    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        try:
            if file_path and os.path.exists(file_path):
                return os.path.getsize(file_path)
            return 687168  # Default file size
        except Exception as e:
            logger.warning(f"[TTS] Could not get file size for {file_path}: {e}")
            return 687168