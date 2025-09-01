from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
from pathlib import Path

class TranslationService:
    _instance = None
    _models_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TranslationService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._models_loaded:
            self._load_models()
            TranslationService._models_loaded = True

    def _load_models(self):
        """Load AI4Bharat translation models with caching."""
        try:
            cache_dir = Path(os.environ.get("TRANSFORMERS_CACHE", Path.home() / ".cache" / "huggingface" / "transformers"))
            cache_dir.mkdir(parents=True, exist_ok=True)

            print("🚀 Loading AI4Bharat translation models...")

            # Load en→indic model
            self.tokenizer_en_indic = AutoTokenizer.from_pretrained(
                "ai4bharat/IndicTrans2-en-indic-1B",
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            self.model_en_indic = AutoModelForSeq2SeqLM.from_pretrained(
                "ai4bharat/IndicTrans2-en-indic-1B",
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            # Load indic→en model
            self.tokenizer_indic_en = AutoTokenizer.from_pretrained(
                "ai4bharat/IndicTrans2-indic-en-1B",
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            self.model_indic_en = AutoModelForSeq2SeqLM.from_pretrained(
                "ai4bharat/IndicTrans2-indic-en-1B",
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"✅ Using device: {self.device}")

            self.model_en_indic.to(self.device).eval()
            self.model_indic_en.to(self.device).eval()

            print("✅ AI4Bharat translation models loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading translation models: {str(e)}")
            raise e

    def _translate(self, text: str, source: str, target: str) -> str:
        """Translate text from source language to target language."""
        try:
            if not text.strip():
                return text

            target_token = f"<2{target}> {text}"

            if source == "en":
                tokenizer, model = self.tokenizer_en_indic, self.model_en_indic
            else:
                tokenizer, model = self.tokenizer_indic_en, self.model_indic_en

            with torch.no_grad():
                inputs = tokenizer(target_token, return_tensors="pt", padding=True).to(self.device)
                outputs = model.generate(**inputs, max_length=256)
                return tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            print(f"⚠️ Translation error ({source}→{target}): {str(e)}")
            return text  # Fallback to original text

    def translate_to_all(self, title: str, description: str, source_lang: str):
        """Translate text to all supported languages except the source."""
        try:
            languages = ["en", "hi", "kn"]
            result = {}

            for lang in languages:
                if lang == source_lang:
                    continue
                result[lang] = {
                    "title": self._translate(title, source_lang, lang),
                    "description": self._translate(description, source_lang, lang)
                }
            return result

        except Exception as e:
            print(f"⚠️ Error in translate_to_all: {str(e)}")
            return {}

    def is_models_loaded(self) -> bool:
        return self._models_loaded and hasattr(self, 'model_en_indic') and hasattr(self, 'model_indic_en')

    def get_model_info(self) -> dict:
        return {
            "models_loaded": self.is_models_loaded(),
            "device": str(self.device) if hasattr(self, 'device') else "unknown",
            "en_indic_model": "ai4bharat/IndicTrans2-en-indic-1B",
            "indic_en_model": "ai4bharat/IndicTrans2-indic-en-1B"
        }

translation_service = TranslationService()