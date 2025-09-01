from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
from pathlib import Path
import asyncio

LANGUAGE_TAGS = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "kn": "kan_Knda"
}


class TranslationService:
    _instance = None
    _models_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TranslationService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Don't load models at startup, only initialize placeholders
        if not hasattr(self, "device"):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer_en_indic = None
            self.model_en_indic = None
            self.tokenizer_indic_en = None
            self.model_indic_en = None

    def _load_models(self):
        """Load AI4Bharat translation models if not already loaded."""
        if self._models_loaded:
            return

        try:
            cache_dir = Path(
                os.environ.get(
                    "TRANSFORMERS_CACHE",
                    Path.home() / ".cache" / "huggingface" / "transformers"
                )
            )
            cache_dir.mkdir(parents=True, exist_ok=True)

            print("🚀 Loading AI4Bharat translation models...")

            # en→indic
            self.tokenizer_en_indic = AutoTokenizer.from_pretrained(
                "ai4bharat/IndicTrans2-en-indic-1B",
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            self.model_en_indic = AutoModelForSeq2SeqLM.from_pretrained(
                "ai4bharat/IndicTrans2-en-indic-1B",
                trust_remote_code=True,
                cache_dir=cache_dir
            ).to(self.device).eval()

            # indic→en
            self.tokenizer_indic_en = AutoTokenizer.from_pretrained(
                "ai4bharat/IndicTrans2-indic-en-1B",
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            self.model_indic_en = AutoModelForSeq2SeqLM.from_pretrained(
                "ai4bharat/IndicTrans2-indic-en-1B",
                trust_remote_code=True,
                cache_dir=cache_dir
            ).to(self.device).eval()

            TranslationService._models_loaded = True
            print(f"✅ Models loaded successfully on device: {self.device}")

        except Exception as e:
            print(f"❌ Error loading translation models: {str(e)}")
            raise e

    LANGUAGE_TAGS = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "kn": "kan_Knda"
}

def _translate(self, text: str, source: str, target: str) -> str:
    """Translate text from source to target, lazy-loading models if needed."""
    try:
        if not text.strip():
            return text

        # Load models only when needed
        if not self._models_loaded:
            self._load_models()

        # ✅ Map short code ("en", "hi", "kn") to IndicTrans2 tag
        target_tag = LANGUAGE_TAGS.get(target)
        if not target_tag:
            raise ValueError(f"Unsupported target language: {target}")

        input_text = f"<2{target_tag}> {text}"

        if source == "en":
            tokenizer, model = self.tokenizer_en_indic, self.model_en_indic
        else:
            tokenizer, model = self.tokenizer_indic_en, self.model_indic_en

        with torch.no_grad():
            inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(self.device)
            outputs = model.generate(**inputs, max_length=256)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except Exception as e:
        print(f"⚠️ Translation error ({source}→{target}): {str(e)}")
        return text  # fallback
        
    async def translate_async(self, text: str, source: str, target: str) -> str:
        """Async wrapper so routes can `await` translation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._translate, text, source, target)

    def translate_to_all(self, title: str, description: str, source_lang: str):
        """Translate text to all supported languages except the source."""
        try:
            if not self._models_loaded:
                self._load_models()

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
        return self._models_loaded

    def get_model_info(self) -> dict:
        return {
            "models_loaded": self.is_models_loaded(),
            "device": str(self.device),
            "en_indic_model": "ai4bharat/IndicTrans2-en-indic-1B",
            "indic_en_model": "ai4bharat/IndicTrans2-indic-en-1B"
        }


# ✅ Export a single instance
translation_service = TranslationService()