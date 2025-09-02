import torch
import os
from pathlib import Path
import asyncio
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit import IndicProcessor

# Select model size from env: "1B" or "dist-200M"
MODEL_SIZE = os.getenv("MODEL_SIZE", "1B")

MODEL_NAMES = {
    "en_indic": f"ai4bharat/indictrans2-en-indic-{MODEL_SIZE}",
    "indic_en": f"ai4bharat/indictrans2-indic-en-{MODEL_SIZE}"
}


class TranslationService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TranslationService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "device"):
            # Prefer CUDA if available, else CPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Processor handles normalization, tagging, and detokenization
            self.ip = IndicProcessor(inference=True)

            # Placeholders for lazy model load
            self.tokenizer_en_indic = None
            self.model_en_indic = None
            self.tokenizer_indic_en = None
            self.model_indic_en = None

    def _get_cache_dir(self):
        """Resolve Hugging Face cache dir inside container."""
        cache_dir = Path(
            os.environ.get(
                "TRANSFORMERS_CACHE",
                Path.home() / ".cache" / "huggingface" / "transformers"
            )
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _load_en_indic(self):
        """Load en→indic model lazily."""
        if self.model_en_indic is None:
            print(f"🚀 Loading {MODEL_NAMES['en_indic']} ...")
            cache_dir = self._get_cache_dir()
            self.tokenizer_en_indic = AutoTokenizer.from_pretrained(
                MODEL_NAMES["en_indic"], trust_remote_code=True, cache_dir=cache_dir
            )
            self.model_en_indic = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAMES["en_indic"], trust_remote_code=True, cache_dir=cache_dir
            ).to(self.device).eval()

    def _load_indic_en(self):
        """Load indic→en model lazily."""
        if self.model_indic_en is None:
            print(f"🚀 Loading {MODEL_NAMES['indic_en']} ...")
            cache_dir = self._get_cache_dir()
            self.tokenizer_indic_en = AutoTokenizer.from_pretrained(
                MODEL_NAMES["indic_en"], trust_remote_code=True, cache_dir=cache_dir
            )
            self.model_indic_en = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAMES["indic_en"], trust_remote_code=True, cache_dir=cache_dir
            ).to(self.device).eval()

    def _translate(self, text: str, source: str, target: str) -> str:
        """Translate text using IndicTrans2 official pipeline."""
        try:
            if not text.strip():
                return text

            # Pick direction
            if source == "en":
                self._load_en_indic()
                tokenizer, model = self.tokenizer_en_indic, self.model_en_indic
            else:
                self._load_indic_en()
                tokenizer, model = self.tokenizer_indic_en, self.model_indic_en

            # Preprocess (adds tags + normalization)
            batch = self.ip.preprocess_batch([text], src_lang=source, tgt_lang=target)

            with torch.no_grad():
                inputs = tokenizer(
                    batch,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                ).to(self.device)

                outputs = model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )

            # Decode with proper target tokenizer context
            with tokenizer.as_target_tokenizer():
                decoded = tokenizer.batch_decode(
                    outputs.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

            # Postprocess (detokenization/entity replacement)
            translations = self.ip.postprocess_batch(decoded, lang=target)

            return translations[0] if translations else text

        except Exception as e:
            print(f"⚠️ Translation error ({source}→{target}): {str(e)}")
            return text

    async def translate_async(self, text: str, source: str, target: str) -> str:
        """Async wrapper so routes can await translation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._translate, text, source, target)

    def translate_to_all(self, title: str, description: str, source_lang: str):
        """Translate title/description to all supported langs except source."""
        result = {}
        try:
            languages = ["en", "hi", "kn"]
            for lang in languages:
                if lang == source_lang:
                    continue
                result[lang] = {
                    "title": self._translate(title, source_lang, lang),
                    "description": self._translate(description, source_lang, lang),
                }
            return result
        except Exception as e:
            print(f"⚠️ Error in translate_to_all: {str(e)}")
            return {}

    def get_model_info(self) -> dict:
        return {
            "device": str(self.device),
            "en_indic_model": MODEL_NAMES["en_indic"] if self.model_en_indic else None,
            "indic_en_model": MODEL_NAMES["indic_en"] if self.model_indic_en else None,
        }


# ✅ Export singleton instance
translation_service = TranslationService()
