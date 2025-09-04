import torch
import os
from pathlib import Path
import asyncio
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit import IndicProcessor
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Select model size from env: "1B" or "dist-200M"
MODEL_SIZE = os.getenv("MODEL_SIZE", "dist-200M")  # Use smaller model by default

MODEL_NAMES = {
    "en_indic": f"ai4bharat/indictrans2-en-indic-{MODEL_SIZE}",
    "indic_en": f"ai4bharat/indictrans2-indic-en-{MODEL_SIZE}",
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
            logger.info(f"Using device: {self.device}")

            # Processor handles normalization, tagging, and detokenization
            self.ip = IndicProcessor(inference=True)

            # Placeholders for lazy model load
            self.tokenizer_en_indic = None
            self.model_en_indic = None
            self.tokenizer_indic_en = None
            self.model_indic_en = None
            
            # Track loading status
            self.loading_en_indic = False
            self.loading_indic_en = False

    def _get_cache_dir(self):
        """Resolve Hugging Face cache dir inside container."""
        cache_dir = Path(
            os.environ.get(
                "TRANSFORMERS_CACHE",
                Path.home() / ".cache" / "huggingface" / "transformers",
            )
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _load_en_indic(self):
        """Load en→indic model lazily."""
        if self.model_en_indic is None and not self.loading_en_indic:
            try:
                self.loading_en_indic = True
                logger.info(f"🚀 Loading {MODEL_NAMES['en_indic']} ...")
                cache_dir = self._get_cache_dir()
                
                # Check if model files exist in cache
                model_path = cache_dir / "models--ai4bharat--indictrans2-en-indic-dist-200M"
                if model_path.exists():
                    logger.info("📦 Loading model from local cache...")
                else:
                    logger.info("🌐 Downloading model from Hugging Face Hub...")
                
                self.tokenizer_en_indic = AutoTokenizer.from_pretrained(
                    MODEL_NAMES["en_indic"], 
                    trust_remote_code=True, 
                    cache_dir=cache_dir
                )
                
                self.model_en_indic = AutoModelForSeq2SeqLM.from_pretrained(
                    MODEL_NAMES["en_indic"], 
                    trust_remote_code=True, 
                    cache_dir=cache_dir,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32  # Use half precision for GPU
                ).to(self.device).eval()
                
                logger.info("✅ EN→Indic model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load EN→Indic model: {str(e)}")
                raise
            finally:
                self.loading_en_indic = False

    def _load_indic_en(self):
        """Load indic→en model lazily."""
        if self.model_indic_en is None and not self.loading_indic_en:
            try:
                self.loading_indic_en = True
                logger.info(f"🚀 Loading {MODEL_NAMES['indic_en']} ...")
                cache_dir = self._get_cache_dir()
                
                self.tokenizer_indic_en = AutoTokenizer.from_pretrained(
                    MODEL_NAMES["indic_en"], 
                    trust_remote_code=True, 
                    cache_dir=cache_dir
                )
                
                self.model_indic_en = AutoModelForSeq2SeqLM.from_pretrained(
                    MODEL_NAMES["indic_en"], 
                    trust_remote_code=True, 
                    cache_dir=cache_dir,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32  # Use half precision for GPU
                ).to(self.device).eval()
                
                logger.info("✅ Indic→EN model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load Indic→EN model: {str(e)}")
                raise
            finally:
                self.loading_indic_en = False

    def _translate(self, text: str, source: str, target: str) -> str:
        """Translate text using IndicTrans2 official pipeline."""
        try:
            if not text.strip():
                return text

            logger.info(f"Translating: '{text[:50]}...' from {source} to {target}")

            # Map ISO short codes to IndicTrans tags
            def _to_indictrans_tag(code: str) -> str:
                mapping = {
                    "en": "eng_Latn",
                    "hi": "hin_Deva",
                    "kn": "kan_Knda",
                }
                if code not in mapping:
                    raise ValueError(f"Unsupported language code: {code}")
                return mapping[code]

            src_tag = _to_indictrans_tag(source)
            tgt_tag = _to_indictrans_tag(target)

            # Pick correct direction and ensure model is ready
            def _ensure_ready_en_indic(timeout_sec: float = 600.0):
                start = time.monotonic()
                # Trigger load if not started
                self._load_en_indic()
                while self.model_en_indic is None or self.tokenizer_en_indic is None:
                    if time.monotonic() - start > timeout_sec:
                        raise TimeoutError("Timeout waiting for EN→Indic model to load")
                    # If not actively loading, try loading again
                    if not self.loading_en_indic:
                        self._load_en_indic()
                    time.sleep(0.5)

            def _ensure_ready_indic_en(timeout_sec: float = 600.0):
                start = time.monotonic()
                # Trigger load if not started
                self._load_indic_en()
                while self.model_indic_en is None or self.tokenizer_indic_en is None:
                    if time.monotonic() - start > timeout_sec:
                        raise TimeoutError("Timeout waiting for Indic→EN model to load")
                    # If not actively loading, try loading again
                    if not self.loading_indic_en:
                        self._load_indic_en()
                    time.sleep(0.5)

            if source == "en":
                logger.info("Ensuring EN->Indic model is ready")
                _ensure_ready_en_indic()
                tokenizer, model = self.tokenizer_en_indic, self.model_en_indic
            else:
                logger.info("Ensuring Indic->EN model is ready")
                _ensure_ready_indic_en()
                tokenizer, model = self.tokenizer_indic_en, self.model_indic_en

            logger.info("Models loaded, starting preprocessing...")

            # Preprocess (adds tags + normalization)
            try:
                batch = self.ip.preprocess_batch([text], src_lang=src_tag, tgt_lang=tgt_tag)
                logger.info(f"Preprocessed batch: {batch}")
            except Exception as e:
                logger.error(f"Preprocessing failed: {str(e)}")
                raise

            logger.info("Starting tokenization...")
            try:
                with torch.no_grad():
                    inputs = tokenizer(
                        batch,
                        truncation=True,
                        padding="longest",
                        return_tensors="pt",
                        return_attention_mask=True,
                    ).to(self.device)
                    
                    logger.info(f"Input shape: {inputs['input_ids'].shape}")
                    
                    logger.info("Starting model generation...")
                    outputs = model.generate(
                        **inputs,
                        use_cache=True,
                        min_length=0,
                        max_length=256,
                        num_beams=5,
                        num_return_sequences=1,
                    )
                    
                    logger.info(f"Generation completed, output shape: {outputs.shape}")

            except Exception as e:
                logger.error(f"Generation failed: {str(e)}")
                raise

            logger.info("Starting decoding...")
            try:
                # Decode with proper target tokenizer context
                with tokenizer.as_target_tokenizer():
                    decoded = tokenizer.batch_decode(
                        outputs.detach().cpu().tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                
                logger.info(f"Decoded output: {decoded}")

                # Postprocess (detokenization/entity replacement)
                translations = self.ip.postprocess_batch(decoded, lang=tgt_tag)
                result = translations[0] if translations else text
                
                logger.info(f"Final translation result: '{result}'")
                return result

            except Exception as e:
                logger.error(f"Decoding/postprocessing failed: {str(e)}")
                raise

        except Exception as e:
            import traceback
            logger.error(f"Full traceback:")
            traceback.print_exc()
            logger.error(f"⚠️ Translation error ({source}→{target}): {str(e)}")
            # Don't return fallback - let the error bubble up so we can see what's happening
            raise

    async def translate_async(self, text: str, source: str, target: str) -> str:
        """Async wrapper so routes can await translation."""
        try:
            # Use a larger thread pool for CPU-intensive tasks
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(executor, self._translate, text, source, target)
                return result
        except Exception as e:
            logger.error(f"Error in translate_async: {str(e)}")
            raise

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
            logger.error(f"⚠️ Error in translate_to_all: {str(e)}")
            return {}

    @property
    def is_models_loaded(self) -> bool:
        """Check if any models are loaded"""
        return (self.model_en_indic is not None) or (self.model_indic_en is not None)
    
    def get_model_info(self) -> dict:
        return {
            "device": str(self.device),
            "model_size": MODEL_SIZE,
            "en_indic_model": MODEL_NAMES["en_indic"] if self.model_en_indic else None,
            "indic_en_model": MODEL_NAMES["indic_en"] if self.model_indic_en else None,
            "en_indic_loaded": self.model_en_indic is not None,
            "indic_en_loaded": self.model_indic_en is not None,
            "any_model_loaded": self.is_models_loaded,
        }

    def warmup(self) -> None:
        """Preload tokenizers/models and prime preprocessing resources to avoid first-call latency."""
        try:
            logger.info("🔥 Warmup: loading models and priming preprocess...")
            # Load both directions
            self._load_en_indic()
            self._load_indic_en()

            # Prime IndicProcessor resources and tokenizers with a tiny sample
            try:
                _ = self.ip.preprocess_batch(["warmup"], src_lang="eng_Latn", tgt_lang="hin_Deva")
            except Exception as e:
                logger.warning(f"Warmup preprocess failed (continuing): {e}")

            # Run a very small forward pass to initialize model graph/caches
            try:
                if self.tokenizer_en_indic and self.model_en_indic:
                    with torch.no_grad():
                        inputs = self.tokenizer_en_indic(["warmup"], return_tensors="pt").to(self.device)
                        _ = self.model_en_indic.generate(**inputs, max_length=8)
            except Exception as e:
                logger.warning(f"Warmup generation failed (continuing): {e}")

            logger.info("✅ Warmup completed")
        except Exception as e:
            logger.error(f"Warmup error: {e}")


# ✅ Export singleton instance
translation_service = TranslationService()