import torch
import os
from pathlib import Path
import asyncio
import logging
import threading
import pickle
import hashlib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from IndicTransToolkit import IndicProcessor
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed model names for dist-200M
MODEL_NAMES = {
    "en_indic": "ai4bharat/indictrans2-en-indic-dist-200M",
    "indic_en": "ai4bharat/indictrans2-indic-en-dist-200M",
}

class TranslationService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TranslationService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "device"):
            # Force CPU-only for lightweight deployment
            self.device = torch.device("cpu")
            logger.info("[Translation] Using CPU for dist-200M model")

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

            # Cache dirs
            os.environ.setdefault("HF_HOME", "/app/.cache/huggingface")
            os.environ.setdefault("HF_HUB_CACHE", "/app/.cache/huggingface/hub")
            os.environ.setdefault("TRANSFORMERS_CACHE", "/app/.cache/huggingface/transformers")
            Path("/app/.cache/huggingface/hub").mkdir(parents=True, exist_ok=True)
            Path("/app/.cache/huggingface/transformers").mkdir(parents=True, exist_ok=True)
            
            # Model caching setup
            self.cache_dir = Path("/tmp/translation_cache")
            self.cache_dir.mkdir(exist_ok=True)
            self.en_indic_cache = self.cache_dir / "en_indic_model.pkl"
            self.indic_en_cache = self.cache_dir / "indic_en_model.pkl"

    def _get_cache_dir(self):
        """Resolve Hugging Face cache dir inside container"""
        cache_dir = Path(
            os.environ.get("HF_HUB_CACHE", "/app/.cache/huggingface/hub")
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _fix_tokenizer_specials(self, tokenizer):
        """Ensure pad/eos tokens are set (avoid NoneType issues)"""
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return tokenizer

    def _cache_model(self, model_type):
        """Cache the loaded model to disk"""
        try:
            if model_type == "en_indic" and self.model_en_indic is not None:
                cache_data = {
                    'model_state_dict': self.model_en_indic.state_dict(),
                    'tokenizer_config': self.tokenizer_en_indic.get_vocab(),
                    'model_config': self.model_en_indic.config.to_dict()
                }
                with open(self.en_indic_cache, 'wb') as f:
                    pickle.dump(cache_data, f)
                logger.info(f"Cached {model_type} model")
                
            elif model_type == "indic_en" and self.model_indic_en is not None:
                cache_data = {
                    'model_state_dict': self.model_indic_en.state_dict(),
                    'tokenizer_config': self.tokenizer_indic_en.get_vocab(),
                    'model_config': self.model_indic_en.config.to_dict()
                }
                with open(self.indic_en_cache, 'wb') as f:
                    pickle.dump(cache_data, f)
                logger.info(f"Cached {model_type} model")
                
        except Exception as e:
            logger.warning(f"Failed to cache {model_type} model: {e}")

    def _load_cached_model(self, model_type):
        """Load model from cache"""
        try:
            cache_file = self.en_indic_cache if model_type == "en_indic" else self.indic_en_cache
            
            if not cache_file.exists():
                return False
                
            logger.info(f"Loading {model_type} model from cache...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Recreate model from cache
            if model_type == "en_indic":
                self.model_en_indic = AutoModelForSeq2SeqLM.from_config(
                    AutoConfig.from_dict(cache_data['model_config'])
                )
                self.model_en_indic.load_state_dict(cache_data['model_state_dict'])
                self.model_en_indic.to(self.device).eval()
                
                # Recreate tokenizer
                self.tokenizer_en_indic = AutoTokenizer.from_pretrained(
                    MODEL_NAMES["en_indic"],
                    trust_remote_code=True
                )
                self.tokenizer_en_indic = self._fix_tokenizer_specials(self.tokenizer_en_indic)
                
            else:  # indic_en
                self.model_indic_en = AutoModelForSeq2SeqLM.from_config(
                    AutoConfig.from_dict(cache_data['model_config'])
                )
                self.model_indic_en.load_state_dict(cache_data['model_state_dict'])
                self.model_indic_en.to(self.device).eval()
                
                self.tokenizer_indic_en = AutoTokenizer.from_pretrained(
                    MODEL_NAMES["indic_en"],
                    trust_remote_code=True
                )
                self.tokenizer_indic_en = self._fix_tokenizer_specials(self.tokenizer_indic_en)
            
            logger.info(f"Successfully loaded {model_type} model from cache")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load {model_type} model from cache: {e}")
            return False

    def _ensure_en_indic_model(self):
        if self.model_en_indic is None and not self.loading_en_indic:
            try:
                self.loading_en_indic = True
                
                # Check for cached model first
                if self._load_cached_model("en_indic"):
                    logger.info("Loaded EN→Indic model from cache")
                    return
                
                # Load from HuggingFace if no cache
                logger.info("Loading EN→Indic model from HuggingFace...")
                cache_dir = self._get_cache_dir()
                local_path = cache_dir / "models--ai4bharat--indictrans2-en-indic-dist-200M"

                if local_path.exists():
                    logger.info(f"Loading EN→Indic from local cache: {local_path}")
                    snapshots = list((local_path / "snapshots").glob("*"))
                    model_path = str(snapshots[0]) if snapshots else MODEL_NAMES["en_indic"]
                else:
                    logger.info("Loading EN→Indic from HuggingFace...")
                    model_path = MODEL_NAMES["en_indic"]

                self.tokenizer_en_indic = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    local_files_only=local_path.exists(),
                )
                self.tokenizer_en_indic = self._fix_tokenizer_specials(self.tokenizer_en_indic)

                self.model_en_indic = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    local_files_only=local_path.exists(),
                    torch_dtype=torch.float32
                ).to(self.device).eval()

                # Cache the loaded model
                self._cache_model("en_indic")
                logger.info("EN→Indic dist-200M model loaded and cached successfully")
            finally:
                self.loading_en_indic = False

    def _ensure_indic_en_model(self):
        if self.model_indic_en is None and not self.loading_indic_en:
            try:
                self.loading_indic_en = True
                
                # Check for cached model first
                if self._load_cached_model("indic_en"):
                    logger.info("Loaded Indic→EN model from cache")
                    return
                
                # Load from HuggingFace if no cache
                logger.info("Loading Indic→EN model from HuggingFace...")
                cache_dir = self._get_cache_dir()
                local_path = cache_dir / "models--ai4bharat--indictrans2-indic-en-dist-200M"

                if local_path.exists():
                    logger.info(f"Loading Indic→EN from local cache: {local_path}")
                    snapshots = list((local_path / "snapshots").glob("*"))
                    model_path = str(snapshots[0]) if snapshots else MODEL_NAMES["indic_en"]
                else:
                    logger.info("Loading Indic→EN from HuggingFace...")
                    model_path = MODEL_NAMES["indic_en"]

                self.tokenizer_indic_en = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    local_files_only=local_path.exists(),
                )
                self.tokenizer_indic_en = self._fix_tokenizer_specials(self.tokenizer_indic_en)

                self.model_indic_en = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    local_files_only=local_path.exists(),
                    torch_dtype=torch.float32
                ).to(self.device).eval()

                # Cache the loaded model
                self._cache_model("indic_en")
                logger.info("Indic→EN dist-200M model loaded and cached successfully")
            finally:
                self.loading_indic_en = False

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        logger.info(f"Translating: '{text[:50]}...' from {source_lang} to {target_lang}")
        if source_lang == "en" and target_lang in ["hi", "kn"]:
            return self._translate(text, source_lang, target_lang)
        elif source_lang in ["hi", "kn"] and target_lang == "en":
            return self._translate(text, source_lang, target_lang)
        else:
            raise ValueError(f"Unsupported translation: {source_lang} → {target_lang}")

    def _translate(self, text: str, source: str, target: str) -> str:
        if not text.strip():
            return text

        mapping = {"en": "eng_Latn", "hi": "hin_Deva", "kn": "kan_Knda"}
        src_tag = mapping[source]
        tgt_tag = mapping[target]

        if source == "en":
            self._ensure_en_indic_model()
            tokenizer, model = self.tokenizer_en_indic, self.model_en_indic
        else:
            self._ensure_indic_en_model()
            tokenizer, model = self.tokenizer_indic_en, self.model_indic_en

        # Preprocess
        batch = self.ip.preprocess_batch([text], src_lang=src_tag, tgt_lang=tgt_tag)
        if isinstance(batch, tuple):
            batch = batch[0]
        if not isinstance(batch, list):
            raise ValueError(f"Unexpected batch type from preprocess: {type(batch)}")

        # Tokenize
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                use_cache=False,
                min_length=0,
                max_length=256,
                num_beams=3,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        decoded = tokenizer.batch_decode(
            outputs.detach().cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Postprocess
        translations = self.ip.postprocess_batch(decoded, lang=tgt_tag)
        return translations[0] if translations else text

    @property
    def is_models_loaded(self) -> bool:
        return (self.model_en_indic is not None) or (self.model_indic_en is not None)

    def get_model_info(self) -> dict:
        return {
            "device": str(self.device),
            "model_size": "dist-200M",
            "en_indic_model": MODEL_NAMES["en_indic"] if self.model_en_indic else None,
            "indic_en_model": MODEL_NAMES["indic_en"] if self.model_indic_en else None,
            "en_indic_loaded": self.model_en_indic is not None,
            "indic_en_loaded": self.model_indic_en is not None,
            "any_model_loaded": self.is_models_loaded,
        }

    def warmup(self) -> None:
        logger.info("Warmup: loading models and priming...")
        
        start_time = time.time()
        
        def load_en_indic():
            self._ensure_en_indic_model()
            logger.info("EN→Indic model ready")
        
        def load_indic_en():
            self._ensure_indic_en_model()
            logger.info("Indic→EN model ready")
        
        # Start both model loading in parallel
        thread1 = threading.Thread(target=load_en_indic)
        thread2 = threading.Thread(target=load_indic_en)
        
        thread1.start()
        thread2.start()
        
        # Wait for both to complete
        thread1.join()
        thread2.join()
        
        load_time = time.time() - start_time
        logger.info(f"Both models loaded in {load_time:.2f} seconds")
        
        # Test both directions
        try:
            test_en_hi = self.translate("Hello", "en", "hi")
            test_hi_en = self.translate("नमस्ते", "hi", "en")
            logger.info(f"Warmup tests: EN→HI: {test_en_hi}, HI→EN: {test_hi_en}")
        except Exception as e:
            logger.warning(f"Warmup test failed: {e}")

    async def translate_to_all_async(self, title: str, description: str, source_lang: str) -> dict:
        """
        Translate title and description to all supported languages (Hindi, Kannada, English).
        Returns a dictionary with translations for each language.
        """
        logger.info(f"Starting translation to all languages from {source_lang}")
        
        # Define target languages based on source
        if source_lang == "en":
            target_langs = ["hi", "kn"]
        elif source_lang in ["hi", "kn"]:
            target_langs = ["en"]
        else:
            logger.warning(f"Unsupported source language: {source_lang}, using original text")
            return {
                "hi": {"title": title, "description": description},
                "kn": {"title": title, "description": description},
                "en": {"title": title, "description": description}
            }
        
        result = {}
        
        # Add source language as-is
        result[source_lang] = {"title": title, "description": description}
        
        # Translate to target languages
        for target_lang in target_langs:
            try:
                logger.info(f"Translating to {target_lang}...")
                translated_title = self.translate(title, source_lang, target_lang)
                translated_description = self.translate(description, source_lang, target_lang)
                result[target_lang] = {
                    "title": translated_title,
                    "description": translated_description
                }
                logger.info(f"Successfully translated to {target_lang}")
            except Exception as e:
                logger.error(f"Failed to translate to {target_lang}: {e}")
                # Use original text as fallback
                result[target_lang] = {"title": title, "description": description}
        
        # Ensure all languages are present
        for lang in ["hi", "kn", "en"]:
            if lang not in result:
                result[lang] = {"title": title, "description": description}
        
        logger.info(f"Translation to all languages completed. Available languages: {list(result.keys())}")
        return result

translation_service = TranslationService()
