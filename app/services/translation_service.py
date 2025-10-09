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

    def _get_cache_dir(self):
        """Resolve HF hub cache directory inside container (matches build preload)."""
        cache_dir = Path(
            os.environ.get("HF_HUB_CACHE", "/app/.cache/huggingface/hub")
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _resolve_local_snapshot_dir(self, model_name: str) -> Path | None:
        """Return latest local snapshot path for a given repo inside HF hub cache.
        Example: /app/.cache/huggingface/hub/models--{org}--{repo}/snapshots/{rev}
        """
        try:
            hub_cache = Path(os.environ.get("HF_HUB_CACHE", "/app/.cache/huggingface/hub"))
            repo_dir = hub_cache / f"models--{model_name.replace('/', '--')}" / "snapshots"
            if not repo_dir.exists():
                return None
            # Choose most recently modified snapshot
            candidates = [p for p in repo_dir.iterdir() if p.is_dir()]
            if not candidates:
                return None
            latest = max(candidates, key=lambda p: p.stat().st_mtime)
            return latest
        except Exception:
            return None

    def _fix_tokenizer_specials(self, tokenizer):
        """Ensure pad/eos tokens are set (avoid NoneType issues)"""
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return tokenizer

    def _load_cached_model(self, model_type):
        """Load model strictly from local Transformers cache (offline)."""
        try:
            cache_dir = self._get_cache_dir()
            model_name = MODEL_NAMES["en_indic"] if model_type == "en_indic" else MODEL_NAMES["indic_en"]

            logger.info(f"Loading {model_type} model from local cache: {cache_dir}")

            model_name_full = MODEL_NAMES["en_indic"] if model_type == "en_indic" else MODEL_NAMES["indic_en"]
            local_snapshot = self._resolve_local_snapshot_dir(model_name_full)
            load_path = str(local_snapshot) if local_snapshot else model_name_full
            if local_snapshot is None:
                logger.warning(f"No local snapshot found for {model_name_full}; attempting ID-based local load")

            if model_type == "en_indic":
                self.tokenizer_en_indic = AutoTokenizer.from_pretrained(
                    load_path,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
                self.tokenizer_en_indic = self._fix_tokenizer_specials(self.tokenizer_en_indic)

                self.model_en_indic = AutoModelForSeq2SeqLM.from_pretrained(
                    load_path,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    local_files_only=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                )
                self.model_en_indic = self.model_en_indic.to(self.device).eval()
            else:
                self.tokenizer_indic_en = AutoTokenizer.from_pretrained(
                    load_path,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
                self.tokenizer_indic_en = self._fix_tokenizer_specials(self.tokenizer_indic_en)

                self.model_indic_en = AutoModelForSeq2SeqLM.from_pretrained(
                    load_path,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    local_files_only=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                )
                self.model_indic_en = self.model_indic_en.to(self.device).eval()

            logger.info(f"Successfully loaded {model_type} model from local cache")
            return True
        except Exception as e:
            logger.warning(f"Failed to load {model_type} model from local cache: {e}")
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
                logger.info("Loading EN→Indic model from local cache (strict)...")
                cache_dir = self._get_cache_dir()
                model_path = MODEL_NAMES["en_indic"]

                self.tokenizer_en_indic = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
                self.tokenizer_en_indic = self._fix_tokenizer_specials(self.tokenizer_en_indic)

                # Load model with proper initialization
                self.model_en_indic = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    local_files_only=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                )
                
                # Force model initialization by moving to device
                self.model_en_indic = self.model_en_indic.to(self.device)
                self.model_en_indic.eval()
                
                # Force data loading with a dummy forward pass
                with torch.no_grad():
                    dummy_input = self.tokenizer_en_indic("test", return_tensors="pt", padding=True, truncation=True).to(self.device)
                    _ = self.model_en_indic(**dummy_input)

                logger.info("EN→Indic dist-200M model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load EN→Indic model: {e}")
                raise
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
                logger.info("Loading Indic→EN model from local cache (strict)...")
                cache_dir = self._get_cache_dir()
                model_path = MODEL_NAMES["indic_en"]

                self.tokenizer_indic_en = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
                self.tokenizer_indic_en = self._fix_tokenizer_specials(self.tokenizer_indic_en)

                # Load model with proper initialization
                self.model_indic_en = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    local_files_only=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                )
                
                # Force model initialization by moving to device
                self.model_indic_en = self.model_indic_en.to(self.device)
                self.model_indic_en.eval()
                
                # Force data loading with a dummy forward pass
                with torch.no_grad():
                    dummy_input = self.tokenizer_indic_en("test", return_tensors="pt", padding=True, truncation=True).to(self.device)
                    _ = self.model_indic_en(**dummy_input)

                logger.info("Indic→EN dist-200M model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Indic→EN model: {e}")
                raise
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
        try:
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

            # Add validation that models are properly loaded
            if model is None or tokenizer is None:
                logger.error(f"{source}→{target} model not loaded properly")
                return text  # Return original text as fallback

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
            if isinstance(translations, tuple):
                translations = translations[0]
            return translations[0] if translations else text
            
        except Exception as e:
            logger.error(f"Translation error {source}→{target}: {e}")
            return text  # Return original text on error

    def _translate_batch(self, texts: list[str], source: str, target: str) -> list[str]:
        """Translate a list of texts in a single model call for efficiency.
        Preserves ordering; on error returns original texts.
        """
        try:
            # Short-circuit empty batch
            if not texts:
                return []

            # If all strings are empty/whitespace, return as-is
            if all((t is None) or (isinstance(t, str) and not t.strip()) for t in texts):
                return texts

            # Normalize inputs to strings
            safe_texts = [(t if isinstance(t, str) else "") for t in texts]

            mapping = {"en": "eng_Latn", "hi": "hin_Deva", "kn": "kan_Knda"}
            src_tag = mapping[source]
            tgt_tag = mapping[target]

            if source == "en":
                self._ensure_en_indic_model()
                tokenizer, model = self.tokenizer_en_indic, self.model_en_indic
            else:
                self._ensure_indic_en_model()
                tokenizer, model = self.tokenizer_indic_en, self.model_indic_en

            if model is None or tokenizer is None:
                logger.error(f"{source}→{target} model not loaded properly (batch)")
                return texts

            # Preprocess batch
            batch = self.ip.preprocess_batch(safe_texts, src_lang=src_tag, tgt_lang=tgt_tag)
            if isinstance(batch, tuple):
                batch = batch[0]
            if not isinstance(batch, list):
                raise ValueError(f"Unexpected batch type from preprocess: {type(batch)}")

            # Tokenize once
            inputs = tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate once
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

            # Decode and postprocess
            decoded = tokenizer.batch_decode(
                outputs.detach().cpu(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            translations = self.ip.postprocess_batch(decoded, lang=tgt_tag)
            if isinstance(translations, tuple):
                translations = translations[0]

            # Safety: ensure same length
            if not isinstance(translations, list) or len(translations) != len(safe_texts):
                logger.warning("Batch translation length mismatch; falling back to originals")
                return texts

            return translations
        except Exception as e:
            logger.error(f"Batch translation error {source}→{target}: {e}")
            return texts

    def _translate_batch_via_en(self, texts: list[str], source: str, target: str) -> list[str]:
        """Translate between Indic languages by pivoting through English (source→en→target)."""
        try:
            if source == "en" or target == "en":
                return self._translate_batch(texts, source, target)

            # Step 1: source → en
            to_en = self._translate_batch(texts, source, "en")
            if not isinstance(to_en, list) or len(to_en) != len(texts):
                logger.warning("Pivot step source→en failed; returning originals")
                return texts

            # Step 2: en → target
            to_target = self._translate_batch(to_en, "en", target)
            if not isinstance(to_target, list) or len(to_target) != len(texts):
                logger.warning("Pivot step en→target failed; returning originals")
                return texts

            return to_target
        except Exception as e:
            logger.error(f"Pivot batch translation error {source}→{target}: {e}")
            return texts

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
            try:
                self._ensure_en_indic_model()
                logger.info("EN→Indic model ready")
            except Exception as e:
                logger.error(f"Failed to load EN→Indic model: {e}")
        
        def load_indic_en():
            try:
                self._ensure_indic_en_model()
                logger.info("Indic→EN model ready")
            except Exception as e:
                logger.error(f"Failed to load Indic→EN model: {e}")
        
        # Start both model loading in parallel
        thread1 = threading.Thread(target=load_en_indic)
        thread2 = threading.Thread(target=load_indic_en)
        
        thread1.start()
        thread2.start()
        
        # Wait for both to complete
        thread1.join()
        thread2.join()
        
        load_time = time.time() - start_time
        logger.info(f"Model loading completed in {load_time:.2f} seconds")
        
        # Test both directions if models loaded successfully
        try:
            if self.model_en_indic is not None:
                test_en_hi = self.translate("Hello", "en", "hi")
                logger.info(f"Warmup test EN→HI: {test_en_hi}")
            
            if self.model_indic_en is not None:
                test_hi_en = self.translate("नमस्ते", "hi", "en")
                logger.info(f"Warmup test HI→EN: {test_hi_en}")
                
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
        elif source_lang == "hi":
            target_langs = ["en", "kn"]  
        elif source_lang == "kn":
            target_langs = ["en", "hi"]  
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
        
        # Translate to target languages in parallel, batching title+description per language
        async def translate_one_language(tgt: str):
            try:
                logger.info(f"Translating to {tgt} (batched)...")
                # If both source and target are Indic (non-EN), pivot via EN
                if source_lang in ["hi", "kn"] and tgt in ["hi", "kn"]:
                    batch = await asyncio.to_thread(self._translate_batch_via_en, [title, description], source_lang, tgt)
                else:
                    batch = await asyncio.to_thread(self._translate_batch, [title, description], source_lang, tgt)
                if not isinstance(batch, list) or len(batch) != 2:
                    raise ValueError("Unexpected batch output shape")
                return tgt, {"title": batch[0], "description": batch[1]}
            except Exception as e:
                logger.error(f"Failed to translate to {tgt}: {e}")
                return tgt, {"title": title, "description": description}

        tasks = [translate_one_language(t) for t in target_langs]
        if tasks:
            results = await asyncio.gather(*tasks)
            for lang_key, payload in results:
                result[lang_key] = payload
        
        # Ensure all languages are present
        for lang in ["hi", "kn", "en"]:
            if lang not in result:
                result[lang] = {"title": title, "description": description}
        
        logger.info(f"Translation to all languages completed. Available languages: {list(result.keys())}")
        return result

translation_service = TranslationService()