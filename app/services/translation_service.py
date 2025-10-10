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
                
                # Force initialization with dummy forward pass
                with torch.no_grad():
                    dummy_input = self.tokenizer_en_indic("test", return_tensors="pt", padding=True, truncation=True).to(self.device)
                    _ = self.model_en_indic(**dummy_input)
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
                
                # Force initialization with dummy forward pass
                with torch.no_grad():
                    dummy_input = self.tokenizer_indic_en("test", return_tensors="pt", padding=True, truncation=True).to(self.device)
                    _ = self.model_indic_en(**dummy_input)

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
                    self.loading_en_indic = False
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
                
                # Force model initialization by moving to device and loading weights
                self.model_en_indic = self.model_en_indic.to(self.device)
                self.model_en_indic.eval()
                
                # Force data loading with a dummy forward pass to ensure weights are loaded
                with torch.no_grad():
                    dummy_input = self.tokenizer_en_indic("test", return_tensors="pt", padding=True, truncation=True).to(self.device)
                    try:
                        _ = self.model_en_indic(**dummy_input)
                    except Exception as e:
                        logger.error(f"Dummy forward pass failed: {e}")
                        raise RuntimeError(f"Model initialization failed: {e}")
                
                # Verify model is properly on device
                if hasattr(self.model_en_indic, 'device') and str(self.model_en_indic.device) != str(self.device):
                    logger.error(f"Model not on expected device: {self.model_en_indic.device} vs {self.device}")
                    raise RuntimeError(f"Model device mismatch after loading")

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
                    self.loading_indic_en = False
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
                
                # Force model initialization by moving to device and loading weights
                self.model_indic_en = self.model_indic_en.to(self.device)
                self.model_indic_en.eval()
                
                # Force data loading with a dummy forward pass to ensure weights are loaded
                with torch.no_grad():
                    dummy_input = self.tokenizer_indic_en("test", return_tensors="pt", padding=True, truncation=True).to(self.device)
                    try:
                        _ = self.model_indic_en(**dummy_input)
                    except Exception as e:
                        logger.error(f"Dummy forward pass failed: {e}")
                        raise RuntimeError(f"Model initialization failed: {e}")
                
                # Verify model is properly on device
                if hasattr(self.model_indic_en, 'device') and str(self.model_indic_en.device) != str(self.device):
                    logger.error(f"Model not on expected device: {self.model_indic_en.device} vs {self.device}")
                    raise RuntimeError(f"Model device mismatch after loading")

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
        elif source_lang in ["hi", "kn"] and target_lang in ["hi", "kn"] and source_lang != target_lang:
            # Pivot through English for HI↔KN
            return self._translate_via_en(text, source_lang, target_lang)
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
                raise RuntimeError(f"{source}→{target} model not loaded properly")
            
            # Validate device consistency
            if hasattr(model, 'device') and str(model.device) != str(self.device):
                logger.error(f"Model device mismatch: model on {model.device}, expected {self.device}")
                raise RuntimeError(f"Model device mismatch: model on {model.device}, expected {self.device}")

            # Preprocess
            batch = self.ip.preprocess_batch([text], src_lang=src_tag, tgt_lang=tgt_tag)
            if isinstance(batch, tuple):
                batch = batch[0]
            if not isinstance(batch, list):
                raise ValueError(f"Unexpected batch type from preprocess: {type(batch)}")
            
            # Validate batch is not empty
            if not batch or len(batch) == 0:
                logger.error(f"Preprocessing returned empty batch for {source}→{target}")
                raise RuntimeError(f"Preprocessing returned empty batch for {source}→{target}")

            # Tokenize
            try:
                inputs = tokenizer(
                    batch,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                    max_length=256,
                )
                
                # Validate tokenizer output
                if inputs is None:
                    logger.error(f"Tokenizer returned None for {source}→{target}")
                    raise RuntimeError(f"Tokenizer returned None for {source}→{target}")
                
                # Validate all tensor values are not None before moving to device
                for k, v in inputs.items():
                    if v is None:
                        logger.error(f"Tokenizer returned None for key '{k}' in {source}→{target}")
                        raise RuntimeError(f"Tokenizer returned None for key '{k}'")
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            except Exception as e:
                logger.error(f"Tokenization error for {source}→{target}: {e}")
                raise RuntimeError(f"Tokenization failed for {source}→{target}: {e}")

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=128,
                    num_beams=1,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Validate outputs
            if outputs is None:
                logger.error(f"Model generate returned None for {source}→{target}")
                raise RuntimeError(f"Model generate returned None for {source}→{target}")

            # Decode
            decoded = tokenizer.batch_decode(
                outputs.detach().cpu(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # Validate decoded output
            if not decoded or len(decoded) == 0:
                logger.error(f"Tokenizer decode returned empty for {source}→{target}")
                raise RuntimeError(f"Tokenizer decode returned empty for {source}→{target}")

            # Postprocess
            translations = self.ip.postprocess_batch(decoded, lang=tgt_tag)
            
            # Validate postprocessed output
            if not translations or len(translations) == 0:
                logger.error(f"Postprocess returned empty for {source}→{target}")
                raise RuntimeError(f"Postprocess returned empty for {source}→{target}")
                
            # Handle tuple return from postprocess_batch
            if isinstance(translations, tuple):
                translations = translations[0]
                
            if not translations:
                logger.error(f"Final translations is empty for {source}→{target}")
                raise RuntimeError(f"Final translations is empty for {source}→{target}")
                
            return translations[0]
            
        except Exception as e:
            logger.error(f"Translation error {source}→{target}: {e}")
            raise

    def _translate_via_en(self, text: str, source: str, target: str) -> str:
        """Translate between Indic languages by pivoting through English."""
        try:
            if not text.strip():
                return text
            
            logger.info(f"Pivoting {source}→{target} through English")
            
            # Step 1: source → en
            to_en = self._translate(text, source, "en")
            if not to_en or to_en == text:
                logger.error(f"Pivot step {source}→en failed")
                raise RuntimeError(f"Pivot step {source}→en failed")
            
            # Step 2: en → target
            to_target = self._translate(to_en, "en", target)
            if not to_target or to_target == to_en:
                logger.error(f"Pivot step en→{target} failed")
                raise RuntimeError(f"Pivot step en→{target} failed")
            
            return to_target
        except Exception as e:
            logger.error(f"Pivot translation error {source}→{target}: {e}")
            raise

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
        
        # Verify models loaded
        if self.model_en_indic is None and self.model_indic_en is None:
            logger.error("CRITICAL: No models loaded during warmup!")
            raise RuntimeError("Model loading failed - no models available")
        
        # Test both directions if models loaded successfully
        try:
            if self.model_en_indic is not None:
                test_en_hi = self._translate("Hello", "en", "hi")
                logger.info(f"Warmup test EN→HI: {test_en_hi}")
            else:
                logger.warning("EN→Indic model not loaded, skipping warmup test")
            
            if self.model_indic_en is not None:
                test_hi_en = self._translate("नमस्ते", "hi", "en")
                logger.info(f"Warmup test HI→EN: {test_hi_en}")
            else:
                logger.warning("Indic→EN model not loaded, skipping warmup test")
                
        except Exception as e:
            logger.error(f"Warmup test failed: {e}")
            raise RuntimeError(f"Warmup test failed: {e}")

    async def translate_to_all_async(self, title: str, description: str, source_lang: str) -> dict:
        """
        Translate title and description to all supported languages (Hindi, Kannada, English).
        Returns a dictionary with translations for each language.
        """
        logger.info(f"Starting translation to all languages from {source_lang}")
        
        # Define target languages based on source
        if source_lang == "en":
            target_langs = ["hi", "kn"]
        elif source_lang == "kn":
            target_langs = ["en", "hi"]
        elif source_lang == "hi":
            target_langs = ["en", "kn"]
        else:
            logger.error(f"Unsupported source language: {source_lang}")
            raise ValueError(f"Unsupported source language: {source_lang}")
        
        result = {}
        
        # Add source language as-is
        result[source_lang] = {"title": title, "description": description}
        
        # Translate to target languages sequentially to avoid CPU contention
        for target_lang in target_langs:
            try:
                logger.info(f"Translating to {target_lang}...")
                translated_title = await asyncio.to_thread(self.translate, title, source_lang, target_lang)
                translated_description = await asyncio.to_thread(self.translate, description, source_lang, target_lang)
                result[target_lang] = {
                    "title": translated_title,
                    "description": translated_description
                }
                logger.info(f"Successfully translated to {target_lang}")
            except Exception as e:
                logger.error(f"Failed to translate to {target_lang}: {e}")
                raise
        
        # # Ensure all languages are present
        # for lang in ["hi", "kn", "en"]:
        #     if lang not in result:
        #         result[lang] = {"title": title, "description": description}
        
        logger.info(f"Translation to all languages completed. Available languages: {list(result.keys())}")
        return result

translation_service = TranslationService()