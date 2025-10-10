import torch
import os
from pathlib import Path
import asyncio
import logging
import threading
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys

# Add IndicTrans2 to path
sys.path.append('/app/IndicTrans2')

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

            # Initialize IndicTrans2 components
            try:
                logger.info("Initializing IndicTrans2 components...")
                # Import IndicTrans2 components
                import sys
                sys.path.append('/app/IndicTrans2')
                from IndicTrans2.inference.engine import Model
                from IndicTrans2.inference.engine import InferenceEngine
                self.Model = Model
                self.InferenceEngine = InferenceEngine
                logger.info("IndicTrans2 components initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize IndicTrans2 components: {e}")
                raise RuntimeError(f"IndicTrans2 initialization failed: {e}")

            # Placeholders for lazy model load
            self.en_indic_model = None
            self.indic_en_model = None
            self.en_indic_engine = None
            self.indic_en_engine = None

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
        """Resolve HF hub cache directory inside container."""
        cache_dir = Path(
            os.environ.get("HF_HUB_CACHE", "/app/.cache/huggingface/hub")
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _ensure_en_indic_model(self):
        """Load EN->Indic model using IndicTrans2 native approach."""
        if self.en_indic_model is None and not self.loading_en_indic:
            try:
                self.loading_en_indic = True
                logger.info("Starting EN->Indic model loading with IndicTrans2...")
                
                cache_dir = self._get_cache_dir()
                model_path = MODEL_NAMES["en_indic"]

                # Load model using IndicTrans2 native approach
                logger.info(f"Loading EN->Indic model from {model_path}")
                self.en_indic_model = self.Model(
                    model_path,
                    device=self.device
                )
                
                # Initialize inference engine
                logger.info("Initializing EN->Indic inference engine...")
                self.en_indic_engine = self.InferenceEngine(
                    self.en_indic_model
                )
                
                logger.info("EN->Indic model loaded successfully with IndicTrans2")
                
            except Exception as e:
                logger.error(f"Failed to load EN->Indic model: {e}")
                self.en_indic_model = None
                self.en_indic_engine = None
                raise
            finally:
                self.loading_en_indic = False

    def _ensure_indic_en_model(self):
        """Load Indic->EN model using IndicTrans2 native approach."""
        if self.indic_en_model is None and not self.loading_indic_en:
            try:
                self.loading_indic_en = True
                logger.info("Starting Indic->EN model loading with IndicTrans2...")
                
                cache_dir = self._get_cache_dir()
                model_path = MODEL_NAMES["indic_en"]

                # Load model using IndicTrans2 native approach
                logger.info(f"Loading Indic->EN model from {model_path}")
                self.indic_en_model = self.Model(
                    model_path,
                    device=self.device
                )
                
                # Initialize inference engine
                logger.info("Initializing Indic->EN inference engine...")
                self.indic_en_engine = self.InferenceEngine(
                    self.indic_en_model
                )
                
                logger.info("Indic->EN model loaded successfully with IndicTrans2")
                
            except Exception as e:
                logger.error(f"Failed to load Indic->EN model: {e}")
                self.indic_en_model = None
                self.indic_en_engine = None
                raise
            finally:
                self.loading_indic_en = False

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using IndicTrans2 native API."""
        logger.info(f"Translating: '{text[:50]}...' from {source_lang} to {target_lang}")
        
        if source_lang == "en" and target_lang in ["hi", "kn"]:
            return self._translate_en_to_indic(text, target_lang)
        elif source_lang in ["hi", "kn"] and target_lang == "en":
            return self._translate_indic_to_en(text, source_lang)
        elif source_lang in ["hi", "kn"] and target_lang in ["hi", "kn"] and source_lang != target_lang:
            # Pivot through English for HI↔KN
            return self._translate_via_en(text, source_lang, target_lang)
        else:
            raise ValueError(f"Unsupported translation: {source_lang} -> {target_lang}")

    def _translate_en_to_indic(self, text: str, target_lang: str) -> str:
        """Translate from English to Indic language using IndicTrans2."""
        try:
            if not text.strip():
                return text

            logger.info(f"Translating EN->{target_lang} with IndicTrans2")

            # Load model if not already loaded
            self._ensure_en_indic_model()
            
            if self.en_indic_engine is None:
                raise RuntimeError("EN->Indic engine not loaded")
            
            # Use IndicTrans2 native translation
            result = self.en_indic_engine.translate(
                text,
                src_lang="eng_Latn",
                tgt_lang=self._get_lang_code(target_lang)
            )
            
            logger.info(f"Translation successful: '{result[:50]}...'")
            return result
            
        except Exception as e:
            logger.error(f"Translation error EN->{target_lang}: {e}")
            raise

    def _translate_indic_to_en(self, text: str, source_lang: str) -> str:
        """Translate from Indic language to English using IndicTrans2."""
        try:
            if not text.strip():
                return text

            logger.info(f"Translating {source_lang}->EN with IndicTrans2")
            
            # Load model if not already loaded
            self._ensure_indic_en_model()
            
            if self.indic_en_engine is None:
                raise RuntimeError("Indic->EN engine not loaded")
            
            # Use IndicTrans2 native translation
            result = self.indic_en_engine.translate(
                text,
                src_lang=self._get_lang_code(source_lang),
                tgt_lang="eng_Latn"
            )
            
            logger.info(f"Translation successful: '{result[:50]}...'")
            return result
            
        except Exception as e:
            logger.error(f"Translation error {source_lang}->EN: {e}")
            raise

    def _translate_via_en(self, text: str, source: str, target: str) -> str:
        """Translate between Indic languages by pivoting through English."""
        try:
            if not text.strip():
                return text
            
            logger.info(f"Pivoting {source}->{target} through English")
            
            # Step 1: source -> en
            to_en = self._translate_indic_to_en(text, source)
            if not to_en or to_en == text:
                logger.error(f"Pivot step {source}->en failed")
                raise RuntimeError(f"Pivot step {source}->en failed")
            
            # Step 2: en -> target
            to_target = self._translate_en_to_indic(to_en, target)
            if not to_target or to_target == to_en:
                logger.error(f"Pivot step en->{target} failed")
                raise RuntimeError(f"Pivot step en->{target} failed")
            
            return to_target
        except Exception as e:
            logger.error(f"Pivot translation error {source}->{target}: {e}")
            raise

    def _get_lang_code(self, lang: str) -> str:
        """Convert language code to IndicTrans2 format."""
        lang_map = {
            "en": "eng_Latn",
            "hi": "hin_Deva", 
            "kn": "kan_Knda"
        }
        return lang_map.get(lang, lang)

    @property
    def is_models_loaded(self) -> bool:
        """Check if any models are loaded."""
        return (self.en_indic_model is not None) or (self.indic_en_model is not None)

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "device": str(self.device),
            "model_size": "dist-200M",
            "en_indic_model": MODEL_NAMES["en_indic"] if self.en_indic_model else None,
            "indic_en_model": MODEL_NAMES["indic_en"] if self.indic_en_model else None,
            "en_indic_loaded": self.en_indic_model is not None,
            "indic_en_loaded": self.indic_en_model is not None,
            "any_model_loaded": self.is_models_loaded,
        }

    def warmup(self) -> None:
        """Warmup models using IndicTrans2 native approach."""
        logger.info("Warmup: loading IndicTrans2 models...")
        
        start_time = time.time()
        
        # Track loading results
        en_indic_loaded = False
        indic_en_loaded = False
        en_indic_error = None
        indic_en_error = None
        
        def load_en_indic():
            nonlocal en_indic_loaded, en_indic_error
            try:
                logger.info("Starting EN->Indic model loading...")
                self._ensure_en_indic_model()
                en_indic_loaded = True
                logger.info("EN->Indic model ready")
            except Exception as e:
                en_indic_error = e
                logger.error(f"Failed to load EN->Indic model: {e}")
        
        def load_indic_en():
            nonlocal indic_en_loaded, indic_en_error
            try:
                logger.info("Starting Indic->EN model loading...")
                self._ensure_indic_en_model()
                indic_en_loaded = True
                logger.info("Indic->EN model ready")
            except Exception as e:
                indic_en_error = e
                logger.error(f"Failed to load Indic->EN model: {e}")
        
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
        logger.info(f"EN->Indic loaded: {en_indic_loaded}, Indic->EN loaded: {indic_en_loaded}")
        
        # Verify models loaded
        if not en_indic_loaded and not indic_en_loaded:
            error_msg = "Model loading failed - no models available"
            if en_indic_error:
                error_msg += f" (EN->Indic error: {en_indic_error})"
            if indic_en_error:
                error_msg += f" (Indic->EN error: {indic_en_error})"
            
            logger.error(f"CRITICAL: {error_msg}")
            raise RuntimeError(error_msg)
        
        # Test translations if models loaded successfully
        try:
            if en_indic_loaded and self.en_indic_engine is not None:
                logger.info("Testing EN->HI translation...")
                test_en_hi = self._translate_en_to_indic("Hello", "hi")
                logger.info(f"Warmup test EN->HI: {test_en_hi}")
            else:
                logger.warning("EN->Indic model not loaded, skipping warmup test")
            
            if indic_en_loaded and self.indic_en_engine is not None:
                logger.info("Testing HI->EN translation...")
                test_hi_en = self._translate_indic_to_en("नमस्ते", "hi")
                logger.info(f"Warmup test HI->EN: {test_hi_en}")
            else:
                logger.warning("Indic->EN model not loaded, skipping warmup test")
                
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
        
        logger.info(f"Translation to all languages completed. Available languages: {list(result.keys())}")
        return result

# Create singleton instance
translation_service = TranslationService()