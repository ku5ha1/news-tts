import torch
import os
import asyncio
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from threading import Lock

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle IndicTransToolkit import with fallback
try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    logger.error("Failed to import IndicProcessor from IndicTransToolkit.processor")
    IndicProcessor = None

# Model names for 1B (CORRECT MODELS)
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
            
            # CRITICAL: Enable CPU optimizations
            torch.set_num_threads(2)  # Use both P1V3 cores
            torch.set_num_interop_threads(2)
            
            logger.info("[Translation] Using CPU for 1B model with optimizations")

            # Initialize IndicTransToolkit components with graceful degradation
            self.ip = None
            self._initialization_error = None
            
            try:
                if IndicProcessor is None:
                    logger.warning("IndicProcessor is not available - using fallback transformers-only mode")
                    self.ip = None  # Use fallback mode
                    logger.info("Using fallback translation mode without IndicTransToolkit")
                else:
                    logger.info("Initializing IndicTransToolkit components...")
                    self.ip = IndicProcessor(inference=True)
                    logger.info("IndicTransToolkit components initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize IndicTransToolkit components: {e}")
                self._initialization_error = str(e)
                # Don't raise - allow service to start in degraded mode

            # Placeholders for lazy model load
            self.en_indic_model = None
            self.indic_en_model = None
            self.en_indic_tokenizer = None
            self.indic_en_tokenizer = None

            # Thread locks for model operations
            self.en_indic_lock = Lock()
            self.indic_en_lock = Lock()

            # Track loading status
            self.loading_en_indic = False
            self.loading_indic_en = False
            
            # Circuit breaker state
            self._circuit_open = False
            self._failure_count = 0
            self._last_failure_time = None

    def _check_circuit_breaker(self):
        """Check if circuit breaker should prevent operations."""
        if self._circuit_open:
            import time
            # Reset circuit after 5 minutes
            if time.time() - self._last_failure_time > 300:
                self._circuit_open = False
                self._failure_count = 0
                logger.info("Circuit breaker reset - attempting operations again")
            else:
                raise RuntimeError("Circuit breaker is open - translation service unavailable")
    
    def _record_failure(self):
        """Record a failure for circuit breaker."""
        import time
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        # Open circuit after 3 consecutive failures
        if self._failure_count >= 3:
            self._circuit_open = True
            logger.error(f"Circuit breaker opened after {self._failure_count} failures")
    
    def _record_success(self):
        """Record a success for circuit breaker."""
        self._failure_count = 0
        self._circuit_open = False
    
    def get_health_status(self):
        """Get health status of the translation service."""
        return {
            "initialization_error": self._initialization_error,
            "circuit_open": self._circuit_open,
            "failure_count": self._failure_count,
            "models_loaded": {
                "en_indic": self.en_indic_model is not None,
                "indic_en": self.indic_en_model is not None
            },
            "indictrans_available": self.ip is not None
        }

    def _ensure_en_indic_model(self):
        """Load EN->Indic model and tokenizer if not already loaded."""
        if self.en_indic_model is None and not self.loading_en_indic:
            with self.en_indic_lock:
                if self.en_indic_model is None:
                    try:
                        self.loading_en_indic = True
                        model_name = MODEL_NAMES["en_indic"]
                        logger.info(f"Loading EN->Indic 1B model: {model_name}")
                                
                        self.en_indic_tokenizer = AutoTokenizer.from_pretrained(
                                    model_name, trust_remote_code=True
                                )
                        self.en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
                                    model_name, trust_remote_code=True
                                )
                        self.en_indic_model.eval()  # Set to evaluation mode
                                
                        logger.info("EN->Indic 1B model loaded successfully")
                    except Exception as e:
                        logger.error(f"Failed to load EN->Indic model: {e}")
                        self.en_indic_model = None
                        self.en_indic_tokenizer = None
                        raise
                    finally:
                        self.loading_en_indic = False

    def _ensure_indic_en_model(self):
        """Load Indic->EN model and tokenizer if not already loaded."""
        if self.indic_en_model is None and not self.loading_indic_en:
            with self.indic_en_lock:
                if self.indic_en_model is None:
                    try:
                        self.loading_indic_en = True
                        model_name = MODEL_NAMES["indic_en"]
                        logger.info(f"Loading Indic->EN 1B model: {model_name}")
                        
                        self.indic_en_tokenizer = AutoTokenizer.from_pretrained(
                            model_name, trust_remote_code=True
                        )
                        self.indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name, trust_remote_code=True
                        )
                        self.indic_en_model.eval()  # Set to evaluation mode
                        
                        logger.info("Indic->EN 1B model loaded successfully")
                    except Exception as e:
                        logger.error(f"Failed to load Indic->EN model: {e}")
                        self.indic_en_model = None
                        self.indic_en_tokenizer = None
                        raise
                    finally:
                        self.loading_indic_en = False

    def _normalize_lang_code(self, lang: str) -> str:
        """Normalize language codes from detection to internal format."""
        lang = lang.lower().strip()
        
        if lang in ["en", "english"]:
            return "english"
        elif lang in ["hi", "hindi"]:
            return "hindi"
        elif lang in ["kn", "kannada"]:
            return "kannada"
        else:
            return "english"

    def _get_lang_code(self, lang: str) -> str:
        """Convert language name to IndicTrans2 language code."""
        lang_map = {
            "hindi": "hin_Deva",
            "kannada": "kan_Knda",
            "english": "eng_Latn"
        }
        return lang_map.get(lang.lower(), "hin_Deva")

    def _translate_en_to_indic_batch(self, texts: list, target_langs: list) -> dict:
        """BATCH translate English to multiple Indic languages in ONE model call."""
        with self.en_indic_lock:
            try:
                self._ensure_en_indic_model()
                if self.en_indic_model is None or self.en_indic_tokenizer is None:
                    raise RuntimeError("EN->Indic model not loaded")
                
                logger.info(f"Batch translating {len(texts)} texts to {len(target_langs)} languages")
                
                # Prepare batch for all target languages
                all_batches = []
                for text, target_lang in zip(texts, target_langs):
                    batch = self.ip.preprocess_batch(
                        [text], 
                        src_lang="eng_Latn", 
                        tgt_lang=self._get_lang_code(target_lang)
                    )
                    all_batches.extend(batch)
                
                # Tokenize all at once
                inputs = self.en_indic_tokenizer(
                    all_batches,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                ).to(self.device)

                # Single inference for all languages
                with torch.no_grad():
                    generated_tokens = self.en_indic_model.generate(
                        **inputs,
                        use_cache=False,
                        min_length=0,
                        max_length=128,
                        num_beams=1,  # Greedy for speed
                        num_return_sequences=1,
                        pad_token_id=self.en_indic_tokenizer.pad_token_id,
                        early_stopping=True,
                    )

                # Decode all results
                with self.en_indic_tokenizer.as_target_tokenizer():
                    decoded = self.en_indic_tokenizer.batch_decode(
                        generated_tokens.detach().cpu().tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )

                # Postprocess each language
                results = {}
                for i, target_lang in enumerate(target_langs):
                    translated = self.ip.postprocess_batch(
                        [decoded[i]], 
                        lang=self._get_lang_code(target_lang)
                    )
                    results[target_lang] = translated[0] if translated else texts[i]
                    logger.info(f"Translated to {target_lang}: {results[target_lang][:50]}...")
                
                return results
                
            except Exception as e:
                logger.error(f"Batch translation error: {e}")
                raise

    def _translate_indic_to_en(self, text: str, source_lang: str) -> str:
        """Translate Indic language text to English."""
        with self.indic_en_lock:
            try:
                logger.info(f"Translating {source_lang}->EN: {text[:50]}...")
                
                self._ensure_indic_en_model()
                if self.indic_en_model is None or self.indic_en_tokenizer is None:
                    raise RuntimeError("Indic->EN model not loaded")

                batch = self.ip.preprocess_batch(
                    [text], 
                    src_lang=self._get_lang_code(source_lang), 
                    tgt_lang="eng_Latn"
                )

                inputs = self.indic_en_tokenizer(
                    batch,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                ).to(self.device)

                with torch.no_grad():
                    generated_tokens = self.indic_en_model.generate(
                        **inputs,
                        use_cache=False,
                        min_length=0,
                        max_length=128,
                        num_beams=1,
                        num_return_sequences=1,
                        pad_token_id=self.indic_en_tokenizer.pad_token_id,
                        early_stopping=True,
                    )

                with self.indic_en_tokenizer.as_target_tokenizer():
                    generated_tokens = self.indic_en_tokenizer.batch_decode(
                        generated_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                
                translations = self.ip.postprocess_batch(
                    generated_tokens, 
                    lang="eng_Latn"
                )

                result = translations[0] if translations else text
                logger.info(f"Translation result: {result[:50]}...")
                return result
                    
            except Exception as e:
                logger.error(f"Translation error {source_lang}->EN: {e}")
            raise

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Single translation method."""
        try:
            # Check if IndicTransToolkit is available
            if self.ip is None:
                logger.warning("IndicTransToolkit not available - returning original text")
                return text
            
            source_lang = self._normalize_lang_code(source_lang)
            target_lang = self._normalize_lang_code(target_lang)
            
            if source_lang == target_lang:
                return text
            
            if source_lang == "english":
                results = self._translate_en_to_indic_batch([text], [target_lang])
                return results[target_lang]
            elif target_lang == "english":
                return self._translate_indic_to_en(text, source_lang)
            else:
                # Translate through English
                english_text = self._translate_indic_to_en(text, source_lang)
                results = self._translate_en_to_indic_batch([english_text], [target_lang])
                return results[target_lang]
        except Exception as e:
            logger.error(f"Translation failed {source_lang}->{target_lang}: {e}")
            raise

    async def translate_to_all_async(self, title: str, description: str, source_lang: str) -> dict:
        """BATCH translate to multiple languages efficiently."""
        source_lang = self._normalize_lang_code(source_lang)
        
        if source_lang == "english":
            target_languages = ["hindi", "kannada"]
        elif source_lang == "kannada":
            target_languages = ["english", "hindi"]
        else:
            target_languages = ["english", "hindi"]
        
        # PRE-LOAD models
        logger.info("Pre-loading models...")
        if source_lang == "english":
                self._ensure_en_indic_model()
        else:
                self._ensure_indic_en_model()
        logger.info("Models pre-loaded successfully")
        
        # Combine title and description
        combined_text = f"{title}. {description}"
        
        # BATCH translate to all target languages at once
        loop = asyncio.get_event_loop()
        
        if source_lang == "english":
            # Batch translate to both Hindi and Kannada in ONE call
            texts = [combined_text, combined_text]
            results = await loop.run_in_executor(
                None,
                self._translate_en_to_indic_batch,
                texts,
                target_languages
            )
        else:
            # First translate to English, then batch to others
            english_text = await loop.run_in_executor(
                None,
                self._translate_indic_to_en,
                combined_text,
                source_lang
            )
            
            # Filter out english if it's already a target
            indic_targets = [lang for lang in target_languages if lang != "english"]
            
            results = {"english": english_text}
            
            if indic_targets:
                self._ensure_en_indic_model()
                texts = [english_text] * len(indic_targets)
                indic_results = await loop.run_in_executor(
                    None,
                    self._translate_en_to_indic_batch,
                    texts,
                    indic_targets
                )
                results.update(indic_results)
        
        # Split back into title and description
        translations = {}
        for lang, translated_text in results.items():
            parts = translated_text.split('. ', 1)
            translations[lang] = {
                "title": parts[0] if parts else translated_text,
                "description": parts[1] if len(parts) > 1 else translated_text
            }
        
        logger.info(f"Batch translation completed: {list(translations.keys())}")
        return translations

# Create singleton instance
translation_service = TranslationService()