import torch
import os
import asyncio
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit import IndicProcessor
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model names for dist-200M
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

            # Initialize IndicTransToolkit components
            try:
                logger.info("Initializing IndicTransToolkit components...")
                self.ip = IndicProcessor(inference=True)
                logger.info("IndicTransToolkit components initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize IndicTransToolkit components: {e}")
                raise RuntimeError(f"IndicTransToolkit initialization failed: {e}")

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

    def _ensure_en_indic_model(self):
        """Load EN->Indic model and tokenizer if not already loaded."""
        if self.en_indic_model is None and not self.loading_en_indic:
            with self.en_indic_lock:  # Thread-safe loading
                if self.en_indic_model is None:  # Double-check
                    try:
                        self.loading_en_indic = True
                        model_name = MODEL_NAMES["en_indic"]
                        logger.info(f"Loading EN->Indic model: {model_name}")
                        
                        # Load tokenizer and model
                        self.en_indic_tokenizer = AutoTokenizer.from_pretrained(
                            model_name, trust_remote_code=True
                        )
                        self.en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name, trust_remote_code=True
                        )
                        
                        logger.info("EN->Indic model loaded successfully")
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
            with self.indic_en_lock:  # Thread-safe loading
                if self.indic_en_model is None:  # Double-check
                    try:
                        self.loading_indic_en = True
                        model_name = MODEL_NAMES["indic_en"]
                        logger.info(f"Loading Indic->EN model: {model_name}")
                        
                        # Load tokenizer and model
                        self.indic_en_tokenizer = AutoTokenizer.from_pretrained(
                            model_name, trust_remote_code=True
                        )
                        self.indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name, trust_remote_code=True
                        )
                        
                        logger.info("Indic->EN model loaded successfully")
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
        
        # Map detection codes to internal codes (only 3 languages supported)
        if lang in ["en", "english"]:
            return "english"
        elif lang in ["hi", "hindi"]:
            return "hindi"
        elif lang in ["kn", "kannada"]:
            return "kannada"
        else:
            # Default to English if unknown
            return "english"

    def _get_lang_code(self, lang: str) -> str:
        """Convert language name to IndicTrans2 language code."""
        lang_map = {
            "hindi": "hin_Deva",
            "kannada": "kan_Knda",
            "english": "eng_Latn"
        }
        return lang_map.get(lang.lower(), "hin_Deva")

    def _translate_en_to_indic(self, text: str, target_lang: str) -> str:
        """Translate English text to target Indic language using IndicTransToolkit."""
        # Use lock to serialize model access
        with self.en_indic_lock:
            try:
                logger.info(f"Translating EN->{target_lang}: {text[:50]}...")
                
                # Ensure model is loaded
                self._ensure_en_indic_model()
                if self.en_indic_model is None or self.en_indic_tokenizer is None:
                    raise RuntimeError("EN->Indic model not loaded")
                
                # Preprocess text
                batch = self.ip.preprocess_batch(
                    [text], 
                    src_lang="eng_Latn", 
                    tgt_lang=self._get_lang_code(target_lang)
                )

                # Tokenize
                inputs = self.en_indic_tokenizer(
                    batch,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                ).to(self.device)

                # Generate translation
                with torch.no_grad():
                    generated_tokens = self.en_indic_model.generate(
                        **inputs,
                        use_cache=False,  # FIX: Disable cache to avoid past_key_values bug
                        min_length=0,
                        max_length=256,
                        num_beams=5,
                        num_return_sequences=1,
                        pad_token_id=self.en_indic_tokenizer.pad_token_id,
                    )

                # Decode tokens
                with self.en_indic_tokenizer.as_target_tokenizer():
                    generated_tokens = self.en_indic_tokenizer.batch_decode(
                        generated_tokens.detach().cpu().tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )

                # Postprocess
                translations = self.ip.postprocess_batch(
                    generated_tokens, 
                    lang=self._get_lang_code(target_lang)
                )

                result = translations[0] if translations else text
                logger.info(f"Translation result: {result[:50]}...")
                return result
                    
            except Exception as e:
                logger.error(f"Translation error EN->{target_lang}: {e}")
                raise

    def _translate_indic_to_en(self, text: str, source_lang: str) -> str:
        """Translate Indic language text to English using IndicTransToolkit."""
        # Use lock to serialize model access
        with self.indic_en_lock:
            try:
                logger.info(f"Translating {source_lang}->EN: {text[:50]}...")
                
                # Ensure model is loaded
                self._ensure_indic_en_model()
                if self.indic_en_model is None or self.indic_en_tokenizer is None:
                    raise RuntimeError("Indic->EN model not loaded")

                # Preprocess text
                batch = self.ip.preprocess_batch(
                    [text], 
                    src_lang=self._get_lang_code(source_lang), 
                    tgt_lang="eng_Latn"
                )

                # Tokenize
                inputs = self.indic_en_tokenizer(
                    batch,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                ).to(self.device)

                # Generate translation
                with torch.no_grad():
                    generated_tokens = self.indic_en_model.generate(
                        **inputs,
                        use_cache=False,  # FIX: Disable cache to avoid past_key_values bug
                        min_length=0,
                        max_length=256,
                        num_beams=5,
                        num_return_sequences=1,
                        pad_token_id=self.indic_en_tokenizer.pad_token_id,
                    )

                # Decode tokens
                with self.indic_en_tokenizer.as_target_tokenizer():
                    generated_tokens = self.indic_en_tokenizer.batch_decode(
                        generated_tokens.detach().cpu().tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )

                # Postprocess
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
        """Main translation method."""
        try:
            # Normalize language codes
            source_lang = self._normalize_lang_code(source_lang)
            target_lang = self._normalize_lang_code(target_lang)
            
            if source_lang == target_lang:
                return text
            
            if source_lang == "english":
                return self._translate_en_to_indic(text, target_lang)
            elif target_lang == "english":
                return self._translate_indic_to_en(text, source_lang)
            else:
                # Translate through English
                english_text = self._translate_indic_to_en(text, source_lang)
                return self._translate_en_to_indic(english_text, target_lang)
        except Exception as e:
            logger.error(f"Translation failed {source_lang}->{target_lang}: {e}")
            raise

    async def translate_to_all_async(self, title: str, description: str, source_lang: str) -> dict:
        """Translate title and description to specific languages - SEQUENTIAL for stability."""
        # Normalize source language
        source_lang = self._normalize_lang_code(source_lang)
        
        # Define target languages based on source
        if source_lang == "english":
            target_languages = ["hi", "kn"]  # Use short codes for API compatibility
        elif source_lang == "kannada":
            target_languages = ["en", "hi"]
        else:
            # For other languages, translate to English and Hindi
            target_languages = ["en", "hi"]
        
        # PRE-LOAD MODELS to avoid contention during execution
        logger.info("Pre-loading models to avoid contention...")
        if source_lang == "english":
            self._ensure_en_indic_model()
        else:
            self._ensure_indic_en_model()
            # If we need both models (indic->indic via English), load both
            if source_lang != "english" and "english" not in target_languages:
                self._ensure_en_indic_model()
        logger.info("Models pre-loaded successfully")
        
        translations = {}
        
        # SEQUENTIAL TRANSLATION - More reliable than parallel on CPU
        loop = asyncio.get_event_loop()
        
        for lang in target_languages:
            if lang.lower() != source_lang.lower():
                try:
                    # Combine title and description
                    combined_text = f"{title}. {description}"
                    
                    logger.info(f"Translating {source_lang}->{lang}...")
                    
                    # Run in executor to avoid blocking
                    translated_text = await loop.run_in_executor(
                        None, 
                        self.translate,
                        combined_text,
                        source_lang,
                        lang
                    )
                    
                    # Split back into title and description
                    parts = translated_text.split('. ', 1)
                    translated_title = parts[0] if parts else translated_text
                    translated_description = parts[1] if len(parts) > 1 else translated_text
                    
                    translations[lang] = {
                        "title": translated_title,
                        "description": translated_description
                    }
                    
                    logger.info(f"Completed translation to {lang}")
                    
                except Exception as e:
                    logger.error(f"Failed to translate to {lang}: {e}")
                    translations[lang] = {"title": title, "description": description}
        
        return translations

# Create singleton instance
translation_service = TranslationService()