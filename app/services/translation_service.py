import torch
import os
import asyncio
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit import IndicProcessor

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

            # Track loading status
            self.loading_en_indic = False
            self.loading_indic_en = False

    def _ensure_en_indic_model(self):
        """Load EN->Indic model and tokenizer if not already loaded."""
        if self.en_indic_model is None and not self.loading_en_indic:
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
                    use_cache=False,  # Disable cache to avoid past_key_values issue
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                    do_sample=False,  # Use deterministic generation
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
                    use_cache=False,  # Disable cache to avoid past_key_values issue
                        min_length=0,
                    max_length=256,
                    num_beams=5,
                        num_return_sequences=1,
                    do_sample=False,  # Use deterministic generation
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
        """Translate title and description to specific languages based on source."""
        # Normalize source language
        source_lang = self._normalize_lang_code(source_lang)
        
        # Define target languages based on source
        if source_lang == "english":
            target_languages = ["hindi", "kannada"]
        elif source_lang == "kannada":
            target_languages = ["english", "hindi"]
        else:
            # For other languages, translate to English and Hindi
            target_languages = ["english", "hindi"]
        
        translations = {}
        
        # Run translations in parallel
        tasks = []
        for lang in target_languages:
            if lang.lower() != source_lang.lower():
                task = asyncio.create_task(
                    self._translate_async(title, description, source_lang, lang)
                )
                tasks.append((lang, task))
        
        # Collect results
        for lang, task in tasks:
            try:
                result = await task
                translations[lang] = result
            except Exception as e:
                logger.error(f"Failed to translate to {lang}: {e}")
                translations[lang] = {"title": title, "description": description}
        
        return translations

    async def _translate_async(self, title: str, description: str, source_lang: str, target_lang: str) -> dict:
        """Async wrapper for translation."""
        loop = asyncio.get_event_loop()
        
        # Run translation in thread pool to avoid blocking
        translated_title = await loop.run_in_executor(
            None, self.translate, title, source_lang, target_lang
        )
        translated_description = await loop.run_in_executor(
            None, self.translate, description, source_lang, target_lang
        )
        
        return {
                    "title": translated_title,
                    "description": translated_description
                }

    def warmup(self):
        """Warmup method for service initialization."""
        try:
            logger.info("Starting translation service warmup...")
            
            # Test translation to ensure models work
            test_text = "Hello world"
            test_result = self.translate(test_text, "english", "hindi")
            
            logger.info(f"Warmup successful: '{test_text}' -> '{test_result}'")
            return True
            
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            raise
        
# Create singleton instance
translation_service = TranslationService()
