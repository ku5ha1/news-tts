import torch
import os
import asyncio
import logging
from typing import Optional, Dict, Any, Tuple
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor


load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
token=os.getenv('HUGGINGFACE_ACCESSTOKEN')

MODEL_NAMES = {
    "en_indic": "ai4bharat/indictrans2-en-indic-dist-200M",
    "indic_en": "ai4bharat/indictrans2-indic-en-dist-200M",
}

class TranslationService:
    """
    Translation Service with dual model support (EN->Indic, Indic->EN).
    Follows ASR pattern: single model instance per worker, async execution.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TranslationService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            # Device detection
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"[Translation] Using device: {self.device}")

            # Model references (lazy loaded)
            self.ip: Optional[Any] = None
            self.en_indic_tokenizer: Optional[Any] = None
            self.en_indic_model: Optional[Any] = None
            self.indic_en_tokenizer: Optional[Any] = None
            self.indic_en_model: Optional[Any] = None
            
            # Initialization flags
            self._en_indic_loaded = False
            self._indic_en_loaded = False
            self._lock = asyncio.Lock()
            self._initialized = True
            
            logger.info("[Translation] Service initialized (models will load on first use)")

    async def _ensure_models_loaded(self):
        """Ensure models are loaded (lazy initialization with lock)."""
        # Quick check without lock (fast path)
        if self._en_indic_loaded and self._indic_en_loaded:
            return
        
        # Only acquire lock if models not loaded
        async with self._lock:
            # Double-check after acquiring lock
            if self._en_indic_loaded and self._indic_en_loaded:
                return
            
            try:
                # Initialize IndicTransToolkit components
                if self.ip is None:
                    logger.info("Initializing IndicTransToolkit components...")
                    self.ip = IndicProcessor(inference=True)
                    logger.info("IndicTransToolkit components initialized successfully")

                # Load models in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                
                # Load EN->Indic model
                if not self._en_indic_loaded:
                    logger.info(f"Loading EN->Indic model: {MODEL_NAMES['en_indic']}")
                    
                    self.en_indic_tokenizer, self.en_indic_model = await asyncio.gather(
                        loop.run_in_executor(
                            None,
                            lambda: AutoTokenizer.from_pretrained(
                                MODEL_NAMES["en_indic"],
                                trust_remote_code=True,
                                token=os.getenv("HUGGINGFACE_ACCESSTOKEN")
                            )
                        ),
                        loop.run_in_executor(
                            None,
                            lambda: AutoModelForSeq2SeqLM.from_pretrained(
                                MODEL_NAMES["en_indic"],
                                trust_remote_code=True,
                                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                                token=token
                            ).to(self.device)
                        )
                    )
                    
                    self.en_indic_model.eval()
                    self._en_indic_loaded = True
                    logger.info(f"EN->Indic model loaded successfully on {self.device}")

                # Load Indic->EN model
                if not self._indic_en_loaded:
                    logger.info(f"Loading Indic->EN model: {MODEL_NAMES['indic_en']}")
                    
                    self.indic_en_tokenizer, self.indic_en_model = await asyncio.gather(
                        loop.run_in_executor(
                            None,
                            lambda: AutoTokenizer.from_pretrained(
                                MODEL_NAMES["indic_en"],
                                trust_remote_code=True,
                                token=os.getenv("HUGGINGFACE_ACCESSTOKEN")
                            )
                        ),
                        loop.run_in_executor(
                            None,
                            lambda: AutoModelForSeq2SeqLM.from_pretrained(
                                MODEL_NAMES["indic_en"],
                                trust_remote_code=True,
                                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                                token=os.getenv("HUGGINGFACE_ACCESSTOKEN")
                            ).to(self.device)
                        )
                    )
                    
                    self.indic_en_model.eval()
                    self._indic_en_loaded = True
                    logger.info(f"Indic->EN model loaded successfully on {self.device}")

                logger.info("All translation models loaded successfully")
                
            except ImportError as e:
                logger.error(f"Failed to import translation models: {e}")
                raise RuntimeError(f"Translation model import failed: {e}")
            except OSError as e:
                logger.error(f"Failed to load translation models from disk: {e}")
                raise RuntimeError(f"Translation model loading failed: {e}")
            except Exception as e:
                logger.error(f"Failed to load translation models: {e}")
                raise RuntimeError(f"Translation model loading failed: {e}")

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

    def _translate_en_to_indic_blocking(self, text: str, target_lang: str) -> str:
        """
        Translate English to Indic language (blocking operation).
        Runs in thread pool via run_in_executor.
        """
        try:
            if not self._en_indic_loaded:
                raise RuntimeError("EN->Indic model not loaded")
            
            src_lang, tgt_lang = "eng_Latn", self._get_lang_code(target_lang)
            
            # Preprocess
            batch = self.ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
            
            # Tokenize
            inputs = self.en_indic_tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)

            # Generate
            with torch.no_grad():
                generated_tokens = self.en_indic_model.generate(
                    **inputs,
                    use_cache=False,
                    max_length=256,
                    num_beams=1,
                    num_return_sequences=1,
                )

            # Decode
            generated_tokens = self.en_indic_tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # Postprocess
            translations = self.ip.postprocess_batch(generated_tokens, lang=tgt_lang)
            
            result = translations[0] if translations else text
            logger.info(f"Translated EN->{target_lang}: {result[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Translation error EN->{target_lang}: {e}")
            raise

    def _translate_indic_to_en_blocking(self, text: str, source_lang: str) -> str:
        """
        Translate Indic language to English (blocking operation).
        Runs in thread pool via run_in_executor.
        """
        try:
            if not self._indic_en_loaded:
                raise RuntimeError("Indic->EN model not loaded")
            
            src_lang, tgt_lang = self._get_lang_code(source_lang), "eng_Latn"
            
            # Preprocess
            batch = self.ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
            
            # Tokenize
            inputs = self.indic_en_tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)

            # Generate
            with torch.no_grad():
                generated_tokens = self.indic_en_model.generate(
                    **inputs,
                    use_cache=False,
                    max_length=256,
                    num_beams=1,
                    num_return_sequences=1,
                )

            # Decode
            generated_tokens = self.indic_en_tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # Postprocess
            translations = self.ip.postprocess_batch(generated_tokens, lang=tgt_lang)
            
            result = translations[0] if translations else text
            logger.info(f"Translated {source_lang}->EN: {result[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Translation error {source_lang}->EN: {e}")
            raise

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Single translation method (async version).
        For backward compatibility with any code using single translations.
        """
        try:
            # Ensure models are loaded
            await self._ensure_models_loaded()
            
            source_lang = self._normalize_lang_code(source_lang)
            target_lang = self._normalize_lang_code(target_lang)
            
            if source_lang == target_lang:
                return text
            
            loop = asyncio.get_event_loop()
            
            if source_lang == "english":
                return await loop.run_in_executor(
                    None,
                    self._translate_en_to_indic_blocking,
                    text,
                    target_lang
                )
            elif target_lang == "english":
                return await loop.run_in_executor(
                    None,
                    self._translate_indic_to_en_blocking,
                    text,
                    source_lang
                )
            else:
                # Translate through English (two steps in sequence)
                english_text = await loop.run_in_executor(
                    None,
                    self._translate_indic_to_en_blocking,
                    text,
                    source_lang
                )
                return await loop.run_in_executor(
                    None,
                    self._translate_en_to_indic_blocking,
                    english_text,
                    target_lang
                )
        except Exception as e:
            logger.error(f"Translation failed {source_lang}->{target_lang}: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._en_indic_loaded and self._indic_en_loaded
    
    async def cleanup(self):
        """Cleanup model resources."""
        if self.en_indic_model:
            del self.en_indic_model
            self.en_indic_model = None
        if self.en_indic_tokenizer:
            del self.en_indic_tokenizer
            self.en_indic_tokenizer = None
        self._en_indic_loaded = False
        
        if self.indic_en_model:
            del self.indic_en_model
            self.indic_en_model = None
        if self.indic_en_tokenizer:
            del self.indic_en_tokenizer
            self.indic_en_tokenizer = None
        self._indic_en_loaded = False
        
        if self.ip:
            del self.ip
            self.ip = None
        
        logger.info("Translation service cleaned up")

    async def translate_to_all_async(self, title: str, description: str, source_lang: str) -> Dict[str, Dict[str, str]]:
        """
        Translate to multiple languages with parallel execution.
        
        Key improvements:
        1. Lazy model loading (ASR pattern)
        2. Parallel translation of multiple languages (2x faster)
        3. Batch title + description together where possible
        
        Args:
            title: Title text to translate
            description: Description text to translate
            source_lang: Source language code
            
        Returns:
            Dict mapping language to {title, description} translations
        """
        try:
            # Ensure models are loaded
            await self._ensure_models_loaded()
            
            source_lang = self._normalize_lang_code(source_lang)
            
            # Determine target languages
            if source_lang == "english":
                target_languages = ["hindi", "kannada"]
            elif source_lang == "kannada":
                target_languages = ["english", "hindi"]
            else:
                target_languages = ["english", "hindi"]
            
            loop = asyncio.get_event_loop()
            translations = {}
            
            if source_lang == "english":
                # Parallel translation: Hindi and Kannada simultaneously
                logger.info(f"Translating from English to {target_languages} (parallel)")
                
                # Create all translation tasks
                tasks = []
                for target_lang in target_languages:
                    # Task for title
                    title_task = loop.run_in_executor(
                        None,
                        self._translate_en_to_indic_blocking,
                        title,
                        target_lang
                    )
                    # Task for description
                    desc_task = loop.run_in_executor(
                        None,
                        self._translate_en_to_indic_blocking,
                        description,
                        target_lang
                    )
                    tasks.append((target_lang, title_task, desc_task))
                
                # Execute all translations in parallel
                for target_lang, title_task, desc_task in tasks:
                    translated_title, translated_description = await asyncio.gather(
                        title_task, desc_task
                    )
                    
                    translations[target_lang] = {
                        "title": translated_title,
                        "description": translated_description
                    }
                    logger.info(f"Completed {target_lang} translation")
                
            else:
                # First translate to English (title and description in parallel)
                logger.info(f"Translating from {source_lang} to English")
                
                english_title_task = loop.run_in_executor(
                    None,
                    self._translate_indic_to_en_blocking,
                    title,
                    source_lang
                )
                english_desc_task = loop.run_in_executor(
                    None,
                    self._translate_indic_to_en_blocking,
                    description,
                    source_lang
                )
                
                english_title, english_description = await asyncio.gather(
                    english_title_task, english_desc_task
                )
                
                translations["english"] = {
                    "title": english_title,
                    "description": english_description
                }
                logger.info("Completed English translation")
                
                # Then translate English to other languages (parallel)
                other_langs = [lang for lang in target_languages if lang != "english"]
                if other_langs:
                    logger.info(f"Translating from English to {other_langs} (parallel)")
                    
                    tasks = []
                    for target_lang in other_langs:
                        title_task = loop.run_in_executor(
                            None,
                            self._translate_en_to_indic_blocking,
                            english_title,
                            target_lang
                        )
                        desc_task = loop.run_in_executor(
                            None,
                            self._translate_en_to_indic_blocking,
                            english_description,
                            target_lang
                        )
                        tasks.append((target_lang, title_task, desc_task))
                    
                    # Execute in parallel
                    for target_lang, title_task, desc_task in tasks:
                        translated_title, translated_description = await asyncio.gather(
                            title_task, desc_task
                        )
                        
                        translations[target_lang] = {
                            "title": translated_title,
                            "description": translated_description
                        }
                        logger.info(f"Completed {target_lang} translation")
            
            logger.info(f"All translations completed: {list(translations.keys())}")
            return translations
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise

# Create singleton instance
translation_service = TranslationService()