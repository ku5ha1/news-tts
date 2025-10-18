import torch
import os
import asyncio
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model names for 200M models
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
            # Device detection as per official snippet
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"[Translation] Using device: {self.device}")

            # Initialize IndicTransToolkit components
            try:
                logger.info("Initializing IndicTransToolkit components...")
                self.ip = IndicProcessor(inference=True)
                logger.info("IndicTransToolkit components initialized successfully")
            except ImportError as e:
                logger.error(f"Failed to import IndicTransToolkit components: {e}")
                raise RuntimeError(f"IndicTransToolkit import failed: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize IndicTransToolkit components: {e}")
                raise RuntimeError(f"IndicTransToolkit initialization failed: {e}")

            # Load models immediately as per official snippet
            try:
                logger.info("Loading EN->Indic model immediately...")
                self.en_indic_tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_NAMES["en_indic"], trust_remote_code=True, token=os.getenv("HUGGINGFACE_ACCESSTOKEN")
                )
                self.en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
                    MODEL_NAMES["en_indic"], 
                    trust_remote_code=True, 
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    token=os.getenv("HUGGINGFACE_ACCESSTOKEN"),
                    # attn_implementation="flash_attention_2" if self.device.type == "cuda" else None
                ).to(self.device)
                self.en_indic_model.eval()
                logger.info("EN->Indic model loaded successfully")
            except ImportError as e:
                logger.error(f"Failed to import EN->Indic model: {e}")
                raise RuntimeError(f"EN->Indic model import failed: {e}")
            except OSError as e:
                logger.error(f"Failed to load EN->Indic model from disk: {e}")
                raise RuntimeError(f"EN->Indic model loading failed: {e}")
            except Exception as e:
                logger.error(f"Failed to load EN->Indic model: {e}")
                raise RuntimeError(f"EN->Indic model loading failed: {e}")

            try:
                logger.info("Loading Indic->EN model immediately...")
                self.indic_en_tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_NAMES["indic_en"], trust_remote_code=True, token=os.getenv("HUGGINGFACE_ACCESSTOKEN")
                )
                self.indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(
                    MODEL_NAMES["indic_en"], 
                    trust_remote_code=True, 
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    token=os.getenv("HUGGINGFACE_ACCESSTOKEN")
                ).to(self.device)
                self.indic_en_model.eval()
                logger.info("Indic->EN model loaded successfully")
            except ImportError as e:
                logger.error(f"Failed to import Indic->EN model: {e}")
                raise RuntimeError(f"Indic->EN model import failed: {e}")
            except OSError as e:
                logger.error(f"Failed to load Indic->EN model from disk: {e}")
                raise RuntimeError(f"Indic->EN model loading failed: {e}")
            except Exception as e:
                logger.error(f"Failed to load Indic->EN model: {e}")
                raise RuntimeError(f"Indic->EN model loading failed: {e}")

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

    def _translate_en_to_indic(self, text: str, target_lang: str) -> str:
        """Translate English to Indic language using official snippet approach."""
        try:
            src_lang, tgt_lang = "eng_Latn", self._get_lang_code(target_lang)
            
            # Preprocess as per official snippet
            batch = self.ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
            
            # Tokenize as per official snippet
            inputs = self.en_indic_tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)

            # Generate with safe parameters to avoid KV cache issues
            with torch.no_grad():
                generated_tokens = self.en_indic_model.generate(
                    **inputs,
                    use_cache=False,  # Disable cache to avoid KV cache issues
                    max_length=256,
                    num_beams=1,      # Single beam to avoid beam search conflicts
                    num_return_sequences=1,
                )

            # Decode as per official snippet
            generated_tokens = self.en_indic_tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # Postprocess as per official snippet
            translations = self.ip.postprocess_batch(generated_tokens, lang=tgt_lang)
            
            result = translations[0] if translations else text
            logger.info(f"Translated to {target_lang}: {result[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Translation error EN->{target_lang}: {e}")
            raise

    def _translate_indic_to_en(self, text: str, source_lang: str) -> str:
        """Translate Indic language to English using official snippet approach."""
        try:
            src_lang, tgt_lang = self._get_lang_code(source_lang), "eng_Latn"
            
            # Preprocess as per official snippet
            batch = self.ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
            
            # Tokenize as per official snippet
            inputs = self.indic_en_tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)

            # Generate as per official snippet
            with torch.no_grad():
                generated_tokens = self.indic_en_model.generate(
                    **inputs,
                    use_cache=False,
                    max_length=256,
                    num_beams=1,
                    num_return_sequences=1,
                )

            # Decode as per official snippet
            generated_tokens = self.indic_en_tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # Postprocess as per official snippet
            translations = self.ip.postprocess_batch(generated_tokens, lang=tgt_lang)
            
            result = translations[0] if translations else text
            logger.info(f"Translated to English: {result[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Translation error {source_lang}->EN: {e}")
            raise

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Single translation method."""
        try:
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
        """Translate to multiple languages using official snippet approach."""
        try:
            source_lang = self._normalize_lang_code(source_lang)
            
            if source_lang == "english":
                target_languages = ["hindi", "kannada"]
            elif source_lang == "kannada":
                target_languages = ["english", "hindi"]
            else:
                target_languages = ["english", "hindi"]
            
            # Run translations in executor to avoid blocking
            loop = asyncio.get_event_loop()
            translations = {}
            
            if source_lang == "english":
                # Translate title and description separately for better accuracy
                for target_lang in target_languages:
                    logger.info(f"Translating to {target_lang}: title='{title[:50]}...', description='{description[:50]}...'")
                    
                    # Translate title separately
                    translated_title = await loop.run_in_executor(
                        None,
                        self._translate_en_to_indic,
                        title,
                        target_lang
                    )
                    
                    # Translate description separately
                    translated_description = await loop.run_in_executor(
                        None,
                        self._translate_en_to_indic,
                        description,
                        target_lang
                    )
                    
                    translations[target_lang] = {
                        "title": translated_title,
                        "description": translated_description
                    }
                    
                    logger.info(f"Translated {target_lang}: title='{translated_title[:50]}...', description='{translated_description[:50]}...'")
            else:
                # First translate to English
                english_title = await loop.run_in_executor(
                    None,
                    self._translate_indic_to_en,
                    title,
                    source_lang
                )
                english_description = await loop.run_in_executor(
                    None,
                    self._translate_indic_to_en,
                    description,
                    source_lang
                )
                
                translations["english"] = {
                    "title": english_title,
                    "description": english_description
                }
                
                # Then translate English to other languages
                for target_lang in target_languages:
                    if target_lang != "english":
                        logger.info(f"Translating English to {target_lang}: title='{english_title[:50]}...', description='{english_description[:50]}...'")
                        
                        translated_title = await loop.run_in_executor(
                            None,
                            self._translate_en_to_indic,
                            english_title,
                            target_lang
                        )
                        
                        translated_description = await loop.run_in_executor(
                            None,
                            self._translate_en_to_indic,
                            english_description,
                            target_lang
                        )
                        
                        translations[target_lang] = {
                            "title": translated_title,
                            "description": translated_description
                        }
                        
                        logger.info(f"Translated {target_lang}: title='{translated_title[:50]}...', description='{translated_description[:50]}...'")
            
            logger.info(f"Translation completed: {list(translations.keys())}")
            return translations
            
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            raise

# Create singleton instance
translation_service = TranslationService()