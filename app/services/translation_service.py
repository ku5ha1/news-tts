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


    def _get_cache_dir(self):
        """Resolve Hugging Face cache dir inside container."""
        cache_dir = Path(
            os.environ.get(
                "TRANSFORMERS_CACHE",
                "/app/.cache/huggingface/transformers",
            )
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir


    def _ensure_en_indic_model(self):
        """Load EN→Indic model (use local path first)."""
        if self.model_en_indic is None and not self.loading_en_indic:
            try:
                self.loading_en_indic = True
                
                # Check for baked-in model first
                local_path = "/app/models/indictrans2-en-indic-dist-200M"
                if os.path.exists(local_path):
                    logger.info(f"Loading EN→Indic from local path: {local_path}")
                    model_path = local_path
                else:
                    logger.info("Loading EN→Indic from HuggingFace...")
                    model_path = MODEL_NAMES["en_indic"]
                
                cache_dir = self._get_cache_dir()
                
                self.tokenizer_en_indic = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )

                self.model_en_indic = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float32  # CPU uses float32
                ).to(self.device).eval()
                
                logger.info("EN→Indic dist-200M model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load EN→Indic model: {str(e)}", exc_info=True)
                raise
            finally:
                self.loading_en_indic = False


    def _ensure_indic_en_model(self):
        """Load Indic→EN model (use local path first)."""
        if self.model_indic_en is None and not self.loading_indic_en:
            try:
                self.loading_indic_en = True
                
                # Check for baked-in model first
                local_path = "/app/models/indictrans2-indic-en-dist-200M"
                if os.path.exists(local_path):
                    logger.info(f"Loading Indic→EN from local path: {local_path}")
                    model_path = local_path
                else:
                    logger.info("Loading Indic→EN from HuggingFace...")
                    model_path = MODEL_NAMES["indic_en"]
                
                cache_dir = self._get_cache_dir()
                
                self.tokenizer_indic_en = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )

                self.model_indic_en = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float32  # CPU uses float32
                ).to(self.device).eval()
                
                logger.info("Indic→EN dist-200M model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Indic→EN model: {str(e)}", exc_info=True)
                raise
            finally:
                self.loading_indic_en = False


    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Main translate method with correct routing."""
        logger.info(f"Translating: '{text[:50]}...' from {source_lang} to {target_lang}")
        
        # Route to correct model based on direction
        if source_lang == "en" and target_lang in ["hi", "kn"]:
            # English to Indic
            logger.info("Using EN→Indic model")
            return self._translate(text, source_lang, target_lang)
        elif source_lang in ["hi", "kn"] and target_lang == "en":
            # Indic to English  
            logger.info("Using Indic→EN model")
            return self._translate(text, source_lang, target_lang)
        else:
            raise ValueError(f"Unsupported translation: {source_lang} → {target_lang}")


    def _translate(self, text: str, source: str, target: str) -> str:
        """Translate text using IndicTrans2 dist-200M models."""
        try:
            if not text.strip():
                return text

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
            def _ensure_ready_en_indic(timeout_sec: float = 300.0):
                start = time.monotonic()
                self._ensure_en_indic_model()
                while self.model_en_indic is None or self.tokenizer_en_indic is None:
                    if time.monotonic() - start > timeout_sec:
                        raise TimeoutError("Timeout waiting for EN→Indic model to load")
                    if not self.loading_en_indic:
                        self._ensure_en_indic_model()
                    time.sleep(0.5)

            def _ensure_ready_indic_en(timeout_sec: float = 300.0):
                start = time.monotonic()
                self._ensure_indic_en_model()
                while self.model_indic_en is None or self.tokenizer_indic_en is None:
                    if time.monotonic() - start > timeout_sec:
                        raise TimeoutError("Timeout waiting for Indic→EN model to load")
                    if not self.loading_indic_en:
                        self._ensure_indic_en_model()
                    time.sleep(0.5)

            if source == "en":
                logger.info("Ensuring EN->Indic dist-200M model is ready")
                _ensure_ready_en_indic()
                tokenizer, model = self.tokenizer_en_indic, self.model_en_indic
            else:
                logger.info("Ensuring Indic->EN dist-200M model is ready")
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
                        max_length=512  # Limit for dist-200M
                    ).to(self.device)
                    
                    logger.info(f"Input shape: {inputs['input_ids'].shape}")
                    
                    logger.info("Starting model generation...")
                    outputs = model.generate(
                        **inputs,
                        use_cache=True,
                        min_length=0,
                        max_length=256,  # Smaller max length for dist-200M
                        num_beams=3,    # Reduced beams for speed
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
                logger.info("Postprocess.start")
                translations = self.ip.postprocess_batch(decoded, lang=tgt_tag)
                logger.info("Postprocess.end")
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
            raise


    async def translate_async(self, text: str, source: str, target: str) -> str:
        """Async wrapper so routes can await translation."""
        try:
            import concurrent.futures
            logger.info(f"translate_async.start {source}->{target} text_len={len(text)}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(executor, self.translate, text, source, target)
                logger.info(f"translate_async.done {source}->{target} result_len={len(result) if result else 0}")
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
                    "title": self.translate(title, source_lang, lang),
                    "description": self.translate(description, source_lang, lang),
                }
            return result
        except Exception as e:
            logger.error(f"Error in translate_to_all: {str(e)}")
            return {}


    async def translate_to_all_async(self, title: str, description: str, source_lang: str):
        """Translate to all supported langs concurrently."""
        try:
            import concurrent.futures
            languages = ["en", "hi", "kn"]
            loop = asyncio.get_event_loop()
            logger.info(f"translate_to_all_async.start source={source_lang} langs={languages}")

            async def run_for_lang(lang: str):
                if lang == source_lang:
                    logger.info(f"translate_to_all_async.skip source_lang==target_lang {lang}")
                    return None, None
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    per_call_timeout = float(os.getenv("TRANSLATION_PER_CALL_TIMEOUT", "30"))
                    per_call_timeout_retry = float(os.getenv("TRANSLATION_PER_CALL_TIMEOUT_RETRY", str(min(60, int(per_call_timeout) * 2))))
                    logger.info(f"translate_to_all_async.lang.start {source_lang}->{lang} (title) timeout={per_call_timeout}s")
                    try:
                        t = await asyncio.wait_for(
                            loop.run_in_executor(executor, self.translate, title, source_lang, lang),
                            timeout=per_call_timeout,
                        )
                        logger.info(f"translate_to_all_async.lang.done {source_lang}->{lang} (title)")
                    except asyncio.TimeoutError as e:
                        logger.warning(f"translate_to_all_async.lang.timeout {source_lang}->{lang} (title) t={per_call_timeout}s; retrying with {per_call_timeout_retry}s")
                        try:
                            t = await asyncio.wait_for(
                                loop.run_in_executor(executor, self.translate, title, source_lang, lang),
                                timeout=per_call_timeout_retry,
                            )
                            logger.info(f"translate_to_all_async.lang.done.retry {source_lang}->{lang} (title)")
                        except Exception as e2:
                            logger.warning(f"translate_to_all_async.lang.retry_failed {source_lang}->{lang} (title) err={e2}; using original")
                            t = title
                    except Exception as e:
                        logger.warning(f"translate_to_all_async.lang.error {source_lang}->{lang} (title) err={e}; using original")
                        t = title

                    logger.info(f"translate_to_all_async.lang.start {source_lang}->{lang} (description) timeout={per_call_timeout}s")
                    try:
                        d = await asyncio.wait_for(
                            loop.run_in_executor(executor, self.translate, description, source_lang, lang),
                            timeout=per_call_timeout,
                        )
                        logger.info(f"translate_to_all_async.lang.done {source_lang}->{lang} (description)")
                    except asyncio.TimeoutError as e:
                        logger.warning(f"translate_to_all_async.lang.timeout {source_lang}->{lang} (description) t={per_call_timeout}s; retrying with {per_call_timeout_retry}s")
                        try:
                            d = await asyncio.wait_for(
                                loop.run_in_executor(executor, self.translate, description, source_lang, lang),
                                timeout=per_call_timeout_retry,
                            )
                            logger.info(f"translate_to_all_async.lang.done.retry {source_lang}->{lang} (description)")
                        except Exception as e2:
                            logger.warning(f"translate_to_all_async.lang.retry_failed {source_lang}->{lang} (description) err={e2}; using original")
                            d = description
                    except Exception as e:
                        logger.warning(f"translate_to_all_async.lang.error {source_lang}->{lang} (description) err={e}; using original")
                        d = description
                return lang, {"title": t, "description": d}

            tasks = [run_for_lang(l) for l in languages]
            results = await asyncio.gather(*tasks)
            out = {}
            for lang, payload in results:
                if lang and payload:
                    out[lang] = payload
            logger.info(f"translate_to_all_async.done langs={list(out.keys())}")
            return out
        except Exception as e:
            logger.error(f"Error in translate_to_all_async: {str(e)}")
            return {}


    @property
    def is_models_loaded(self) -> bool:
        """Check if any models are loaded"""
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
        """Warm up the translation models with a simple test."""
        try:
            logger.info("Warmup: loading dist-200M models and priming preprocess...")
            
            # Load both models
            self._ensure_en_indic_model()
            self._ensure_indic_en_model()
            
            # Simple test translation (FIXED: no tuple unpacking)
            test_result = self.translate("Hello", "en", "hi")
            
            if test_result:
                logger.info("Warmup completed successfully")
            else:
                logger.info("Warmup completed with warnings")
                
        except Exception as e:
            logger.warning(f"Warmup failed (continuing): {e}")
            # Don't raise - let the service continue


# Singleton instance
translation_service = TranslationService()
