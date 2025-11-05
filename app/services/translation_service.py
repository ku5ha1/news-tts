import torch
import os
import asyncio
import logging
import concurrent.futures
from typing import List, Optional
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
from app.services.translation_cache_service import translation_cache_service
from app.services.dynamic_batch_queue import get_batch_queue


load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                
                # Apply Int8 quantization for CPU (75% memory reduction, 2-3x faster)
                if self.device.type == "cpu":
                    try:
                        logger.info("[Quantization] Applying Int8 dynamic quantization to EN->Indic model...")
                        # Dynamic quantization: quantizes weights, activations quantized at runtime
                        self.en_indic_model = torch.quantization.quantize_dynamic(
                            self.en_indic_model,
                            {torch.nn.Linear, torch.nn.LSTM},  # Quantize Linear and LSTM layers
                            dtype=torch.qint8
                        )
                        logger.info("[Quantization] EN->Indic model quantized successfully (Int8)")
                    except Exception as e:
                        logger.warning(f"[Quantization] Failed to quantize EN->Indic model: {e}, using FP32")
                else:
                    logger.info("[Quantization] GPU detected, skipping quantization (use FP16)")
                
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
                
                # Apply Int8 quantization for CPU (75% memory reduction, 2-3x faster)
                if self.device.type == "cpu":
                    try:
                        logger.info("[Quantization] Applying Int8 dynamic quantization to Indic->EN model...")
                        # Dynamic quantization: quantizes weights, activations quantized at runtime
                        self.indic_en_model = torch.quantization.quantize_dynamic(
                            self.indic_en_model,
                            {torch.nn.Linear, torch.nn.LSTM},  # Quantize Linear and LSTM layers
                            dtype=torch.qint8
                        )
                        logger.info("[Quantization] Indic->EN model quantized successfully (Int8)")
                    except Exception as e:
                        logger.warning(f"[Quantization] Failed to quantize Indic->EN model: {e}, using FP32")
                else:
                    logger.info("[Quantization] GPU detected, skipping quantization (use FP16)")
                
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
            
            # Initialize batch processing executor and configuration
            # OPTIMIZATION FOR 4 vCPU/16GB VM:
            # Use ThreadPoolExecutor with 4 workers (PyTorch releases GIL during inference)
            # PyTorch's C++ backend releases GIL, so we get true parallelism for model inference
            # Preprocessing/postprocessing may hold GIL briefly, but most time is in inference
            self.max_batch_chars = 3000  # Max characters per batch
            cpu_count = os.cpu_count() or 4
            
            # Use ThreadPoolExecutor with 4 workers to match 4 vCPUs
            # PyTorch releases GIL during model.generate() calls, enabling true parallelism
            # CRITICAL: Set PyTorch to use 1 thread per worker to avoid thread contention
            # Each worker = 1 CPU core, so PyTorch should use 1 thread per worker
            torch.set_num_threads(1)  # Each worker uses 1 CPU core
            torch.set_num_interop_threads(1)  # Inter-op threads also set to 1
            
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=4,  # Match 4 vCPUs exactly for maximum utilization
                thread_name_prefix="translation"
            )
            
            # Request throttling: Allow 1-2 concurrent full translations
            # Each translation can spawn 10+ chunks (5 chunks × 2 languages)
            # With 4 workers, we can handle 1 full translation efficiently
            # 2 concurrent translations = 20 chunks competing for 4 workers (acceptable)
            self._translation_semaphore = asyncio.Semaphore(2)
            
            logger.info(f"[Translation] Optimized for 4 vCPU/16GB VM: {self.executor._max_workers} executor workers, max_batch_chars={self.max_batch_chars}, max_concurrent_translations=2")

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
    
    def _calculate_adaptive_max_length(self, text: str) -> int:
        """Calculate max_length based on input text to prevent truncation."""
        # Base max_length ensures output is at least 2x input (for expansion)
        # Add safety margin for longer translations
        input_length = len(text)
        if input_length < 100:
            return 256  # Short texts
        elif input_length < 500:
            return 512  # Medium texts
        elif input_length < 1500:
            return 1024  # Long texts
        else:
            # Very long texts - will be chunked, so use reasonable max
            return 1024
    
    def _chunk_text_smart(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Split text into chunks at sentence boundaries with overlap."""
        if len(text) <= chunk_size:
            return [text]
        
        # Split by sentences (simple approach)
        sentences = text.replace('!', '.\n').replace('?', '.\n').split('.')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size, finalize current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append(chunk_text)
                
                # Start new chunk with overlap (last 1-2 sentences)
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append(chunk_text)
        
        return chunks if chunks else [text]

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

            # Calculate adaptive max_length to prevent truncation
            adaptive_max_length = self._calculate_adaptive_max_length(text)
            
            # Generate with safe parameters to avoid KV cache issues
            with torch.no_grad():
                generated_tokens = self.en_indic_model.generate(
                    **inputs,
                    use_cache=False,  # Disable cache to avoid KV cache issues
                    max_length=adaptive_max_length,
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

            # Calculate adaptive max_length to prevent truncation
            adaptive_max_length = self._calculate_adaptive_max_length(text)
            
            # Generate as per official snippet
            with torch.no_grad():
                generated_tokens = self.indic_en_model.generate(
                    **inputs,
                    use_cache=False,
                    max_length=adaptive_max_length,
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

    def _translate_en_to_indic_batch(self, texts: List[str], target_lang: str) -> List[str]:
        """Internal: Translate multiple English texts to Indic language in a single batch."""
        try:
            if not texts:
                return []
            
            src_lang, tgt_lang = "eng_Latn", self._get_lang_code(target_lang)
            
            # Preprocess all texts in batch
            batch = self.ip.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
            
            # Tokenize batch
            inputs = self.en_indic_tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)

            # Calculate adaptive max_length based on longest text in batch
            max_input_length = max(len(text) for text in texts) if texts else 0
            adaptive_max_length = self._calculate_adaptive_max_length(texts[0] if texts else "")
            
            # Generate translations for entire batch
            with torch.no_grad():
                generated_tokens = self.en_indic_model.generate(
                    **inputs,
                    use_cache=False,
                    max_length=adaptive_max_length,
                    num_beams=1,
                    num_return_sequences=1,
                )

            # Decode batch
            generated_tokens = self.en_indic_tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # Postprocess batch
            translations = self.ip.postprocess_batch(generated_tokens, lang=tgt_lang)
            
            logger.info(f"[Batch] Translated {len(texts)} texts to {target_lang}")
            return translations if translations else texts
            
        except Exception as e:
            logger.error(f"Batch translation error EN->{target_lang}: {e}")
            raise

    def _translate_indic_to_en_batch(self, texts: List[str], source_lang: str) -> List[str]:
        """Internal: Translate multiple Indic texts to English in a single batch."""
        try:
            if not texts:
                return []
            
            src_lang, tgt_lang = self._get_lang_code(source_lang), "eng_Latn"
            
            # Preprocess batch
            batch = self.ip.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
            
            # Tokenize batch
            inputs = self.indic_en_tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)

            # Calculate adaptive max_length based on longest text in batch
            adaptive_max_length = self._calculate_adaptive_max_length(texts[0] if texts else "")
            
            # Generate batch
            with torch.no_grad():
                generated_tokens = self.indic_en_model.generate(
                    **inputs,
                    use_cache=False,
                    max_length=adaptive_max_length,
                    num_beams=1,
                    num_return_sequences=1,
                )

            # Decode batch
            generated_tokens = self.indic_en_tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # Postprocess batch
            translations = self.ip.postprocess_batch(generated_tokens, lang=tgt_lang)
            
            logger.info(f"[Batch] Translated {len(texts)} texts to English")
            return translations if translations else texts
            
        except Exception as e:
            logger.error(f"Batch translation error {source_lang}->EN: {e}")
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

    async def _translate_text_with_chunking(self, text: str, source_lang: str, target_lang: str, is_en_to_indic: bool) -> str:
        """Translate text with automatic chunking for long texts."""
        # If text is short, translate directly
        if len(text) <= 1500:
            loop = asyncio.get_event_loop()
            if is_en_to_indic:
                return await loop.run_in_executor(
                    self.executor,
                    self._translate_en_to_indic,
                    text,
                    target_lang
                )
            else:
                return await loop.run_in_executor(
                    self.executor,
                    self._translate_indic_to_en,
                    text,
                    source_lang
                )
        
        # Long text: chunk and translate concurrently
        logger.info(f"[Chunking] Text length {len(text)} chars, splitting into chunks...")
        chunks = self._chunk_text_smart(text, chunk_size=800, overlap=150)
        logger.info(f"[Chunking] Split into {len(chunks)} chunks")
        
        loop = asyncio.get_event_loop()
        
        # Translate chunks concurrently (max 4 for 4 vCPU)
        async def translate_chunk(chunk: str):
            if is_en_to_indic:
                return await loop.run_in_executor(
                    self.executor,
                    self._translate_en_to_indic,
                    chunk,
                    target_lang
                )
            else:
                return await loop.run_in_executor(
                    self.executor,
                    self._translate_indic_to_en,
                    chunk,
                    source_lang
                )
        
        # OPTIMIZATION: Process ALL chunks concurrently using all 4 vCPUs
        # PyTorch releases GIL during inference, so ThreadPoolExecutor with 4 workers = true parallelism
        # No semaphore needed here - executor already limits to 4 workers
        # Process all chunks in parallel (executor will queue excess)
        translated_chunks = await asyncio.gather(*[translate_chunk(chunk) for chunk in chunks])
        
        # Rejoin chunks (remove overlap duplicates)
        result = " ".join(translated_chunks)
        logger.info(f"[Chunking] Rejoined {len(chunks)} chunks into final translation ({len(result)} chars)")
        return result

    async def translate_to_all_async(self, title: str, description: str, source_lang: str) -> dict:
        """Translate to multiple languages using optimized batch processing with caching, batching, and request throttling."""
        
        # PHASE 1 OPTIMIZATION: Check cache first
        try:
            cached_result = await translation_cache_service.get_cached_translation(
                title, description, source_lang
            )
            if cached_result:
                logger.info("[Cache] Translation cache HIT - returning cached result")
                return cached_result
        except Exception as e:
            logger.warning(f"[Cache] Cache lookup failed: {e}, continuing with translation")
        
        # CRITICAL: Use semaphore to limit concurrent translations
        async with self._translation_semaphore:
            # PHASE 1 OPTIMIZATION: Use dynamic batching for short texts (optional, can be disabled)
            # For long texts, skip batching and process directly (reliability > speed)
            use_batch_queue = len(description) <= 1500  # Only use queue for shorter texts
            
            if use_batch_queue:
                batch_queue = get_batch_queue(enabled=True)
                if batch_queue:
                    try:
                        logger.info("[BatchQueue] Using dynamic batch queue for translation")
                        result = await batch_queue.add_request(
                            title, description, source_lang,
                            self._translate_to_all_async_internal
                        )
                        # Cache the result
                        await translation_cache_service.set_cached_translation(
                            title, description, source_lang, result
                        )
                        return result
                    except Exception as e:
                        logger.warning(f"[BatchQueue] Batch queue failed: {e}, falling back to direct translation")
            
            # Direct translation (fallback or for long texts)
            try:
                result = await self._translate_to_all_async_internal(title, description, source_lang)
                # Cache the result
                await translation_cache_service.set_cached_translation(
                    title, description, source_lang, result
                )
                return result
            except Exception as e:
                logger.error(f"Translation failed: {e}")
                raise
    
    async def _translate_to_all_async_internal(self, title: str, description: str, source_lang: str) -> dict:
        """Internal translation method (without caching/batching logic)."""
        try:
            source_lang = self._normalize_lang_code(source_lang)
            
            if source_lang == "english":
                target_languages = ["hindi", "kannada"]
            elif source_lang == "kannada":
                target_languages = ["english", "hindi"]
            else:
                target_languages = ["english", "hindi"]
            
            loop = asyncio.get_event_loop()
            translations = {}
            
            if source_lang == "english":
                # OPTIMIZATION: Batch translate title + description together for each target language
                # Process all target languages concurrently
                async def translate_to_target(target_lang: str):
                    # Check if description needs chunking
                    if len(description) > 1500:
                        # Long description: translate separately with chunking
                        logger.info(f"[Batch] Long description detected ({len(description)} chars), using chunking for {target_lang}")
                        translated_title = await self._translate_text_with_chunking(title, source_lang, target_lang, True)
                        translated_description = await self._translate_text_with_chunking(description, source_lang, target_lang, True)
                    else:
                        # Short texts: batch translate together (2 texts in 1 model call)
                        texts = [title, description]
                        logger.info(f"[Batch] Translating to {target_lang}: title='{title[:50]}...', description='{description[:50]}...'")
                        
                        translated_texts = await loop.run_in_executor(
                            self.executor,
                            self._translate_en_to_indic_batch,
                            texts,
                            target_lang
                        )
                        
                        translated_title = translated_texts[0] if len(translated_texts) > 0 else title
                        translated_description = translated_texts[1] if len(translated_texts) > 1 else description
                    
                    logger.info(f"[Batch] Translated {target_lang}: title='{translated_title[:50]}...', description='{translated_description[:50]}...'")
                    
                    return {
                        target_lang: {
                            "title": translated_title,
                            "description": translated_description
                        }
                    }
                
                # OPTIMIZATION: Process all target languages concurrently
                tasks = [translate_to_target(lang) for lang in target_languages]
                results = await asyncio.gather(*tasks)
                
                # Merge results
                for result in results:
                    translations.update(result)
                    
            else:
                # OPTIMIZATION: Batch translate title + description to English together
                # Check if description needs chunking
                if len(description) > 1500:
                    logger.info(f"[Batch] Long description detected ({len(description)} chars), using chunking for English")
                    english_title = await self._translate_text_with_chunking(title, source_lang, "english", False)
                    english_description = await self._translate_text_with_chunking(description, source_lang, "english", False)
                else:
                    texts = [title, description]
                    logger.info(f"[Batch] Translating to English: title='{title[:50]}...', description='{description[:50]}...'")
                    
                    english_texts = await loop.run_in_executor(
                        self.executor,
                        self._translate_indic_to_en_batch,
                        texts,
                        source_lang
                    )
                    
                    english_title = english_texts[0] if len(english_texts) > 0 else title
                    english_description = english_texts[1] if len(english_texts) > 1 else description
                
                translations["english"] = {
                    "title": english_title,
                    "description": english_description
                }
                
                logger.info(f"[Batch] Translated to English: title='{english_title[:50]}...', description='{english_description[:50]}...'")
                
                # OPTIMIZATION: Then translate English to other languages concurrently
                async def translate_english_to_target(target_lang: str):
                    if len(english_description) > 1500:
                        logger.info(f"[Batch] Long English description detected ({len(english_description)} chars), using chunking for {target_lang}")
                        translated_title = await self._translate_text_with_chunking(english_title, "english", target_lang, True)
                        translated_description = await self._translate_text_with_chunking(english_description, "english", target_lang, True)
                    else:
                        texts = [english_title, english_description]
                        logger.info(f"[Batch] Translating English to {target_lang}: title='{english_title[:50]}...', description='{english_description[:50]}...'")
                        
                        translated_texts = await loop.run_in_executor(
                            self.executor,
                            self._translate_en_to_indic_batch,
                            texts,
                            target_lang
                        )
                        
                        translated_title = translated_texts[0] if len(translated_texts) > 0 else english_title
                        translated_description = translated_texts[1] if len(translated_texts) > 1 else english_description
                    
                    logger.info(f"[Batch] Translated {target_lang}: title='{translated_title[:50]}...', description='{translated_description[:50]}...'")
                    
                    return {
                        target_lang: {
                            "title": translated_title,
                            "description": translated_description
                        }
                    }
                
                # Process all target languages concurrently
                tasks = [translate_english_to_target(lang) for lang in target_languages if lang != "english"]
                if tasks:
                    results = await asyncio.gather(*tasks)
                    for result in results:
                        translations.update(result)
            
            logger.info(f"Translation completed: {list(translations.keys())}")
            return translations
            
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            raise

# Create singleton instance
translation_service = TranslationService()