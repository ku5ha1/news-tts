import importlib
import logging
import os
from typing import Any, Dict, List

import torch
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def _load_indic_processor() -> Any:
    module = importlib.import_module("IndicTransToolkit.processor")  # type: ignore[import]
    return module.IndicProcessor


load_dotenv()


logger = logging.getLogger(__name__)


MODEL_NAMES = {
    "en_indic": "ai4bharat/indictrans2-en-indic-dist-200M",
    "indic_en": "ai4bharat/indictrans2-indic-en-dist-200M",
}


class TranslationWorker:
    """CPU-friendly translation runtime that lives inside worker processes."""

    def __init__(self) -> None:
        logging.basicConfig(level=logging.INFO)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[TranslationWorker] Using device: {self.device}")

        try:
            logger.info("[TranslationWorker] Initialising IndicProcessor ...")
            IndicProcessor = _load_indic_processor()
            self.ip = IndicProcessor(inference=True)
        except ImportError as exc:  # pragma: no cover - infrastructure failure
            logger.error("IndicProcessor import failed", exc_info=True)
            raise RuntimeError(f"IndicTransToolkit import failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover
            logger.error("IndicProcessor initialisation failed", exc_info=True)
            raise RuntimeError(f"IndicTransToolkit initialisation failed: {exc}") from exc

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        self._load_models()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_models(self) -> None:
        token = os.getenv("HUGGINGFACE_ACCESSTOKEN")

        try:
            logger.info("[TranslationWorker] Loading EN->Indic model ...")
            self.en_indic_tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAMES["en_indic"],
                trust_remote_code=True,
                token=token,
            )
            self.en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAMES["en_indic"],
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                token=token,
            ).to(self.device)
            self.en_indic_model.eval()

            if self.device.type == "cpu":
                try:
                    logger.info("[TranslationWorker] Quantising EN->Indic model (int8) ...")
                    self.en_indic_model = torch.quantization.quantize_dynamic(
                        self.en_indic_model,
                        {torch.nn.Linear, torch.nn.LSTM},
                        dtype=torch.qint8,
                    )
                except Exception as exc:  # pragma: no cover - int8 optional
                    logger.warning("EN->Indic quantisation failed: %s", exc)

            logger.info("[TranslationWorker] EN->Indic model ready")
        except Exception as exc:  # pragma: no cover - infrastructure failure
            logger.error("Failed to load EN->Indic model", exc_info=True)
            raise RuntimeError(f"EN->Indic model loading failed: {exc}") from exc

        try:
            logger.info("[TranslationWorker] Loading Indic->EN model ...")
            self.indic_en_tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAMES["indic_en"],
                trust_remote_code=True,
                token=token,
            )
            self.indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAMES["indic_en"],
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                token=token,
            ).to(self.device)
            self.indic_en_model.eval()

            if self.device.type == "cpu":
                try:
                    logger.info("[TranslationWorker] Quantising Indic->EN model (int8) ...")
                    self.indic_en_model = torch.quantization.quantize_dynamic(
                        self.indic_en_model,
                        {torch.nn.Linear, torch.nn.LSTM},
                        dtype=torch.qint8,
                    )
                except Exception as exc:  # pragma: no cover
                    logger.warning("Indic->EN quantisation failed: %s", exc)

            logger.info("[TranslationWorker] Indic->EN model ready")
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to load Indic->EN model", exc_info=True)
            raise RuntimeError(f"Indic->EN model loading failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_lang_code(lang: str) -> str:
        lang = lang.lower().strip()
        if lang in ("en", "english"):
            return "english"
        if lang in ("hi", "hindi"):
            return "hindi"
        if lang in ("kn", "kannada"):
            return "kannada"
        return "english"

    @staticmethod
    def _get_lang_code(lang: str) -> str:
        lang_map = {
            "hindi": "hin_Deva",
            "kannada": "kan_Knda",
            "english": "eng_Latn",
        }
        return lang_map.get(lang.lower(), "hin_Deva")

    @staticmethod
    def _calculate_adaptive_max_length(text: str) -> int:
        length = len(text)
        if length < 100:
            return 256
        if length < 500:
            return 512
        if length < 1500:
            return 1024
        return 1024

    @staticmethod
    def _chunk_text_smart(text: str, chunk_size: int = 800, overlap: int = 80) -> List[str]:
        if len(text) <= chunk_size:
            return [text]

        sentences = text.replace("!", ".\n").replace("?", ".\n").split(".")
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            size = len(sentence)
            if current and current_len + size > chunk_size:
                chunk_text = ". ".join(current) + "."
                chunks.append(chunk_text)

                overlap_sentences = current[-2:] if len(current) >= 2 else current
                current = overlap_sentences + [sentence]
                current_len = sum(len(s) for s in current)
            else:
                current.append(sentence)
                current_len += size

        if current:
            chunk_text = ". ".join(current) + "."
            chunks.append(chunk_text)

        return chunks if chunks else [text]

    # ------------------------------------------------------------------
    # Core translation primitives
    # ------------------------------------------------------------------
    def _translate_en_to_indic(self, text: str, target_lang: str) -> str:
        tgt_lang = self._get_lang_code(target_lang)
        src_lang = "eng_Latn"

        batch = self.ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = self.en_indic_tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        adaptive_max_length = self._calculate_adaptive_max_length(text)

        with torch.no_grad():
            generated_tokens = self.en_indic_model.generate(
                **inputs,
                use_cache=False,
                max_length=adaptive_max_length,
                num_beams=1,
                num_return_sequences=1,
            )

        decoded = self.en_indic_tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        translations = self.ip.postprocess_batch(decoded, lang=tgt_lang)
        return translations[0] if translations else text

    def _translate_indic_to_en(self, text: str, source_lang: str) -> str:
        src_lang = self._get_lang_code(source_lang)
        tgt_lang = "eng_Latn"

        batch = self.ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = self.indic_en_tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        adaptive_max_length = self._calculate_adaptive_max_length(text)

        with torch.no_grad():
            generated_tokens = self.indic_en_model.generate(
                **inputs,
                use_cache=False,
                max_length=adaptive_max_length,
                num_beams=1,
                num_return_sequences=1,
            )

        decoded = self.indic_en_tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        translations = self.ip.postprocess_batch(decoded, lang=tgt_lang)
        return translations[0] if translations else text

    def _translate_en_to_indic_batch(self, texts: List[str], target_lang: str) -> List[str]:
        if not texts:
            return []

        tgt_lang = self._get_lang_code(target_lang)
        batch = self.ip.preprocess_batch(texts, src_lang="eng_Latn", tgt_lang=tgt_lang)
        inputs = self.en_indic_tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        max_length = self._calculate_adaptive_max_length(max(texts, key=len))

        with torch.no_grad():
            generated_tokens = self.en_indic_model.generate(
                **inputs,
                use_cache=False,
                max_length=max_length,
                num_beams=1,
                num_return_sequences=1,
            )

        decoded = self.en_indic_tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        translations = self.ip.postprocess_batch(decoded, lang=tgt_lang)
        return translations if translations else texts

    def _translate_indic_to_en_batch(self, texts: List[str], source_lang: str) -> List[str]:
        if not texts:
            return []

        src_lang = self._get_lang_code(source_lang)
        batch = self.ip.preprocess_batch(texts, src_lang=src_lang, tgt_lang="eng_Latn")
        inputs = self.indic_en_tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        max_length = self._calculate_adaptive_max_length(max(texts, key=len))

        with torch.no_grad():
            generated_tokens = self.indic_en_model.generate(
                **inputs,
                use_cache=False,
                max_length=max_length,
                num_beams=1,
                num_return_sequences=1,
            )

        decoded = self.indic_en_tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        translations = self.ip.postprocess_batch(decoded, lang="eng_Latn")
        return translations if translations else texts

    # ------------------------------------------------------------------
    # Higher-level helpers
    # ------------------------------------------------------------------
    def _translate_text_with_chunking(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        is_en_to_indic: bool,
    ) -> str:
        if len(text) <= 1500:
            return (
                self._translate_en_to_indic(text, target_lang)
                if is_en_to_indic
                else self._translate_indic_to_en(text, source_lang)
            )

        chunks = self._chunk_text_smart(text, chunk_size=800, overlap=80)
        translations: List[str] = []
        for chunk in chunks:
            translated = (
                self._translate_en_to_indic(chunk, target_lang)
                if is_en_to_indic
                else self._translate_indic_to_en(chunk, source_lang)
            )
            translations.append(translated)

        return " ".join(translations)

    # ------------------------------------------------------------------
    # Public APIs used by ProcessPool workers
    # ------------------------------------------------------------------
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        source = self._normalize_lang_code(source_lang)
        target = self._normalize_lang_code(target_lang)

        if source == target:
            return text

        if source == "english":
            return self._translate_en_to_indic(text, target)
        if target == "english":
            return self._translate_indic_to_en(text, source)

        english = self._translate_indic_to_en(text, source)
        return self._translate_en_to_indic(english, target)

    def translate_to_all(self, title: str, description: str, source_lang: str) -> Dict[str, Dict[str, str]]:
        source_lang = self._normalize_lang_code(source_lang)
        translations: Dict[str, Dict[str, str]] = {}

        if source_lang == "english":
            target_languages = ["hindi", "kannada"]
            for target in target_languages:
                if len(description) > 1500:
                    translated_title = self._translate_en_to_indic(title, target)
                    translated_description = self._translate_text_with_chunking(
                        description, source_lang, target, True
                    )
                else:
                    translated_texts = self._translate_en_to_indic_batch([title, description], target)
                    translated_title = translated_texts[0] if len(translated_texts) > 0 else title
                    translated_description = (
                        translated_texts[1] if len(translated_texts) > 1 else description
                    )

                translations[target] = {
                    "title": translated_title,
                    "description": translated_description,
                }

        else:
            if len(description) > 1500:
                english_title = self._translate_indic_to_en(title, source_lang)
                english_description = self._translate_text_with_chunking(
                    description, source_lang, "english", False
                )
            else:
                english_texts = self._translate_indic_to_en_batch([title, description], source_lang)
                english_title = english_texts[0] if len(english_texts) > 0 else title
                english_description = english_texts[1] if len(english_texts) > 1 else description

            translations["english"] = {
                "title": english_title,
                "description": english_description,
            }

            downstream_targets = ["hindi", "kannada"]
            for target in downstream_targets:
                if target == source_lang:
                    translations[target] = {
                        "title": title,
                        "description": description,
                    }
                    continue

                if len(english_description) > 1500:
                    translated_title = self._translate_en_to_indic(english_title, target)
                    translated_description = self._translate_text_with_chunking(
                        english_description, "english", target, True
                    )
                else:
                    translated_texts = self._translate_en_to_indic_batch(
                        [english_title, english_description], target
                    )
                    translated_title = translated_texts[0] if len(translated_texts) > 0 else english_title
                    translated_description = (
                        translated_texts[1] if len(translated_texts) > 1 else english_description
                    )

                translations[target] = {
                    "title": translated_title,
                    "description": translated_description,
                }

        return translations


# ----------------------------------------------------------------------
# ProcessPool helpers
# ----------------------------------------------------------------------
_worker_instance: TranslationWorker | None = None


def _get_worker() -> TranslationWorker:
    global _worker_instance
    if _worker_instance is None:
        _worker_instance = TranslationWorker()
    return _worker_instance


def initialize_worker() -> None:
    """Initialiser hook for ProcessPoolExecutor."""
    _get_worker()


def translate_text_worker(text: str, source_lang: str, target_lang: str) -> str:
    worker = _get_worker()
    return worker.translate(text, source_lang, target_lang)


def translate_title_description_worker(
    title: str, description: str, source_lang: str
) -> Dict[str, Dict[str, str]]:
    worker = _get_worker()
    return worker.translate_to_all(title, description, source_lang)


def warmup_worker() -> bool:
    """Utility used in health checks to ensure models are loaded."""
    _get_worker()
    return True


