import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from app.services.dynamic_batch_queue import get_batch_queue
from app.services.translation_cache_service import translation_cache_service
from app.services.translation_worker import (
    initialize_worker,
    translate_text_worker,
    translate_title_description_worker,
    warmup_worker,
)


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslationService:
    """Coordinates translation requests via a process-based worker pool."""

    _instance: Optional["TranslationService"] = None

    def __new__(cls) -> "TranslationService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialised", False):  # pragma: no cover - defensive guard
            return

        self._initialised = True
        self.max_workers = min(4, os.cpu_count() or 4)
        self._process_executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=initialize_worker,
        )
        self._translation_semaphore = asyncio.Semaphore(2)
        logger.info(
            "[Translation] ProcessPoolExecutor initialised (workers=%s)", self.max_workers
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _run_sync(self, func, *args):
        future = self._process_executor.submit(func, *args)
        return future.result()

    async def _run_async(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._process_executor, func, *args)

    async def _get_cached_translation(
        self, title: str, description: str, source_lang: str
    ) -> Optional[Dict[str, Dict[str, str]]]:
        try:
            cached = await translation_cache_service.get_cached_translation(
                title, description, source_lang
            )
            if cached:
                logger.info("[Cache] HIT for translation request")
            return cached
        except Exception as exc:  # pragma: no cover - cache infra failure
            logger.warning("[Cache] lookup failed: %s", exc)
            return None

    async def _store_cache(
        self, title: str, description: str, source_lang: str, translations: Dict[str, Any]
    ) -> None:
        try:
            await translation_cache_service.set_cached_translation(
                title, description, source_lang, translations
            )
        except Exception as exc:  # pragma: no cover - cache infra failure
            logger.warning("[Cache] store failed: %s", exc)

    async def _translate_to_all_async_internal(
        self, title: str, description: str, source_lang: str
    ) -> Dict[str, Dict[str, str]]:
        return await self._run_async(
            translate_title_description_worker, title, description, source_lang
        )

    async def _translate_via_batch_queue(
        self, title: str, description: str, source_lang: str
    ) -> Dict[str, Dict[str, str]]:
        batch_queue = get_batch_queue(enabled=True)
        if not batch_queue:
            return await self._translate_to_all_async_internal(
                title, description, source_lang
            )

        return await batch_queue.add_request(
            title,
            description,
            source_lang,
            self._translate_to_all_async_internal,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            return self._run_sync(
                translate_text_worker,
                text,
                source_lang,
                target_lang,
            )
        except Exception as exc:
            logger.error(
                "[Translation] translate failed %s->%s: %s",
                source_lang,
                target_lang,
                exc,
                exc_info=True,
            )
            raise

    async def translate_to_all_async(
        self, title: str, description: str, source_lang: str
    ) -> Dict[str, Dict[str, str]]:
        cached = await self._get_cached_translation(title, description, source_lang)
        if cached:
            return cached

        async with self._translation_semaphore:
            use_batch_queue = len(description) <= 1500
            result: Optional[Dict[str, Dict[str, str]]] = None

            if use_batch_queue:
                try:
                    logger.info("[BatchQueue] Attempting queued translation")
                    result = await self._translate_via_batch_queue(
                        title, description, source_lang
                    )
                except Exception as exc:
                    logger.warning("[BatchQueue] Failed (%s), falling back", exc)
                    result = None

            if result is None:
                result = await self._translate_to_all_async_internal(
                    title, description, source_lang
                )

            await self._store_cache(title, description, source_lang, result)
            return result

    async def warmup(self) -> bool:
        try:
            return await self._run_async(warmup_worker)
        except Exception as exc:  # pragma: no cover
            logger.error("[Translation] Warmup failed: %s", exc)
            return False


translation_service = TranslationService()