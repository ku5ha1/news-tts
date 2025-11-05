import asyncio
import logging
from typing import Callable, Any, Optional
import random

logger = logging.getLogger(__name__)

async def retry_with_exponential_backoff(
    func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_multiplier: float = 2.0,
    jitter: bool = True,
    **kwargs
) -> Any:
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if attempt == max_retries:
                logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}. Last error: {e}")
                raise e
            
            # Calculate delay with exponential backoff
            delay = min(base_delay * (backoff_multiplier ** attempt), max_delay)
            
            # Add jitter to prevent thundering herd
            if jitter:
                delay += random.uniform(0, delay * 0.1)
            
            logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s")
            await asyncio.sleep(delay)
    
    # This should never be reached, but just in case
    raise last_exception

async def retry_translation_with_timeout(
    translation_service,
    title: str,
    description: str,
    source_lang: str,
    timeout: float = 90.0,
    max_retries: int = 1
) -> dict:
    """
    Retry translation with adaptive timeout based on text length.
    For long texts, increases timeout to ensure reliability.
    """
    # Calculate adaptive timeout: base + extra time for long texts
    total_chars = len(title) + len(description)
    if total_chars > 2000:
        # Very long text with chunking: need much more time
        # Formula: base + (num_chunks × time_per_batch × 2 languages) + buffer
        # Chunk size is 800 chars, so estimate number of chunks
        num_chunks = max(1, (total_chars - 1500) // 800 + 1)
        # Each batch takes ~35-40s on CPU (realistic measurement from logs)
        # Process in batches of 4 (parallel on 4 vCPUs)
        num_batches = (num_chunks + 3) // 4  # Round up division for batches
        batch_time = num_batches * 40  # 40s per batch (realistic for CPU)
        adaptive_timeout = 60 + (batch_time * 2) + 60  # base + (batches × langs) + larger buffer
        adaptive_timeout = min(adaptive_timeout, 300.0)  # Cap at 5 minutes for sequential processing
        logger.info(f"[Timeout] Adaptive timeout: {adaptive_timeout:.1f}s for {total_chars} chars ({num_chunks} chunks, {num_batches} batches) (base: {timeout}s)")
        timeout = adaptive_timeout
    elif total_chars > 1500:
        # Long text: moderate increase
        timeout = timeout * 1.5
        logger.info(f"[Timeout] Increased timeout to {timeout:.1f}s for long text ({total_chars} chars)")
    
    async def translate_with_timeout():
        try:
            return await asyncio.wait_for(
                translation_service.translate_to_all_async(title, description, source_lang),
                timeout=timeout
            )
        except asyncio.TimeoutError as e:
            logger.error(f"[Timeout] Translation timed out after {timeout:.1f}s for {len(title)+len(description)} chars")
            raise
        except Exception as e:
            logger.error(f"[Translation] Error during translation: {type(e).__name__}: {str(e)}")
            raise
    
    return await retry_with_exponential_backoff(
        translate_with_timeout,
        max_retries=max_retries,
        base_delay=2.0,  # Start with 2 seconds
        max_delay=15.0,  # Max 15 seconds between retries
        backoff_multiplier=1.5,  # Gentle backoff
        jitter=True
    )
