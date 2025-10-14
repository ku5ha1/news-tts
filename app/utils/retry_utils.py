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
    max_retries: int = 3
) -> dict:
    async def translate_with_timeout():
        return await asyncio.wait_for(
            translation_service.translate_to_all_async(title, description, source_lang),
            timeout=timeout
        )
    
    return await retry_with_exponential_backoff(
        translate_with_timeout,
        max_retries=max_retries,
        base_delay=2.0,  # Start with 2 seconds
        max_delay=15.0,  # Max 15 seconds between retries
        backoff_multiplier=1.5,  # Gentle backoff
        jitter=True
    )
