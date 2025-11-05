import asyncio
import logging
import time
from typing import List, Dict, Any, Callable, Awaitable, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    """Represents a translation request in the batch queue."""
    title: str
    description: str
    source_lang: str
    future: asyncio.Future
    timestamp: float

class DynamicBatchQueue:
    """
    Dynamic batching queue that collects requests and processes them in batches.
    Optimizes throughput by batching multiple requests together.
    """
    
    def __init__(
        self,
        batch_size: int = 4,
        max_wait_ms: int = 50,
        max_batch_chars: int = 3000
    ):
        """
        Initialize dynamic batch queue.
        
        Args:
            batch_size: Maximum number of requests per batch
            max_wait_ms: Maximum time to wait before processing batch (milliseconds)
            max_batch_chars: Maximum characters per batch
        """
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.max_batch_chars = max_batch_chars
        self.queue: List[BatchRequest] = []
        self.queue_lock = asyncio.Lock()
        self.processing = False
        self.batch_processor_task: Optional[asyncio.Task] = None
        
    async def add_request(
        self,
        title: str,
        description: str,
        source_lang: str,
        translate_func: Callable[[str, str, str], Awaitable[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Add a translation request to the batch queue.
        
        Returns:
            Translation results dictionary
        """
        future = asyncio.Future()
        request = BatchRequest(
            title=title,
            description=description,
            source_lang=source_lang,
            future=future,
            timestamp=time.time()
        )
        
        async with self.queue_lock:
            self.queue.append(request)
            
            # Start batch processor if not running
            if not self.processing:
                self.processing = True
                self.batch_processor_task = asyncio.create_task(
                    self._process_batches(translate_func)
                )
            
            # Trigger immediate processing if batch is full
            if len(self.queue) >= self.batch_size:
                # Wake up the batch processor
                pass
        
        # Wait for result
        try:
            result = await future
            return result
        except Exception as e:
            logger.error(f"[BatchQueue] Request failed: {e}")
            raise
    
    async def _process_batches(
        self,
        translate_func: Callable[[str, str, str], Awaitable[Dict[str, Any]]]
    ):
        """Process batches from the queue."""
        while True:
            try:
                # Wait for batch to fill or timeout
                await asyncio.sleep(self.max_wait_ms / 1000.0)
                
                async with self.queue_lock:
                    if not self.queue:
                        # Queue is empty, stop processing
                        self.processing = False
                        break
                    
                    # Get batch of requests
                    batch = self._get_batch()
                    if not batch:
                        continue
                
                # Process batch concurrently (not as a single batch to model)
                # This allows better parallelism while still benefiting from queueing
                logger.info(f"[BatchQueue] Processing batch of {len(batch)} requests")
                
                # Process requests concurrently (up to batch_size)
                tasks = []
                for req in batch:
                    task = asyncio.create_task(
                        self._process_single_request(req, translate_func)
                    )
                    tasks.append(task)
                
                # Wait for all requests in batch to complete
                await asyncio.gather(*tasks, return_exceptions=True)
                
            except Exception as e:
                logger.error(f"[BatchQueue] Error processing batch: {e}")
                # Continue processing
                await asyncio.sleep(0.1)
        
        self.processing = False
    
    def _get_batch(self) -> List[BatchRequest]:
        """Extract a batch of requests from the queue."""
        if not self.queue:
            return []
        
        batch = []
        total_chars = 0
        oldest_time = self.queue[0].timestamp if self.queue else time.time()
        
        # Check if oldest request has waited long enough
        wait_time = (time.time() - oldest_time) * 1000  # milliseconds
        
        # Get batch if:
        # 1. Queue is full (batch_size reached), OR
        # 2. Oldest request has waited max_wait_ms, OR
        # 3. Batch chars limit would be exceeded
        should_process = (
            len(self.queue) >= self.batch_size or
            wait_time >= self.max_wait_ms
        )
        
        if should_process:
            # Take up to batch_size requests
            while self.queue and len(batch) < self.batch_size:
                req = self.queue[0]
                req_chars = len(req.title) + len(req.description)
                
                # Check if adding this request would exceed char limit
                if total_chars + req_chars > self.max_batch_chars and batch:
                    break
                
                batch.append(self.queue.pop(0))
                total_chars += req_chars
        
        return batch
    
    async def _process_single_request(
        self,
        request: BatchRequest,
        translate_func: Callable[[str, str, str], Awaitable[Dict[str, Any]]]
    ):
        """Process a single request from the batch."""
        try:
            result = await translate_func(
                request.title,
                request.description,
                request.source_lang
            )
            request.future.set_result(result)
        except Exception as e:
            logger.error(f"[BatchQueue] Request processing failed: {e}")
            request.future.set_exception(e)

# Global batch queue instance (optional, can be disabled)
_dynamic_batch_queue: Optional[DynamicBatchQueue] = None

def get_batch_queue(enabled: bool = True) -> Optional[DynamicBatchQueue]:
    """Get or create the global batch queue."""
    global _dynamic_batch_queue
    if enabled and _dynamic_batch_queue is None:
        _dynamic_batch_queue = DynamicBatchQueue(
            batch_size=4,
            max_wait_ms=50,
            max_batch_chars=3000
        )
    return _dynamic_batch_queue if enabled else None

