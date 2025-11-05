import hashlib
import json
import logging
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from app.config.settings import settings

logger = logging.getLogger(__name__)

class TranslationCacheService:
    """Service for caching translation results to improve performance."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TranslationCacheService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self.client = None
            self.db = None
            self.collection = None
            self.cache_enabled = True
            self.cache_ttl_hours = 24  # Cache for 24 hours
            self._init_cache()
    
    def _init_cache(self):
        """Initialize MongoDB connection for caching."""
        try:
            if not settings.DATABASE_NAME:
                logger.warning("[Cache] DATABASE_NAME not set, caching disabled")
                self.cache_enabled = False
                return
            
            database_name = settings.DATABASE_NAME
            self.client = AsyncIOMotorClient(settings.MONGO_URI)
            self.db = self.client[database_name]
            self.collection = self.db["translation_cache"]
            
            # Create TTL index for automatic expiration (run in background)
            # Note: We can't use asyncio.create_task here in __init__, so we'll create index on first use
            self._ttl_index_created = False
            
            logger.info("[Cache] Translation cache service initialized")
        except Exception as e:
            logger.warning(f"[Cache] Failed to initialize cache: {e}, caching disabled")
            self.cache_enabled = False
    
    async def _ensure_ttl_index(self):
        """Ensure TTL index exists (lazy creation on first use)."""
        if self._ttl_index_created or self.collection is None:
            return
        
        try:
            await self.collection.create_index(
                "expires_at",
                expireAfterSeconds=0  # Delete documents when expires_at is reached
            )
            self._ttl_index_created = True
            logger.info("[Cache] TTL index created successfully")
        except Exception as e:
            # Index might already exist, which is fine
            if "already exists" not in str(e).lower():
                logger.warning(f"[Cache] TTL index creation failed: {e}")
            self._ttl_index_created = True
    
    def _generate_cache_key(self, title: str, description: str, source_lang: str) -> str:
        """Generate cache key from translation inputs."""
        # Create a hash of the inputs
        cache_input = {
            "title": title.strip().lower(),
            "description": description.strip().lower(),
            "source_lang": source_lang.lower()
        }
        cache_string = json.dumps(cache_input, sort_keys=True)
        cache_hash = hashlib.sha256(cache_string.encode('utf-8')).hexdigest()
        return f"trans:{cache_hash}"
    
    async def get_cached_translation(
        self, 
        title: str, 
        description: str, 
        source_lang: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached translation if available."""
        if not self.cache_enabled or self.collection is None:
            return None
        
        try:
            # Ensure TTL index exists
            await self._ensure_ttl_index()
            
            cache_key = self._generate_cache_key(title, description, source_lang)
            cached_doc = await self.collection.find_one({"cache_key": cache_key})
            
            if cached_doc:
                # Check if cache is still valid
                expires_at = cached_doc.get("expires_at")
                if expires_at and datetime.utcnow() < expires_at:
                    logger.info(f"[Cache] Cache HIT for key: {cache_key[:16]}...")
                    return cached_doc.get("translations")
                else:
                    # Cache expired, delete it
                    await self.collection.delete_one({"cache_key": cache_key})
                    logger.info(f"[Cache] Cache EXPIRED for key: {cache_key[:16]}...")
                    return None
            
            logger.debug(f"[Cache] Cache MISS for key: {cache_key[:16]}...")
            return None
            
        except Exception as e:
            logger.warning(f"[Cache] Error retrieving cache: {e}")
            return None
    
    async def set_cached_translation(
        self,
        title: str,
        description: str,
        source_lang: str,
        translations: Dict[str, Any]
    ) -> bool:
        """Cache translation results."""
        if not self.cache_enabled or self.collection is None:
            return False
        
        try:
            # Ensure TTL index exists
            await self._ensure_ttl_index()
            
            cache_key = self._generate_cache_key(title, description, source_lang)
            expires_at = datetime.utcnow() + timedelta(hours=self.cache_ttl_hours)
            
            cache_doc = {
                "cache_key": cache_key,
                "title": title[:100],  # Store first 100 chars for reference
                "source_lang": source_lang,
                "translations": translations,
                "created_at": datetime.utcnow(),
                "expires_at": expires_at
            }
            
            # Upsert: update if exists, insert if not
            await self.collection.update_one(
                {"cache_key": cache_key},
                {"$set": cache_doc},
                upsert=True
            )
            
            logger.info(f"[Cache] Cached translation for key: {cache_key[:16]}...")
            return True
            
        except Exception as e:
            logger.warning(f"[Cache] Error caching translation: {e}")
            return False
    
    async def clear_cache(self, older_than_hours: Optional[int] = None) -> int:
        """Clear cache entries. If older_than_hours is None, clear all."""
        if not self.cache_enabled or self.collection is None:
            return 0
        
        try:
            if older_than_hours:
                cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
                result = await self.collection.delete_many({"expires_at": {"$lt": cutoff_time}})
            else:
                result = await self.collection.delete_many({})
            
            logger.info(f"[Cache] Cleared {result.deleted_count} cache entries")
            return result.deleted_count
        except Exception as e:
            logger.error(f"[Cache] Error clearing cache: {e}")
            return 0

# Create singleton instance
translation_cache_service = TranslationCacheService()

