import logging
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from app.config.settings import settings
import asyncio

logger = logging.getLogger(__name__)

# Configuration constants
MAX_RETRIES = 2

class DBService:
    def __init__(self):
        self.connected = False
        self.client = None
        self.db = None
        self.collection = None

        try:
            if not settings.DATABASE_NAME:
                logger.error("DATABASE_NAME is missing - cannot connect to database")
                return
            else:
                database_name = settings.DATABASE_NAME

            logger.info(f"[MongoDB] Connecting to database: {database_name}")
            self.client = AsyncIOMotorClient(settings.MONGO_URI)
            self.db = self.client[database_name]
            self.collection = self.db["news"]
            self.connected = True
            logger.info("[MongoDB] Initialized successfully")
        except ImportError as e:
            logger.error(f"[MongoDB] Import error: {str(e)}", exc_info=True)
        except ConnectionError as e:
            logger.error(f"[MongoDB] Connection error: {str(e)}", exc_info=True)
        except Exception as e:
            logger.error(f"[MongoDB] Initialization error: {str(e)}", exc_info=True)

    async def insert_news(self, data: dict):
        """Insert news document into MongoDB"""
        if not self.connected or not self.client:
            raise RuntimeError("Database not connected")

        try:
            logger.info(f"[MongoDB] Insert.start id={data.get('_id')}")
            result = await self.collection.insert_one(data)
            logger.info(f"[MongoDB] Insert.done id={result.inserted_id}")
            return result.inserted_id
        except ConnectionError as e:
            logger.error(f"[MongoDB] Insert connection error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Database connection failed: {str(e)}")
        except Exception as e:
            logger.error(f"[MongoDB] Insert.failed error={str(e)}", exc_info=True)
            raise RuntimeError(f"Database insert failed: {str(e)}")

    async def update_news_fields(self, news_id: str | ObjectId, updates: dict, retries: int = MAX_RETRIES) -> bool:
        """Update specific fields on a news document with optional retries."""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot update - not connected")
            return False

        oid = ObjectId(news_id) if not isinstance(news_id, ObjectId) else news_id

        for attempt in range(1, retries + 1):
            try:
                logger.info(f"[MongoDB] Update.start attempt={attempt} id={oid} fields={len(updates)}")
                result = await self.collection.update_one({"_id": oid}, {"$set": updates})
                if result.modified_count > 0:
                    logger.info(f"[MongoDB] Update.done id={oid} modified={result.modified_count}")
                    return True
                else:
                    logger.warning(f"[MongoDB] Update.none id={oid}")
                    return False
            except Exception as e:
                logger.error(f"[MongoDB] Update.failed attempt={attempt} id={news_id} error={str(e)}", exc_info=True)
                if attempt < retries:
                    await asyncio.sleep(1)  # small delay before retry
                    continue
                return False

    async def is_connected(self) -> bool:
        """Check if MongoDB is connected"""
        if not self.client:
            return False
        try:
            await self.client.admin.command('ping')
            self.connected = True
            logger.info("[MongoDB] Connection verified")
            return True
        except Exception as e:
            logger.error(f"[MongoDB] Connection check failed: {str(e)}")
            self.connected = False
            return False

    async def get_news_by_id(self, news_id: str):
        """Get news by ID"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get document - not connected")
            return None
        try:
            oid = ObjectId(news_id) if not isinstance(news_id, ObjectId) else news_id
            logger.info(f"[MongoDB] Fetching document: {oid}")
            result = await self.collection.find_one({"_id": oid})
            if result:
                logger.info(f"[MongoDB] Document found: {oid}")
            else:
                logger.warning(f"[MongoDB] Document not found: {oid}")
            return result
        except Exception as e:
            logger.error(f"[MongoDB] Get error for {news_id}: {str(e)}", exc_info=True)
            return None
