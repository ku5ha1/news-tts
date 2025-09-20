import logging
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from app.config.settings import Settings

logger = logging.getLogger(__name__)

class DBService:
    def __init__(self):
        self.settings = Settings()
        
        try:
            if not self.settings.DATABASE_NAME:
                logger.error("DATABASE_NAME is missing - cannot connect to database")
                self.connected = False
                self.client = None
                self.db = None
                self.collection = None
                return
            else:
                database_name = self.settings.DATABASE_NAME
                
            logger.info(f"[MongoDB] Connecting to database: {database_name}")
            self.client = AsyncIOMotorClient(self.settings.MONGO_URI)
            self.db = self.client[database_name]
            self.collection = self.db["news"]
            self.connected = True
            logger.info("[MongoDB] Initialized successfully")
            
        except Exception as e:
            logger.error(f"[MongoDB] Initialization error: {str(e)}", exc_info=True)
            self.connected = False
            self.client = None

    async def insert_news(self, data: dict):
        """Insert news document into MongoDB"""
        if not self.connected or not self.client:
            raise RuntimeError("Database not connected")
            
        try:
            logger.info(f"[MongoDB] Insert.start id={data.get('_id')}")
            result = await self.collection.insert_one(data)
            logger.info(f"[MongoDB] Insert.done id={result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"[MongoDB] Insert.failed error={str(e)}", exc_info=True)
            self.connected = False
            raise RuntimeError(f"Database insert failed: {str(e)}")

    async def update_news_fields(self, news_id: str | ObjectId, updates: dict) -> bool:
        """Update specific fields on a news document."""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot update - not connected")
            return False
            
        try:
            oid = ObjectId(news_id) if not isinstance(news_id, ObjectId) else news_id
            logger.info(f"[MongoDB] Update.start id={oid} fields={len(updates)}")
            result = await self.collection.update_one({"_id": oid}, {"$set": updates})
            
            if result.modified_count > 0:
                logger.info(f"[MongoDB] Update.done id={oid} modified={result.modified_count}")
                return True
            else:
                logger.warning(f"[MongoDB] Update.none id={oid}")
                return False
                
        except Exception as e:
            logger.error(f"[MongoDB] Update.failed id={news_id} error={str(e)}", exc_info=True)
            return False

    async def is_connected(self) -> bool:
        """Check if MongoDB is connected"""
        if not self.client:
            return False
            
        try:
            # Ping the database
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
