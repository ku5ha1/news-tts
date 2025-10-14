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

    async def get_user_by_id(self, user_id: ObjectId, collection_name: str = "users"):
        """Get user by ID from specified collection"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get user - not connected")
            return None
        try:
            logger.info(f"[MongoDB] Fetching user: {user_id} from {collection_name}")
            collection = self.db[collection_name]
            result = await collection.find_one({"_id": user_id})
            if result:
                logger.info(f"[MongoDB] User found: {user_id}")
            else:
                logger.warning(f"[MongoDB] User not found: {user_id}")
            return result
        except Exception as e:
            logger.error(f"[MongoDB] Error getting user {user_id}: {str(e)}")
            return None

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

    # Category methods
    async def insert_category(self, data: dict):
        """Insert category document into MongoDB"""
        if not self.connected or not self.client:
            raise RuntimeError("Database not connected")
        try:
            logger.info(f"[MongoDB] Insert category start id={data.get('_id')}")
            collection = self.db["categories"]
            result = await collection.insert_one(data)
            logger.info(f"[MongoDB] Insert category done id={result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"[MongoDB] Insert category failed error={str(e)}", exc_info=True)
            raise RuntimeError(f"Database insert failed: {str(e)}")

    async def get_category_by_id(self, category_id: str | ObjectId):
        """Get category by ID"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get category - not connected")
            return None
        try:
            oid = ObjectId(category_id) if not isinstance(category_id, ObjectId) else category_id
            logger.info(f"[MongoDB] Fetching category: {oid}")
            collection = self.db["categories"]
            result = await collection.find_one({"_id": oid})
            if result:
                logger.info(f"[MongoDB] Category found: {oid}")
            else:
                logger.warning(f"[MongoDB] Category not found: {oid}")
            return result
        except Exception as e:
            logger.error(f"[MongoDB] Get category error for {category_id}: {str(e)}", exc_info=True)
            return None

    async def get_categories_paginated(self, skip: int = 0, limit: int = 20, status_filter: str = None):
        """Get categories with pagination and optional status filter"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get categories - not connected")
            return [], 0
        try:
            collection = self.db["categories"]
            query = {}
            if status_filter:
                query["status"] = status_filter
            
            # Get total count
            total = await collection.count_documents(query)
            
            # Get paginated results
            cursor = collection.find(query).skip(skip).limit(limit)
            categories = await cursor.to_list(length=limit)
            
            logger.info(f"[MongoDB] Found {len(categories)} categories (total: {total})")
            return categories, total
        except Exception as e:
            logger.error(f"[MongoDB] Get categories error: {str(e)}", exc_info=True)
            return [], 0

    async def update_category_fields(self, category_id: str | ObjectId, updates: dict, retries: int = MAX_RETRIES) -> bool:
        """Update specific fields on a category document with optional retries."""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot update category - not connected")
            return False

        oid = ObjectId(category_id) if not isinstance(category_id, ObjectId) else category_id

        for attempt in range(1, retries + 1):
            try:
                logger.info(f"[MongoDB] Update category start attempt={attempt} id={oid} fields={len(updates)}")
                collection = self.db["categories"]
                result = await collection.update_one({"_id": oid}, {"$set": updates})
                if result.modified_count > 0:
                    logger.info(f"[MongoDB] Update category done id={oid} modified={result.modified_count}")
                    return True
                else:
                    logger.warning(f"[MongoDB] Update category none id={oid}")
                    return False
            except Exception as e:
                logger.error(f"[MongoDB] Update category failed attempt={attempt} id={category_id} error={str(e)}", exc_info=True)
                if attempt < retries:
                    await asyncio.sleep(1)
                    continue
                return False

    async def delete_category(self, category_id: str | ObjectId) -> bool:
        """Delete a category"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot delete category - not connected")
            return False
        try:
            oid = ObjectId(category_id) if not isinstance(category_id, ObjectId) else category_id
            logger.info(f"[MongoDB] Deleting category: {oid}")
            collection = self.db["categories"]
            result = await collection.delete_one({"_id": oid})
            if result.deleted_count > 0:
                logger.info(f"[MongoDB] Category deleted: {oid}")
                return True
            else:
                logger.warning(f"[MongoDB] Category not found for deletion: {oid}")
                return False
        except Exception as e:
            logger.error(f"[MongoDB] Delete category error for {category_id}: {str(e)}", exc_info=True)
            return False

    # Long Video methods
    async def insert_longvideo(self, data: dict):
        """Insert long video document into MongoDB"""
        if not self.connected or not self.client:
            raise RuntimeError("Database not connected")
        try:
            logger.info(f"[MongoDB] Insert longvideo start id={data.get('_id')}")
            collection = self.db["longvideos"]
            result = await collection.insert_one(data)
            logger.info(f"[MongoDB] Insert longvideo done id={result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"[MongoDB] Insert longvideo failed error={str(e)}", exc_info=True)
            raise RuntimeError(f"Database insert failed: {str(e)}")

    async def get_longvideo_by_id(self, video_id: str | ObjectId):
        """Get long video by ID"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get longvideo - not connected")
            return None
        try:
            oid = ObjectId(video_id) if not isinstance(video_id, ObjectId) else video_id
            logger.info(f"[MongoDB] Fetching longvideo: {oid}")
            collection = self.db["longvideos"]
            result = await collection.find_one({"_id": oid})
            if result:
                logger.info(f"[MongoDB] Longvideo found: {oid}")
            else:
                logger.warning(f"[MongoDB] Longvideo not found: {oid}")
            return result
        except Exception as e:
            logger.error(f"[MongoDB] Get longvideo error for {video_id}: {str(e)}", exc_info=True)
            return None

    async def get_longvideos_paginated(self, skip: int = 0, limit: int = 20, status_filter: str = None, category_filter: str = None):
        """Get long videos with pagination and optional filters"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get longvideos - not connected")
            return [], 0
        try:
            collection = self.db["longvideos"]
            query = {}
            if status_filter:
                query["status"] = status_filter
            if category_filter:
                query["category"] = ObjectId(category_filter)
            
            # Get total count
            total = await collection.count_documents(query)
            
            # Get paginated results
            cursor = collection.find(query).skip(skip).limit(limit)
            videos = await cursor.to_list(length=limit)
            
            logger.info(f"[MongoDB] Found {len(videos)} longvideos (total: {total})")
            return videos, total
        except Exception as e:
            logger.error(f"[MongoDB] Get longvideos error: {str(e)}", exc_info=True)
            return [], 0

    async def update_longvideo_fields(self, video_id: str | ObjectId, updates: dict, retries: int = MAX_RETRIES) -> bool:
        """Update specific fields on a long video document with optional retries."""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot update longvideo - not connected")
            return False

        oid = ObjectId(video_id) if not isinstance(video_id, ObjectId) else video_id

        for attempt in range(1, retries + 1):
            try:
                logger.info(f"[MongoDB] Update longvideo start attempt={attempt} id={oid} fields={len(updates)}")
                collection = self.db["longvideos"]
                result = await collection.update_one({"_id": oid}, {"$set": updates})
                if result.modified_count > 0:
                    logger.info(f"[MongoDB] Update longvideo done id={oid} modified={result.modified_count}")
                    return True
                else:
                    logger.warning(f"[MongoDB] Update longvideo none id={oid}")
                    return False
            except Exception as e:
                logger.error(f"[MongoDB] Update longvideo failed attempt={attempt} id={video_id} error={str(e)}", exc_info=True)
                if attempt < retries:
                    await asyncio.sleep(1)
                    continue
                return False

    async def delete_longvideo(self, video_id: str | ObjectId) -> bool:
        """Delete a long video"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot delete longvideo - not connected")
            return False
        try:
            oid = ObjectId(video_id) if not isinstance(video_id, ObjectId) else video_id
            logger.info(f"[MongoDB] Deleting longvideo: {oid}")
            collection = self.db["longvideos"]
            result = await collection.delete_one({"_id": oid})
            if result.deleted_count > 0:
                logger.info(f"[MongoDB] Longvideo deleted: {oid}")
                return True
            else:
                logger.warning(f"[MongoDB] Longvideo not found for deletion: {oid}")
                return False
        except Exception as e:
            logger.error(f"[MongoDB] Delete longvideo error for {video_id}: {str(e)}", exc_info=True)
            return False