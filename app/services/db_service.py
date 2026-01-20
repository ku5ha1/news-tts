import logging
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from app.config.settings import settings
import asyncio
from typing import Optional, List

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

    async def get_news_paginated(self, skip: int = 0, limit: int = 20, status_filter: str = None, district_slug_filter: str = None, date_filter: str = None, news_type_filter: str = None):
        """Get news with pagination and optional status/district/date/newsType filters"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get news - not connected")
            return [], 0
        try:
            query = {}
            if status_filter:
                query["status"] = status_filter
            if district_slug_filter:
                query["district_slug"] = district_slug_filter
            if news_type_filter:
                query["newsType"] = news_type_filter
            if date_filter:
                # Parse date string (YYYY-MM-DD) and create date range for the entire day
                from datetime import datetime, timedelta
                try:
                    target_date = datetime.strptime(date_filter, "%Y-%m-%d")
                    start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
                    end_of_day = start_of_day + timedelta(days=1) - timedelta(microseconds=1)
                    query["publishedAt"] = {
                        "$gte": start_of_day,
                        "$lte": end_of_day
                    }
                    logger.info(f"[MongoDB] Date filter applied: {start_of_day} to {end_of_day}")
                except ValueError:
                    logger.warning(f"[MongoDB] Invalid date format: {date_filter}, expected YYYY-MM-DD")
            
            # Get total count
            total = await self.collection.count_documents(query)
            
            # Get paginated results, sorted by published date (newest first)
            cursor = self.collection.find(query).sort("publishedAt", -1).skip(skip).limit(limit)
            news_list = await cursor.to_list(length=limit)
            
            logger.info(f"[MongoDB] Found {len(news_list)} news (total: {total})")
            return news_list, total
        except Exception as e:
            logger.error(f"[MongoDB] Get news paginated error: {str(e)}", exc_info=True)
            return [], 0

    async def delete_news(self, news_id: str | ObjectId) -> bool:
        """Delete a news document"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot delete news - not connected")
            return False
        try:
            oid = ObjectId(news_id) if not isinstance(news_id, ObjectId) else news_id
            logger.info(f"[MongoDB] Deleting news: {oid}")
            result = await self.collection.delete_one({"_id": oid})
            if result.deleted_count > 0:
                logger.info(f"[MongoDB] News deleted: {oid}")
                return True
            else:
                logger.warning(f"[MongoDB] News not found for deletion: {oid}")
                return False
        except Exception as e:
            logger.error(f"[MongoDB] Delete news error for {news_id}: {str(e)}", exc_info=True)
            return False

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
                if result.matched_count > 0:
                    logger.info(f"[MongoDB] Update longvideo done id={oid} matched={result.matched_count} modified={result.modified_count}")
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

    # Short Video methods
    async def insert_shortvideo(self, data: dict):
        """Insert short video document into MongoDB"""
        if not self.connected or not self.client:
            raise RuntimeError("Database not connected")
        try:
            logger.info(f"[MongoDB] Insert shortvideo start id={data.get('_id')}")
            collection = self.db["videos"]
            result = await collection.insert_one(data)
            logger.info(f"[MongoDB] Insert shortvideo done id={result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"[MongoDB] Insert shortvideo failed error={str(e)}", exc_info=True)
            raise RuntimeError(f"Database insert failed: {str(e)}")

    async def get_shortvideo_by_id(self, video_id: str | ObjectId):
        """Get short video by ID"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get shortvideo - not connected")
            return None
        try:
            oid = ObjectId(video_id) if not isinstance(video_id, ObjectId) else video_id
            logger.info(f"[MongoDB] Fetching shortvideo: {oid}")
            collection = self.db["videos"]
            result = await collection.find_one({"_id": oid})
            if result:
                logger.info(f"[MongoDB] Shortvideo found: {oid}")
            else:
                logger.warning(f"[MongoDB] Shortvideo not found: {oid}")
            return result
        except Exception as e:
            logger.error(f"[MongoDB] Get shortvideo error for {video_id}: {str(e)}", exc_info=True)
            return None

    async def get_shortvideos_paginated(self, skip: int = 0, limit: int = 20, status_filter: str = None, category_filter: str = None):
        """Get short videos with pagination and optional filters"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get shortvideos - not connected")
            return [], 0
        try:
            collection = self.db["videos"]
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
            
            logger.info(f"[MongoDB] Found {len(videos)} shortvideos (total: {total})")
            return videos, total
        except Exception as e:
            logger.error(f"[MongoDB] Get shortvideos error: {str(e)}", exc_info=True)
            return [], 0

    async def update_shortvideo_fields(self, video_id: str | ObjectId, updates: dict, retries: int = MAX_RETRIES) -> bool:
        """Update specific fields on a short video document with optional retries."""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot update shortvideo - not connected")
            return False

        oid = ObjectId(video_id) if not isinstance(video_id, ObjectId) else video_id

        for attempt in range(1, retries + 1):
            try:
                logger.info(f"[MongoDB] Update shortvideo start attempt={attempt} id={oid} fields={len(updates)}")
                collection = self.db["videos"]
                result = await collection.update_one({"_id": oid}, {"$set": updates})
                if result.modified_count > 0:
                    logger.info(f"[MongoDB] Update shortvideo done id={oid} modified={result.modified_count}")
                    return True
                else:
                    logger.warning(f"[MongoDB] Update shortvideo none id={oid}")
                    return False
            except Exception as e:
                logger.error(f"[MongoDB] Update shortvideo failed attempt={attempt} id={video_id} error={str(e)}", exc_info=True)
                if attempt < retries:
                    await asyncio.sleep(1)
                    continue
                return False

    async def delete_shortvideo(self, video_id: str | ObjectId) -> bool:
        """Delete a short video"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot delete shortvideo - not connected")
            return False
        try:
            oid = ObjectId(video_id) if not isinstance(video_id, ObjectId) else video_id
            logger.info(f"[MongoDB] Deleting shortvideo: {oid}")
            collection = self.db["videos"]
            result = await collection.delete_one({"_id": oid})
            if result.deleted_count > 0:
                logger.info(f"[MongoDB] Shortvideo deleted: {oid}")
                return True
            else:
                logger.warning(f"[MongoDB] Shortvideo not found for deletion: {oid}")
                return False
        except Exception as e:
            logger.error(f"[MongoDB] Delete shortvideo error for {video_id}: {str(e)}", exc_info=True)
            return False

    # Photo methods
    async def insert_photo(self, data: dict):
        """Insert photo document into MongoDB"""
        if not self.connected or not self.client:
            raise RuntimeError("Database not connected")
        try:
            logger.info(f"[MongoDB] Insert photo start id={data.get('_id')}")
            collection = self.db["photos"]
            result = await collection.insert_one(data)
            logger.info(f"[MongoDB] Insert photo done id={result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"[MongoDB] Insert photo failed error={str(e)}", exc_info=True)
            raise RuntimeError(f"Database insert failed: {str(e)}")

    async def get_photo_by_id(self, photo_id: str | ObjectId):
        """Get photo by ID"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get photo - not connected")
            return None
        try:
            oid = ObjectId(photo_id) if not isinstance(photo_id, ObjectId) else photo_id
            logger.info(f"[MongoDB] Fetching photo: {oid}")
            collection = self.db["photos"]
            result = await collection.find_one({"_id": oid})
            if result:
                logger.info(f"[MongoDB] Photo found: {oid}")
            else:
                logger.warning(f"[MongoDB] Photo not found: {oid}")
            return result
        except Exception as e:
            logger.error(f"[MongoDB] Get photo error for {photo_id}: {str(e)}", exc_info=True)
            return None

    async def get_photos_paginated(self, skip: int = 0, limit: int = 20, status_filter: str = None, category_filter: str = None):
        """Get photos with pagination and optional status and category filters"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get photos - not connected")
            return [], 0
        try:
            collection = self.db["photos"]
            query = {}
            if status_filter:
                query["status"] = status_filter
            if category_filter:
                try:
                    query["category"] = ObjectId(category_filter)
                except Exception:
                    logger.warning(f"[MongoDB] Invalid category ObjectId: {category_filter}")
                    return [], 0
            
            # Get total count
            total = await collection.count_documents(query)
            
            # Get paginated results
            cursor = collection.find(query).skip(skip).limit(limit)
            photos = await cursor.to_list(length=limit)
            
            logger.info(f"[MongoDB] Found {len(photos)} photos (total: {total})")
            return photos, total
        except Exception as e:
            logger.error(f"[MongoDB] Get photos error: {str(e)}", exc_info=True)
            return [], 0

    async def update_photo_fields(self, photo_id: str | ObjectId, updates: dict, retries: int = MAX_RETRIES) -> bool:
        """Update specific fields on a photo document with optional retries."""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot update photo - not connected")
            return False

        oid = ObjectId(photo_id) if not isinstance(photo_id, ObjectId) else photo_id

        for attempt in range(1, retries + 1):
            try:
                logger.info(f"[MongoDB] Update photo start attempt={attempt} id={oid} fields={len(updates)}")
                collection = self.db["photos"]
                result = await collection.update_one({"_id": oid}, {"$set": updates})
                if result.modified_count > 0:
                    logger.info(f"[MongoDB] Update photo done id={oid} modified={result.modified_count}")
                    return True
                else:
                    logger.warning(f"[MongoDB] Update photo none id={oid}")
                    return False
            except Exception as e:
                logger.error(f"[MongoDB] Update photo failed attempt={attempt} id={photo_id} error={str(e)}", exc_info=True)
                if attempt < retries:
                    await asyncio.sleep(1)
                    continue
                return False

    async def delete_photo(self, photo_id: str | ObjectId) -> bool:
        """Delete a photo"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot delete photo - not connected")
            return False
        try:
            oid = ObjectId(photo_id) if not isinstance(photo_id, ObjectId) else photo_id
            logger.info(f"[MongoDB] Deleting photo: {oid}")
            collection = self.db["photos"]
            result = await collection.delete_one({"_id": oid})
            if result.deleted_count > 0:
                logger.info(f"[MongoDB] Photo deleted: {oid}")
                return True
            else:
                logger.warning(f"[MongoDB] Photo not found for deletion: {oid}")
                return False
        except Exception as e:
            logger.error(f"[MongoDB] Delete photo error for {photo_id}: {str(e)}", exc_info=True)
            return False

    # Magazine methods
    async def insert_magazine(self, data: dict):
        """Insert magazine document into MongoDB"""
        if not self.connected or not self.client:
            raise RuntimeError("Database not connected")
        try:
            logger.info(f"[MongoDB] Insert magazine start id={data.get('_id')}")
            collection = self.db["magazines"]
            result = await collection.insert_one(data)
            logger.info(f"[MongoDB] Insert magazine done id={result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"[MongoDB] Insert magazine failed error={str(e)}", exc_info=True)
            raise RuntimeError(f"Database insert failed: {str(e)}")

    async def get_magazine_by_id(self, magazine_id: str | ObjectId):
        """Get magazine by ID"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get magazine - not connected")
            return None
        try:
            oid = ObjectId(magazine_id) if not isinstance(magazine_id, ObjectId) else magazine_id
            logger.info(f"[MongoDB] Fetching magazine: {oid}")
            collection = self.db["magazines"]
            result = await collection.find_one({"_id": oid})
            if result:
                logger.info(f"[MongoDB] Magazine found: {oid}")
            else:
                logger.warning(f"[MongoDB] Magazine not found: {oid}")
            return result
        except Exception as e:
            logger.error(f"[MongoDB] Get magazine error for {magazine_id}: {str(e)}", exc_info=True)
            return None

    async def get_magazines_paginated(self, skip: int = 0, limit: int = 20, status_filter: str = None):
        """Get magazines with pagination and optional status filter"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get magazines - not connected")
            return [], 0
        try:
            collection = self.db["magazines"]
            query = {}
            if status_filter:
                query["status"] = status_filter
            
            # Get total count
            total = await collection.count_documents(query)
            
            # Get paginated results
            cursor = collection.find(query).skip(skip).limit(limit)
            magazines = await cursor.to_list(length=limit)
            
            logger.info(f"[MongoDB] Found {len(magazines)} magazines (total: {total})")
            return magazines, total
        except Exception as e:
            logger.error(f"[MongoDB] Get magazines error: {str(e)}", exc_info=True)
            return [], 0

    async def update_magazine_fields(self, magazine_id: str | ObjectId, updates: dict, retries: int = MAX_RETRIES) -> bool:
        """Update specific fields on a magazine document with optional retries."""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot update magazine - not connected")
            return False

        oid = ObjectId(magazine_id) if not isinstance(magazine_id, ObjectId) else magazine_id

        for attempt in range(1, retries + 1):
            try:
                logger.info(f"[MongoDB] Update magazine start attempt={attempt} id={oid} fields={len(updates)}")
                collection = self.db["magazines"]
                result = await collection.update_one({"_id": oid}, {"$set": updates})
                if result.modified_count > 0:
                    logger.info(f"[MongoDB] Update magazine done id={oid} modified={result.modified_count}")
                    return True
                else:
                    logger.warning(f"[MongoDB] Update magazine none id={oid}")
                    return False
            except Exception as e:
                logger.error(f"[MongoDB] Update magazine failed attempt={attempt} id={magazine_id} error={str(e)}", exc_info=True)
                if attempt < retries:
                    await asyncio.sleep(1)
                    continue
                return False

    async def delete_magazine(self, magazine_id: str | ObjectId) -> bool:
        """Delete a magazine"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot delete magazine - not connected")
            return False
        try:
            oid = ObjectId(magazine_id) if not isinstance(magazine_id, ObjectId) else magazine_id
            logger.info(f"[MongoDB] Deleting magazine: {oid}")
            collection = self.db["magazines"]
            result = await collection.delete_one({"_id": oid})
            if result.deleted_count > 0:
                logger.info(f"[MongoDB] Magazine deleted: {oid}")
                return True
            else:
                logger.warning(f"[MongoDB] Magazine not found for deletion: {oid}")
                return False
        except Exception as e:
            logger.error(f"[MongoDB] Delete magazine error for {magazine_id}: {str(e)}", exc_info=True)
            return False

    # Magazine2 methods
    async def insert_magazine2(self, data: dict):
        """Insert magazine2 document into MongoDB"""
        if not self.connected or not self.client:
            raise RuntimeError("Database not connected")
        try:
            logger.info(f"[MongoDB] Insert magazine2 start id={data.get('_id')}")
            collection = self.db["magazine2"]
            result = await collection.insert_one(data)
            logger.info(f"[MongoDB] Insert magazine2 done id={result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"[MongoDB] Insert magazine2 failed error={str(e)}", exc_info=True)
            raise RuntimeError(f"Database insert failed: {str(e)}")

    async def get_magazine2_by_id(self, magazine2_id: str | ObjectId):
        """Get magazine2 by ID"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get magazine2 - not connected")
            return None
        try:
            oid = ObjectId(magazine2_id) if not isinstance(magazine2_id, ObjectId) else magazine2_id
            logger.info(f"[MongoDB] Fetching magazine2: {oid}")
            collection = self.db["magazine2"]
            result = await collection.find_one({"_id": oid})
            if result:
                logger.info(f"[MongoDB] Magazine2 found: {oid}")
            else:
                logger.warning(f"[MongoDB] Magazine2 not found: {oid}")
            return result
        except Exception as e:
            logger.error(f"[MongoDB] Get magazine2 error for {magazine2_id}: {str(e)}", exc_info=True)
            return None

    async def get_magazine2s_paginated(self, skip: int = 0, limit: int = 20, status_filter: str = None):
        """Get magazine2s with pagination and optional status filter"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get magazine2s - not connected")
            return [], 0
        try:
            collection = self.db["magazine2"]
            query = {}
            if status_filter:
                query["status"] = status_filter
            
            # Get total count
            total = await collection.count_documents(query)
            
            # Get paginated results
            cursor = collection.find(query).skip(skip).limit(limit)
            magazine2s = await cursor.to_list(length=limit)
            
            logger.info(f"[MongoDB] Found {len(magazine2s)} magazine2s (total: {total})")
            return magazine2s, total
        except Exception as e:
            logger.error(f"[MongoDB] Get magazine2s error: {str(e)}", exc_info=True)
            return [], 0

    async def update_magazine2_fields(self, magazine2_id: str | ObjectId, updates: dict, retries: int = MAX_RETRIES) -> bool:
        """Update specific fields on a magazine2 document with optional retries."""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot update magazine2 - not connected")
            return False

        oid = ObjectId(magazine2_id) if not isinstance(magazine2_id, ObjectId) else magazine2_id

        for attempt in range(1, retries + 1):
            try:
                logger.info(f"[MongoDB] Update magazine2 start attempt={attempt} id={oid} fields={len(updates)}")
                collection = self.db["magazine2"]
                result = await collection.update_one({"_id": oid}, {"$set": updates})
                if result.modified_count > 0:
                    logger.info(f"[MongoDB] Update magazine2 done id={oid} modified={result.modified_count}")
                    return True
                else:
                    logger.warning(f"[MongoDB] Update magazine2 none id={oid}")
                    return False
            except Exception as e:
                logger.error(f"[MongoDB] Update magazine2 failed attempt={attempt} id={magazine2_id} error={str(e)}", exc_info=True)
                if attempt < retries:
                    await asyncio.sleep(1)
                    continue
                return False

    async def delete_magazine2(self, magazine2_id: str | ObjectId) -> bool:
        """Delete a magazine2"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot delete magazine2 - not connected")
            return False
        try:
            oid = ObjectId(magazine2_id) if not isinstance(magazine2_id, ObjectId) else magazine2_id
            logger.info(f"[MongoDB] Deleting magazine2: {oid}")
            collection = self.db["magazine2"]
            result = await collection.delete_one({"_id": oid})
            if result.deleted_count > 0:
                logger.info(f"[MongoDB] Magazine2 deleted: {oid}")
                return True
            else:
                logger.warning(f"[MongoDB] Magazine2 not found for deletion: {oid}")
                return False
        except Exception as e:
            logger.error(f"[MongoDB] Delete magazine2 error for {magazine2_id}: {str(e)}", exc_info=True)
            return False

    # StaticPage methods
    async def insert_staticpage(self, staticpage_document: dict) -> bool:
        """Insert a new static page document."""
        try:
            collection = self.db["staticpages"]
            result = await collection.insert_one(staticpage_document)
            logger.info(f"[MongoDB] StaticPage inserted: {result.inserted_id}")
            return True
        except Exception as e:
            logger.error(f"[MongoDB] Insert staticpage error: {str(e)}", exc_info=True)
            return False

    async def get_staticpage_by_id(self, staticpage_id: ObjectId) -> Optional[dict]:
        """Get a static page by ID."""
        try:
            collection = self.db["staticpages"]
            staticpage = await collection.find_one({"_id": staticpage_id})
            if staticpage:
                logger.info(f"[MongoDB] StaticPage found: {staticpage_id}")
                return staticpage
            else:
                logger.warning(f"[MongoDB] StaticPage not found: {staticpage_id}")
                return None
        except Exception as e:
            logger.error(f"[MongoDB] Get staticpage error for {staticpage_id}: {str(e)}", exc_info=True)
            return None

    async def get_staticpages_paginated(self, skip: int = 0, limit: int = 20, status_filter: str = None) -> tuple[List[dict], int]:
        """Get paginated static pages with optional status filter."""
        try:
            collection = self.db["staticpages"]
            
            # Build filter
            filter_dict = {}
            if status_filter:
                filter_dict["status"] = status_filter
            
            # Get total count
            total = await collection.count_documents(filter_dict)
            
            # Get paginated results
            cursor = collection.find(filter_dict).sort("createdTime", -1).skip(skip).limit(limit)
            staticpages = await cursor.to_list(length=limit)
            
            logger.info(f"[MongoDB] StaticPages paginated: {len(staticpages)}/{total} (skip={skip}, limit={limit}, status={status_filter})")
            return staticpages, total
        except Exception as e:
            logger.error(f"[MongoDB] Get staticpages paginated error: {str(e)}", exc_info=True)
            return [], 0

    async def update_staticpage_fields(self, staticpage_id: ObjectId, updates: dict) -> bool:
        """Update specific fields of a static page."""
        try:
            collection = self.db["staticpages"]
            result = await collection.update_one(
                {"_id": staticpage_id},
                {"$set": updates}
            )
            if result.modified_count > 0:
                logger.info(f"[MongoDB] StaticPage updated: {staticpage_id}")
                return True
            else:
                logger.warning(f"[MongoDB] StaticPage not found for update: {staticpage_id}")
                return False
        except Exception as e:
            logger.error(f"[MongoDB] Update staticpage error for {staticpage_id}: {str(e)}", exc_info=True)
            return False

    async def delete_staticpage(self, staticpage_id: ObjectId) -> bool:
        """Delete a static page."""
        try:
            collection = self.db["staticpages"]
            result = await collection.delete_one({"_id": staticpage_id})
            if result.deleted_count > 0:
                logger.info(f"[MongoDB] StaticPage deleted: {staticpage_id}")
                return True
            else:
                logger.warning(f"[MongoDB] StaticPage not found for deletion: {staticpage_id}")
                return False
        except Exception as e:
            logger.error(f"[MongoDB] Delete staticpage error for {staticpage_id}: {str(e)}", exc_info=True)
            return False

    # LatestNotification methods
    async def insert_latestnotification(self, data: dict) -> bool:
        """Insert a new latest notification document."""
        try:
            collection = self.db["latestnotifications"]
            result = await collection.insert_one(data)
            logger.info(f"[MongoDB] LatestNotification inserted: {result.inserted_id}")
            return True
        except Exception as e:
            logger.error(f"[MongoDB] Insert latestnotification error: {str(e)}", exc_info=True)
            return False

    async def get_latestnotification_by_id(self, latestnotification_id: ObjectId) -> Optional[dict]:
        """Get a latest notification by ID."""
        try:
            collection = self.db["latestnotifications"]
            latestnotification = await collection.find_one({"_id": latestnotification_id})
            if latestnotification:
                logger.info(f"[MongoDB] LatestNotification found: {latestnotification_id}")
                return latestnotification
            else:
                logger.warning(f"[MongoDB] LatestNotification not found: {latestnotification_id}")
                return None
        except Exception as e:
            logger.error(f"[MongoDB] Get latestnotification error for {latestnotification_id}: {str(e)}", exc_info=True)
            return None

    async def get_latestnotifications_paginated(self, skip: int = 0, limit: int = 20) -> tuple[List[dict], int]:
        """Get paginated latest notifications."""
        try:
            collection = self.db["latestnotifications"]
            
            # Get total count
            total = await collection.count_documents({})
            
            # Get paginated results
            cursor = collection.find({}).sort("createdAt", -1).skip(skip).limit(limit)
            latestnotifications = await cursor.to_list(length=limit)
            
            logger.info(f"[MongoDB] LatestNotifications paginated: {len(latestnotifications)}/{total} (skip={skip}, limit={limit})")
            return latestnotifications, total
        except Exception as e:
            logger.error(f"[MongoDB] Get latestnotifications paginated error: {str(e)}", exc_info=True)
            return [], 0

    async def update_latestnotification_fields(self, latestnotification_id: ObjectId, updates: dict) -> bool:
        """Update specific fields of a latest notification."""
        try:
            collection = self.db["latestnotifications"]
            result = await collection.update_one(
                {"_id": latestnotification_id},
                {"$set": updates}
            )
            if result.modified_count > 0:
                logger.info(f"[MongoDB] LatestNotification updated: {latestnotification_id}")
                return True
            else:
                logger.warning(f"[MongoDB] LatestNotification not found for update: {latestnotification_id}")
                return False
        except Exception as e:
            logger.error(f"[MongoDB] Update latestnotification error for {latestnotification_id}: {str(e)}", exc_info=True)
            return False

    async def delete_latestnotification(self, latestnotification_id: ObjectId) -> bool:
        """Delete a latest notification."""
        try:
            collection = self.db["latestnotifications"]
            result = await collection.delete_one({"_id": latestnotification_id})
            if result.deleted_count > 0:
                logger.info(f"[MongoDB] LatestNotification deleted: {latestnotification_id}")
                return True
            else:
                logger.warning(f"[MongoDB] LatestNotification not found for deletion: {latestnotification_id}")
                return False
        except Exception as e:
            logger.error(f"[MongoDB] Delete latestnotification error for {latestnotification_id}: {str(e)}", exc_info=True)
            return False

    # NewArticle methods
    async def insert_newarticle(self, data: dict) -> bool:
        """Insert a new article document."""
        try:
            collection = self.db["newarticles"]
            result = await collection.insert_one(data)
            logger.info(f"[MongoDB] NewArticle inserted: {result.inserted_id}")
            return True
        except Exception as e:
            logger.error(f"[MongoDB] Insert newarticle error: {str(e)}", exc_info=True)
            return False

    async def get_newarticle_by_id(self, newarticle_id: ObjectId) -> Optional[dict]:
        """Get a new article by ID."""
        try:
            collection = self.db["newarticles"]
            newarticle = await collection.find_one({"_id": newarticle_id})
            if newarticle:
                logger.info(f"[MongoDB] NewArticle found: {newarticle_id}")
                return newarticle
            else:
                logger.warning(f"[MongoDB] NewArticle not found: {newarticle_id}")
                return None
        except Exception as e:
            logger.error(f"[MongoDB] Get newarticle error for {newarticle_id}: {str(e)}", exc_info=True)
            return None

    async def get_newarticles_paginated(self, skip: int = 0, limit: int = 20) -> tuple[List[dict], int]:
        """Get paginated new articles."""
        try:
            collection = self.db["newarticles"]
            
            # Get total count
            total = await collection.count_documents({})
            
            # Get paginated results
            cursor = collection.find({}).sort("createdAt", -1).skip(skip).limit(limit)
            newarticles = await cursor.to_list(length=limit)
            
            logger.info(f"[MongoDB] NewArticles paginated: {len(newarticles)}/{total} (skip={skip}, limit={limit})")
            return newarticles, total
        except Exception as e:
            logger.error(f"[MongoDB] Get newarticles paginated error: {str(e)}", exc_info=True)
            return [], 0

    async def update_newarticle_fields(self, newarticle_id: ObjectId, updates: dict) -> bool:
        """Update specific fields of a new article."""
        try:
            collection = self.db["newarticles"]
            result = await collection.update_one(
                {"_id": newarticle_id},
                {"$set": updates}
            )
            if result.modified_count > 0:
                logger.info(f"[MongoDB] NewArticle updated: {newarticle_id}")
                return True
            else:
                logger.warning(f"[MongoDB] NewArticle not found for update: {newarticle_id}")
                return False
        except Exception as e:
            logger.error(f"[MongoDB] Update newarticle error for {newarticle_id}: {str(e)}", exc_info=True)
            return False

    async def delete_newarticle(self, newarticle_id: ObjectId) -> bool:
        """Delete a new article."""
        try:
            collection = self.db["newarticles"]
            result = await collection.delete_one({"_id": newarticle_id})
            if result.deleted_count > 0:
                logger.info(f"[MongoDB] NewArticle deleted: {newarticle_id}")
                return True
            else:
                logger.warning(f"[MongoDB] NewArticle not found for deletion: {newarticle_id}")
                return False
        except Exception as e:
            logger.error(f"[MongoDB] Delete newarticle error for {newarticle_id}: {str(e)}", exc_info=True)
            return False 
   # Photo Category methods
    async def insert_photo_category(self, data: dict):
        """Insert photo category document into MongoDB"""
        if not self.connected or not self.client:
            raise RuntimeError("Database not connected")
        try:
            logger.info(f"[MongoDB] Insert photo category start id={data.get('_id')}")
            collection = self.db["photo_categories"]
            result = await collection.insert_one(data)
            logger.info(f"[MongoDB] Insert photo category done id={result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"[MongoDB] Insert photo category failed error={str(e)}", exc_info=True)
            raise RuntimeError(f"Database insert failed: {str(e)}")

    async def get_photo_category_by_id(self, category_id: str | ObjectId):
        """Get photo category by ID"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get photo category - not connected")
            return None
        try:
            oid = ObjectId(category_id) if not isinstance(category_id, ObjectId) else category_id
            logger.info(f"[MongoDB] Fetching photo category: {oid}")
            collection = self.db["photo_categories"]
            result = await collection.find_one({"_id": oid})
            if result:
                logger.info(f"[MongoDB] Photo category found: {oid}")
            else:
                logger.warning(f"[MongoDB] Photo category not found: {oid}")
            return result
        except Exception as e:
            logger.error(f"[MongoDB] Get photo category error for {category_id}: {str(e)}", exc_info=True)
            return None

    async def get_photo_categories_paginated(self, skip: int = 0, limit: int = 20):
        """Get photo categories with pagination"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get photo categories - not connected")
            return [], 0
        try:
            collection = self.db["photo_categories"]
            query = {}
            
            # Get total count
            total = await collection.count_documents(query)
            
            # Get paginated results
            cursor = collection.find(query).skip(skip).limit(limit)
            categories = await cursor.to_list(length=limit)
            
            logger.info(f"[MongoDB] Found {len(categories)} photo categories (total: {total})")
            return categories, total
        except Exception as e:
            logger.error(f"[MongoDB] Get photo categories error: {str(e)}", exc_info=True)
            return [], 0

    async def update_photo_category_fields(self, category_id: str | ObjectId, updates: dict, retries: int = MAX_RETRIES) -> bool:
        """Update specific fields on a photo category document with optional retries."""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot update photo category - not connected")
            return False

        oid = ObjectId(category_id) if not isinstance(category_id, ObjectId) else category_id

        for attempt in range(1, retries + 1):
            try:
                logger.info(f"[MongoDB] Update photo category start attempt={attempt} id={oid} fields={len(updates)}")
                collection = self.db["photo_categories"]
                result = await collection.update_one({"_id": oid}, {"$set": updates})
                if result.modified_count > 0:
                    logger.info(f"[MongoDB] Update photo category done id={oid} modified={result.modified_count}")
                    return True
                else:
                    logger.warning(f"[MongoDB] Update photo category none id={oid}")
                    return False
            except Exception as e:
                logger.error(f"[MongoDB] Update photo category failed attempt={attempt} id={category_id} error={str(e)}", exc_info=True)
                if attempt < retries:
                    await asyncio.sleep(1)
                    continue
                return False

    async def delete_photo_category(self, category_id: str | ObjectId) -> bool:
        """Delete a photo category"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot delete photo category - not connected")
            return False
        try:
            oid = ObjectId(category_id) if not isinstance(category_id, ObjectId) else category_id
            logger.info(f"[MongoDB] Deleting photo category: {oid}")
            collection = self.db["photo_categories"]
            result = await collection.delete_one({"_id": oid})
            if result.deleted_count > 0:
                logger.info(f"[MongoDB] Photo category deleted: {oid}")
                return True
            else:
                logger.warning(f"[MongoDB] Photo category not found for deletion: {oid}")
                return False
        except Exception as e:
            logger.error(f"[MongoDB] Delete photo category error for {category_id}: {str(e)}", exc_info=True)
            return False    # Video Category methods

    async def insert_video_category(self, data: dict):
        """Insert video category document into MongoDB"""
        if not self.connected or not self.client:
            raise RuntimeError("Database not connected")
        try:
            logger.info(f"[MongoDB] Insert video category start id={data.get('_id')}")
            collection = self.db["video_categories"]
            result = await collection.insert_one(data)
            logger.info(f"[MongoDB] Insert video category done id={result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"[MongoDB] Insert video category failed error={str(e)}", exc_info=True)
            raise RuntimeError(f"Database insert failed: {str(e)}")

    async def get_video_category_by_id(self, category_id: str | ObjectId):
        """Get video category by ID"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get video category - not connected")
            return None
        try:
            oid = ObjectId(category_id) if not isinstance(category_id, ObjectId) else category_id
            logger.info(f"[MongoDB] Fetching video category: {oid}")
            collection = self.db["video_categories"]
            result = await collection.find_one({"_id": oid})
            if result:
                logger.info(f"[MongoDB] Video category found: {oid}")
            else:
                logger.warning(f"[MongoDB] Video category not found: {oid}")
            return result
        except Exception as e:
            logger.error(f"[MongoDB] Get video category error for {category_id}: {str(e)}", exc_info=True)
            return None

    async def get_video_categories_paginated(self, skip: int = 0, limit: int = 20):
        """Get video categories with pagination"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get video categories - not connected")
            return [], 0
        try:
            collection = self.db["video_categories"]
            query = {}
            
            # Get total count
            total = await collection.count_documents(query)
            
            # Get paginated results
            cursor = collection.find(query).skip(skip).limit(limit)
            categories = await cursor.to_list(length=limit)
            
            logger.info(f"[MongoDB] Found {len(categories)} video categories (total: {total})")
            return categories, total
        except Exception as e:
            logger.error(f"[MongoDB] Get video categories error: {str(e)}", exc_info=True)
            return [], 0

    async def update_video_category_fields(self, category_id: str | ObjectId, updates: dict, retries: int = MAX_RETRIES) -> bool:
        """Update specific fields on a video category document with optional retries."""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot update video category - not connected")
            return False

        oid = ObjectId(category_id) if not isinstance(category_id, ObjectId) else category_id

        for attempt in range(1, retries + 1):
            try:
                logger.info(f"[MongoDB] Update video category start attempt={attempt} id={oid} fields={len(updates)}")
                collection = self.db["video_categories"]
                result = await collection.update_one({"_id": oid}, {"$set": updates})
                if result.modified_count > 0:
                    logger.info(f"[MongoDB] Update video category done id={oid} modified={result.modified_count}")
                    return True
                else:
                    logger.warning(f"[MongoDB] Update video category none id={oid}")
                    return False
            except Exception as e:
                logger.error(f"[MongoDB] Update video category failed attempt={attempt} id={category_id} error={str(e)}", exc_info=True)
                if attempt < retries:
                    await asyncio.sleep(1)
                    continue
                return False

    async def delete_video_category(self, category_id: str | ObjectId) -> bool:
        """Delete a video category"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot delete video category - not connected")
            return False
        try:
            oid = ObjectId(category_id) if not isinstance(category_id, ObjectId) else category_id
            logger.info(f"[MongoDB] Deleting video category: {oid}")
            collection = self.db["video_categories"]
            result = await collection.delete_one({"_id": oid})
            if result.deleted_count > 0:
                logger.info(f"[MongoDB] Video category deleted: {oid}")
                return True
            else:
                logger.warning(f"[MongoDB] Video category not found for deletion: {oid}")
                return False
        except Exception as e:
            logger.error(f"[MongoDB] Delete video category error for {category_id}: {str(e)}", exc_info=True)
            return False    
# District methods
    async def insert_district(self, data: dict):
        """Insert district document into MongoDB"""
        if not self.connected or not self.client:
            raise RuntimeError("Database not connected")
        try:
            logger.info(f"[MongoDB] Insert district start id={data.get('_id')}")
            collection = self.db["districts"]
            result = await collection.insert_one(data)
            logger.info(f"[MongoDB] Insert district done id={result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"[MongoDB] Insert district failed error={str(e)}", exc_info=True)
            raise RuntimeError(f"Database insert failed: {str(e)}")

    async def get_district_by_id(self, district_id: str | ObjectId):
        """Get district by ID"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get district - not connected")
            return None
        try:
            oid = ObjectId(district_id) if not isinstance(district_id, ObjectId) else district_id
            logger.info(f"[MongoDB] Fetching district: {oid}")
            collection = self.db["districts"]
            result = await collection.find_one({"_id": oid})
            if result:
                logger.info(f"[MongoDB] District found: {oid}")
            else:
                logger.warning(f"[MongoDB] District not found: {oid}")
            return result
        except Exception as e:
            logger.error(f"[MongoDB] Get district error for {district_id}: {str(e)}", exc_info=True)
            return None

    async def get_district_by_slug(self, district_slug: str):
        """Get district by slug"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get district by slug - not connected")
            return None
        try:
            logger.info(f"[MongoDB] Fetching district by slug: {district_slug}")
            collection = self.db["districts"]
            result = await collection.find_one({"district_slug": district_slug})
            if result:
                logger.info(f"[MongoDB] District found by slug: {district_slug}")
            else:
                logger.warning(f"[MongoDB] District not found by slug: {district_slug}")
            return result
        except Exception as e:
            logger.error(f"[MongoDB] Get district by slug error for {district_slug}: {str(e)}", exc_info=True)
            return None

    async def get_districts_paginated(self, skip: int = 0, limit: int = 20):
        """Get districts with pagination"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot get districts - not connected")
            return [], 0
        try:
            collection = self.db["districts"]
            query = {}
            
            # Get total count
            total = await collection.count_documents(query)
            
            # Get paginated results sorted by district_name
            cursor = collection.find(query).sort("district_name", 1).skip(skip).limit(limit)
            districts = await cursor.to_list(length=limit)
            
            logger.info(f"[MongoDB] Found {len(districts)} districts (total: {total})")
            return districts, total
        except Exception as e:
            logger.error(f"[MongoDB] Get districts error: {str(e)}", exc_info=True)
            return [], 0

    async def update_district_fields(self, district_id: str | ObjectId, updates: dict, retries: int = MAX_RETRIES) -> bool:
        """Update specific fields on a district document with optional retries."""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot update district - not connected")
            return False

        oid = ObjectId(district_id) if not isinstance(district_id, ObjectId) else district_id

        for attempt in range(1, retries + 1):
            try:
                logger.info(f"[MongoDB] Update district start attempt={attempt} id={oid} fields={len(updates)}")
                collection = self.db["districts"]
                result = await collection.update_one({"_id": oid}, {"$set": updates})
                if result.modified_count > 0:
                    logger.info(f"[MongoDB] Update district done id={oid} modified={result.modified_count}")
                    return True
                else:
                    logger.warning(f"[MongoDB] Update district none id={oid}")
                    return False
            except Exception as e:
                logger.error(f"[MongoDB] Update district failed attempt={attempt} id={district_id} error={str(e)}", exc_info=True)
                if attempt < retries:
                    await asyncio.sleep(1)
                    continue
                return False

    async def delete_district(self, district_id: str | ObjectId) -> bool:
        """Delete a district"""
        if not self.connected or not self.client:
            logger.error("[MongoDB] Cannot delete district - not connected")
            return False
        try:
            oid = ObjectId(district_id) if not isinstance(district_id, ObjectId) else district_id
            logger.info(f"[MongoDB] Deleting district: {oid}")
            collection = self.db["districts"]
            result = await collection.delete_one({"_id": oid})
            if result.deleted_count > 0:
                logger.info(f"[MongoDB] District deleted: {oid}")
                return True
            else:
                logger.warning(f"[MongoDB] District not found for deletion: {oid}")
                return False
        except Exception as e:
            logger.error(f"[MongoDB] Delete district error for {district_id}: {str(e)}", exc_info=True)
            return False