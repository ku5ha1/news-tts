from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from app.config.settings import Settings

class DBService:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncIOMotorClient(self.settings.MONGO_URI)
        self.db = self.client[self.settings.DATABASE_NAME]
        self.collection = self.db["news"]
        self.connected = True

    async def insert_news(self, data: dict):
        """Insert news document into MongoDB"""
        try:
            result = await self.collection.insert_one(data)
            return result.inserted_id
        except Exception as e:
            print(f"Database insert error: {str(e)}")
            self.connected = False
            raise e

    async def update_news_fields(self, news_id: str | ObjectId, updates: dict) -> bool:
        """Update specific fields on a news document."""
        try:
            oid = ObjectId(news_id) if not isinstance(news_id, ObjectId) else news_id
            result = await self.collection.update_one({"_id": oid}, {"$set": updates})
            return result.modified_count > 0
        except Exception as e:
            print(f"Database update error: {str(e)}")
            return False

    async def is_connected(self) -> bool:
        """Check if MongoDB is connected"""
        try:
            # Ping the database
            await self.client.admin.command('ping')
            self.connected = True
            return True
        except Exception as e:
            print(f"Database connection error: {str(e)}")
            self.connected = False
            return False

    async def get_news_by_id(self, news_id: str):
        """Get news by ID"""
        try:
            oid = ObjectId(news_id) if not isinstance(news_id, ObjectId) else news_id
            return await self.collection.find_one({"_id": oid})
        except Exception as e:
            print(f"Database get error: {str(e)}")
            return None
