from motor.motor_asyncio import AsyncIOMotorClient
from app.config.settings import Settings

class DBService:
    def __init__(self):
        self.settings = Settings()
        self.client = AsyncIOMotorClient(self.settings.MONGO_URI)
        self.db = self.client[self.settings.DATABASE_NAME]
        self.collection = self.db["news"]

    async def insert_news(self, data: dict):
        result = await self.collection.insert_one(data)
        return result.inserted_id
