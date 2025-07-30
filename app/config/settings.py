from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGO_URI: str
    DATABASE_NAME: str = "news_service"
    FIREBASE_CREDENTIALS: str
    ELEVENLABS_API_KEY: str

    class Config:
        env_file = ".env"
