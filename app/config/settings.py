from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    CORS_ORIGINS: Optional[str] = None
    
    # Database
    MONGO_URI: Optional[str] = None
    DATABASE_NAME: Optional[str] = None
    
    # Firebase
    FIREBASE_SERVICE_ACCOUNT_BASE64: Optional[str] = None
    FIREBASE_STORAGE_BUCKET: Optional[str] = None
    
    # AI Models
    AI4BHARAT_MODELS_PATH: Optional[str] = None
    ELEVENLABS_API_KEY: Optional[str] = None
    
    # Logging
    LOG_LEVEL: Optional[str] = None

    class Config:
        env_file = ".env"
