from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    CORS_ORIGINS: str
    
    # Database
    MONGO_URI: str
    DATABASE_NAME: str
    
    # Firebase
    FIREBASE_SERVICE_ACCOUNT_BASE64: str
    FIREBASE_STORAGE_BUCKET: str
    
    # AI Models
    AI4BHARAT_MODELS_PATH: str
    ELEVENLABS_API_KEY: str
    
    # Cloud Configuration
    CLOUD_PROVIDER: str
    # These fields are now optional, which is what we need to avoid the validation error.
    # We'll default them to None if they are not found in the environment.
    REGION: Optional[str] = None
    CLUSTER_NAME: Optional[str] = None
    SERVICE_NAME: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str

    class Config:
        env_file = ".env"
