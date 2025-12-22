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
    
    # Azure Blob Storage
    AZURE_STORAGE_ACCOUNT_NAME: Optional[str] = None
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = None
    AZURE_STORAGE_ACCESS_KEY: Optional[str] = None
    AZURE_STORAGE_AUDIOFIELD_CONTAINER: Optional[str] = None
    AZURE_STORAGE_MAGAZINE_CONTAINER: Optional[str] = None
    AZURE_STORAGE_MAGAZINE2_CONTAINER: Optional[str] = None
    AZURE_STORAGE_OUTPUT_CONTAINER_NAME: Optional[str] = None
    
    # Azure Document Intelligence
    DOCINT_ENDPOINT: Optional[str] = None
    DOCINT_KEY: Optional[str] = None
    
    # Azure Speech (for TTS)
    AZURE_SPEECH_KEY: Optional[str] = None
    AZURE_SPEECH_REGION: Optional[str] = None
    AZURE_SPEECH_ENDPOINT: Optional[str] = None
    
    # Qdrant Cloud (Vector Database)
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: Optional[str] = "magazine2_search"
    
    # AI Models
    AI4BHARAT_MODELS_PATH: Optional[str] = None
    ELEVENLABS_API_KEY: Optional[str] = None
    ELEVENLABS_VOICE_ID: Optional[str] = None
    
    # HuggingFace Cache (for FastEmbed models)
    HF_HOME: Optional[str] = None
    HF_HUB_CACHE: Optional[str] = None
    TRANSFORMERS_CACHE: Optional[str] = None
    
    # Authentication
    JWT_SECRET: Optional[str] = None
    
    # Logging
    LOG_LEVEL: Optional[str] = None

    class Config:
        env_file = ".env"
        extra = "ignore"

# Create global settings instance
settings = Settings()
