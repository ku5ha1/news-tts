from pydantic_settings import BaseSettings
from typing import Optional, Any

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
<<<<<<< HEAD
    
    # ElevenLabs TTS
    ELEVENLABS_API_KEY: Optional[str] = None
    ELEVENLABS_VOICE_ID: Optional[str] = None
    
    # Google Cloud Translate
    GOOGLE_TRANSLATE_API_KEY: Optional[str] = None
=======
    AZURE_STORAGE_MAGAZINE_CONTAINER: Optional[str] = None
    AZURE_STORAGE_MAGAZINE2_CONTAINER: Optional[str] = None
    
    # AI Models
    AI4BHARAT_MODELS_PATH: Optional[str] = None
    ELEVENLABS_API_KEY: Optional[str] = None
    ELEVENLABS_VOICE_ID: Optional[str] = None
    
    # Authentication
    JWT_SECRET: Optional[str] = None
>>>>>>> 0f1b80f4a9e37b585911f0fe0f7c4e0bbec6734c
    
    # Logging
    LOG_LEVEL: Optional[str] = None
    
    # Azure (for deployment)
    APP_SERVICE_NAME: Optional[str] = None
    RESOURCE_GROUP: Optional[str] = None
    ACR_NAME: Optional[str] = None
    IMAGE_NAME: Optional[str] = None
    
    # Translation settings
    TRANSLATION_PER_CALL_TIMEOUT: Optional[str] = "90"
    TRANSLATION_PER_CALL_TIMEOUT_RETRY: Optional[str] = "120"
    
    # HuggingFace settings (for compatibility)
    HF_HOME: Optional[str] = None
    HF_HUB_CACHE: Optional[str] = None
    TRANSFORMERS_CACHE: Optional[str] = None
    HF_HUB_OFFLINE: Optional[str] = "0"
    TRUST_REMOTE_CODE: Optional[str] = "1"
    
    # Model settings
    MODEL_SIZE: Optional[str] = None
    AI4BHARAT_MODELS_PATH: Optional[str] = None
    
    # Azure credentials
    SECRET: Optional[str] = None
    VALUE: Optional[str] = None
    AZURE_CREDENTIALS: Optional[str] = None

    class Config:
        env_file = ".env"
<<<<<<< HEAD
        extra = "ignore"  # Ignore extra fields instead of raising errors
=======
        extra = "ignore"
>>>>>>> 0f1b80f4a9e37b585911f0fe0f7c4e0bbec6734c

# Create global settings instance
settings = Settings()
