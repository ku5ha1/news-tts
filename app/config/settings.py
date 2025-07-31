from pydantic_settings import BaseSettings

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
    
    # AWS/Azure Configuration
    CLOUD_PROVIDER: str
    REGION: str
    CLUSTER_NAME: str
    SERVICE_NAME: str
    
    # Logging
    LOG_LEVEL: str

    class Config:
        env_file = ".env"
