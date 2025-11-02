from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
<<<<<<< HEAD
from app.api import news
=======
from app.api import news, category, longvideo, shortvideo, photo, magazine, magazine2, staticpage, search, latestnotification
>>>>>>> 0f1b80f4a9e37b585911f0fe0f7c4e0bbec6734c
from app.config.settings import settings
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from time import perf_counter
from pathlib import Path

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
log = logging.getLogger("news-tts-service") 

_is_ready = False
_model_err: str | None = None

async def background_google_translate_test():
    """Background task to test Google Translate API availability."""
    global _is_ready, _model_err
    
    try:
        log.info("Testing Google Cloud Translate API...")
        
        # Test Google Translate service
        try:
            from app.services.google_translate_service import google_translate_service
            
            # Test translation to ensure API works
            log.info("Testing Google Translate service with sample text...")
            test_result = await google_translate_service.translate_to_all_async(
                title="Hello world",
                description="This is a test",
                source_lang="english"
            )
            
            log.info(f"Google Translate API test completed successfully: 'Hello world' -> {test_result}")
            _is_ready = True
            _model_err = None
            
        except Exception as e:
            log.warning(f"Google Translate API test failed: {e}")
            # Don't fail - allow app to start in degraded mode
            _is_ready = False
            _model_err = f"Google Translate API unavailable: {str(e)}"
            log.info("Application will start in degraded mode - translation features may not work")
    
    except Exception as e:
        log.error(f"Google Translate API test failed: {e}")
        _is_ready = False
        _model_err = f"Google Translate API test failed: {str(e)}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = None   
    try:
        log.info("Starting services...")
<<<<<<< HEAD
        log.info("Using Google Cloud Translate API for translation (no local models)")
        log.info(f"Google Translate API Key: {'SET' if os.getenv('GOOGLE_TRANSLATE_API_KEY') else 'NOT SET'}")
=======
        
        # Comprehensive environment variable logging
        log.info("=== ENVIRONMENT VARIABLES DEBUG ===")
        critical_env_vars = [
            "DATABASE_NAME", "MONGO_URI", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID",
            "AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_CONNECTION_STRING", 
            "AZURE_STORAGE_ACCESS_KEY", "AZURE_STORAGE_AUDIOFIELD_CONTAINER",
            "CORS_ORIGIN", "HF_HOME", "HF_HUB_OFFLINE", "TRUST_REMOTE_CODE"
        ]
        
        for var in critical_env_vars:
            value = os.getenv(var)
            if value:
                # Mask sensitive values
                if any(sensitive in var.upper() for sensitive in ["KEY", "PASSWORD", "SECRET", "TOKEN", "CONNECTION"]):
                    masked_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
                    log.info(f"  {var}: {masked_value}")
                else:
                    log.info(f"  {var}: {value}")
            else:
                log.warning(f"  {var}: NOT SET")
        
        log.info(f"Environment: HF_HOME={os.getenv('HF_HOME')}, TRANSFORMERS_CACHE={os.getenv('TRANSFORMERS_CACHE')}")
        log.info(f"Offline mode: HF_HUB_OFFLINE={os.getenv('HF_HUB_OFFLINE')}")
        
        # Test settings instance
        log.info("=== SETTINGS INSTANCE DEBUG ===")
        log.info(f"Settings DATABASE_NAME: {getattr(settings, 'DATABASE_NAME', 'NOT_FOUND')}")
        log.info(f"Settings MONGO_URI: {'SET' if getattr(settings, 'MONGO_URI', None) else 'NOT_SET'}")
        log.info(f"Settings AZURE_STORAGE_ACCOUNT_NAME: {'SET' if getattr(settings, 'AZURE_STORAGE_ACCOUNT_NAME', None) else 'NOT_SET'}")
        log.info("=== END DEBUG ===")
>>>>>>> 0f1b80f4a9e37b585911f0fe0f7c4e0bbec6734c

        # Start Google Translate API test in background (non-blocking)
        log.info("Starting Google Translate API test in background...")
        task = asyncio.create_task(background_google_translate_test())
        
        # Mark as ready immediately for fast startup
        global _is_ready, _model_err
        _is_ready = True
        log.info("Application ready - Google Translate API test running in background")

    except Exception as e:
        log.exception(f"Service initialization failed: {e}")
        _model_err = str(e)
        _is_ready = False
        # Log the error but don't crash - let the app start and show error in health check
        log.error("Service initialization failed, but allowing app to start for debugging")

    yield  # lifespan yields to FastAPI's runtime

    # Cleanup / finalization phase
    if task:
        try:
            await task
        except Exception as e:
            log.exception(f"Background Google Translate test task failed: {e}")

    

def _parse_cors(origins_env: str | None) -> list[str]:
    if not origins_env:
        return ["*"]
    value = origins_env.strip()
    if value == "*" or value == '["*"]':
        return ["*"]
    
    out = [o.strip() for o in value.split(",") if o.strip()]
    return out or ["*"]

app = FastAPI(
    title="news translation api",
    description="using Google Cloud Translate API and ElevenLabs TTS",
    version="2.0.0-dev",  
    lifespan=lifespan,
)

_cors_origins = _parse_cors(getattr(settings, 'CORS_ORIGINS', None))  
_allow_credentials = False if _cors_origins == ["*"] else True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(news.router, prefix="/api", tags=["news"])
app.include_router(category.router, prefix="/api/categories", tags=["categories"])
app.include_router(longvideo.router, prefix="/api/longvideos", tags=["longvideos"])
app.include_router(shortvideo.router, prefix="/api/shortvideos", tags=["shortvideos"])
app.include_router(photo.router, prefix="/api/photos", tags=["photos"])
app.include_router(magazine.router, prefix="/api/magazines", tags=["magazines"])
app.include_router(magazine2.router, prefix="/api/magazine2", tags=["magazine2"])
app.include_router(staticpage.router, prefix="/api/staticpages", tags=["staticpages"])
app.include_router(latestnotification.router, prefix="/api/latestnotifications", tags=["latestnotifications"])
app.include_router(search.router, tags=["search"])

@app.get("/health")
async def health():
    global _is_ready, _model_err
    
    status = "ok" if _is_ready else "loading"
    if _model_err:
        status = "error"
<<<<<<< HEAD
=======

    hf_home = os.getenv("HF_HOME", "/mnt/models") 
    hf_home_exists = os.path.exists(hf_home)
    hf_home_writable = False
    
    if hf_home_exists:
        try:
            test_file = os.path.join(hf_home, "health_check.tmp")
            with open(test_file, "w") as f:
                f.write("health_check")
            os.remove(test_file)
            hf_home_writable = True
        except Exception:
            hf_home_writable = False
>>>>>>> 0f1b80f4a9e37b585911f0fe0f7c4e0bbec6734c
    
    return {
        "status": status, 
        "service": "news-tts", 
        "version": "2.0.0-dev",
        "translation": "Google Cloud Translate API",
        "tts": "ElevenLabs API",
        "storage": "Azure Blob Storage",
        "ready": _is_ready,
        "error": _model_err,
        "diagnostics": {
<<<<<<< HEAD
            "google_translate_api_key": "SET" if os.getenv("GOOGLE_TRANSLATE_API_KEY") else "NOT SET",
            "azure_storage_account": "SET" if os.getenv("AZURE_STORAGE_ACCOUNT_NAME") else "NOT SET",
            "azure_storage_container": "SET" if os.getenv("AZURE_STORAGE_AUDIOFIELD_CONTAINER") else "NOT SET",
            "branch": "develop (Google Translate + Azure Blob)",
            "startup_mode": "fast (no model loading)"
=======
            "hf_home": hf_home,
            "hf_home_exists": hf_home_exists,
            "hf_home_writable": hf_home_writable,
            "hf_hub_offline": os.getenv("HF_HUB_OFFLINE", "not_set"),
            "trust_remote_code": os.getenv("TRUST_REMOTE_CODE", "not_set"),
            "transformers_cache": os.getenv("TRANSFORMERS_CACHE", "not_set"),
            "azure_storage_account": "SET" if os.getenv("AZURE_STORAGE_ACCOUNT_NAME") else "NOT SET",
            "azure_storage_container": "SET" if os.getenv("AZURE_STORAGE_AUDIOFIELD_CONTAINER") else "NOT SET"
>>>>>>> 0f1b80f4a9e37b585911f0fe0f7c4e0bbec6734c
        }
    }

@app.get("/")
async def root():
    return {
<<<<<<< HEAD
        "message": "Lightweight Translation & TTS (Develop Branch)", 
        "translation": "Google Cloud Translate API",
        "tts": "ElevenLabs API",
        "branch": "develop",
        "startup": "fast (no model loading)",
=======
>>>>>>> 0f1b80f4a9e37b585911f0fe0f7c4e0bbec6734c
        "docs": "/docs", 
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    import os

    num_workers = 25  
    print(f"Starting server with {num_workers} workers on {os.cpu_count()} CPU cores")
    
    # Check if SSL certificates exist
    ssl_cert_path = "/etc/letsencrypt/live/diprkarnataka.duckdns.org/fullchain.pem"
    ssl_key_path = "/etc/letsencrypt/live/diprkarnataka.duckdns.org/privkey.pem"
    
    if os.path.exists(ssl_cert_path) and os.path.exists(ssl_key_path):
        print("SSL certificates found, starting HTTPS server on port 443")
        uvicorn.run(
            "app.main:app", 
            host="0.0.0.0", 
            port=443,
            workers=num_workers,
            ssl_keyfile=ssl_key_path,
            ssl_certfile=ssl_cert_path
        )
    else:
        print("SSL certificates not found, starting HTTP server on port 8080")
        uvicorn.run(
            "app.main:app", 
            host="0.0.0.0", 
            port=int(os.environ.get("PORT", 8080)),
            workers=num_workers
        )