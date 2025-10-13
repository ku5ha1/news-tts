from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import news
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
    task = None  # ✅ define upfront to avoid unbound reference
    try:
        log.info("Starting services...")
        log.info("Using Google Cloud Translate API for translation (no local models)")
        log.info(f"Google Translate API Key: {'SET' if os.getenv('GOOGLE_TRANSLATE_API_KEY') else 'NOT SET'}")

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

@app.get("/health")
async def health():
    global _is_ready, _model_err
    
    status = "ok" if _is_ready else "loading"
    if _model_err:
        status = "error"
    
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
            "google_translate_api_key": "SET" if os.getenv("GOOGLE_TRANSLATE_API_KEY") else "NOT SET",
            "azure_storage_account": "SET" if os.getenv("AZURE_STORAGE_ACCOUNT_NAME") else "NOT SET",
            "azure_storage_container": "SET" if os.getenv("AZURE_STORAGE_AUDIOFIELD_CONTAINER") else "NOT SET",
            "branch": "develop (Google Translate + Azure Blob)",
            "startup_mode": "fast (no model loading)"
        }
    }

@app.get("/")
async def root():
    return {
        "message": "Lightweight Translation & TTS (Develop Branch)", 
        "translation": "Google Cloud Translate API",
        "tts": "ElevenLabs API",
        "branch": "develop",
        "startup": "fast (no model loading)",
        "docs": "/docs", 
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))