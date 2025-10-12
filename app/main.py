from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import news
from app.config.settings import Settings
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from time import perf_counter
from pathlib import Path

settings = Settings()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
log = logging.getLogger("news-tts-service") 

_is_ready = False
_model_err: str | None = None

async def background_model_preload():
    """Background task to preload translation models without blocking startup."""
    global _is_ready, _model_err
    
    try:
        log.info("Starting background model preloading...")
        
        # Check if HF_HOME is accessible (graceful degradation)
        hf_home = os.getenv("HF_HOME", "/mnt/models")
        if not os.path.exists(hf_home):
            log.warning(f"HF_HOME directory {hf_home} does not exist - models may not be available")
            # Don't fail - allow app to start and show error in health check
        else:
            log.info(f"HF_HOME {hf_home} exists - checking accessibility")
            # Check if we can read the cache directory
            try:
                test_file = os.path.join(hf_home, "test_read.tmp")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                log.info(f"HF_HOME {hf_home} is accessible")
            except Exception as e:
                log.warning(f"Cannot access HF_HOME {hf_home}: {e} - models may not work")
                # Don't fail - allow app to start
        
        # Try to initialize translation service lazily
        try:
            from app.services.translation_service import translation_service
            
            # Test translation to ensure models work using translate_to_all_async
            log.info("Testing translation service with sample text...")
            test_result = await translation_service.translate_to_all_async(
                title="Hello world",
                description="This is a test",
                source_lang="english"
            )
            
            log.info(f"Background model preloading completed successfully: 'Hello world' -> {test_result}")
            _is_ready = True
            _model_err = None
            
        except Exception as e:
            log.warning(f"Translation service test failed: {e}")
            # Don't fail - allow app to start in degraded mode
            _is_ready = False
            _model_err = f"Translation service unavailable: {str(e)}"
            log.info("Application will start in degraded mode - translation features may not work")

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = None  # ✅ define upfront to avoid unbound reference
    try:
        log.info("Starting services...")
        log.info(f"Environment: HF_HOME={os.getenv('HF_HOME')}, TRANSFORMERS_CACHE={os.getenv('TRANSFORMERS_CACHE')}")
        log.info(f"Offline mode: HF_HUB_OFFLINE={os.getenv('HF_HUB_OFFLINE')}")

        # Start background model preloading (non-blocking)
        task = asyncio.create_task(background_model_preload())
        log.info("Background model preload started")

        log.info("ElevenLabs TTS service ready - no warmup needed")
        # _is_ready remains False until models are loaded

    except Exception as e:
        log.exception(f"Service initialization failed: {e}")
        global _model_err, _is_ready
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
            log.exception(f"Background model preloading task failed: {e}")

    

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
    description="using indic trans 2 dist 200M and elevenlabs",
    version="2.0.0",  
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
    
    # Additional diagnostics
    hf_home = os.getenv("HF_HOME", "/mnt/models")  # Fixed default
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
    
    return {
        "status": status, 
        "service": "news-tts", 
        "version": "2.0.0",
        "translation": "IndicTrans2 200M",
        "tts": "ElevenLabs API",
        "ready": _is_ready,
        "error": _model_err,
        "diagnostics": {
            "hf_home": hf_home,
            "hf_home_exists": hf_home_exists,
            "hf_home_writable": hf_home_writable,
            "hf_hub_offline": os.getenv("HF_HUB_OFFLINE", "not_set"),
            "trust_remote_code": os.getenv("TRUST_REMOTE_CODE", "not_set"),
            "transformers_cache": os.getenv("TRANSFORMERS_CACHE", "not_set")
        }
    }

@app.get("/")
async def root():
    return {
        "message": "Lightweight Translation & TTS", 
        "translation": "IndicTrans2 1B",
        "tts": "ElevenLabs API",
        "docs": "/docs", 
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
