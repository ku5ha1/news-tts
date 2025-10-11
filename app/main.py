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
        from app.services.translation_service import translation_service
        
        # Test translation to ensure models work using translate_to_all_async
        test_result = await translation_service.translate_to_all_async(
            title="Hello world",
            description="This is a test",
            source_lang="english"
        )
        
        log.info(f"Background model preloading completed: 'Hello world' -> {test_result}")
        _is_ready = True
        _model_err = None
        
    except Exception as e:
        log.exception(f"Background model preloading failed: {e}")
        _model_err = str(e)
        _is_ready = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = None  # ✅ define upfront to avoid unbound reference
    try:
        log.info("Starting services...")

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

_cors_origins = _parse_cors(settings.CORS_ORIGINS)  
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
    
    return {
        "status": "ok", 
        "service": "news-tts", 
        "version": "2.0.0",
        "translation": "1B",
        "tts": "ElevenLabs API"
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
