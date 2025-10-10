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

async def preload_models() -> None:
    global _is_ready, _model_err
    try:
        t0 = perf_counter()
        log.info("🔁 Skipping model preload - IndicTrans2 handles model loading natively...")
        
        # Ensure cache directory exists
        cache_dir = os.getenv("HF_HUB_CACHE", "/app/.cache/huggingface/hub")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        dt = perf_counter() - t0
        _is_ready = True
        log.info(f"IndicTrans2 setup ready in {dt:.2f}s")
        log.info("Models will be loaded on-demand by IndicTrans2")
        log.info("TTS will use ElevenLabs API - no local model preloading needed")
        
    except Exception as e:
        _model_err = repr(e)
        _is_ready = False
        log.exception("IndicTrans2 setup failed")
        
        # Add specific guidance for IndicTrans2 errors
        if "IndicTrans2" in str(e) or "Model" in str(e):
            log.error("CRITICAL: IndicTrans2 setup issue detected!")
            log.error("This error suggests IndicTrans2 failed to initialize properly.")
            log.error("Please check IndicTrans2 installation and repository clone.")
            log.error("Or run the diagnostic script: python app/scripts/fix_indictrans.py")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        log.info("Starting service warmup...")
        from app.services.translation_service import translation_service
    
        await asyncio.to_thread(translation_service.warmup)
        log.info("Translation service warmup completed successfully")
        log.info("ElevenLabs TTS service ready - no warmup needed")
        global _is_ready
        _is_ready = True
    except Exception as e:
        log.exception(f"Warmup failed: {e}")
        global _model_err
        _model_err = str(e)
        _is_ready = False
    yield
    

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

@app.middleware("http")
async def readiness_gate(request, call_next):
    path = request.url.path
    # Gate translation-heavy endpoints until warmup completes
    if path in ("/api/create", "/api/translate") and not _is_ready:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Service warming up, please retry shortly",
                "ready": _is_ready,
                "error": _model_err,
            },
        )
    return await call_next(request)

@app.get("/health")
async def health():
    
    return {
        "status": "ok", 
        "service": "news-tts", 
        "version": "2.0.0",
        "translation": "dist-200M",
        "tts": "ElevenLabs API"
    }

@app.get("/ready")
async def ready():
    
    return {
        "ready": _is_ready,
        "error": _model_err,
        "service": "news-tts",
        "version": "2.0.0",
        "translation_models": "dist-200M loaded" if _is_ready else "loading...",
        "tts_service": "ElevenLabs API ready"
    }

@app.get("/")
async def root():
    return {
        "message": "Lightweight Translation & TTS", 
        "translation": "IndicTrans2 dist-200M",
        "tts": "ElevenLabs API",
        "docs": "/docs", 
        "health": "/health", 
        "ready": "/ready"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
