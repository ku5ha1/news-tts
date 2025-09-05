from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import news
from app.config.settings import Settings
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from time import perf_counter

settings = Settings()

# ---------- logging ----------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
log = logging.getLogger("news-tts-service") 

# ---------- model state ----------
_is_ready = False
_model_err: str | None = None

async def preload_models() -> None:
    """
    Preload heavy models so the first request is fast.
    This assumes you already baked weights into the image via TRANSFORMERS_CACHE.
    """
    global _is_ready, _model_err
    try:
        t0 = perf_counter()
        log.info("🔁 Preloading ai4bharat models...")
        # Import here to avoid slowing startup if module import is heavy.
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        base = "ai4bharat"
        cache_dir = os.getenv("TRANSFORMERS_CACHE", "/app/.cache/huggingface/transformers")

        # Load both directions (baked weights should make this quick)
        AutoTokenizer.from_pretrained(f"{base}/IndicTrans2-en-indic-1B", trust_remote_code=True, cache_dir=cache_dir)
        AutoModelForSeq2SeqLM.from_pretrained(f"{base}/IndicTrans2-en-indic-1B", trust_remote_code=True, cache_dir=cache_dir)
        AutoTokenizer.from_pretrained(f"{base}/IndicTrans2-indic-en-1B", trust_remote_code=True, cache_dir=cache_dir)
        AutoModelForSeq2SeqLM.from_pretrained(f"{base}/IndicTrans2-indic-en-1B", trust_remote_code=True, cache_dir=cache_dir)

        dt = perf_counter() - t0
        _is_ready = True
        log.info(f"✅ Models ready in {dt:.2f}s")
    except Exception as e:
        _model_err = repr(e)
        _is_ready = False
        log.exception("❌ Model preload failed")

# ---------- lifespan: load on startup ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kick model loading without blocking ASGI startup
    # Optional global preload for HF cache: await preload_models()
    # Service-level warmup to avoid first-request stalls
    try:
        from app.services.translation_service import translation_service
        # Run warmup in background so startup isn't blocked for long
        asyncio.create_task(asyncio.to_thread(translation_service.warmup))
    except Exception:
        log.exception("Warmup scheduling failed")
    yield
    # (Optional) cleanup here

def _parse_cors(origins_env: str | None) -> list[str]:
    if not origins_env:
        return ["*"]
    value = origins_env.strip()
    if value == "*" or value == '["*"]':
        return ["*"]
    # split + strip empties
    out = [o.strip() for o in value.split(",") if o.strip()]
    return out or ["*"]

app = FastAPI(
    title="FastAPI News Translation & TTS Service",
    description="Microservice for news translation and text-to-speech generation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
_cors_origins = _parse_cors(os.getenv("CORS_ORIGINS"))
# If using wildcard origins, do not allow credentials per CORS spec
_allow_credentials = False if _cors_origins == ["*"] else True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(news.router, prefix="/api", tags=["news"])

# ---------- health/readiness ----------
@app.get("/health")
async def health():
    """Liveness: cheap, never touches models."""
    return {"status": "ok", "service": "news-tts", "version": "1.0.0"}

@app.get("/ready")
async def ready():
    """Readiness: reports model state."""
    return {
        "ready": _is_ready,
        "error": _model_err,
        "service": "news-tts",
        "version": "1.0.0",
    }

@app.get("/")
async def root():
    return {"message": "Translation & TTS", "docs": "/docs", "health": "/health", "ready": "/ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
