from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import news
from app.config.settings import Settings
import os 

settings = Settings()

app = FastAPI(
    title="FastAPI News Translation & TTS Service",
    description="Microservice for news translation and text-to-speech generation",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(news.router, prefix="/api", tags=["news"])

@app.get("/health")
async def health():
    """Root health check endpoint"""
    return {
        "status": "healthy",
        "service": "FastAPI News Translation & TTS Service",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FastAPI News Translation & TTS Service",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))