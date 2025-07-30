from fastapi import FastAPI
from app.api import news
from app.config.settings import Settings

settings = Settings()

app = FastAPI(title="FastAPI News Service")

# Routers
app.include_router(news.router, prefix="/api/news", tags=["news"])

@app.get("/health")
async def health():
    return {"status": "ok"}
