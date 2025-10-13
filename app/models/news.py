from pydantic import BaseModel
from typing import Optional, Dict, List, Literal
from datetime import datetime

class NewsCreateRequest(BaseModel):
    title: str
    description: str
    category: str
    author: str
    newsImage: str
    publishedAt: datetime
    magazineType: Literal["magazine", "magazine2"]
    newsType: Literal["statenews", "districtnews", "specialnews"]

class TranslationData(BaseModel):
    title: str
    description: str
    audio_description: str

class NewsResponse(BaseModel):
    success: bool
    data: Dict

class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str

class TranslationResponse(BaseModel):
    success: bool
    data: Dict

class TTSRequest(BaseModel):
    text: str
    language: str
    voice_id: Optional[str] = None

class TTSResponse(BaseModel):
    success: bool
    data: Dict

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    database_connected: bool
    firebase_connected: bool
    timestamp: datetime
    version: str