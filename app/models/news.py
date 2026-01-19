from pydantic import BaseModel
from typing import Optional, Dict, List, Literal
from datetime import datetime

class NewsCreateRequest(BaseModel):
    title: str
    description: str
    # category: str
    author: str
    newsImage: str
    publishedAt: datetime
    magazineType: Literal["magazine", "magazine2"]
    newsType: Literal["statenews", "districtnews", "specialnews", "articles"]
    district_slug: Optional[str] = None  # For district news filtering
    source: Optional[str] = None  # Source link for articles 

class NewsUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    # category: Optional[str] = None
    author: Optional[str] = None
    newsImage: Optional[str] = None
    magazineType: Optional[Literal["magazine", "magazine2"]] = None
    newsType: Optional[Literal["statenews", "districtnews", "specialnews", "articles"]] = None
    district_slug: Optional[str] = None  # For district news filtering
    source: Optional[str] = None  # Source link for articles
    status: Optional[Literal["pending", "approved", "rejected"]] = None
    isLive: Optional[bool] = None

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
    azure_blob_connected: bool
    timestamp: datetime
    version: str