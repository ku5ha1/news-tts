from pydantic import BaseModel
from typing import Optional, Dict

class NewsCreateRequest(BaseModel):
    title: str
    description: str
    source_language: Optional[str] = None  # en/hi/kn

class TranslationData(BaseModel):
    title: str
    description: str
    audio_url: Optional[str] = None

class NewsResponse(BaseModel):
    id: str
    source_language: str
    translations: Dict[str, TranslationData]
