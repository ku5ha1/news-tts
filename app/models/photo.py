from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class PhotoCreateRequest(BaseModel):
    title: str
    photoImage: str
    category: str  # ObjectId as string

class PhotoResponse(BaseModel):
    success: bool
    data: Dict

class PhotoUpdateRequest(BaseModel):
    title: Optional[str] = None
    photoImage: Optional[str] = None
    category: Optional[str] = None  # ObjectId as string
    status: Optional[str] = None

class PhotoListResponse(BaseModel):
    success: bool
    data: Dict
    total: int
    page: int
    page_size: int
