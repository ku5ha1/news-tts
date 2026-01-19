from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class VideoCategoryCreateRequest(BaseModel):
    category_name: str

class VideoCategoryResponse(BaseModel):
    success: bool
    data: Dict

class VideoCategoryUpdateRequest(BaseModel):
    category_name: Optional[str] = None

class VideoCategoryListResponse(BaseModel):
    success: bool
    data: Dict
    total: int
    page: int
    page_size: int