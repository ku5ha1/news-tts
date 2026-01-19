from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class PhotoCategoryCreateRequest(BaseModel):
    category_name: str

class PhotoCategoryResponse(BaseModel):
    success: bool
    data: Dict

class PhotoCategoryUpdateRequest(BaseModel):
    category_name: Optional[str] = None

class PhotoCategoryListResponse(BaseModel):
    success: bool
    data: Dict
    total: int
    page: int
    page_size: int