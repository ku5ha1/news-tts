from pydantic import BaseModel
from typing import Optional, Dict, Literal
from datetime import datetime

class CategoryCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None

class CategoryResponse(BaseModel):
    success: bool
    data: Dict

class CategoryUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[Literal["pending", "approved", "rejected"]] = None

class CategoryListResponse(BaseModel):
    success: bool
    data: Dict
    total: int
    page: int
    page_size: int
