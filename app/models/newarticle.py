from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from bson import ObjectId

class NewArticleCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, description="Article title")
    link: str = Field(..., min_length=1, description="Article link")

class NewArticleUpdateRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, description="Article title")
    link: Optional[str] = Field(None, min_length=1, description="Article link")

class NewArticleResponse(BaseModel):
    success: bool
    data: Dict[str, Any]

class NewArticleListResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    total: int
    page: int
    page_size: int

