from pydantic import BaseModel
from typing import Optional, Dict, Literal
from datetime import datetime

class ShortVideoCreateRequest(BaseModel):
    title: str
    description: str
    thumbnail: str
    video_url: str
    # category: str  # ObjectId as string
    magazineType: Literal["magazine", "magazine2"]
    newsType: Literal["statenews", "districtnews", "specialnews"]

class ShortVideoResponse(BaseModel):
    success: bool
    data: Dict

class ShortVideoUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    thumbnail: Optional[str] = None
    video_url: Optional[str] = None
    # category: Optional[str] = None
    magazineType: Optional[Literal["magazine", "magazine2"]] = None
    newsType: Optional[Literal["statenews", "districtnews", "specialnews"]] = None
    status: Optional[Literal["pending", "approved", "rejected"]] = None

class ShortVideoListResponse(BaseModel):
    success: bool
    data: Dict
    total: int
    page: int
    page_size: int
