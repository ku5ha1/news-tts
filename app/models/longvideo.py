from pydantic import BaseModel
from typing import Optional, Dict, Literal
from datetime import datetime

class LongVideoCreateRequest(BaseModel):
    title: str
    description: str
    thumbnail: str
    video_url: str
    category: str  # ObjectId as string
    magazineType: Literal["magazine", "magazine2"]
    newsType: Literal["statenews", "districtnews", "specialnews"]
    Topics: Optional[str] = None 

class LongVideoResponse(BaseModel):
    success: bool
    data: Dict

class LongVideoUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    thumbnail: Optional[str] = None
    video_url: Optional[str] = None
    category: Optional[str] = None  # ObjectId as string
    magazineType: Optional[Literal["magazine", "magazine2"]] = None
    newsType: Optional[Literal["statenews", "districtnews", "specialnews"]] = None
    Topics: Optional[str] = None
    status: Optional[Literal["pending", "approved", "rejected"]] = None

class LongVideoListResponse(BaseModel):
    success: bool
    data: Dict
    total: int
    page: int
    page_size: int
