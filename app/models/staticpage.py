from pydantic import BaseModel
from typing import Optional, Dict, List

class StaticPageCreateRequest(BaseModel):
    staticpageName: str
    staticpageImage: str
    staticpageLink: str

class StaticPageUpdateRequest(BaseModel):
    staticpageName: Optional[str] = None
    staticpageImage: Optional[str] = None
    staticpageLink: Optional[str] = None
    status: Optional[str] = None

class StaticPageResponse(BaseModel):
    success: bool
    data: Dict

class StaticPageListResponse(BaseModel):
    success: bool
    data: Dict
    total: int
    page: int
    page_size: int
