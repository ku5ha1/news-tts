from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class Magazine2CreateRequest(BaseModel):
    title: str
    description: str
    editionNumber: str
    publishedMonth: str
    publishedYear: str
    magazineThumbnail: str
    magazinePdf: str

class Magazine2Response(BaseModel):
    success: bool
    data: Dict

class Magazine2UpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    editionNumber: Optional[str] = None
    publishedMonth: Optional[str] = None
    publishedYear: Optional[str] = None
    magazineThumbnail: Optional[str] = None
    magazinePdf: Optional[str] = None
    status: Optional[str] = None

class Magazine2ListResponse(BaseModel):
    success: bool
    data: Dict
    total: int
    page: int
    page_size: int
