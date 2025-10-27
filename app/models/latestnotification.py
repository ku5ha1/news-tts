from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from bson import ObjectId

class LatestNotificationCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, description="Notification title")
    link: str = Field(..., min_length=1, description="Notification link")

class LatestNotificationUpdateRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, description="Notification title")
    link: Optional[str] = Field(None, min_length=1, description="Notification link")

class LatestNotificationResponse(BaseModel):
    success: bool
    data: Dict[str, Any]

class LatestNotificationListResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    total: int
    page: int
    page_size: int
