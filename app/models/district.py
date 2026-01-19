from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class DistrictCreateRequest(BaseModel):
    district_name: str
    district_code: Optional[str] = None

class DistrictResponse(BaseModel):
    success: bool
    data: Dict

class DistrictUpdateRequest(BaseModel):
    district_name: Optional[str] = None
    district_code: Optional[str] = None

class DistrictListResponse(BaseModel):
    success: bool
    data: Dict
    total: int
    page: int
    page_size: int