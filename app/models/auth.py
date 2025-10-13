from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

class UserResponse(BaseModel):
    user_id: str
    email: str
    displayName: Optional[str] = None
    role: str
    profileImage: Optional[str] = None
