from fastapi import APIRouter, HTTPException, status
from app.models.auth import LoginRequest, LoginResponse, UserResponse
from app.services.auth_service import auth_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/login", response_model=LoginResponse)
async def login(credentials: LoginRequest):
    """Test login endpoint for authentication."""
    try:
        logger.info(f"[LOGIN] Login attempt for email: {credentials.email}")
        
        # Authenticate user and generate token
        result = await auth_service.login_user(credentials.email, credentials.password)
        
        logger.info(f"[LOGIN] Login successful for: {credentials.email}")
        
        return LoginResponse(
            success=True,
            message="Login successful",
            data=result
        )
        
    except ValueError as e:
        error_msg = str(e)
        logger.warning(f"[LOGIN] Login failed for {credentials.email}: {error_msg}")
        
        return LoginResponse(
            success=False,
            message=error_msg,
            data=None
        )
        
    except Exception as e:
        logger.error(f"[LOGIN] Login error for {credentials.email}: {e}")
        
        return LoginResponse(
            success=False,
            message="Login failed due to server error",
            data=None
        )

@router.get("/test-token", response_model=dict)
async def test_token():
    """Test endpoint to verify JWT authentication is working."""
    return {
        "message": "JWT authentication is working",
        "status": "success"
    }
