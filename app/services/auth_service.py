import os
import jwt
import logging
from typing import Optional, Dict, Any
from bson import ObjectId
from app.services.db_service import get_db_service

logger = logging.getLogger(__name__)


class AuthService:
    """Service for JWT token verification and user authentication."""
    
    # Constants
    USERS_COLLECTION = "users"
    ALLOWED_ROLES = ["admin", "moderator"]
    
    def __init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET")
        if not self.jwt_secret:
            raise RuntimeError("JWT_SECRET environment variable not set")
        
        self.db_service = get_db_service()
        logger.info("[Auth] AuthService initialized")
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and extract payload."""
        try:
            # Decode JWT token
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            logger.info(f"[Auth] JWT token verified for user: {payload.get('user_id', 'unknown')}")
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("[Auth] JWT token has expired")
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"[Auth] Invalid JWT token: {e}")
            raise ValueError("Invalid token")
        except Exception as e:
            logger.error(f"[Auth] JWT verification error: {e}")
            raise ValueError("Token verification failed")
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user from database by ID."""
        try:
            # Convert string ID to ObjectId
            object_id = ObjectId(user_id)
            
            # Query users collection
            user = await self.db_service.get_user_by_id(object_id, self.USERS_COLLECTION)
            
            if user:
                logger.info(f"[Auth] User found: {user.get('email', 'unknown')} with role: {user.get('role', 'unknown')}")
            else:
                logger.warning(f"[Auth] User not found: {user_id}")
            
            return user
            
        except Exception as e:
            logger.error(f"[Auth] Database error getting user {user_id}: {e}")
            return None
    
    async def validate_user_role(self, user_id: str) -> bool:
        """Validate if user has admin or moderator role."""
        try:
            user = await self.get_user_by_id(user_id)
            
            if not user:
                logger.warning(f"[Auth] User not found for role validation: {user_id}")
                return False
            
            user_role = user.get("role")
            is_allowed = user_role in self.ALLOWED_ROLES
            
            logger.info(f"[Auth] Role validation for {user_id}: {user_role} -> {'ALLOWED' if is_allowed else 'DENIED'}")
            return is_allowed
            
        except Exception as e:
            logger.error(f"[Auth] Role validation error for {user_id}: {e}")
            return False
    
    async def authenticate_user(self, token: str) -> Dict[str, Any]:
        """Complete authentication flow: verify token and validate role."""
        try:
            # Step 1: Verify JWT token
            payload = self.verify_jwt_token(token)
            user_id = payload.get("user_id")
            
            if not user_id:
                raise ValueError("Token missing user_id")
            
            # Step 2: Get user from database
            user = await self.get_user_by_id(user_id)
            if not user:
                raise ValueError("User not found")
            
            # Step 3: Validate user role
            if not await self.validate_user_role(user_id):
                raise ValueError("Insufficient permissions")
            
            logger.info(f"[Auth] Authentication successful for user: {user.get('email', user_id)}")
            return user
            
        except ValueError as e:
            logger.warning(f"[Auth] Authentication failed: {e}")
            raise
        except Exception as e:
            logger.error(f"[Auth] Authentication error: {e}")
            raise ValueError("Authentication failed")


# Create singleton instance
auth_service = AuthService()
