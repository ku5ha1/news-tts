import os
import jwt
import logging
from typing import Optional, Dict, Any
from bson import ObjectId
from app.services.db_service import DBService

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
        
        self.db_service = DBService()
        logger.info("[Auth] AuthService initialized")
    
    def decode_jwt_token(self, token: str) -> Dict[str, Any]:
        """Decode JWT token without signature verification (just extract payload)."""
        try:
            # Decode JWT token WITHOUT signature verification
            payload = jwt.decode(token, options={"verify_signature": False})
            
            if 'id' in payload and 'user_id' not in payload:
                payload['user_id'] = payload['id']
            
            logger.info(f"[Auth] JWT token decoded for user: {payload.get('user_id', payload.get('id', 'unknown'))}")
            return payload
            
        except Exception as e:
            logger.error(f"[Auth] JWT decode error: {e}")
            raise ValueError("Token decode failed")
    
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
    
    async def get_user_from_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Get user from JWT token payload (for external tokens like MERN)."""
        try:
            # Decode token and get payload (no signature verification)
            payload = self.decode_jwt_token(token)
            
            # Try to get user from database first
            user_id = payload.get('user_id', payload.get('id'))
            if user_id:
                user = await self.get_user_by_id(user_id)
                if user:
                    return user

            logger.info(f"[Auth] Creating user object from token payload for external user: {payload.get('email', 'unknown')}")
            
            user_from_token = {
                "_id": ObjectId(user_id) if user_id else None,
                "email": payload.get('email'),
                "role": payload.get('role', 'user'),
                "displayName": payload.get('displayName'),
                "profileImage": payload.get('profileImage'),
                "phone_Number": payload.get('phone_Number'),
                "source": "external_token" 
            }
            
            return user_from_token
            
        except Exception as e:
            logger.error(f"[Auth] Error getting user from token: {e}")
            return None
    
    async def authenticate_user(self, token: str) -> Dict[str, Any]:
        """Complete authentication flow: verify token and validate role."""
        try:
            user = await self.get_user_from_token(token)
            if not user:
                raise ValueError("User not found")
            
            # Step 2: Validate user role
            user_role = user.get("role", "user")
            if user_role not in self.ALLOWED_ROLES:
                raise ValueError("Insufficient permissions")
            
            logger.info(f"[Auth] Authentication successful for user: {user.get('email', 'unknown')} with role: {user_role}")
            return user
            
        except ValueError as e:
            logger.warning(f"[Auth] Authentication failed: {e}")
            raise
        except Exception as e:
            logger.error(f"[Auth] Authentication error: {e}")
            raise ValueError("Authentication failed")

auth_service = AuthService()