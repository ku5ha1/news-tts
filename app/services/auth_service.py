import os
import jwt
import logging
import hashlib
from datetime import datetime, timedelta
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
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256 (simple hashing for testing)"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    async def authenticate_user_credentials(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with email and password."""
        try:
            logger.info(f"[Auth] Attempting login for email: {email}")
            
            # Hash the provided password
            hashed_password = self.hash_password(password)
            
            # Query users collection for email and password
            collection = self.db_service.db[self.USERS_COLLECTION]
            user = await collection.find_one({
                "email": email,
                "password": hashed_password
            })
            
            if user:
                logger.info(f"[Auth] Login successful for: {email} with role: {user.get('role', 'unknown')}")
                return user
            else:
                logger.warning(f"[Auth] Login failed for: {email}")
                return None
                
        except Exception as e:
            logger.error(f"[Auth] Login error for {email}: {e}")
            return None
    
    def generate_jwt_token(self, user: Dict[str, Any]) -> str:
        """Generate JWT token for authenticated user."""
        try:
            # Create payload with user information
            payload = {
                "user_id": str(user["_id"]),
                "email": user["email"],
                "role": user.get("role", "user"),
                "iat": datetime.utcnow(),  # Issued at
                "exp": datetime.utcnow() + timedelta(hours=24)  # Expires in 24 hours
            }
            
            # Generate JWT token
            token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
            
            logger.info(f"[Auth] JWT token generated for user: {user['email']}")
            return token
            
        except Exception as e:
            logger.error(f"[Auth] JWT generation error: {e}")
            raise ValueError("Token generation failed")
    
    async def login_user(self, email: str, password: str) -> Dict[str, Any]:
        """Complete login flow: authenticate credentials and generate token."""
        try:
            # Step 1: Authenticate credentials
            user = await self.authenticate_user_credentials(email, password)
            if not user:
                raise ValueError("Invalid email or password")
            
            # Step 2: Generate JWT token
            token = self.generate_jwt_token(user)
            
            # Step 3: Return user data and token
            return {
                "token": token,
                "user": {
                    "user_id": str(user["_id"]),
                    "email": user["email"],
                    "displayName": user.get("displayName"),
                    "role": user.get("role", "user"),
                    "profileImage": user.get("profileImage")
                }
            }
            
        except ValueError as e:
            logger.warning(f"[Auth] Login failed: {e}")
            raise
        except Exception as e:
            logger.error(f"[Auth] Login error: {e}")
            raise ValueError("Login failed")

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
