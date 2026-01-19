from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
from bson import ObjectId
import asyncio
import os
from app.models.category import (
    CategoryCreateRequest, CategoryResponse,
    CategoryUpdateRequest, CategoryListResponse
)
from app.services.db_service import DBService
from app.services.auth_service import auth_service
from app.utils.language_detection import detect_language
from app.utils.retry_utils import retry_translation_with_timeout
from app.utils.json_encoder import to_extended_json
import logging

logger = logging.getLogger(__name__)

DEFAULT_TRANSLATION_TIMEOUT = 90.0
DEFAULT_PAGE_SIZE = 20

router = APIRouter()

db_service = None

# Authentication setup
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Authentication dependency to verify JWT token and user role."""
    try:
        # Extract Bearer token
        token = credentials.credentials
        
        # Authenticate user (verify token + validate role)
        user = await auth_service.authenticate_user(token)
        
        logger.info(f"[AUTH] User authenticated: {user.get('email', 'unknown')} with role: {user.get('role', 'unknown')}")
        return user
        
    except ValueError as e:
        error_msg = str(e)
        if "expired" in error_msg.lower():
            logger.warning(f"[AUTH] Token expired: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        elif "invalid" in error_msg.lower():
            logger.warning(f"[AUTH] Invalid token: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        elif "not found" in error_msg.lower():
            logger.warning(f"[AUTH] User not found: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        elif "permissions" in error_msg.lower():
            logger.warning(f"[AUTH] Insufficient permissions: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions. Only admin and moderator roles can create categories.",
            )
        else:
            logger.error(f"[AUTH] Authentication error: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except Exception as e:
        logger.error(f"[AUTH] Unexpected authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error",
        )

def get_db_service():
    """Lazy import of DB service to avoid module-level failures."""
    global db_service
    if db_service is None:
        try:
            db_service = DBService()
        except ImportError as e:
            logger.error(f"Failed to import DB service: {e}")
            raise HTTPException(
                status_code=503, 
                detail=f"Database service import failed: {str(e)}"
            )
        except RuntimeError as e:
            logger.error(f"DB service runtime error: {e}")
            raise HTTPException(
                status_code=503, 
                detail=f"Database service unavailable: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize DB service: {e}")
            raise HTTPException(
                status_code=503, 
                detail=f"Database service unavailable: {str(e)}"
            )
    return db_service

def get_translation_service():
    """Lazy import of translation service to avoid module-level failures."""
    try:
        from app.services.translation_service import translation_service
        return translation_service
    except ImportError as e:
        logger.error(f"Failed to import translation service: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"Translation service import failed: {str(e)}"
        )
    except RuntimeError as e:
        logger.error(f"Translation service runtime error: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"Translation service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to import translation service: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"Translation service unavailable: {str(e)}"
        )

# Removed local to_extended_json - now using universal to_extended_json from utils

@router.post("/create", response_model=CategoryResponse)
async def create_category(
    payload: CategoryCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create category with translation to Hindi, Kannada, and English."""
    category_id = ObjectId()
    logger.info(f"[CATEGORY-CREATE] start category_id={category_id} user={current_user.get('email', 'unknown')}")

    try:
        # Validate input
        if not payload.name.strip():
            raise HTTPException(status_code=400, detail="Category name cannot be empty")

        # Detect source language
        source_lang = detect_language(payload.name)
        logger.info(f"[CATEGORY-CREATE] detected_language={source_lang} category_id={category_id}")

        # Translation with timeout
        try:
            timeout_sec = float(os.getenv("TRANSLATION_PER_CALL_TIMEOUT", str(DEFAULT_TRANSLATION_TIMEOUT)))
        except (ValueError, TypeError):
            timeout_sec = DEFAULT_TRANSLATION_TIMEOUT

        try:
            translation_service = get_translation_service()
            translations = await retry_translation_with_timeout(
                translation_service,
                payload.name,
                "",
                source_lang,
                timeout=timeout_sec,
                max_retries=3
            )
            logger.info(f"[CATEGORY-CREATE] translation.done langs={list(translations.keys())} category_id={category_id}")
        except asyncio.TimeoutError:
            logger.error(f"[CATEGORY-CREATE] translation timed out after {timeout_sec}s for category_id={category_id}")
            raise HTTPException(status_code=504, detail="Translation timed out")
        except Exception as e:
            logger.error(f"[CATEGORY-CREATE] translation failed for category_id={category_id}: {e}")
            if "IndicTrans2" in str(e) or "Model" in str(e):
                logger.error("This appears to be an IndicTrans2 model loading issue")
            raise

        # Determine status based on user role
        user_role = current_user.get("role", "")
        if user_role == "admin":
            status = "approved"
            logger.info(f"[CATEGORY-CREATE] Admin user creating category - status=approved category_id={category_id}")
        else:
            status = "pending"
            logger.info(f"[CATEGORY-CREATE] Non-admin user creating category - status=pending category_id={category_id}")

        # Create category document
        category_document = {
            "_id": category_id,
            "name": payload.name,
            "description": payload.description or "",
            "hindi": translations.get("hindi", {}).get("title", payload.name),
            "kannada": translations.get("kannada", {}).get("title", payload.name),
            "English": translations.get("english", {}).get("title", payload.name),
            "createdBy": ObjectId(current_user.get("_id")) if current_user.get("_id") else None,
            "status": status,
            "createdTime": datetime.utcnow(),
            "last_updated": datetime.utcnow(),
        }

        # Insert into DB
        await asyncio.wait_for(get_db_service().insert_category(category_document), timeout=15.0)

        response_doc = to_extended_json(category_document)
        logger.info(f"[CATEGORY-CREATE] success category_id={category_id}")
        return CategoryResponse(success=True, data=response_doc)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CATEGORY-CREATE] failed category_id={category_id} error={e}")
        try:
            await get_db_service().update_category_fields(category_id, {"_deleted_due_to_error": True})
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error creating category: {str(e)}")

@router.get("/list", response_model=CategoryListResponse)
async def list_categories(
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    status_filter: str = None
):
    """List categories with pagination and optional status filter."""
    try:
        logger.info(f"[CATEGORY-LIST] page={page} page_size={page_size} status={status_filter}")
        
        # Calculate skip
        skip = (page - 1) * page_size
        
        # Get categories from DB
        categories, total = await get_db_service().get_categories_paginated(
            skip=skip, 
            limit=page_size, 
            status_filter=status_filter
        )
        
        # Format response
        formatted_categories = [to_extended_json(cat) for cat in categories]
        
        return CategoryListResponse(
            success=True,
            data={"categories": formatted_categories},
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"[CATEGORY-LIST] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing categories: {str(e)}")

@router.get("/{category_id}", response_model=CategoryResponse)
async def get_category(
    category_id: str
):
    """Get a specific category by ID."""
    try:
        logger.info(f"[CATEGORY-GET] category_id={category_id}")
        
        category = await get_db_service().get_category_by_id(ObjectId(category_id))
        if not category:
            raise HTTPException(status_code=404, detail="Category not found")
        
        response_doc = to_extended_json(category)
        return CategoryResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CATEGORY-GET] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting category: {str(e)}")

@router.put("/{category_id}", response_model=CategoryResponse)
async def update_category(
    category_id: str,
    payload: CategoryUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update a category."""
    try:
        logger.info(f"[CATEGORY-UPDATE] category_id={category_id}")
        
        # Check if category exists
        existing_category = await get_db_service().get_category_by_id(ObjectId(category_id))
        if not existing_category:
            raise HTTPException(status_code=404, detail="Category not found")
        
        # Prepare update fields
        updates = {"last_updated": datetime.utcnow()}
        
        # Handle name update with translation
        if payload.name is not None:
            updates["name"] = payload.name
            
            # Re-translate the name if it's being updated
            try:
                # Detect source language
                source_lang = detect_language(payload.name)
                logger.info(f"[CATEGORY-UPDATE] detected_language={source_lang} for name update")
                
                # Translation with timeout
                try:
                    timeout_sec = float(os.getenv("TRANSLATION_PER_CALL_TIMEOUT", str(DEFAULT_TRANSLATION_TIMEOUT)))
                except (ValueError, TypeError):
                    timeout_sec = DEFAULT_TRANSLATION_TIMEOUT
                
                try:
                    translation_service = get_translation_service()
                    translations = await retry_translation_with_timeout(
                        translation_service,
                        payload.name,
                        "",
                        source_lang,
                        timeout=timeout_sec,
                        max_retries=3
                    )
                    logger.info(f"[CATEGORY-UPDATE] translation.done langs={list(translations.keys())} for name update")
                except asyncio.TimeoutError:
                    logger.error(f"[CATEGORY-UPDATE] translation timed out after {timeout_sec}s for name update")
                    raise HTTPException(status_code=504, detail="Translation timed out")
                except Exception as e:
                    logger.error(f"[CATEGORY-UPDATE] translation failed for name update: {e}")
                    if "IndicTrans2" in str(e) or "Model" in str(e):
                        logger.error("This appears to be an IndicTrans2 model loading issue")
                    raise
                
                # Handle bidirectional translation - ensure all three languages are present
                # If source is English, add original text as English translation
                if source_lang == "en":
                    translations["english"] = {
                        "title": payload.name,
                        "description": ""
                    }
                    logger.info(f"[CATEGORY-UPDATE] added original text as English translation for source=en")
                
                # If source is Kannada, add original text as Kannada translation  
                elif source_lang == "kn":
                    translations["kannada"] = {
                        "title": payload.name,
                        "description": ""
                    }
                    logger.info(f"[CATEGORY-UPDATE] added original text as Kannada translation for source=kn")
                
                # If source is Hindi, add original text as Hindi translation
                elif source_lang == "hi":
                    translations["hindi"] = {
                        "title": payload.name,
                        "description": ""
                    }
                    logger.info(f"[CATEGORY-UPDATE] added original text as Hindi translation for source=hi")
                
                # Update translation fields
                updates["hindi"] = translations.get("hindi", {}).get("title", payload.name)
                updates["kannada"] = translations.get("kannada", {}).get("title", payload.name)
                updates["English"] = translations.get("english", {}).get("title", payload.name)
                
                logger.info(f"[CATEGORY-UPDATE] updated translations: hindi='{updates['hindi'][:20]}...', kannada='{updates['kannada'][:20]}...', english='{updates['English'][:20]}...'")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[CATEGORY-UPDATE] translation error for name update: {e}")
                raise HTTPException(status_code=500, detail=f"Translation failed for name update: {str(e)}")
            
        if payload.description is not None:
            updates["description"] = payload.description
            
        # Handle status updates (admin/moderator only)
        if payload.status is not None:
            user_role = current_user.get("role", "")
            if user_role in ["admin", "moderator"]:
                updates["status"] = payload.status
            else:
                logger.warning(f"[CATEGORY-UPDATE] Non-admin/moderator user tried to update status: {current_user.get('email')}")
        
        # Update in DB
        success = await get_db_service().update_category_fields(ObjectId(category_id), updates)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update category")
        
        # Get updated category
        updated_category = await get_db_service().get_category_by_id(ObjectId(category_id))
        response_doc = to_extended_json(updated_category)
        
        return CategoryResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CATEGORY-UPDATE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating category: {str(e)}")

@router.delete("/{category_id}")
async def delete_category(
    category_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a category."""
    try:
        logger.info(f"[CATEGORY-DELETE] category_id={category_id}")
        
        # Check if category exists
        existing_category = await get_db_service().get_category_by_id(ObjectId(category_id))
        if not existing_category:
            raise HTTPException(status_code=404, detail="Category not found")
        
        # Delete from DB
        success = await get_db_service().delete_category(ObjectId(category_id))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete category")
        
        return {"success": True, "message": "Category deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CATEGORY-DELETE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting category: {str(e)}")
