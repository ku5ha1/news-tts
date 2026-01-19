from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
from bson import ObjectId
import asyncio
from app.models.video_category import (
    VideoCategoryCreateRequest, VideoCategoryResponse,
    VideoCategoryUpdateRequest, VideoCategoryListResponse
)
from app.services.db_service import DBService
from app.services.auth_service import auth_service
from app.utils.language_detection import detect_language
from app.utils.retry_utils import retry_translation_with_timeout
from app.utils.json_encoder import to_extended_json
import logging
import os

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
                detail="Insufficient permissions. Only admin and moderator roles can manage video categories.",
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

@router.post("/create", response_model=VideoCategoryResponse)
async def create_video_category(
    payload: VideoCategoryCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create video category with translation to Hindi, Kannada, and English."""
    category_id = ObjectId()
    logger.info(f"[VIDEO-CATEGORY-CREATE] start category_id={category_id} user={current_user.get('email', 'unknown')}")

    try:
        # Validate input
        if not payload.category_name.strip():
            raise HTTPException(status_code=400, detail="Category name cannot be empty")

        # Detect source language from category_name
        source_lang = detect_language(payload.category_name)
        logger.info(f"[VIDEO-CATEGORY-CREATE] detected_language={source_lang} category_id={category_id}")

        # Translation with timeout
        try:
            timeout_sec = float(os.getenv("TRANSLATION_PER_CALL_TIMEOUT", str(DEFAULT_TRANSLATION_TIMEOUT)))
        except (ValueError, TypeError):
            timeout_sec = DEFAULT_TRANSLATION_TIMEOUT

        try:
            translation_service = get_translation_service()
            translations = await retry_translation_with_timeout(
                translation_service,
                payload.category_name,
                "",
                source_lang,
                timeout=timeout_sec,
                max_retries=3
            )
            logger.info(f"[VIDEO-CATEGORY-CREATE] translation.done langs={list(translations.keys())} category_id={category_id}")
        except asyncio.TimeoutError:
            logger.error(f"[VIDEO-CATEGORY-CREATE] translation timed out after {timeout_sec}s for category_id={category_id}")
            raise HTTPException(status_code=504, detail="Translation timed out")
        except Exception as e:
            logger.error(f"[VIDEO-CATEGORY-CREATE] translation failed for category_id={category_id}: {e}")
            if "IndicTrans2" in str(e) or "Model" in str(e):
                logger.error("This appears to be an IndicTrans2 model loading issue")
            raise

        # Create video category document
        category_document = {
            "_id": category_id,
            "category_name": payload.category_name,
            "date_created": datetime.utcnow(),
            "created_by": ObjectId(current_user.get("_id")) if current_user.get("_id") else None,
            "hindi": translations.get("hindi", {}).get("title", payload.category_name),
            "kannada": translations.get("kannada", {}).get("title", payload.category_name),
            "english": translations.get("english", {}).get("title", payload.category_name),
        }

        # Insert into DB
        await asyncio.wait_for(get_db_service().insert_video_category(category_document), timeout=15.0)

        response_doc = to_extended_json(category_document)
        logger.info(f"[VIDEO-CATEGORY-CREATE] success category_id={category_id}")
        return VideoCategoryResponse(success=True, data=response_doc)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[VIDEO-CATEGORY-CREATE] failed category_id={category_id} error={e}")
        try:
            await get_db_service().update_video_category_fields(category_id, {"_deleted_due_to_error": True})
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error creating video category: {str(e)}")

@router.get("/list", response_model=VideoCategoryListResponse)
async def list_video_categories(
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    current_user: dict = Depends(get_current_user)
):
    """List video categories with pagination."""
    try:
        logger.info(f"[VIDEO-CATEGORY-LIST] page={page} page_size={page_size}")
        
        # Calculate skip
        skip = (page - 1) * page_size
        
        # Get video categories from DB
        categories, total = await get_db_service().get_video_categories_paginated(
            skip=skip, 
            limit=page_size
        )
        
        # Format response
        formatted_categories = [to_extended_json(category) for category in categories]
        
        return VideoCategoryListResponse(
            success=True,
            data={"video_categories": formatted_categories},
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"[VIDEO-CATEGORY-LIST] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing video categories: {str(e)}")

@router.get("/{category_id}", response_model=VideoCategoryResponse)
async def get_video_category(
    category_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific video category by ID."""
    try:
        logger.info(f"[VIDEO-CATEGORY-GET] category_id={category_id}")
        
        category = await get_db_service().get_video_category_by_id(ObjectId(category_id))
        if not category:
            raise HTTPException(status_code=404, detail="Video category not found")
        
        response_doc = to_extended_json(category)
        return VideoCategoryResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[VIDEO-CATEGORY-GET] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting video category: {str(e)}")

@router.put("/{category_id}", response_model=VideoCategoryResponse)
async def update_video_category(
    category_id: str,
    payload: VideoCategoryUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update a video category."""
    try:
        logger.info(f"[VIDEO-CATEGORY-UPDATE] category_id={category_id}")
        
        # Check if video category exists
        existing_category = await get_db_service().get_video_category_by_id(ObjectId(category_id))
        if not existing_category:
            raise HTTPException(status_code=404, detail="Video category not found")
        
        # Prepare update fields
        updates = {}
        
        # Handle category_name update with translation
        if payload.category_name is not None:
            updates["category_name"] = payload.category_name
            
            # Re-translate the category_name if it's being updated
            try:
                # Detect source language
                source_lang = detect_language(payload.category_name)
                logger.info(f"[VIDEO-CATEGORY-UPDATE] detected_language={source_lang} for category_name update")
                
                # Translation with timeout
                try:
                    timeout_sec = float(os.getenv("TRANSLATION_PER_CALL_TIMEOUT", str(DEFAULT_TRANSLATION_TIMEOUT)))
                except (ValueError, TypeError):
                    timeout_sec = DEFAULT_TRANSLATION_TIMEOUT
                
                try:
                    translation_service = get_translation_service()
                    translations = await retry_translation_with_timeout(
                        translation_service,
                        payload.category_name,
                        "",
                        source_lang,
                        timeout=timeout_sec,
                        max_retries=3
                    )
                    logger.info(f"[VIDEO-CATEGORY-UPDATE] translation.done langs={list(translations.keys())} for category_name update")
                except asyncio.TimeoutError:
                    logger.error(f"[VIDEO-CATEGORY-UPDATE] translation timed out after {timeout_sec}s for category_name update")
                    raise HTTPException(status_code=504, detail="Translation timed out")
                except Exception as e:
                    logger.error(f"[VIDEO-CATEGORY-UPDATE] translation failed for category_name update: {e}")
                    if "IndicTrans2" in str(e) or "Model" in str(e):
                        logger.error("This appears to be an IndicTrans2 model loading issue")
                    raise
                
                # Handle bidirectional translation - ensure all three languages are present
                # If source is English, add original text as English translation
                if source_lang == "en":
                    translations["english"] = {
                        "title": payload.category_name,
                        "description": ""
                    }
                    logger.info(f"[VIDEO-CATEGORY-UPDATE] added original text as English translation for source=en")
                
                # If source is Kannada, add original text as Kannada translation  
                elif source_lang == "kn":
                    translations["kannada"] = {
                        "title": payload.category_name,
                        "description": ""
                    }
                    logger.info(f"[VIDEO-CATEGORY-UPDATE] added original text as Kannada translation for source=kn")
                
                # If source is Hindi, add original text as Hindi translation
                elif source_lang == "hi":
                    translations["hindi"] = {
                        "title": payload.category_name,
                        "description": ""
                    }
                    logger.info(f"[VIDEO-CATEGORY-UPDATE] added original text as Hindi translation for source=hi")
                
                # Update translation fields (video categories store translations as strings, not objects)
                updates["hindi"] = translations.get("hindi", {}).get("title", payload.category_name)
                updates["kannada"] = translations.get("kannada", {}).get("title", payload.category_name)
                updates["english"] = translations.get("english", {}).get("title", payload.category_name)
                
                logger.info(f"[VIDEO-CATEGORY-UPDATE] updated translations: hindi='{updates['hindi'][:20]}...', kannada='{updates['kannada'][:20]}...', english='{updates['english'][:20]}...'")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[VIDEO-CATEGORY-UPDATE] translation error for category_name update: {e}")
                raise HTTPException(status_code=500, detail=f"Translation failed for category_name update: {str(e)}")
        
        # Update in DB
        success = await get_db_service().update_video_category_fields(ObjectId(category_id), updates)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update video category")
        
        # Get updated video category
        updated_category = await get_db_service().get_video_category_by_id(ObjectId(category_id))
        response_doc = to_extended_json(updated_category)
        
        return VideoCategoryResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[VIDEO-CATEGORY-UPDATE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating video category: {str(e)}")

@router.delete("/{category_id}")
async def delete_video_category(
    category_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a video category."""
    try:
        logger.info(f"[VIDEO-CATEGORY-DELETE] category_id={category_id}")
        
        # Check if video category exists
        existing_category = await get_db_service().get_video_category_by_id(ObjectId(category_id))
        if not existing_category:
            raise HTTPException(status_code=404, detail="Video category not found")
        
        # Delete from DB
        success = await get_db_service().delete_video_category(ObjectId(category_id))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete video category")
        
        return {"success": True, "message": "Video category deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[VIDEO-CATEGORY-DELETE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting video category: {str(e)}")