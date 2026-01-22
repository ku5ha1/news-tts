from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
from bson import ObjectId
import asyncio
import os
from app.models.photo import (
    PhotoCreateRequest, PhotoResponse,
    PhotoUpdateRequest, PhotoListResponse
)
from app.services.db_singleton import get_db_service
from app.services.auth_service import auth_service
from app.utils.language_detection import detect_language
from app.utils.retry_utils import retry_translation_with_timeout
from app.utils.json_encoder import to_extended_json
import logging

logger = logging.getLogger(__name__)

DEFAULT_TRANSLATION_TIMEOUT = 90.0
DEFAULT_PAGE_SIZE = 20

router = APIRouter()

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
                detail="Insufficient permissions. Only admin and moderator roles can create photos.",
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

@router.post("/create", response_model=PhotoResponse)
async def create_photo(
    payload: PhotoCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create photo with translation to Hindi, Kannada, and English."""
    photo_id = ObjectId()
    logger.info(f"[PHOTO-CREATE] start photo_id={photo_id} user={current_user.get('email', 'unknown')}")

    try:
        # Validate input
        if not payload.title.strip():
            raise HTTPException(status_code=400, detail="Title cannot be empty")
        if not payload.photoImage.strip():
            raise HTTPException(status_code=400, detail="Photo image URL cannot be empty")
        if not payload.category.strip():
            raise HTTPException(status_code=400, detail="Category cannot be empty")
        
        # Validate category ObjectId
        try:
            category_object_id = ObjectId(payload.category)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid category ID format")

        # Detect source language from title
        source_lang = detect_language(payload.title)
        logger.info(f"[PHOTO-CREATE] detected_language={source_lang} photo_id={photo_id}")

        # Translation with timeout
        try:
            timeout_sec = float(os.getenv("TRANSLATION_PER_CALL_TIMEOUT", str(DEFAULT_TRANSLATION_TIMEOUT)))
        except (ValueError, TypeError):
            timeout_sec = DEFAULT_TRANSLATION_TIMEOUT

        try:
            translation_service = get_translation_service()
            translations = await retry_translation_with_timeout(
                translation_service,
                payload.title,
                "",
                source_lang,
                timeout=timeout_sec,
                max_retries=3
            )
            logger.info(f"[PHOTO-CREATE] translation.done langs={list(translations.keys())} photo_id={photo_id}")
        except asyncio.TimeoutError:
            logger.error(f"[PHOTO-CREATE] translation timed out after {timeout_sec}s for photo_id={photo_id}")
            raise HTTPException(status_code=504, detail="Translation timed out")
        except Exception as e:
            logger.error(f"[PHOTO-CREATE] translation failed for photo_id={photo_id}: {e}")
            if "IndicTrans2" in str(e) or "Model" in str(e):
                logger.error("This appears to be an IndicTrans2 model loading issue")
            raise

        # Determine status based on user role
        user_role = current_user.get("role", "")
        if user_role == "admin":
            status = "approved"
            logger.info(f"[PHOTO-CREATE] Admin user creating photo - status=approved photo_id={photo_id}")
        else:
            status = "pending"
            logger.info(f"[PHOTO-CREATE] Non-admin user creating photo - status=pending photo_id={photo_id}")

        # Create photo document
        photo_document = {
            "_id": photo_id,
            "title": payload.title,
            "photoImage": payload.photoImage,
            "category": category_object_id,
            "createdBy": ObjectId(current_user.get("_id")) if current_user.get("_id") else None,
            "status": status,
            "createdTime": datetime.utcnow(),
            "hindi": translations.get("hindi", {}).get("title", payload.title),
            "kannada": translations.get("kannada", {}).get("title", payload.title),
            "english": translations.get("english", {}).get("title", payload.title),
        }

        # Insert into DB
        await asyncio.wait_for(get_db_service().insert_photo(photo_document), timeout=15.0)

        response_doc = to_extended_json(photo_document)
        logger.info(f"[PHOTO-CREATE] success photo_id={photo_id}")
        return PhotoResponse(success=True, data=response_doc)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PHOTO-CREATE] failed photo_id={photo_id} error={e}")
        try:
            await get_db_service().update_photo_fields(photo_id, {"_deleted_due_to_error": True})
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error creating photo: {str(e)}")

@router.get("/list", response_model=PhotoListResponse)
async def list_photos(
    status_filter: str = None,
    category_filter: str = None
):
    """List all photos with optional filters."""
    try:
        logger.info(f"[PHOTO-LIST] status={status_filter} category={category_filter}")
        
        # Get photos from DB with proper pagination
        page_size = 1000  # Cap at 1000 for performance
        photos, total = await get_db_service().get_photos_paginated(
            skip=0, 
            limit=page_size,
            status_filter=status_filter,
            category_filter=category_filter
        )
        
        # Format response
        formatted_photos = [to_extended_json(photo) for photo in photos]
        
        return PhotoListResponse(
            success=True,
            data={"photos": formatted_photos},
            total=total,
            page=1,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"[PHOTO-LIST] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing photos: {str(e)}")

@router.get("/{photo_id}", response_model=PhotoResponse)
async def get_photo(
    photo_id: str
):
    """Get a specific photo by ID."""
    try:
        logger.info(f"[PHOTO-GET] photo_id={photo_id}")
        
        photo = await get_db_service().get_photo_by_id(ObjectId(photo_id))
        if not photo:
            raise HTTPException(status_code=404, detail="Photo not found")
        
        response_doc = to_extended_json(photo)
        return PhotoResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PHOTO-GET] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting photo: {str(e)}")

@router.put("/{photo_id}", response_model=PhotoResponse)
async def update_photo(
    photo_id: str,
    payload: PhotoUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update a photo."""
    try:
        logger.info(f"[PHOTO-UPDATE] photo_id={photo_id}")
        
        # Check if photo exists
        existing_photo = await get_db_service().get_photo_by_id(ObjectId(photo_id))
        if not existing_photo:
            raise HTTPException(status_code=404, detail="Photo not found")
        
        # Prepare update fields
        updates = {}
        
        # Handle title update with translation
        if payload.title is not None:
            updates["title"] = payload.title
            
            # Re-translate the title if it's being updated
            try:
                # Detect source language
                source_lang = detect_language(payload.title)
                logger.info(f"[PHOTO-UPDATE] detected_language={source_lang} for title update")
                
                # Translation with timeout
                try:
                    timeout_sec = float(os.getenv("TRANSLATION_PER_CALL_TIMEOUT", str(DEFAULT_TRANSLATION_TIMEOUT)))
                except (ValueError, TypeError):
                    timeout_sec = DEFAULT_TRANSLATION_TIMEOUT
                
                try:
                    translation_service = get_translation_service()
                    translations = await retry_translation_with_timeout(
                        translation_service,
                        payload.title,
                        "",
                        source_lang,
                        timeout=timeout_sec,
                        max_retries=3
                    )
                    logger.info(f"[PHOTO-UPDATE] translation.done langs={list(translations.keys())} for title update")
                except asyncio.TimeoutError:
                    logger.error(f"[PHOTO-UPDATE] translation timed out after {timeout_sec}s for title update")
                    raise HTTPException(status_code=504, detail="Translation timed out")
                except Exception as e:
                    logger.error(f"[PHOTO-UPDATE] translation failed for title update: {e}")
                    if "IndicTrans2" in str(e) or "Model" in str(e):
                        logger.error("This appears to be an IndicTrans2 model loading issue")
                    raise
                
                # Handle bidirectional translation - ensure all three languages are present
                # If source is English, add original text as English translation
                if source_lang == "en":
                    translations["english"] = {
                        "title": payload.title,
                        "description": ""
                    }
                    logger.info(f"[PHOTO-UPDATE] added original text as English translation for source=en")
                
                # If source is Kannada, add original text as Kannada translation  
                elif source_lang == "kn":
                    translations["kannada"] = {
                        "title": payload.title,
                        "description": ""
                    }
                    logger.info(f"[PHOTO-UPDATE] added original text as Kannada translation for source=kn")
                
                # If source is Hindi, add original text as Hindi translation
                elif source_lang == "hi":
                    translations["hindi"] = {
                        "title": payload.title,
                        "description": ""
                    }
                    logger.info(f"[PHOTO-UPDATE] added original text as Hindi translation for source=hi")
                
                # Update translation fields (photos store translations as strings, not objects)
                updates["hindi"] = translations.get("hindi", {}).get("title", payload.title)
                updates["kannada"] = translations.get("kannada", {}).get("title", payload.title)
                updates["english"] = translations.get("english", {}).get("title", payload.title)
                
                logger.info(f"[PHOTO-UPDATE] updated translations: hindi='{updates['hindi'][:20]}...', kannada='{updates['kannada'][:20]}...', english='{updates['english'][:20]}...'")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[PHOTO-UPDATE] translation error for title update: {e}")
                raise HTTPException(status_code=500, detail=f"Translation failed for title update: {str(e)}")
            
        if payload.photoImage is not None:
            updates["photoImage"] = payload.photoImage
            
        if payload.category is not None:
            try:
                updates["category"] = ObjectId(payload.category)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid category ID format")
            
        # Handle status updates (admin/moderator only)
        if payload.status is not None:
            user_role = current_user.get("role", "")
            if user_role in ["admin", "moderator"]:
                updates["status"] = payload.status
            else:
                logger.warning(f"[PHOTO-UPDATE] Non-admin/moderator user tried to update status: {current_user.get('email')}")
        
        # Update in DB
        success = await get_db_service().update_photo_fields(ObjectId(photo_id), updates)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update photo")
        
        # Get updated photo
        updated_photo = await get_db_service().get_photo_by_id(ObjectId(photo_id))
        response_doc = to_extended_json(updated_photo)
        
        return PhotoResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PHOTO-UPDATE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating photo: {str(e)}")

@router.delete("/{photo_id}")
async def delete_photo(
    photo_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a photo."""
    try:
        logger.info(f"[PHOTO-DELETE] photo_id={photo_id}")
        
        # Check if photo exists
        existing_photo = await get_db_service().get_photo_by_id(ObjectId(photo_id))
        if not existing_photo:
            raise HTTPException(status_code=404, detail="Photo not found")
        
        # Delete from DB
        success = await get_db_service().delete_photo(ObjectId(photo_id))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete photo")
        
        return {"success": True, "message": "Photo deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PHOTO-DELETE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting photo: {str(e)}")
