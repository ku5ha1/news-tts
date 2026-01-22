from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
from bson import ObjectId
import asyncio
import os
from app.models.longvideo import (
    LongVideoCreateRequest, LongVideoResponse,
    LongVideoUpdateRequest, LongVideoListResponse
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
                detail="Insufficient permissions. Only admin and moderator roles can create long videos.",
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

@router.post("/create", response_model=LongVideoResponse)
async def create_long_video(
    payload: LongVideoCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create long video with translation to Hindi, Kannada, and English."""
    video_id = ObjectId()
    logger.info(f"[LONGVIDEO-CREATE] start video_id={video_id} user={current_user.get('email', 'unknown')}")

    try:
        # Validate input
        if not payload.title.strip():
            raise HTTPException(status_code=400, detail="Title cannot be empty")
        if not payload.description.strip():
            raise HTTPException(status_code=400, detail="Description cannot be empty")
        if not payload.thumbnail.strip():
            raise HTTPException(status_code=400, detail="Thumbnail URL cannot be empty")
        if not payload.video_url.strip():
            raise HTTPException(status_code=400, detail="Video URL cannot be empty")
        if not payload.category.strip():
            raise HTTPException(status_code=400, detail="Category cannot be empty")

        # Validate category ObjectId
        try:
            category_object_id = ObjectId(payload.category)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid category ID format")

        # Validate Topics ObjectId if provided
        topics_object_id = None
        if payload.Topics:
            try:
                topics_object_id = ObjectId(payload.Topics)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid Topics ID format")

        # Detect source language from title + description
        combined_text = f"{payload.title} {payload.description}"
        source_lang = detect_language(combined_text)
        logger.info(f"[LONGVIDEO-CREATE] detected_language={source_lang} video_id={video_id}")

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
                payload.description,
                source_lang,
                timeout=timeout_sec,
                max_retries=3
            )
            logger.info(f"[LONGVIDEO-CREATE] translation.done langs={list(translations.keys())} video_id={video_id}")
        except asyncio.TimeoutError:
            logger.error(f"[LONGVIDEO-CREATE] translation timed out after {timeout_sec}s for video_id={video_id}")
            raise HTTPException(status_code=504, detail="Translation timed out")
        except Exception as e:
            logger.error(f"[LONGVIDEO-CREATE] translation failed for video_id={video_id}: {e}")
            if "IndicTrans2" in str(e) or "Model" in str(e):
                logger.error("This appears to be an IndicTrans2 model loading issue")
            raise

        # Determine status based on user role
        user_role = current_user.get("role", "")
        if user_role == "admin":
            status = "approved"
            logger.info(f"[LONGVIDEO-CREATE] Admin user creating long video - status=approved video_id={video_id}")
        else:
            status = "pending"
            logger.info(f"[LONGVIDEO-CREATE] Non-admin user creating long video - status=pending video_id={video_id}")

        # Create long video document
        long_video_document = {
            "_id": video_id,
            "title": payload.title,
            "description": payload.description,
            "thumbnail": payload.thumbnail,
            "video_url": payload.video_url,
            "category": category_object_id,
            "magazineType": payload.magazineType,
            "newsType": payload.newsType,
            "Topics": topics_object_id,
            "likedBy": [],
            "total_Likes": 0,
            "Total_views": 0,
            "Comments": [],
            "status": status,
            "createdBy": ObjectId(current_user.get("_id")) if current_user.get("_id") else None,
            "createdAt": datetime.utcnow(),
            "hindi": {
                "title": translations.get("hindi", {}).get("title", payload.title),
                "description": translations.get("hindi", {}).get("description", payload.description),
            },
            "kannada": {
                "title": translations.get("kannada", {}).get("title", payload.title),
                "description": translations.get("kannada", {}).get("description", payload.description),
            },
            "english": {
                "title": translations.get("english", {}).get("title", payload.title),
                "description": translations.get("english", {}).get("description", payload.description),
            },
        }

        # Insert into DB
        await asyncio.wait_for(get_db_service().insert_longvideo(long_video_document), timeout=15.0)

        response_doc = to_extended_json(long_video_document)
        logger.info(f"[LONGVIDEO-CREATE] success video_id={video_id}")
        return LongVideoResponse(success=True, data=response_doc)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[LONGVIDEO-CREATE] failed video_id={video_id} error={e}")
        try:
            await get_db_service().update_longvideo_fields(video_id, {"_deleted_due_to_error": True})
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error creating long video: {str(e)}")

@router.get("/list", response_model=LongVideoListResponse)
async def list_long_videos(
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    status_filter: str = None,
    category_filter: str = None
):
    """List long videos with pagination and optional filters."""
    try:
        logger.info(f"[LONGVIDEO-LIST] page={page} page_size={page_size} status={status_filter} category={category_filter}")
        
        # Calculate skip
        skip = (page - 1) * page_size
        
        # Get long videos from DB
        videos, total = await get_db_service().get_longvideos_paginated(
            skip=skip, 
            limit=page_size, 
            status_filter=status_filter,
            category_filter=category_filter
        )
        
        # Format response
        formatted_videos = [to_extended_json(video) for video in videos]
        
        return LongVideoListResponse(
            success=True,
            data={"longvideos": formatted_videos},
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"[LONGVIDEO-LIST] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing long videos: {str(e)}")

@router.get("/{video_id}", response_model=LongVideoResponse)
async def get_long_video(
    video_id: str
):
    """Get a specific long video by ID."""
    try:
        logger.info(f"[LONGVIDEO-GET] video_id={video_id}")
        
        video = await get_db_service().get_longvideo_by_id(ObjectId(video_id))
        if not video:
            raise HTTPException(status_code=404, detail="Long video not found")
        
        response_doc = to_extended_json(video)
        return LongVideoResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[LONGVIDEO-GET] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting long video: {str(e)}")

@router.put("/{video_id}", response_model=LongVideoResponse)
async def update_long_video(
    video_id: str,
    payload: LongVideoUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update a long video."""
    try:
        logger.info(f"[LONGVIDEO-UPDATE] video_id={video_id}")
        
        # Check if video exists
        existing_video = await get_db_service().get_longvideo_by_id(ObjectId(video_id))
        if not existing_video:
            raise HTTPException(status_code=404, detail="Long video not found")
        
        # Prepare update fields
        updates = {}
        
        # Handle title and description updates with translation
        if payload.title is not None or payload.description is not None:
            # Get current values for fields that aren't being updated
            current_title = payload.title if payload.title is not None else existing_video.get("title", "")
            current_description = payload.description if payload.description is not None else existing_video.get("description", "")
            
            # Update the fields
            if payload.title is not None:
                updates["title"] = payload.title
            if payload.description is not None:
                updates["description"] = payload.description
            
            # Re-translate if title or description is being updated
            try:
                # Detect source language from combined title + description
                combined_text = f"{current_title} {current_description}"
                source_lang = detect_language(combined_text)
                logger.info(f"[LONGVIDEO-UPDATE] detected_language={source_lang} for title/description update")
                
                # Translation with timeout
                try:
                    timeout_sec = float(os.getenv("TRANSLATION_PER_CALL_TIMEOUT", str(DEFAULT_TRANSLATION_TIMEOUT)))
                except (ValueError, TypeError):
                    timeout_sec = DEFAULT_TRANSLATION_TIMEOUT
                
                try:
                    translation_service = get_translation_service()
                    translations = await retry_translation_with_timeout(
                        translation_service,
                        current_title,
                        current_description,
                        source_lang,
                        timeout=timeout_sec,
                        max_retries=3
                    )
                    logger.info(f"[LONGVIDEO-UPDATE] translation.done langs={list(translations.keys())} for title/description update")
                except asyncio.TimeoutError:
                    logger.error(f"[LONGVIDEO-UPDATE] translation timed out after {timeout_sec}s for title/description update")
                    raise HTTPException(status_code=504, detail="Translation timed out")
                except Exception as e:
                    logger.error(f"[LONGVIDEO-UPDATE] translation failed for title/description update: {e}")
                    if "IndicTrans2" in str(e) or "Model" in str(e):
                        logger.error("This appears to be an IndicTrans2 model loading issue")
                    raise
                
                # Handle bidirectional translation - ensure all three languages are present
                # If source is English, add original text as English translation
                if source_lang == "en":
                    translations["english"] = {
                        "title": current_title,
                        "description": current_description
                    }
                    logger.info(f"[LONGVIDEO-UPDATE] added original text as English translation for source=en")
                
                # If source is Kannada, add original text as Kannada translation  
                elif source_lang == "kn":
                    translations["kannada"] = {
                        "title": current_title,
                        "description": current_description
                    }
                    logger.info(f"[LONGVIDEO-UPDATE] added original text as Kannada translation for source=kn")
                
                # If source is Hindi, add original text as Hindi translation
                elif source_lang == "hi":
                    translations["hindi"] = {
                        "title": current_title,
                        "description": current_description
                    }
                    logger.info(f"[LONGVIDEO-UPDATE] added original text as Hindi translation for source=hi")
                
                # Update translation fields
                updates["hindi"] = {
                    "title": translations.get("hindi", {}).get("title", current_title),
                    "description": translations.get("hindi", {}).get("description", current_description)
                }
                updates["kannada"] = {
                    "title": translations.get("kannada", {}).get("title", current_title),
                    "description": translations.get("kannada", {}).get("description", current_description)
                }
                updates["english"] = {
                    "title": translations.get("english", {}).get("title", current_title),
                    "description": translations.get("english", {}).get("description", current_description)
                }
                
                logger.info(f"[LONGVIDEO-UPDATE] updated translations for title/description update")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[LONGVIDEO-UPDATE] translation error for title/description update: {e}")
                raise HTTPException(status_code=500, detail=f"Translation failed for title/description update: {str(e)}")
            
        if payload.thumbnail is not None:
            updates["thumbnail"] = payload.thumbnail
            
        if payload.video_url is not None:
            updates["video_url"] = payload.video_url
            
        if payload.category is not None:
            try:
                updates["category"] = ObjectId(payload.category)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid category ID format")
                
        if payload.magazineType is not None:
            updates["magazineType"] = payload.magazineType
            
        if payload.newsType is not None:
            updates["newsType"] = payload.newsType
            
        if payload.Topics is not None:
            if payload.Topics:
                try:
                    updates["Topics"] = ObjectId(payload.Topics)
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid Topics ID format")
            else:
                updates["Topics"] = None
            
        # Handle status updates (admin/moderator only)
        if payload.status is not None:
            user_role = current_user.get("role", "")
            if user_role in ["admin", "moderator"]:
                updates["status"] = payload.status
            else:
                logger.warning(f"[LONGVIDEO-UPDATE] Non-admin/moderator user tried to update status: {current_user.get('email')}")
        
        # Update in DB
        success = await get_db_service().update_longvideo_fields(ObjectId(video_id), updates)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update long video")
        
        # Get updated video
        updated_video = await get_db_service().get_longvideo_by_id(ObjectId(video_id))
        response_doc = to_extended_json(updated_video)
        
        return LongVideoResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[LONGVIDEO-UPDATE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating long video: {str(e)}")

@router.delete("/{video_id}")
async def delete_long_video(
    video_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a long video."""
    try:
        logger.info(f"[LONGVIDEO-DELETE] video_id={video_id}")
        
        # Check if video exists
        existing_video = await get_db_service().get_longvideo_by_id(ObjectId(video_id))
        if not existing_video:
            raise HTTPException(status_code=404, detail="Long video not found")
        
        # Delete from DB
        success = await get_db_service().delete_longvideo(ObjectId(video_id))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete long video")
        
        return {"success": True, "message": "Long video deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[LONGVIDEO-DELETE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting long video: {str(e)}")
