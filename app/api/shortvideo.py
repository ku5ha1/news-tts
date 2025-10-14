from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
from bson import ObjectId
import asyncio
import os
from app.models.shortvideo import (
    ShortVideoCreateRequest, ShortVideoResponse,
    ShortVideoUpdateRequest, ShortVideoListResponse
)
from app.services.db_service import DBService
from app.services.auth_service import auth_service
from app.utils.language_detection import detect_language
from app.utils.retry_utils import retry_translation_with_timeout
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
                detail="Insufficient permissions. Only admin and moderator roles can create short videos.",
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

def _to_extended_json(document: dict) -> dict:
    def oidify(value):
        try:
            return {"$oid": str(ObjectId(value))}
        except Exception:
            return {"$oid": str(value)} if isinstance(value, ObjectId) else value

    def dateify(value: datetime):
        return {"$date": value.replace(microsecond=0).isoformat() + "Z"}

    # Shallow copy
    doc = dict(document)

    # ObjectId fields
    for key in ["_id", "category", "createdBy"]:
        if key in doc:
            val = doc[key]
            if isinstance(val, ObjectId) or (isinstance(val, str) and len(val) == 24):
                doc[key] = oidify(val)

    # Date fields
    for key in ["createdAt"]:
        if key in doc and isinstance(doc[key], datetime):
            doc[key] = dateify(doc[key])

    return doc

@router.post("/create", response_model=ShortVideoResponse)
async def create_short_video(
    payload: ShortVideoCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create short video with translation to Hindi, Kannada, and English."""
    video_id = ObjectId()
    logger.info(f"[SHORTVIDEO-CREATE] start video_id={video_id} user={current_user.get('email', 'unknown')}")

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

        # Detect source language from title + description
        combined_text = f"{payload.title} {payload.description}"
        source_lang = detect_language(combined_text)
        logger.info(f"[SHORTVIDEO-CREATE] detected_language={source_lang} video_id={video_id}")

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
            logger.info(f"[SHORTVIDEO-CREATE] translation.done langs={list(translations.keys())} video_id={video_id}")
        except asyncio.TimeoutError:
            logger.error(f"[SHORTVIDEO-CREATE] translation timed out after {timeout_sec}s for video_id={video_id}")
            raise HTTPException(status_code=504, detail="Translation timed out")
        except Exception as e:
            logger.error(f"[SHORTVIDEO-CREATE] translation failed for video_id={video_id}: {e}")
            if "IndicTrans2" in str(e) or "Model" in str(e):
                logger.error("This appears to be an IndicTrans2 model loading issue")
            raise

        # Determine status based on user role
        user_role = current_user.get("role", "")
        if user_role == "admin":
            status = "approved"
            logger.info(f"[SHORTVIDEO-CREATE] Admin user creating short video - status=approved video_id={video_id}")
        else:
            status = "pending"
            logger.info(f"[SHORTVIDEO-CREATE] Non-admin user creating short video - status=pending video_id={video_id}")

        # Create short video document
        short_video_document = {
            "_id": video_id,
            "title": payload.title,
            "description": payload.description,
            "thumbnail": payload.thumbnail,
            "video_url": payload.video_url,
            "category": category_object_id,
            "magazineType": payload.magazineType,
            "newsType": payload.newsType,
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
        await asyncio.wait_for(get_db_service().insert_shortvideo(short_video_document), timeout=15.0)

        response_doc = _to_extended_json(short_video_document)
        logger.info(f"[SHORTVIDEO-CREATE] success video_id={video_id}")
        return ShortVideoResponse(success=True, data=response_doc)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SHORTVIDEO-CREATE] failed video_id={video_id} error={e}")
        try:
            await get_db_service().update_shortvideo_fields(video_id, {"_deleted_due_to_error": True})
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error creating short video: {str(e)}")

@router.get("/list", response_model=ShortVideoListResponse)
async def list_short_videos(
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    status_filter: str = None,
    category_filter: str = None,
    current_user: dict = Depends(get_current_user)
):
    """List short videos with pagination and optional filters."""
    try:
        logger.info(f"[SHORTVIDEO-LIST] page={page} page_size={page_size} status={status_filter} category={category_filter}")
        
        # Calculate skip
        skip = (page - 1) * page_size
        
        # Get short videos from DB
        videos, total = await get_db_service().get_shortvideos_paginated(
            skip=skip, 
            limit=page_size, 
            status_filter=status_filter,
            category_filter=category_filter
        )
        
        # Format response
        formatted_videos = [_to_extended_json(video) for video in videos]
        
        return ShortVideoListResponse(
            success=True,
            data={"shortvideos": formatted_videos},
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"[SHORTVIDEO-LIST] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing short videos: {str(e)}")

@router.get("/{video_id}", response_model=ShortVideoResponse)
async def get_short_video(
    video_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific short video by ID."""
    try:
        logger.info(f"[SHORTVIDEO-GET] video_id={video_id}")
        
        video = await get_db_service().get_shortvideo_by_id(ObjectId(video_id))
        if not video:
            raise HTTPException(status_code=404, detail="Short video not found")
        
        response_doc = _to_extended_json(video)
        return ShortVideoResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SHORTVIDEO-GET] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting short video: {str(e)}")

@router.put("/{video_id}", response_model=ShortVideoResponse)
async def update_short_video(
    video_id: str,
    payload: ShortVideoUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update a short video."""
    try:
        logger.info(f"[SHORTVIDEO-UPDATE] video_id={video_id}")
        
        # Check if video exists
        existing_video = await get_db_service().get_shortvideo_by_id(ObjectId(video_id))
        if not existing_video:
            raise HTTPException(status_code=404, detail="Short video not found")
        
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
                logger.info(f"[SHORTVIDEO-UPDATE] detected_language={source_lang} for title/description update")
                
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
                    logger.info(f"[SHORTVIDEO-UPDATE] translation.done langs={list(translations.keys())} for title/description update")
                except asyncio.TimeoutError:
                    logger.error(f"[SHORTVIDEO-UPDATE] translation timed out after {timeout_sec}s for title/description update")
                    raise HTTPException(status_code=504, detail="Translation timed out")
                except Exception as e:
                    logger.error(f"[SHORTVIDEO-UPDATE] translation failed for title/description update: {e}")
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
                    logger.info(f"[SHORTVIDEO-UPDATE] added original text as English translation for source=en")
                
                # If source is Kannada, add original text as Kannada translation  
                elif source_lang == "kn":
                    translations["kannada"] = {
                        "title": current_title,
                        "description": current_description
                    }
                    logger.info(f"[SHORTVIDEO-UPDATE] added original text as Kannada translation for source=kn")
                
                # If source is Hindi, add original text as Hindi translation
                elif source_lang == "hi":
                    translations["hindi"] = {
                        "title": current_title,
                        "description": current_description
                    }
                    logger.info(f"[SHORTVIDEO-UPDATE] added original text as Hindi translation for source=hi")
                
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
                
                logger.info(f"[SHORTVIDEO-UPDATE] updated translations for title/description update")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[SHORTVIDEO-UPDATE] translation error for title/description update: {e}")
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
            
        # Handle status updates (admin/moderator only)
        if payload.status is not None:
            user_role = current_user.get("role", "")
            if user_role in ["admin", "moderator"]:
                updates["status"] = payload.status
            else:
                logger.warning(f"[SHORTVIDEO-UPDATE] Non-admin/moderator user tried to update status: {current_user.get('email')}")
        
        # Update in DB
        success = await get_db_service().update_shortvideo_fields(ObjectId(video_id), updates)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update short video")
        
        # Get updated video
        updated_video = await get_db_service().get_shortvideo_by_id(ObjectId(video_id))
        response_doc = _to_extended_json(updated_video)
        
        return ShortVideoResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SHORTVIDEO-UPDATE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating short video: {str(e)}")

@router.delete("/{video_id}")
async def delete_short_video(
    video_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a short video."""
    try:
        logger.info(f"[SHORTVIDEO-DELETE] video_id={video_id}")
        
        # Check if video exists
        existing_video = await get_db_service().get_shortvideo_by_id(ObjectId(video_id))
        if not existing_video:
            raise HTTPException(status_code=404, detail="Short video not found")
        
        # Delete from DB
        success = await get_db_service().delete_shortvideo(ObjectId(video_id))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete short video")
        
        return {"success": True, "message": "Short video deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SHORTVIDEO-DELETE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting short video: {str(e)}")
