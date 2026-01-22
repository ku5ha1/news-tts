from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
from bson import ObjectId
import asyncio
import os
from app.models.latestnotification import (
    LatestNotificationCreateRequest, LatestNotificationResponse,
    LatestNotificationUpdateRequest, LatestNotificationListResponse
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
                detail="Insufficient permissions. Only admin and moderator roles can create notifications.",
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

@router.post("/create", response_model=LatestNotificationResponse)
async def create_latest_notification(
    payload: LatestNotificationCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create latest notification with translation to Hindi, Kannada, and English."""
    latestnotification_id = ObjectId()
    logger.info(f"[LATESTNOTIFICATION-CREATE] start latestnotification_id={latestnotification_id} user={current_user.get('email', 'unknown')}")

    try:
        # Validate input
        if not payload.title.strip():
            raise HTTPException(status_code=400, detail="Notification title cannot be empty")
        if not payload.link.strip():
            raise HTTPException(status_code=400, detail="Notification link cannot be empty")

        # Detect source language from title
        source_lang = detect_language(payload.title)
        logger.info(f"[LATESTNOTIFICATION-CREATE] detected_language={source_lang} latestnotification_id={latestnotification_id}")

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
            logger.info(f"[LATESTNOTIFICATION-CREATE] translation.done langs={list(translations.keys())} latestnotification_id={latestnotification_id}")
        except asyncio.TimeoutError:
            logger.error(f"[LATESTNOTIFICATION-CREATE] translation timed out after {timeout_sec}s for latestnotification_id={latestnotification_id}")
            raise HTTPException(status_code=504, detail="Translation timed out")
        except Exception as e:
            logger.error(f"[LATESTNOTIFICATION-CREATE] translation failed for latestnotification_id={latestnotification_id}: {e}")
            if "IndicTrans2" in str(e) or "Model" in str(e):
                logger.error("This appears to be an IndicTrans2 model loading issue")
            raise

        # Create latest notification document
        latestnotification_document = {
            "_id": latestnotification_id,
            "title": payload.title,
            "link": payload.link,
            "createdAt": datetime.utcnow(),
            "hindi": translations.get("hindi", {}).get("title", payload.title),
            "kannada": translations.get("kannada", {}).get("title", payload.title),
            "English": translations.get("english", {}).get("title", payload.title),
        }

        # Insert into DB
        await asyncio.wait_for(get_db_service().insert_latestnotification(latestnotification_document), timeout=15.0)

        response_doc = to_extended_json(latestnotification_document)
        logger.info(f"[LATESTNOTIFICATION-CREATE] success latestnotification_id={latestnotification_id}")
        return LatestNotificationResponse(success=True, data=response_doc)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[LATESTNOTIFICATION-CREATE] failed latestnotification_id={latestnotification_id} error={e}")
        try:
            await get_db_service().update_latestnotification_fields(latestnotification_id, {"_deleted_due_to_error": True})
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error creating latest notification: {str(e)}")

@router.get("/list", response_model=LatestNotificationListResponse)
async def list_latest_notifications(
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE
):
    """List latest notifications with pagination."""
    try:
        logger.info(f"[LATESTNOTIFICATION-LIST] page={page} page_size={page_size}")
        
        # Calculate skip
        skip = (page - 1) * page_size
        
        # Get latest notifications from DB
        latestnotifications, total = await get_db_service().get_latestnotifications_paginated(
            skip=skip, 
            limit=page_size
        )
        
        # Format response
        formatted_latestnotifications = [to_extended_json(notification) for notification in latestnotifications]
        
        return LatestNotificationListResponse(
            success=True,
            data={"latestnotifications": formatted_latestnotifications},
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"[LATESTNOTIFICATION-LIST] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing latest notifications: {str(e)}")

@router.get("/{latestnotification_id}", response_model=LatestNotificationResponse)
async def get_latest_notification(
    latestnotification_id: str
):
    """Get a specific latest notification by ID."""
    try:
        logger.info(f"[LATESTNOTIFICATION-GET] latestnotification_id={latestnotification_id}")
        
        latestnotification = await get_db_service().get_latestnotification_by_id(ObjectId(latestnotification_id))
        if not latestnotification:
            raise HTTPException(status_code=404, detail="Latest notification not found")
        
        response_doc = to_extended_json(latestnotification)
        return LatestNotificationResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[LATESTNOTIFICATION-GET] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting latest notification: {str(e)}")

@router.put("/{latestnotification_id}", response_model=LatestNotificationResponse)
async def update_latest_notification(
    latestnotification_id: str,
    payload: LatestNotificationUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update a latest notification."""
    try:
        logger.info(f"[LATESTNOTIFICATION-UPDATE] latestnotification_id={latestnotification_id}")
        
        # Check if latest notification exists
        existing_latestnotification = await get_db_service().get_latestnotification_by_id(ObjectId(latestnotification_id))
        if not existing_latestnotification:
            raise HTTPException(status_code=404, detail="Latest notification not found")
        
        # Prepare update fields
        updates = {}
        
        # Handle title update with translation
        if payload.title is not None:
            updates["title"] = payload.title
            
            # Re-translate the title if it's being updated
            try:
                # Detect source language
                source_lang = detect_language(payload.title)
                logger.info(f"[LATESTNOTIFICATION-UPDATE] detected_language={source_lang} for title update")
                
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
                    logger.info(f"[LATESTNOTIFICATION-UPDATE] translation.done langs={list(translations.keys())} for title update")
                except asyncio.TimeoutError:
                    logger.error(f"[LATESTNOTIFICATION-UPDATE] translation timed out after {timeout_sec}s for title update")
                    raise HTTPException(status_code=504, detail="Translation timed out")
                except Exception as e:
                    logger.error(f"[LATESTNOTIFICATION-UPDATE] translation failed for title update: {e}")
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
                    logger.info(f"[LATESTNOTIFICATION-UPDATE] added original text as English translation for source=en")
                
                # If source is Kannada, add original text as Kannada translation  
                elif source_lang == "kn":
                    translations["kannada"] = {
                        "title": payload.title,
                        "description": ""
                    }
                    logger.info(f"[LATESTNOTIFICATION-UPDATE] added original text as Kannada translation for source=kn")
                
                # If source is Hindi, add original text as Hindi translation
                elif source_lang == "hi":
                    translations["hindi"] = {
                        "title": payload.title,
                        "description": ""
                    }
                    logger.info(f"[LATESTNOTIFICATION-UPDATE] added original text as Hindi translation for source=hi")
                
                # Update translation fields
                updates["hindi"] = translations.get("hindi", {}).get("title", payload.title)
                updates["kannada"] = translations.get("kannada", {}).get("title", payload.title)
                updates["English"] = translations.get("english", {}).get("title", payload.title)
                
                logger.info(f"[LATESTNOTIFICATION-UPDATE] updated translations: hindi='{updates['hindi'][:20]}...', kannada='{updates['kannada'][:20]}...', english='{updates['English'][:20]}...'")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[LATESTNOTIFICATION-UPDATE] translation error for title update: {e}")
                raise HTTPException(status_code=500, detail=f"Translation failed for title update: {str(e)}")
            
        if payload.link is not None:
            updates["link"] = payload.link
        
        # Update in DB
        success = await get_db_service().update_latestnotification_fields(ObjectId(latestnotification_id), updates)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update latest notification")
        
        # Get updated latest notification
        updated_latestnotification = await get_db_service().get_latestnotification_by_id(ObjectId(latestnotification_id))
        response_doc = to_extended_json(updated_latestnotification)
        
        return LatestNotificationResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[LATESTNOTIFICATION-UPDATE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating latest notification: {str(e)}")

@router.delete("/{latestnotification_id}")
async def delete_latest_notification(
    latestnotification_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a latest notification."""
    try:
        logger.info(f"[LATESTNOTIFICATION-DELETE] latestnotification_id={latestnotification_id}")
        
        # Check if latest notification exists
        existing_latestnotification = await get_db_service().get_latestnotification_by_id(ObjectId(latestnotification_id))
        if not existing_latestnotification:
            raise HTTPException(status_code=404, detail="Latest notification not found")
        
        # Delete from DB
        success = await get_db_service().delete_latestnotification(ObjectId(latestnotification_id))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete latest notification")
        
        return {"success": True, "message": "Latest notification deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[LATESTNOTIFICATION-DELETE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting latest notification: {str(e)}")
