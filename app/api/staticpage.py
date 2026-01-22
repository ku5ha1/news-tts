from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
from bson import ObjectId
import asyncio
import os
from app.models.staticpage import (
    StaticPageCreateRequest, StaticPageResponse,
    StaticPageUpdateRequest, StaticPageListResponse
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
                detail="Insufficient permissions. Only admin and moderator roles can create static pages.",
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

@router.post("/create", response_model=StaticPageResponse)
async def create_static_page(
    payload: StaticPageCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create static page with translation to Hindi, Kannada, and English."""
    staticpage_id = ObjectId()
    logger.info(f"[STATICPAGE-CREATE] start staticpage_id={staticpage_id} user={current_user.get('email', 'unknown')}")

    try:
        # Validate input
        if not payload.staticpageName.strip():
            raise HTTPException(status_code=400, detail="Static page name cannot be empty")
        if not payload.staticpageImage.strip():
            raise HTTPException(status_code=400, detail="Static page image URL cannot be empty")
        if not payload.staticpageLink.strip():
            raise HTTPException(status_code=400, detail="Static page link cannot be empty")

        # Detect source language from staticpageName
        source_lang = detect_language(payload.staticpageName)
        logger.info(f"[STATICPAGE-CREATE] detected_language={source_lang} staticpage_id={staticpage_id}")

        # Translation with timeout
        try:
            timeout_sec = float(os.getenv("TRANSLATION_PER_CALL_TIMEOUT", str(DEFAULT_TRANSLATION_TIMEOUT)))
        except (ValueError, TypeError):
            timeout_sec = DEFAULT_TRANSLATION_TIMEOUT

        try:
            translation_service = get_translation_service()
            translations = await retry_translation_with_timeout(
                translation_service,
                payload.staticpageName,
                "",
                source_lang,
                timeout=timeout_sec,
                max_retries=3
            )
            logger.info(f"[STATICPAGE-CREATE] translation.done langs={list(translations.keys())} staticpage_id={staticpage_id}")
        except asyncio.TimeoutError:
            logger.error(f"[STATICPAGE-CREATE] translation timed out after {timeout_sec}s for staticpage_id={staticpage_id}")
            raise HTTPException(status_code=504, detail="Translation timed out")
        except Exception as e:
            logger.error(f"[STATICPAGE-CREATE] translation failed for staticpage_id={staticpage_id}: {e}")
            if "IndicTrans2" in str(e) or "Model" in str(e):
                logger.error("This appears to be an IndicTrans2 model loading issue")
            raise

        # Determine status based on user role
        user_role = current_user.get("role", "")
        if user_role == "admin":
            status = "approved"
            logger.info(f"[STATICPAGE-CREATE] Admin user creating static page - status=approved staticpage_id={staticpage_id}")
        else:
            status = "pending"
            logger.info(f"[STATICPAGE-CREATE] Non-admin user creating static page - status=pending staticpage_id={staticpage_id}")

        # Create static page document
        staticpage_document = {
            "_id": staticpage_id,
            "staticpageName": payload.staticpageName,
            "staticpageImage": payload.staticpageImage,
            "staticpageLink": payload.staticpageLink,
            "createdBy": ObjectId(current_user.get("_id")) if current_user.get("_id") else None,
            "status": status,
            "createdTime": datetime.utcnow(),
            "last_updated": datetime.utcnow(),
            "hindi": translations.get("hindi", {}).get("title", payload.staticpageName),
            "kannada": translations.get("kannada", {}).get("title", payload.staticpageName),
            "English": translations.get("english", {}).get("title", payload.staticpageName),
        }

        # Insert into DB
        await asyncio.wait_for(get_db_service().insert_staticpage(staticpage_document), timeout=15.0)

        response_doc = to_extended_json(staticpage_document)
        logger.info(f"[STATICPAGE-CREATE] success staticpage_id={staticpage_id}")
        return StaticPageResponse(success=True, data=response_doc)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[STATICPAGE-CREATE] failed staticpage_id={staticpage_id} error={e}")
        try:
            await get_db_service().update_staticpage_fields(staticpage_id, {"_deleted_due_to_error": True})
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error creating static page: {str(e)}")

@router.get("/list", response_model=StaticPageListResponse)
async def list_static_pages(
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    status_filter: str = None
):
    """List static pages with pagination and optional status filter."""
    try:
        logger.info(f"[STATICPAGE-LIST] page={page} page_size={page_size} status={status_filter}")
        
        # Calculate skip
        skip = (page - 1) * page_size
        
        # Get static pages from DB
        staticpages, total = await get_db_service().get_staticpages_paginated(
            skip=skip, 
            limit=page_size, 
            status_filter=status_filter
        )
        
        # Format response
        formatted_staticpages = [to_extended_json(staticpage) for staticpage in staticpages]
        
        return StaticPageListResponse(
            success=True,
            data={"staticpages": formatted_staticpages},
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"[STATICPAGE-LIST] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing static pages: {str(e)}")

@router.get("/{staticpage_id}", response_model=StaticPageResponse)
async def get_static_page(
    staticpage_id: str
):
    """Get a specific static page by ID."""
    try:
        logger.info(f"[STATICPAGE-GET] staticpage_id={staticpage_id}")
        
        staticpage = await get_db_service().get_staticpage_by_id(ObjectId(staticpage_id))
        if not staticpage:
            raise HTTPException(status_code=404, detail="Static page not found")
        
        response_doc = to_extended_json(staticpage)
        return StaticPageResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[STATICPAGE-GET] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting static page: {str(e)}")

@router.put("/{staticpage_id}", response_model=StaticPageResponse)
async def update_static_page(
    staticpage_id: str,
    payload: StaticPageUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update a static page."""
    try:
        logger.info(f"[STATICPAGE-UPDATE] staticpage_id={staticpage_id}")
        
        # Check if static page exists
        existing_staticpage = await get_db_service().get_staticpage_by_id(ObjectId(staticpage_id))
        if not existing_staticpage:
            raise HTTPException(status_code=404, detail="Static page not found")
        
        # Prepare update fields
        updates = {}
        
        # Handle staticpageName update with translation
        if payload.staticpageName is not None:
            updates["staticpageName"] = payload.staticpageName
            
            # Re-translate the staticpageName if it's being updated
            try:
                # Detect source language
                source_lang = detect_language(payload.staticpageName)
                logger.info(f"[STATICPAGE-UPDATE] detected_language={source_lang} for staticpageName update")
                
                # Translation with timeout
                try:
                    timeout_sec = float(os.getenv("TRANSLATION_PER_CALL_TIMEOUT", str(DEFAULT_TRANSLATION_TIMEOUT)))
                except (ValueError, TypeError):
                    timeout_sec = DEFAULT_TRANSLATION_TIMEOUT
                
                try:
                    translation_service = get_translation_service()
                    translations = await retry_translation_with_timeout(
                        translation_service,
                        payload.staticpageName,
                        "",
                        source_lang,
                        timeout=timeout_sec,
                        max_retries=3
                    )
                    logger.info(f"[STATICPAGE-UPDATE] translation.done langs={list(translations.keys())} for staticpageName update")
                except asyncio.TimeoutError:
                    logger.error(f"[STATICPAGE-UPDATE] translation timed out after {timeout_sec}s for staticpageName update")
                    raise HTTPException(status_code=504, detail="Translation timed out")
                except Exception as e:
                    logger.error(f"[STATICPAGE-UPDATE] translation failed for staticpageName update: {e}")
                    if "IndicTrans2" in str(e) or "Model" in str(e):
                        logger.error("This appears to be an IndicTrans2 model loading issue")
                    raise
                
                # Handle bidirectional translation - ensure all three languages are present
                # If source is English, add original text as English translation
                if source_lang == "en":
                    translations["english"] = {
                        "title": payload.staticpageName,
                        "description": ""
                    }
                    logger.info(f"[STATICPAGE-UPDATE] added original text as English translation for source=en")
                
                # If source is Kannada, add original text as Kannada translation  
                elif source_lang == "kn":
                    translations["kannada"] = {
                        "title": payload.staticpageName,
                        "description": ""
                    }
                    logger.info(f"[STATICPAGE-UPDATE] added original text as Kannada translation for source=kn")
                
                # If source is Hindi, add original text as Hindi translation
                elif source_lang == "hi":
                    translations["hindi"] = {
                        "title": payload.staticpageName,
                        "description": ""
                    }
                    logger.info(f"[STATICPAGE-UPDATE] added original text as Hindi translation for source=hi")
                
                # Update translation fields (static pages store translations as strings, not objects)
                updates["hindi"] = translations.get("hindi", {}).get("title", payload.staticpageName)
                updates["kannada"] = translations.get("kannada", {}).get("title", payload.staticpageName)
                updates["English"] = translations.get("english", {}).get("title", payload.staticpageName)
                
                logger.info(f"[STATICPAGE-UPDATE] updated translations: hindi='{updates['hindi'][:20]}...', kannada='{updates['kannada'][:20]}...', english='{updates['English'][:20]}...'")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[STATICPAGE-UPDATE] translation error for staticpageName update: {e}")
                raise HTTPException(status_code=500, detail=f"Translation failed for staticpageName update: {str(e)}")
            
        if payload.staticpageImage is not None:
            updates["staticpageImage"] = payload.staticpageImage
            
        if payload.staticpageLink is not None:
            updates["staticpageLink"] = payload.staticpageLink
            
        # Handle status updates (admin/moderator only)
        if payload.status is not None:
            user_role = current_user.get("role", "")
            if user_role in ["admin", "moderator"]:
                updates["status"] = payload.status
            else:
                logger.warning(f"[STATICPAGE-UPDATE] Non-admin/moderator user tried to update status: {current_user.get('email')}")
        
        # Always update last_updated timestamp
        updates["last_updated"] = datetime.utcnow()
        
        # Update in DB
        success = await get_db_service().update_staticpage_fields(ObjectId(staticpage_id), updates)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update static page")
        
        # Get updated static page
        updated_staticpage = await get_db_service().get_staticpage_by_id(ObjectId(staticpage_id))
        response_doc = to_extended_json(updated_staticpage)
        
        return StaticPageResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[STATICPAGE-UPDATE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating static page: {str(e)}")

@router.delete("/{staticpage_id}")
async def delete_static_page(
    staticpage_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a static page."""
    try:
        logger.info(f"[STATICPAGE-DELETE] staticpage_id={staticpage_id}")
        
        # Check if static page exists
        existing_staticpage = await get_db_service().get_staticpage_by_id(ObjectId(staticpage_id))
        if not existing_staticpage:
            raise HTTPException(status_code=404, detail="Static page not found")
        
        # Delete from DB
        success = await get_db_service().delete_staticpage(ObjectId(staticpage_id))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete static page")
        
        return {"success": True, "message": "Static page deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[STATICPAGE-DELETE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting static page: {str(e)}")
