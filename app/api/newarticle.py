from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
from bson import ObjectId
import asyncio
import os
from app.models.newarticle import (
    NewArticleCreateRequest, NewArticleResponse,
    NewArticleUpdateRequest, NewArticleListResponse
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
                detail="Insufficient permissions. Only admin and moderator roles can create articles.",
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
    for key in ["_id"]:
        if key in doc:
            val = doc[key]
            if isinstance(val, ObjectId) or (isinstance(val, str) and len(val) == 24):
                doc[key] = oidify(val)

    # Date fields
    for key in ["createdAt"]:
        if key in doc and isinstance(doc[key], datetime):
            doc[key] = dateify(doc[key])

    return doc

@router.post("/create", response_model=NewArticleResponse)
async def create_new_article(
    payload: NewArticleCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create new article with translation to Hindi, Kannada, and English."""
    newarticle_id = ObjectId()
    logger.info(f"[NEWARTICLE-CREATE] start newarticle_id={newarticle_id} user={current_user.get('email', 'unknown')}")

    try:
        # Validate input
        if not payload.title.strip():
            raise HTTPException(status_code=400, detail="Article title cannot be empty")
        if not payload.link.strip():
            raise HTTPException(status_code=400, detail="Article link cannot be empty")

        # Detect source language from title
        source_lang = detect_language(payload.title)
        logger.info(f"[NEWARTICLE-CREATE] detected_language={source_lang} newarticle_id={newarticle_id}")

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
            logger.info(f"[NEWARTICLE-CREATE] translation.done langs={list(translations.keys())} newarticle_id={newarticle_id}")
        except asyncio.TimeoutError:
            logger.error(f"[NEWARTICLE-CREATE] translation timed out after {timeout_sec}s for newarticle_id={newarticle_id}")
            raise HTTPException(status_code=504, detail="Translation timed out")
        except Exception as e:
            logger.error(f"[NEWARTICLE-CREATE] translation failed for newarticle_id={newarticle_id}: {e}")
            if "IndicTrans2" in str(e) or "Model" in str(e):
                logger.error("This appears to be an IndicTrans2 model loading issue")
            raise

        # Create new article document
        newarticle_document = {
            "_id": newarticle_id,
            "title": payload.title,
            "link": payload.link,
            "createdAt": datetime.utcnow(),
            "hindi": translations.get("hindi", {}).get("title", payload.title),
            "kannada": translations.get("kannada", {}).get("title", payload.title),
            "English": translations.get("english", {}).get("title", payload.title),
        }

        # Insert into DB
        await asyncio.wait_for(get_db_service().insert_newarticle(newarticle_document), timeout=15.0)

        response_doc = _to_extended_json(newarticle_document)
        logger.info(f"[NEWARTICLE-CREATE] success newarticle_id={newarticle_id}")
        return NewArticleResponse(success=True, data=response_doc)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[NEWARTICLE-CREATE] failed newarticle_id={newarticle_id} error={e}")
        try:
            await get_db_service().update_newarticle_fields(newarticle_id, {"_deleted_due_to_error": True})
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error creating new article: {str(e)}")

@router.get("/list", response_model=NewArticleListResponse)
async def list_new_articles(
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE
):
    """List new articles with pagination."""
    try:
        logger.info(f"[NEWARTICLE-LIST] page={page} page_size={page_size}")
        
        # Calculate skip
        skip = (page - 1) * page_size
        
        # Get new articles from DB
        newarticles, total = await get_db_service().get_newarticles_paginated(
            skip=skip, 
            limit=page_size
        )
        
        # Format response
        formatted_newarticles = [_to_extended_json(article) for article in newarticles]
        
        return NewArticleListResponse(
            success=True,
            data={"newarticles": formatted_newarticles},
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"[NEWARTICLE-LIST] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing new articles: {str(e)}")

@router.get("/{newarticle_id}", response_model=NewArticleResponse)
async def get_new_article(
    newarticle_id: str
):
    """Get a specific new article by ID."""
    try:
        logger.info(f"[NEWARTICLE-GET] newarticle_id={newarticle_id}")
        
        newarticle = await get_db_service().get_newarticle_by_id(ObjectId(newarticle_id))
        if not newarticle:
            raise HTTPException(status_code=404, detail="New article not found")
        
        response_doc = _to_extended_json(newarticle)
        return NewArticleResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[NEWARTICLE-GET] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting new article: {str(e)}")

@router.put("/{newarticle_id}", response_model=NewArticleResponse)
async def update_new_article(
    newarticle_id: str,
    payload: NewArticleUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update a new article."""
    try:
        logger.info(f"[NEWARTICLE-UPDATE] newarticle_id={newarticle_id}")
        
        # Check if new article exists
        existing_newarticle = await get_db_service().get_newarticle_by_id(ObjectId(newarticle_id))
        if not existing_newarticle:
            raise HTTPException(status_code=404, detail="New article not found")
        
        # Prepare update fields
        updates = {}
        
        # Handle title update with translation
        if payload.title is not None:
            updates["title"] = payload.title
            
            # Re-translate the title if it's being updated
            try:
                # Detect source language
                source_lang = detect_language(payload.title)
                logger.info(f"[NEWARTICLE-UPDATE] detected_language={source_lang} for title update")
                
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
                    logger.info(f"[NEWARTICLE-UPDATE] translation.done langs={list(translations.keys())} for title update")
                except asyncio.TimeoutError:
                    logger.error(f"[NEWARTICLE-UPDATE] translation timed out after {timeout_sec}s for title update")
                    raise HTTPException(status_code=504, detail="Translation timed out")
                except Exception as e:
                    logger.error(f"[NEWARTICLE-UPDATE] translation failed for title update: {e}")
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
                    logger.info(f"[NEWARTICLE-UPDATE] added original text as English translation for source=en")
                
                # If source is Kannada, add original text as Kannada translation  
                elif source_lang == "kn":
                    translations["kannada"] = {
                        "title": payload.title,
                        "description": ""
                    }
                    logger.info(f"[NEWARTICLE-UPDATE] added original text as Kannada translation for source=kn")
                
                # If source is Hindi, add original text as Hindi translation
                elif source_lang == "hi":
                    translations["hindi"] = {
                        "title": payload.title,
                        "description": ""
                    }
                    logger.info(f"[NEWARTICLE-UPDATE] added original text as Hindi translation for source=hi")
                
                # Update translation fields
                updates["hindi"] = translations.get("hindi", {}).get("title", payload.title)
                updates["kannada"] = translations.get("kannada", {}).get("title", payload.title)
                updates["English"] = translations.get("english", {}).get("title", payload.title)
                
                logger.info(f"[NEWARTICLE-UPDATE] updated translations: hindi='{updates['hindi'][:20]}...', kannada='{updates['kannada'][:20]}...', english='{updates['English'][:20]}...'")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[NEWARTICLE-UPDATE] translation error for title update: {e}")
                raise HTTPException(status_code=500, detail=f"Translation failed for title update: {str(e)}")
            
        if payload.link is not None:
            updates["link"] = payload.link
        
        # Update in DB
        success = await get_db_service().update_newarticle_fields(ObjectId(newarticle_id), updates)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update new article")
        
        # Get updated new article
        updated_newarticle = await get_db_service().get_newarticle_by_id(ObjectId(newarticle_id))
        response_doc = _to_extended_json(updated_newarticle)
        
        return NewArticleResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[NEWARTICLE-UPDATE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating new article: {str(e)}")

@router.delete("/{newarticle_id}")
async def delete_new_article(
    newarticle_id: str
):
    """Delete a new article."""
    try:
        logger.info(f"[NEWARTICLE-DELETE] newarticle_id={newarticle_id}")
        
        # Check if new article exists
        existing_newarticle = await get_db_service().get_newarticle_by_id(ObjectId(newarticle_id))
        if not existing_newarticle:
            raise HTTPException(status_code=404, detail="New article not found")
        
        # Delete from DB
        success = await get_db_service().delete_newarticle(ObjectId(newarticle_id))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete new article")
        
        return {"success": True, "message": "New article deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[NEWARTICLE-DELETE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting new article: {str(e)}")

