from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
from bson import ObjectId
import asyncio
import re
from app.models.district import (
    DistrictCreateRequest, DistrictResponse,
    DistrictUpdateRequest, DistrictListResponse
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

def generate_slug(text: str) -> str:
    """Generate URL-friendly slug from district name."""
    # Convert to lowercase and replace spaces with hyphens
    slug = text.lower().strip()
    # Remove special characters except hyphens and alphanumeric
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    # Replace multiple spaces/hyphens with single hyphen
    slug = re.sub(r'[\s-]+', '-', slug)
    # Remove leading/trailing hyphens
    slug = slug.strip('-')
    return slug

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
                detail="Insufficient permissions. Only admin and moderator roles can manage districts.",
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

@router.post("/create", response_model=DistrictResponse)
async def create_district(
    payload: DistrictCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create district with translation to Hindi, Kannada, and English."""
    district_id = ObjectId()
    logger.info(f"[DISTRICT-CREATE] start district_id={district_id} user={current_user.get('email', 'unknown')}")

    try:
        # Validate input
        if not payload.district_name.strip():
            raise HTTPException(status_code=400, detail="District name cannot be empty")

        # Generate slug
        district_slug = generate_slug(payload.district_name)
        if not district_slug:
            raise HTTPException(status_code=400, detail="Invalid district name for URL generation")

        # Check if slug already exists
        existing_district = await get_db_service().get_district_by_slug(district_slug)
        if existing_district:
            raise HTTPException(status_code=400, detail=f"District with similar name already exists: {existing_district.get('district_name')}")

        # Detect source language from district_name
        source_lang = detect_language(payload.district_name)
        logger.info(f"[DISTRICT-CREATE] detected_language={source_lang} district_id={district_id}")

        # Translation with timeout
        try:
            timeout_sec = float(os.getenv("TRANSLATION_PER_CALL_TIMEOUT", str(DEFAULT_TRANSLATION_TIMEOUT)))
        except (ValueError, TypeError):
            timeout_sec = DEFAULT_TRANSLATION_TIMEOUT

        try:
            translation_service = get_translation_service()
            translations = await retry_translation_with_timeout(
                translation_service,
                payload.district_name,
                "",
                source_lang,
                timeout=timeout_sec,
                max_retries=3
            )
            logger.info(f"[DISTRICT-CREATE] translation.done langs={list(translations.keys())} district_id={district_id}")
        except asyncio.TimeoutError:
            logger.error(f"[DISTRICT-CREATE] translation timed out after {timeout_sec}s for district_id={district_id}")
            raise HTTPException(status_code=504, detail="Translation timed out")
        except Exception as e:
            logger.error(f"[DISTRICT-CREATE] translation failed for district_id={district_id}: {e}")
            if "IndicTrans2" in str(e) or "Model" in str(e):
                logger.error("This appears to be an IndicTrans2 model loading issue")
            raise

        # Create district document
        district_document = {
            "_id": district_id,
            "district_name": payload.district_name,
            "district_slug": district_slug,
            "district_code": payload.district_code,
            "date_created": datetime.utcnow(),
            "created_by": ObjectId(current_user.get("_id")) if current_user.get("_id") else None,
            "hindi": translations.get("hindi", {}).get("title", payload.district_name),
            "kannada": translations.get("kannada", {}).get("title", payload.district_name),
            "english": translations.get("english", {}).get("title", payload.district_name),
        }

        # Insert into DB
        await asyncio.wait_for(get_db_service().insert_district(district_document), timeout=15.0)

        response_doc = to_extended_json(district_document)
        logger.info(f"[DISTRICT-CREATE] success district_id={district_id} slug={district_slug}")
        return DistrictResponse(success=True, data=response_doc)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DISTRICT-CREATE] failed district_id={district_id} error={e}")
        try:
            await get_db_service().update_district_fields(district_id, {"_deleted_due_to_error": True})
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error creating district: {str(e)}")

@router.get("/list", response_model=DistrictListResponse)
async def list_districts(
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    current_user: dict = Depends(get_current_user)
):
    """List districts with pagination."""
    try:
        logger.info(f"[DISTRICT-LIST] page={page} page_size={page_size}")
        
        # Calculate skip
        skip = (page - 1) * page_size
        
        # Get districts from DB
        districts, total = await get_db_service().get_districts_paginated(
            skip=skip, 
            limit=page_size
        )
        
        # Format response
        formatted_districts = [to_extended_json(district) for district in districts]
        
        return DistrictListResponse(
            success=True,
            data={"districts": formatted_districts},
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"[DISTRICT-LIST] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing districts: {str(e)}")

@router.get("/{district_id}", response_model=DistrictResponse)
async def get_district(
    district_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific district by ID."""
    try:
        logger.info(f"[DISTRICT-GET] district_id={district_id}")
        
        district = await get_db_service().get_district_by_id(ObjectId(district_id))
        if not district:
            raise HTTPException(status_code=404, detail="District not found")
        
        response_doc = to_extended_json(district)
        return DistrictResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DISTRICT-GET] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting district: {str(e)}")

@router.get("/slug/{district_slug}", response_model=DistrictResponse)
async def get_district_by_slug(
    district_slug: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific district by slug."""
    try:
        logger.info(f"[DISTRICT-GET-SLUG] district_slug={district_slug}")
        
        district = await get_db_service().get_district_by_slug(district_slug)
        if not district:
            raise HTTPException(status_code=404, detail="District not found")
        
        response_doc = to_extended_json(district)
        return DistrictResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DISTRICT-GET-SLUG] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting district by slug: {str(e)}")

@router.put("/{district_id}", response_model=DistrictResponse)
async def update_district(
    district_id: str,
    payload: DistrictUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update a district."""
    try:
        logger.info(f"[DISTRICT-UPDATE] district_id={district_id}")
        
        # Check if district exists
        existing_district = await get_db_service().get_district_by_id(ObjectId(district_id))
        if not existing_district:
            raise HTTPException(status_code=404, detail="District not found")
        
        # Prepare update fields
        updates = {}
        
        # Handle district_name update with translation and slug regeneration
        if payload.district_name is not None:
            updates["district_name"] = payload.district_name
            
            # Generate new slug
            new_slug = generate_slug(payload.district_name)
            if not new_slug:
                raise HTTPException(status_code=400, detail="Invalid district name for URL generation")
            
            # Check if new slug conflicts with existing districts (excluding current)
            existing_with_slug = await get_db_service().get_district_by_slug(new_slug)
            if existing_with_slug and str(existing_with_slug.get("_id")) != district_id:
                raise HTTPException(status_code=400, detail=f"District with similar name already exists: {existing_with_slug.get('district_name')}")
            
            updates["district_slug"] = new_slug
            
            # Re-translate the district_name if it's being updated
            try:
                # Detect source language
                source_lang = detect_language(payload.district_name)
                logger.info(f"[DISTRICT-UPDATE] detected_language={source_lang} for district_name update")
                
                # Translation with timeout
                try:
                    timeout_sec = float(os.getenv("TRANSLATION_PER_CALL_TIMEOUT", str(DEFAULT_TRANSLATION_TIMEOUT)))
                except (ValueError, TypeError):
                    timeout_sec = DEFAULT_TRANSLATION_TIMEOUT
                
                try:
                    translation_service = get_translation_service()
                    translations = await retry_translation_with_timeout(
                        translation_service,
                        payload.district_name,
                        "",
                        source_lang,
                        timeout=timeout_sec,
                        max_retries=3
                    )
                    logger.info(f"[DISTRICT-UPDATE] translation.done langs={list(translations.keys())} for district_name update")
                except asyncio.TimeoutError:
                    logger.error(f"[DISTRICT-UPDATE] translation timed out after {timeout_sec}s for district_name update")
                    raise HTTPException(status_code=504, detail="Translation timed out")
                except Exception as e:
                    logger.error(f"[DISTRICT-UPDATE] translation failed for district_name update: {e}")
                    if "IndicTrans2" in str(e) or "Model" in str(e):
                        logger.error("This appears to be an IndicTrans2 model loading issue")
                    raise
                
                # Handle bidirectional translation - ensure all three languages are present
                # If source is English, add original text as English translation
                if source_lang == "en":
                    translations["english"] = {
                        "title": payload.district_name,
                        "description": ""
                    }
                    logger.info(f"[DISTRICT-UPDATE] added original text as English translation for source=en")
                
                # If source is Kannada, add original text as Kannada translation  
                elif source_lang == "kn":
                    translations["kannada"] = {
                        "title": payload.district_name,
                        "description": ""
                    }
                    logger.info(f"[DISTRICT-UPDATE] added original text as Kannada translation for source=kn")
                
                # If source is Hindi, add original text as Hindi translation
                elif source_lang == "hi":
                    translations["hindi"] = {
                        "title": payload.district_name,
                        "description": ""
                    }
                    logger.info(f"[DISTRICT-UPDATE] added original text as Hindi translation for source=hi")
                
                # Update translation fields
                updates["hindi"] = translations.get("hindi", {}).get("title", payload.district_name)
                updates["kannada"] = translations.get("kannada", {}).get("title", payload.district_name)
                updates["english"] = translations.get("english", {}).get("title", payload.district_name)
                
                logger.info(f"[DISTRICT-UPDATE] updated translations and slug: {new_slug}")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[DISTRICT-UPDATE] translation error for district_name update: {e}")
                raise HTTPException(status_code=500, detail=f"Translation failed for district_name update: {str(e)}")
        
        if payload.district_code is not None:
            updates["district_code"] = payload.district_code
        
        # Update in DB
        success = await get_db_service().update_district_fields(ObjectId(district_id), updates)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update district")
        
        # Get updated district
        updated_district = await get_db_service().get_district_by_id(ObjectId(district_id))
        response_doc = to_extended_json(updated_district)
        
        return DistrictResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DISTRICT-UPDATE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating district: {str(e)}")

@router.delete("/{district_id}")
async def delete_district(
    district_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a district."""
    try:
        logger.info(f"[DISTRICT-DELETE] district_id={district_id}")
        
        # Check if district exists
        existing_district = await get_db_service().get_district_by_id(ObjectId(district_id))
        if not existing_district:
            raise HTTPException(status_code=404, detail="District not found")
        
        # Delete from DB
        success = await get_db_service().delete_district(ObjectId(district_id))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete district")
        
        return {"success": True, "message": "District deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DISTRICT-DELETE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting district: {str(e)}")

# Public endpoint for frontend district news (no authentication required)
@router.get("/news/{district_slug}")
async def get_district_news(
    district_slug: str,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE
):
    """Get news for a specific district by slug (public endpoint)."""
    try:
        logger.info(f"[DISTRICT-NEWS] district_slug={district_slug} page={page} page_size={page_size}")
        
        # Validate district exists
        district = await get_db_service().get_district_by_slug(district_slug)
        if not district:
            raise HTTPException(status_code=404, detail="District not found")
        
        # Calculate skip
        skip = (page - 1) * page_size
        
        # Get news for this district
        news_list, total = await get_db_service().get_news_paginated(
            skip=skip,
            limit=page_size,
            status_filter="approved",  # Only approved news
            district_slug_filter=district_slug
        )
        
        # Format response
        formatted_news = [to_extended_json(news) for news in news_list]
        
        return {
            "success": True,
            "district": to_extended_json(district),
            "data": {
                "news": formatted_news,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size if total > 0 else 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DISTRICT-NEWS] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting district news: {str(e)}")