from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
from bson import ObjectId
import asyncio
import os
from app.models.magazine import (
    MagazineCreateRequest, MagazineResponse,
    MagazineUpdateRequest, MagazineListResponse
)
from app.services.db_service import DBService
from app.services.auth_service import auth_service
from app.services.azure_blob_service import AzureBlobService
from app.utils.language_detection import detect_language
from app.utils.retry_utils import retry_translation_with_timeout
import logging

logger = logging.getLogger(__name__)

DEFAULT_TRANSLATION_TIMEOUT = 90.0
DEFAULT_PAGE_SIZE = 20
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
ALLOWED_PDF_EXTENSIONS = {'.pdf'}

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
                detail="Insufficient permissions. Only admin and moderator roles can create magazines.",
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

def get_azure_blob_service():
    """Lazy import of Azure blob service to avoid module-level failures."""
    try:
        return AzureBlobService()
    except ImportError as e:
        logger.error(f"Failed to import Azure blob service: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"Azure blob service import failed: {str(e)}"
        )
    except RuntimeError as e:
        logger.error(f"Azure blob service runtime error: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"Azure blob service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to initialize Azure blob service: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"Azure blob service unavailable: {str(e)}"
        )

def validate_file(file: UploadFile, file_type: str) -> None:
    """Validate uploaded file for type and size."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail=f"{file_type} file is required")
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"{file_type} file size exceeds maximum allowed size of {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Check file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_type == "thumbnail":
        if file_extension not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid thumbnail file type. Allowed types: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
            )
    elif file_type == "pdf":
        if file_extension not in ALLOWED_PDF_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid PDF file type. Only PDF files are allowed."
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
    for key in ["_id", "createdBy"]:
        if key in doc:
            val = doc[key]
            if isinstance(val, ObjectId) or (isinstance(val, str) and len(val) == 24):
                doc[key] = oidify(val)

    # Date fields
    for key in ["createdTime", "last_updated"]:
        if key in doc and isinstance(doc[key], datetime):
            doc[key] = dateify(doc[key])

    return doc

@router.post("/create", response_model=MagazineResponse)
async def create_magazine(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Create magazine with either file uploads or URL strings."""
    magazine_id = ObjectId()
    logger.info(f"[MAGAZINE-CREATE] start magazine_id={magazine_id} user={current_user.get('email', 'unknown')}")

    uploaded_files = []  # Track uploaded files for cleanup on failure
    
    try:
        # Check content type to determine if it's JSON or multipart
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            # Handle JSON request (backward compatibility)
            body = await request.json()
            title = body.get("title", "").strip()
            description = body.get("description", "").strip()
            editionNumber = body.get("editionNumber", "").strip()
            publishedMonth = body.get("publishedMonth", "").strip()
            publishedYear = body.get("publishedYear", "").strip()
            magazineThumbnail = body.get("magazineThumbnail", "").strip()
            magazinePdf = body.get("magazinePdf", "").strip()
            
            # Validate input fields
            if not title:
                raise HTTPException(status_code=400, detail="Title cannot be empty")
            if not description:
                raise HTTPException(status_code=400, detail="Description cannot be empty")
            if not editionNumber:
                raise HTTPException(status_code=400, detail="Edition number cannot be empty")
            if not publishedMonth:
                raise HTTPException(status_code=400, detail="Published month cannot be empty")
            if not publishedYear:
                raise HTTPException(status_code=400, detail="Published year cannot be empty")
            if not magazineThumbnail:
                raise HTTPException(status_code=400, detail="Magazine thumbnail URL cannot be empty")
            if not magazinePdf:
                raise HTTPException(status_code=400, detail="Magazine PDF URL cannot be empty")
            
            # Use URLs directly (no file upload)
            thumbnail_url = magazineThumbnail
            pdf_url = magazinePdf
            logger.info(f"[MAGAZINE-CREATE] JSON request - using provided URLs magazine_id={magazine_id}")
            
        elif "multipart/form-data" in content_type:
            # Handle multipart request (file uploads)
            form = await request.form()
            title = form.get("title", "").strip()
            description = form.get("description", "").strip()
            editionNumber = form.get("editionNumber", "").strip()
            publishedMonth = form.get("publishedMonth", "").strip()
            publishedYear = form.get("publishedYear", "").strip()
            magazineThumbnail = form.get("magazineThumbnail")
            magazinePdf = form.get("magazinePdf")
            
            # Validate input fields
            if not title:
                raise HTTPException(status_code=400, detail="Title cannot be empty")
            if not description:
                raise HTTPException(status_code=400, detail="Description cannot be empty")
            if not editionNumber:
                raise HTTPException(status_code=400, detail="Edition number cannot be empty")
            if not publishedMonth:
                raise HTTPException(status_code=400, detail="Published month cannot be empty")
            if not publishedYear:
                raise HTTPException(status_code=400, detail="Published year cannot be empty")

            # Validate uploaded files
            validate_file(magazineThumbnail, "thumbnail")
            validate_file(magazinePdf, "pdf")
            
            logger.info(f"[MAGAZINE-CREATE] file validation passed magazine_id={magazine_id}")

            # Upload files to Azure Blob Storage
            try:
                azure_service = get_azure_blob_service()
                if not azure_service.is_connected():
                    raise HTTPException(status_code=503, detail="Azure Blob Storage is not available")
                
                # Upload thumbnail
                thumbnail_url = azure_service.upload_magazine_file(
                    magazineThumbnail, publishedYear, publishedMonth, str(magazine_id), "thumbnail"
                )
                uploaded_files.append(("thumbnail", thumbnail_url))
                logger.info(f"[MAGAZINE-CREATE] thumbnail uploaded magazine_id={magazine_id}")
                
                # Upload PDF
                pdf_url = azure_service.upload_magazine_file(
                    magazinePdf, publishedYear, publishedMonth, str(magazine_id), "pdf"
                )
                uploaded_files.append(("pdf", pdf_url))
                logger.info(f"[MAGAZINE-CREATE] PDF uploaded magazine_id={magazine_id}")
                
            except Exception as e:
                logger.error(f"[MAGAZINE-CREATE] file upload failed magazine_id={magazine_id}: {e}")
                # Clean up any uploaded files
                azure_service = get_azure_blob_service()
                for file_type, url in uploaded_files:
                    try:
                        azure_service.delete_magazine_file(url)
                        logger.info(f"[MAGAZINE-CREATE] cleaned up {file_type} file")
                    except:
                        pass
                raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Content-Type must be either application/json or multipart/form-data")

        # Detect source language from title + description
        combined_text = f"{title} {description}"
        source_lang = detect_language(combined_text)
        logger.info(f"[MAGAZINE-CREATE] detected_language={source_lang} magazine_id={magazine_id}")

        # Translation with timeout
        try:
            timeout_sec = float(os.getenv("TRANSLATION_PER_CALL_TIMEOUT", str(DEFAULT_TRANSLATION_TIMEOUT)))
        except (ValueError, TypeError):
            timeout_sec = DEFAULT_TRANSLATION_TIMEOUT

        try:
            translation_service = get_translation_service()
            translations = await retry_translation_with_timeout(
                translation_service,
                title,
                description,
                source_lang,
                timeout=timeout_sec,
                max_retries=3
            )
            logger.info(f"[MAGAZINE-CREATE] translation.done langs={list(translations.keys())} magazine_id={magazine_id}")
        except asyncio.TimeoutError:
            logger.error(f"[MAGAZINE-CREATE] translation timed out after {timeout_sec}s for magazine_id={magazine_id}")
            # Clean up uploaded files
            azure_service = get_azure_blob_service()
            for file_type, url in uploaded_files:
                try:
                    azure_service.delete_magazine_file(url)
                except:
                    pass
            raise HTTPException(status_code=504, detail="Translation timed out")
        except Exception as e:
            logger.error(f"[MAGAZINE-CREATE] translation failed for magazine_id={magazine_id}: {e}")
            # Clean up uploaded files
            azure_service = get_azure_blob_service()
            for file_type, url in uploaded_files:
                try:
                    azure_service.delete_magazine_file(url)
                except:
                    pass
            if "IndicTrans2" in str(e) or "Model" in str(e):
                logger.error("This appears to be an IndicTrans2 model loading issue")
            raise

        # Determine status based on user role
        user_role = current_user.get("role", "")
        if user_role == "admin":
            status = "approved"
            logger.info(f"[MAGAZINE-CREATE] Admin user creating magazine - status=approved magazine_id={magazine_id}")
        else:
            status = "pending"
            logger.info(f"[MAGAZINE-CREATE] Non-admin user creating magazine - status=pending magazine_id={magazine_id}")

        # Create magazine document
        magazine_document = {
            "_id": magazine_id,
            "title": title,
            "description": description,
            "editionNumber": editionNumber,
            "publishedMonth": publishedMonth,
            "publishedYear": publishedYear,
            "magazineThumbnail": thumbnail_url,
            "magazinePdf": pdf_url,
            "createdBy": ObjectId(current_user.get("_id")) if current_user.get("_id") else None,
            "status": status,
            "createdTime": datetime.utcnow(),
            "last_updated": datetime.utcnow(),
            "hindi": {
                "title": translations.get("hindi", {}).get("title", title),
                "description": translations.get("hindi", {}).get("description", description),
            },
            "kannada": {
                "title": translations.get("kannada", {}).get("title", title),
                "description": translations.get("kannada", {}).get("description", description),
            },
            "english": {
                "title": translations.get("english", {}).get("title", title),
                "description": translations.get("english", {}).get("description", description),
            },
        }

        # Insert into DB
        await asyncio.wait_for(get_db_service().insert_magazine(magazine_document), timeout=15.0)

        response_doc = _to_extended_json(magazine_document)
        logger.info(f"[MAGAZINE-CREATE] success magazine_id={magazine_id}")
        return MagazineResponse(success=True, data=response_doc)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MAGAZINE-CREATE] failed magazine_id={magazine_id} error={e}")
        # Clean up uploaded files on any error
        try:
            azure_service = get_azure_blob_service()
            for file_type, url in uploaded_files:
                try:
                    azure_service.delete_magazine_file(url)
                    logger.info(f"[MAGAZINE-CREATE] cleaned up {file_type} file due to error")
                except:
                    pass
        except:
            pass
        try:
            await get_db_service().update_magazine_fields(magazine_id, {"_deleted_due_to_error": True})
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error creating magazine: {str(e)}")

@router.get("/list", response_model=MagazineListResponse)
async def list_magazines(
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    status_filter: str = None,
    current_user: dict = Depends(get_current_user)
):
    """List magazines with pagination and optional filters."""
    try:
        logger.info(f"[MAGAZINE-LIST] page={page} page_size={page_size} status={status_filter}")
        
        # Calculate skip
        skip = (page - 1) * page_size
        
        # Get magazines from DB
        magazines, total = await get_db_service().get_magazines_paginated(
            skip=skip, 
            limit=page_size, 
            status_filter=status_filter
        )
        
        # Format response
        formatted_magazines = [_to_extended_json(magazine) for magazine in magazines]
        
        return MagazineListResponse(
            success=True,
            data={"magazines": formatted_magazines},
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"[MAGAZINE-LIST] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing magazines: {str(e)}")

@router.get("/{magazine_id}", response_model=MagazineResponse)
async def get_magazine(
    magazine_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific magazine by ID."""
    try:
        logger.info(f"[MAGAZINE-GET] magazine_id={magazine_id}")
        
        magazine = await get_db_service().get_magazine_by_id(ObjectId(magazine_id))
        if not magazine:
            raise HTTPException(status_code=404, detail="Magazine not found")
        
        response_doc = _to_extended_json(magazine)
        return MagazineResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MAGAZINE-GET] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting magazine: {str(e)}")

@router.put("/{magazine_id}", response_model=MagazineResponse)
async def update_magazine(
    magazine_id: str,
    payload: MagazineUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update a magazine."""
    try:
        logger.info(f"[MAGAZINE-UPDATE] magazine_id={magazine_id}")
        
        # Check if magazine exists
        existing_magazine = await get_db_service().get_magazine_by_id(ObjectId(magazine_id))
        if not existing_magazine:
            raise HTTPException(status_code=404, detail="Magazine not found")
        
        # Prepare update fields
        updates = {}
        
        # Handle title and description updates with translation
        if payload.title is not None or payload.description is not None:
            # Get current values for fields that aren't being updated
            current_title = payload.title if payload.title is not None else existing_magazine.get("title", "")
            current_description = payload.description if payload.description is not None else existing_magazine.get("description", "")
            
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
                logger.info(f"[MAGAZINE-UPDATE] detected_language={source_lang} for title/description update")
                
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
                    logger.info(f"[MAGAZINE-UPDATE] translation.done langs={list(translations.keys())} for title/description update")
                except asyncio.TimeoutError:
                    logger.error(f"[MAGAZINE-UPDATE] translation timed out after {timeout_sec}s for title/description update")
                    raise HTTPException(status_code=504, detail="Translation timed out")
                except Exception as e:
                    logger.error(f"[MAGAZINE-UPDATE] translation failed for title/description update: {e}")
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
                    logger.info(f"[MAGAZINE-UPDATE] added original text as English translation for source=en")
                
                # If source is Kannada, add original text as Kannada translation  
                elif source_lang == "kn":
                    translations["kannada"] = {
                        "title": current_title,
                        "description": current_description
                    }
                    logger.info(f"[MAGAZINE-UPDATE] added original text as Kannada translation for source=kn")
                
                # If source is Hindi, add original text as Hindi translation
                elif source_lang == "hi":
                    translations["hindi"] = {
                        "title": current_title,
                        "description": current_description
                    }
                    logger.info(f"[MAGAZINE-UPDATE] added original text as Hindi translation for source=hi")
                
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
                
                logger.info(f"[MAGAZINE-UPDATE] updated translations for title/description update")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[MAGAZINE-UPDATE] translation error for title/description update: {e}")
                raise HTTPException(status_code=500, detail=f"Translation failed for title/description update: {str(e)}")
            
        if payload.editionNumber is not None:
            updates["editionNumber"] = payload.editionNumber
            
        if payload.publishedMonth is not None:
            updates["publishedMonth"] = payload.publishedMonth
            
        if payload.publishedYear is not None:
            updates["publishedYear"] = payload.publishedYear
            
        if payload.magazineThumbnail is not None:
            updates["magazineThumbnail"] = payload.magazineThumbnail
            
        if payload.magazinePdf is not None:
            updates["magazinePdf"] = payload.magazinePdf
            
        # Handle status updates (admin/moderator only)
        if payload.status is not None:
            user_role = current_user.get("role", "")
            if user_role in ["admin", "moderator"]:
                updates["status"] = payload.status
            else:
                logger.warning(f"[MAGAZINE-UPDATE] Non-admin/moderator user tried to update status: {current_user.get('email')}")
        
        # Update in DB
        success = await get_db_service().update_magazine_fields(ObjectId(magazine_id), updates)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update magazine")
        
        # Get updated magazine
        updated_magazine = await get_db_service().get_magazine_by_id(ObjectId(magazine_id))
        response_doc = _to_extended_json(updated_magazine)
        
        return MagazineResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MAGAZINE-UPDATE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating magazine: {str(e)}")

@router.delete("/{magazine_id}")
async def delete_magazine(
    magazine_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a magazine."""
    try:
        logger.info(f"[MAGAZINE-DELETE] magazine_id={magazine_id}")
        
        # Check if magazine exists
        existing_magazine = await get_db_service().get_magazine_by_id(ObjectId(magazine_id))
        if not existing_magazine:
            raise HTTPException(status_code=404, detail="Magazine not found")
        
        # Delete from DB
        success = await get_db_service().delete_magazine(ObjectId(magazine_id))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete magazine")
        
        return {"success": True, "message": "Magazine deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MAGAZINE-DELETE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting magazine: {str(e)}")
