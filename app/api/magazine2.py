from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form
from typing import Optional
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
from bson import ObjectId
import asyncio
import os
from app.models.magazine2 import (
    Magazine2CreateRequest, Magazine2Response,
    Magazine2UpdateRequest, Magazine2ListResponse
)
from app.services.db_singleton import get_db_service
from app.services.auth_service import auth_service
from app.services.azure_blob_service import AzureBlobService
from app.utils.language_detection import detect_language
from app.utils.retry_utils import retry_translation_with_timeout
from app.utils.json_encoder import to_extended_json
import logging

logger = logging.getLogger(__name__)

DEFAULT_TRANSLATION_TIMEOUT = 90.0
DEFAULT_PAGE_SIZE = 20
MAX_FILE_SIZE = 4 * 1024 * 1024  # 4MB
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
ALLOWED_PDF_EXTENSIONS = {'.pdf'}

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
                detail="Insufficient permissions. Only admin and moderator roles can create magazine2.",
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

# Removed local to_extended_json - now using universal to_extended_json from utils

@router.post("/create", response_model=Magazine2Response)
async def create_magazine2(
    title: str = Form(...),
    description: str = Form(...),
    editionNumber: str = Form(...),
    publishedMonth: str = Form(...),
    publishedYear: str = Form(...),
    magazineThumbnail: UploadFile = File(...),
    magazinePdf: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Create magazine2 with file uploads to Azure Blob Storage."""
    magazine2_id = ObjectId()
    logger.info(f"[MAGAZINE2-CREATE] start magazine2_id={magazine2_id} user={current_user.get('email', 'unknown')}")

    uploaded_files = []  # Track uploaded files for cleanup on failure
    
    try:
        # Validate input fields
        if not title.strip():
            raise HTTPException(status_code=400, detail="Title cannot be empty")
        if not description.strip():
            raise HTTPException(status_code=400, detail="Description cannot be empty")
        if not editionNumber.strip():
            raise HTTPException(status_code=400, detail="Edition number cannot be empty")
        if not publishedMonth.strip():
            raise HTTPException(status_code=400, detail="Published month cannot be empty")
        if not publishedYear.strip():
            raise HTTPException(status_code=400, detail="Published year cannot be empty")

        # Validate uploaded files
        validate_file(magazineThumbnail, "thumbnail")
        validate_file(magazinePdf, "pdf")
        
        logger.info(f"[MAGAZINE2-CREATE] file validation passed magazine2_id={magazine2_id}")

        # Upload files to Azure Blob Storage
        try:
            azure_service = get_azure_blob_service()
            if not azure_service.is_connected():
                raise HTTPException(status_code=503, detail="Azure Blob Storage is not available")
            
            # Upload thumbnail
            thumbnail_url = azure_service.upload_magazine2_file(
                magazineThumbnail, publishedYear, publishedMonth, str(magazine2_id), "thumbnail"
            )
            uploaded_files.append(("thumbnail", thumbnail_url))
            logger.info(f"[MAGAZINE2-CREATE] thumbnail uploaded magazine2_id={magazine2_id}")
            
            # Upload PDF
            pdf_url = azure_service.upload_magazine2_file(
                magazinePdf, publishedYear, publishedMonth, str(magazine2_id), "pdf"
            )
            uploaded_files.append(("pdf", pdf_url))
            logger.info(f"[MAGAZINE2-CREATE] PDF uploaded magazine2_id={magazine2_id}")
            
        except Exception as e:
            logger.error(f"[MAGAZINE2-CREATE] file upload failed magazine2_id={magazine2_id}: {e}")
            # Clean up any uploaded files
            azure_service = get_azure_blob_service()
            for file_type, url in uploaded_files:
                try:
                    azure_service.delete_magazine2_file(url)
                    logger.info(f"[MAGAZINE2-CREATE] cleaned up {file_type} file")
                except:
                    pass
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

        # Detect source language from title + description
        combined_text = f"{title} {description}"
        source_lang = detect_language(combined_text)
        logger.info(f"[MAGAZINE2-CREATE] detected_language={source_lang} magazine2_id={magazine2_id}")

        # Translation with timeout - CRITICAL: Translation fields are required in schema
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
            logger.info(f"[MAGAZINE2-CREATE] translation.done langs={list(translations.keys())} magazine2_id={magazine2_id}")
        except asyncio.TimeoutError:
            logger.error(f"[MAGAZINE2-CREATE] translation timed out after {timeout_sec}s for magazine2_id={magazine2_id}")
            # Clean up uploaded files
            azure_service = get_azure_blob_service()
            for file_type, url in uploaded_files:
                try:
                    azure_service.delete_magazine2_file(url)
                except:
                    pass
            raise HTTPException(status_code=504, detail="Translation timed out - required for magazine2 creation")
        except Exception as e:
            logger.error(f"[MAGAZINE2-CREATE] translation failed for magazine2_id={magazine2_id}: {e}")
            # Clean up uploaded files
            azure_service = get_azure_blob_service()
            for file_type, url in uploaded_files:
                try:
                    azure_service.delete_magazine2_file(url)
                except:
                    pass
            if "IndicTrans2" in str(e) or "Model" in str(e):
                logger.error("This appears to be an IndicTrans2 model loading issue")
            raise HTTPException(status_code=500, detail=f"Translation failed - required for magazine2 creation: {str(e)}")

        # Handle bidirectional translation - ensure all three languages are present
        # Source language can be either English or Kannada, we need all three languages
        
        # If source is English, add original text as English translation
        if source_lang == "en":
            translations["english"] = {
                "title": title,
                "description": description
            }
            logger.info(f"[MAGAZINE2-CREATE] added original text as English translation for source=en")
        
        # If source is Kannada, add original text as Kannada translation  
        elif source_lang == "kn":
            translations["kannada"] = {
                "title": title,
                "description": description
            }
            logger.info(f"[MAGAZINE2-CREATE] added original text as Kannada translation for source=kn")
        
        # If source is Hindi, add original text as Hindi translation
        elif source_lang == "hi":
            translations["hindi"] = {
                "title": title,
                "description": description
            }
            logger.info(f"[MAGAZINE2-CREATE] added original text as Hindi translation for source=hi")

        # Validate translation results - CRITICAL: All translation fields are required
        hindi_title = translations.get("hindi", {}).get("title", "")
        hindi_description = translations.get("hindi", {}).get("description", "")
        kannada_title = translations.get("kannada", {}).get("title", "")
        kannada_description = translations.get("kannada", {}).get("description", "")
        english_title = translations.get("english", {}).get("title", "")
        english_description = translations.get("english", {}).get("description", "")

        if not hindi_title or not hindi_description:
            raise HTTPException(status_code=500, detail="Hindi translation failed - required for magazine2")
        if not kannada_title or not kannada_description:
            raise HTTPException(status_code=500, detail="Kannada translation failed - required for magazine2")
        if not english_title or not english_description:
            raise HTTPException(status_code=500, detail="English translation failed - required for magazine2")

        # Determine status based on user role
        user_role = current_user.get("role", "")
        if user_role == "admin":
            status = "approved"
            logger.info(f"[MAGAZINE2-CREATE] Admin user creating magazine2 - status=approved magazine2_id={magazine2_id}")
        else:
            status = "pending"
            logger.info(f"[MAGAZINE2-CREATE] Non-admin user creating magazine2 - status=pending magazine2_id={magazine2_id}")

        # Create magazine2 document
        magazine2_document = {
            "_id": magazine2_id,
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
                "title": hindi_title,
                "description": hindi_description,
            },
            "kannada": {
                "title": kannada_title,
                "description": kannada_description,
            },
            "english": {
                "title": english_title,
                "description": english_description,
            },
        }

        # Insert into DB
        await asyncio.wait_for(get_db_service().insert_magazine2(magazine2_document), timeout=15.0)

        # Trigger search pipeline if status is approved
        if status == "approved":
            try:
                logger.info(f"[MAGAZINE2-CREATE] triggering search pipeline for approved magazine2_id={magazine2_id}")
                from app.services.magazine2_pipeline import Magazine2Pipeline
                pipeline = Magazine2Pipeline()
                result = await pipeline.process_single_magazine(str(magazine2_id))
                if result["success"]:
                    logger.info(f"[MAGAZINE2-CREATE] search pipeline completed successfully for magazine2_id={magazine2_id}")
                else:
                    logger.warning(f"[MAGAZINE2-CREATE] search pipeline failed for magazine2_id={magazine2_id}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"[MAGAZINE2-CREATE] search pipeline error for magazine2_id={magazine2_id}: {e}")
                # Don't fail the main operation if search pipeline fails

        response_doc = to_extended_json(magazine2_document)
        logger.info(f"[MAGAZINE2-CREATE] success magazine2_id={magazine2_id}")
        return Magazine2Response(success=True, data=response_doc)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MAGAZINE2-CREATE] failed magazine2_id={magazine2_id} error={e}")
        # Clean up uploaded files on any error
        try:
            azure_service = get_azure_blob_service()
            for file_type, url in uploaded_files:
                try:
                    azure_service.delete_magazine2_file(url)
                    logger.info(f"[MAGAZINE2-CREATE] cleaned up {file_type} file due to error")
                except:
                    pass
        except:
            pass
        try:
            await get_db_service().update_magazine2_fields(magazine2_id, {"_deleted_due_to_error": True})
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error creating magazine2: {str(e)}")

@router.get("/list", response_model=Magazine2ListResponse)
async def list_magazine2s(
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    status_filter: str = None
):
    """List magazine2s with pagination and optional filters."""
    try:
        logger.info(f"[MAGAZINE2-LIST] page={page} page_size={page_size} status={status_filter}")
        
        # Calculate skip
        skip = (page - 1) * page_size
        
        # Get magazine2s from DB
        magazine2s, total = await get_db_service().get_magazine2s_paginated(
            skip=skip, 
            limit=page_size, 
            status_filter=status_filter
        )
        
        # Format response
        formatted_magazine2s = [to_extended_json(magazine2) for magazine2 in magazine2s]
        
        return Magazine2ListResponse(
            success=True,
            data={"magazine2s": formatted_magazine2s},
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"[MAGAZINE2-LIST] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing magazine2s: {str(e)}")

@router.get("/{magazine2_id}", response_model=Magazine2Response)
async def get_magazine2(
    magazine2_id: str
):
    """Get a specific magazine2 by ID."""
    try:
        logger.info(f"[MAGAZINE2-GET] magazine2_id={magazine2_id}")
        
        magazine2 = await get_db_service().get_magazine2_by_id(ObjectId(magazine2_id))
        if not magazine2:
            raise HTTPException(status_code=404, detail="Magazine2 not found")
        
        response_doc = to_extended_json(magazine2)
        return Magazine2Response(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MAGAZINE2-GET] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting magazine2: {str(e)}")

@router.put("/{magazine2_id}", response_model=Magazine2Response)
async def update_magazine2(
    magazine2_id: str,
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    editionNumber: Optional[str] = Form(None),
    publishedMonth: Optional[str] = Form(None),
    publishedYear: Optional[str] = Form(None),
    magazineThumbnail: Optional[UploadFile] = File(None),
    magazinePdf: Optional[UploadFile] = File(None),
    status: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """Update a magazine2 with optional file uploads to Azure Blob Storage."""
    try:
        logger.info(f"[MAGAZINE2-UPDATE] start magazine2_id={magazine2_id} user={current_user.get('email', 'unknown')}")
        
        # Check if magazine2 exists
        existing_magazine2 = await get_db_service().get_magazine2_by_id(ObjectId(magazine2_id))
        if not existing_magazine2:
            raise HTTPException(status_code=404, detail="Magazine2 not found")
        
        uploaded_files = []  # Track uploaded files for cleanup on failure
        old_files_to_delete = []  # Track old files to delete
        
        # Initialize azure service early for cleanup operations
        azure_service = get_azure_blob_service()
        
        try:
            # Prepare update fields
            updates = {}
            
            # Handle file uploads first
            if not azure_service.is_connected():
                raise HTTPException(status_code=503, detail="Azure Blob Storage is not available")
            
            # Get current values for file operations
            current_published_year = publishedYear if publishedYear is not None else existing_magazine2.get("publishedYear", "")
            current_published_month = publishedMonth if publishedMonth is not None else existing_magazine2.get("publishedMonth", "")
            
            # Handle thumbnail update
            if magazineThumbnail is not None:
                validate_file(magazineThumbnail, "thumbnail")
                
                # Upload new thumbnail first to get the new URL
                thumbnail_url = azure_service.upload_magazine2_file(
                    magazineThumbnail, current_published_year, current_published_month, magazine2_id, "thumbnail"
                )
                uploaded_files.append(("thumbnail", thumbnail_url))
                updates["magazineThumbnail"] = thumbnail_url
                
                # Only delete old thumbnail if the blob path is different
                old_thumbnail_url = existing_magazine2.get("magazineThumbnail")
                if old_thumbnail_url and old_thumbnail_url != thumbnail_url:
                    old_files_to_delete.append(("thumbnail", old_thumbnail_url))
                    logger.info(f"[MAGAZINE2-UPDATE] thumbnail path changed, will delete old file")
                else:
                    logger.info(f"[MAGAZINE2-UPDATE] thumbnail path unchanged, using overwrite")
                
                logger.info(f"[MAGAZINE2-UPDATE] thumbnail uploaded magazine2_id={magazine2_id}")
            
            # Handle PDF update
            if magazinePdf is not None:
                validate_file(magazinePdf, "pdf")
                
                # Upload new PDF first to get the new URL
                pdf_url = azure_service.upload_magazine2_file(
                    magazinePdf, current_published_year, current_published_month, magazine2_id, "pdf"
                )
                uploaded_files.append(("pdf", pdf_url))
                updates["magazinePdf"] = pdf_url
                
                # Only delete old PDF if the blob path is different
                old_pdf_url = existing_magazine2.get("magazinePdf")
                if old_pdf_url and old_pdf_url != pdf_url:
                    old_files_to_delete.append(("pdf", old_pdf_url))
                    logger.info(f"[MAGAZINE2-UPDATE] PDF path changed, will delete old file")
                else:
                    logger.info(f"[MAGAZINE2-UPDATE] PDF path unchanged, using overwrite")
                
                logger.info(f"[MAGAZINE2-UPDATE] PDF uploaded magazine2_id={magazine2_id}")
            
            # Handle title and description updates with translation
            if title is not None or description is not None:
                # Get current values for fields that aren't being updated
                current_title = title if title is not None else existing_magazine2.get("title", "")
                current_description = description if description is not None else existing_magazine2.get("description", "")
                
                # Update the fields
                if title is not None:
                    updates["title"] = title
                if description is not None:
                    updates["description"] = description
                
                # Re-translate if title or description is being updated
                try:
                    # Detect source language from combined title + description
                    combined_text = f"{current_title} {current_description}"
                    source_lang = detect_language(combined_text)
                    logger.info(f"[MAGAZINE2-UPDATE] detected_language={source_lang} for title/description update")
                    
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
                        logger.info(f"[MAGAZINE2-UPDATE] translation.done langs={list(translations.keys())} for title/description update")
                    except asyncio.TimeoutError:
                        logger.error(f"[MAGAZINE2-UPDATE] translation timed out after {timeout_sec}s for title/description update")
                        raise HTTPException(status_code=504, detail="Translation timed out")
                    except Exception as e:
                        logger.error(f"[MAGAZINE2-UPDATE] translation failed for title/description update: {e}")
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
                        logger.info(f"[MAGAZINE2-UPDATE] added original text as English translation for source=en")
                    
                    # If source is Kannada, add original text as Kannada translation  
                    elif source_lang == "kn":
                        translations["kannada"] = {
                            "title": current_title,
                            "description": current_description
                        }
                        logger.info(f"[MAGAZINE2-UPDATE] added original text as Kannada translation for source=kn")
                    
                    # If source is Hindi, add original text as Hindi translation
                    elif source_lang == "hi":
                        translations["hindi"] = {
                            "title": current_title,
                            "description": current_description
                        }
                        logger.info(f"[MAGAZINE2-UPDATE] added original text as Hindi translation for source=hi")
                    
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
                    
                    logger.info(f"[MAGAZINE2-UPDATE] updated translations for title/description update")
                    
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"[MAGAZINE2-UPDATE] translation error for title/description update: {e}")
                    raise HTTPException(status_code=500, detail=f"Translation failed for title/description update: {str(e)}")
                
            # Handle other field updates
            if editionNumber is not None:
                updates["editionNumber"] = editionNumber
                
            if publishedMonth is not None:
                updates["publishedMonth"] = publishedMonth
                
            if publishedYear is not None:
                updates["publishedYear"] = publishedYear
                
            # Handle status updates (admin/moderator only)
            if status is not None:
                user_role = current_user.get("role", "")
                if user_role in ["admin", "moderator"]:
                    updates["status"] = status
                    # Track if status is changing to approved for search pipeline trigger
                    status_changing_to_approved = (status == "approved" and existing_magazine2.get("status") != "approved")
                else:
                    logger.warning(f"[MAGAZINE2-UPDATE] Non-admin/moderator user tried to update status: {current_user.get('email')}")
                    status_changing_to_approved = False
            else:
                status_changing_to_approved = False
            
            # Update last_updated timestamp
            updates["last_updated"] = datetime.utcnow()
            
            # Update in DB
            success = await get_db_service().update_magazine2_fields(ObjectId(magazine2_id), updates)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to update magazine2")
            
            # Trigger search pipeline if status changed to approved
            if status_changing_to_approved:
                try:
                    logger.info(f"[MAGAZINE2-UPDATE] triggering search pipeline for status change to approved magazine2_id={magazine2_id}")
                    from app.services.magazine2_pipeline import Magazine2Pipeline
                    pipeline = Magazine2Pipeline()
                    result = await pipeline.process_single_magazine(magazine2_id)
                    if result["success"]:
                        logger.info(f"[MAGAZINE2-UPDATE] search pipeline completed successfully for magazine2_id={magazine2_id}")
                    else:
                        logger.warning(f"[MAGAZINE2-UPDATE] search pipeline failed for magazine2_id={magazine2_id}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"[MAGAZINE2-UPDATE] search pipeline error for magazine2_id={magazine2_id}: {e}")
                    # Don't fail the main operation if search pipeline fails
            
            # Delete old files after successful database update
            for file_type, old_url in old_files_to_delete:
                try:
                    azure_service.delete_magazine2_file(old_url)
                    logger.info(f"[MAGAZINE2-UPDATE] deleted old {file_type} file: {old_url}")
                except Exception as e:
                    logger.warning(f"[MAGAZINE2-UPDATE] failed to delete old {file_type} file: {e}")
            
            # Get updated magazine2
            updated_magazine2 = await get_db_service().get_magazine2_by_id(ObjectId(magazine2_id))
            response_doc = to_extended_json(updated_magazine2)
            
            logger.info(f"[MAGAZINE2-UPDATE] success magazine2_id={magazine2_id}")
            return Magazine2Response(success=True, data=response_doc)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[MAGAZINE2-UPDATE] failed magazine2_id={magazine2_id} error={e}")
            # Clean up any uploaded files on error
            try:
                for file_type, url in uploaded_files:
                    try:
                        azure_service.delete_magazine2_file(url)
                        logger.info(f"[MAGAZINE2-UPDATE] cleaned up {file_type} file due to error")
                    except:
                        pass
            except:
                pass
            raise HTTPException(status_code=500, detail=f"Error updating magazine2: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MAGAZINE2-UPDATE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating magazine2: {str(e)}")

@router.delete("/{magazine2_id}")
async def delete_magazine2(
    magazine2_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a magazine2."""
    try:
        logger.info(f"[MAGAZINE2-DELETE] magazine2_id={magazine2_id}")
        
        # Check if magazine2 exists
        existing_magazine2 = await get_db_service().get_magazine2_by_id(ObjectId(magazine2_id))
        if not existing_magazine2:
            raise HTTPException(status_code=404, detail="Magazine2 not found")
        
        # Delete from DB
        success = await get_db_service().delete_magazine2(ObjectId(magazine2_id))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete magazine2")
        
        return {"success": True, "message": "Magazine2 deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MAGAZINE2-DELETE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting magazine2: {str(e)}")
