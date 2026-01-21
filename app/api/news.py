from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
from time import perf_counter
from bson import ObjectId
import asyncio
import os
from app.models.news import (
    NewsCreateRequest, NewsResponse, NewsUpdateRequest,
    TranslationRequest, TranslationResponse,
    TTSRequest, TTSResponse, HealthResponse
)
from app.services.tts_service import TTSService
from app.services.azure_blob_service import AzureBlobService
from app.services.db_service import DBService
from app.services.auth_service import auth_service
from app.utils.language_detection import detect_language
from app.utils.retry_utils import retry_translation_with_timeout, retry_with_exponential_backoff
from app.utils.json_encoder import to_extended_json
import logging 

logger = logging.getLogger(__name__)

DEFAULT_TRANSLATION_TIMEOUT = 90.0
DEFAULT_TTS_TIMEOUT = 30.0
TTS_RETRY_TIMEOUT = 60.0  # 60 seconds for retry attempts
MAX_TTS_RETRIES = 2  # Maximum retry attempts for TTS
MAX_RETRIES = 2
DEFAULT_PAGE_SIZE = 20

router = APIRouter()

tts_service = None
azure_blob_service = None
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
                detail="Insufficient permissions. Only admin and moderator roles can create news.",
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

def get_tts_service():
    """Lazy import of TTS service to avoid module-level failures."""
    global tts_service
    if tts_service is None:
        try:
            tts_service = TTSService()
        except ImportError as e:
            logger.error(f"Failed to import TTS service: {e}")
            raise HTTPException(
                status_code=503, 
                detail=f"TTS service import failed: {str(e)}"
            )
        except RuntimeError as e:
            logger.error(f"TTS service runtime error: {e}")
            raise HTTPException(
                status_code=503, 
                detail=f"TTS service unavailable: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize TTS service: {e}")
            raise HTTPException(
                status_code=503, 
                detail=f"TTS service unavailable: {str(e)}"
            )
    return tts_service

def get_azure_blob_service():
    """Lazy import of Azure Blob service to avoid module-level failures."""
    global azure_blob_service
    if azure_blob_service is None:
        try:
            azure_blob_service = AzureBlobService()
        except ImportError as e:
            logger.error(f"Failed to import Azure Blob service: {e}")
            raise HTTPException(
                status_code=503, 
                detail=f"Azure Blob service import failed: {str(e)}"
            )
        except RuntimeError as e:
            logger.error(f"Azure Blob service runtime error: {e}")
            raise HTTPException(
                status_code=503, 
                detail=f"Azure Blob service unavailable: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Azure Blob service: {e}")
            raise HTTPException(
                status_code=503, 
                detail=f"Azure Blob service unavailable: {str(e)}"
            )
    return azure_blob_service

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

# Removed local _to_extended_json - now using universal to_extended_json from utils


async def _generate_audio_with_retry(text: str, language: str, max_retries: int = MAX_TTS_RETRIES) -> str:
    """Generate audio with timeout and retry logic."""
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"[TTS-RETRY] Attempt {attempt + 1}/{max_retries + 1} for {language}")
            
            # Use timeout for TTS generation
            audio_file = await asyncio.wait_for(
                asyncio.to_thread(get_tts_service().generate_audio, text, language),
                timeout=TTS_RETRY_TIMEOUT
            )
            
            logger.info(f"[TTS-RETRY] Success on attempt {attempt + 1} for {language}")
            return audio_file
            
        except asyncio.TimeoutError:
            logger.warning(f"[TTS-RETRY] Timeout on attempt {attempt + 1} for {language}")
            if attempt < max_retries:
                wait_time = (attempt + 1) * 5  # Exponential backoff: 5s, 10s, 15s
                logger.info(f"[TTS-RETRY] Waiting {wait_time}s before retry for {language}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"[TTS-RETRY] All attempts failed for {language}")
                raise RuntimeError(f"TTS generation failed after {max_retries + 1} attempts for {language}")
        
        except Exception as e:
            logger.warning(f"[TTS-RETRY] Error on attempt {attempt + 1} for {language}: {e}")
            if attempt < max_retries:
                wait_time = (attempt + 1) * 3  # Shorter backoff for other errors: 3s, 6s, 9s
                logger.info(f"[TTS-RETRY] Waiting {wait_time}s before retry for {language}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"[TTS-RETRY] All attempts failed for {language}")
                raise RuntimeError(f"TTS generation failed after {max_retries + 1} attempts for {language}: {str(e)}")


async def _translate_and_attach_translations(document_id: ObjectId, payload: NewsCreateRequest, source_lang: str):
    """Background job: translate to all languages, update document, then generate TTS."""
    try:
        # Verify document exists
        existing_doc = await get_db_service().get_news_by_id(document_id)
        if not existing_doc:
            logger.warning(f"[BG-TRANSLATE] Document not found, skipping translation doc={document_id}")
            return

        logger.info(f"[BG-TRANSLATE] Starting translation for doc={document_id} source_lang={source_lang}")

        start_time = datetime.utcnow()
        attempts = int(existing_doc.get("translationAttempts", 0) or 0) + 1

        await get_db_service().update_news_fields(
            document_id,
            {
                "translationStatus": "processing",
                "translationAttempts": attempts,
                "translationLastAttempt": start_time,
                "translationError": "",
                "last_updated": datetime.utcnow(),
            },
        )

        # Perform translation (no timeout in background)
        try:
            translation_service = get_translation_service()
            translations = await translation_service.translate_to_all_async(
                payload.title,
                payload.description,
                source_lang
            )
            logger.info(f"[BG-TRANSLATE] Translation completed langs={list(translations.keys())} doc={document_id}")
        except Exception as e:
            logger.error(f"[BG-TRANSLATE] Translation failed for doc={document_id}: {e}")
            await get_db_service().update_news_fields(
                document_id,
                {
                    "translationStatus": "failed",
                    "translationError": str(e)[:512],
                    "translationAttempts": attempts,
                    "last_updated": datetime.utcnow(),
                },
            )
            # Don't crash, just log and return - document still has original text
            return

        # Update document with translations
        updates = {
            "hindi.title": translations.get("hindi", {}).get("title", payload.title),
            "hindi.description": translations.get("hindi", {}).get("description", payload.description),
            "kannada.title": translations.get("kannada", {}).get("title", payload.title),
            "kannada.description": translations.get("kannada", {}).get("description", payload.description),
            "English.title": translations.get("english", {}).get("title", payload.title),
            "English.description": translations.get("english", {}).get("description", payload.description),
            "last_updated": datetime.utcnow(),
            "translationStatus": "completed",
            "translationError": "",
            "translationAttempts": attempts,
            "translationCompletedAt": datetime.utcnow(),
            "translationDurationSeconds": (datetime.utcnow() - start_time).total_seconds(),
        }

        ok = await get_db_service().update_news_fields(document_id, updates)
        if not ok:
            logger.error(f"[BG-TRANSLATE] Mongo update returned modified_count=0 for doc={document_id}")
            return
        else:
            logger.info(f"[BG-TRANSLATE] Successfully updated document {document_id} with translations")

        # Now generate TTS for all languages
        logger.info(f"[BG-TRANSLATE] Starting TTS generation for doc={document_id}")
        await _generate_and_attach_audio(document_id, payload, translations, source_lang)

    except Exception as e:
        logger.error(f"[BG-TRANSLATE] Unexpected background translation error for {document_id}: {e}")
        await get_db_service().update_news_fields(
            document_id,
            {
                "translationStatus": "failed",
                "translationError": str(e)[:512],
                "last_updated": datetime.utcnow(),
            },
        )


async def _generate_and_attach_audio(document_id: ObjectId, payload: NewsCreateRequest, translations: dict, source_lang: str):
    """Background job: generate TTS sequentially, upload to Firebase, update Mongo doc."""
    try:
        # Verify document exists
        existing_doc = await get_db_service().get_news_by_id(document_id)
        if not existing_doc:
            logger.warning(f"[BG-TTS] Document not found, skipping TTS generation doc={document_id}")
            return

        updates = {"last_updated": datetime.utcnow()}
        lang_map = {"en": "English", "hi": "hindi", "kn": "kannada"}
        translation_keys = {"en": "english", "hi": "hindi", "kn": "kannada"}
        any_success = False

        # Process languages sequentially
        for lang in ["en", "hi", "kn"]:
            try:
                if lang == source_lang:
                    text = f"{payload.title}. {payload.description}"
                else:
                    translation_key = translation_keys[lang]
                    text = f"{translations.get(translation_key, {}).get('title', payload.title)}. " \
                           f"{translations.get(translation_key, {}).get('description', payload.description)}"

                text = text[:1200]  # truncate

                logger.info(f"[BG-TTS] Generating audio for {lang} (doc={document_id})")

                # Generate audio with retry logic and timeout
                audio_file = await _generate_audio_with_retry(text, lang)
                
                # Upload to Azure Blob with timeout
                audio_url = await asyncio.wait_for(
                    asyncio.to_thread(get_azure_blob_service().upload_audio, audio_file, lang, str(document_id)),
                    timeout=30.0  # 30 second timeout for upload
                )

                # Update DB fields
                field = f"{lang_map[lang]}.audio_description"
                updates[field] = audio_url
                any_success = True
                logger.info(f"[BG-TTS] Completed audio for {lang} -> {audio_url}")

                await asyncio.sleep(2)  # slight delay between langs

            except asyncio.TimeoutError as e:
                logger.error(f"[BG-TTS] Timeout for {lang} (doc={document_id}): {e}")
                continue
            except Exception as e:
                logger.error(f"[BG-TTS] Failed audio for {lang} (doc={document_id}): {e}")
                continue

        # Update MongoDB with generated audio URLs
        ok = await get_db_service().update_news_fields(document_id, updates)
        if not ok:
            logger.error(f"[BG-TTS] Mongo update returned modified_count=0 for doc={document_id}")
        else:
            logger.info(f"[BG-TTS] Successfully updated document {document_id} with audio URLs")

    except Exception as e:
        logger.error(f"[BG-TTS] Unexpected background TTS error for {document_id}: {e}")



# async def _translate_and_attach(document_id: ObjectId, payload: NewsCreateRequest, source_lang: str):
#     """Background: translate to target langs, update doc, then generate TTS and attach URLs."""
#     try:
#         translations = await translation_service.translate_to_all_async(
#             payload.title,
#             payload.description,
#             source_lang,
#         )

#         updates = {
#             "hindi.title": translations.get("hi", {}).get("title", payload.title),
#             "hindi.description": translations.get("hi", {}).get("description", payload.description),
#             "kannada.title": translations.get("kn", {}).get("title", payload.title),
#             "kannada.description": translations.get("kn", {}).get("description", payload.description),
#             "English.title": translations.get("en", {}).get("title", payload.title),
#             "English.description": translations.get("en", {}).get("description", payload.description),
#             "last_updated": datetime.utcnow(),
#         }

#         try:
#             await db_service.update_news_fields(document_id, updates)


@router.get("/", response_model=NewsResponse)
async def get_all_news(
    status: str = None,
    district_slug: str = None,
    date: str = None
):
    """Get all news with optional status/district/date filters."""
    try:
        t0 = perf_counter()
        logger.info(f"[GET_ALL] start status={status} district_slug={district_slug} date={date}")
        
        # Get all news from database using db_service (no pagination)
        news_list, total = await get_db_service().get_news_paginated(
            skip=0,
            limit=10000,  # Large limit to get all results
            status_filter=status,
            district_slug_filter=district_slug,
            date_filter=date
        )
        logger.info(
            "[GET_ALL] fetched %s/%s items in %.3fs (status=%s district=%s date=%s)",
            len(news_list),
            total,
            perf_counter() - t0,
            status,
            district_slug,
            date,
        )
        
        # Convert to extended JSON format
        news_list_json = [to_extended_json(doc) for doc in news_list]
        
        logger.info(f"[GET_ALL] success - found {len(news_list)}/{total} news items")
        
        return NewsResponse(
            success=True,
            data={
                "news": news_list_json,
                "total": total
            }
        )
            
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GET_ALL] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")


@router.get("/{news_id}", response_model=NewsResponse)
async def get_news_by_id(
    news_id: str
):
    """Get a single news article by ID."""
    try:
        t0 = perf_counter()
        logger.info(f"[GET_BY_ID] start news_id={news_id}")
        
        # Validate ObjectId format
        try:
            ObjectId(news_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid news ID format")
        
        # Get news from database
        news = await get_db_service().get_news_by_id(ObjectId(news_id))
        if not news:
            raise HTTPException(status_code=404, detail="News not found")
        logger.info("[GET_BY_ID] fetched 1 item in %.3fs news_id=%s", perf_counter() - t0, news_id)
        
        # Convert to extended JSON format
        news_json = to_extended_json(news)
        
        logger.info(f"[GET_BY_ID] success news_id={news_id}")
        return NewsResponse(success=True, data=news_json)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GET_BY_ID] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")


@router.post("/create", response_model=NewsResponse)
async def create_news(
    payload: NewsCreateRequest, 
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Create news doc immediately; translation and TTS run in background."""
    document_id = ObjectId()
    logger.info(f"[CREATE] start doc={document_id} user={current_user.get('email', 'unknown')}")

    try:
        source_lang = detect_language(payload.title + " " + payload.description)
        logger.info(f"[CREATE] detected_language={source_lang} doc={document_id}")

        # Determine status and isLive based on user role
        user_role = current_user.get("role", "")
        if user_role == "admin":
            status = "approved"
            isLive = True
            logger.info(f"[CREATE] Admin user creating news - status=approved, isLive=True doc={document_id}")
        else:
            status = "pending"
            isLive = False
            logger.info(f"[CREATE] Non-admin user creating news - status=pending, isLive=False doc={document_id}")

        # Create news document with original text and empty translations
        # Translations will be filled in by background task
        news_document = {
            "_id": document_id,
            "title": payload.title,
            "description": payload.description,
            "newsImage": payload.newsImage,
            # "category": ObjectId(payload.category) if isinstance(payload.category, str) and len(payload.category) == 24 else payload.category,
            "author": payload.author,
            "publishedAt": datetime.utcnow(),
            "magazineType": payload.magazineType,
            "newsType": payload.newsType,
            "district_slug": payload.district_slug,
            "source": payload.source,
            "isLive": isLive,
            "views": 0,
            "total_Likes": 0,
            "comments": [],
            "likedBy": [],
            "createdTime": datetime.utcnow(),
            "last_updated": datetime.utcnow(),
            "createdBy": ObjectId(current_user.get("_id")) if current_user.get("_id") else None,
            "status": status,
            "translationStatus": "pending",
            "translationAttempts": 0,
            "translationError": "",
            "translationLastAttempt": None,
            "translationCompletedAt": None,
            "translationDurationSeconds": 0.0,
            "hindi": {
                "title": "",
                "description": "",
                "audio_description": "",
            },
            "kannada": {
                "title": "",
                "description": "",
                "audio_description": "",
            },
            "English": {
                "title": "",
                "description": "",
                "audio_description": "",
            },
        }

        # Insert into DB
        await asyncio.wait_for(get_db_service().insert_news(news_document), timeout=15.0)
        logger.info(f"[CREATE] Document created successfully doc={document_id}")

        # Schedule translation in background (will also trigger TTS after completion)
        background_tasks.add_task(
            _translate_and_attach_translations, document_id, payload, source_lang
        )
        logger.info(f"[CREATE] Background translation task scheduled doc={document_id}")

        response_doc = to_extended_json(news_document)
        return NewsResponse(success=True, data=response_doc)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CREATE] failed doc={document_id} error={e}")
        try:
            await get_db_service().update_news_fields(document_id, {"_deleted_due_to_error": True})
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error creating news: {str(e)}")


@router.put("/{news_id}", response_model=NewsResponse)
async def update_news(
    news_id: str,
    payload: NewsUpdateRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Update a news article."""
    try:
        logger.info(f"[UPDATE] start news_id={news_id} user={current_user.get('email', 'unknown')}")
        
        # Check if news exists
        existing_news = await get_db_service().get_news_by_id(ObjectId(news_id))
        if not existing_news:
            raise HTTPException(status_code=404, detail="News not found")
        
        # Check permissions
        user_role = current_user.get("role", "")
        user_id = current_user.get("id")
        news_creator_id = existing_news.get("createdBy")
        
        # Admin can update any news, others can only update their own
        if user_role != "admin" and (not user_id or not news_creator_id or str(news_creator_id) != str(user_id)):
            raise HTTPException(status_code=403, detail="You can only update your own news articles")
        
        # Prepare update fields
        updates = {}
        
        # Handle title and description updates with translation
        if payload.title is not None or payload.description is not None:
            # Get current values for fields that aren't being updated
            current_title = payload.title if payload.title is not None else existing_news.get("title", "")
            current_description = payload.description if payload.description is not None else existing_news.get("description", "")
            
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
                logger.info(f"[UPDATE] detected_language={source_lang} for title/description update")
                
                # Translation with timeout
                try:
                    timeout_sec = float(os.getenv("TRANSLATION_PER_CALL_TIMEOUT", str(DEFAULT_TRANSLATION_TIMEOUT)))
                except (ValueError, TypeError):
                    timeout_sec = DEFAULT_TRANSLATION_TIMEOUT
                
                translation_start = datetime.utcnow()
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
                    logger.info(f"[UPDATE] translation.done langs={list(translations.keys())} for title/description update")
                except asyncio.TimeoutError:
                    logger.error(f"[UPDATE] translation timed out after {timeout_sec}s for title/description update")
                    raise HTTPException(status_code=504, detail="Translation timed out")
                except Exception as e:
                    logger.error(f"[UPDATE] translation failed for title/description update: {e}")
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
                    logger.info(f"[UPDATE] added original text as English translation for source=en")
                
                # If source is Kannada, add original text as Kannada translation  
                elif source_lang == "kn":
                    translations["kannada"] = {
                        "title": current_title,
                        "description": current_description
                    }
                    logger.info(f"[UPDATE] added original text as Kannada translation for source=kn")
                
                # If source is Hindi, add original text as Hindi translation
                elif source_lang == "hi":
                    translations["hindi"] = {
                        "title": current_title,
                        "description": current_description
                    }
                    logger.info(f"[UPDATE] added original text as Hindi translation for source=hi")
                
                # Update translation fields
                duration_seconds = (datetime.utcnow() - translation_start).total_seconds()
                updates["hindi"] = {
                    "title": translations.get("hindi", {}).get("title", current_title),
                    "description": translations.get("hindi", {}).get("description", current_description),
                    "audio_description": existing_news.get("hindi", {}).get("audio_description", "")
                }
                updates["kannada"] = {
                    "title": translations.get("kannada", {}).get("title", current_title),
                    "description": translations.get("kannada", {}).get("description", current_description),
                    "audio_description": existing_news.get("kannada", {}).get("audio_description", "")
                }
                updates["English"] = {
                    "title": translations.get("english", {}).get("title", current_title),
                    "description": translations.get("english", {}).get("description", current_description),
                    "audio_description": existing_news.get("English", {}).get("audio_description", "")
                }
                updates["translationStatus"] = "completed"
                updates["translationError"] = ""
                updates["translationLastAttempt"] = translation_start
                updates["translationCompletedAt"] = datetime.utcnow()
                updates["translationDurationSeconds"] = duration_seconds
                
                logger.info(f"[UPDATE] updated translations for title/description update")
                
                # Schedule TTS regeneration in background if title/description changed
                background_tasks.add_task(
                    _generate_and_attach_audio, ObjectId(news_id), 
                    type('obj', (object,), {'title': current_title, 'description': current_description})(),
                    translations, source_lang
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[UPDATE] translation error for title/description update: {e}")
                raise HTTPException(status_code=500, detail=f"Translation failed for title/description update: {str(e)}")
            
        # Handle other field updates
        # if payload.category is not None:
        #     updates["category"] = ObjectId(payload.category) if isinstance(payload.category, str) and len(payload.category) == 24 else payload.category
            
        if payload.author is not None:
            updates["author"] = payload.author
            
        if payload.newsImage is not None:
            updates["newsImage"] = payload.newsImage
            
        if payload.magazineType is not None:
            updates["magazineType"] = payload.magazineType
            
        if payload.newsType is not None:
            updates["newsType"] = payload.newsType
            
        if payload.district_slug is not None:
            updates["district_slug"] = payload.district_slug
            
        if payload.source is not None:
            updates["source"] = payload.source
            
        # Handle status and isLive (admin/moderator only)
        if payload.status is not None:
            if user_role in ["admin", "moderator"]:
                updates["status"] = payload.status
            else:
                logger.warning(f"[UPDATE] Non-admin/moderator user tried to update status: {current_user.get('email')}")
                
        if payload.isLive is not None:
            if user_role in ["admin", "moderator"]:
                updates["isLive"] = payload.isLive
            else:
                logger.warning(f"[UPDATE] Non-admin/moderator user tried to update isLive: {current_user.get('email')}")
        
        # Always update last_updated timestamp
        updates["last_updated"] = datetime.utcnow()
        
        # Update in DB
        success = await get_db_service().update_news_fields(ObjectId(news_id), updates)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update news")
        
        # Get updated news
        updated_news = await get_db_service().get_news_by_id(ObjectId(news_id))
        response_doc = to_extended_json(updated_news)
        
        logger.info(f"[UPDATE] success news_id={news_id}")
        return NewsResponse(success=True, data=response_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[UPDATE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating news: {str(e)}")


@router.delete("/{news_id}", response_model=NewsResponse)
async def delete_news(
    news_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a news article."""
    try:
        logger.info(f"[DELETE] start news_id={news_id} user={current_user.get('email', 'unknown')}")
        
        # Check if news exists
        existing_news = await get_db_service().get_news_by_id(ObjectId(news_id))
        if not existing_news:
            raise HTTPException(status_code=404, detail="News not found")
        
        # Check permissions
        user_role = current_user.get("role", "")
        user_id = current_user.get("id")
        news_creator_id = existing_news.get("createdBy")
        
        # Admin can delete any news, others can only delete their own
        if user_role != "admin" and (not user_id or not news_creator_id or str(news_creator_id) != str(user_id)):
            raise HTTPException(status_code=403, detail="You can only delete your own news articles")
        
        # Delete the news document
        success = await get_db_service().delete_news(ObjectId(news_id))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete news")
        
        logger.info(f"[DELETE] success news_id={news_id}")
        return NewsResponse(success=True, data={"_id": {"$oid": news_id}, "deleted": True})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DELETE] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting news: {str(e)}")


@router.post("/translate", response_model=TranslationResponse)
async def translate_text(payload: TranslationRequest):
    """Translate text between languages"""
    try:
        logger.info(f"Received translation request: {payload.source_language} -> {payload.target_language}")
        logger.info(f"Request payload: {payload}")
        
        # Validate input
        if not payload.text.strip():
            logger.error("Empty text provided")
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if payload.source_language == payload.target_language:
            logger.info("Source and target languages are the same, returning original text")
            return TranslationResponse(
                success=True,
                data={
                    "original_text": payload.text,
                    "translated_text": payload.text,
                    "source_language": payload.source_language,
                    "target_language": payload.target_language,
                    "translation_time": 0.0
                }
            )

        start_time = datetime.utcnow()
        logger.info("Starting translation process...")
        
        # Add timeout for translation with retry logic
        try:
            logger.info("Calling translation_service.translate with retry...")
            translation_service = get_translation_service()
            
            # Use retry logic for direct translation
            timeout_sec = float(os.getenv("TRANSLATION_PER_CALL_TIMEOUT", "90.0"))
            translated_text = await retry_with_exponential_backoff(
                lambda: asyncio.wait_for(
                    asyncio.to_thread(
                        translation_service.translate,
                        payload.text,
                        payload.source_language,
                        payload.target_language
                    ),
                    timeout=timeout_sec
                ),
                max_retries=3,
                base_delay=1.0,
                max_delay=30.0
            )
            logger.info(f"Translation service returned: '{translated_text}'")
            
        except asyncio.TimeoutError:
            logger.error(f"Translation timed out after {timeout_sec}s")
            raise HTTPException(status_code=504, detail="Translation timed out")
        except Exception as e:
            logger.error(f"Translation service threw exception: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
        
        translation_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Translation completed in {translation_time:.2f} seconds")

        response = TranslationResponse(
            success=True,
            data={
                "original_text": payload.text,
                "translated_text": translated_text,
                "source_language": payload.source_language,
                "target_language": payload.target_language,
                "translation_time": translation_time
            }
        )
        
        logger.info(f"Returning response: {response}")
        return response
        
    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in translate_text: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Translation error: {str(e)}"
        )   
        

@router.post("/tts/generate", response_model=TTSResponse)
async def generate_tts(payload: TTSRequest):
    """Generate TTS audio"""
    try:
        audio_file = await asyncio.to_thread(get_tts_service().generate_audio, payload.text, payload.language)
        audio_url = await asyncio.to_thread(get_azure_blob_service().upload_audio, audio_file, payload.language) 
        duration = get_tts_service().get_audio_duration(audio_file)
        file_size = get_tts_service().get_file_size(audio_file)

        return TTSResponse(
            success=True,
            data={
                "audio_url": audio_url,
                "duration": duration,
                "file_size": file_size,
                "language": payload.language
            }
        )
    except Exception as e:
        logger.error(f"TTS endpoint error: {str(e)}", exc_info=True)  # ADD logging
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=True, # type: ignore
        database_connected=await get_db_service().is_connected(),
        azure_blob_connected=get_azure_blob_service().is_connected(),
        timestamp=datetime.utcnow(),
        version="1.0.0"
    )

# News Type Filtering Endpoints
@router.get("/news/getNewsByNewsType/articles", response_model=NewsResponse)
async def get_articles(
    date: str = None
):
    """Get all articles with optional date filter."""
    try:
        t0 = perf_counter()
        logger.info(f"[GET_ARTICLES] start date={date}")
        
        # Get all articles from database (no pagination)
        news_list, total = await get_db_service().get_news_paginated(
            skip=0,
            limit=10000,  # Large limit to get all results
            status_filter="approved",  # Only approved news
            news_type_filter="articles",
            date_filter=date
        )
        logger.info(
            "[GET_ARTICLES] fetched %s/%s items in %.3fs date=%s",
            len(news_list),
            total,
            perf_counter() - t0,
            date,
        )
        
        # Convert to extended JSON format
        news_list_json = [to_extended_json(doc) for doc in news_list]
        
        logger.info(f"[GET_ARTICLES] success - found {len(news_list)}/{total} articles")
        
        return NewsResponse(
            success=True,
            data={
                "news": news_list_json,
                "total": total
            }
        )
        
    except Exception as e:
        logger.error(f"[GET_ARTICLES] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching articles: {str(e)}")

@router.get("/news/getNewsByNewsType/specialnews", response_model=NewsResponse)
async def get_special_news(
    date: str = None
):
    """Get all special news with optional date filter."""
    try:
        t0 = perf_counter()
        logger.info(f"[GET_SPECIALNEWS] start date={date}")
        
        # Get all special news from database (no pagination)
        news_list, total = await get_db_service().get_news_paginated(
            skip=0,
            limit=10000,  # Large limit to get all results
            status_filter="approved",  # Only approved news
            news_type_filter="specialnews",
            date_filter=date
        )
        logger.info(
            "[GET_SPECIALNEWS] fetched %s/%s items in %.3fs date=%s",
            len(news_list),
            total,
            perf_counter() - t0,
            date,
        )
        
        # Convert to extended JSON format
        news_list_json = [to_extended_json(doc) for doc in news_list]
        
        logger.info(f"[GET_SPECIALNEWS] success - found {len(news_list)}/{total} special news")
        
        return NewsResponse(
            success=True,
            data={
                "news": news_list_json,
                "total": total
            }
        )
        
    except Exception as e:
        logger.error(f"[GET_SPECIALNEWS] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching special news: {str(e)}")

@router.get("/news/getNewsByNewsType/districtnews", response_model=NewsResponse)
async def get_district_news_by_type(
    date: str = None
):
    """Get all district news with optional date filter."""
    try:
        t0 = perf_counter()
        logger.info(f"[GET_DISTRICTNEWS] start date={date}")
        
        # Get all district news from database (no pagination)
        news_list, total = await get_db_service().get_news_paginated(
            skip=0,
            limit=10000,  # Large limit to get all results
            status_filter="approved",  # Only approved news
            news_type_filter="districtnews",
            date_filter=date
        )
        logger.info(
            "[GET_DISTRICTNEWS] fetched %s/%s items in %.3fs date=%s",
            len(news_list),
            total,
            perf_counter() - t0,
            date,
        )
        
        # Convert to extended JSON format
        news_list_json = [to_extended_json(doc) for doc in news_list]
        
        logger.info(f"[GET_DISTRICTNEWS] success - found {len(news_list)}/{total} district news")
        
        return NewsResponse(
            success=True,
            data={
                "news": news_list_json,
                "total": total
            }
        )
        
    except Exception as e:
        logger.error(f"[GET_DISTRICTNEWS] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching district news: {str(e)}")

@router.get("/news/getNewsByNewsType/statenews", response_model=NewsResponse)
async def get_state_news(
    date: str = None
):
    """Get all state news with optional date filter."""
    try:
        t0 = perf_counter()
        logger.info(f"[GET_STATENEWS] start date={date}")
        
        # Get all state news from database (no pagination)
        news_list, total = await get_db_service().get_news_paginated(
            skip=0,
            limit=10000,  # Large limit to get all results
            status_filter="approved",  # Only approved news
            news_type_filter="statenews",
            date_filter=date
        )
        logger.info(
            "[GET_STATENEWS] fetched %s/%s items in %.3fs date=%s",
            len(news_list),
            total,
            perf_counter() - t0,
            date,
        )
        
        # Convert to extended JSON format
        news_list_json = [to_extended_json(doc) for doc in news_list]
        
        logger.info(f"[GET_STATENEWS] success - found {len(news_list)}/{total} state news")
        
        return NewsResponse(
            success=True,
            data={
                "news": news_list_json,
                "total": total
            }
        )
        
    except Exception as e:
        logger.error(f"[GET_STATENEWS] error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching state news: {str(e)}")