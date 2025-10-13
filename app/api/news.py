from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime
from bson import ObjectId
import asyncio
import os
from app.models.news import (
    NewsCreateRequest, NewsResponse,
    TranslationRequest, TranslationResponse,
    TTSRequest, TTSResponse, HealthResponse
)
# REMOVED: Module-level import of translation_service - will import lazily
from app.services.tts_service import TTSService
from app.services.azure_blob_service import AzureBlobService
from app.services.db_service import DBService
from app.utils.language_detection import detect_language
import logging 

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services that are safe to import at module level
# Use lazy initialization to avoid module-level failures
tts_service = None
azure_blob_service = None
db_service = None

def get_tts_service():
    """Lazy import of TTS service to avoid module-level failures."""
    global tts_service
    if tts_service is None:
        try:
            tts_service = TTSService()
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
        except Exception as e:
            logger.error(f"Failed to initialize DB service: {e}")
            raise HTTPException(
                status_code=503, 
                detail=f"Database service unavailable: {str(e)}"
            )
    return db_service

def get_google_translate_service():
    """Lazy import of Google Translate service to avoid module-level failures."""
    try:
        from app.services.google_translate_service import google_translate_service
        return google_translate_service
    except Exception as e:
        logger.error(f"Failed to import Google Translate service: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"Google Translate service unavailable: {str(e)}"
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
    for key in ["publishedAt", "createdTime", "last_updated"]:
        if key in doc and isinstance(doc[key], datetime):
            doc[key] = dateify(doc[key])

    return doc


async def _generate_and_attach_audio(document_id: ObjectId, payload: NewsCreateRequest, translations: dict, source_lang: str):
    """Background job: generate TTS sequentially, upload to Firebase, update Mongo doc."""
    try:
        # Verify document exists
        existing_doc = await get_db_service().get_news_by_id(document_id)
        if not existing_doc:
            logger.warning(f"[BG-TTS] Document not found, skipping TTS generation doc={document_id}")
            return

        updates = {"last_updated": datetime.utcnow()}
        lang_map = {"hi": "hindi", "kn": "kannada", "en": "English"}
        any_success = False

        # Process languages sequentially
        for lang in ["en", "hi", "kn"]:
            try:
                if lang == source_lang:
                    text = f"{payload.title}. {payload.description}"
                else:
                    text = f"{translations.get(lang, {}).get('title', payload.title)}. " \
                           f"{translations.get(lang, {}).get('description', payload.description)}"

                text = text[:1200]  # truncate

                logger.info(f"[BG-TTS] Generating audio for {lang} (doc={document_id})")

                # Run blocking TTS & Azure Blob calls in separate threads
                audio_file = await asyncio.to_thread(get_tts_service().generate_audio, text, lang)
                audio_url = await asyncio.to_thread(get_azure_blob_service().upload_audio, audio_file, lang, str(document_id))

                # Update DB fields
                field = f"{lang_map[lang]}.audio_description"
                updates[field] = audio_url
                any_success = True
                logger.info(f"[BG-TTS] Completed audio for {lang} -> {audio_url}")

                await asyncio.sleep(2)  # slight delay between langs

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
#         except Exception as e:
#             logger.error(f"[BG] Mongo update (translations) failed for doc={document_id}: {e}")

#         # proceed to audio generation
#         await _generate_and_attach_audio(document_id, payload, translations, source_lang)
#     except Exception as e:
#         logger.error(f"[BG] translate+tts failed for doc={document_id}: {e}")


# def run_in_thread(coro):
#     """Run an async coroutine in a new event loop inside a thread."""
#     try:
#         asyncio.run(coro)
#     except Exception as e:
#         logger.error(f"[BG_TASK] error in background task: {e}")


@router.post("/create", response_model=NewsResponse)
async def create_news(payload: NewsCreateRequest, background_tasks: BackgroundTasks):
    """Create news doc; translation synchronous, TTS runs in background."""
    document_id = ObjectId()
    logger.info(f"[CREATE] start doc={document_id}")

    try:
        source_lang = detect_language(payload.title + " " + payload.description)
        logger.info(f"[CREATE] detected_language={source_lang} doc={document_id}")

        # Translation with timeout (no fallback)
        try:
            timeout_sec = float(os.getenv("TRANSLATION_PER_CALL_TIMEOUT", "90"))
        except Exception:
            timeout_sec = 90.0

        try:
            google_translate_service = get_google_translate_service()
            translations = await asyncio.wait_for(
                google_translate_service.translate_to_all_async(payload.title, payload.description, source_lang),
                timeout=timeout_sec
            )
            logger.info(f"[CREATE] Google Translate done langs={list(translations.keys())} doc={document_id}")
        except asyncio.TimeoutError:
            logger.error(f"[CREATE] Google Translate timed out after {timeout_sec}s for doc={document_id}")
            raise HTTPException(status_code=504, detail="Translation timed out")
        except Exception as e:
            logger.error(f"[CREATE] Google Translate failed for doc={document_id}: {e}")
            # Add specific guidance for Google Translate errors
            if "GOOGLE_TRANSLATE_API_KEY" in str(e):
                logger.error("This appears to be a Google Translate API key issue")
            raise

        # Create news document
        news_document = {
            "_id": document_id,
            "title": payload.title,
            "description": payload.description,
            "newsImage": payload.newsImage,
            "category": ObjectId(payload.category) if isinstance(payload.category, str) and len(payload.category) == 24 else payload.category,
            "author": payload.author,
            "publishedAt": payload.publishedAt,
            "magazineType": payload.magazineType,
            "newsType": payload.newsType,
            "isLive": False,
            "views": 0,
            "total_Likes": 0,
            "comments": [],
            "likedBy": [],
            "createdTime": datetime.utcnow(),
            "last_updated": datetime.utcnow(),
            "hindi": {
                "title": translations.get("hi", {}).get("title", payload.title),
                "description": translations.get("hi", {}).get("description", payload.description),
                "audio_description": "",
            },
            "kannada": {
                "title": translations.get("kn", {}).get("title", payload.title),
                "description": translations.get("kn", {}).get("description", payload.description),
                "audio_description": "",
            },
            "English": {
                "title": translations.get("en", {}).get("title", payload.title),
                "description": translations.get("en", {}).get("description", payload.description),
                "audio_description": "",
            },
        }

        # Insert into DB
        await asyncio.wait_for(get_db_service().insert_news(news_document), timeout=15.0)

        # Schedule TTS in background
        background_tasks.add_task(
            _generate_and_attach_audio, document_id, payload, translations, source_lang
        )

        response_doc = _to_extended_json(news_document)
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
        
        # Add timeout for translation (5 minutes for model loading)
        try:
            logger.info("Calling translation_service.translate...")
            translation_service = get_translation_service()
            translated_text = translation_service.translate(
                payload.text, 
                payload.source_language, 
                payload.target_language
            )
            logger.info(f"Translation service returned: '{translated_text}'")
            
        except Exception as e:
            logger.error(f"Translation service threw exception: {str(e)}")
            raise
        
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