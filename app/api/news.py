from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime
from bson import ObjectId
import asyncio
from app.models.news import (
    NewsCreateRequest, NewsResponse,
    TranslationRequest, TranslationResponse,
    TTSRequest, TTSResponse, HealthResponse
)
from app.services.translation_service import translation_service
from app.services.tts_service import TTSService
from app.services.firebase_service import FirebaseService
from app.services.db_service import DBService
from app.utils.language_detection import detect_language
import logging 

logger = logging.getLogger(__name__)

router = APIRouter()

tts_service = TTSService()
firebase_service = FirebaseService()
db_service = DBService()

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
        # First, verify the document exists
        existing_doc = await db_service.get_news_by_id(document_id)
        if not existing_doc:
            logger.warning(f"[BG-TTS] Document not found, skipping TTS generation doc={document_id}")
            return
            
        updates = {"last_updated": datetime.utcnow()}
        lang_map = {"hi": "hindi", "kn": "kannada", "en": "English"}
        any_success = False
        
        # Process languages SEQUENTIALLY with reasonable timeouts
        for lang in ["en", "hi", "kn"]:
            try:
                if lang == source_lang:
                    text = f"{payload.title}. {payload.description}"
                else:
                    text = f"{translations.get(lang, {}).get('title', payload.title)}. {translations.get(lang, {}).get('description', payload.description)}"
                
                # Truncate excessive input
                text = text[:1200]
                logger.info(f"[BG-TTS] Generating audio for {lang} (doc={document_id})")
                
                # Generate audio with timeout
                audio_file = await asyncio.wait_for(
                    asyncio.to_thread(tts_service.generate_audio, text, lang), 
                    timeout=45.0
                )
                
                # Upload to Firebase with timeout
                audio_url = await asyncio.wait_for(
                    asyncio.to_thread(firebase_service.upload_audio, audio_file, lang, str(document_id)), 
                    timeout=30.0
                )
                
                # Update success
                field = f"{lang_map[lang]}.audio_description"
                updates[field] = audio_url
                any_success = True
                logger.info(f"[BG-TTS] Completed audio for {lang} -> {audio_url}")
                
                # Add delay between requests to respect rate limits
                await asyncio.sleep(2)  # Reduced from 10s to 2s
                    
            except asyncio.TimeoutError:
                logger.error(f"[BG-TTS] {lang} timed out for doc={document_id}")
                continue
            except Exception as e:
                logger.error(f"[BG-TTS] {lang} failed for doc={document_id}: {e}")
                continue

        # Set isLive only if at least one audio succeeded
        updates["isLive"] = any_success 

        # Update MongoDB
        try:
            ok = await db_service.update_news_fields(document_id, updates)
            if not ok:
                logger.error(f"[BG-TTS] Mongo update returned modified_count=0 for doc={document_id}")
            else:
                logger.info(f"[BG-TTS] Successfully updated document {document_id} with audio URLs")
        except Exception as e:
            logger.error(f"[BG-TTS] Mongo update failed for doc={document_id}: {e}")
            
    except Exception as e:
        logger.error(f"Background TTS job failed for {document_id}: {e}")



async def _translate_and_attach(document_id: ObjectId, payload: NewsCreateRequest, source_lang: str):
    """Background: translate to target langs, update doc, then generate TTS and attach URLs."""
    try:
        translations = await translation_service.translate_to_all_async(
            payload.title,
            payload.description,
            source_lang,
        )

        updates = {
            "hindi.title": translations.get("hi", {}).get("title", payload.title),
            "hindi.description": translations.get("hi", {}).get("description", payload.description),
            "kannada.title": translations.get("kn", {}).get("title", payload.title),
            "kannada.description": translations.get("kn", {}).get("description", payload.description),
            "English.title": translations.get("en", {}).get("title", payload.title),
            "English.description": translations.get("en", {}).get("description", payload.description),
            "last_updated": datetime.utcnow(),
        }

        try:
            await db_service.update_news_fields(document_id, updates)
        except Exception as e:
            logger.error(f"[BG] Mongo update (translations) failed for doc={document_id}: {e}")

        # proceed to audio generation
        await _generate_and_attach_audio(document_id, payload, translations, source_lang)
    except Exception as e:
        logger.error(f"[BG] translate+tts failed for doc={document_id}: {e}")

@router.post("/create", response_model=NewsResponse)
async def create_news(payload: NewsCreateRequest, background_tasks: BackgroundTasks):
    """Create doc after doing translations synchronously; TTS runs in background."""
    document_id = ObjectId()
    logger.info(f"[CREATE] start doc={document_id}")
    
    try:
        source_lang = detect_language(payload.title + " " + payload.description)
        logger.info(f"[CREATE] detected_language={source_lang} doc={document_id}")

        # Initialize translations with original text as fallback
        translations = {}
        
        # Try to translate with reasonable timeout
        try:
            logger.info(f"[CREATE] translation.start doc={document_id}")
            translations = await asyncio.wait_for(
                translation_service.translate_to_all_async(
                    payload.title,
                    payload.description,
                    source_lang,
                ),
                timeout=20.0,
            )
            logger.info(f"[CREATE] translation.done langs={list(translations.keys()) if translations else []} doc={document_id}")
        except asyncio.TimeoutError:
            logger.warning(f"[CREATE] translation.timeout doc={document_id}; using original text")
            # Use original text as fallback
            translations = {
                "hi": {"title": payload.title, "description": payload.description},
                "kn": {"title": payload.title, "description": payload.description},
                "en": {"title": payload.title, "description": payload.description}
            }
        except Exception as e:
            logger.error(f"[CREATE] translation.failed doc={document_id} error={e}")
            # Use original text as fallback
            translations = {
                "hi": {"title": payload.title, "description": payload.description},
                "kn": {"title": payload.title, "description": payload.description},
                "en": {"title": payload.title, "description": payload.description}
            }

        # Create the news document FIRST - this ensures no duplication
        news_document = {
            "_id": document_id,
            "title": payload.title,
            "description": payload.description,
            "newsImage": payload.newsImage,
            "category": ObjectId(payload.category) if isinstance(payload.category, str) and len(payload.category) == 24 else payload.category,
            "author": payload.author,
            "publishedAt": payload.publishedAt,
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

        # Insert into DB - this is critical and should not be skipped
        try:
            logger.info(f"[CREATE] db.insert.start doc={document_id}")
            await asyncio.wait_for(db_service.insert_news(news_document), timeout=15.0)
            logger.info(f"[CREATE] db.insert.done doc={document_id}")
        except asyncio.TimeoutError:
            logger.error(f"[CREATE] db.insert.timeout doc={document_id}; document may not be saved")
            raise HTTPException(status_code=500, detail="Database insert timed out")
        except Exception as e:
            logger.error(f"[CREATE] db.insert.failed doc={document_id} error={e}")
            raise HTTPException(status_code=500, detail=f"Database insert failed: {str(e)}")

        # Schedule background TTS generation ONLY after successful DB insert
        logger.info(f"[CREATE] bg_tts.schedule doc={document_id}")
        background_tasks.add_task(_generate_and_attach_audio, document_id, payload, translations, source_lang)

        # Return immediately so LB doesn't timeout
        response_doc = _to_extended_json(news_document)
        logger.info(f"[CREATE] response.return doc={document_id}")
        return NewsResponse(success=True, data=response_doc)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"[CREATE] failed doc={document_id} error={e}")
        # Clean up: try to delete the document if it was partially created
        try:
            await db_service.update_news_fields(document_id, {"_deleted_due_to_error": True})
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
        audio_file = await asyncio.to_thread(tts_service.generate_audio, payload.text, payload.language)
        audio_url = await asyncio.to_thread(firebase_service.upload_audio, audio_file, payload.language)  # ADD await asyncio.to_thread
        duration = tts_service.get_audio_duration(audio_file)
        file_size = tts_service.get_file_size(audio_file)

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
        models_loaded=translation_service.is_models_loaded, # type: ignore
        database_connected=await db_service.is_connected(),
        firebase_connected=firebase_service.is_connected(),
        timestamp=datetime.utcnow(),
        version="1.0.0"
    )