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
    """Background job: generate TTS for langs, upload to Firebase, update Mongo doc."""
    try:
        async def process_lang(lang):
            try:
                if lang == source_lang:
                    text = f"{payload.title}. {payload.description}"
                else:
                    text = f"{translations.get(lang, {}).get('title', payload.title)}. {translations.get(lang, {}).get('description', payload.description)}"
                # Truncate excessive input to keep CPU inference bounded
                text = text[:1200]
                logger.info(f"[BG-TTS] Generating audio for {lang} (doc={document_id})")
                audio_file = await asyncio.to_thread(tts_service.generate_audio, text, lang)
                audio_url = await asyncio.to_thread(firebase_service.upload_audio, audio_file, lang, str(document_id))
                logger.info(f"[BG-TTS] Uploaded audio for {lang} -> {audio_url}")
                return lang, audio_url
            except Exception as e:
                logger.error(f"[BG-TTS] {lang} failed: {e}")
                return e

        audio_results = await asyncio.gather(*(process_lang(lang) for lang in ["en", "hi", "kn"]), return_exceptions=True)

        # Build update fields only for successes
        updates = {"last_updated": datetime.utcnow()}
        lang_map = {"hi": "hindi", "kn": "kannada", "en": "English"}
        any_success = False
        for item in audio_results:
            if isinstance(item, tuple) and len(item) == 2:
                lang, url = item
                field = f"{lang_map[lang]}.audio_description"
                updates[field] = url
                any_success = True

        # If at least one audio succeeded, set isLive true; else keep false but mark last_updated
        updates["isLive"] = any_success

        try:
            ok = await db_service.update_news_fields(document_id, updates)
            if not ok:
                logger.error(f"[BG-TTS] Mongo update returned modified_count=0 for doc={document_id}")
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
    """Create doc after doing translations synchronously; TTS runs in background.

    Returns the news document containing translated title/description fields so
    clients can use translations immediately while audio is generated later.
    """
    try:
        document_id = ObjectId()
        source_lang = detect_language(payload.title + " " + payload.description)

        # Translate synchronously so response contains translations, but cap time
        try:
            translations = await asyncio.wait_for(
                translation_service.translate_to_all_async(
                    payload.title,
                    payload.description,
                    source_lang,
                ),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            logger.warning("/create translation timed out; returning originals and continuing BG TTS")
            translations = {}
        except Exception as e:
            logger.error(f"/create translation failed: {e}")
            translations = {}

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

        # Insert into DB with timeout to avoid request hang
        try:
            await asyncio.wait_for(db_service.insert_news(news_document), timeout=15.0)
        except asyncio.TimeoutError:
            logger.error("Mongo insert_news timed out; proceeding to schedule BG tasks")
        except Exception as e:
            logger.error(f"Mongo insert_news failed: {e}")

        # Schedule only TTS (translations already done)
        background_tasks.add_task(_generate_and_attach_audio, document_id, payload, translations, source_lang)

        # Return immediately so LB doesn't timeout
        response_doc = _to_extended_json(news_document)
        return NewsResponse(success=True, data=response_doc)
    except Exception as e:
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
            logger.info("Calling translation_service.translate_async...")
            translated_text = await asyncio.wait_for(
                translation_service.translate_async(
                    payload.text, 
                    payload.source_language, 
                    payload.target_language
                ),
                timeout=300.0  # 5 minutes timeout
            )
            logger.info(f"Translation service returned: '{translated_text}'")
            
        except asyncio.TimeoutError:
            logger.error("Translation request timed out")
            raise HTTPException(
                status_code=504, 
                detail="Translation request timed out. This may happen during initial model loading."
            )
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
        audio_url = firebase_service.upload_audio(audio_file, payload.language)
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