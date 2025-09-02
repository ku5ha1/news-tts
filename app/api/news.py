from fastapi import APIRouter, HTTPException
from datetime import datetime
import uuid
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

@router.post("/create", response_model=NewsResponse)
async def create_news(payload: NewsCreateRequest):
    """Main endpoint for news creation with translation and TTS"""
    try:
        document_id = str(uuid.uuid4())
        source_lang = detect_language(payload.title + " " + payload.description)

        # Translate to all required languages asynchronously
        translations = await translation_service.translate_to_all(
            payload.title, payload.description, source_lang
        )

        # Generate TTS audio for each language concurrently
        async def process_lang(lang):
            if lang == source_lang:
                text = f"{payload.title}. {payload.description}"
            else:
                text = f"{translations[lang]['title']}. {translations[lang]['description']}"
            audio_file = await asyncio.to_thread(tts_service.generate_audio, text, lang)
            audio_url = firebase_service.upload_audio(audio_file, lang, document_id)
            return lang, audio_url

        audio_results = await asyncio.gather(*(process_lang(lang) for lang in ["en", "hi", "kn"]))
        audio_urls = dict(audio_results)

        news_document = {
            "_id": document_id,
            "title": payload.title,
            "description": payload.description,
            "newsImage": payload.newsImage,
            "category": payload.category,
            "author": payload.author,
            "publishedAt": payload.publishedAt,
            "isLive": True,
            "views": 0,
            "total_Likes": 0,
            "comments": [],
            "likedBy": [],
            "createdTime": datetime.utcnow(),
            "hindi": {
                "title": translations.get("hi", {}).get("title", payload.title),
                "description": translations.get("hi", {}).get("description", payload.description),
                "audio_description": audio_urls.get("hi", "")
            },
            "kannada": {
                "title": translations.get("kn", {}).get("title", payload.title),
                "description": translations.get("kn", {}).get("description", payload.description),
                "audio_description": audio_urls.get("kn", "")
            },
            "English": {
                "title": translations.get("en", {}).get("title", payload.title),
                "description": translations.get("en", {}).get("description", payload.description),
                "audio_description": audio_urls.get("en", "")
            }
        }

        inserted_id = await db_service.insert_news(news_document)
        news_document["_id"] = str(inserted_id)
        return NewsResponse(success=True, data=news_document)

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
        models_loaded=translation_service.is_models_loaded(), # type: ignore
        database_connected=await db_service.is_connected(),
        firebase_connected=firebase_service.is_connected(),
        timestamp=datetime.utcnow(),
        version="1.0.0"
    )
