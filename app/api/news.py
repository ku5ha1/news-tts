from fastapi import APIRouter, HTTPException
from datetime import datetime
import uuid
from app.models.news import NewsCreateRequest, NewsResponse, TranslationRequest, TranslationResponse, TTSRequest, TTSResponse, HealthResponse
from app.services.translation_service import TranslationService
from app.services.tts_service import TTSService
from app.services.firebase_service import FirebaseService
from app.services.db_service import DBService
from app.utils.language_detection import detect_language

router = APIRouter()

translation_service = TranslationService()
tts_service = TTSService()
firebase_service = FirebaseService()
db_service = DBService()

@router.post("/create", response_model=NewsResponse)
async def create_news(payload: NewsCreateRequest):
    """Main endpoint for news creation with translation and TTS"""
    try:
        # Generate document ID for organizing audio files
        document_id = str(uuid.uuid4())
        
        # 1. Detect language if not provided
        source_lang = detect_language(payload.title + " " + payload.description)
        
        # 2. Translate to all languages (en, hi, kn)
        translations = translation_service.translate_to_all(payload.title, payload.description, source_lang)
        
        # 3. Generate audio for each language
        audio_urls = {}
        for lang in ["en", "hi", "kn"]:
            if lang == source_lang:
                # Use original text for source language
                text = f"{payload.title}. {payload.description}"
            else:
                # Use translated text
                text = f"{translations[lang]['title']}. {translations[lang]['description']}"
            
            # Generate audio
            audio_file = tts_service.generate_audio(text, lang)
            # Upload to Firebase with document ID for organization
            audio_url = firebase_service.upload_audio(audio_file, lang, document_id)
            audio_urls[lang] = audio_url
        
        # 4. Prepare news document for MongoDB
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
        
        # 5. Save to MongoDB
        inserted_id = await db_service.insert_news(news_document)
        news_document["_id"] = str(inserted_id)
        
        return NewsResponse(success=True, data=news_document)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating news: {str(e)}")

@router.post("/translate", response_model=TranslationResponse)
async def translate_text(payload: TranslationRequest):
    """Translate text between languages"""
    try:
        start_time = datetime.utcnow()
        
        translated_text = translation_service._translate(
            payload.text, 
            payload.source_language, 
            payload.target_language
        )
        
        end_time = datetime.utcnow()
        translation_time = (end_time - start_time).total_seconds()
        
        return TranslationResponse(
            success=True,
            data={
                "original_text": payload.text,
                "translated_text": translated_text,
                "source_language": payload.source_language,
                "target_language": payload.target_language,
                "translation_time": translation_time
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

@router.post("/tts/generate", response_model=TTSResponse)
async def generate_tts(payload: TTSRequest):
    """Generate TTS audio"""
    try:
        # Generate audio
        audio_file = tts_service.generate_audio(payload.text, payload.language)
        
        # Upload to Firebase
        audio_url = firebase_service.upload_audio(audio_file, payload.language)
        
        # Get audio metadata
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
        models_loaded=True,  # Translation models are loaded
        database_connected=await db_service.is_connected(),
        firebase_connected=firebase_service.is_connected(),
        timestamp=datetime.utcnow(),
        version="1.0.0"
    )
