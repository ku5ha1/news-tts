from fastapi import APIRouter, HTTPException
from app.models.news import NewsCreateRequest, NewsResponse
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

@router.post("/createNews", response_model=NewsResponse)
async def create_news(payload: NewsCreateRequest):
    # 1. Detect language if not provided
    source_lang = payload.source_language or detect_language(payload.title + " " + payload.description)
    
    # 2. Translate to other two languages
    translations = translation_service.translate_to_all(payload.title, payload.description, source_lang)
    
    # 3. Generate audio (placeholders now)
    audio_urls = {}
    for lang, text in translations.items():
        audio_file = tts_service.generate_audio(text['title'] + " " + text['description'], lang)
        audio_urls[lang] = firebase_service.upload_audio(audio_file, lang)

    # 4. Save to MongoDB
    news_document = {
        "source_language": source_lang,
        "translations": {
            lang: {
                "title": translations[lang]['title'],
                "description": translations[lang]['description'],
                "audio_url": audio_urls[lang]
            } for lang in translations
        }
    }
    inserted_id = await db_service.insert_news(news_document)

    return NewsResponse(id=str(inserted_id), **news_document)
