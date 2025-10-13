import os
import asyncio
import logging
import httpx
from typing import Dict

logger = logging.getLogger(__name__)

class GoogleTranslateService:
    """Google Cloud Translate service for news translation using REST API."""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GOOGLE_TRANSLATE_API_KEY environment variable not set")
        
        self.base_url = "https://translation.googleapis.com/language/translate/v2"
        logger.info("[GoogleTranslate] Service initialized successfully")
    
    def _normalize_lang_code(self, lang: str) -> str:
        """Normalize language codes to Google Translate format."""
        lang = lang.lower().strip()
        
        # Map internal language codes to Google Translate codes
        lang_map = {
            "en": "en",
            "english": "en", 
            "hi": "hi",
            "hindi": "hi",
            "kn": "kn", 
            "kannada": "kn"
        }
        
        return lang_map.get(lang, "en")
    
    async def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using Google Cloud Translate REST API."""
        try:
            source_code = self._normalize_lang_code(source_lang)
            target_code = self._normalize_lang_code(target_lang)
            
            if source_code == target_code:
                return text
            
            logger.info(f"[GoogleTranslate] Translating from {source_code} to {target_code}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    params={"key": self.api_key},
                    json={
                        "q": text,
                        "source": source_code,
                        "target": target_code,
                        "format": "text"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    translated_text = result["data"]["translations"][0]["translatedText"]
                    logger.info(f"[GoogleTranslate] Translation successful: '{text[:50]}...' -> '{translated_text[:50]}...'")
                    return translated_text
                else:
                    logger.error(f"[GoogleTranslate] API error: {response.status_code} - {response.text}")
                    raise Exception(f"Translation API error: {response.status_code}")
            
        except Exception as e:
            logger.error(f"[GoogleTranslate] Translation failed {source_lang}->{target_lang}: {e}")
            raise
    
    async def translate_to_all_async(self, title: str, description: str, source_lang: str) -> Dict[str, Dict[str, str]]:
        """Translate title and description to all target languages."""
        try:
            source_lang = self._normalize_lang_code(source_lang)
            logger.info(f"[GoogleTranslate] Starting batch translation from {source_lang}")
            
            # Define target languages based on source
            if source_lang == "en":
                target_languages = ["hi", "kn"]
            elif source_lang == "hi":
                target_languages = ["en", "kn"]
            elif source_lang == "kn":
                target_languages = ["en", "hi"]
            else:
                # Default: translate to Hindi and Kannada
                target_languages = ["hi", "kn"]
            
            results = {}
            
            # Translate title and description separately for better accuracy
            for target_lang in target_languages:
                try:
                    # Translate title separately
                    translated_title = await self._translate_text(
                        title,
                        source_lang,
                        target_lang
                    )
                    
                    # Translate description separately
                    translated_description = await self._translate_text(
                        description,
                        source_lang,
                        target_lang
                    )
                    
                    results[target_lang] = {
                        "title": translated_title,
                        "description": translated_description
                    }
                    
                    logger.info(f"[GoogleTranslate] Completed translation to {target_lang}")
                    
                except Exception as e:
                    logger.error(f"[GoogleTranslate] Failed to translate to {target_lang}: {e}")
                    # Use original text as fallback
                    results[target_lang] = {
                        "title": title,
                        "description": description
                    }
            
            # Add source language if not already included
            if source_lang not in results:
                results[source_lang] = {
                    "title": title,
                    "description": description
                }
            
            logger.info(f"[GoogleTranslate] Batch translation completed: {list(results.keys())}")
            return results
            
        except Exception as e:
            logger.error(f"[GoogleTranslate] Batch translation failed: {e}")
            raise

# Create singleton instance
google_translate_service = GoogleTranslateService()
