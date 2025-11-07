#!/usr/bin/env python3
"""
Re-translate and regenerate TTS for specific documents
Fixes truncation issues in Hindi/Kannada translations

Usage:
    python app/scripts/retranslate_document.py <document_id>
    
Example:
    python app/scripts/retranslate_document.py 6900ad4d663742d6666f0a33
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from bson import ObjectId

# Add project root to path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.db_service import DBService
from app.services.translation_service import translation_service
from app.services.tts_service import TTSService
from app.services.azure_blob_service import AzureBlobService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("retranslate_document")


class DocumentRetranslator:
    """Re-translate and regenerate TTS for specific documents"""
    
    def __init__(self):
        self.db_service = DBService()
        self.db = self.db_service.db
        self.news_collection = self.db["news"]
        
        try:
            self.tts_service = TTSService()
            logger.info("✓ TTS Service initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize TTS Service: {e}")
            raise
        
        try:
            self.blob_service = AzureBlobService()
            logger.info("✓ Azure Blob Service initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize Blob Service: {e}")
            raise
    
    async def fetch_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Fetch document from database by ID"""
        try:
            document = await self.news_collection.find_one({"_id": ObjectId(doc_id)})
            if not document:
                logger.error(f"✗ Document not found: {doc_id}")
                return None
            
            logger.info(f"✓ Document fetched: {doc_id}")
            logger.info(f"  Title: {document.get('title', 'N/A')[:50]}...")
            return document
        except Exception as e:
            logger.error(f"✗ Failed to fetch document {doc_id}: {e}")
            return None
    
    async def retranslate_document(self, doc_id: str) -> bool:
        """Re-translate document and regenerate TTS audio"""
        logger.info("=" * 70)
        logger.info(f"Processing document: {doc_id}")
        logger.info("=" * 70)
        
        # Fetch document
        document = await self.fetch_document(doc_id)
        if not document:
            return False
        
        # Get English content
        english_title = document.get("title", "")
        english_description = document.get("description", "")
        
        if not english_title or not english_description:
            logger.error("✗ Missing English title or description")
            return False
        
        logger.info(f"English content:")
        logger.info(f"  Title length: {len(english_title)} chars")
        logger.info(f"  Description length: {len(english_description)} chars")
        
        # Step 1: Re-translate
        logger.info("\n[Step 1/3] Re-translating content...")
        try:
            translations = await translation_service.translate_to_all_async(
                title=english_title,
                description=english_description,
                source_lang="english"
            )
            logger.info("✓ Translation completed")
            
            # Log translation results
            for lang, content in translations.items():
                title_len = len(content.get("title", ""))
                desc_len = len(content.get("description", ""))
                logger.info(f"  {lang.capitalize()}: title={title_len} chars, desc={desc_len} chars")
        
        except Exception as e:
            logger.error(f"✗ Translation failed: {e}")
            return False
        
        # Step 2: Generate TTS audio for both languages
        logger.info("\n[Step 2/3] Generating TTS audio...")
        
        audio_urls = {}
        lang_code_map = {"hindi": "hi", "kannada": "kn"}
        
        for lang_name, lang_code in lang_code_map.items():
            if lang_name not in translations:
                logger.warning(f"⚠ Skipping {lang_name}: not in translations")
                continue
            
            translated_desc = translations[lang_name].get("description", "")
            if not translated_desc:
                logger.warning(f"⚠ Skipping {lang_name}: empty description")
                continue
            
            try:
                logger.info(f"  Generating {lang_name} audio...")
                
                # Generate audio file
                audio_path = self.tts_service.generate_audio(translated_desc, lang_code)
                
                # Upload to Azure Blob
                blob_path = f"{doc_id}/{lang_name}-{Path(audio_path).stem}.mp3"
                audio_url = self.blob_service.upload_audio(audio_path, blob_path)
                
                audio_urls[lang_name] = audio_url
                logger.info(f"  ✓ {lang_name} audio uploaded: {audio_url}")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to generate {lang_name} audio: {e}")
                # Continue with other languages even if one fails
        
        # Step 3: Update database
        logger.info("\n[Step 3/3] Updating database...")
        
        update_data = {}
        
        # Update Hindi
        if "hindi" in translations:
            update_data["hindi.title"] = translations["hindi"]["title"]
            update_data["hindi.description"] = translations["hindi"]["description"]
            if "hindi" in audio_urls:
                update_data["hindi.audio_description"] = audio_urls["hindi"]
        
        # Update Kannada
        if "kannada" in translations:
            update_data["kannada.title"] = translations["kannada"]["title"]
            update_data["kannada.description"] = translations["kannada"]["description"]
            if "kannada" in audio_urls:
                update_data["kannada.audio_description"] = audio_urls["kannada"]
        
        try:
            result = await self.news_collection.update_one(
                {"_id": ObjectId(doc_id)},
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                logger.info(f"✓ Database updated successfully")
                logger.info(f"  Modified fields: {len(update_data)}")
                return True
            else:
                logger.warning("⚠ No changes made to database")
                return False
                
        except Exception as e:
            logger.error(f"✗ Database update failed: {e}")
            return False


async def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python app/scripts/retranslate_document.py <document_id>")
        print("\nExample:")
        print("  python app/scripts/retranslate_document.py 6900ad4d663742d6666f0a33")
        sys.exit(1)
    
    doc_id = sys.argv[1].strip()
    
    # Validate ObjectId format
    try:
        ObjectId(doc_id)
    except Exception:
        logger.error(f"✗ Invalid document ID format: {doc_id}")
        logger.error("  Document ID must be a valid 24-character hex string")
        sys.exit(1)
    
    logger.info("Document Re-translation and TTS Regeneration Script")
    logger.info(f"Document ID: {doc_id}")
    logger.info("")
    
    try:
        retranslator = DocumentRetranslator()
        success = await retranslator.retranslate_document(doc_id)
        
        logger.info("")
        logger.info("=" * 70)
        if success:
            logger.info("✓ SUCCESS: Document retranslated and TTS regenerated")
            logger.info("=" * 70)
            sys.exit(0)
        else:
            logger.info("✗ FAILED: Document processing failed")
            logger.info("=" * 70)
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

