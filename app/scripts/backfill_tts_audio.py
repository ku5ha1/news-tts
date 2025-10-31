import os
import sys
import time
import uuid
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict

from dotenv import load_dotenv
from bson import ObjectId
from pymongo import MongoClient

# Ensure project root is on sys.path when running as a script
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.tts_service import TTSService  # noqa: E402
from app.services.azure_blob_service import AzureBlobService  # noqa: E402

try:
    import azure.cognitiveservices.speech as speechsdk  # noqa: E402
    AZURE_TTS_AVAILABLE = True
except ImportError:
    AZURE_TTS_AVAILABLE = False


log = logging.getLogger("backfill_tts")


LANG_KEY_MAP: Dict[str, str] = {"en": "English", "hi": "hindi", "kn": "kannada"}

AZURE_VOICE_MAP = {
    "en": "en-IN-KavyaNeural",
    "hi": "hi-IN-SwaraNeural",
    "kn": "kn-IN-SapnaNeural",
}

AZURE_LOCALE_MAP = {
    "en": "en-IN",
    "hi": "hi-IN",
    "kn": "kn-IN",
}


def _get_text_for_language(document: dict, lang: str) -> Optional[str]:
    """Build input text for a given language from document fields.

    - English: prefer `English.title/description`, fallback to root `title/description`.
    - Hindi/Kannada: require the corresponding sub-document; skip if missing.
    """
    key = LANG_KEY_MAP[lang]
    if lang == "en":
        title = (
            (document.get("English") or {}).get("title")
            or document.get("title")
        )
        desc = (
            (document.get("English") or {}).get("description")
            or document.get("description")
        )
    else:
        sub = document.get(key) or {}
        title = sub.get("title")
        desc = sub.get("description")

    if not title and not desc:
        return None

    combined = f"{title or ''}. {desc or ''}".strip()
    # Align with service behavior: cap to around 1200 chars
    return combined[:1200] if combined else None


def _azure_tts_synthesize(
    text: str, lang: str, output_path: str, speech_key: str, speech_region: str
) -> str:
    """Synthesize text to speech using Azure TTS and save to MP3 file."""
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    
    # Set voice directly in SpeechConfig instead of using SSML
    voice_name = AZURE_VOICE_MAP.get(lang, "en-IN-KavyaNeural")
    speech_config.speech_synthesis_voice_name = voice_name
    
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )

    # Use plain text synthesis instead of SSML
    result = synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        log.info(f"Azure TTS synthesized {lang} ({len(text)} chars)")
        return output_path
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speechsdk.CancellationDetails(result)
        raise RuntimeError(f"Azure TTS cancelled: {cancellation_details.reason}")
    else:
        raise RuntimeError(f"Azure TTS failed: {result.reason}")


def _needs_audio(document: dict, lang: str) -> bool:
    key = LANG_KEY_MAP[lang]
    sub = document.get(key) or {}
    audio_url = sub.get("audio_description", "")
    return audio_url == ""  # Only empty strings, not missing fields


def _has_any_missing_audio(document: dict, langs: list[str]) -> bool:
    """Check if document has any missing audio fields for the specified languages."""
    return any(_needs_audio(document, lang) for lang in langs)


def backfill(
    mongo_uri: str,
    database_name: str,
    collection_name: str,
    voice_id: str,
    langs: list[str],
    dry_run: bool = False,
    force_overwrite: bool = False,
    voice_id_en: Optional[str] = None,
    voice_id_hi: Optional[str] = None,
    voice_id_kn: Optional[str] = None,
    max_chars_en: int = 1200,
    max_chars_hi: int = 1200,
    max_chars_kn: int = 800,
    sleep_between_langs: float = 0.0,
    sleep_between_docs: float = 0.0,
    provider: str = "elevenlabs",  # elevenlabs or azure
    limit: Optional[int] = None,
    resume_from: Optional[str] = None,
) -> None:
    load_dotenv()

    # Initialize Azure Blob service
    azure = AzureBlobService()
    if not azure.is_connected():
        raise RuntimeError("Azure Blob Storage not connected - check env vars")

    # Initialize TTS provider
    if provider == "azure":
        if not AZURE_TTS_AVAILABLE:
            raise RuntimeError("Azure TTS not available - install azure-cognitiveservices-speech")
        speech_key = os.getenv("AZURE_SPEECH_KEY")
        speech_region = os.getenv("AZURE_SPEECH_REGION")
        if not speech_key or not speech_region:
            raise RuntimeError("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION env vars required")
        tts_service = None  # Will use _azure_tts_synthesize directly
    else:  # elevenlabs
        os.environ["ELEVENLABS_VOICE_ID"] = voice_id
        tts = TTSService()
        if voice_id_en:
            tts.voice_mapping["en"] = voice_id_en
        if voice_id_hi:
            tts.voice_mapping["hi"] = voice_id_hi
        if voice_id_kn:
            tts.voice_mapping["kn"] = voice_id_kn
        tts_service = tts

    # Mongo (sync) for simple batch updates
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    db = client[database_name]
    coll = db[collection_name]

    # Count documents with empty audio fields (not missing, just empty)
    # Query for documents with missing audio fields
    # For Kannada regeneration, we want ALL documents that have Kannada content
    if "kn" in langs and len(langs) == 1:
        # Special case: if only Kannada is specified, get ALL documents with Kannada content
        missing_audio_query = {
            f"{LANG_KEY_MAP['kn']}.title": {"$exists": True, "$ne": ""},
            f"{LANG_KEY_MAP['kn']}.description": {"$exists": True, "$ne": ""}
        }
        log.info("Kannada-only mode: Processing ALL documents with Kannada content for regeneration")
    else:
        # Normal mode: only documents with missing audio
        missing_audio_query = {
            "$or": [
                {f"{LANG_KEY_MAP[lang]}.audio_description": ""}
                for lang in langs
            ]
        }
    
    # Add resume condition if specified
    if resume_from:
        try:
            resume_id = ObjectId(resume_from)
            missing_audio_query["_id"] = {"$gt": resume_id}
            log.info(f"Resuming from document ID: {resume_from}")
        except Exception as e:
            log.error(f"Invalid resume_from ID: {e}")
            raise
    
    total_missing = coll.count_documents(missing_audio_query)
    total_docs = coll.count_documents({})
    
    if "kn" in langs and len(langs) == 1:
        log.info(f"Kannada regeneration mode: Found {total_missing} documents with Kannada content out of {total_docs} total documents")
    else:
        log.info(f"Found {total_missing} documents with missing audio out of {total_docs} total documents")
    
    if limit:
        log.info(f"TESTING MODE: Processing only {limit} document(s)")

    processed = 0
    updated = 0
    failed = 0

    # Process in batches to avoid cursor timeout
    batch_size = 20
    skip = 0
    
    while True:
        try:
            # Get batch of documents with missing audio
            docs = list(coll.find(missing_audio_query, projection=None)
                       .skip(skip)
                       .limit(batch_size))
            
            if not docs:
                log.info("No more documents to process")
                break
                
            log.info(f"Processing batch {skip//batch_size + 1}: {len(docs)} documents")
            
            for doc in docs:
                if limit and processed >= limit:
                    break
                    
                processed += 1
                doc_id = doc.get("_id")
                doc_id_str = str(doc_id)

                # Skip if document doesn't have any missing audio for our target languages
                # Exception: For Kannada-only mode, process all documents regardless
                if not _has_any_missing_audio(doc, langs) and not ("kn" in langs and len(langs) == 1):
                    log.info(f"[{doc_id_str}] Skipping - no missing audio for target languages")
                    continue

                updates: Dict[str, object] = {}
                doc_updated = False

                for lang in langs:
                    try:
                        # For Kannada-only mode, always regenerate even if audio exists
                        if not _needs_audio(doc, lang):
                            if not force_overwrite and not ("kn" in langs and len(langs) == 1):
                                continue

                        text = _get_text_for_language(doc, lang)
                        if not text:
                            log.info(f"[{doc_id_str}] Skip {lang}: no text available")
                            continue

                        # Per-language truncation
                        if lang == "en":
                            text = text[:max_chars_en]
                        elif lang == "hi":
                            text = text[:max_chars_hi]
                        elif lang == "kn":
                            text = text[:max_chars_kn]

                        log.info(f"[{doc_id_str}] Generating TTS for {lang} ({len(text)} chars)")

                        # Simple retry loop (2 retries -> 3 total attempts)
                        attempts = 3
                        audio_file_path: Optional[str] = None
                        temp_path = Path("/tmp") / f"audio_{lang}_{uuid.uuid4().hex[:8]}.mp3"
                        
                        for attempt in range(1, attempts + 1):
                            try:
                                if provider == "azure":
                                    audio_file_path = _azure_tts_synthesize(text, lang, str(temp_path), speech_key, speech_region)
                                else:  # elevenlabs
                                    audio_file_path = tts_service.generate_audio(text, lang)
                                break
                            except Exception as e:
                                log.warning(
                                    f"[{doc_id_str}] TTS attempt {attempt}/{attempts} failed for {lang}: {e}"
                                )
                                if attempt < attempts:
                                    time.sleep(2 * attempt)

                        if not audio_file_path:
                            failed += 1
                            log.error(f"[{doc_id_str}] TTS failed for {lang} after retries")
                            continue

                        # Upload to Azure
                        existing_url = ((doc.get(LANG_KEY_MAP[lang]) or {}).get("audio_description") or "").strip()
                        
                        # For Kannada-only mode, always overwrite existing audio
                        if existing_url and (force_overwrite or ("kn" in langs and len(langs) == 1)):
                            # Overwrite same blob path to keep URL stable
                            audio_url = existing_url
                            if not dry_run:
                                audio_url = azure.upload_audio_to_existing_url(audio_file_path, existing_url)
                            log.info(f"[{doc_id_str}] {lang} overwritten at same URL")
                            # No need to set field if URL unchanged; still bump last_updated below
                        else:
                            audio_url = azure.upload_audio(audio_file_path, lang, document_id=doc_id_str)
                            field_path = f"{LANG_KEY_MAP[lang]}.audio_description"
                            updates[field_path] = audio_url
                            log.info(f"[{doc_id_str}] {lang} -> {audio_url}")

                        doc_updated = True

                        # Best-effort local cleanup
                        try:
                            Path(audio_file_path).unlink(missing_ok=True)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        if sleep_between_langs:
                            time.sleep(sleep_between_langs)
                    except Exception as e:
                        failed += 1
                        log.exception(f"[{doc_id_str}] Error processing {lang}: {e}")

                if doc_updated and updates:
                    log.info(f"[{doc_id_str}] Updating {len(updates)} field(s)")
                    if not dry_run:
                        # Use $currentDate to set last_updated server-side
                        update_ops = {"$set": updates, "$currentDate": {"last_updated": True}}
                        coll.update_one({"_id": doc_id}, update_ops)
                    updated += 1

                if sleep_between_docs:
                    time.sleep(sleep_between_docs)
            
            # Move to next batch
            skip += batch_size
            
            # Small delay between batches to avoid overwhelming the system
            time.sleep(1)
            
        except Exception as e:
            log.error(f"Error processing batch starting at {skip}: {e}")
            # Continue with next batch
            skip += batch_size
            continue

    log.info(
        f"Done. processed={processed} updated={updated} failed_lang_ops={failed}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill missing TTS audio_description fields")
    p.add_argument("--mongo-uri", required=True, help="MongoDB connection string")
    p.add_argument("--db", required=True, help="Database name (e.g., test)")
    p.add_argument("--collection", default="news", help="Collection name (default: news)")
    p.add_argument(
        "--voice-id",
        required=True,
        help="ElevenLabs voice ID to use for all languages",
    )
    p.add_argument("--voice-id-en", help="Voice ID for English (overrides --voice-id)")
    p.add_argument("--voice-id-hi", help="Voice ID for Hindi (overrides --voice-id)")
    p.add_argument("--voice-id-kn", help="Voice ID for Kannada (overrides --voice-id)")
    p.add_argument(
        "--langs",
        default="en,hi,kn",
        help="Comma-separated languages to process (subset of en,hi,kn)",
    )
    p.add_argument("--force-overwrite", action="store_true", help="Regenerate and overwrite even if audio exists")
    p.add_argument("--max-chars-en", type=int, default=1200)
    p.add_argument("--max-chars-hi", type=int, default=1200)
    p.add_argument("--max-chars-kn", type=int, default=800)
    p.add_argument("--sleep-between-langs", type=float, default=0.0)
    p.add_argument("--sleep-between-docs", type=float, default=0.0)
    p.add_argument("--provider", default="elevenlabs", choices=["elevenlabs", "azure"], help="TTS provider to use (default: elevenlabs)")
    p.add_argument("--limit", type=int, help="Limit number of documents to process (for testing)")
    p.add_argument("--dry-run", action="store_true", help="Do not write updates")
    p.add_argument("--resume-from", type=str, help="Resume from this document ID (ObjectId)")
    p.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    langs = [s.strip() for s in args.langs.split(",") if s.strip() in ("en", "hi", "kn")]
    if not langs:
        raise SystemExit("No valid languages specified (choose from en,hi,kn)")

    backfill(
        mongo_uri=args.mongo_uri,
        database_name=args.db,
        collection_name=args.collection,
        voice_id=args.voice_id,
        langs=langs,
        dry_run=args.dry_run,
        force_overwrite=args.force_overwrite,
        voice_id_en=args.voice_id_en,
        voice_id_hi=args.voice_id_hi,
        voice_id_kn=args.voice_id_kn,
        max_chars_en=args.max_chars_en,
        max_chars_hi=args.max_chars_hi,
        max_chars_kn=args.max_chars_kn,
        sleep_between_langs=args.sleep_between_langs,
        sleep_between_docs=args.sleep_between_docs,
        provider=args.provider,
        limit=args.limit,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()