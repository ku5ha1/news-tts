import os
import uuid
import json
import base64
import logging
from firebase_admin import credentials, initialize_app, storage
from app.config.settings import Settings

logger = logging.getLogger(__name__)

def normalize_bucket_name(name: str) -> str:
    if not name:
        return name
    if name.startswith("gs://"):
        name = name.replace("gs://", "", 1)
    
    return name.strip()

class FirebaseService:
    def __init__(self):
        self.settings = Settings()
        self.bucket_name = normalize_bucket_name(self.settings.FIREBASE_STORAGE_BUCKET or "")

        try:
            if not self.settings.FIREBASE_SERVICE_ACCOUNT_BASE64:
                logger.warning("FIREBASE_SERVICE_ACCOUNT_BASE64 is missing - Firebase will be disabled")
                self.connected = False
                return

            service_account_info = json.loads(
                base64.b64decode(self.settings.FIREBASE_SERVICE_ACCOUNT_BASE64).decode("utf-8")
            )

            cred = credentials.Certificate(service_account_info)
            self.app = initialize_app(cred, {"storageBucket": self.bucket_name})
            self.bucket = storage.bucket(self.bucket_name)
            self.connected = True
            logger.info(f"Firebase initialized with bucket: {self.bucket_name}")

        except Exception as e:
            logger.error(f"Firebase initialization error: {str(e)}", exc_info=True)
            self.connected = False

    def upload_audio(self, file_path: str, language: str, document_id: str = None) -> str:
        logger.info(f"[Firebase] Upload.start path={file_path} lang={language} doc={document_id}")
        
        if not self.connected:
            raise RuntimeError("Firebase not connected")

        if not file_path:
            raise ValueError(f"No file path provided for {language}")
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        if not document_id:
            document_id = str(uuid.uuid4()).replace("-", "")

        language_prefix = {"hi": "hindi", "kn": "kannada", "en": "english"}.get(language, language)
        file_extension = os.path.splitext(file_path)[1] or ".wav"

        unique_suffix = str(uuid.uuid4())
        blob_name = f"news-audio/{document_id}/{language_prefix}-{unique_suffix}{file_extension}"

        try:
            blob = self.bucket.blob(blob_name)
            logger.info(f"[Firebase] Uploading blob: {blob_name}")
            blob.upload_from_filename(file_path)
            logger.info(f"[Firebase] Upload.done: {blob_name}")

            try:
                blob.make_public()
                logger.info(f"[Firebase] File made public: {blob_name}")
            except Exception as e:
                logger.warning(f"[Firebase] Could not make file public: {e}")

            public_url = f"gs://{self.bucket.name}/{blob_name}"
            logger.info(f"[Firebase] Upload.complete url={public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"[Firebase] Upload.failed path={file_path} lang={language} error={e}", exc_info=True)
            raise RuntimeError(f"Firebase upload failed: {e}")

    def delete_audio(self, file_url: str) -> bool:
        try:
            if not self.connected:
                logger.warning("[Firebase] Not connected - cannot delete")
                return False

            # Remove gs:// prefix if passed
            file_url = file_url.replace("gs://", "")
            parts = file_url.split("/", 1)
            if len(parts) != 2:
                raise ValueError("Invalid gs:// URL")

            blob_path = parts[1]
            blob = self.bucket.blob(blob_path)
            blob.delete()
            logger.info(f"[Firebase] Deleted: {blob_path}")
            return True

        except Exception as e:
            logger.error(f"[Firebase] Delete error: {str(e)}")
            return False

    def is_connected(self) -> bool:
        return self.connected
