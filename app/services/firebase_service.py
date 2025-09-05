import os
import uuid
import json
import base64
from firebase_admin import credentials, initialize_app, storage
from app.config.settings import Settings

def normalize_bucket_name(name: str) -> str:
    if not name:
        return name
    if name.startswith("gs://"):
        name = name.replace("gs://", "", 1)
    # Modern Firebase Storage buckets use ".firebasestorage.app" format
    # Keep the bucket name as-is since Firebase Admin SDK supports both formats
    return name.strip()

class FirebaseService:
    def __init__(self):
        self.settings = Settings()
        self.bucket_name = normalize_bucket_name(self.settings.FIREBASE_STORAGE_BUCKET)

        try:
            if not self.settings.FIREBASE_SERVICE_ACCOUNT_BASE64:
                raise ValueError("FIREBASE_SERVICE_ACCOUNT_BASE64 is missing in environment variables")

            service_account_info = json.loads(
                base64.b64decode(self.settings.FIREBASE_SERVICE_ACCOUNT_BASE64).decode("utf-8")
            )

            cred = credentials.Certificate(service_account_info)
            self.app = initialize_app(cred, {"storageBucket": self.bucket_name})
            self.bucket = storage.bucket(self.bucket_name)
            self.connected = True
            print(f"✅ Firebase initialized with bucket: {self.bucket_name}")

        except Exception as e:
            print(f"Firebase initialization error: {str(e)}")
            self.connected = False

    def upload_audio(self, file_path: str, language: str, document_id: str = None) -> str:
        if not self.connected:
            raise RuntimeError("Firebase not connected")

        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"No audio file generated for {language}")

        if not document_id:
            document_id = str(uuid.uuid4()).replace("-", "")

        language_prefix = {"hi": "hindi", "kn": "kannada", "en": "english"}.get(language, language)
        file_extension = os.path.splitext(file_path)[1] or ".mp3"

        unique_suffix = str(uuid.uuid4())
        blob_name = f"news-audio/{document_id}/{language_prefix}-{unique_suffix}{file_extension}"

        blob = self.bucket.blob(blob_name)
        blob.upload_from_filename(file_path)

        try:
            blob.make_public()
        except Exception:
            pass

        return f"gs://{self.bucket.name}/{blob_name}"

    def delete_audio(self, file_url: str) -> bool:
        try:
            if not self.connected:
                return False

            # Remove gs:// prefix if passed
            file_url = file_url.replace("gs://", "")
            parts = file_url.split("/", 1)
            if len(parts) != 2:
                raise ValueError("Invalid gs:// URL")

            blob_path = parts[1]
            blob = self.bucket.blob(blob_path)
            blob.delete()
            return True

        except Exception as e:
            print(f"Firebase delete error: {str(e)}")
            return False

    def is_connected(self) -> bool:
        return self.connected
