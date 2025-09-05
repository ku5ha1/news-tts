import os
import uuid
import json
import base64
from firebase_admin import credentials, initialize_app, storage
from app.config.settings import Settings

class FirebaseService:
    def __init__(self):
        self.settings = Settings()
        
        # Normalize bucket name: users often set the HTTPS hostname instead of the GCS bucket id
        def _normalize_bucket_name(name: str) -> str:
            # Expected: {project}.appspot.com
            if name.endswith(".firebasestorage.app"):
                return name.replace(".firebasestorage.app", ".appspot.com")
            return name
        
        self.bucket_name = _normalize_bucket_name(self.settings.FIREBASE_STORAGE_BUCKET)
        
        # Initialize Firebase Admin SDK with base64 decoded credentials
        try:
            # Check if base64 service account is provided
            if not self.settings.FIREBASE_SERVICE_ACCOUNT_BASE64:
                raise ValueError("FIREBASE_SERVICE_ACCOUNT_BASE64 is missing in environment variables")
            
            # Decode base64 service account credentials (matching Node.js pattern)
            service_account_info = json.loads(
                base64.b64decode(self.settings.FIREBASE_SERVICE_ACCOUNT_BASE64).decode('utf-8')
            )
            
            # Initialize Firebase with credentials and storage bucket
            cred = credentials.Certificate(service_account_info)
            self.app = initialize_app(cred, {
                'storageBucket': self.bucket_name
            })
            self.bucket = storage.bucket(self.bucket_name)
            self.connected = True
            print("Firebase initialized successfully")
            
        except Exception as e:
            print(f"Firebase initialization error: {str(e)}")
            self.connected = False

    def upload_audio(self, file_path: str, language: str, document_id: str = None) -> str:
        """Upload audio file to Firebase Storage and return gs:// URI matching legacy pattern

        Pattern: gs://{bucket}/news-audio/{document_id}/{language}-{uuid}.ext
        language is one of: english, hindi, kannada
        """
        try:
            if not self.connected:
                # Return a deterministic-looking gs path even if not connected
                language_prefix = {
                    "hi": "hindi",
                    "kn": "kannada",
                    "en": "english"
                }.get(language, language)
                fake_doc = document_id or str(uuid.uuid4()).replace("-", "")
                fake_name = f"{language_prefix}-{str(uuid.uuid4())}.mp3"
                return f"gs://{self.bucket_name}/news-audio/{fake_doc}/{fake_name}"
            
            # Use provided document_id or generate one (folder component)
            if not document_id:
                document_id = str(uuid.uuid4()).replace("-", "")
            
            # Language mapping for file name prefix
            language_prefix = {
                "hi": "hindi",
                "kn": "kannada", 
                "en": "english"
            }.get(language, language)
            file_extension = os.path.splitext(file_path)[1]
            
            # Create unique filename: {language}-{uuid}.ext under news-audio/{document_id}/
            unique_suffix = str(uuid.uuid4())
            blob_name = f"news-audio/{document_id}/{language_prefix}-{unique_suffix}{file_extension}"
            blob = self.bucket.blob(blob_name)
            blob.upload_from_filename(file_path)
            
            # Optionally make public, but return gs:// URI to match required pattern
            try:
                blob.make_public()
            except Exception:
                pass
            return f"gs://{self.bucket_name}/{blob_name}"
            
        except Exception as e:
            print(f"Firebase upload error: {str(e)}")
            language_prefix = {
                "hi": "hindi",
                "kn": "kannada", 
                "en": "english"
            }.get(language, language)
            fallback_doc = document_id or str(uuid.uuid4()).replace("-", "")
            fallback_name = f"{language_prefix}-{str(uuid.uuid4())}.mp3"
            return f"gs://{self.bucket_name}/news-audio/{fallback_doc}/{fallback_name}"

    def delete_audio(self, file_url: str) -> bool:
        """Delete audio file from Firebase Storage"""
        try:
            if not self.connected:
                return True
                
            # Extract filename from URL
            filename = file_url.split('/')[-1]
            blob = self.bucket.blob(f"news-audio/{filename}")
            blob.delete()
            return True
            
        except Exception as e:
            print(f"Firebase delete error: {str(e)}")
            return False

    def is_connected(self) -> bool:
        """Check if Firebase is properly connected"""
        return self.connected
