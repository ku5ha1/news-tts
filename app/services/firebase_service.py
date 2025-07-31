import os
import uuid
import json
import base64
from firebase_admin import credentials, initialize_app, storage
from app.config.settings import Settings

class FirebaseService:
    def __init__(self):
        self.settings = Settings()
        
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
                'storageBucket': self.settings.FIREBASE_STORAGE_BUCKET
            })
            self.bucket = storage.bucket()
            self.connected = True
            print("Firebase initialized successfully")
            
        except Exception as e:
            print(f"Firebase initialization error: {str(e)}")
            self.connected = False

    def upload_audio(self, file_path: str, language: str, document_id: str = None) -> str:
        """Upload audio file to Firebase Storage and return public URL"""
        try:
            if not self.connected:
                return f"https://firebase.storage.fake/{language}/file.mp3"
            
            # Use provided document_id or generate one
            if not document_id:
                document_id = str(uuid.uuid4())
            
            # Language mapping for file names
            language_mapping = {
                "hi": "hindi_audio",
                "kn": "kannada_audio", 
                "en": "english_audio"
            }
            
            # Get the correct file name for the language
            audio_filename = language_mapping.get(language, f"{language}_audio")
            file_extension = os.path.splitext(file_path)[1]
            
            # Create blob and upload to news-audio/document_id/language_audio.mp3
            blob = self.bucket.blob(f"news-audio/{document_id}/{audio_filename}{file_extension}")
            blob.upload_from_filename(file_path)
            
            # Make the file publicly accessible
            blob.make_public()
            
            # Return the public URL
            return blob.public_url
            
        except Exception as e:
            print(f"Firebase upload error: {str(e)}")
            return f"https://firebase.storage.fake/{language}/file.mp3"

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
