import os
import uuid
import logging
from typing import Optional
from azure.storage.blob import BlobServiceClient, BlobClient, ContentSettings
from fastapi import UploadFile
from app.config.settings import settings

logger = logging.getLogger(__name__)

class AzureBlobService:
    def __init__(self):
        self.account_name = settings.AZURE_STORAGE_ACCOUNT_NAME
        self.connection_string = settings.AZURE_STORAGE_CONNECTION_STRING
        self.access_key = settings.AZURE_STORAGE_ACCESS_KEY
        self.container_name = settings.AZURE_STORAGE_AUDIOFIELD_CONTAINER
        self.magazine_container_name = settings.AZURE_STORAGE_MAGAZINE_CONTAINER
        self.magazine2_container_name = settings.AZURE_STORAGE_MAGAZINE2_CONTAINER

        try:
            if not self.account_name or not self.container_name:
                logger.warning("Azure Storage account name or container name is missing - Azure Blob will be disabled")
                self.connected = False
                return

            # Initialize BlobServiceClient
            if self.connection_string:
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            elif self.account_name and self.access_key:
                # Alternative: use account name and access key
                account_url = f"https://{self.account_name}.blob.core.windows.net"
                self.blob_service_client = BlobServiceClient(account_url=account_url, credential=self.access_key)
            else:
                logger.warning("Azure Storage credentials not provided - Azure Blob will be disabled")
                self.connected = False
                return

            # Test connection by getting container properties
            self.container_client = self.blob_service_client.get_container_client(self.container_name)
            self.container_client.get_container_properties()
            
            # Initialize magazine container client
            self.magazine_container_client = self.blob_service_client.get_container_client(self.magazine_container_name)
            self.magazine_container_client.get_container_properties()
            
            # Initialize magazine2 container client
            self.magazine2_container_client = self.blob_service_client.get_container_client(self.magazine2_container_name)
            self.magazine2_container_client.get_container_properties()
            
            self.connected = True
            logger.info(f"Azure Blob Storage initialized with account: {self.account_name}, containers: {self.container_name}, {self.magazine_container_name}, {self.magazine2_container_name}")

        except ImportError as e:
            logger.error(f"Azure Storage SDK import error: {str(e)}", exc_info=True)
            self.connected = False
        except ConnectionError as e:
            logger.error(f"Azure Storage connection error: {str(e)}", exc_info=True)
            self.connected = False
        except Exception as e:
            logger.error(f"Azure Blob Storage initialization error: {str(e)}", exc_info=True)
            self.connected = False

    def upload_audio(self, file_path: str, language: str, document_id: str = None) -> str:
        logger.info(f"[AzureBlob] Upload.start path={file_path} lang={language} doc={document_id}")
        
        if not self.connected:
            raise RuntimeError("Azure Blob Storage not connected")

        if not file_path:
            raise ValueError(f"No file path provided for {language}")
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        if not document_id:
            document_id = str(uuid.uuid4()).replace("-", "")

        language_prefix = {"hi": "hindi", "kn": "kannada", "en": "english"}.get(language, language)
        file_extension = os.path.splitext(file_path)[1] or ".mp3"

        unique_suffix = str(uuid.uuid4())
        blob_name = f"{document_id}/{language_prefix}-{unique_suffix}{file_extension}"

        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            
            logger.info(f"[AzureBlob] Uploading blob: {blob_name}")
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            logger.info(f"[AzureBlob] Upload.done: {blob_name}")

            # Generate public URL
            public_url = f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}"
            logger.info(f"[AzureBlob] Upload.complete url={public_url}")
            return public_url
            
        except FileNotFoundError as e:
            logger.error(f"[AzureBlob] File not found: {str(e)}", exc_info=True)
            raise RuntimeError(f"Audio file not found: {str(e)}")
        except ConnectionError as e:
            logger.error(f"[AzureBlob] Connection error during upload: {str(e)}", exc_info=True)
            raise RuntimeError(f"Azure Storage connection failed: {str(e)}")
        except Exception as e:
            logger.error(f"[AzureBlob] Upload.failed path={file_path} lang={language} error={e}", exc_info=True)
            raise RuntimeError(f"Azure Blob upload failed: {e}")

    def upload_audio_to_existing_url(self, file_path: str, existing_url: str) -> str:
        """Overwrite an existing blob path with a new audio file.

        existing_url must target this account and the audio container.
        Returns the same URL when successful.
        """
        if not self.connected:
            raise RuntimeError("Azure Blob Storage not connected")

        if not file_path:
            raise ValueError("No file path provided")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        expected_prefix = f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/"
        if not existing_url.startswith(expected_prefix):
            raise ValueError("URL does not point to the configured audio container")

        blob_path = existing_url.replace(expected_prefix, "")

        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            return existing_url
        except Exception as e:
            logger.error(f"[AzureBlob] Overwrite failed for {blob_path}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Azure Blob overwrite failed: {e}")

    def delete_audio(self, file_url: str) -> bool:
        try:
            if not self.connected:
                logger.warning("[AzureBlob] Not connected - cannot delete")
                return False

            # Extract blob name from URL
            # Expected format: https://{account}.blob.core.windows.net/{container}/{blob_path}
            if not file_url.startswith(f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/"):
                logger.error(f"[AzureBlob] Invalid URL format: {file_url}")
                return False

            blob_path = file_url.replace(f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/", "")
            blob_client = self.container_client.get_blob_client(blob_path)
            blob_client.delete_blob()
            logger.info(f"[AzureBlob] Deleted: {blob_path}")
            return True

        except Exception as e:
            logger.error(f"[AzureBlob] Delete error: {str(e)}")
            return False

    def is_connected(self) -> bool:
        return self.connected

    def upload_magazine_file(self, file: UploadFile, published_year: str, published_month: str, magazine_id: str, file_type: str) -> str:
        logger.info(f"[AzureBlob] Magazine upload start: {file_type} for magazine {magazine_id}")
        
        if not self.connected:
            raise RuntimeError("Azure Blob Storage not connected")

        if not file:
            raise ValueError(f"No file provided for {file_type}")
            
        if file_type not in ['thumbnail', 'pdf']:
            raise ValueError("file_type must be 'thumbnail' or 'pdf'")

        # Generate blob name based on file type
        if file_type == 'thumbnail':
            # Get file extension
            file_extension = os.path.splitext(file.filename)[1] if file.filename else '.jpg'
            blob_name = f"{published_year}/{published_month}/{magazine_id}_thumbnail{file_extension}"
        else:  # pdf
            blob_name = f"{published_year}/{published_month}/{magazine_id}_{published_month}_{published_year}.pdf"

        try:
            blob_client = self.magazine_container_client.get_blob_client(blob_name)
            
            logger.info(f"[AzureBlob] Uploading magazine {file_type}: {blob_name}")
            
            # Read file content and upload
            file_content = file.file.read()
            blob_client.upload_blob(file_content, overwrite=True)
            
            # Set HTTP headers for PDF files to display inline
            if file_type == 'pdf':
                content_settings = ContentSettings(
                    content_type="application/pdf",
                    content_disposition=f"inline; filename=\"{blob_name}\""
                )
                blob_client.set_http_headers(content_settings=content_settings)
                logger.info(f"[AzureBlob] Set PDF headers for inline display: {blob_name}")
            
            logger.info(f"[AzureBlob] Magazine {file_type} upload done: {blob_name}")

            # Generate public URL
            public_url = f"https://{self.account_name}.blob.core.windows.net/{self.magazine_container_name}/{blob_name}"
            logger.info(f"[AzureBlob] Magazine {file_type} upload complete: {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"[AzureBlob] Magazine {file_type} upload failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Magazine {file_type} upload failed: {e}")

    def delete_magazine_file(self, file_url: str) -> bool:
        """
        Delete magazine file from Azure Blob Storage
        
        Args:
            file_url: Public URL of the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.connected:
                logger.warning("[AzureBlob] Not connected - cannot delete magazine file")
                return False

            # Extract blob name from URL
            if not file_url.startswith(f"https://{self.account_name}.blob.core.windows.net/{self.magazine_container_name}/"):
                logger.error(f"[AzureBlob] Invalid magazine file URL format: {file_url}")
                return False

            blob_path = file_url.replace(f"https://{self.account_name}.blob.core.windows.net/{self.magazine_container_name}/", "")
            blob_client = self.magazine_container_client.get_blob_client(blob_path)
            blob_client.delete_blob()
            logger.info(f"[AzureBlob] Deleted magazine file: {blob_path}")
            return True

        except Exception as e:
            logger.error(f"[AzureBlob] Magazine file delete error: {str(e)}")
            return False

    def get_container_url(self) -> str:
        """Get the base URL for the container"""
        if not self.connected:
            return ""
        return f"https://{self.account_name}.blob.core.windows.net/{self.container_name}"
    
    def get_magazine_container_url(self) -> str:
        """Get the base URL for the magazine container"""
        if not self.connected:
            return ""
        return f"https://{self.account_name}.blob.core.windows.net/{self.magazine_container_name}"

    def upload_magazine2_file(self, file: UploadFile, published_year: str, published_month: str, magazine2_id: str, file_type: str) -> str:
        logger.info(f"[AzureBlob] Magazine2 upload start: {file_type} for magazine2 {magazine2_id}")
        
        if not self.connected:
            raise RuntimeError("Azure Blob Storage not connected")

        if not file:
            raise ValueError(f"No file provided for {file_type}")
            
        if file_type not in ['thumbnail', 'pdf']:
            raise ValueError("file_type must be 'thumbnail' or 'pdf'")

        # Generate blob name based on file type
        if file_type == 'thumbnail':
            # Get file extension
            file_extension = os.path.splitext(file.filename)[1] if file.filename else '.jpg'
            blob_name = f"{published_year}/{published_month}/{magazine2_id}_thumbnail{file_extension}"
        else:  # pdf
            blob_name = f"{published_year}/{published_month}/{magazine2_id}_{published_month}_{published_year}.pdf"

        try:
            blob_client = self.magazine2_container_client.get_blob_client(blob_name)
            
            logger.info(f"[AzureBlob] Uploading magazine2 {file_type}: {blob_name}")
            
            # Read file content and upload
            file_content = file.file.read()
            blob_client.upload_blob(file_content, overwrite=True)
            
            # Set HTTP headers for PDF files to display inline
            if file_type == 'pdf':
                content_settings = ContentSettings(
                    content_type="application/pdf",
                    content_disposition=f"inline; filename=\"{blob_name}\""
                )
                blob_client.set_http_headers(content_settings=content_settings)
                logger.info(f"[AzureBlob] Set PDF headers for inline display: {blob_name}")
            
            logger.info(f"[AzureBlob] Magazine2 {file_type} upload done: {blob_name}")

            # Generate public URL
            public_url = f"https://{self.account_name}.blob.core.windows.net/{self.magazine2_container_name}/{blob_name}"
            logger.info(f"[AzureBlob] Magazine2 {file_type} upload complete: {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"[AzureBlob] Magazine2 {file_type} upload failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Magazine2 {file_type} upload failed: {e}")

    def update_blob_content_disposition(self, container_name: str, blob_name: str, content_disposition: str = 'inline') -> bool:
        """Update the content disposition of an existing blob."""
        try:
            if not self.connected:
                logger.warning("Azure Blob Storage not connected")
                return False
                
            container_client = self.blob_service_client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
            
            # Get existing blob properties
            blob_properties = blob_client.get_blob_properties()
            
            # Set HTTP headers with new disposition
            content_settings = ContentSettings(
                content_type=blob_properties.content_settings.content_type or "application/pdf",
                content_disposition=f"{content_disposition}; filename=\"{blob_name}\""
            )
            
            # Set blob HTTP headers
            blob_client.set_http_headers(content_settings=content_settings)
            
            logger.info(f"[AzureBlob] Updated content disposition for {blob_name} to {content_disposition}")
            return True
            
        except Exception as e:
            logger.error(f"[AzureBlob] Failed to update content disposition for {blob_name}: {str(e)}")
            return False

    def delete_magazine2_file(self, file_url: str) -> bool:
        try:
            if not self.connected:
                logger.warning("[AzureBlob] Not connected - cannot delete magazine2 file")
                return False

            # Extract blob name from URL
            if not file_url.startswith(f"https://{self.account_name}.blob.core.windows.net/{self.magazine2_container_name}/"):
                logger.error(f"[AzureBlob] Invalid magazine2 file URL format: {file_url}")
                return False

            blob_path = file_url.replace(f"https://{self.account_name}.blob.core.windows.net/{self.magazine2_container_name}/", "")
            blob_client = self.magazine2_container_client.get_blob_client(blob_path)
            blob_client.delete_blob()
            logger.info(f"[AzureBlob] Deleted magazine2 file: {blob_path}")
            return True

        except Exception as e:
            logger.error(f"[AzureBlob] Magazine2 file delete error: {str(e)}")
            return False

    def get_magazine2_container_url(self) -> str:
        """Get the base URL for the magazine2 container"""
        if not self.connected:
            return ""
        return f"https://{self.account_name}.blob.core.windows.net/{self.magazine2_container_name}"