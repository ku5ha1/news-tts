import os
import uuid
import logging
from typing import Optional
from azure.storage.blob import BlobServiceClient, BlobClient
from app.config.settings import Settings

logger = logging.getLogger(__name__)

class AzureBlobService:
    def __init__(self):
        self.settings = Settings()
        self.account_name = self.settings.AZURE_STORAGE_ACCOUNT_NAME
        self.connection_string = self.settings.AZURE_STORAGE_CONNECTION_STRING
        self.access_key = self.settings.AZURE_STORAGE_ACCESS_KEY
        self.container_name = self.settings.AZURE_STORAGE_AUDIOFIELD_CONTAINER

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
            
            self.connected = True
            logger.info(f"Azure Blob Storage initialized with account: {self.account_name}, container: {self.container_name}")

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
            
        except Exception as e:
            logger.error(f"[AzureBlob] Upload.failed path={file_path} lang={language} error={e}", exc_info=True)
            raise RuntimeError(f"Azure Blob upload failed: {e}")

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

    def get_container_url(self) -> str:
        """Get the base URL for the container"""
        if not self.connected:
            return ""
        return f"https://{self.account_name}.blob.core.windows.net/{self.container_name}"
