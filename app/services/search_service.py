import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from azure.storage.blob import BlobServiceClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticSearch,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField
)
from dotenv import load_dotenv
import requests
from io import BytesIO


load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchService:
    """Main search service for Magazine2 PDF processing and search functionality"""
    
    def __init__(self):
        """Initialize Azure service clients"""
        self._setup_azure_clients()
        self._setup_search_config()
    
    def _setup_azure_clients(self):
        """Setup Azure service clients"""
        # Azure Storage
        self.storage_conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not self.storage_conn_str:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found")
        
        self.blob_service = BlobServiceClient.from_connection_string(self.storage_conn_str)
        self.magazine2_container = self.blob_service.get_container_client(
            os.getenv("AZURE_STORAGE_MAGAZINE2_CONTAINER")
        )
        
        # Output container is optional - use magazine2 container if not specified
        output_container_name = os.getenv("AZURE_STORAGE_OUTPUT_CONTAINER_NAME")
        if output_container_name:
            self.output_container = self.blob_service.get_container_client(output_container_name)
        else:
            # Use magazine2 container as fallback
            self.output_container = self.magazine2_container
            logger.warning("AZURE_STORAGE_OUTPUT_CONTAINER_NAME not set, using magazine2 container for output")
        
        # Azure Document Intelligence
        docint_endpoint = os.getenv("DOCINT_ENDPOINT")
        docint_key = os.getenv("DOCINT_KEY")
        if not docint_endpoint or not docint_key:
            raise ValueError("DOCINT_ENDPOINT or DOCINT_KEY not found")
        
        self.docint_client = DocumentIntelligenceClient(
            docint_endpoint, 
            AzureKeyCredential(docint_key)
        )
        
        # Azure Cognitive Search
        search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        search_key = os.getenv("AZURE_SEARCH_KEY")
        self.search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        
        if not search_endpoint or not search_key:
            raise ValueError("AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_KEY not found")
        
        self.search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=self.search_index_name,
            credential=AzureKeyCredential(search_key)
        )
        self.search_index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=AzureKeyCredential(search_key)
        )
        
        # Azure OpenAI
        self.aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        self.aoai_key = os.getenv("AZURE_OPENAI_KEY")
        self.embed_model = os.getenv("AZURE_OPENAI_EMBED_MODEL", "text-embedding-3-large")
        
        if not self.aoai_endpoint or not self.aoai_key:
            raise ValueError("AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_KEY not found")
    
    def _setup_search_config(self):
        """Setup search configuration"""
        self.batch_size = 16
        self.max_retries = 5
        self.backoff_factor = 2
    
    def _first_or_str(self, x, default=""):
        """Helper to ensure string fields are primitives, not arrays"""
        if isinstance(x, list):
            return str(x[0]) if x else default
        return str(x) if x is not None else default
    
    def _first_or_int(self, x, default=0):
        """Helper to ensure int fields are primitives, not arrays"""
        if isinstance(x, list):
            x = x[0] if x else default
        try:
            return int(x)
        except Exception:
            return default
    
    def create_search_index(self) -> bool:
        """Create or update the search index"""
        try:
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="title", type=SearchFieldDataType.String),
                SearchableField(name="description", type=SearchFieldDataType.String),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SimpleField(name="magazine_id", type=SearchFieldDataType.String),
                SimpleField(name="published_year", type=SearchFieldDataType.Int32),
                SimpleField(name="published_month", type=SearchFieldDataType.String),
                SimpleField(name="edition_number", type=SearchFieldDataType.String),
                SimpleField(name="pdf_url", type=SearchFieldDataType.String),
                SimpleField(name="thumbnail_url", type=SearchFieldDataType.String),
                SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), 
                          vector_search_dimensions=3072, vector_search_profile_name="vectorSearchProfile")
            ]
            
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="vectorSearchProfile",
                        algorithm_configuration_name="hnswConfig"
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(name="hnswConfig")
                ]
            )
            
            semantic_config = SemanticConfiguration(
                name="semanticConfig",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[
                        SemanticField(field_name="content"),
                        SemanticField(field_name="description")
                    ]
                )
            )
            
            semantic_search = SemanticSearch(configurations=[semantic_config])
            
            index = SearchIndex(
                name=self.search_index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search
            )
            
            self.search_index_client.create_or_update_index(index)
            logger.info(f"Search index '{self.search_index_name}' created/updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create search index: {e}")
            return False
    
    def ocr_pdf_from_blob(self, blob_name: str) -> Optional[Dict[str, Any]]:
        """Extract text from PDF using Azure Document Intelligence"""
        try:
            logger.info(f"Starting OCR for PDF: {blob_name}")
            
            # Download PDF from blob storage
            pdf_bytes = self.magazine2_container.download_blob(blob_name).readall()
            
            # Process with Document Intelligence - use body parameter with bytes
            poller = self.docint_client.begin_analyze_document("prebuilt-read", body=pdf_bytes)
            result = poller.result()
            
            # Structure the result
            structured_result = {"pages": []}
            for page in result.pages:
                page_text = "\n".join([line.content for line in page.lines])
                structured_result["pages"].append({
                    "page_number": page.page_number,
                    "content": page_text,
                    "lines": [{"text": line.content, "polygon": line.polygon} for line in page.lines]
                })
            
            logger.info(f"OCR completed for PDF: {blob_name}")
            return structured_result
            
        except Exception as e:
            logger.error(f"OCR failed for {blob_name}: {e}")
            return None
    
    def save_ocr_result(self, blob_name: str, ocr_result: Dict[str, Any]) -> bool:
        """Save OCR result to output container"""
        try:
            json_path = blob_name.replace(".pdf", ".json")
            self.output_container.upload_blob(
                name=json_path,
                data=json.dumps(ocr_result, ensure_ascii=False, indent=2),
                overwrite=True
            )
            logger.info(f"OCR result saved: {json_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save OCR result: {e}")
            return False
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts using Azure OpenAI"""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            url = f"{self.aoai_endpoint}/openai/deployments/{self.embed_model}/embeddings?api-version=2024-06-01"
            headers = {"api-key": self.aoai_key, "Content-Type": "application/json"}
            payload = {"input": batch}
            
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = requests.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    embeddings.extend([item["embedding"] for item in data["data"]])
                    break  # success
                except requests.exceptions.HTTPError as e:
                    if resp.status_code == 429:
                        sleep_time = self.backoff_factor ** attempt
                        logger.warning(f"429 Too Many Requests. Retrying in {sleep_time}s... (Attempt {attempt})")
                        time.sleep(sleep_time)
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Embedding failed: {e}")
                    if attempt == self.max_retries:
                        raise
                    time.sleep(self.backoff_factor ** attempt)
        
        return embeddings
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into semantic chunks"""
        if not text.strip():
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def process_magazine2_pdf(self, magazine_data: Dict[str, Any]) -> bool:
        """Process a single Magazine2 PDF through the complete pipeline"""
        try:
            magazine_id = str(magazine_data["_id"])  # Convert ObjectId to string
            pdf_url = magazine_data["magazinePdf"]
            
            # Extract blob name from URL (preserve full path including folders)
            # URL format: https://diprstorage.blob.core.windows.net/marchofkarnataka/2025/January/filename.pdf
            # We need: 2025/January/filename.pdf
            url_parts = pdf_url.split("/")
            if len(url_parts) >= 5:
                # Get everything after the container name (index 3)
                blob_name = "/".join(url_parts[4:])
            else:
                # Fallback to just filename if URL structure is unexpected
                blob_name = url_parts[-1]
            
            # Step 1: OCR
            ocr_result = self.ocr_pdf_from_blob(blob_name)
            if not ocr_result:
                return False
            
            # Step 2: Save OCR result
            self.save_ocr_result(blob_name, ocr_result)
            
            # Step 3: Extract and chunk text
            all_text = " ".join([page["content"] for page in ocr_result["pages"]])
            chunks = self.chunk_text(all_text)
            
            if not chunks:
                logger.warning(f"No text chunks found for {magazine_id}")
                return False
            
            # Step 4: Generate embeddings
            embeddings = self.generate_embeddings(chunks)
            
            # Step 5: Create search documents
            search_documents = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Use helper functions to ensure all fields are primitives
                title = self._first_or_str(magazine_data.get("title"))
                description = self._first_or_str(magazine_data.get("description"))
                published_month = self._first_or_str(magazine_data.get("publishedMonth"))
                edition_number = self._first_or_str(magazine_data.get("editionNumber"))
                thumbnail_url = self._first_or_str(magazine_data.get("magazineThumbnail"))
                published_year = self._first_or_int(magazine_data.get("publishedYear"))
                
                doc = {
                    "id": f"{magazine_id}_chunk_{i}",
                    "title": title,
                    "description": description,
                    "content": chunk,
                    "magazine_id": magazine_id,
                    "published_year": published_year,
                    "published_month": published_month,
                    "edition_number": edition_number,
                    "pdf_url": pdf_url,
                    "thumbnail_url": thumbnail_url,
                    "contentVector": embedding
                }
                search_documents.append(doc)
            
            # Step 6: Upload to search index
            # Debug: Log the complete document structure to identify array fields
            if search_documents:
                import json
                sample_doc = search_documents[0]
                logger.info("=== COMPLETE DOCUMENT STRUCTURE ===")
                for key, value in sample_doc.items():
                    logger.info(f"Field '{key}': type={type(value)}, value={value if not isinstance(value, list) else f'[array with {len(value)} items]'}")
                logger.info("=== END DOCUMENT STRUCTURE ===")
            
            # Ensure proper serialization for vector fields
            # Convert embeddings to proper format for Azure Search
            for doc in search_documents:
                if 'contentVector' in doc and isinstance(doc['contentVector'], list):
                    # Ensure all values are floats and properly formatted
                    doc['contentVector'] = [float(x) for x in doc['contentVector']]
            
            # Additional validation: ensure all fields are properly serialized
            try:
                # Test serialization before uploading
                json.dumps(search_documents[0])
                logger.info("Document serialization test passed")
            except Exception as e:
                logger.error(f"Document serialization test failed: {e}")
                raise ValueError(f"Document serialization failed: {e}")
            
            self.search_client.upload_documents(documents=search_documents)
            
            logger.info(f"Successfully processed {magazine_id}: {len(chunks)} chunks indexed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process magazine {magazine_id}: {e}")
            return False
    
    def search_documents(self, query: str, top: int = 10) -> List[Dict[str, Any]]:
        """Search documents using semantic and vector search"""
        try:
            # Generate query embedding
            query_embeddings = self.generate_embeddings([query])
            query_vector = query_embeddings[0]
            
            # Perform search
            results = self.search_client.search(
                search_text=query,
                vector_queries=[VectorizedQuery(vector=query_vector, fields="contentVector")],
                semantic_configuration_name="semanticConfig",
                highlight_fields=["content"],
                top=top,
                include_total_count=True
            )
            
            search_results = []
            for result in results:
                search_results.append({
                    "id": result["id"],
                    "title": result.get("title", ""),
                    "description": result.get("description", ""),
                    "content": result.get("content", ""),
                    "magazine_id": result.get("magazine_id", ""),
                    "published_year": result.get("published_year", 0),
                    "published_month": result.get("published_month", ""),
                    "edition_number": result.get("edition_number", ""),
                    "pdf_url": result.get("pdf_url", ""),
                    "thumbnail_url": result.get("thumbnail_url", ""),
                    "score": result.get("@search.score", 0)
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_processed_files(self) -> List[str]:
        """Get list of already processed files"""
        try:
            processed_files = []
            for blob in self.output_container.list_blobs():
                if blob.name.endswith(".json"):
                    # Extract magazine_id from JSON filename
                    magazine_id = blob.name.replace(".json", "").split("_")[0]
                    processed_files.append(magazine_id)
            return processed_files
        except Exception as e:
            logger.error(f"Failed to get processed files: {e}")
            return []
    
    def is_file_processed(self, magazine_id: str) -> bool:
        """Check if a magazine PDF has already been processed"""
        processed_files = self.get_processed_files()
        return magazine_id in processed_files
