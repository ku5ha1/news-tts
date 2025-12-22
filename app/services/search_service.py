import logging
from typing import List, Dict, Any, Optional
from azure.storage.blob import BlobServiceClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import gc
import time
import json

from app.config.settings import settings
from app.services.qdrant_service import QdrantService
from app.services.local_embedding_service import LocalEmbeddingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchService:
    """Refactored search service using Qdrant Cloud and local embeddings"""
    
    def __init__(self):
        """Initialize services: Azure Document Intelligence + Qdrant + Local Embeddings"""
        self._setup_azure_clients()
        self._setup_local_services()
        self._setup_search_config()
    
    def _setup_azure_clients(self):
        """Setup Azure service clients (Storage + Document Intelligence only)"""
        # Azure Storage
        self.storage_conn_str = settings.AZURE_STORAGE_CONNECTION_STRING
        if not self.storage_conn_str:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found")
        
        self.blob_service = BlobServiceClient.from_connection_string(self.storage_conn_str)
        self.magazine2_container = self.blob_service.get_container_client(
            settings.AZURE_STORAGE_MAGAZINE2_CONTAINER
        )
        
        # Output container for OCR results
        output_container_name = settings.AZURE_STORAGE_OUTPUT_CONTAINER_NAME
        if output_container_name:
            self.output_container = self.blob_service.get_container_client(output_container_name)
        else:
            self.output_container = self.magazine2_container
            logger.warning("AZURE_STORAGE_OUTPUT_CONTAINER_NAME not set, using magazine2 container for output")
        
        # Azure Document Intelligence (ONLY Azure service we keep)
        docint_endpoint = settings.DOCINT_ENDPOINT
        docint_key = settings.DOCINT_KEY
        if not docint_endpoint or not docint_key:
            raise ValueError("DOCINT_ENDPOINT or DOCINT_KEY not found")
        
        self.docint_client = DocumentIntelligenceClient(
            docint_endpoint, 
            AzureKeyCredential(docint_key)
        )
        logger.info("[SearchService] Azure Document Intelligence initialized")
    
    def _setup_local_services(self):
        """Setup local Qdrant and embedding services"""
        try:
            # Initialize Qdrant Cloud service
            self.qdrant_service = QdrantService()
            logger.info("[SearchService] Qdrant Cloud service initialized")
            
            # Initialize local embedding service
            self.embedding_service = LocalEmbeddingService()
            logger.info("[SearchService] Local embedding service initialized")
            
        except Exception as e:
            logger.error(f"[SearchService] Failed to initialize local services: {e}")
            raise
    
    def _setup_search_config(self):
        """Setup search configuration"""
        self.embedding_batch_size = 4  # Keep CPU usage stable
        self.upsert_batch_size = 100  # Batch size for Qdrant upserts
    
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
        """Create Qdrant collection (replaces Azure Search index creation)"""
        try:
            return self.qdrant_service.create_collection(force_recreate=False)
        except Exception as e:
            logger.error(f"[SearchService] Failed to create Qdrant collection: {e}")
            return False
    
    def ocr_pdf_from_blob(self, blob_name: str) -> Optional[Dict[str, Any]]:
        """Extract text from PDF using Azure Document Intelligence (UNCHANGED)"""
        try:
            logger.info(f"[SearchService] Starting OCR for PDF: {blob_name}")
            
            # Download PDF from blob storage
            pdf_bytes = self.magazine2_container.download_blob(blob_name).readall()
            
            # Process with Document Intelligence
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
            
            logger.info(f"[SearchService] OCR completed for PDF: {blob_name}")
            return structured_result
            
        except Exception as e:
            logger.error(f"[SearchService] OCR failed for {blob_name}: {e}")
            return None
    
    def save_ocr_result(self, blob_name: str, ocr_result: Dict[str, Any]) -> bool:
        """Save OCR result to output container (UNCHANGED)"""
        try:
            json_path = blob_name.replace(".pdf", ".json")
            self.output_container.upload_blob(
                name=json_path,
                data=json.dumps(ocr_result, ensure_ascii=False, indent=2),
                overwrite=True
            )
            logger.info(f"[SearchService] OCR result saved: {json_path}")
            return True
        except Exception as e:
            logger.error(f"[SearchService] Failed to save OCR result: {e}")
            return False
    
    def chunk_text_semantically(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Enhanced chunking that preserves semantic context (UNCHANGED)"""
        if not text.strip():
            return []
        
        # Split by sentences first to preserve semantic boundaries
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence exceeds chunk size, finalize current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text.strip())
                
                # Start new chunk with overlap from previous chunk
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple rules"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def process_magazine2_pdf(self, magazine_data: Dict[str, Any]) -> bool:
        """
        REFACTORED: Process PDF using local embeddings and Qdrant Cloud
        
        Flow:
        1. OCR with Azure Document Intelligence (kept)
        2. Generate embeddings with LocalEmbeddingService (NEW)
        3. Upsert to Qdrant Cloud (NEW)
        """
        try:
            magazine_id = str(magazine_data["_id"])
            pdf_url = magazine_data["magazinePdf"]
            
            logger.info(f"[SearchService] Processing magazine: {magazine_id}")
            
            # Extract blob name from URL
            url_parts = pdf_url.split("/")
            if len(url_parts) >= 5:
                blob_name = "/".join(url_parts[4:])
            else:
                blob_name = url_parts[-1]
            
            # Step 1: Get OCR result (from existing JSON or fresh OCR)
            ocr_result = self._get_or_create_ocr_result(blob_name)
            if not ocr_result:
                return False
            
            # Step 2: Create documents with page-aware chunking
            documents, texts = self._create_documents_with_pages(ocr_result, magazine_data)
            
            if not documents or not texts:
                logger.warning(f"[SearchService] No documents created for {magazine_id}")
                return False
            
            # Step 3: Generate embeddings using LOCAL service (CRITICAL CHANGE)
            try:
                logger.info(f"[SearchService] Generating embeddings for {len(texts)} chunks")
                dense_embeddings, sparse_embeddings = self.embedding_service.generate_hybrid_embeddings(
                    texts, 
                    batch_size=self.embedding_batch_size
                )
                logger.info(f"[SearchService] Embeddings generated successfully")
                
            except Exception as e:
                logger.error(f"[SearchService] Embedding generation failed: {e}")
                return False
            
            # Step 4: Upsert to Qdrant Cloud (CRITICAL CHANGE)
            try:
                success = self.qdrant_service.upsert_documents(
                    documents=documents,
                    dense_embeddings=dense_embeddings,
                    sparse_embeddings=sparse_embeddings
                )
                
                if not success:
                    logger.error(f"[SearchService] Failed to upsert documents to Qdrant")
                    return False
                
                logger.info(f"[SearchService] Successfully processed {magazine_id}: {len(documents)} chunks indexed")
                return True
                
            except Exception as e:
                logger.error(f"[SearchService] Qdrant upsert failed: {e}")
                return False
            
        except Exception as e:
            logger.error(f"[SearchService] Failed to process magazine {magazine_id}: {e}")
            return False
        
        finally:
            # CRITICAL: Force garbage collection to keep RAM < 3GB
            gc.collect()
            logger.info("[SearchService] Garbage collection completed")
    
    def _get_or_create_ocr_result(self, blob_name: str) -> Optional[Dict[str, Any]]:
        """Get existing OCR result or create new one (UNCHANGED)"""
        try:
            json_path = blob_name.replace(".pdf", ".json")
            
            try:
                json_blob = self.output_container.download_blob(json_path)
                json_data = json_blob.readall().decode('utf-8')
                ocr_result = json.loads(json_data)
                logger.info(f"[SearchService] Using existing OCR result: {json_path}")
                return ocr_result
            except Exception:
                logger.info(f"[SearchService] No existing OCR result found, performing fresh OCR: {blob_name}")
                pass
            
            # Perform fresh OCR
            ocr_result = self.ocr_pdf_from_blob(blob_name)
            if ocr_result:
                self.save_ocr_result(blob_name, ocr_result)
            
            return ocr_result
            
        except Exception as e:
            logger.error(f"[SearchService] Failed to get/create OCR result for {blob_name}: {e}")
            return None
    
    def _create_documents_with_pages(
        self, 
        ocr_result: Dict[str, Any], 
        magazine_data: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Create document metadata and extract texts for embedding
        
        Returns:
            Tuple of (documents, texts) where documents contain metadata and texts are for embedding
        """
        try:
            magazine_id = str(magazine_data["_id"])
            
            # Extract metadata
            title = self._first_or_str(magazine_data.get("title"))
            description = self._first_or_str(magazine_data.get("description"))
            published_month = self._first_or_str(magazine_data.get("publishedMonth"))
            edition_number = self._first_or_str(magazine_data.get("editionNumber"))
            thumbnail_url = self._first_or_str(magazine_data.get("magazineThumbnail"))
            published_year = self._first_or_int(magazine_data.get("publishedYear"))
            pdf_url = magazine_data["magazinePdf"]
            
            documents = []
            texts = []
            
            # Process each page separately to maintain page numbers
            for page in ocr_result["pages"]:
                page_number = page["page_number"]
                page_content = page["content"]
                
                if not page_content.strip():
                    continue
                
                # Apply semantic chunking to this page
                page_chunks = self.chunk_text_semantically(page_content)
                
                for chunk_index, chunk in enumerate(page_chunks):
                    # Create document ID
                    doc_id = f"{magazine_id}_page_{page_number}_chunk_{chunk_index}"
                    
                    # Create document metadata (for Qdrant payload)
                    doc = {
                        "id": doc_id,
                        "title": title,
                        "description": description,
                        "content": chunk,
                        "magazine_id": magazine_id,
                        "page_number": page_number,
                        "chunk_position": chunk_index,
                        "published_year": published_year,
                        "published_month": published_month,
                        "edition_number": edition_number,
                        "pdf_url": pdf_url,
                        "thumbnail_url": thumbnail_url
                    }
                    
                    documents.append(doc)
                    texts.append(chunk)
            
            logger.info(f"[SearchService] Created {len(documents)} documents with page information")
            return documents, texts
            
        except Exception as e:
            logger.error(f"[SearchService] Failed to create documents: {e}")
            return [], []
    
    def search_documents(
        self, 
        query: str, 
        top: int = 10, 
        filters: Optional[Dict[str, Any]] = None, 
        vector_weight: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        REFACTORED: Search using Qdrant Cloud with RRF fusion and rescoring
        
        Flow:
        1. Generate query embeddings with LocalEmbeddingService (NEW)
        2. Perform hybrid search with Qdrant (NEW)
        3. Map results to existing SearchResult format
        """
        try:
            logger.info(f"[SearchService] Search query: '{query}'")
            t0 = time.time()
            
            # Step 1: Generate query embeddings using LOCAL service (CRITICAL CHANGE)
            try:
                dense_vector, sparse_vector = self.embedding_service.generate_query_embeddings(query)
                logger.info(f"[SearchService] Query embeddings generated")
            except Exception as e:
                logger.error(f"[SearchService] Query embedding failed: {e}")
                return []
            
            # Step 2: Perform hybrid search with Qdrant (CRITICAL CHANGE)
            # Uses RRF fusion + rescoring with oversampling=2.0 (configured in QdrantService)
            try:
                qdrant_results = self.qdrant_service.hybrid_search(
                    query_dense=dense_vector,
                    query_sparse=sparse_vector,
                    top=top,
                    filters=filters
                )
                
                if not qdrant_results:
                    logger.info(f"[SearchService] No results found for query: '{query}'")
                    return []
                
            except Exception as e:
                logger.error(f"[SearchService] Qdrant search failed: {e}")
                return []
            
            # Step 3: Map Qdrant results to our existing SearchResult format
            mapped_results = self._map_qdrant_results(qdrant_results, query)
            
            t_search = (time.time() - t0) * 1000.0
            logger.info(f"[SearchService] Search completed: {len(mapped_results)} results in {t_search:.1f}ms")
            
            return mapped_results
            
        except Exception as e:
            logger.error(f"[SearchService] Search failed: {e}")
            return []
    
    def _map_qdrant_results(self, qdrant_results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Map Qdrant ScoredPoint results to our existing SearchResult format
        
        Preserves fields: page_number, edition_number, content snippet, score
        """
        mapped_results = []
        
        for result in qdrant_results:
            # Extract content and create snippet
            full_content = result.get("content", "")
            snippet = self._create_snippet(full_content, query)
            
            mapped_result = {
                "id": result.get("id", ""),
                "title": result.get("title", ""),
                "description": result.get("description", ""),
                "content": snippet,  # Use snippet instead of full content
                "magazine_id": result.get("magazine_id", ""),
                "page_number": result.get("page_number", 1),
                "chunk_position": result.get("chunk_position", 0),
                "published_year": result.get("published_year", 0),
                "published_month": result.get("published_month", ""),
                "edition_number": result.get("edition_number", ""),
                "pdf_url": result.get("pdf_url", ""),
                "thumbnail_url": result.get("thumbnail_url", ""),
                "score": result.get("score", 0.0)
            }
            
            mapped_results.append(mapped_result)
        
        return mapped_results
    
    def _create_snippet(self, content: str, query: str, max_length: int = 500) -> str:
        """Create a snippet from content, highlighting query terms"""
        if len(content) <= max_length:
            return content
        
        # Find query terms in content
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        # Find first occurrence of any query term
        first_pos = len(content)
        for term in query_terms:
            pos = content_lower.find(term)
            if pos != -1 and pos < first_pos:
                first_pos = pos
        
        # Create snippet around first occurrence
        if first_pos < len(content):
            start = max(0, first_pos - 100)
            end = min(len(content), first_pos + max_length - 100)
            snippet = content[start:end]
            
            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."
            
            return snippet
        
        # Fallback: return first max_length characters
        return content[:max_length] + "..."
    
    def get_processed_files(self) -> List[str]:
        """Get list of already processed files (UNCHANGED)"""
        try:
            processed_files = []
            for blob in self.output_container.list_blobs():
                if blob.name.endswith(".json"):
                    magazine_id = blob.name.replace(".json", "").split("_")[0]
                    processed_files.append(magazine_id)
            return processed_files
        except Exception as e:
            logger.error(f"[SearchService] Failed to get processed files: {e}")
            return []
    
    def is_file_processed(self, magazine_id: str) -> bool:
        """Check if a magazine PDF has already been processed (UNCHANGED)"""
        processed_files = self.get_processed_files()
        return magazine_id in processed_files
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the search service configuration"""
        return {
            "embedding_service": self.embedding_service.get_model_info(),
            "qdrant_collection": self.qdrant_service.get_collection_info(),
            "embedding_batch_size": self.embedding_batch_size,
            "upsert_batch_size": self.upsert_batch_size
        }
