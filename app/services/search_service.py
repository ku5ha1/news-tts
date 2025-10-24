import os
import json
import time
import logging
import math
import difflib
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
        """Create or update the search index with enhanced schema"""
        try:
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="title", type=SearchFieldDataType.String),
                SearchableField(name="description", type=SearchFieldDataType.String),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SimpleField(name="magazine_id", type=SearchFieldDataType.String),
                SimpleField(name="page_number", type=SearchFieldDataType.Int32),  # NEW FIELD
                SimpleField(name="chunk_position", type=SearchFieldDataType.Int32),  # NEW FIELD
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
                    raw_embeddings = [item["embedding"] for item in data["data"]]
                    # Optional query/doc normalization can be applied at search-time; keep raw here
                    embeddings.extend(raw_embeddings)
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
        """Split text into semantic chunks (legacy method for compatibility)"""
        if not text.strip():
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def chunk_text_semantically(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Enhanced chunking that preserves semantic context"""
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
        # Simple sentence splitting - can be enhanced with NLTK if needed
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def process_magazine2_pdf(self, magazine_data: Dict[str, Any]) -> bool:
        """Process a single Magazine2 PDF through the enhanced pipeline with page-aware chunking"""
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
            
            # Step 1: Get OCR result (from existing JSON or fresh OCR)
            ocr_result = self._get_or_create_ocr_result(blob_name)
            if not ocr_result:
                return False
            
            # Step 2: Apply NEW page-aware chunking logic
            search_documents = self._create_search_documents_with_pages(
                ocr_result, magazine_data
            )
            
            if not search_documents:
                logger.warning(f"No search documents created for {magazine_id}")
                return False
            
            # Step 3: Upload to search index with enhanced validation
            try:
                # Test serialization before uploading
                import json
                json.dumps(search_documents[0])
                logger.info("Document serialization test passed")
            except Exception as e:
                logger.error(f"Document serialization test failed: {e}")
                raise ValueError(f"Document serialization failed: {e}")
            
            self.search_client.upload_documents(documents=search_documents)
            
            logger.info(f"Successfully processed {magazine_id}: {len(search_documents)} chunks indexed with page numbers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process magazine {magazine_id}: {e}")
            return False
    
    def _get_or_create_ocr_result(self, blob_name: str) -> Optional[Dict[str, Any]]:
        """Get existing OCR result or create new one"""
        try:
            # Try to load existing JSON first
            json_path = blob_name.replace(".pdf", ".json")
            
            try:
                json_blob = self.output_container.download_blob(json_path)
                json_data = json_blob.readall().decode('utf-8')
                ocr_result = json.loads(json_data)
                logger.info(f"Using existing OCR result: {json_path}")
                return ocr_result
            except Exception:
                # JSON doesn't exist or is corrupted, do fresh OCR
                logger.info(f"No existing OCR result found, performing fresh OCR: {blob_name}")
                pass
            
            # Perform fresh OCR
            ocr_result = self.ocr_pdf_from_blob(blob_name)
            if ocr_result:
                # Save the new OCR result
                self.save_ocr_result(blob_name, ocr_result)
            
            return ocr_result
            
        except Exception as e:
            logger.error(f"Failed to get/create OCR result for {blob_name}: {e}")
            return None
    
    def _create_search_documents_with_pages(self, ocr_result: Dict[str, Any], magazine_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create search documents using NEW page-aware chunking logic"""
        try:
            magazine_id = str(magazine_data["_id"])
            
            # Extract metadata using helper functions
            title = self._first_or_str(magazine_data.get("title"))
            description = self._first_or_str(magazine_data.get("description"))
            published_month = self._first_or_str(magazine_data.get("publishedMonth"))
            edition_number = self._first_or_str(magazine_data.get("editionNumber"))
            thumbnail_url = self._first_or_str(magazine_data.get("magazineThumbnail"))
            published_year = self._first_or_int(magazine_data.get("publishedYear"))
            pdf_url = magazine_data["magazinePdf"]
            
            search_documents = []
            
            # NEW: Process each page separately to maintain page numbers
            for page in ocr_result["pages"]:
                page_number = page["page_number"]
                page_content = page["content"]
                
                if not page_content.strip():
                    continue  # Skip empty pages
                    
                # Apply NEW semantic chunking to this page
                page_chunks = self.chunk_text_semantically(page_content)
                
                if page_chunks:
                    # Generate embeddings for this page's chunks
                    page_embeddings = self.generate_embeddings(page_chunks)
                    
                    for chunk_index, (chunk, embedding) in enumerate(zip(page_chunks, page_embeddings)):
                        doc = {
                            "id": f"{magazine_id}_page_{page_number}_chunk_{chunk_index}",  # NEW ID format
                            "title": title,
                            "description": description,
                            "content": chunk,
                            "magazine_id": magazine_id,
                            "page_number": page_number,  # NEW FIELD
                            "chunk_position": chunk_index,  # NEW FIELD
                            "published_year": published_year,
                            "published_month": published_month,
                            "edition_number": edition_number,
                            "pdf_url": pdf_url,
                            "thumbnail_url": thumbnail_url,
                            "contentVector": [float(x) for x in embedding]  # Ensure float format
                        }
                        search_documents.append(doc)
            
            logger.info(f"Created {len(search_documents)} search documents with page information")
            return search_documents
            
        except Exception as e:
            logger.error(f"Failed to create search documents: {e}")
            return []
    
    def search_documents(self, query: str, top: int = 10, filters: Optional[Dict[str, Any]] = None, vector_weight: Optional[float] = None) -> List[Dict[str, Any]]:
        """Enhanced search with improved scoring and page number support"""
        try:
            # Preprocess query for better matching
            processed_query = self._preprocess_query(query)
            expanded_query = self._expand_query(processed_query)
            
            # Generate query embedding
            query_embeddings = self.generate_embeddings([query])
            query_vector = query_embeddings[0]
            
            # L2-normalize query vector for stable cosine-like behavior
            try:
                norm = math.sqrt(sum(x * x for x in query_vector)) or 1.0
                query_vector = [x / norm for x in query_vector]
            except Exception:
                pass
            
            vector_query = VectorizedQuery(vector=query_vector, fields="contentVector")
            
            # Dynamic vector weight based on query type
            effective_weight = self._calculate_vector_weight(query, vector_weight)
            
            try:
                vector_query.weight = float(effective_weight)
            except Exception:
                pass

            search_kwargs: Dict[str, Any] = {
                "search_text": expanded_query,  # Use expanded query for better recall
                "vector_queries": [vector_query],
                "semantic_configuration_name": "semanticConfig",
                "highlight_fields": "content",
                "highlight_pre_tag": "<mark>",
                "highlight_post_tag": "</mark>",
                "top": top * 2,  # Get more results for re-ranking
                "include_total_count": True,
            }

            # Apply filters
            if filters:
                clauses = []
                for key, value in filters.items():
                    if value is None:
                        continue
                    if isinstance(value, int):
                        clauses.append(f"{key} eq {value}")
                    else:
                        safe_val = str(value).replace("'", "''")
                        clauses.append(f"{key} eq '{safe_val}'")
                if clauses:
                    search_kwargs["filter"] = " and ".join(clauses)

            t0 = time.time()
            results = self.search_client.search(**search_kwargs)
            t_search = (time.time() - t0) * 1000.0
            
            # Enhanced result processing with re-ranking
            processed_results = self._process_and_rerank_results(
                results, query, processed_query
            )
            
            # Apply enhanced deduplication
            deduplicated_results = self._enhanced_deduplication(processed_results)
            
            # Return top results after re-ranking
            final_results = deduplicated_results[:top]
            
            logger.info(f"[ENHANCED_SEARCH] query='{query}' vector_weight={effective_weight} results={len(final_results)} t_search_ms={t_search:.1f}")
            return final_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better matching"""
        query = query.strip().lower()
        # Remove extra spaces
        query = " ".join(query.split())
        return query
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms"""
        # Simple query expansion - can be enhanced with domain-specific synonyms
        expansions = {
            "bridge": "bridge infrastructure crossing construction",
            "temple": "temple shrine religious architecture",
            "festival": "festival celebration cultural event",
            "government": "government administration policy"
        }
        
        words = query.split()
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            if word in expansions:
                expanded_words.extend(expansions[word].split())
        
        return " ".join(expanded_words)
    
    def _calculate_vector_weight(self, query: str, provided_weight: Optional[float]) -> float:
        """Calculate dynamic vector weight based on query characteristics"""
        if provided_weight is not None:
            return provided_weight
        
        # Analyze query to determine optimal vector weight
        query_lower = query.lower()
        
        # Infrastructure/technical queries benefit from higher vector weight
        if any(term in query_lower for term in ["bridge", "infrastructure", "construction", "engineering"]):
            return 6.0
        
        # General queries use balanced weight
        if any(term in query_lower for term in ["karnataka", "bangalore", "government"]):
            return 4.0
        
        # Tourism/cultural queries use lower vector weight (favor exact matches)
        if any(term in query_lower for term in ["festival", "temple", "culture", "tourism"]):
            return 3.0
        
        return 4.0  # Default
    
    def _process_and_rerank_results(self, results, original_query: str, processed_query: str) -> List[Dict[str, Any]]:
        """Process results and apply custom re-ranking"""
        processed_results = []
        
        for result in results:
            # Extract highlighted content
            highlights = result.get("@search.highlights", {})
            highlighted_content = highlights.get("content", [])
            
            if highlighted_content:
                content = " ... ".join(highlighted_content)
            else:
                full_content = result.get("content", "")
                content = full_content[:500] + "..." if len(full_content) > 500 else full_content

            # Calculate enhanced relevance score
            base_score = result.get("@search.score", 0)
            enhanced_score = self._calculate_enhanced_score(
                result, original_query, processed_query, base_score
            )

            processed_result = {
                "id": result["id"],
                "title": result.get("title", ""),
                "description": result.get("description", ""),
                "content": content,
                "magazine_id": result.get("magazine_id", ""),
                "page_number": result.get("page_number", 1),  # NEW FIELD
                "chunk_position": result.get("chunk_position", 0),  # NEW FIELD
                "published_year": result.get("published_year", 0),
                "published_month": result.get("published_month", ""),
                "edition_number": result.get("edition_number", ""),
                "pdf_url": result.get("pdf_url", ""),
                "thumbnail_url": result.get("thumbnail_url", ""),
                "score": enhanced_score,
                "original_score": base_score
            }
            processed_results.append(processed_result)
        
        # Sort by enhanced score
        processed_results.sort(key=lambda x: x["score"], reverse=True)
        return processed_results
    
    def _calculate_enhanced_score(self, result: Dict, query: str, processed_query: str, base_score: float) -> float:
        """Calculate enhanced relevance score using multiple factors"""
        content = result.get("content", "").lower()
        title = result.get("title", "").lower()
        query_terms = processed_query.split()
        
        # Factor 1: Base search score (40%)
        score = base_score * 0.4
        
        # Factor 2: Term frequency in content (25%)
        term_freq_score = 0
        for term in query_terms:
            term_freq_score += content.count(term) * 0.1
        score += min(term_freq_score, 1.0) * 0.25
        
        # Factor 3: Term proximity (20%)
        proximity_score = self._calculate_proximity_score(content, query_terms)
        score += proximity_score * 0.2
        
        # Factor 4: Title relevance (10%)
        title_score = 0
        for term in query_terms:
            if term in title:
                title_score += 0.2
        score += min(title_score, 1.0) * 0.1
        
        # Factor 5: Content type relevance (5%)
        content_type_score = self._calculate_content_type_score(content, query)
        score += content_type_score * 0.05
        
        return score
    
    def _calculate_proximity_score(self, content: str, query_terms: List[str]) -> float:
        """Calculate score based on proximity of query terms"""
        if len(query_terms) < 2:
            return 1.0
        
        words = content.split()
        positions = {}
        
        # Find positions of each query term
        for term in query_terms:
            positions[term] = [i for i, word in enumerate(words) if term in word.lower()]
        
        if not all(positions.values()):
            return 0.0
        
        # Calculate minimum distance between terms
        min_distance = float('inf')
        for i, term1 in enumerate(query_terms):
            for j, term2 in enumerate(query_terms[i+1:], i+1):
                for pos1 in positions[term1]:
                    for pos2 in positions[term2]:
                        distance = abs(pos1 - pos2)
                        min_distance = min(min_distance, distance)
        
        # Convert distance to score (closer = higher score)
        if min_distance == float('inf'):
            return 0.0
        
        return max(0, 1.0 - (min_distance / 50.0))  # Normalize to 0-1
    
    def _calculate_content_type_score(self, content: str, query: str) -> float:
        """Boost score based on content type relevance to query"""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Infrastructure queries should boost infrastructure content
        if "bridge" in query_lower or "infrastructure" in query_lower:
            if any(term in content_lower for term in ["construction", "engineering", "infrastructure", "project"]):
                return 1.0
            if any(term in content_lower for term in ["playground", "game", "children"]):
                return 0.2  # Penalize playground equipment for infrastructure queries
        
        return 0.5  # Neutral score
    
    def _enhanced_deduplication(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced deduplication using improved similarity detection"""
        if len(results) <= 1:
            return results
        
        # Group by magazine_id first
        magazine_groups = {}
        for result in results:
            mag_id = result["magazine_id"]
            if mag_id not in magazine_groups:
                magazine_groups[mag_id] = []
            magazine_groups[mag_id].append(result)
        
        deduplicated = []
        
        for mag_id, mag_results in magazine_groups.items():
            if len(mag_results) == 1:
                deduplicated.extend(mag_results)
                continue
            
            # For multiple results from same magazine, keep diverse ones
            kept_results = [mag_results[0]]  # Always keep highest scoring
            
            for candidate in mag_results[1:]:
                is_duplicate = False
                candidate_content = candidate["content"].lower().replace("<mark>", "").replace("</mark>", "")
                
                for kept in kept_results:
                    kept_content = kept["content"].lower().replace("<mark>", "").replace("</mark>", "")
                    
                    # Use improved similarity detection
                    similarity = self._calculate_text_similarity(candidate_content, kept_content)
                    
                    if similarity > 0.85:  # Threshold for considering duplicate
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    kept_results.append(candidate)
                    
                # Limit results per magazine
                if len(kept_results) >= 3:
                    break
            
            deduplicated.extend(kept_results)
        
        return deduplicated
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        import difflib
        return difflib.SequenceMatcher(a=text1, b=text2).ratio()
    
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
