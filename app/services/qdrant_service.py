import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVectorParams,
    SparseIndexParams,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    QuantizationSearchParams,
    OptimizersConfigDiff,
)

from app.config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantService:
    """Service for managing Qdrant Cloud vector database operations"""
    
    def __init__(self):
        """Initialize Qdrant Cloud client"""
        self.qdrant_url = settings.QDRANT_URL
        self.qdrant_api_key = settings.QDRANT_API_KEY
        self.collection_name = settings.QDRANT_COLLECTION_NAME or "magazine2_search"
        
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables")
        
        try:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
            )
            logger.info(f"[Qdrant] Connected to Qdrant Cloud: {self.qdrant_url}")
            
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"[Qdrant] Available collections: {collections}")
            
        except Exception as e:
            logger.error(f"[Qdrant] Failed to connect: {e}")
            raise
    
    def create_collection(self, force_recreate: bool = False) -> bool:
        """
        Create Qdrant collection with Named Vectors for hybrid search
        
        Configuration:
        - Dense vector: 1024-dim with Binary Quantization
        - Sparse vector: BM25-based for keyword search
        - Optimized for <3GB RAM usage on VM
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if collection_exists:
                if force_recreate:
                    logger.info(f"[Qdrant] Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"[Qdrant] Collection '{self.collection_name}' already exists")
                    return True
            
            logger.info(f"[Qdrant] Creating collection: {self.collection_name}")
            
            # Create collection with Named Vectors
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=1024,
                        distance=Distance.COSINE,
                        quantization_config=ScalarQuantization(
                            scalar=ScalarQuantizationConfig(
                                type=ScalarType.INT8,
                                quantile=0.99,
                                always_ram=True
                            )
                        )
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=False  # Keep sparse index in RAM for speed
                        )
                    )
                },
                optimizers_config=OptimizersConfigDiff(
                    memmap_threshold=20000  # Use memory-mapped files for large collections
                )
            )
            
            logger.info(f"[Qdrant] Collection '{self.collection_name}' created successfully")
            logger.info("[Qdrant] Configuration: 1024-dim dense (scalar quantized INT8) + sparse (BM25)")
            return True
            
        except Exception as e:
            logger.error(f"[Qdrant] Failed to create collection: {e}")
            return False
    
    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """Get collection information"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"[Qdrant] Failed to get collection info: {e}")
            return None
    
    def delete_collection(self) -> bool:
        """Delete the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"[Qdrant] Collection '{self.collection_name}' deleted")
            return True
        except Exception as e:
            logger.error(f"[Qdrant] Failed to delete collection: {e}")
            return False
    
    def upsert_documents(
        self,
        documents: List[Dict[str, Any]],
        dense_embeddings: List[List[float]],
        sparse_embeddings: List[Dict[str, Any]]
    ) -> bool:
        """
        Upsert documents with dense and sparse vectors
        
        Args:
            documents: List of document metadata (id, content, magazine_id, page_number, etc.)
            dense_embeddings: List of 1024-dim dense vectors
            sparse_embeddings: List of sparse vectors (indices + values)
        """
        try:
            if len(documents) != len(dense_embeddings) or len(documents) != len(sparse_embeddings):
                raise ValueError("Documents, dense_embeddings, and sparse_embeddings must have same length")
            
            points = []
            for doc, dense_vec, sparse_vec in zip(documents, dense_embeddings, sparse_embeddings):
                point = PointStruct(
                    id=doc["id"],
                    vector={
                        "dense": dense_vec,
                        "sparse": sparse_vec
                    },
                    payload={
                        "title": doc.get("title", ""),
                        "description": doc.get("description", ""),
                        "content": doc.get("content", ""),
                        "magazine_id": doc.get("magazine_id", ""),
                        "page_number": doc.get("page_number", 1),
                        "chunk_position": doc.get("chunk_position", 0),
                        "published_year": doc.get("published_year", 0),
                        "published_month": doc.get("published_month", ""),
                        "edition_number": doc.get("edition_number", ""),
                        "pdf_url": doc.get("pdf_url", ""),
                        "thumbnail_url": doc.get("thumbnail_url", "")
                    }
                )
                points.append(point)
            
            # Upsert in batches to manage memory
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.info(f"[Qdrant] Upserted batch {i//batch_size + 1}: {len(batch)} points")
            
            logger.info(f"[Qdrant] Successfully upserted {len(points)} documents")
            return True
            
        except Exception as e:
            logger.error(f"[Qdrant] Failed to upsert documents: {e}")
            return False
    
    def hybrid_search(
        self,
        query_dense: List[float],
        query_sparse: Dict[str, Any],
        top: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using RRF (Reciprocal Rank Fusion) with rescoring
        
        Args:
            query_dense: 1024-dim dense query vector
            query_sparse: Sparse query vector (indices + values)
            top: Number of results to return
            filters: Optional filters for payload fields
        
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Perform dense vector search
            dense_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("dense", query_dense),
                limit=top * 2,  # Oversample by 2x for better accuracy
                search_params=QuantizationSearchParams(
                    rescore=True,  # Enable rescoring for binary quantization
                    oversampling=2.0  # Oversample by 2x for accuracy
                )
            )
            
            # Perform sparse vector search
            sparse_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("sparse", query_sparse),
                limit=top * 2
            )
            
            # Implement RRF (Reciprocal Rank Fusion) manually
            rrf_scores = {}
            k = 60  # RRF constant
            
            # Score dense results
            for rank, result in enumerate(dense_results, 1):
                doc_id = result.id
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
            
            # Score sparse results
            for rank, result in enumerate(sparse_results, 1):
                doc_id = result.id
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
            
            # Create a map of doc_id to result for easy lookup
            results_map = {}
            for result in dense_results + sparse_results:
                if result.id not in results_map:
                    results_map[result.id] = result
            
            # Sort by RRF score and get top results
            sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top]
            
            # Map results to our format
            final_results = []
            for doc_id, rrf_score in sorted_ids:
                result = results_map[doc_id]
                final_result = {
                    "id": result.id,
                    "score": rrf_score,
                    **result.payload
                }
                final_results.append(final_result)
            
            logger.info(f"[Qdrant] Hybrid search returned {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"[Qdrant] Hybrid search failed: {e}")
            return []
    
    def delete_by_magazine_id(self, magazine_id: str) -> bool:
        """Delete all documents for a specific magazine"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector={
                    "filter": {
                        "must": [
                            {
                                "key": "magazine_id",
                                "match": {"value": magazine_id}
                            }
                        ]
                    }
                }
            )
            logger.info(f"[Qdrant] Deleted documents for magazine_id: {magazine_id}")
            return True
        except Exception as e:
            logger.error(f"[Qdrant] Failed to delete documents: {e}")
            return False
