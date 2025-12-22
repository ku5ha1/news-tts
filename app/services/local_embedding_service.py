import logging
from typing import List, Dict, Any
from fastembed import TextEmbedding, SparseTextEmbedding
import gc
import os

from app.config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalEmbeddingService:
    """Service for generating embeddings locally using FastEmbed"""
    
    def __init__(self):
        """Initialize FastEmbed models for dense and sparse embeddings"""
        try:
            logger.info("[LocalEmbedding] Initializing FastEmbed models...")
            
            # Dense embedding model: mixedbread-ai/mxbai-embed-large-v1 (1024-dim)
            cache_dir = settings.HF_HUB_CACHE or os.path.expanduser("~/.cache/huggingface/hub")
            self.dense_model = TextEmbedding(
                model_name="mixedbread-ai/mxbai-embed-large-v1",
                max_length=512,  # Maximum token length for input
                cache_dir=cache_dir
            )
            logger.info("[LocalEmbedding] Dense model loaded: mixedbread-ai/mxbai-embed-large-v1")
            
            # Sparse embedding model: Qdrant/bm25 for keyword search
            self.sparse_model = SparseTextEmbedding(
                model_name="Qdrant/bm25",
                cache_dir=cache_dir
            )
            logger.info("[LocalEmbedding] Sparse model loaded: Qdrant/bm25")
            
            # Configuration
            self.dense_batch_size = 4  # Keep CPU usage stable
            self.target_dim = 1024  # Target dimension for Qdrant collection (mixedbread model is 1024-dim)
            
            logger.info(f"[LocalEmbedding] Configuration: batch_size={self.dense_batch_size}, target_dim={self.target_dim}")
            
        except Exception as e:
            logger.error(f"[LocalEmbedding] Failed to initialize models: {e}")
            raise
    
    def generate_dense_embeddings(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """
        Generate dense embeddings using snowflake-arctic-embed-l-v2.0
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing (default: 4)
        
        Returns:
            List of 1024-dimensional dense vectors
        """
        if batch_size is None:
            batch_size = self.dense_batch_size
        
        try:
            logger.info(f"[LocalEmbedding] Generating dense embeddings for {len(texts)} texts (batch_size={batch_size})")
            
            all_embeddings = []
            
            # Process in batches to manage memory
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Generate embeddings for batch
                batch_embeddings = list(self.dense_model.embed(batch))
                
                # mixedbread-ai/mxbai-embed-large-v1 outputs 1024 dimensions
                embeddings_list = [
                    embedding.tolist() 
                    for embedding in batch_embeddings
                ]
                
                all_embeddings.extend(embeddings_list)
                
                # Force garbage collection after each batch to free memory
                gc.collect()
                
                logger.info(f"[LocalEmbedding] Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            logger.info(f"[LocalEmbedding] Generated {len(all_embeddings)} dense embeddings (dim={self.target_dim})")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"[LocalEmbedding] Failed to generate dense embeddings: {e}")
            raise
    
    def generate_sparse_embeddings(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Generate sparse embeddings using Qdrant/bm25
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of sparse vectors in Qdrant format (indices + values)
        """
        try:
            logger.info(f"[LocalEmbedding] Generating sparse embeddings for {len(texts)} texts")
            
            # Generate sparse embeddings
            sparse_embeddings = list(self.sparse_model.embed(texts))
            
            # Convert to Qdrant sparse vector format
            qdrant_sparse_vectors = []
            for sparse_embedding in sparse_embeddings:
                # FastEmbed returns sparse vectors with indices and values
                sparse_vector = {
                    "indices": sparse_embedding.indices.tolist(),
                    "values": sparse_embedding.values.tolist()
                }
                qdrant_sparse_vectors.append(sparse_vector)
            
            logger.info(f"[LocalEmbedding] Generated {len(qdrant_sparse_vectors)} sparse embeddings")
            return qdrant_sparse_vectors
            
        except Exception as e:
            logger.error(f"[LocalEmbedding] Failed to generate sparse embeddings: {e}")
            raise
    
    def generate_hybrid_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = None
    ) -> tuple[List[List[float]], List[Dict[str, Any]]]:
        """
        Generate both dense and sparse embeddings for hybrid search
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for dense embedding generation
        
        Returns:
            Tuple of (dense_embeddings, sparse_embeddings)
        """
        try:
            logger.info(f"[LocalEmbedding] Generating hybrid embeddings for {len(texts)} texts")
            
            # Generate dense embeddings
            dense_embeddings = self.generate_dense_embeddings(texts, batch_size)
            
            # Generate sparse embeddings
            sparse_embeddings = self.generate_sparse_embeddings(texts)
            
            logger.info("[LocalEmbedding] Hybrid embeddings generated successfully")
            return dense_embeddings, sparse_embeddings
            
        except Exception as e:
            logger.error(f"[LocalEmbedding] Failed to generate hybrid embeddings: {e}")
            raise
    
    def generate_query_embeddings(self, query: str) -> tuple[List[float], Dict[str, Any]]:
        """
        Generate embeddings for a single query (optimized for search)
        
        Args:
            query: Query string
        
        Returns:
            Tuple of (dense_vector, sparse_vector)
        """
        try:
            # Generate dense embedding
            dense_embeddings = self.generate_dense_embeddings([query], batch_size=1)
            dense_vector = dense_embeddings[0]
            
            # Generate sparse embedding
            sparse_embeddings = self.generate_sparse_embeddings([query])
            sparse_vector = sparse_embeddings[0]
            
            return dense_vector, sparse_vector
            
        except Exception as e:
            logger.error(f"[LocalEmbedding] Failed to generate query embeddings: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "dense_model": "mixedbread-ai/mxbai-embed-large-v1",
            "sparse_model": "Qdrant/bm25",
            "target_dimension": self.target_dim,
            "batch_size": self.dense_batch_size,
            "max_token_length": 512
        }
    
    def cleanup(self):
        """Cleanup models and free memory"""
        try:
            del self.dense_model
            del self.sparse_model
            gc.collect()
            logger.info("[LocalEmbedding] Models cleaned up and memory freed")
        except Exception as e:
            logger.error(f"[LocalEmbedding] Failed to cleanup models: {e}")
