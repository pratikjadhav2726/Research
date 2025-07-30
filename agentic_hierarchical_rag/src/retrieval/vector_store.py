"""Vector store implementation for efficient similarity search."""

import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict, Any
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """Simple vector store using Faiss for similarity search."""
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        metric: str = "cosine",
    ):
        """Initialize the vector store.
        
        Args:
            dimension: Dimension of vectors
            index_type: Type of Faiss index ("flat", "ivf", "hnsw")
            metric: Distance metric ("cosine", "l2", "ip")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        
        # Create Faiss index
        self.index = self._create_index()
        
        # Store metadata separately
        self.metadata: List[Dict[str, Any]] = []
        
    def _create_index(self) -> faiss.Index:
        """Create Faiss index based on configuration.
        
        Returns:
            Faiss index
        """
        if self.metric == "cosine":
            # For cosine similarity, we normalize vectors and use inner product
            if self.index_type == "flat":
                index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
        
        elif self.metric == "l2":
            if self.index_type == "flat":
                index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
                
        elif self.metric == "ip":
            # Inner product without normalization
            if self.index_type == "flat":
                index = faiss.IndexFlatIP(self.dimension)
            else:
                raise ValueError(f"Inner product only supported for flat index")
        
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
            
        return index
    
    def add(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """Add vectors to the store.
        
        Args:
            embeddings: Vectors to add (shape: [n, dimension])
            metadata: Optional metadata for each vector
            
        Returns:
            List of indices for added vectors
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
            
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} doesn't match "
                f"index dimension {self.dimension}"
            )
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            embeddings = self._normalize_vectors(embeddings)
        
        # Get current size before adding
        start_idx = self.index.ntotal
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        if metadata:
            if len(metadata) != len(embeddings):
                raise ValueError("Metadata length must match embeddings length")
            self.metadata.extend(metadata)
        else:
            # Add empty metadata
            self.metadata.extend([{} for _ in range(len(embeddings))])
        
        # Return indices
        indices = list(range(start_idx, self.index.ntotal))
        logger.info(f"Added {len(embeddings)} vectors to store")
        
        return indices
    
    def search(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        filter_fn: Optional[callable] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[List[Dict[str, Any]]]]:
        """Search for similar vectors.
        
        Args:
            query_vectors: Query vectors (shape: [n_queries, dimension])
            k: Number of results per query
            filter_fn: Optional function to filter results based on metadata
            
        Returns:
            Tuple of (distances, indices, metadata)
        """
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
            
        if query_vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query_vectors.shape[1]} doesn't match "
                f"index dimension {self.dimension}"
            )
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            query_vectors = self._normalize_vectors(query_vectors)
        
        # Search
        distances, indices = self.index.search(query_vectors, k)
        
        # Convert distances to similarities for cosine
        if self.metric == "cosine":
            # Inner product of normalized vectors = cosine similarity
            similarities = distances
        else:
            # For L2, convert to similarity (inverse of distance)
            similarities = 1 / (1 + distances)
        
        # Apply filter if provided
        if filter_fn:
            filtered_results = []
            for i in range(len(query_vectors)):
                query_results = []
                for j in range(k):
                    idx = indices[i, j]
                    if idx >= 0 and idx < len(self.metadata):
                        meta = self.metadata[idx]
                        if filter_fn(meta):
                            query_results.append((similarities[i, j], idx, meta))
                
                # Sort by similarity and take top k
                query_results.sort(key=lambda x: x[0], reverse=True)
                query_results = query_results[:k]
                
                # Reconstruct arrays
                if query_results:
                    sim, idx, meta = zip(*query_results)
                    similarities[i, :len(sim)] = sim
                    indices[i, :len(idx)] = idx
                else:
                    similarities[i, :] = -1
                    indices[i, :] = -1
        
        # Gather metadata
        metadata_results = []
        for i in range(len(query_vectors)):
            query_metadata = []
            for j in range(k):
                idx = indices[i, j]
                if idx >= 0 and idx < len(self.metadata):
                    query_metadata.append(self.metadata[idx])
                else:
                    query_metadata.append({})
            metadata_results.append(query_metadata)
        
        return similarities, indices, metadata_results
    
    def get_by_indices(self, indices: List[int]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Get vectors and metadata by indices.
        
        Args:
            indices: List of indices
            
        Returns:
            Tuple of (vectors, metadata)
        """
        vectors = []
        metadata = []
        
        for idx in indices:
            if 0 <= idx < self.index.ntotal:
                # Reconstruct vector
                vec = self.index.reconstruct(int(idx))
                vectors.append(vec)
                
                # Get metadata
                if idx < len(self.metadata):
                    metadata.append(self.metadata[idx])
                else:
                    metadata.append({})
        
        if vectors:
            return np.vstack(vectors), metadata
        else:
            return np.array([]), []
    
    def remove(self, indices: List[int]) -> None:
        """Remove vectors by indices.
        
        Note: This is not supported by all Faiss index types.
        
        Args:
            indices: Indices to remove
        """
        if hasattr(self.index, 'remove_ids'):
            self.index.remove_ids(np.array(indices, dtype=np.int64))
            # Note: This doesn't update metadata indices, which is a limitation
            logger.warning("Metadata indices may be inconsistent after removal")
        else:
            raise NotImplementedError(
                f"Removal not supported for index type {self.index_type}"
            )
    
    def clear(self) -> None:
        """Clear all vectors from the store."""
        self.index.reset()
        self.metadata.clear()
        logger.info("Cleared vector store")
    
    def save(self, path: Path) -> None:
        """Save the vector store to disk.
        
        Args:
            path: Path to save directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save index
        index_path = path / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        
        # Save configuration
        config = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "ntotal": self.index.ntotal,
        }
        config_path = path / "config.pkl"
        with open(config_path, "wb") as f:
            pickle.dump(config, f)
        
        logger.info(f"Saved vector store to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        """Load a vector store from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            Loaded VectorStore instance
        """
        path = Path(path)
        
        # Load configuration
        config_path = path / "config.pkl"
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        
        # Create instance
        store = cls(
            dimension=config["dimension"],
            index_type=config["index_type"],
            metric=config["metric"]
        )
        
        # Load index
        index_path = path / "index.faiss"
        store.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, "rb") as f:
            store.metadata = pickle.load(f)
        
        logger.info(f"Loaded vector store from {path}")
        return store
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity.
        
        Args:
            vectors: Vectors to normalize
            
        Returns:
            Normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        return vectors / norms
    
    @property
    def size(self) -> int:
        """Get number of vectors in the store."""
        return self.index.ntotal