"""Embedding generation for text chunks and queries."""

from typing import List, Union, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Embedder:
    """Handles text embedding generation using various models."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress: bool = True,
    ):
        """Initialize the embedder.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use (cuda/cpu), auto-detected if None
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to normalize embeddings
            show_progress: Whether to show progress bar
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress = show_progress
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load the model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Embedder initialized with model {model_name} on {self.device}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def embed_texts(
        self,
        texts: Union[str, List[str]],
        convert_to_numpy: bool = True,
        show_progress_bar: Optional[bool] = None,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts to embed
            convert_to_numpy: Whether to convert to numpy arrays
            show_progress_bar: Override default progress bar setting
            
        Returns:
            Embeddings as numpy array or list of arrays
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
            
        # Determine whether to show progress
        if show_progress_bar is None:
            show_progress_bar = self.show_progress and len(texts) > 10
            
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=show_progress_bar,
        )
        
        # Return single embedding if single input
        if single_input:
            return embeddings[0]
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query.
        
        This is a convenience method that ensures single query handling
        and always returns a numpy array.
        
        Args:
            query: Query text to embed
            
        Returns:
            Query embedding as numpy array
        """
        return self.embed_texts(query, convert_to_numpy=True, show_progress_bar=False)
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Embed a batch of texts with optional custom batch size.
        
        Args:
            texts: List of texts to embed
            batch_size: Override default batch size
            
        Returns:
            Embeddings as numpy array of shape (n_texts, embedding_dim)
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Process in batches for memory efficiency
        all_embeddings = []
        
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Generating embeddings",
            disable=not self.show_progress,
        ):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            all_embeddings.append(embeddings)
        
        # Concatenate all embeddings
        return np.vstack(all_embeddings)
    
    def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        metric: str = "cosine",
    ) -> np.ndarray:
        """Compute similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings (shape: [n, d] or [d])
            embeddings2: Second set of embeddings (shape: [m, d] or [d])
            metric: Similarity metric ("cosine" or "dot")
            
        Returns:
            Similarity matrix of shape [n, m] or scalar if both inputs are 1D
        """
        # Ensure 2D arrays
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)
            
        if metric == "cosine":
            # Normalize if not already normalized
            if not self.normalize_embeddings:
                embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
                embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
            # Cosine similarity is just dot product of normalized vectors
            similarity = np.dot(embeddings1, embeddings2.T)
        elif metric == "dot":
            similarity = np.dot(embeddings1, embeddings2.T)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
            
        # Return scalar if both inputs were 1D
        if similarity.shape == (1, 1):
            return similarity[0, 0]
        return similarity
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "max_sequence_length": self.model.max_seq_length,
        }
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save embeddings to disk with optional metadata.
        
        Args:
            embeddings: Embeddings to save
            filepath: Path to save file
            metadata: Optional metadata to save alongside embeddings
        """
        save_dict = {
            "embeddings": embeddings,
            "model_info": self.get_model_info(),
        }
        if metadata:
            save_dict["metadata"] = metadata
            
        np.savez_compressed(filepath, **save_dict)
        logger.info(f"Saved embeddings to {filepath}")
    
    @staticmethod
    def load_embeddings(filepath: str) -> Dict[str, Any]:
        """Load embeddings from disk.
        
        Args:
            filepath: Path to saved embeddings
            
        Returns:
            Dictionary with embeddings and metadata
        """
        data = np.load(filepath, allow_pickle=True)
        return {
            "embeddings": data["embeddings"],
            "model_info": data["model_info"].item() if "model_info" in data else None,
            "metadata": data["metadata"].item() if "metadata" in data else None,
        }