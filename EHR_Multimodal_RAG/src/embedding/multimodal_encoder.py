"""
Multimodal Encoder for EHR Multimodal RAG System

This module implements advanced multimodal encoding techniques specifically
designed for healthcare data, incorporating clinical concept alignment,
hierarchical embeddings, and cross-modal understanding.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoImageProcessor,
    CLIPModel, CLIPProcessor, CLIPTextModel, CLIPVisionModel
)
from sentence_transformers import SentenceTransformer
import clip

from ..utils.clinical_validators import validate_clinical_text
from .clinical_alignment import ClinicalConceptAligner
from .hierarchical_embedder import HierarchicalEmbedder


@dataclass
class MultimodalEmbedding:
    """Container for multimodal embeddings with metadata."""
    text_embedding: Optional[torch.Tensor] = None
    image_embedding: Optional[torch.Tensor] = None
    unified_embedding: Optional[torch.Tensor] = None
    hierarchical_embedding: Optional[Dict[str, torch.Tensor]] = None
    clinical_concepts: Optional[List[str]] = None
    confidence_scores: Optional[Dict[str, float]] = None
    modality_weights: Optional[Dict[str, float]] = None


class ClinicalCLIPEncoder(nn.Module):
    """
    Clinical CLIP encoder with medical domain adaptation.
    
    This encoder extends the standard CLIP architecture with:
    - Medical vocabulary understanding
    - Clinical concept alignment
    - Healthcare-specific fine-tuning
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Load base CLIP model
        model_name = config.get('model_name', 'openai/clip-vit-base-patch32')
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        
        # Clinical text encoder (specialized for medical text)
        clinical_text_model = config.get('clinical_text_model', 
                                       'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        self.clinical_text_encoder = AutoModel.from_pretrained(clinical_text_model)
        self.clinical_tokenizer = AutoTokenizer.from_pretrained(clinical_text_model)
        
        # Embedding dimensions
        self.embedding_dim = config.get('embedding_dim', 768)
        self.clip_dim = self.clip_model.config.projection_dim
        self.clinical_dim = self.clinical_text_encoder.config.hidden_size
        
        # Projection layers for dimension alignment
        self.clip_text_proj = nn.Linear(self.clip_dim, self.embedding_dim)
        self.clip_vision_proj = nn.Linear(self.clip_dim, self.embedding_dim)
        self.clinical_text_proj = nn.Linear(self.clinical_dim, self.embedding_dim)
        
        # Cross-modal attention for better alignment
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Clinical concept alignment layer
        self.concept_alignment = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_text(self, texts: List[str], use_clinical: bool = True) -> torch.Tensor:
        """
        Encode text using both CLIP and clinical encoders.
        
        Args:
            texts: List of text strings to encode
            use_clinical: Whether to use clinical text encoder
            
        Returns:
            Text embeddings tensor
        """
        device = next(self.parameters()).device
        
        # CLIP text encoding
        clip_inputs = self.clip_processor(text=texts, return_tensors="pt", 
                                        padding=True, truncation=True)
        clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
        
        with torch.no_grad():
            clip_text_features = self.clip_model.get_text_features(**clip_inputs)
        
        clip_embeddings = self.clip_text_proj(clip_text_features)
        
        if use_clinical:
            # Clinical text encoding
            clinical_inputs = self.clinical_tokenizer(
                texts, return_tensors="pt", padding=True, 
                truncation=True, max_length=512
            )
            clinical_inputs = {k: v.to(device) for k, v in clinical_inputs.items()}
            
            clinical_outputs = self.clinical_text_encoder(**clinical_inputs)
            clinical_embeddings = self.clinical_text_proj(
                clinical_outputs.pooler_output
            )
            
            # Combine CLIP and clinical embeddings
            combined_embeddings = torch.stack([clip_embeddings, clinical_embeddings], dim=1)
            
            # Apply cross-attention
            attended_embeddings, _ = self.cross_attention(
                combined_embeddings, combined_embeddings, combined_embeddings
            )
            
            # Pool the attended embeddings
            text_embeddings = attended_embeddings.mean(dim=1)
        else:
            text_embeddings = clip_embeddings
        
        # Apply concept alignment
        text_embeddings = self.concept_alignment(text_embeddings)
        
        return F.normalize(text_embeddings, p=2, dim=-1)
    
    def encode_image(self, images: Union[torch.Tensor, List]) -> torch.Tensor:
        """
        Encode images using CLIP vision encoder.
        
        Args:
            images: Images to encode (tensor or list of PIL images)
            
        Returns:
            Image embeddings tensor
        """
        device = next(self.parameters()).device
        
        if isinstance(images, list):
            # Process PIL images
            clip_inputs = self.clip_processor(images=images, return_tensors="pt")
            clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
        else:
            # Handle tensor input
            clip_inputs = {"pixel_values": images.to(device)}
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**clip_inputs)
        
        image_embeddings = self.clip_vision_proj(image_features)
        
        return F.normalize(image_embeddings, p=2, dim=-1)
    
    def forward(self, texts: Optional[List[str]] = None, 
                images: Optional[Union[torch.Tensor, List]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multimodal encoding.
        
        Args:
            texts: Optional list of texts
            images: Optional images
            
        Returns:
            Dictionary containing embeddings and similarity scores
        """
        outputs = {}
        
        if texts is not None:
            text_embeddings = self.encode_text(texts)
            outputs['text_embeddings'] = text_embeddings
        
        if images is not None:
            image_embeddings = self.encode_image(images)
            outputs['image_embeddings'] = image_embeddings
        
        # Compute cross-modal similarity if both modalities present
        if texts is not None and images is not None:
            # Scaled similarity (temperature scaling)
            similarity = torch.matmul(text_embeddings, image_embeddings.T) * self.temperature.exp()
            outputs['similarity_matrix'] = similarity
        
        return outputs


class MultimodalEncoder:
    """
    Advanced multimodal encoder for EHR data with clinical intelligence.
    
    Features:
    - Clinical concept alignment
    - Hierarchical embedding structure
    - Cross-modal attention mechanisms
    - Medical domain adaptation
    - Privacy-preserving encoding
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize clinical CLIP encoder
        self.clinical_clip = ClinicalCLIPEncoder(config.get('encoder', {}))
        self.clinical_clip.to(self.device)
        
        # Initialize clinical concept aligner
        if config.get('clinical_knowledge', {}).get('ontologies', {}).get('enable', True):
            self.concept_aligner = ClinicalConceptAligner(
                config.get('clinical_knowledge', {})
            )
        else:
            self.concept_aligner = None
        
        # Initialize hierarchical embedder
        if config.get('hierarchical', {}).get('enable', True):
            self.hierarchical_embedder = HierarchicalEmbedder(
                config.get('hierarchical', {})
            )
        else:
            self.hierarchical_embedder = None
        
        # Embedding cache for efficiency
        self.embedding_cache = {}
        self.cache_enabled = config.get('cache_embeddings', True)
        
    def encode_clinical_text(self, text: str, 
                           extract_concepts: bool = True) -> MultimodalEmbedding:
        """
        Encode clinical text with concept extraction and validation.
        
        Args:
            text: Clinical text to encode
            extract_concepts: Whether to extract clinical concepts
            
        Returns:
            MultimodalEmbedding object
        """
        # Validate clinical text
        if not validate_clinical_text(text):
            self.logger.warning("Clinical text validation failed")
        
        # Check cache
        cache_key = f"text_{hash(text)}"
        if self.cache_enabled and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Encode text
        with torch.no_grad():
            text_embedding = self.clinical_clip.encode_text([text])
        
        # Extract clinical concepts
        clinical_concepts = []
        if extract_concepts and self.concept_aligner:
            clinical_concepts = self.concept_aligner.extract_concepts(text)
        
        # Generate hierarchical embeddings
        hierarchical_embedding = None
        if self.hierarchical_embedder:
            hierarchical_embedding = self.hierarchical_embedder.encode_hierarchical(
                text, clinical_concepts
            )
        
        # Calculate confidence scores
        confidence_scores = self._calculate_text_confidence(text, text_embedding)
        
        embedding = MultimodalEmbedding(
            text_embedding=text_embedding,
            unified_embedding=text_embedding,
            hierarchical_embedding=hierarchical_embedding,
            clinical_concepts=clinical_concepts,
            confidence_scores=confidence_scores,
            modality_weights={'text': 1.0}
        )
        
        # Cache result
        if self.cache_enabled:
            self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    def encode_medical_image(self, image: Union[torch.Tensor, np.ndarray], 
                           metadata: Optional[Dict[str, Any]] = None) -> MultimodalEmbedding:
        """
        Encode medical image with clinical context.
        
        Args:
            image: Medical image to encode
            metadata: Optional image metadata (modality, body part, etc.)
            
        Returns:
            MultimodalEmbedding object
        """
        # Convert numpy array to tensor if needed
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        
        # Ensure proper dimensions and device
        if len(image.shape) == 3 and image.shape[0] not in [1, 3]:
            image = image.permute(2, 0, 1)  # HWC to CHW
        
        if len(image.shape) == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)  # Grayscale to RGB
        
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Check cache
        cache_key = f"image_{hash(image.cpu().numpy().tobytes())}"
        if self.cache_enabled and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Encode image
        with torch.no_grad():
            image_embedding = self.clinical_clip.encode_image(image)
        
        # Extract clinical concepts from metadata
        clinical_concepts = []
        if metadata and self.concept_aligner:
            metadata_text = self._metadata_to_text(metadata)
            clinical_concepts = self.concept_aligner.extract_concepts(metadata_text)
        
        # Generate hierarchical embeddings
        hierarchical_embedding = None
        if self.hierarchical_embedder and metadata:
            hierarchical_embedding = self.hierarchical_embedder.encode_hierarchical(
                self._metadata_to_text(metadata), clinical_concepts
            )
        
        # Calculate confidence scores
        confidence_scores = self._calculate_image_confidence(image, image_embedding)
        
        embedding = MultimodalEmbedding(
            image_embedding=image_embedding,
            unified_embedding=image_embedding,
            hierarchical_embedding=hierarchical_embedding,
            clinical_concepts=clinical_concepts,
            confidence_scores=confidence_scores,
            modality_weights={'image': 1.0}
        )
        
        # Cache result
        if self.cache_enabled:
            self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    def encode_multimodal(self, text: Optional[str] = None, 
                         image: Optional[Union[torch.Tensor, np.ndarray]] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> MultimodalEmbedding:
        """
        Encode multimodal input (text + image) with cross-modal alignment.
        
        Args:
            text: Optional clinical text
            image: Optional medical image
            metadata: Optional metadata
            
        Returns:
            MultimodalEmbedding object with unified representation
        """
        if text is None and image is None:
            raise ValueError("At least one modality (text or image) must be provided")
        
        # Encode individual modalities
        text_embedding = None
        image_embedding = None
        all_concepts = []
        
        if text is not None:
            text_result = self.encode_clinical_text(text)
            text_embedding = text_result.text_embedding
            if text_result.clinical_concepts:
                all_concepts.extend(text_result.clinical_concepts)
        
        if image is not None:
            image_result = self.encode_medical_image(image, metadata)
            image_embedding = image_result.image_embedding
            if image_result.clinical_concepts:
                all_concepts.extend(image_result.clinical_concepts)
        
        # Create unified embedding
        unified_embedding = self._create_unified_embedding(
            text_embedding, image_embedding
        )
        
        # Generate hierarchical embeddings
        hierarchical_embedding = None
        if self.hierarchical_embedder:
            combined_text = text or ""
            if metadata:
                combined_text += " " + self._metadata_to_text(metadata)
            
            hierarchical_embedding = self.hierarchical_embedder.encode_hierarchical(
                combined_text, all_concepts
            )
        
        # Calculate modality weights
        modality_weights = self._calculate_modality_weights(
            text_embedding, image_embedding
        )
        
        # Calculate confidence scores
        confidence_scores = self._calculate_multimodal_confidence(
            text_embedding, image_embedding, unified_embedding
        )
        
        return MultimodalEmbedding(
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            unified_embedding=unified_embedding,
            hierarchical_embedding=hierarchical_embedding,
            clinical_concepts=list(set(all_concepts)),  # Remove duplicates
            confidence_scores=confidence_scores,
            modality_weights=modality_weights
        )
    
    def _create_unified_embedding(self, text_embedding: Optional[torch.Tensor],
                                image_embedding: Optional[torch.Tensor]) -> torch.Tensor:
        """Create unified embedding from text and image embeddings."""
        if text_embedding is not None and image_embedding is not None:
            # Weighted combination with learned attention
            embeddings = torch.stack([text_embedding.squeeze(), image_embedding.squeeze()])
            
            # Simple weighted average (could be replaced with learned attention)
            weights = F.softmax(torch.ones(2), dim=0)
            unified = torch.sum(embeddings * weights.unsqueeze(-1), dim=0)
            
            return unified.unsqueeze(0)
        
        elif text_embedding is not None:
            return text_embedding
        
        elif image_embedding is not None:
            return image_embedding
        
        else:
            raise ValueError("At least one embedding must be provided")
    
    def _calculate_modality_weights(self, text_embedding: Optional[torch.Tensor],
                                  image_embedding: Optional[torch.Tensor]) -> Dict[str, float]:
        """Calculate adaptive weights for different modalities."""
        weights = {}
        
        if text_embedding is not None and image_embedding is not None:
            # Calculate weights based on embedding magnitudes and uncertainty
            text_magnitude = torch.norm(text_embedding).item()
            image_magnitude = torch.norm(image_embedding).item()
            
            total_magnitude = text_magnitude + image_magnitude
            weights['text'] = text_magnitude / total_magnitude
            weights['image'] = image_magnitude / total_magnitude
        
        elif text_embedding is not None:
            weights['text'] = 1.0
        
        elif image_embedding is not None:
            weights['image'] = 1.0
        
        return weights
    
    def _calculate_text_confidence(self, text: str, 
                                 embedding: torch.Tensor) -> Dict[str, float]:
        """Calculate confidence scores for text encoding."""
        scores = {}
        
        # Embedding magnitude as confidence indicator
        scores['embedding_confidence'] = torch.norm(embedding).item()
        
        # Text length and complexity
        scores['text_length_score'] = min(len(text.split()) / 50.0, 1.0)
        
        # Medical terminology density (if concept aligner available)
        if self.concept_aligner:
            concepts = self.concept_aligner.extract_concepts(text)
            scores['medical_concept_density'] = len(concepts) / max(len(text.split()), 1)
        
        return scores
    
    def _calculate_image_confidence(self, image: torch.Tensor, 
                                  embedding: torch.Tensor) -> Dict[str, float]:
        """Calculate confidence scores for image encoding."""
        scores = {}
        
        # Embedding magnitude
        scores['embedding_confidence'] = torch.norm(embedding).item()
        
        # Image quality metrics
        scores['image_variance'] = torch.var(image).item()
        scores['image_mean'] = torch.mean(image).item()
        
        return scores
    
    def _calculate_multimodal_confidence(self, text_embedding: Optional[torch.Tensor],
                                       image_embedding: Optional[torch.Tensor],
                                       unified_embedding: torch.Tensor) -> Dict[str, float]:
        """Calculate confidence scores for multimodal encoding."""
        scores = {}
        
        # Unified embedding confidence
        scores['unified_confidence'] = torch.norm(unified_embedding).item()
        
        # Cross-modal consistency
        if text_embedding is not None and image_embedding is not None:
            similarity = F.cosine_similarity(
                text_embedding, image_embedding, dim=-1
            ).item()
            scores['cross_modal_consistency'] = similarity
        
        return scores
    
    def _metadata_to_text(self, metadata: Dict[str, Any]) -> str:
        """Convert image metadata to text description."""
        text_parts = []
        
        if 'modality' in metadata:
            text_parts.append(f"Modality: {metadata['modality']}")
        
        if 'body_part' in metadata:
            text_parts.append(f"Body part: {metadata['body_part']}")
        
        if 'study_description' in metadata:
            text_parts.append(f"Study: {metadata['study_description']}")
        
        if 'findings' in metadata:
            text_parts.append(f"Findings: {metadata['findings']}")
        
        return ". ".join(text_parts)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.config.get('embedding_dim', 768)
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
    
    def save_model(self, path: str):
        """Save the encoder model."""
        torch.save({
            'model_state_dict': self.clinical_clip.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path: str):
        """Load the encoder model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.clinical_clip.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']