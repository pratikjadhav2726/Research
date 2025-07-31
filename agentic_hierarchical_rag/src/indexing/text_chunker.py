"""Text chunking utilities for document processing."""

from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    start_idx: int
    end_idx: int
    chunk_id: int
    metadata: Dict[str, Any]
    
    def __len__(self) -> int:
        """Get length of chunk content."""
        return len(self.content)
    
    def word_count(self) -> int:
        """Get word count of chunk."""
        return len(self.content.split())


class TextChunker:
    """Handles text chunking with various strategies."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: ChunkingStrategy = ChunkingStrategy.SLIDING_WINDOW,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
    ):
        """Initialize the text chunker.
        
        Args:
            chunk_size: Target size for chunks (in characters)
            chunk_overlap: Overlap between chunks (for sliding window)
            strategy: Chunking strategy to use
            min_chunk_size: Minimum allowed chunk size
            max_chunk_size: Maximum allowed chunk size
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """Chunk text according to the configured strategy.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of text chunks
        """
        if metadata is None:
            metadata = {}
            
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._fixed_size_chunking(text, metadata)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunking(text, metadata)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._sentence_chunking(text, metadata)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self._paragraph_chunking(text, metadata)
        elif self.strategy == ChunkingStrategy.SLIDING_WINDOW:
            return self._sliding_window_chunking(text, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
    
    def _fixed_size_chunking(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[TextChunk]:
        """Simple fixed-size chunking."""
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(text), self.chunk_size):
            chunk_content = text[i:i + self.chunk_size]
            if len(chunk_content) >= self.min_chunk_size:
                chunk = TextChunk(
                    content=chunk_content.strip(),
                    start_idx=i,
                    end_idx=min(i + self.chunk_size, len(text)),
                    chunk_id=chunk_id,
                    metadata=metadata.copy()
                )
                chunks.append(chunk)
                chunk_id += 1
                
        return chunks
    
    def _sliding_window_chunking(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[TextChunk]:
        """Sliding window chunking with overlap."""
        chunks = []
        chunk_id = 0
        stride = self.chunk_size - self.chunk_overlap
        
        for i in range(0, len(text), stride):
            chunk_content = text[i:i + self.chunk_size]
            if len(chunk_content) >= self.min_chunk_size:
                chunk = TextChunk(
                    content=chunk_content.strip(),
                    start_idx=i,
                    end_idx=min(i + self.chunk_size, len(text)),
                    chunk_id=chunk_id,
                    metadata=metadata.copy()
                )
                chunks.append(chunk)
                chunk_id += 1
                
            # Stop if we've reached the end
            if i + self.chunk_size >= len(text):
                break
                
        return chunks
    
    def _sentence_chunking(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[TextChunk]:
        """Chunk by sentences, respecting size limits."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0
        start_idx = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed max size, create chunk
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunk_content = " ".join(current_chunk)
                chunk = TextChunk(
                    content=chunk_content,
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_content),
                    chunk_id=chunk_id,
                    metadata=metadata.copy()
                )
                chunks.append(chunk)
                chunk_id += 1
                
                # Reset for next chunk
                start_idx = start_idx + len(chunk_content) + 1
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size + 1  # +1 for space
        
        # Add final chunk
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            chunk = TextChunk(
                content=chunk_content,
                start_idx=start_idx,
                end_idx=len(text),
                chunk_id=chunk_id,
                metadata=metadata.copy()
            )
            chunks.append(chunk)
            
        return chunks
    
    def _paragraph_chunking(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[TextChunk]:
        """Chunk by paragraphs."""
        # Split by double newlines or multiple spaces
        paragraphs = re.split(r'\n\n+|\r\n\r\n+', text)
        chunks = []
        chunk_id = 0
        current_position = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) >= self.min_chunk_size:
                # If paragraph is too large, split it further
                if len(paragraph) > self.max_chunk_size:
                    # Use sentence chunking for large paragraphs
                    sub_chunks = self._sentence_chunking(paragraph, metadata)
                    for sub_chunk in sub_chunks:
                        sub_chunk.chunk_id = chunk_id
                        sub_chunk.start_idx += current_position
                        sub_chunk.end_idx += current_position
                        chunks.append(sub_chunk)
                        chunk_id += 1
                else:
                    chunk = TextChunk(
                        content=paragraph,
                        start_idx=current_position,
                        end_idx=current_position + len(paragraph),
                        chunk_id=chunk_id,
                        metadata=metadata.copy()
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    
            current_position += len(paragraph) + 2  # Account for paragraph separator
            
        return chunks
    
    def _semantic_chunking(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[TextChunk]:
        """Semantic chunking based on topic boundaries.
        
        This is a simplified version. In practice, you might use
        more sophisticated methods like TextTiling or topic modeling.
        """
        # For now, fall back to paragraph chunking with overlap
        paragraphs = re.split(r'\n\n+|\r\n\r\n+', text)
        chunks = []
        chunk_id = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if len(paragraph) < self.min_chunk_size:
                continue
                
            # Add context from previous paragraph if available
            if i > 0 and self.chunk_overlap > 0:
                prev_sentences = sent_tokenize(paragraphs[i-1])
                if prev_sentences:
                    # Add last sentence from previous paragraph
                    paragraph = prev_sentences[-1] + " " + paragraph
            
            # Add context from next paragraph if available
            if i < len(paragraphs) - 1 and self.chunk_overlap > 0:
                next_sentences = sent_tokenize(paragraphs[i+1])
                if next_sentences:
                    # Add first sentence from next paragraph
                    paragraph = paragraph + " " + next_sentences[0]
            
            chunk = TextChunk(
                content=paragraph,
                start_idx=0,  # Would need proper tracking in production
                end_idx=len(paragraph),
                chunk_id=chunk_id,
                metadata=metadata.copy()
            )
            chunks.append(chunk)
            chunk_id += 1
            
        return chunks
    
    def merge_small_chunks(
        self,
        chunks: List[TextChunk],
        min_size: Optional[int] = None
    ) -> List[TextChunk]:
        """Merge chunks that are too small.
        
        Args:
            chunks: List of chunks to process
            min_size: Minimum size (uses self.min_chunk_size if None)
            
        Returns:
            List of merged chunks
        """
        if min_size is None:
            min_size = self.min_chunk_size
            
        merged_chunks = []
        current_merged = None
        
        for chunk in chunks:
            if len(chunk) < min_size:
                if current_merged is None:
                    current_merged = chunk
                else:
                    # Merge with current
                    current_merged = TextChunk(
                        content=current_merged.content + " " + chunk.content,
                        start_idx=current_merged.start_idx,
                        end_idx=chunk.end_idx,
                        chunk_id=current_merged.chunk_id,
                        metadata=current_merged.metadata
                    )
            else:
                if current_merged is not None:
                    merged_chunks.append(current_merged)
                    current_merged = None
                merged_chunks.append(chunk)
        
        # Add any remaining merged chunk
        if current_merged is not None:
            merged_chunks.append(current_merged)
            
        return merged_chunks