"""Query structures for Agentic Hierarchical RAG."""

from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime


class QueryType(Enum):
    """Types of queries based on abstraction level."""
    FACTUAL = "factual"  # Specific facts, dates, numbers
    THEMATIC = "thematic"  # Broad themes, summaries
    COMPARATIVE = "comparative"  # Comparisons across topics
    ANALYTICAL = "analytical"  # Deep analysis, reasoning
    EXPLORATORY = "exploratory"  # Open-ended exploration


@dataclass
class Query:
    """Represents a user query with metadata for processing.
    
    Attributes:
        query_id: Unique identifier for the query
        text: The original query text
        query_type: Classified type of query
        embedding: Vector embedding of the query
        metadata: Additional metadata
        timestamp: When the query was created
        abstraction_level: Suggested abstraction level (0=leaf, higher=more abstract)
        confidence_scores: Confidence scores for different aspects
    """
    
    query_id: str
    text: str
    query_type: Optional[QueryType] = None
    embedding: Optional[Any] = None  # numpy array
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    abstraction_level: Optional[int] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize query attributes."""
        # Add default confidence scores if not provided
        if "query_type_confidence" not in self.confidence_scores:
            self.confidence_scores["query_type_confidence"] = 0.0
        if "abstraction_confidence" not in self.confidence_scores:
            self.confidence_scores["abstraction_confidence"] = 0.0
    
    def is_factual(self) -> bool:
        """Check if query is factual type."""
        return self.query_type == QueryType.FACTUAL
    
    def is_thematic(self) -> bool:
        """Check if query is thematic type."""
        return self.query_type == QueryType.THEMATIC
    
    def requires_high_abstraction(self) -> bool:
        """Check if query requires high-level abstraction."""
        return self.query_type in [QueryType.THEMATIC, QueryType.COMPARATIVE]
    
    def requires_detailed_search(self) -> bool:
        """Check if query requires detailed leaf-level search."""
        return self.query_type == QueryType.FACTUAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary."""
        return {
            "query_id": self.query_id,
            "text": self.text,
            "query_type": self.query_type.value if self.query_type else None,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "abstraction_level": self.abstraction_level,
            "confidence_scores": self.confidence_scores
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Query":
        """Create query from dictionary."""
        import numpy as np
        
        embedding = None
        if data.get("embedding") is not None:
            embedding = np.array(data["embedding"])
        
        query_type = None
        if data.get("query_type"):
            query_type = QueryType(data["query_type"])
        
        timestamp = datetime.fromisoformat(data["timestamp"])
        
        return cls(
            query_id=data["query_id"],
            text=data["text"],
            query_type=query_type,
            embedding=embedding,
            metadata=data.get("metadata", {}),
            timestamp=timestamp,
            abstraction_level=data.get("abstraction_level"),
            confidence_scores=data.get("confidence_scores", {})
        )


@dataclass
class QueryAnalysis:
    """Analysis results for a query.
    
    Contains the agent's analysis of the query including
    determined type, abstraction level, and search strategy.
    """
    
    query: Query
    suggested_levels: List[int]  # Suggested tree levels to search
    search_strategy: str  # e.g., "top_down", "bottom_up", "targeted"
    reasoning: str  # Agent's reasoning for the strategy
    requires_retrieval: bool = True
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary."""
        return {
            "query": self.query.to_dict(),
            "suggested_levels": self.suggested_levels,
            "search_strategy": self.search_strategy,
            "reasoning": self.reasoning,
            "requires_retrieval": self.requires_retrieval,
            "confidence": self.confidence
        }