"""Response structures for Agentic Hierarchical RAG."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .node import HierarchicalNode
from .query import Query


@dataclass
class RetrievalResponse:
    """Response from the retrieval phase.
    
    Contains retrieved nodes and metadata about the retrieval process.
    """
    
    query: Query
    retrieved_nodes: List[HierarchicalNode]
    retrieval_strategy: str
    levels_searched: List[int]
    total_nodes_examined: int
    retrieval_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_context(self) -> str:
        """Get concatenated context from retrieved nodes."""
        contexts = []
        for node in self.retrieved_nodes:
            contexts.append(node.get_text_for_retrieval())
        return "\n\n".join(contexts)
    
    def get_sources(self) -> List[Dict[str, Any]]:
        """Get source information for citations."""
        sources = []
        for node in self.retrieved_nodes:
            source = {
                "node_id": node.node_id,
                "level": node.level,
                "type": node.node_type.value,
                "relevance_score": node.relevance_score,
                "metadata": node.metadata
            }
            sources.append(source)
        return sources
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "query": self.query.to_dict(),
            "retrieved_nodes": [node.to_dict() for node in self.retrieved_nodes],
            "retrieval_strategy": self.retrieval_strategy,
            "levels_searched": self.levels_searched,
            "total_nodes_examined": self.total_nodes_examined,
            "retrieval_time_ms": self.retrieval_time_ms,
            "metadata": self.metadata
        }


@dataclass
class GenerationResponse:
    """Response from the generation phase.
    
    Contains the generated answer and metadata about the generation process.
    """
    
    query: Query
    answer: str
    retrieval_response: RetrievalResponse
    model_name: str
    generation_time_ms: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    confidence_score: float = 0.0
    citations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def has_citations(self) -> bool:
        """Check if response includes citations."""
        return len(self.citations) > 0
    
    def get_total_time_ms(self) -> float:
        """Get total time for retrieval and generation."""
        return self.retrieval_response.retrieval_time_ms + self.generation_time_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "query": self.query.to_dict(),
            "answer": self.answer,
            "retrieval_response": self.retrieval_response.to_dict(),
            "model_name": self.model_name,
            "generation_time_ms": self.generation_time_ms,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "confidence_score": self.confidence_score,
            "citations": self.citations,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CritiqueResponse:
    """Response from the self-critique phase.
    
    Contains evaluation of retrieved context and generated answer.
    """
    
    retrieval_quality: str  # "sufficient", "insufficient", "partial"
    answer_quality: str  # "supported", "unsupported", "partial"
    missing_information: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    
    def requires_refinement(self) -> bool:
        """Check if refinement is needed."""
        return self.retrieval_quality != "sufficient" or self.answer_quality != "supported"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert critique to dictionary."""
        return {
            "retrieval_quality": self.retrieval_quality,
            "answer_quality": self.answer_quality,
            "missing_information": self.missing_information,
            "suggested_actions": self.suggested_actions,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }