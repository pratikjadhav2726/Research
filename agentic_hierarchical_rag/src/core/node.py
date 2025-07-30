"""Hierarchical node structure for RAPTOR-style tree."""

from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime


class NodeType(Enum):
    """Types of nodes in the hierarchical tree."""
    LEAF = "leaf"  # Original text chunks
    INTERMEDIATE = "intermediate"  # Clustered summaries
    ROOT = "root"  # Top-level summary


@dataclass
class HierarchicalNode:
    """Represents a node in the hierarchical RAG tree.
    
    Attributes:
        node_id: Unique identifier for the node
        node_type: Type of node (leaf, intermediate, root)
        content: Text content of the node
        embedding: Vector embedding of the content
        level: Level in the hierarchy (0 for leaves, increases upward)
        parent_id: ID of parent node (None for root)
        children_ids: List of child node IDs
        metadata: Additional metadata (source, timestamp, etc.)
        summary: Summary text (for intermediate/root nodes)
        relevance_score: Relevance score from last retrieval
    """
    
    node_id: str
    node_type: NodeType
    content: str
    embedding: Optional[np.ndarray] = None
    level: int = 0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None
    relevance_score: float = 0.0
    
    def __post_init__(self):
        """Validate node after initialization."""
        if self.node_type == NodeType.LEAF and self.level != 0:
            raise ValueError("Leaf nodes must have level 0")
        
        if self.node_type == NodeType.ROOT and self.parent_id is not None:
            raise ValueError("Root node cannot have a parent")
        
        # Add creation timestamp if not present
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().isoformat()
    
    def add_child(self, child_id: str) -> None:
        """Add a child node ID."""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
    
    def remove_child(self, child_id: str) -> None:
        """Remove a child node ID."""
        if child_id in self.children_ids:
            self.children_ids.remove(child_id)
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.node_type == NodeType.LEAF
    
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.node_type == NodeType.ROOT
    
    def get_text_for_retrieval(self) -> str:
        """Get the text content for retrieval operations."""
        if self.summary and self.node_type != NodeType.LEAF:
            return self.summary
        return self.content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "content": self.content,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "level": self.level,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata,
            "summary": self.summary,
            "relevance_score": self.relevance_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HierarchicalNode":
        """Create node from dictionary."""
        embedding = None
        if data.get("embedding") is not None:
            embedding = np.array(data["embedding"])
        
        return cls(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            content=data["content"],
            embedding=embedding,
            level=data["level"],
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            metadata=data.get("metadata", {}),
            summary=data.get("summary"),
            relevance_score=data.get("relevance_score", 0.0)
        )