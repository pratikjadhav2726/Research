"""Agentic Hierarchical RAG implementation."""

from .core import (
    HierarchicalNode,
    NodeType,
    HierarchicalTree,
    Query,
    QueryType,
    RetrievalResponse,
    GenerationResponse,
)

from .indexing import (
    RAPTORIndexer,
    TextChunker,
    ChunkingStrategy,
    Embedder,
)

from .retrieval import (
    HierarchicalRetriever,
    VectorStore,
)

from .agents import (
    AgenticController,
    QueryAnalyzer,
    ReflectionToken,
    ReflectionTokens,
)

from .evaluation import (
    Evaluator,
    RAGMetrics,
    Benchmark,
    BenchmarkDataset,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "HierarchicalNode",
    "NodeType", 
    "HierarchicalTree",
    "Query",
    "QueryType",
    "RetrievalResponse",
    "GenerationResponse",
    
    # Indexing
    "RAPTORIndexer",
    "TextChunker",
    "ChunkingStrategy",
    "Embedder",
    
    # Retrieval
    "HierarchicalRetriever",
    "VectorStore",
    
    # Agents
    "AgenticController",
    "QueryAnalyzer",
    "ReflectionToken",
    "ReflectionTokens",
    
    # Evaluation
    "Evaluator",
    "RAGMetrics",
    "Benchmark",
    "BenchmarkDataset",
]