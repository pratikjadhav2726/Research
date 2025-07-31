"""Main interface for Agentic Hierarchical RAG."""

from typing import List, Dict, Any, Optional, Callable, Union
from pathlib import Path
import logging

from src import (
    HierarchicalTree,
    RAPTORIndexer,
    Embedder,
    TextChunker,
    ChunkingStrategy,
    HierarchicalRetriever,
    AgenticController,
    QueryAnalyzer,
    GenerationResponse,
)

logger = logging.getLogger(__name__)


class AgenticHierarchicalRAG:
    """Main interface for Agentic Hierarchical RAG system.
    
    This class provides a simple interface to:
    1. Build a hierarchical tree from documents
    2. Query the tree using agentic navigation
    3. Evaluate system performance
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        summarizer: Optional[Callable[[List[str]], str]] = None,
        generator: Optional[Callable[[str, str], Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the AH-RAG system.
        
        Args:
            embedding_model: Name of the embedding model to use
            summarizer: Function to summarize text chunks
            generator: Function to generate answers
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize embedder
        self.embedder = Embedder(
            model_name=embedding_model,
            batch_size=self.config.get("embedding_batch_size", 32),
            normalize_embeddings=True,
        )
        
        # Set up summarizer
        if summarizer is None:
            self.summarizer = self._default_summarizer
        else:
            self.summarizer = summarizer
        
        # Set up generator
        if generator is None:
            self.generator = self._default_generator
        else:
            self.generator = generator
        
        # Initialize components
        self.tree: Optional[HierarchicalTree] = None
        self.indexer: Optional[RAPTORIndexer] = None
        self.retriever: Optional[HierarchicalRetriever] = None
        self.controller: Optional[AgenticController] = None
        
        logger.info("Initialized Agentic Hierarchical RAG system")
    
    def index_documents(
        self,
        documents: Union[List[str], List[Dict[str, Any]]],
        save_tree: Optional[Path] = None,
        show_progress: bool = True,
    ) -> HierarchicalTree:
        """Index documents and build hierarchical tree.
        
        Args:
            documents: List of documents (strings or dicts with 'content' key)
            save_tree: Optional path to save the tree
            show_progress: Whether to show progress bars
            
        Returns:
            Built hierarchical tree
        """
        # Prepare documents
        if isinstance(documents[0], str):
            documents = [{"content": doc} for doc in documents]
        
        # Initialize text chunker
        chunker = TextChunker(
            chunk_size=self.config.get("chunk_size", 512),
            chunk_overlap=self.config.get("chunk_overlap", 50),
            strategy=ChunkingStrategy(self.config.get("chunking_strategy", "sliding_window")),
        )
        
        # Initialize RAPTOR indexer
        self.indexer = RAPTORIndexer(
            embedder=self.embedder,
            summarizer=self.summarizer,
            chunker=chunker,
            clustering_method=self.config.get("clustering_method", "gmm"),
            min_cluster_size=self.config.get("min_cluster_size", 3),
            max_cluster_size=self.config.get("max_cluster_size", 10),
            reduction_factor=self.config.get("reduction_factor", 0.5),
            max_levels=self.config.get("max_levels", 5),
        )
        
        # Build tree
        logger.info(f"Building hierarchical tree from {len(documents)} documents")
        self.tree = self.indexer.build_tree(documents, show_progress=show_progress)
        
        # Save tree if requested
        if save_tree:
            self.tree.save(save_tree)
            logger.info(f"Saved tree to {save_tree}")
        
        # Initialize retriever
        self.retriever = HierarchicalRetriever(self.tree, self.embedder)
        
        # Initialize agentic controller
        self.controller = AgenticController(
            tree=self.tree,
            retriever=self.retriever,
            generator=self.generator,
            query_analyzer=QueryAnalyzer(),
            max_iterations=self.config.get("max_iterations", 5),
            confidence_threshold=self.config.get("confidence_threshold", 0.7),
        )
        
        return self.tree
    
    def load_tree(self, path: Path) -> HierarchicalTree:
        """Load a pre-built tree from disk.
        
        Args:
            path: Path to saved tree
            
        Returns:
            Loaded tree
        """
        logger.info(f"Loading tree from {path}")
        self.tree = HierarchicalTree.load(path)
        
        # Initialize retriever and controller
        self.retriever = HierarchicalRetriever(self.tree, self.embedder)
        self.controller = AgenticController(
            tree=self.tree,
            retriever=self.retriever,
            generator=self.generator,
            query_analyzer=QueryAnalyzer(),
            max_iterations=self.config.get("max_iterations", 5),
            confidence_threshold=self.config.get("confidence_threshold", 0.7),
        )
        
        return self.tree
    
    def query(self, query_text: str) -> GenerationResponse:
        """Query the system.
        
        Args:
            query_text: User query
            
        Returns:
            Generation response with answer and metadata
        """
        if self.controller is None:
            raise ValueError("System not initialized. Call index_documents() or load_tree() first.")
        
        return self.controller.process_query(query_text)
    
    def get_tree_stats(self) -> Dict[str, Any]:
        """Get statistics about the hierarchical tree.
        
        Returns:
            Tree statistics
        """
        if self.tree is None:
            return {"error": "No tree loaded"}
        
        stats = self.tree.get_statistics()
        stats["retriever_stats"] = self.retriever.get_retrieval_stats() if self.retriever else {}
        
        return stats
    
    def _default_summarizer(self, texts: List[str]) -> str:
        """Default summarizer using simple concatenation.
        
        In practice, this should use an LLM to generate summaries.
        
        Args:
            texts: List of texts to summarize
            
        Returns:
            Summary text
        """
        # Simple implementation - in practice, use an LLM
        combined = " ".join(texts)
        
        # Truncate to reasonable length
        max_length = 500
        if len(combined) > max_length:
            combined = combined[:max_length] + "..."
        
        return f"Summary of {len(texts)} texts: {combined}"
    
    def _default_generator(self, prompt: str, query: str) -> Dict[str, Any]:
        """Default generator using simple template.
        
        In practice, this should use an LLM to generate answers.
        
        Args:
            prompt: Full prompt with context
            query: Original query
            
        Returns:
            Generation result dictionary
        """
        # Simple implementation - in practice, use an LLM
        return {
            "answer": f"Based on the provided context, here is an answer to '{query}': [Generated answer would go here]",
            "model": "default",
            "total_tokens": 100,
            "prompt_tokens": 80,
            "completion_tokens": 20,
        }
    
    @classmethod
    def from_config(cls, config_path: Path) -> "AgenticHierarchicalRAG":
        """Create instance from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configured AH-RAG instance
        """
        import yaml
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        return cls(
            embedding_model=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            config=config
        )