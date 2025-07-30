"""RAPTOR-style hierarchical indexing implementation."""

import uuid
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import logging
from tqdm import tqdm
from pathlib import Path

from ..core import HierarchicalNode, NodeType, HierarchicalTree
from .text_chunker import TextChunker, TextChunk, ChunkingStrategy
from .embedder import Embedder

logger = logging.getLogger(__name__)


class RAPTORIndexer:
    """Implements RAPTOR-style recursive abstractive processing for tree-organized retrieval."""
    
    def __init__(
        self,
        embedder: Embedder,
        summarizer: Callable[[List[str]], str],
        chunker: Optional[TextChunker] = None,
        clustering_method: str = "gmm",
        min_cluster_size: int = 3,
        max_cluster_size: int = 10,
        reduction_factor: float = 0.5,
        max_levels: int = 5,
    ):
        """Initialize the RAPTOR indexer.
        
        Args:
            embedder: Embedder instance for generating embeddings
            summarizer: Function that takes list of texts and returns summary
            chunker: Text chunker instance (creates default if None)
            clustering_method: Method for clustering ("gmm" or "kmeans")
            min_cluster_size: Minimum size for a cluster
            max_cluster_size: Maximum size for a cluster
            reduction_factor: Factor by which nodes reduce at each level
            max_levels: Maximum number of levels in the tree
        """
        self.embedder = embedder
        self.summarizer = summarizer
        self.chunker = chunker or TextChunker(
            chunk_size=512,
            chunk_overlap=50,
            strategy=ChunkingStrategy.SLIDING_WINDOW
        )
        self.clustering_method = clustering_method
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.reduction_factor = reduction_factor
        self.max_levels = max_levels
        
    def build_tree(
        self,
        documents: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> HierarchicalTree:
        """Build a hierarchical tree from documents.
        
        Args:
            documents: List of documents with 'content' and optional 'metadata'
            show_progress: Whether to show progress bars
            
        Returns:
            HierarchicalTree instance
        """
        logger.info(f"Building RAPTOR tree from {len(documents)} documents")
        
        # Initialize tree
        tree = HierarchicalTree()
        
        # Step 1: Create leaf nodes from documents
        logger.info("Creating leaf nodes...")
        leaf_nodes = self._create_leaf_nodes(documents, show_progress)
        
        # Add leaf nodes to tree
        for node in leaf_nodes:
            tree.add_node(node)
            
        logger.info(f"Created {len(leaf_nodes)} leaf nodes")
        
        # Step 2: Recursively build higher levels
        current_level_nodes = leaf_nodes
        current_level = 0
        
        while len(current_level_nodes) > 1 and current_level < self.max_levels - 1:
            logger.info(f"Building level {current_level + 1}...")
            
            # Cluster and summarize current level
            next_level_nodes = self._build_next_level(
                current_level_nodes,
                current_level + 1,
                tree,
                show_progress
            )
            
            if not next_level_nodes:
                break
                
            # Add new nodes to tree
            for node in next_level_nodes:
                tree.add_node(node)
                
            logger.info(f"Created {len(next_level_nodes)} nodes at level {current_level + 1}")
            
            current_level_nodes = next_level_nodes
            current_level += 1
            
        # Step 3: Create root node if we have multiple top-level nodes
        if len(current_level_nodes) > 1:
            root_node = self._create_root_node(current_level_nodes, tree)
            tree.add_node(root_node)
            logger.info("Created root node")
            
        logger.info(f"Tree construction complete. Total nodes: {len(tree.nodes)}")
        logger.info(f"Tree statistics: {tree.get_statistics()}")
        
        return tree
    
    def _create_leaf_nodes(
        self,
        documents: List[Dict[str, Any]],
        show_progress: bool
    ) -> List[HierarchicalNode]:
        """Create leaf nodes from documents.
        
        Args:
            documents: List of documents
            show_progress: Whether to show progress
            
        Returns:
            List of leaf nodes
        """
        leaf_nodes = []
        
        for doc_idx, doc in enumerate(tqdm(documents, desc="Processing documents", disable=not show_progress)):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            metadata["source_doc_idx"] = doc_idx
            
            # Chunk the document
            chunks = self.chunker.chunk_text(content, metadata)
            
            # Create nodes for each chunk
            for chunk in chunks:
                node_id = str(uuid.uuid4())
                node = HierarchicalNode(
                    node_id=node_id,
                    node_type=NodeType.LEAF,
                    content=chunk.content,
                    level=0,
                    metadata={
                        **chunk.metadata,
                        "chunk_id": chunk.chunk_id,
                        "start_idx": chunk.start_idx,
                        "end_idx": chunk.end_idx,
                    }
                )
                
                # Generate embedding
                node.embedding = self.embedder.embed_texts(chunk.content)
                
                leaf_nodes.append(node)
                
        return leaf_nodes
    
    def _build_next_level(
        self,
        nodes: List[HierarchicalNode],
        level: int,
        tree: HierarchicalTree,
        show_progress: bool
    ) -> List[HierarchicalNode]:
        """Build the next level of the tree through clustering and summarization.
        
        Args:
            nodes: Nodes from current level
            level: Level number for new nodes
            tree: Tree being built
            show_progress: Whether to show progress
            
        Returns:
            List of nodes for the next level
        """
        # Get embeddings for clustering
        embeddings = np.array([node.embedding for node in nodes])
        
        # Determine number of clusters
        n_clusters = self._determine_n_clusters(len(nodes))
        
        if n_clusters <= 1:
            return []
            
        # Perform clustering
        clusters = self._cluster_nodes(embeddings, n_clusters)
        
        # Create new nodes for each cluster
        new_nodes = []
        cluster_assignments = {}
        
        # Group nodes by cluster
        for idx, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_assignments:
                cluster_assignments[cluster_id] = []
            cluster_assignments[cluster_id].append(nodes[idx])
            
        # Create summary nodes for each cluster
        for cluster_id, cluster_nodes in tqdm(
            cluster_assignments.items(),
            desc=f"Creating level {level} nodes",
            disable=not show_progress
        ):
            if len(cluster_nodes) < self.min_cluster_size:
                continue
                
            # Create new node
            new_node = self._create_summary_node(
                cluster_nodes,
                level,
                tree
            )
            new_nodes.append(new_node)
            
        return new_nodes
    
    def _cluster_nodes(
        self,
        embeddings: np.ndarray,
        n_clusters: int
    ) -> np.ndarray:
        """Cluster nodes based on their embeddings.
        
        Args:
            embeddings: Node embeddings
            n_clusters: Number of clusters
            
        Returns:
            Cluster assignments
        """
        if self.clustering_method == "gmm":
            model = GaussianMixture(
                n_components=n_clusters,
                random_state=42,
                n_init=3
            )
        elif self.clustering_method == "kmeans":
            model = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
            
        clusters = model.fit_predict(embeddings)
        return clusters
    
    def _determine_n_clusters(self, n_nodes: int) -> int:
        """Determine optimal number of clusters based on reduction factor.
        
        Args:
            n_nodes: Number of nodes to cluster
            
        Returns:
            Number of clusters
        """
        # Target number of nodes for next level
        target_nodes = max(1, int(n_nodes * self.reduction_factor))
        
        # Average cluster size
        avg_cluster_size = (self.min_cluster_size + self.max_cluster_size) / 2
        
        # Calculate number of clusters
        n_clusters = max(1, int(n_nodes / avg_cluster_size))
        
        # Ensure we're actually reducing the number of nodes
        n_clusters = min(n_clusters, target_nodes)
        
        return n_clusters
    
    def _create_summary_node(
        self,
        cluster_nodes: List[HierarchicalNode],
        level: int,
        tree: HierarchicalTree
    ) -> HierarchicalNode:
        """Create a summary node for a cluster of nodes.
        
        Args:
            cluster_nodes: Nodes in the cluster
            level: Level for the new node
            tree: Tree being built
            
        Returns:
            New summary node
        """
        # Collect texts for summarization
        texts = []
        for node in cluster_nodes:
            if node.summary:
                texts.append(node.summary)
            else:
                texts.append(node.content)
                
        # Generate summary
        summary = self.summarizer(texts)
        
        # Create new node
        node_id = str(uuid.uuid4())
        new_node = HierarchicalNode(
            node_id=node_id,
            node_type=NodeType.INTERMEDIATE,
            content=summary,
            summary=summary,
            level=level,
            metadata={
                "n_children": len(cluster_nodes),
                "child_levels": list(set(n.level for n in cluster_nodes)),
            }
        )
        
        # Generate embedding for summary
        new_node.embedding = self.embedder.embed_texts(summary)
        
        # Set up parent-child relationships
        for child_node in cluster_nodes:
            child_node.parent_id = node_id
            new_node.add_child(child_node.node_id)
            
        return new_node
    
    def _create_root_node(
        self,
        top_nodes: List[HierarchicalNode],
        tree: HierarchicalTree
    ) -> HierarchicalNode:
        """Create the root node of the tree.
        
        Args:
            top_nodes: Top-level nodes
            tree: Tree being built
            
        Returns:
            Root node
        """
        # Collect texts for final summary
        texts = []
        for node in top_nodes:
            if node.summary:
                texts.append(node.summary)
            else:
                texts.append(node.content)
                
        # Generate root summary
        summary = self.summarizer(texts)
        
        # Create root node
        node_id = str(uuid.uuid4())
        root_node = HierarchicalNode(
            node_id=node_id,
            node_type=NodeType.ROOT,
            content=summary,
            summary=summary,
            level=max(n.level for n in top_nodes) + 1,
            metadata={
                "n_children": len(top_nodes),
                "total_nodes": len(tree.nodes),
            }
        )
        
        # Generate embedding
        root_node.embedding = self.embedder.embed_texts(summary)
        
        # Set up parent-child relationships
        for child_node in top_nodes:
            child_node.parent_id = node_id
            root_node.add_child(child_node.node_id)
            
        return root_node
    
    def update_tree(
        self,
        tree: HierarchicalTree,
        new_documents: List[Dict[str, Any]],
        rebalance: bool = True
    ) -> HierarchicalTree:
        """Update an existing tree with new documents.
        
        Args:
            tree: Existing tree to update
            new_documents: New documents to add
            rebalance: Whether to rebalance affected parts of the tree
            
        Returns:
            Updated tree
        """
        # This is a simplified version - in practice, you'd want more
        # sophisticated update strategies
        logger.info(f"Updating tree with {len(new_documents)} new documents")
        
        # Create new leaf nodes
        new_leaf_nodes = self._create_leaf_nodes(new_documents, show_progress=True)
        
        # Add to tree
        for node in new_leaf_nodes:
            tree.add_node(node)
            
        if rebalance:
            # In a full implementation, you would rebalance affected
            # parts of the tree by re-clustering and re-summarizing
            logger.warning("Tree rebalancing not yet implemented")
            
        return tree