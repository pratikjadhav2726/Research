"""Hierarchical retriever for navigating the RAPTOR tree."""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from collections import defaultdict

from ..core import Query, HierarchicalTree, HierarchicalNode, NodeType
from ..indexing import Embedder
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class HierarchicalRetriever:
    """Retrieves relevant nodes from a hierarchical tree structure.
    
    This retriever supports multiple retrieval strategies:
    - Level-specific retrieval
    - Cross-level retrieval
    - Collapsed tree retrieval
    - Path-based retrieval
    """
    
    def __init__(
        self,
        tree: HierarchicalTree,
        embedder: Embedder,
        vector_stores: Optional[Dict[int, VectorStore]] = None,
    ):
        """Initialize the hierarchical retriever.
        
        Args:
            tree: Hierarchical tree to search
            embedder: Embedder for query encoding
            vector_stores: Optional pre-built vector stores per level
        """
        self.tree = tree
        self.embedder = embedder
        
        # Build or use provided vector stores
        if vector_stores:
            self.vector_stores = vector_stores
        else:
            self.vector_stores = self._build_vector_stores()
    
    def _build_vector_stores(self) -> Dict[int, VectorStore]:
        """Build vector stores for each level of the tree.
        
        Returns:
            Dictionary mapping level to vector store
        """
        logger.info("Building vector stores for each level...")
        vector_stores = {}
        
        # Get embedding dimension from first node
        sample_node = next(iter(self.tree.nodes.values()))
        if sample_node.embedding is None:
            raise ValueError("Nodes must have embeddings before building vector stores")
        
        embedding_dim = len(sample_node.embedding)
        
        # Build store for each level
        for level in sorted(self.tree.levels.keys()):
            nodes = self.tree.get_nodes_at_level(level)
            if not nodes:
                continue
                
            # Create vector store
            store = VectorStore(
                dimension=embedding_dim,
                index_type="flat",  # Use flat for small datasets
                metric="cosine"
            )
            
            # Add embeddings and metadata
            embeddings = []
            metadata = []
            
            for node in nodes:
                if node.embedding is not None:
                    embeddings.append(node.embedding)
                    metadata.append({
                        "node_id": node.node_id,
                        "level": node.level,
                        "type": node.node_type.value,
                        "parent_id": node.parent_id,
                        "children_ids": node.children_ids,
                        **node.metadata
                    })
            
            if embeddings:
                embeddings_array = np.vstack(embeddings)
                store.add(embeddings_array, metadata)
                vector_stores[level] = store
                logger.info(f"Built vector store for level {level} with {len(embeddings)} nodes")
        
        return vector_stores
    
    def retrieve_at_level(
        self,
        query: Query,
        level: int,
        top_k: int = 5,
        filter_fn: Optional[callable] = None
    ) -> List[HierarchicalNode]:
        """Retrieve nodes at a specific level.
        
        Args:
            query: Query object
            level: Level to search
            top_k: Number of results
            filter_fn: Optional filter function
            
        Returns:
            List of relevant nodes
        """
        if level not in self.vector_stores:
            logger.warning(f"No vector store for level {level}")
            return []
        
        # Get query embedding
        if query.embedding is None:
            query.embedding = self.embedder.embed_query(query.text)
        
        # Search in level-specific store
        store = self.vector_stores[level]
        similarities, indices, metadata_list = store.search(
            query.embedding,
            k=top_k,
            filter_fn=filter_fn
        )
        
        # Get nodes and update relevance scores
        nodes = []
        for i, meta_list in enumerate(metadata_list[0]):  # Single query
            if "node_id" in meta_list:
                node = self.tree.get_node(meta_list["node_id"])
                if node:
                    node.relevance_score = float(similarities[0][i])
                    nodes.append(node)
        
        return nodes
    
    def retrieve_collapsed_tree(
        self,
        query: Query,
        top_k: int = 10,
        filter_fn: Optional[callable] = None
    ) -> List[HierarchicalNode]:
        """Retrieve from all levels simultaneously (collapsed tree).
        
        Args:
            query: Query object
            top_k: Total number of results
            filter_fn: Optional filter function
            
        Returns:
            List of relevant nodes from any level
        """
        # Get query embedding
        if query.embedding is None:
            query.embedding = self.embedder.embed_query(query.text)
        
        # Search all levels and collect results
        all_results = []
        
        for level, store in self.vector_stores.items():
            similarities, indices, metadata_list = store.search(
                query.embedding,
                k=top_k,  # Get top_k from each level
                filter_fn=filter_fn
            )
            
            # Add results with level info
            for i, meta_list in enumerate(metadata_list[0]):
                if "node_id" in meta_list:
                    all_results.append({
                        "node_id": meta_list["node_id"],
                        "level": level,
                        "similarity": float(similarities[0][i]),
                        "metadata": meta_list
                    })
        
        # Sort by similarity and take top_k overall
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = all_results[:top_k]
        
        # Get nodes
        nodes = []
        for result in top_results:
            node = self.tree.get_node(result["node_id"])
            if node:
                node.relevance_score = result["similarity"]
                nodes.append(node)
        
        return nodes
    
    def retrieve_with_path(
        self,
        query: Query,
        start_level: int = 0,
        top_k_per_level: int = 3
    ) -> List[HierarchicalNode]:
        """Retrieve nodes along paths from specific level to root.
        
        Args:
            query: Query object
            start_level: Level to start from (usually 0 for leaves)
            top_k_per_level: Number of nodes per level
            
        Returns:
            List of nodes forming paths to root
        """
        # Start with best matches at start level
        start_nodes = self.retrieve_at_level(query, start_level, top_k_per_level)
        
        if not start_nodes:
            return []
        
        # Collect all nodes in paths
        path_nodes = set()
        for node in start_nodes:
            path = self.tree.get_path_to_root(node.node_id)
            path_nodes.update(path)
        
        return list(path_nodes)
    
    def retrieve_subtree(
        self,
        query: Query,
        root_level: int,
        top_k_roots: int = 2
    ) -> List[HierarchicalNode]:
        """Retrieve entire subtrees rooted at high-level nodes.
        
        Args:
            query: Query object
            root_level: Level to find root nodes
            top_k_roots: Number of root nodes
            
        Returns:
            List of all nodes in selected subtrees
        """
        # Find best matching nodes at root level
        root_nodes = self.retrieve_at_level(query, root_level, top_k_roots)
        
        if not root_nodes:
            return []
        
        # Collect all nodes in subtrees
        subtree_nodes = []
        for root in root_nodes:
            subtree = self.tree.get_subtree(root.node_id)
            subtree_nodes.extend(subtree)
        
        return subtree_nodes
    
    def retrieve_diverse(
        self,
        query: Query,
        top_k: int = 10,
        diversity_weight: float = 0.3
    ) -> List[HierarchicalNode]:
        """Retrieve diverse nodes using MMR (Maximal Marginal Relevance).
        
        Args:
            query: Query object
            top_k: Number of results
            diversity_weight: Weight for diversity (0=pure relevance, 1=pure diversity)
            
        Returns:
            List of diverse relevant nodes
        """
        # Get initial candidates (2x top_k)
        candidates = self.retrieve_collapsed_tree(query, top_k * 2)
        
        if not candidates:
            return []
        
        # Get query embedding
        if query.embedding is None:
            query.embedding = self.embedder.embed_query(query.text)
        
        # MMR selection
        selected = []
        remaining = candidates.copy()
        
        # Select first node (highest relevance)
        selected.append(remaining.pop(0))
        
        # Iteratively select diverse nodes
        while len(selected) < top_k and remaining:
            mmr_scores = []
            
            for candidate in remaining:
                # Relevance to query
                relevance = candidate.relevance_score
                
                # Maximum similarity to already selected
                max_sim = 0
                for selected_node in selected:
                    if (candidate.embedding is not None and 
                        selected_node.embedding is not None):
                        sim = self.embedder.compute_similarity(
                            candidate.embedding,
                            selected_node.embedding
                        )
                        max_sim = max(max_sim, float(sim))
                
                # MMR score
                mmr = (1 - diversity_weight) * relevance - diversity_weight * max_sim
                mmr_scores.append(mmr)
            
            # Select node with highest MMR
            best_idx = np.argmax(mmr_scores)
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def retrieve_by_type(
        self,
        query: Query,
        node_types: List[NodeType],
        top_k: int = 10
    ) -> List[HierarchicalNode]:
        """Retrieve nodes of specific types.
        
        Args:
            query: Query object
            node_types: List of node types to retrieve
            top_k: Number of results
            
        Returns:
            List of nodes of specified types
        """
        def type_filter(metadata: Dict[str, Any]) -> bool:
            return metadata.get("type") in [nt.value for nt in node_types]
        
        return self.retrieve_collapsed_tree(query, top_k, filter_fn=type_filter)
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval index.
        
        Returns:
            Dictionary with retrieval statistics
        """
        stats = {
            "levels": len(self.vector_stores),
            "nodes_per_level": {},
            "total_indexed": 0
        }
        
        for level, store in self.vector_stores.items():
            size = store.size
            stats["nodes_per_level"][level] = size
            stats["total_indexed"] += size
        
        return stats