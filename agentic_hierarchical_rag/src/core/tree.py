"""Hierarchical tree structure for managing RAPTOR-style nodes."""

from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import json
import pickle
from pathlib import Path

from .node import HierarchicalNode, NodeType


class HierarchicalTree:
    """Manages the hierarchical tree structure of nodes.
    
    This class provides methods for building, traversing, and querying
    the RAPTOR-style hierarchical tree.
    """
    
    def __init__(self):
        """Initialize an empty hierarchical tree."""
        self.nodes: Dict[str, HierarchicalNode] = {}
        self.root_id: Optional[str] = None
        self.levels: Dict[int, List[str]] = defaultdict(list)
        
    def add_node(self, node: HierarchicalNode) -> None:
        """Add a node to the tree.
        
        Args:
            node: The node to add
            
        Raises:
            ValueError: If node with same ID already exists
        """
        if node.node_id in self.nodes:
            raise ValueError(f"Node with ID {node.node_id} already exists")
        
        self.nodes[node.node_id] = node
        self.levels[node.level].append(node.node_id)
        
        if node.node_type == NodeType.ROOT:
            if self.root_id is not None:
                raise ValueError("Tree already has a root node")
            self.root_id = node.node_id
            
        # Update parent's children list
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].add_child(node.node_id)
    
    def get_node(self, node_id: str) -> Optional[HierarchicalNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_root(self) -> Optional[HierarchicalNode]:
        """Get the root node."""
        return self.nodes.get(self.root_id) if self.root_id else None
    
    def get_leaves(self) -> List[HierarchicalNode]:
        """Get all leaf nodes."""
        return [node for node in self.nodes.values() if node.is_leaf()]
    
    def get_nodes_at_level(self, level: int) -> List[HierarchicalNode]:
        """Get all nodes at a specific level."""
        node_ids = self.levels.get(level, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def get_children(self, node_id: str) -> List[HierarchicalNode]:
        """Get all children of a node."""
        node = self.get_node(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]
    
    def get_parent(self, node_id: str) -> Optional[HierarchicalNode]:
        """Get the parent of a node."""
        node = self.get_node(node_id)
        if not node or not node.parent_id:
            return None
        return self.nodes.get(node.parent_id)
    
    def get_ancestors(self, node_id: str) -> List[HierarchicalNode]:
        """Get all ancestors of a node (parent, grandparent, etc.)."""
        ancestors = []
        current = self.get_node(node_id)
        
        while current and current.parent_id:
            parent = self.get_parent(current.node_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
                
        return ancestors
    
    def get_descendants(self, node_id: str) -> List[HierarchicalNode]:
        """Get all descendants of a node (children, grandchildren, etc.)."""
        descendants = []
        to_visit = [node_id]
        
        while to_visit:
            current_id = to_visit.pop(0)
            children = self.get_children(current_id)
            descendants.extend(children)
            to_visit.extend([child.node_id for child in children])
            
        return descendants
    
    def get_subtree(self, node_id: str) -> List[HierarchicalNode]:
        """Get the entire subtree rooted at a node (including the node itself)."""
        node = self.get_node(node_id)
        if not node:
            return []
        return [node] + self.get_descendants(node_id)
    
    def get_path_to_root(self, node_id: str) -> List[HierarchicalNode]:
        """Get the path from a node to the root."""
        node = self.get_node(node_id)
        if not node:
            return []
        return [node] + self.get_ancestors(node_id)
    
    def traverse_bfs(self, start_id: Optional[str] = None) -> List[HierarchicalNode]:
        """Traverse the tree in breadth-first order.
        
        Args:
            start_id: Node to start from (defaults to root)
            
        Returns:
            List of nodes in BFS order
        """
        if start_id is None:
            start_id = self.root_id
        if not start_id or start_id not in self.nodes:
            return []
            
        visited = set()
        queue = [start_id]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
                
            visited.add(node_id)
            node = self.nodes[node_id]
            result.append(node)
            queue.extend(node.children_ids)
            
        return result
    
    def get_max_depth(self) -> int:
        """Get the maximum depth (level) of the tree."""
        return max(self.levels.keys()) if self.levels else -1
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about the tree structure."""
        stats = {
            "total_nodes": len(self.nodes),
            "max_depth": self.get_max_depth(),
            "leaf_nodes": len(self.get_leaves()),
            "nodes_per_level": {level: len(nodes) for level, nodes in self.levels.items()},
        }
        
        # Calculate average branching factor
        non_leaf_nodes = [n for n in self.nodes.values() if not n.is_leaf()]
        if non_leaf_nodes:
            total_children = sum(len(n.children_ids) for n in non_leaf_nodes)
            stats["avg_branching_factor"] = total_children / len(non_leaf_nodes)
        else:
            stats["avg_branching_factor"] = 0
            
        return stats
    
    def save(self, path: Path, format: str = "json") -> None:
        """Save the tree to disk.
        
        Args:
            path: Path to save the tree
            format: Format to use ("json" or "pickle")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            data = {
                "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
                "root_id": self.root_id,
                "levels": dict(self.levels)
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif format == "pickle":
            with open(path, "wb") as f:
                pickle.dump(self, f)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @classmethod
    def load(cls, path: Path, format: str = "json") -> "HierarchicalTree":
        """Load a tree from disk.
        
        Args:
            path: Path to load the tree from
            format: Format to use ("json" or "pickle")
            
        Returns:
            Loaded HierarchicalTree instance
        """
        path = Path(path)
        
        if format == "json":
            with open(path, "r") as f:
                data = json.load(f)
            
            tree = cls()
            tree.root_id = data["root_id"]
            tree.levels = defaultdict(list, {int(k): v for k, v in data["levels"].items()})
            
            # Recreate nodes
            for node_id, node_data in data["nodes"].items():
                node = HierarchicalNode.from_dict(node_data)
                tree.nodes[node_id] = node
                
            return tree
        elif format == "pickle":
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unknown format: {format}")