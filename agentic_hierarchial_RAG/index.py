from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import INDEX_DIR


@dataclass
class Node:
    """A node in the hierarchical index tree."""

    id: str
    level: int  # 0 = leaf / chunk level, increasing towards root
    text: str
    embedding: List[float]
    children: List[str] = field(default_factory=list)  # IDs of child nodes

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "Node":
        return cls(**d)


class HierarchicalIndex:
    """Stores and retrieves nodes in a hierarchical tree by ID."""

    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.root_id: Optional[str] = None

    # ---------- CRUD helpers ---------

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def create_node(
        self, *, level: int, text: str, embedding: List[float], children: Optional[List[str]] = None
    ) -> str:
        node_id = str(uuid.uuid4())
        self.nodes[node_id] = Node(
            id=node_id, level=level, text=text, embedding=embedding, children=children or []
        )
        return node_id

    def get_node(self, node_id: str) -> Node:
        return self.nodes[node_id]

    # ---------- Persistence ---------

    def _path(self) -> Path:
        return INDEX_DIR / f"{self.name}.json"

    def save(self):
        data = {
            "name": self.name,
            "root_id": self.root_id,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
        }
        self._path().write_text(json.dumps(data))

    @classmethod
    def load(cls, name: str) -> "HierarchicalIndex":
        path = INDEX_DIR / f"{name}.json"
        data = json.loads(path.read_text())
        index = cls(name=data["name"])
        index.root_id = data["root_id"]
        index.nodes = {nid: Node.from_dict(nd) for nid, nd in data["nodes"].items()}
        return index