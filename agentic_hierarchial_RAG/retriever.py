from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL_NAME, TOP_K_CANDIDATES
from .index import HierarchicalIndex, Node


class HierarchicalRetriever:
    """Retrieves relevant nodes from a hierarchical index at a given level."""

    def __init__(self, index_name: str):
        self.index = HierarchicalIndex.load(index_name)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # --------- Public API ----------

    def search(self, query: str, level: int, top_k: int = TOP_K_CANDIDATES) -> List[Tuple[Node, float]]:
        """Return top_k nodes at given level ranked by cosine similarity."""
        q_emb = self.embedder.encode([query])[0]
        # Collect nodes at requested level
        level_nodes = [node for node in self.index.nodes.values() if node.level == level]
        if not level_nodes:
            raise ValueError(f"No nodes found at level {level}. Available levels: {sorted({n.level for n in self.index.nodes.values()})}")
        node_embs = np.array([n.embedding for n in level_nodes])
        # Normalize for cosine similarity
        q_vec = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        n_vecs = node_embs / (np.linalg.norm(node_embs, axis=1, keepdims=True) + 1e-9)
        sims = np.dot(n_vecs, q_vec)
        top_idx = sims.argsort()[-top_k:][::-1]
        return [(level_nodes[i], sims[i]) for i in top_idx]