import math
import os
import random
from pathlib import Path
from typing import List, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from .config import (
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
    SUMMARIZATION_MODEL_NAME,
    MAX_CHILDREN,
)
from .index import HierarchicalIndex


class HierarchicalIndexer:
    """Builds a RAPTOR-style hierarchical index that stores summaries at multiple levels."""

    def __init__(self, name: str):
        self.name = name
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        # BART summarization pipeline (can be any summarizer)
        self.summarizer = pipeline(
            "summarization",
            model=SUMMARIZATION_MODEL_NAME,
            device=0 if os.getenv("CUDA_VISIBLE_DEVICES") else -1,
        )

    # ---------------- Public API ----------------

    def build_from_documents(self, docs: Sequence[str]):
        """Entry point. Builds hierarchy and persists index."""
        # 1. Prepare leaf nodes (chunks)
        leaf_texts = self._chunk_documents(docs)
        leaf_embeddings = self.embedder.encode(leaf_texts, convert_to_numpy=True)

        index = HierarchicalIndex(self.name)
        level_nodes: List[str] = []
        for text, emb in zip(leaf_texts, leaf_embeddings):
            node_id = index.create_node(level=0, text=text, embedding=emb.tolist())
            level_nodes.append(node_id)

        # 2. Recursively build upper levels
        current_level = 0
        while len(level_nodes) > 1:
            current_level += 1
            # Determine number of clusters (max children per parent)
            n_clusters = max(1, math.ceil(len(level_nodes) / MAX_CHILDREN))
            # Collect embeddings for nodes to cluster
            node_embs = np.array([index.get_node(nid).embedding for nid in level_nodes])
            # For small numbers, skip clustering and directly attach to a root
            if len(level_nodes) <= MAX_CHILDREN:
                cluster_labels = np.zeros(len(level_nodes), dtype=int)
                n_clusters = 1
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                cluster_labels = kmeans.fit_predict(node_embs)

            new_level_nodes: List[str] = []
            for cl in range(n_clusters):
                cluster_node_ids = [nid for nid, label in zip(level_nodes, cluster_labels) if label == cl]
                cluster_text = " \n".join([index.get_node(nid).text for nid in cluster_node_ids])
                # Summarize cluster text to create parent text
                summary = self._summarize(cluster_text)
                # Compute embedding for summary
                summary_emb = self.embedder.encode([summary])[0]
                parent_id = index.create_node(
                    level=current_level, text=summary, embedding=summary_emb.tolist(), children=cluster_node_ids
                )
                new_level_nodes.append(parent_id)
            level_nodes = new_level_nodes
        # Last remaining node is root
        index.root_id = level_nodes[0]
        index.save()
        print(f"[Indexer] Saved hierarchical index '{self.name}' with {len(index.nodes)} nodes.")

    # ---------------- Helpers ----------------

    def _chunk_documents(self, docs: Sequence[str], chunk_size: int = 512) -> List[str]:
        """Naive tokenizer: split docs into chunks of ~chunk_size words."""
        leaf_chunks: List[str] = []
        for doc in docs:
            words = doc.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                # Ensure chunk has some minimum length
                if len(chunk.split()) > 20:
                    leaf_chunks.append(chunk)
        return leaf_chunks

    def _summarize(self, text: str, max_len: int = 120) -> str:
        # Try-catch to fall back on naive summarizer if text is too short
        try:
            summary = self.summarizer(text[:4000], max_length=max_len, min_length=30, do_sample=False)[0][
                "summary_text"
            ]
            return summary.strip()
        except Exception:
            # Fallback: take first 3 sentences
            return " ".join(text.split(".")[:3])