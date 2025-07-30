import json
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from ..agent import RetrievalAgent
from ..retriever import HierarchicalRetriever
from .metrics import exact_match, f1_score
from ..config import DATA_DIR


class Benchmark:
    """Runs a basic benchmark given a dataset of {id, question, answer}."""

    def __init__(self, index_name: str, dataset_path: Path):
        self.index_name = index_name
        self.dataset_path = dataset_path
        self.agent = RetrievalAgent(index_name)
        self.leaf_retriever = HierarchicalRetriever(index_name)

    def _load_dataset(self):
        with open(self.dataset_path) as f:
            data = [json.loads(line) for line in f]
        return data

    def run(self, max_examples: int = 50):
        data = self._load_dataset()[:max_examples]
        agent_metrics = {"em": [], "f1": []}
        baseline_metrics = {"em": [], "f1": []}
        for sample in tqdm(data, desc="Benchmark"):
            q, ans = sample["question"], sample["answer"]
            # AH-RAG answer
            pred = self.agent.answer(q)
            agent_metrics["em"].append(exact_match(pred, ans))
            agent_metrics["f1"].append(f1_score(pred, ans))
            # Baseline: retrieve leaf nodes and return concatenated context (simulate naive answer)
            leaf_nodes = [n for n, _ in self.leaf_retriever.search(q, level=0, top_k=3)]
            naive_pred = " ".join(n.text for n in leaf_nodes)[:200]
            baseline_metrics["em"].append(exact_match(naive_pred, ans))
            baseline_metrics["f1"].append(f1_score(naive_pred, ans))
        self._report(agent_metrics, baseline_metrics)

    @staticmethod
    def _report(agent_metrics, baseline_metrics):
        def mean(lst):
            return sum(lst) / (len(lst) or 1)

        print("=== AH-RAG Results ===")
        print(f"EM:  {mean(agent_metrics['em']):.3f}")
        print(f"F1:  {mean(agent_metrics['f1']):.3f}")
        print("\n=== Baseline (leaf-level) Results ===")
        print(f"EM:  {mean(baseline_metrics['em']):.3f}")
        print(f"F1:  {mean(baseline_metrics['f1']):.3f}")