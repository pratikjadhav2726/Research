# Agentic Hierarchical RAG (AH-RAG)

**AH-RAG** implements *Novel Idea 2* from the accompanying research note – an agent-driven, multi-scale Retrieval-Augmented-Generation pipeline that combines:

1. A RAPTOR-style hierarchical index that stores text chunks at the leaf level and recursive, abstractive summaries at higher levels.
2. A Self-RAG-inspired *retrieval agent* that decides – at run-time – which abstraction level(s) to query, and that can iteratively "zoom-in / zoom-out" until sufficient evidence is gathered.
3. A pluggable language-model backend (OpenAI or local HF) used both for summary generation during indexing *and* answer generation during inference.

This repository contains **reference code** strong enough to demonstrate the architecture **yet lightweight** enough to run on a laptop.  Every component is designed to be replaceable with more powerful or domain-specific alternatives.

---
## Directory structure
```
agentic_hierarchial_RAG/
├── __init__.py
├── README.md                ← this file
├── config.py                ← global hyper-parameters & paths
├── index.py                 ← dataclass definitions + (de)serialization helpers
├── ingest.py                ← builds the hierarchical index
├── retriever.py             ← cosine search at an arbitrary level
├── agent.py                 ← the agentic controller (decide / retrieve / refine / generate)
└── evaluation/
    ├── metrics.py           ← EM / F1 helpers
    └── benchmark.py         ← simple evaluation pipeline
```

---
## Quick-start
### 1. Install dependencies
```bash
pip install -r requirements.txt  # or manually install: sentence-transformers, transformers, scikit-learn, tqdm, openai
```

### 2. Index some documents
```python
from agentic_hierarchial_RAG.ingest import HierarchicalIndexer

raw_docs = [open("doc1.txt").read(), open("doc2.txt").read(), ...]
indexer = HierarchicalIndexer(name="demo_corpus")
indexer.build_from_documents(raw_docs)
```
This will create `indexes/demo_corpus.json` storing the full hierarchy.

### 3. Ask questions
```python
from agentic_hierarchial_RAG.agent import RetrievalAgent

agent = RetrievalAgent(index_name="demo_corpus")
print(agent.answer("Summarise the main arguments in section 3"))
```
The agent will:
1. Decide which abstraction level is most appropriate.
2. Retrieve the top-k nodes at that level.
3. If the context seems insufficient it zooms in/out and repeats.
4. Finally calls the generator LLM to craft an answer that cites evidence.

### 4. Run a benchmark (optional)
Prepare a dataset file where each line is a JSON object with `{"question": ..., "answer": ...}` – for instance a subset of *HotpotQA*.
```python
from pathlib import Path
from agentic_hierarchial_RAG.evaluation.benchmark import Benchmark

bench = Benchmark(index_name="demo_corpus", dataset_path=Path("my_eval.jsonl"))
bench.run(max_examples=100)
```
The script reports Exact-Match and F1 for both AH-RAG **and** a naive leaf-level retriever baseline.

---
## Implementation notes
* **Chunking & summaries** – documents are greedily split every 512 words; parents are formed by *K-means* clustering (≤ `MAX_CHILDREN` leaves per parent) followed by abstractive summarisation (Bart-CNN by default).
* **Persistence** – the entire tree lives in a single compact JSON; each node stores its embedding so retrieval requires **no database**.
* **Search** – cosine similarity using *sentence-transformers* embeddings.
* **Agent heuristics** – for brevity only a handful of OpenAI calls are used (one to pick the starting level + one to generate the final answer).  All other logic is heuristic but can be swapped with explicit reflection tokens.

---
## Extending this prototype
* Replace the `SentenceTransformer` or summariser with a domain-specific model.
* Swap the JSON store with a vector DB for scalability.
* Train a small classifier to replace the initial OpenAI call that selects the starting level.
* Implement more sophisticated *self-critique* for deciding when to stop iterating.
* Integrate cross-encoder re-ranking inside `retriever.py` for higher precision.

---
## Licence
MIT