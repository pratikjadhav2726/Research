# Agentic Hierarchical RAG (AH-RAG)

## Overview

Agentic Hierarchical RAG (AH-RAG) is a novel framework that merges the adaptive, decision-making capabilities of Self-RAG with the multi-level abstraction of RAPTOR. This architecture enables intelligent, goal-directed retrieval that dynamically adapts its search strategy based on query requirements.

## Architecture Components

### 1. Hierarchical Indexing (RAPTOR-style)
- **Leaf Nodes**: Granular text chunks from source documents
- **Intermediate Nodes**: Clustered summaries of related chunks
- **Root Node**: High-level abstract summary of entire corpus
- **Implementation**: Recursive clustering and summarization using LLMs

### 2. Agentic Retrieval Controller
- **Self-RAG Integration**: Uses reflection tokens for decision-making
- **Dynamic Strategy**: Determines optimal abstraction level for queries
- **Iterative Refinement**: Can navigate up/down the hierarchy based on needs

### 3. Key Features
- **Adaptive Retrieval**: Decides if/when retrieval is necessary
- **Multi-Scale Navigation**: Zooms in/out of abstraction levels
- **Self-Critique**: Evaluates retrieved context quality
- **Efficiency**: Avoids searching irrelevant abstraction levels

## Implementation Structure

```
agentic_hierarchical_rag/
├── src/
│   ├── core/              # Core data structures and interfaces
│   ├── indexing/          # RAPTOR-style hierarchical indexing
│   ├── retrieval/         # Retrieval strategies and controllers
│   ├── agents/            # Agentic controller implementation
│   └── evaluation/        # Evaluation metrics and pipelines
├── data/
│   ├── raw/               # Raw document corpus
│   ├── processed/         # Processed chunks and summaries
│   └── embeddings/        # Vector embeddings for nodes
├── tests/                 # Unit and integration tests
├── benchmarks/            # Benchmark datasets and scripts
└── configs/               # Configuration files
```

## Key Innovations

1. **Intelligent Abstraction Selection**: The agent analyzes query nature to determine optimal search level
2. **Dynamic Navigation**: Iterative refinement through tree traversal
3. **Self-Aware Retrieval**: Uses critique tokens to evaluate context sufficiency
4. **Efficiency Gains**: Avoids unnecessary searches at inappropriate abstraction levels

## Evaluation Metrics

- **Retrieval Accuracy**: Precision/Recall at different abstraction levels
- **Efficiency**: Query latency and computational cost
- **Context Quality**: Relevance and completeness of retrieved information
- **Adaptability**: Performance across diverse query types

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch or TensorFlow
- Transformers library
- Vector database (e.g., Faiss, Pinecone)
- LLM API access (OpenAI, Anthropic, or local models)

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

```python
from ah_rag import AgenticHierarchicalRAG

# Initialize the system
ah_rag = AgenticHierarchicalRAG(config_path="configs/default.yaml")

# Index documents
ah_rag.index_documents("data/raw/")

# Query the system
response = ah_rag.query("What are the main themes in the document corpus?")
```

## Research Context

This implementation is based on the novel architecture proposed in "Architectures of Understanding: A Deep Dive into Advanced RAG for Next-Generation Context Engineering". AH-RAG addresses key gaps in current RAG systems by combining hierarchical knowledge representation with intelligent, adaptive retrieval strategies.