# Tiered Document Ranking Pipeline for RAG

This project provides a Python-based, multi-stage document retrieval pipeline designed to enhance Retrieval Augmented Generation (RAG) systems. It retrieves relevant documents, re-ranks them for improved precision, and then uses a Large Language Model (LLM) to synthesize a final answer based on the retrieved context.

The pipeline is now structured as a Python module with classes for each major stage, facilitating integration and testing.

## Project Overview

The pipeline follows a tiered approach for efficient and accurate document retrieval and answer synthesis:

1.  **Phase 1 - Retrieve:**
    *   Utilizes BM25 (a keyword-based sparse retrieval method) via the Pyserini library to quickly fetch an initial set of candidate documents (e.g., top 500) from a larger corpus like MSMARCO.
2.  **Phase 2 - Re-Rank:**
    *   Employs a Cross-Encoder model from the `sentence-transformers` library to re-rank a subset of the initially retrieved documents (e.g., top 50). Cross-Encoders provide higher accuracy by performing full attention over the query and document text.
3.  **Phase 3 - Synthesize:**
    *   Passes the top N re-ranked documents (e.g., top 3-10) to a Large Language Model (LLM) like OpenAI's GPT series to generate a concise, context-aware answer to the user's query.

This approach balances the speed of sparse retrieval with the accuracy of neural re-rankers, followed by the generative capabilities of LLMs.

![image](https://github.com/user-attachments/assets/c6d575b0-7355-4b2e-becc-c25d5d4b41eb)
*(Diagram illustrating the three phases of the pipeline)*

## Directory Structure

The project is organized as follows:

```
Tiered Document Ranking Pipeline for RAG/
│
├── tiered_ranking_pipeline.py  # Main pipeline module with core classes (BM25Retriever, etc.)
├── example.py                  # Example script demonstrating how to use the pipeline
├── Readme.md                   # This documentation file
├── requirements.txt            # Python package dependencies
│
├── tests/                      # Directory containing unit tests
│   ├── __init__.py
│   ├── test_data_handling.py
│   ├── test_bm25_retriever.py
│   ├── test_cross_encoder_reranker.py
│   ├── test_llm_synthesizer.py
│   └── test_pipeline_integration.py
│
└── example_msmarco_bm25_index_30docs/ # Example path for a BM25 index (created by example.py)
                                     # The actual index path can be configured.
```

## Setup and Installation

Follow these steps to set up the environment and run the pipeline:

1.  **Prerequisites:**
    *   **Python:** Ensure you have Python 3.8 or newer installed.
    *   **Java (JDK 11):** Pyserini, used for BM25 indexing and retrieval, requires Java JDK 11. Make sure it's installed and the `JAVA_HOME` environment variable is correctly set.
        *   You can check your Java installation with `java -version`.
        *   You can check `JAVA_HOME` with `echo $JAVA_HOME` (Linux/macOS) or `echo %JAVA_HOME%` (Windows).

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv rag_env
    source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
    ```

3.  **Install Python Dependencies:**
    Install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r "Tiered Document Ranking Pipeline for RAG/requirements.txt"
    ```
    *(Note: Adjust the path to `requirements.txt` if you are running this command from a different directory relative to the file.)*

4.  **Set OpenAI API Key:**
    For the LLM synthesis stage (using OpenAI models), you need an API key. Set it as an environment variable:
    ```bash
    export OPENAI_API_KEY="your_actual_openai_api_key_here"
    ```
    On Windows:
    ```bash
    set OPENAI_API_KEY="your_actual_openai_api_key_here"
    ```
    The `LLMSynthesizer` class in the pipeline will automatically try to load this key.

## Running the Pipeline

### Using the Example Script

An `example.py` script is provided to demonstrate a full run of the pipeline with a small dataset subset.

To run the example:
```bash
python "Tiered Document Ranking Pipeline for RAG/example.py"
```
*(Note: Adjust the path to `example.py` if necessary.)*

The example script handles:
*   Loading a small portion of the MSMARCO dataset.
*   Preprocessing documents.
*   Building a temporary BM25 index (if `force_reindex` is True or index doesn't exist).
*   Retrieving and re-ranking documents.
*   Synthesizing an answer using the LLM.

Review `example.py` to see how the pipeline components are instantiated and used.

### Using as a Module

The core logic is encapsulated in classes within `tiered_ranking_pipeline.py`. You can import these classes into your own projects:

```python
from tiered_ranking_pipeline import (
    BM25Retriever,
    CrossEncoderReRanker,
    LLMSynthesizer,
    load_msmarco_dataset,
    preprocess_documents
)

# Initialize and use the components as shown in example.py
# Example:
# dataset = load_msmarco_dataset(subset="train[:100]")
# documents = preprocess_documents(dataset)
# retriever = BM25Retriever(index_path="./my_bm25_index", documents=documents)
# # ... and so on
```

## Running Tests

Unit tests are provided in the `tests/` directory to ensure the functionality of individual components.

1.  **Install Testing Dependencies:**
    If you haven't already, install `pytest` and `pytest-mock`:
    ```bash
    pip install pytest pytest-mock
    ```

2.  **Run Tests:**
    Navigate to the root directory of this project (the one containing the "Tiered Document Ranking Pipeline for RAG" folder) and run:
    ```bash
    python -m pytest "Tiered Document Ranking Pipeline for RAG/tests/"
    ```
    This command will discover and run all tests in the specified directory.

## Configuration

Key aspects of the pipeline can be configured:

*   **Dataset:** The `msmarco_subset_str` parameter in `main()` (and `load_msmarco_dataset`) controls which part of the MSMARCO dataset is used.
*   **BM25 Index Path:** The `bm25_index_name` parameter helps define where the BM25 index is stored/loaded. The `BM25Retriever` also accepts `force_reindex`.
*   **Models:**
    *   Cross-Encoder model: Can be specified when creating `CrossEncoderReRanker` instance. Default is `cross-encoder/ms-marco-MiniLM-L-6-v2`.
    *   LLM model: Can be specified when creating `LLMSynthesizer` instance. Default is `gpt-3.5-turbo`.
*   **OpenAI API Key:** Handled by `LLMSynthesizer` (via argument or `OPENAI_API_KEY` environment variable).
*   **Top-K Values:** Parameters like `bm25_docs_top_k`, `rerank_top_n_count`, and `llm_context_docs_count` in `main()` (and corresponding class methods) control how many documents are processed at each stage.

Refer to the `main()` function in `tiered_ranking_pipeline.py` and the `example.py` script for how these configurations are passed.

## Tech Stack

*   **Python:** Core programming language.
*   **Pyserini:** For BM25 indexing and retrieval. Requires Java JDK 11.
*   **Sentence-Transformers:** For Cross-Encoder models used in the re-ranking phase.
*   **OpenAI API:** For accessing GPT models (e.g., `gpt-3.5-turbo`, `gpt-4`) for answer synthesis.
*   **Hugging Face `datasets`:** For loading and handling datasets like MSMARCO.
*   **Pytest & Pytest-Mock:** For unit testing.
*   **Torch:** As a backend for sentence-transformers.

*(FAISS was mentioned in earlier versions for potential vector search but is not a core component of the current BM25-focused pipeline script.)*

## Future Work

Potential areas for future development include:

*   **Dense Retrieval Integration:** Incorporate dense retrievers (e.g., using FAISS with embeddings from Sentence-Transformers or DPR) as an alternative or complement to BM25 in the first stage.
*   **Advanced Re-ranking:** Explore more sophisticated re-ranking models or multi-stage re-ranking.
*   **LLM Flexibility:** Add support for other LLM providers or locally hosted open-source LLMs (e.g., via Hugging Face Transformers, llama-cpp-python).
*   **Modularity and Abstraction:** Further abstract pipeline stages to allow easier swapping of components (e.g., different retrievers, rerankers, synthesizers).
*   **Performance Optimization:** For larger datasets, optimize indexing and retrieval steps.
*   **Evaluation Framework:** Implement a framework to evaluate the end-to-end performance of the RAG pipeline on benchmark datasets.
*   **Configuration Files:** Move configurable parameters (model names, paths, top-k values) to external configuration files (e.g., YAML, JSON) instead of being hardcoded or passed primarily through function arguments in `main`.
*   **Web API / Service:** Wrap the pipeline in a simple web API (e.g., using FastAPI or Flask) for easier integration into other applications.
