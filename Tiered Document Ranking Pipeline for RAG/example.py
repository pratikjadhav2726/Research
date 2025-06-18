"""
Example script for using the Tiered Document Ranking Pipeline.

This script demonstrates how to instantiate and use the classes from
`tiered_ranking_pipeline.py` to perform a full Retrieve-Rerank-Synthesize
pipeline for a sample query.

Prerequisites:
1.  Ensure all dependencies for `tiered_ranking_pipeline.py` are installed
    (e.g., pyserini, sentence-transformers, datasets, openai, torch).
2.  Pyserini requires Java JDK 11 and `JAVA_HOME` to be set for BM25 indexing.
3.  For the LLM synthesis step, an OpenAI API key is required. It's best to set
    this as an environment variable: `export OPENAI_API_KEY="your_api_key_here"`.
"""

import os
import logging

# Import components from the main pipeline script
from tiered_ranking_pipeline import (
    BM25Retriever,
    CrossEncoderReRanker,
    LLMSynthesizer,
    load_msmarco_dataset,
    preprocess_documents,
    # Optional: load_dependencies_info can be called if desired
    # load_dependencies_info
)

# Configure basic logging for the example script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_example_pipeline():
    """
    Executes a demonstration of the tiered document ranking pipeline.
    """
    logger.info("Starting example run of the Tiered Document Ranking Pipeline...")

    # --- 1. Configuration ---
    # (Users should adjust these for their own use cases)
    sample_query = "What are the symptoms of influenza?"
    # Using a very small subset of MSMARCO for quick example execution.
    # For real use, a larger subset or different dataset would be used.
    dataset_subset = "train[:30]" # Approx 30 query-passage sets from MSMARCO
    # Define a unique index path for this example to avoid conflicts.
    # This path will be used by BM25Retriever.
    example_bm25_index_path = "./example_msmarco_bm25_index_30docs"
    # Forcing re-indexing for this example to ensure it runs end-to-end
    # without assuming a pre-built index. Set to False if index is already built.
    force_reindex_bm25 = True

    # OpenAI API Key:
    # The LLMSynthesizer will try to get this from the environment variable OPENAI_API_KEY.
    # Alternatively, you can pass it directly:
    # openai_api_key_direct = "sk-your_actual_api_key" (NOT RECOMMENDED for shared code)
    # For this example, we rely on the environment variable.
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable is not set.")
        logger.warning("LLM synthesis will likely fail. Please set the API key.")
        # Example could proceed without LLM or use a dummy key if LLMSynthesizer is modified to allow it.

    logger.info(f"Sample Query: '{sample_query}'")
    logger.info(f"Dataset Subset: '{dataset_subset}'")
    logger.info(f"BM25 Index Path: '{example_bm25_index_path}' (Force Re-index: {force_reindex_bm25})")

    # --- 2. Load and Preprocess Data ---
    logger.info("Loading dataset...")
    dataset = load_msmarco_dataset(subset=dataset_subset)
    if not dataset:
        logger.error("Failed to load dataset. Aborting example.")
        return

    logger.info("Preprocessing documents...")
    documents_map = preprocess_documents(dataset) # dict: {doc_id: text}
    if not documents_map:
        logger.error("Failed to preprocess documents. Aborting example.")
        return
    logger.info(f"Successfully preprocessed {len(documents_map)} documents.")

    # --- 3. Initialize BM25Retriever ---
    # This step will build the index if it doesn't exist or if force_reindex_bm25 is True.
    # This might take some time on the first run, especially with larger datasets.
    logger.info("Initializing BM25Retriever...")
    try:
        bm25_retriever = BM25Retriever(
            index_path=example_bm25_index_path,
            documents=documents_map,
            force_reindex=force_reindex_bm25
        )
    except Exception as e:
        logger.error(f"Failed to initialize BM25Retriever: {e}. Aborting example.", exc_info=True)
        logger.error("Ensure Java JDK 11 is installed and JAVA_HOME is set for Pyserini.")
        return
    logger.info("BM25Retriever initialized successfully.")

    # --- 4. Retrieve Documents with BM25 ---
    logger.info("Retrieving documents using BM25...")
    # Parameters for retrieval can be adjusted (e.g., top_k)
    bm25_top_k = 20 # Retrieve top 20 documents for this small example
    bm25_results = bm25_retriever.retrieve(query=sample_query, top_k=bm25_top_k) # List of (doc_id, score, text)
    if not bm25_results:
        logger.warning("BM25 retrieval returned no documents.")
        # Depending on the use case, one might stop or proceed with an empty list.
    else:
        logger.info(f"BM25 retrieved {len(bm25_results)} documents.")
        # Print a sample of BM25 results
        for i, (doc_id, score, text) in enumerate(bm25_results[:3]):
            logger.info(f"  BM25 Rank {i+1}: ID={doc_id}, Score={score:.4f}, Text='{text[:100]}...'")


    # --- 5. Initialize CrossEncoderReRanker ---
    logger.info("Initializing CrossEncoderReRanker...")
    try:
        # Using default model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
        cross_encoder_reranker = CrossEncoderReRanker()
    except Exception as e:
        logger.error(f"Failed to initialize CrossEncoderReRanker: {e}. Aborting example.", exc_info=True)
        return
    logger.info("CrossEncoderReRanker initialized successfully.")

    # --- 6. Re-rank Documents ---
    reranked_results = []
    if bm25_results:
        logger.info("Re-ranking documents using Cross-Encoder...")
        # Parameters for re-ranking can be adjusted (e.g., top_n)
        rerank_top_n = 5 # Re-rank and keep top 5 for this example
        reranked_results = cross_encoder_reranker.rerank(
            query=sample_query,
            bm25_results=bm25_results,
            top_n=rerank_top_n
        ) # List of (doc_id, score, text)
        if not reranked_results:
            logger.warning("Cross-Encoder re-ranking returned no documents.")
        else:
            logger.info(f"Cross-Encoder re-ranked to {len(reranked_results)} documents.")
            # Print a sample of re-ranked results
            for i, (doc_id, score, text) in enumerate(reranked_results[:3]):
                logger.info(f"  CE Rank {i+1}: ID={doc_id}, Score={score:.4f}, Text='{text[:100]}...'")
    else:
        logger.info("Skipping re-ranking as BM25 returned no documents.")


    # --- 7. Initialize LLMSynthesizer ---
    logger.info("Initializing LLMSynthesizer...")
    try:
        # API key is expected to be in OPENAI_API_KEY environment variable.
        # Using default LLM model (e.g., "gpt-3.5-turbo")
        llm_synthesizer = LLMSynthesizer()
    except ValueError as e: # Catch API key error specifically
        logger.error(f"Failed to initialize LLMSynthesizer: {e}.")
        logger.error("Please ensure OPENAI_API_KEY environment variable is set correctly.")
        logger.info("Skipping LLM synthesis step.")
        final_answer = "LLM Synthesizer initialization failed due to missing API key."
    except Exception as e:
        logger.error(f"Failed to initialize LLMSynthesizer: {e}. Skipping LLM synthesis.", exc_info=True)
        final_answer = f"LLM Synthesizer initialization failed: {e}"
    else:
        logger.info("LLMSynthesizer initialized successfully.")
        # --- 8. Synthesize Final Answer ---
        if reranked_results:
            logger.info("Synthesizing final answer using LLM...")
            # Parameters for synthesis can be adjusted (e.g., top_n_docs_for_llm)
            llm_context_docs = 2 # Use top 2 re-ranked documents for LLM context
            final_answer = llm_synthesizer.synthesize(
                query=sample_query,
                reranked_docs=reranked_results,
                top_n_docs_for_llm=llm_context_docs
            )
        elif bm25_results: # Fallback if re-ranking failed but BM25 worked
            logger.info("Synthesizing final answer using LLM with BM25 results (re-ranking was skipped or failed)...")
            llm_context_docs = 2
            final_answer = llm_synthesizer.synthesize(
                query=sample_query,
                reranked_docs=bm25_results, # Pass BM25 results directly
                top_n_docs_for_llm=llm_context_docs
            )
        else:
            logger.info("Skipping LLM synthesis as no documents were retrieved or re-ranked.")
            final_answer = "No documents available to synthesize an answer."

    # --- 9. Display Final Answer ---
    logger.info("\n--- Pipeline Execution Summary ---")
    logger.info(f"Original Query: {sample_query}")
    logger.info(f"Number of documents after BM25 retrieval: {len(bm25_results)}")
    logger.info(f"Number of documents after Cross-Encoder re-ranking: {len(reranked_results)}")
    logger.info(f"Final Synthesized Answer:\n{final_answer}")

    logger.info("\nExample run finished.")
    logger.info("NOTE: For actual use, you would typically use a larger dataset, a persistent index path,")
    logger.info("and ensure your OpenAI API key is securely managed.")


if __name__ == "__main__":
    # Optional: Call this to see dependency reminders from the imported module.
    # from tiered_ranking_pipeline import load_dependencies_info
    # load_dependencies_info()

    run_example_pipeline()
