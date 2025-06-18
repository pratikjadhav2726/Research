"""
Tiered Document Ranking Pipeline for Retrieval Augmented Generation (RAG).

This script implements a multi-stage document retrieval pipeline designed to
enhance Retrieval Augmented Generation systems. The pipeline consists of three main
phases: Retrieve, Re-rank, and Synthesize, using technologies like BM25,
Cross-Encoders, and Large Language Models (LLMs).

This refactored version introduces classes for each pipeline stage, configuration
management for API keys, enhanced error handling, and structured logging for
better traceability and enterprise readiness.
"""

import os
import json
import subprocess
import logging
import torch
import openai
from tqdm.autonotebook import tqdm
from datasets import load_dataset, Dataset as HuggingFaceDataset
from sentence_transformers import CrossEncoder
from pyserini.search.lucene import LuceneSearcher

# --- Configuration ---
# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default model names and paths (can be overridden via main function parameters)
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_LLM_MODEL = "gpt-3.5-turbo" # Changed from gpt-4 for wider accessibility & cost

# --- Utility Functions ---

def load_dependencies_info():
    """
    Provides informational message about installing required Python packages
    and system dependencies (like Java for Pyserini).

    In a production environment, dependencies should be managed by `requirements.txt`
    and proper environment setup. This function serves as a helpful reminder during
    development or manual setup.
    """
    logger.info("Ensuring all dependencies are managed correctly is crucial.")
    logger.info("Python dependencies (typically in requirements.txt): pyserini, sentence-transformers, datasets, tqdm, openai, torch.")
    logger.info("System dependencies: Pyserini requires Java (JDK 11) to be installed and JAVA_HOME environment variable to be set.")

def load_msmarco_dataset(subset: str = "train[:1%]") -> HuggingFaceDataset | None:
    """
    Loads a specified subset of the MSMARCO passage ranking dataset (v1.1).

    Args:
        subset (str, optional): The subset string for Hugging Face Datasets
            (e.g., "train[:1%]", "validation[:100]"). Defaults to "train[:1%]".

    Returns:
        datasets.Dataset | None: The loaded dataset object, or None if loading fails.
    """
    logger.info(f"Attempting to load MSMARCO dataset subset: {subset}...")
    try:
        # MSMARCO v1.1 is specifically for passage ranking task
        dataset = load_dataset("ms_marco", "v1.1", split=subset)
        logger.info(f"Dataset '{subset}' loaded successfully. Number of examples: {len(dataset)}")
        return dataset
    except FileNotFoundError as e:
        logger.error(f"Dataset loading failed: MSMARCO data files not found. Please ensure dataset is available. Error: {e}")
        return None
    except Exception as e: # Catch other potential Hugging Face datasets library errors
        logger.error(f"An unexpected error occurred while loading the MSMARCO dataset: {e}")
        return None

def preprocess_documents(dataset: HuggingFaceDataset) -> dict[str, str]:
    """
    Extracts passages from the MSMARCO dataset and prepares them as a dictionary.

    Args:
        dataset (datasets.Dataset): The loaded MSMARCO dataset.

    Returns:
        dict[str, str]: A dictionary {doc_id: passage_text}.
    """
    logger.info("Preprocessing documents (extracting passages)...")
    processed_docs = {}
    passage_id_counter = 0
    for record in tqdm(dataset, desc="Processing records for passage extraction"):
        if 'passages' in record and 'passage_text' in record['passages']:
            for passage_text in record['passages']['passage_text']:
                if passage_text.strip(): # Ensure passage is not empty
                    processed_docs[str(passage_id_counter)] = passage_text
                    passage_id_counter += 1
        else:
            logger.warning(f"Record missing 'passages' or 'passage_text'. Query ID: {record.get('query_id', 'N/A')}")

    logger.info(f"Preprocessed {len(processed_docs)} non-empty documents (passages).")
    return processed_docs

# --- Pipeline Stage Classes ---

class BM25Retriever:
    """
    Handles BM25 indexing and retrieval using Pyserini.
    """
    def __init__(self, index_path: str, documents: dict[str, str], force_reindex: bool = False):
        """
        Initializes the BM25Retriever, setting up the searcher.
        Builds a new index if one doesn't exist at index_path or if force_reindex is True.

        Args:
            index_path (str): Path to store/load the Lucene index.
            documents (dict[str, str]): Dictionary of documents {doc_id: text}.
            force_reindex (bool): If True, forces rebuilding of the index even if it exists.

        Raises:
            Exception: If index creation or searcher loading fails.
        """
        self.index_path = index_path
        self.documents = documents # Storing documents for retrieval phase
        self.searcher = None

        if force_reindex and os.path.exists(self.index_path):
            logger.info(f"Force re-indexing: Removing existing index at {self.index_path}")
            # Simple removal, for robust removal, use shutil.rmtree if index_path is a directory
            if os.path.isdir(self.index_path):
                import shutil
                shutil.rmtree(self.index_path)
            else: # if it's a file or symlink
                os.remove(self.index_path)


        if not os.path.exists(self.index_path):
            logger.info(f"BM25 index not found at '{self.index_path}'. Creating a new one...")
            self._create_index()
        else:
            logger.info(f"Loading existing BM25 index from '{self.index_path}'.")

        try:
            self.searcher = LuceneSearcher(self.index_path)
            logger.info("BM25 LuceneSearcher loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LuceneSearcher from path '{self.index_path}': {e}", exc_info=True)
            raise  # Re-raise the exception to signal failure

    def _create_index(self):
        """
        Creates a new BM25 index using Pyserini.
        """
        output_dir = os.path.dirname(self.index_path)
        if not output_dir: output_dir = "."
        os.makedirs(output_dir, exist_ok=True)

        jsonl_path = os.path.join(output_dir, f"pyserini_docs_for_{os.path.basename(self.index_path)}.jsonl")

        logger.info(f"Preparing documents in JSONL format at '{jsonl_path}' for indexing.")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for doc_id, content in self.documents.items():
                f.write(json.dumps({"id": doc_id, "contents": content}) + "\n")

        logger.info("Running Pyserini indexing command. This may take some time...")
        logger.info("Requires Java (JDK 11) and Pyserini correctly installed and configured.")

        index_command = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection", "--input", jsonl_path,
            "--index", self.index_path, "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "2", "--storePositions", "--storeDocvectors", "--storeRaw"
        ]

        try:
            logger.info(f"Executing Pyserini index command: {' '.join(index_command)}")
            process = subprocess.run(index_command, check=True, capture_output=True, text=True)
            logger.info("Pyserini indexing process completed.")
            logger.debug(f"Pyserini stdout: {process.stdout}")
            if process.stderr:
                 logger.warning(f"Pyserini stderr: {process.stderr}")
            if not os.path.exists(self.index_path):
                logger.error(f"Index creation failed: Index directory {self.index_path} not found after Pyserini command.")
                raise Exception(f"Pyserini index creation failed for {self.index_path}")
            logger.info(f"BM25 index created successfully at '{self.index_path}'.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Pyserini indexing command failed with return code {e.returncode}.", exc_info=True)
            logger.error(f"Pyserini stdout: {e.stdout}")
            logger.error(f"Pyserini stderr: {e.stderr}")
            raise Exception("Pyserini indexing failed.") from e
        except FileNotFoundError:
            logger.error("'python' command not found for Pyserini indexing. Ensure Python is in PATH.", exc_info=True)
            raise
        finally:
            # Clean up the temporary JSONL file
            if os.path.exists(jsonl_path):
                os.remove(jsonl_path)
                logger.info(f"Removed temporary JSONL file: {jsonl_path}")


    def retrieve(self, query: str, top_k: int = 100) -> list[tuple[str, float, str]]:
        """
        Retrieves top_k documents for a query using the initialized BM25 searcher.

        Args:
            query (str): The search query.
            top_k (int): Number of documents to retrieve. Defaults to 100.

        Returns:
            list[tuple[str, float, str]]: List of (doc_id, score, doc_text).
        """
        if not self.searcher:
            logger.error("BM25 searcher not initialized. Cannot retrieve.")
            return []

        logger.info(f"Retrieving top {top_k} documents for query '{query}' using BM25...")
        hits = self.searcher.search(query, k=top_k)

        results = []
        for hit in hits:
            doc_id = hit.docid
            score = hit.score
            # Assuming self.documents contains the text for all indexed docs
            # This is important if hit.raw or hit.contents is not used or reliable
            doc_text = self.documents.get(doc_id, "Document text not found in preloaded dictionary.")
            results.append((doc_id, score, doc_text))

        logger.info(f"Retrieved {len(results)} documents with BM25.")
        return results

class CrossEncoderReRanker:
    """
    Handles re-ranking of documents using a Cross-Encoder model.
    """
    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL):
        """
        Initializes the CrossEncoderReRanker by loading the specified model.

        Args:
            model_name (str): Hugging Face model name or path for the Cross-Encoder.
        """
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing CrossEncoderReRanker with model: {self.model_name} on device: {self.device}")
        try:
            self.model = CrossEncoder(self.model_name, device=self.device)
            logger.info("Cross-Encoder model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Cross-Encoder model '{self.model_name}': {e}", exc_info=True)
            raise

    def rerank(self, query: str, bm25_results: list[tuple[str, float, str]], top_n: int = 20) -> list[tuple[str, float, str]]:
        """
        Re-ranks documents based on Cross-Encoder scores.

        Args:
            query (str): The original search query.
            bm25_results (list[tuple[str, float, str]]): List of (doc_id, bm25_score, doc_text) from BM25.
            top_n (int): Number of documents to return after re-ranking. Defaults to 20.

        Returns:
            list[tuple[str, float, str]]: Re-ranked list of (doc_id, cross_encoder_score, doc_text).
        """
        if not bm25_results:
            logger.warning("No BM25 results provided to re-rank. Returning empty list.")
            return []

        logger.info(f"Re-ranking {len(bm25_results)} documents with Cross-Encoder model: {self.model_name}...")

        # Format for Cross-Encoder: list of [query, passage_text]
        sentence_pairs = [[query, result[2]] for result in bm25_results]

        logger.info(f"Predicting Cross-Encoder scores for {len(sentence_pairs)} pairs...")
        try:
            cross_encoder_scores = self.model.predict(sentence_pairs, show_progress_bar=True) # tqdm progress bar
        except Exception as e:
            logger.error(f"Error during Cross-Encoder prediction: {e}", exc_info=True)
            return [] # Return empty or handle as appropriate

        reranked_results_details = []
        for i, result in enumerate(bm25_results):
            reranked_results_details.append({
                'doc_id': result[0],
                'score': float(cross_encoder_scores[i]), # Ensure score is float
                'text': result[2]
            })

        reranked_results_details.sort(key=lambda x: x['score'], reverse=True)

        final_results = [(res['doc_id'], res['score'], res['text']) for res in reranked_results_details[:top_n]]
        logger.info(f"Re-ranked and selected top {len(final_results)} documents.")
        return final_results

class LLMSynthesizer:
    """
    Handles answer synthesis using an OpenAI Large Language Model (LLM).
    """
    def __init__(self, openai_api_key: str | None = None, model_name: str = DEFAULT_LLM_MODEL):
        """
        Initializes the LLMSynthesizer.

        Args:
            openai_api_key (str | None, optional): OpenAI API key. If None, it will
                attempt to read from the OPENAI_API_KEY environment variable.
            model_name (str): OpenAI model identifier (e.g., "gpt-3.5-turbo").

        Raises:
            ValueError: If OpenAI API key is not provided and not found in environment variables.
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OpenAI API key not provided and not found in OPENAI_API_KEY environment variable.")
            raise ValueError("OpenAI API key is required for LLMSynthesizer.")

        self.model_name = model_name
        openai.api_key = self.api_key # Set globally for openai library interactions
        logger.info(f"LLMSynthesizer initialized with model: {self.model_name}.")

    def synthesize(self, query: str, reranked_docs: list[tuple[str, float, str]], top_n_docs_for_llm: int = 3) -> str:
        """
        Synthesizes an answer using the LLM based on query and re-ranked documents.

        Args:
            query (str): The original user query.
            reranked_docs (list[tuple[str, float, str]]): Re-ranked documents.
            top_n_docs_for_llm (int): Number of top documents to provide to LLM. Defaults to 3.

        Returns:
            str: The synthesized answer or an error message.
        """
        logger.info(f"Synthesizing answer for query '{query}' using LLM ({self.model_name})...")

        context_docs = reranked_docs[:top_n_docs_for_llm]
        if not context_docs:
            logger.warning("No documents provided to LLM for context.")
            return "No documents were available to synthesize an answer."

        context_parts = [f"Document (ID: {doc[0]}, Score: {doc[1]:.4f}):\n{doc[2]}" for doc in context_docs]
        context_string = "\n\n---\n\n".join(context_parts)

        prompt = (
            f"You are an AI assistant. Answer the query based ONLY on the provided documents.\n"
            f"If the documents do not contain information, state that clearly.\n\n"
            f"Query: \"{query}\"\n\n"
            f"Provided Documents:\n---\n{context_string}\n---\n\n"
            f"Answer:"
        )

        try:
            logger.info(f"Sending request to OpenAI API (model: {self.model_name}). Using {len(context_docs)} documents for context.")
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based ONLY on provided documents. If the answer is not in the documents, say so."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0 # For factual answers
            )
            answer = response.choices[0].message['content'].strip()
            logger.info("Answer successfully synthesized by LLM.")
            return answer
        except openai.error.AuthenticationError as e:
            logger.error(f"OpenAI API Authentication Error: {e}. Check your API key.", exc_info=True)
            return f"OpenAI API Authentication Error: {e}"
        except openai.error.APIError as e: # Catch other OpenAI API specific errors
            logger.error(f"OpenAI API Error: {e}", exc_info=True)
            return f"OpenAI API Error: {e}"
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error during LLM synthesis: {e}", exc_info=True)
            return f"An unexpected error occurred during LLM synthesis: {e}"

# --- Main Orchestration ---
def main(query: str,
         openai_api_key_arg: str | None = None,
         msmarco_subset_str: str = "train[:1%]",
         bm25_index_name: str = "msmarco_bm25_index", # More generic index name
         bm25_docs_top_k: int = 100,
         cross_encoder_model_str: str = DEFAULT_CROSS_ENCODER_MODEL,
         rerank_top_n_count: int = 20,
         llm_model_str: str = DEFAULT_LLM_MODEL,
         llm_context_docs_count: int = 3,
         force_bm25_reindex: bool = False):
    """
    Main function to orchestrate the tiered document ranking and answer synthesis pipeline.
    Uses classes for different stages.
    """
    load_dependencies_info()
    logger.info(f"Starting RAG pipeline for query: '{query}'")

    # --- Configuration for this run ---
    # Construct a unique index path based on subset and index name to avoid collisions
    safe_subset_name = msmarco_subset_str.replace("[:", "_").replace("]", "").replace("%", "percent").replace(" ", "_")
    current_index_path = f"./{bm25_index_name}_{safe_subset_name}"
    logger.info(f"BM25 Index path for this run: {current_index_path}")

    # 1. Load and Preprocess Dataset
    dataset = load_msmarco_dataset(subset=msmarco_subset_str)
    if not dataset:
        logger.critical("Dataset loading failed. Exiting pipeline.")
        return

    documents_dict = preprocess_documents(dataset)
    if not documents_dict:
        logger.critical("No documents were processed from the dataset. Exiting pipeline.")
        return

    # 2. BM25 Retrieval Stage
    try:
        retriever = BM25Retriever(index_path=current_index_path, documents=documents_dict, force_reindex=force_bm25_reindex)
        bm25_retrieved_docs = retriever.retrieve(query, top_k=bm25_docs_top_k)
    except Exception as e:
        logger.critical(f"BM25Retriever initialization or retrieval failed: {e}. Exiting pipeline.", exc_info=True)
        return

    if not bm25_retrieved_docs:
        logger.warning("BM25 returned no results. Further processing may yield no answer.")
        # Depending on desired behavior, could exit or proceed

    # 3. Cross-Encoder Re-ranking Stage
    try:
        reranker = CrossEncoderReRanker(model_name=cross_encoder_model_str)
        if bm25_retrieved_docs:
            reranked_docs_list = reranker.rerank(query, bm25_retrieved_docs, top_n=rerank_top_n_count)
        else:
            logger.info("Skipping re-ranking as BM25 returned no documents.")
            reranked_docs_list = []
    except Exception as e:
        logger.error(f"CrossEncoderReRanker initialization or reranking failed: {e}. Proceeding without re-ranking.", exc_info=True)
        # Fallback: use BM25 results directly if reranker fails, or handle error more strictly
        reranked_docs_list = bm25_retrieved_docs[:rerank_top_n_count] # Simple fallback

    # 4. LLM Answer Synthesis Stage
    llm_generated_answer = "LLM Synthesis skipped or failed."
    if not reranked_docs_list:
        logger.warning("No documents available for LLM synthesis (either BM25 or reranker yielded no results).")
        llm_generated_answer = "No relevant documents found to synthesize an answer."
    else:
        try:
            synthesizer = LLMSynthesizer(openai_api_key=openai_api_key_arg, model_name=llm_model_str)
            llm_generated_answer = synthesizer.synthesize(query, reranked_docs_list, top_n_docs_for_llm=llm_context_docs_count)
        except ValueError as e: # Specifically catch API key issues from constructor
            logger.error(f"LLMSynthesizer setup failed: {e}. Cannot synthesize answer.", exc_info=True)
            llm_generated_answer = f"LLM setup failed: {e}"
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}", exc_info=True)
            llm_generated_answer = f"LLM synthesis encountered an error: {e}"

    # --- Output Results ---
    logger.info("\n--- Final Generated Answer ---")
    logger.info(llm_generated_answer)

    if bm25_retrieved_docs:
        logger.info("\n--- BM25 Top 3 Results (Sample) ---")
        for i, (doc_id, score, text) in enumerate(bm25_retrieved_docs[:3]):
            logger.info(f"Rank {i+1} (ID: {doc_id}, BM25 Score: {score:.4f}): {text[:150]}...")

    if reranked_docs_list and reranked_docs_list != bm25_retrieved_docs[:rerank_top_n_count]: # Avoid double printing if fallback used
        logger.info("\n--- Cross-Encoder Re-ranked Top 3 Results (Sample) ---")
        for i, (doc_id, score, text) in enumerate(reranked_docs_list[:3]):
            logger.info(f"Rank {i+1} (ID: {doc_id}, CE Score: {score:.4f}): {text[:150]}...")

    logger.info("RAG pipeline execution finished.")


if __name__ == "__main__":
    sample_query_main = "What are the main causes of climate change?"

    # API key can be passed as an argument or fetched from ENV by LLMSynthesizer
    # For this example, we try to get it from ENV; if not found, LLMSynthesizer will raise ValueError.
    # Alternatively, pass it directly: main(query=..., openai_api_key_arg="sk-...")
    api_key_from_env = os.getenv("OPENAI_API_KEY")
    if not api_key_from_env:
        logger.warning("OPENAI_API_KEY environment variable not set. LLM Synthesis will fail if not provided elsewhere.")
        # Consider providing a default or dummy key for testing other parts, or exiting.
        # For now, LLMSynthesizer will handle the error.

    # Example main call with some parameters changed from defaults
    main(
        query=sample_query_main,
        openai_api_key_arg=api_key_from_env, # Pass the key (can be None if not set)
        msmarco_subset_str="train[:50]",     # Very small subset for quick testing
        bm25_index_name="msmarco_test_idx",
        bm25_docs_top_k=20,
        rerank_top_n_count=5,
        llm_context_docs_count=2,
        force_bm25_reindex=False # Set to True to rebuild index on first run or after changes
    )

# End of script
