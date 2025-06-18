"""
Integration tests for the main() function orchestrator in tiered_ranking_pipeline.py.
These tests will heavily mock the actual pipeline components.
"""
import pytest
from unittest.mock import patch, MagicMock, call

# Import the main function and other components that might be checked (e.g. logger)
from ..tiered_ranking_pipeline import main
# Import classes to be mocked if type hints are needed for mock instances
from ..tiered_ranking_pipeline import BM25Retriever, CrossEncoderReRanker, LLMSynthesizer

# --- Fixtures for Mock Data ---
@pytest.fixture
def mock_dataset_obj():
    """A mock Hugging Face Dataset object."""
    ds = MagicMock()
    ds.__len__.return_value = 5 # Example length
    return ds

@pytest.fixture
def mock_processed_docs_dict():
    """A mock dictionary of processed documents."""
    return {"id1": "text1", "id2": "text2"}

@pytest.fixture
def mock_bm25_results_list():
    """A mock list of BM25 retrieval results."""
    return [("id1", 0.9, "text1"), ("id2", 0.8, "text2")]

@pytest.fixture
def mock_reranked_results_list():
    """A mock list of re-ranked results."""
    return [("id1", 1.5, "text1"), ("id2", 1.2, "text2")] # Higher scores after CE

@pytest.fixture
def mock_llm_answer_str():
    """A mock string for the LLM's synthesized answer."""
    return "This is the final synthesized answer from the LLM."

# --- Mocks for Pipeline Components ---
@pytest.fixture
def mock_pipeline_components(mocker, mock_dataset_obj, mock_processed_docs_dict,
                             mock_bm25_results_list, mock_reranked_results_list,
                             mock_llm_answer_str):
    """Mocks all major components and functions called by main()."""

    # Mock utility functions
    mock_load_deps = mocker.patch('tiered_ranking_pipeline.load_dependencies_info')
    mock_load_dataset = mocker.patch('tiered_ranking_pipeline.load_msmarco_dataset', return_value=mock_dataset_obj)
    mock_preprocess_docs = mocker.patch('tiered_ranking_pipeline.preprocess_documents', return_value=mock_processed_docs_dict)

    # Mock class constructors and their instances' methods
    mock_bm25_retriever_instance = MagicMock(spec=BM25Retriever) # spec ensures it has BM25Retriever methods
    mock_bm25_retriever_instance.retrieve.return_value = mock_bm25_results_list
    mock_bm25_constructor = mocker.patch('tiered_ranking_pipeline.BM25Retriever', return_value=mock_bm25_retriever_instance)

    mock_ce_reranker_instance = MagicMock(spec=CrossEncoderReRanker)
    mock_ce_reranker_instance.rerank.return_value = mock_reranked_results_list
    mock_ce_constructor = mocker.patch('tiered_ranking_pipeline.CrossEncoderReRanker', return_value=mock_ce_reranker_instance)

    mock_llm_synthesizer_instance = MagicMock(spec=LLMSynthesizer)
    mock_llm_synthesizer_instance.synthesize.return_value = mock_llm_answer_str
    mock_llm_constructor = mocker.patch('tiered_ranking_pipeline.LLMSynthesizer', return_value=mock_llm_synthesizer_instance)

    # Mock logger within the main module
    mock_logger = mocker.patch('tiered_ranking_pipeline.logger')

    return {
        "load_dependencies_info": mock_load_deps,
        "load_msmarco_dataset": mock_load_dataset,
        "preprocess_documents": mock_preprocess_docs,
        "BM25Retriever": mock_bm25_constructor,
        "bm25_retriever_instance": mock_bm25_retriever_instance,
        "CrossEncoderReRanker": mock_ce_constructor,
        "cross_encoder_reranker_instance": mock_ce_reranker_instance,
        "LLMSynthesizer": mock_llm_constructor,
        "llm_synthesizer_instance": mock_llm_synthesizer_instance,
        "logger": mock_logger,
    }


# --- Integration Tests for main() ---

def test_main_pipeline_full_success(mock_pipeline_components, mock_processed_docs_dict,
                                    mock_bm25_results_list, mock_reranked_results_list,
                                    mock_llm_answer_str):
    """
    Test the main() function for a successful run through all stages.
    """
    test_query = "test query for integration"
    test_api_key = "sk-testkey"
    test_subset = "train[:10]"
    test_index_path_base = "test_main_idx" # Corresponds to bm25_index_name in main()
    # Expected index path constructed in main: f"./{test_index_path_base}_{test_subset.replace('[:','_').replace(']','').replace('%','percent')}"
    expected_final_index_path = f"./{test_index_path_base}_train_10"


    main(
        query=test_query,
        openai_api_key_arg=test_api_key,
        msmarco_subset_str=test_subset,
        bm25_index_name=test_index_path_base,
        # Other params will use defaults from main() signature
    )

    # Verify calls to utility functions
    mock_pipeline_components["load_dependencies_info"].assert_called_once()
    mock_pipeline_components["load_msmarco_dataset"].assert_called_once_with(subset=test_subset)
    mock_pipeline_components["preprocess_documents"].assert_called_once_with(mock_pipeline_components["load_msmarco_dataset"].return_value)

    # Verify BM25Retriever instantiation and usage
    mock_pipeline_components["BM25Retriever"].assert_called_once_with(
        index_path=expected_final_index_path,
        documents=mock_processed_docs_dict,
        force_reindex=False # Default from main signature
    )
    mock_pipeline_components["bm25_retriever_instance"].retrieve.assert_called_once_with(
        test_query,
        top_k=100 # Default from main signature
    )

    # Verify CrossEncoderReRanker instantiation and usage
    mock_pipeline_components["CrossEncoderReRanker"].assert_called_once_with(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2" # Default from main / constants
    )
    mock_pipeline_components["cross_encoder_reranker_instance"].rerank.assert_called_once_with(
        test_query,
        mock_bm25_results_list,
        top_n=20 # Default from main signature
    )

    # Verify LLMSynthesizer instantiation and usage
    mock_pipeline_components["LLMSynthesizer"].assert_called_once_with(
        openai_api_key=test_api_key,
        model_name="gpt-3.5-turbo" # Default from main / constants
    )
    mock_pipeline_components["llm_synthesizer_instance"].synthesize.assert_called_once_with(
        test_query,
        mock_reranked_results_list,
        top_n_docs_for_llm=3 # Default from main signature
    )

    # Verify logging output for final answer
    mock_pipeline_components["logger"].info.assert_any_call("\n--- Final Generated Answer ---")
    mock_pipeline_components["logger"].info.assert_any_call(mock_llm_answer_str)


def test_main_pipeline_bm25_fails_to_init(mock_pipeline_components):
    """Test main() when BM25Retriever fails to initialize."""
    mock_pipeline_components["BM25Retriever"].side_effect = Exception("BM25 init error")

    main(query="test query") # Call with minimal args

    mock_pipeline_components["logger"].critical.assert_any_call(
        "BM25Retriever initialization or retrieval failed: BM25 init error. Exiting pipeline.", exc_info=True
    )
    # Ensure subsequent components are not called
    mock_pipeline_components["CrossEncoderReRanker"].assert_not_called()
    mock_pipeline_components["LLMSynthesizer"].assert_not_called()


def test_main_pipeline_bm25_returns_no_results(mock_pipeline_components, mock_processed_docs_dict, mock_llm_answer_str):
    """Test main() when BM25Retriever returns no results."""
    mock_pipeline_components["bm25_retriever_instance"].retrieve.return_value = [] # No BM25 results

    # We still expect reranker to be called, but it should handle empty input gracefully
    # And LLM to be called, but it should also handle empty input for context
    mock_pipeline_components["cross_encoder_reranker_instance"].rerank.return_value = []
    mock_pipeline_components["llm_synthesizer_instance"].synthesize.return_value = "No relevant documents found to synthesize an answer."


    main(query="test query", openai_api_key_arg="key")

    mock_pipeline_components["bm25_retriever_instance"].retrieve.assert_called_once()
    mock_pipeline_components["logger"].warning.assert_any_call("BM25 returned no results. Further processing may yield no answer.")

    # Reranker's rerank method should still be called, even with an empty list
    mock_pipeline_components["cross_encoder_reranker_instance"].rerank.assert_called_once_with(
        "test query",
        [], # Empty list from BM25
        top_n=20 # Default
    )
    mock_pipeline_components["logger"].info.assert_any_call("Skipping re-ranking as BM25 returned no documents.") # This log is from main

    # LLM Synthesizer should be initialized
    mock_pipeline_components["LLMSynthesizer"].assert_called_once()
    # Synthesize should NOT be called if reranked_docs_list is empty (as per logic in main)
    # The synthesize method itself might be called if there's a fallback to BM25 results that are also empty.
    # Based on current `main` logic: if reranked_docs_list is empty, synthesize is not called.
    # The final answer is set to "No relevant documents..."
    mock_pipeline_components["llm_synthesizer_instance"].synthesize.assert_not_called()
    mock_pipeline_components["logger"].warning.assert_any_call("No documents available for LLM synthesis (either BM25 or reranker yielded no results).")
    mock_pipeline_components["logger"].info.assert_any_call("No relevant documents found to synthesize an answer.")


def test_main_pipeline_reranker_fails(mock_pipeline_components, mock_bm25_results_list, mock_llm_answer_str):
    """Test main() when CrossEncoderReRanker fails to initialize or rerank."""
    mock_pipeline_components["CrossEncoderReRanker"].side_effect = Exception("Reranker init error")
    # Fallback in main means it will use BM25 results for LLM if reranker fails

    main(query="test query", openai_api_key_arg="key")

    mock_pipeline_components["logger"].error.assert_any_call(
        "CrossEncoderReRanker initialization or reranking failed: Reranker init error. Proceeding without re-ranking.", exc_info=True
    )
    # LLM should still be called, but with BM25 results (or a slice of them)
    mock_pipeline_components["LLMSynthesizer"].assert_called_once()
    # The reranked_docs_list in main becomes a slice of bm25_results
    expected_fallback_docs_for_llm = mock_bm25_results_list[:20] # Default rerank_top_n_count
    mock_pipeline_components["llm_synthesizer_instance"].synthesize.assert_called_once_with(
        "test query",
        expected_fallback_docs_for_llm, # Fallback docs
        top_n_docs_for_llm=3 # Default
    )


def test_main_pipeline_llm_synthesizer_fails_init(mock_pipeline_components, mock_reranked_results_list):
    """Test main() when LLMSynthesizer fails to initialize (e.g. no API key)."""
    mock_pipeline_components["LLMSynthesizer"].side_effect = ValueError("LLM init error (e.g. API key)")

    main(query="test query") # No API key passed, and assume not in env for this test path

    mock_pipeline_components["logger"].error.assert_any_call(
        "LLMSynthesizer setup failed: LLM init error (e.g. API key). Cannot synthesize answer.", exc_info=True
    )
    # Ensure synthesize method is not called on the instance if constructor fails
    mock_pipeline_components["llm_synthesizer_instance"].synthesize.assert_not_called()
    mock_pipeline_components["logger"].info.assert_any_call("LLM setup failed: LLM init error (e.g. API key)")


def test_main_pipeline_llm_synthesize_method_fails(mock_pipeline_components, mock_reranked_results_list):
    """Test main() when LLMSynthesizer.synthesize method fails."""
    mock_pipeline_components["llm_synthesizer_instance"].synthesize.side_effect = Exception("LLM API call error")

    main(query="test query", openai_api_key_arg="key")

    mock_pipeline_components["logger"].error.assert_any_call(
        "LLM synthesis failed: LLM API call error", exc_info=True
    )
    mock_pipeline_components["logger"].info.assert_any_call("LLM synthesis encountered an error: LLM API call error")

# These tests provide good coverage of the orchestration logic in main().
# They ensure that components are called in the correct order and that
# basic error conditions or alternative paths within main() are handled.
# They do NOT test the internal logic of the individual classes, as those
# are covered by their respective unit test files.
