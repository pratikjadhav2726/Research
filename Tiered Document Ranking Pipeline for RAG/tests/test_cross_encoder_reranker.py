"""
Unit tests for the CrossEncoderReRanker class in tiered_ranking_pipeline.py
"""
import pytest
from unittest.mock import patch, MagicMock
import torch

# Import the class to be tested
from ..tiered_ranking_pipeline import CrossEncoderReRanker, DEFAULT_CROSS_ENCODER_MODEL

@pytest.fixture
def mock_bm25_results_fixture():
    """Provides sample BM25 results."""
    return [
        ("doc1", 0.9, "Text of document 1"),
        ("doc2", 0.8, "Text of document 2, which is relevant."),
        ("doc3", 0.7, "Text of document 3, less relevant."),
        ("doc4", 0.6, "Text of document 4"),
    ]

@pytest.fixture
def cross_encoder_mocks(mocker):
    """Centralized mocks for CrossEncoderReRanker tests."""
    # Mock sentence_transformers.CrossEncoder
    mock_cross_encoder_instance = MagicMock()
    # Simulate predict method, which should return a list or numpy array of scores
    mock_cross_encoder_instance.predict.return_value = [0.1, 0.9, 0.5, 0.3] # Example scores

    mock_cross_encoder_constructor = mocker.patch(
        'tiered_ranking_pipeline.CrossEncoder', # Path to CrossEncoder in the module being tested
        return_value=mock_cross_encoder_instance
    )

    # Mock torch.cuda.is_available
    mock_cuda_available = mocker.patch('torch.cuda.is_available', return_value=False) # Default to CPU

    return {
        "mock_cross_encoder_constructor": mock_cross_encoder_constructor,
        "mock_cross_encoder_instance": mock_cross_encoder_instance,
        "mock_cuda_available": mock_cuda_available,
        "logger": mocker.patch('tiered_ranking_pipeline.logger') # Mock logger
    }

# --- Tests for CrossEncoderReRanker.__init__ ---
def test_crossencoder_init_default_model(cross_encoder_mocks):
    """Test CrossEncoderReRanker initialization with the default model."""
    reranker = CrossEncoderReRanker()

    cross_encoder_mocks["mock_cross_encoder_constructor"].assert_called_once_with(
        DEFAULT_CROSS_ENCODER_MODEL,
        device='cpu' # Because mock_cuda_available returns False
    )
    assert reranker.model is not None
    assert reranker.model_name == DEFAULT_CROSS_ENCODER_MODEL
    assert reranker.device == 'cpu'
    cross_encoder_mocks["logger"].info.assert_any_call(f"Initializing CrossEncoderReRanker with model: {DEFAULT_CROSS_ENCODER_MODEL} on device: cpu")

def test_crossencoder_init_custom_model(cross_encoder_mocks):
    """Test CrossEncoderReRanker initialization with a custom model name."""
    custom_model = "custom/model-name"
    reranker = CrossEncoderReRanker(model_name=custom_model)

    cross_encoder_mocks["mock_cross_encoder_constructor"].assert_called_once_with(
        custom_model,
        device='cpu'
    )
    assert reranker.model_name == custom_model
    cross_encoder_mocks["logger"].info.assert_any_call(f"Initializing CrossEncoderReRanker with model: {custom_model} on device: cpu")


def test_crossencoder_init_cuda_available(cross_encoder_mocks):
    """Test CrossEncoderReRanker initialization when CUDA is available."""
    cross_encoder_mocks["mock_cuda_available"].return_value = True # Simulate CUDA available

    reranker = CrossEncoderReRanker()

    cross_encoder_mocks["mock_cross_encoder_constructor"].assert_called_once_with(
        DEFAULT_CROSS_ENCODER_MODEL,
        device='cuda'
    )
    assert reranker.device == 'cuda'
    cross_encoder_mocks["logger"].info.assert_any_call(f"Initializing CrossEncoderReRanker with model: {DEFAULT_CROSS_ENCODER_MODEL} on device: cuda")

def test_crossencoder_init_model_loading_fails(cross_encoder_mocks):
    """Test CrossEncoderReRanker initialization when model loading fails."""
    cross_encoder_mocks["mock_cross_encoder_constructor"].side_effect = Exception("Model load failed")

    with pytest.raises(Exception, match="Model load failed"):
        CrossEncoderReRanker()
    cross_encoder_mocks["logger"].error.assert_any_call(f"Failed to load Cross-Encoder model '{DEFAULT_CROSS_ENCODER_MODEL}': Model load failed", exc_info=True)


# --- Tests for CrossEncoderReRanker.rerank ---
def test_crossencoder_rerank_success(cross_encoder_mocks, mock_bm25_results_fixture):
    """Test successful re-ranking of documents."""
    reranker = CrossEncoderReRanker() # Uses mocked constructor

    # Mock scores that would be returned by the predict method, corresponding to mock_bm25_results_fixture
    # Original order: doc1, doc2, doc3, doc4
    # Original scores (BM25 like): 0.9, 0.8, 0.7, 0.6
    # New CrossEncoder scores: doc1=0.1, doc2=0.9, doc3=0.5, doc4=0.3
    # Expected sorted order by CE score (desc): doc2, doc3, doc4, doc1

    # The mock_cross_encoder_instance.predict is already set up in cross_encoder_mocks
    # to return [0.1, 0.9, 0.5, 0.3]

    query = "test query"
    top_n = 3
    results = reranker.rerank(query, mock_bm25_results_fixture, top_n=top_n)

    # Verify that predict was called with the correct sentence pairs
    expected_pairs = [[query, doc_text] for _, _, doc_text in mock_bm25_results_fixture]
    cross_encoder_mocks["mock_cross_encoder_instance"].predict.assert_called_once_with(
        expected_pairs,
        show_progress_bar=True
    )

    assert len(results) == top_n
    # Check for correct sorting and content
    assert results[0][0] == "doc2" # doc_id of highest CE score
    assert results[0][1] == 0.9    # CE score
    assert results[0][2] == "Text of document 2, which is relevant." # doc_text

    assert results[1][0] == "doc3"
    assert results[1][1] == 0.5

    assert results[2][0] == "doc4" # Note: If scores were [0.1, 0.9, 0.5, 0.6], then doc4 would be higher than doc1
                                   # With scores [0.1, 0.9, 0.5, 0.3], doc4 is next.
    assert results[2][1] == 0.3

    cross_encoder_mocks["logger"].info.assert_any_call(f"Re-ranked and selected top {top_n} documents.")

def test_crossencoder_rerank_empty_input(cross_encoder_mocks):
    """Test re-ranking with an empty list of BM25 results."""
    reranker = CrossEncoderReRanker()
    results = reranker.rerank("test query", [], top_n=5)

    assert results == []
    cross_encoder_mocks["mock_cross_encoder_instance"].predict.assert_not_called()
    cross_encoder_mocks["logger"].warning.assert_any_call("No BM25 results provided to re-rank. Returning empty list.")

def test_crossencoder_rerank_top_n_greater_than_results(cross_encoder_mocks, mock_bm25_results_fixture):
    """Test re-ranking when top_n is larger than the number of input documents."""
    reranker = CrossEncoderReRanker()
    # predict will return 4 scores based on mock_bm25_results_fixture

    top_n = 10 # More than the 4 documents available
    results = reranker.rerank("test query", mock_bm25_results_fixture, top_n=top_n)

    assert len(results) == len(mock_bm25_results_fixture) # Should return all available, sorted
    assert results[0][0] == "doc2" # Still sorted correctly
    cross_encoder_mocks["logger"].info.assert_any_call(f"Re-ranked and selected top {len(mock_bm25_results_fixture)} documents.")

def test_crossencoder_rerank_prediction_error(cross_encoder_mocks, mock_bm25_results_fixture):
    """Test re-ranking when the model's predict method raises an error."""
    reranker = CrossEncoderReRanker()
    cross_encoder_mocks["mock_cross_encoder_instance"].predict.side_effect = Exception("Prediction failed")

    results = reranker.rerank("test query", mock_bm25_results_fixture, top_n=5)

    assert results == [] # Should return empty list or handle error as defined
    cross_encoder_mocks["logger"].error.assert_any_call("Error during Cross-Encoder prediction: Prediction failed", exc_info=True)
