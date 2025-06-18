"""
Unit tests for data handling functions in tiered_ranking_pipeline.py
(load_msmarco_dataset, preprocess_documents)
"""
import pytest
from unittest.mock import patch, MagicMock

# Import functions to be tested
from ..tiered_ranking_pipeline import load_msmarco_dataset, preprocess_documents

# A fixture to create a mock Hugging Face Dataset object
@pytest.fixture
def mock_hf_dataset_fixture():
    """Creates a mock Hugging Face Dataset object."""
    dataset = MagicMock()
    # Simulate it being iterable and having a __len__ method
    dataset.__len__.return_value = 2 # Example: 2 records
    mock_record_1_passages = ['passage text 1 from record 1', 'passage text 2 from record 1']
    mock_record_2_passages = ['passage text 1 from record 2']

    dataset.__iter__.return_value = [
        {
            'query_id': 'q1',
            'passages': {
                'passage_text': mock_record_1_passages,
                'is_selected': [0, 0], # Dummy data
                'url': ['url1', 'url2'] # Dummy data
            },
            # Add other fields if your preprocess_documents function uses them
        },
        {
            'query_id': 'q2',
            'passages': {
                'passage_text': mock_record_2_passages,
                'is_selected': [1],
                'url': ['url3']
            },
        }
    ]
    return dataset

@pytest.fixture
def mock_empty_hf_dataset_fixture():
    """Creates an empty mock Hugging Face Dataset object."""
    dataset = MagicMock()
    dataset.__len__.return_value = 0
    dataset.__iter__.return_value = []
    return dataset

# Tests for load_msmarco_dataset
@patch('tiered_ranking_pipeline.load_dataset') # Patch where load_dataset is LOOKED UP
def test_load_msmarco_dataset_success(mock_hf_load_dataset, mock_hf_dataset_fixture, caplog):
    """Test successful loading of MSMARCO dataset."""
    mock_hf_load_dataset.return_value = mock_hf_dataset_fixture
    subset_str = "train[:1%]"

    logger = MagicMock() # If tiered_ranking_pipeline.logger is used
    with patch('tiered_ranking_pipeline.logger', logger):
        result_dataset = load_msmarco_dataset(subset=subset_str)

    mock_hf_load_dataset.assert_called_once_with("ms_marco", "v1.1", split=subset_str)
    assert result_dataset is not None
    assert len(result_dataset) == 2 # As defined in mock_hf_dataset_fixture
    logger.info.assert_any_call(f"Attempting to load MSMARCO dataset subset: {subset_str}...")
    logger.info.assert_any_call(f"Dataset '{subset_str}' loaded successfully. Number of examples: {len(mock_hf_dataset_fixture)}")


@patch('tiered_ranking_pipeline.load_dataset')
def test_load_msmarco_dataset_file_not_found(mock_hf_load_dataset, caplog):
    """Test FileNotFoundError during dataset loading."""
    mock_hf_load_dataset.side_effect = FileNotFoundError("Mocked: MSMARCO data files not found")
    subset_str = "train[:1%]"

    logger = MagicMock()
    with patch('tiered_ranking_pipeline.logger', logger):
        result_dataset = load_msmarco_dataset(subset=subset_str)

    assert result_dataset is None
    logger.error.assert_any_call(f"Dataset loading failed: MSMARCO data files not found. Please ensure dataset is available. Error: Mocked: MSMARCO data files not found")

@patch('tiered_ranking_pipeline.load_dataset')
def test_load_msmarco_dataset_other_exception(mock_hf_load_dataset, caplog):
    """Test other exceptions during dataset loading."""
    mock_hf_load_dataset.side_effect = Exception("Mocked: Some other Hugging Face error")
    subset_str = "train[:1%]"

    logger = MagicMock()
    with patch('tiered_ranking_pipeline.logger', logger):
        result_dataset = load_msmarco_dataset(subset=subset_str)

    assert result_dataset is None
    logger.error.assert_any_call(f"An unexpected error occurred while loading the MSMARCO dataset: Mocked: Some other Hugging Face error")


# Tests for preprocess_documents
def test_preprocess_documents_success(mock_hf_dataset_fixture, caplog):
    """Test successful preprocessing of documents."""
    logger = MagicMock()
    with patch('tiered_ranking_pipeline.logger', logger):
        processed_docs = preprocess_documents(mock_hf_dataset_fixture)

    # Expected: 2 passages from record 1, 1 passage from record 2 = 3 total documents
    assert len(processed_docs) == 3
    assert processed_docs["0"] == 'passage text 1 from record 1'
    assert processed_docs["1"] == 'passage text 2 from record 1'
    assert processed_docs["2"] == 'passage text 1 from record 2'
    logger.info.assert_any_call("Preprocessing documents (extracting passages)...")
    logger.info.assert_any_call(f"Preprocessed {len(processed_docs)} non-empty documents (passages).")

def test_preprocess_documents_empty_dataset(mock_empty_hf_dataset_fixture, caplog):
    """Test preprocessing with an empty dataset."""
    logger = MagicMock()
    with patch('tiered_ranking_pipeline.logger', logger):
        processed_docs = preprocess_documents(mock_empty_hf_dataset_fixture)

    assert len(processed_docs) == 0
    logger.info.assert_any_call(f"Preprocessed 0 non-empty documents (passages).")

def test_preprocess_documents_missing_fields(caplog):
    """Test preprocessing with records missing 'passages' or 'passage_text'."""
    mock_dataset_missing_fields = MagicMock()
    mock_dataset_missing_fields.__len__.return_value = 2
    mock_dataset_missing_fields.__iter__.return_value = [
        {'query_id': 'q1'}, # Missing 'passages'
        {'query_id': 'q2', 'passages': {'no_passage_text_field': []}} # Missing 'passage_text'
    ]

    logger = MagicMock()
    with patch('tiered_ranking_pipeline.logger', logger):
        processed_docs = preprocess_documents(mock_dataset_missing_fields)

    assert len(processed_docs) == 0
    # Check that warnings were logged for missing fields
    logger.warning.assert_any_call("Record missing 'passages' or 'passage_text'. Query ID: q1")
    logger.warning.assert_any_call("Record missing 'passages' or 'passage_text'. Query ID: q2")

def test_preprocess_documents_empty_passage_text(caplog):
    """Test preprocessing handles empty or whitespace-only passage texts."""
    mock_dataset_empty_passage = MagicMock()
    mock_dataset_empty_passage.__len__.return_value = 1
    mock_dataset_empty_passage.__iter__.return_value = [
        {
            'query_id': 'q1',
            'passages': {
                'passage_text': ["", "   ", "actual passage 1"],
            }
        }
    ]
    logger = MagicMock()
    with patch('tiered_ranking_pipeline.logger', logger):
        processed_docs = preprocess_documents(mock_dataset_empty_passage)

    assert len(processed_docs) == 1 # Only "actual passage 1" should be included
    assert "0" in processed_docs
    assert processed_docs["0"] == "actual passage 1"
    logger.info.assert_any_call(f"Preprocessed 1 non-empty documents (passages).")

# To run these tests, navigate to the directory containing `tiered_ranking_pipeline.py`
# and run: python -m pytest
# (Assuming pytest is installed and the file structure is correct)
# Tiered Document Ranking Pipeline for RAG/
# |-- tiered_ranking_pipeline.py
# |-- tests/
#     |-- __init__.py
#     |-- test_data_handling.py
#     ... (other test files)
