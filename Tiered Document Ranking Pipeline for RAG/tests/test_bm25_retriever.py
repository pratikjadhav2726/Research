"""
Unit tests for the BM25Retriever class in tiered_ranking_pipeline.py
"""
import pytest
import os
from unittest.mock import patch, MagicMock, call # call for checking multiple calls

# Import the class to be tested
from ..tiered_ranking_pipeline import BM25Retriever

@pytest.fixture
def mock_documents_fixture():
    """Provides a sample documents dictionary."""
    return {
        "doc1": "This is the first document content.",
        "doc2": "Content for the second document is here.",
        "doc3": "The third document has some text."
    }

@pytest.fixture
def bm25_retriever_mocks(mocker):
    """Centralized mocks for BM25Retriever tests."""
    # Mock os.path functions
    mocker.patch('os.path.exists', return_value=False) # Default: index does not exist
    mocker.patch('os.path.dirname', return_value='./test_index_dir') # Mock dirname
    mocker.patch('os.makedirs') # Mock makedirs
    mocker.patch('os.remove') # Mock remove for cleanup jsonl or force_reindex
    mocker.patch('shutil.rmtree') # Mock rmtree for force_reindex on directory

    # Mock Pyserini LuceneSearcher
    mock_lucene_searcher_instance = MagicMock()
    mock_lucene_searcher_constructor = mocker.patch('tiered_ranking_pipeline.LuceneSearcher', return_value=mock_lucene_searcher_instance)

    # Mock subprocess.run for Pyserini indexing command
    mock_subprocess_run = mocker.patch('subprocess.run')
    mock_subprocess_run.return_value = MagicMock(check_returncode=lambda: None, stdout="Indexing successful", stderr="")

    # Mock open for writing JSONL file
    mock_open = mocker.patch('builtins.open', mocker.mock_open())

    return {
        "mock_os_exists": mocker. डबल('os.path.exists'), # Get the patched object
        "mock_os_makedirs": mocker.ubble('os.makedirs'),
        "mock_os_remove": mocker.ubble('os.remove'),
        "mock_shutil_rmtree": mocker.ubble('shutil.rmtree'),
        "mock_lucene_searcher_constructor": mock_lucene_searcher_constructor,
        "mock_lucene_searcher_instance": mock_lucene_searcher_instance,
        "mock_subprocess_run": mock_subprocess_run,
        "mock_open": mock_open,
        "logger": mocker.patch('tiered_ranking_pipeline.logger') # Mock the logger used in the class
    }

# --- Tests for BM25Retriever.__init__ ---
def test_bm25retriever_init_index_creation(bm25_retriever_mocks, mock_documents_fixture, tmp_path):
    """Test BM25Retriever initialization when index needs to be created."""
    index_path = str(tmp_path / "new_index")
    bm25_retriever_mocks["mock_os_exists"].return_value = False # Index does not exist

    retriever = BM25Retriever(index_path=index_path, documents=mock_documents_fixture)

    # Assert JSONL file was written
    jsonl_path = os.path.join(os.path.dirname(index_path), f"pyserini_docs_for_{os.path.basename(index_path)}.jsonl")
    bm25_retriever_mocks["mock_open"].assert_called_once_with(jsonl_path, 'w', encoding='utf-8')
    # Example check for content written (can be more specific)
    # Get all write calls: bm25_retriever_mocks["mock_open"]().write.call_args_list
    # For 3 documents, 3 calls to write json.dumps(...) + "\n"
    assert bm25_retriever_mocks["mock_open"]().write.call_count == len(mock_documents_fixture)


    # Assert Pyserini indexing command was called
    bm25_retriever_mocks["mock_subprocess_run"].assert_called_once()
    args, _ = bm25_retriever_mocks["mock_subprocess_run"].call_args
    assert jsonl_path in args[0] # Check if jsonl_path is in the command
    assert index_path in args[0] # Check if index_path is in the command

    # Assert LuceneSearcher was initialized
    bm25_retriever_mocks["mock_lucene_searcher_constructor"].assert_called_once_with(index_path)
    assert retriever.searcher is not None
    bm25_retriever_mocks["logger"].info.assert_any_call(f"BM25 index not found at '{index_path}'. Creating a new one...")
    bm25_retriever_mocks["logger"].info.assert_any_call(f"BM25 index created successfully at '{index_path}'.")
    bm25_retriever_mocks["mock_os_remove"].assert_called_with(jsonl_path) # Cleanup of jsonl

def test_bm25retriever_init_index_exists(bm25_retriever_mocks, mock_documents_fixture, tmp_path):
    """Test BM25Retriever initialization when index already exists."""
    index_path = str(tmp_path / "existing_index")
    bm25_retriever_mocks["mock_os_exists"].return_value = True # Index exists

    retriever = BM25Retriever(index_path=index_path, documents=mock_documents_fixture)

    # Assert that indexing (subprocess.run, open) was NOT called
    bm25_retriever_mocks["mock_subprocess_run"].assert_not_called()
    bm25_retriever_mocks["mock_open"].assert_not_called()

    # Assert LuceneSearcher was initialized
    bm25_retriever_mocks["mock_lucene_searcher_constructor"].assert_called_once_with(index_path)
    assert retriever.searcher is not None
    bm25_retriever_mocks["logger"].info.assert_any_call(f"Loading existing BM25 index from '{index_path}'.")

def test_bm25retriever_init_force_reindex_directory(bm25_retriever_mocks, mock_documents_fixture, tmp_path):
    """Test BM25Retriever with force_reindex=True for a directory index."""
    index_path = str(tmp_path / "reindex_dir")
    bm25_retriever_mocks["mock_os_exists"].side_effect = [True, False] # Exists then doesn't after removal
    bm25_retriever_mocks["mocker"].patch('os.path.isdir', return_value=True) # Simulate index_path is a directory

    retriever = BM25Retriever(index_path=index_path, documents=mock_documents_fixture, force_reindex=True)

    bm25_retriever_mocks["mock_shutil_rmtree"].assert_called_once_with(index_path)
    bm25_retriever_mocks["mock_os_remove"].assert_not_called() # Not called if it's a directory

    # Check if indexing proceeds after removal
    bm25_retriever_mocks["mock_subprocess_run"].assert_called_once()
    bm25_retriever_mocks["mock_lucene_searcher_constructor"].assert_called_once_with(index_path)
    bm25_retriever_mocks["logger"].info.assert_any_call(f"Force re-indexing: Removing existing index at {index_path}")

def test_bm25retriever_init_force_reindex_file(bm25_retriever_mocks, mock_documents_fixture, tmp_path):
    """Test BM25Retriever with force_reindex=True for a file-based index component (if applicable)."""
    index_path = str(tmp_path / "reindex_file") # Or a specific file if Pyserini creates single file indices
    bm25_retriever_mocks["mock_os_exists"].side_effect = [True, False]
    bm25_retriever_mocks["mocker"].patch('os.path.isdir', return_value=False) # Simulate index_path is a file

    retriever = BM25Retriever(index_path=index_path, documents=mock_documents_fixture, force_reindex=True)

    bm25_retriever_mocks["mock_os_remove"].assert_called_once_with(index_path)
    bm25_retriever_mocks["mock_shutil_rmtree"].assert_not_called()

    bm25_retriever_mocks["mock_subprocess_run"].assert_called_once()
    bm25_retriever_mocks["mock_lucene_searcher_constructor"].assert_called_once_with(index_path)
    bm25_retriever_mocks["logger"].info.assert_any_call(f"Force re-indexing: Removing existing index at {index_path}")


def test_bm25retriever_init_index_creation_fails_command(bm25_retriever_mocks, mock_documents_fixture, tmp_path):
    """Test index creation failure due to Pyserini command error."""
    index_path = str(tmp_path / "fail_index")
    bm25_retriever_mocks["mock_os_exists"].return_value = False
    bm25_retriever_mocks["mock_subprocess_run"].side_effect = subprocess.CalledProcessError(1, "cmd", stderr="Pyserini error")

    with pytest.raises(Exception, match="Pyserini indexing failed."):
        BM25Retriever(index_path=index_path, documents=mock_documents_fixture)

    bm25_retriever_mocks["logger"].error.assert_any_call("Pyserini indexing command failed with return code 1.", exc_info=True)

def test_bm25retriever_init_index_creation_fails_no_dir_after_cmd(bm25_retriever_mocks, mock_documents_fixture, tmp_path):
    """Test index creation failure if directory not found after Pyserini command."""
    index_path = str(tmp_path / "no_dir_index")

    # Simulate os.path.exists for index_path:
    # 1. False (initial check, triggers creation)
    # 2. False (check after subprocess.run, means index not created by Pyserini)
    bm25_retriever_mocks["mock_os_exists"].side_effect = [False, False]

    with pytest.raises(Exception, match=f"Pyserini index creation failed for {index_path}"):
        BM25Retriever(index_path=index_path, documents=mock_documents_fixture)

    bm25_retriever_mocks["logger"].error.assert_any_call(f"Index creation failed: Index directory {index_path} not found after Pyserini command.")


# --- Tests for BM25Retriever.retrieve ---
def test_bm25retriever_retrieve_success(bm25_retriever_mocks, mock_documents_fixture, tmp_path):
    """Test successful document retrieval."""
    index_path = str(tmp_path / "retrieve_idx")
    bm25_retriever_mocks["mock_os_exists"].return_value = True # Assume index exists

    retriever = BM25Retriever(index_path=index_path, documents=mock_documents_fixture)

    # Mock searcher.search results
    mock_hit1 = MagicMock(docid="doc1", score=0.9)
    mock_hit2 = MagicMock(docid="doc2", score=0.8)
    bm25_retriever_mocks["mock_lucene_searcher_instance"].search.return_value = [mock_hit1, mock_hit2]

    query = "test query"
    top_k = 2
    results = retriever.retrieve(query, top_k=top_k)

    bm25_retriever_mocks["mock_lucene_searcher_instance"].search.assert_called_once_with(query, k=top_k)
    assert len(results) == 2
    assert results[0] == ("doc1", 0.9, mock_documents_fixture["doc1"])
    assert results[1] == ("doc2", 0.8, mock_documents_fixture["doc2"])
    bm25_retriever_mocks["logger"].info.assert_any_call(f"Retrieved {len(results)} documents with BM25.")

def test_bm25retriever_retrieve_no_searcher(bm25_retriever_mocks, mock_documents_fixture, tmp_path, caplog):
    """Test retrieval when searcher is not initialized (e.g., init failed)."""
    # Simulate searcher loading failure by making constructor raise an error
    bm25_retriever_mocks["mock_os_exists"].return_value = True # Index exists
    bm25_retriever_mocks["mock_lucene_searcher_constructor"].side_effect = Exception("Failed to load searcher")

    with pytest.raises(Exception, match="Failed to load searcher"): # Exception from __init__
        retriever = BM25Retriever(index_path=str(tmp_path / "fail_load"), documents=mock_documents_fixture)
        # If __init__ itself fails, an instance isn't created, so can't call retrieve.
        # This test setup is more about checking the retrieve path if searcher was None.
        # To test retrieve directly with searcher=None:
        # retriever = BM25Retriever(...)
        # retriever.searcher = None # Manually set for test
        # results = retriever.retrieve("query")
        # assert results == []
        # bm25_retriever_mocks["logger"].error.assert_any_call("BM25 searcher not initialized. Cannot retrieve.")

    # Test the direct path where searcher is None
    # This requires a successfully initialized retriever object first
    index_path = str(tmp_path / "retrieve_idx_searcher_none")
    bm25_retriever_mocks["mock_os_exists"].return_value = True
    bm25_retriever_mocks["mock_lucene_searcher_constructor"].side_effect = None # Reset side effect
    bm25_retriever_mocks["mock_lucene_searcher_constructor"].return_value = bm25_retriever_mocks["mock_lucene_searcher_instance"]


    retriever_instance = BM25Retriever(index_path=index_path, documents=mock_documents_fixture)
    retriever_instance.searcher = None # Manually set searcher to None

    results = retriever_instance.retrieve("some query")
    assert results == []
    bm25_retriever_mocks["logger"].error.assert_any_call("BM25 searcher not initialized. Cannot retrieve.")


def test_bm25retriever_retrieve_doc_text_not_found(bm25_retriever_mocks, mock_documents_fixture, tmp_path):
    """Test retrieval when a doc_id from search results is not in the local documents_map."""
    index_path = str(tmp_path / "retrieve_missing_doc_idx")
    bm25_retriever_mocks["mock_os_exists"].return_value = True

    retriever = BM25Retriever(index_path=index_path, documents=mock_documents_fixture)

    mock_hit_unknown = MagicMock(docid="unknown_doc", score=0.7)
    bm25_retriever_mocks["mock_lucene_searcher_instance"].search.return_value = [mock_hit_unknown]

    results = retriever.retrieve("query", top_k=1)

    assert len(results) == 1
    assert results[0][0] == "unknown_doc"
    assert results[0][2] == "Document text not found in preloaded dictionary."

# Note: Pyserini's LuceneSearcher might raise Java-related errors if not properly configured.
# These tests mock out LuceneSearcher, so they won't catch actual Pyserini/Java issues.
# Integration tests with a real (but small) Pyserini index would be needed for that.
# The current tests focus on the Python logic within BM25Retriever.
