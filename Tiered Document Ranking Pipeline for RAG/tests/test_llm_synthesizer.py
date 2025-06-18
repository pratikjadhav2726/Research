"""
Unit tests for the LLMSynthesizer class in tiered_ranking_pipeline.py
"""
import pytest
import os
from unittest.mock import patch, MagicMock

# Import the class to be tested
from ..tiered_ranking_pipeline import LLMSynthesizer, DEFAULT_LLM_MODEL

@pytest.fixture
def mock_reranked_docs_fixture():
    """Provides sample re-ranked documents."""
    return [
        ("doc2", 0.9, "Relevant text from document 2."),
        ("doc3", 0.5, "Moderately relevant text from document 3."),
        ("doc1", 0.1, "Less relevant text from document 1."),
    ]

@pytest.fixture
def llm_synthesizer_mocks(mocker):
    """Centralized mocks for LLMSynthesizer tests."""
    # Mock os.getenv for API key
    mock_os_getenv = mocker.patch('os.getenv')
    mock_os_getenv.return_value = None # Default: API key not in env

    # Mock openai.ChatCompletion.create
    mock_chat_completion_instance = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message = {'content': "Mocked LLM answer."}
    mock_chat_completion_instance.choices = [mock_choice]

    mock_openai_chatcompletion_create = mocker.patch(
        'openai.ChatCompletion.create', # Path to openai.ChatCompletion.create
        return_value=mock_chat_completion_instance
    )

    # Mock openai.api_key setting (though the library itself handles it)
    # We are more interested in whether our class tries to set it.
    # We can also check by asserting openai.api_key value after __init__
    mocker.patch('openai.api_key', None) # Reset before each test if needed

    return {
        "mock_os_getenv": mock_os_getenv,
        "mock_openai_chatcompletion_create": mock_openai_chatcompletion_create,
        "logger": mocker.patch('tiered_ranking_pipeline.logger') # Mock logger
    }

# --- Tests for LLMSynthesizer.__init__ ---

def test_llmsynthesizer_init_with_api_key_arg(llm_synthesizer_mocks):
    """Test initialization with API key provided as an argument."""
    api_key = "sk-argkey123"
    synthesizer = LLMSynthesizer(openai_api_key=api_key, model_name="gpt-test")

    assert synthesizer.api_key == api_key
    assert synthesizer.model_name == "gpt-test"
    # Check if openai.api_key was set by our class (it is set globally by openai lib)
    # This assertion might be tricky if openai lib internals change.
    # Better to trust the lib sets it if openai.api_key = self.api_key is called.
    # For now, assume it's set if no error.
    llm_synthesizer_mocks["logger"].info.assert_any_call("LLMSynthesizer initialized with model: gpt-test.")
    llm_synthesizer_mocks["mock_os_getenv"].assert_not_called() # Should not call getenv if key is provided

def test_llmsynthesizer_init_with_api_key_env(llm_synthesizer_mocks):
    """Test initialization with API key from environment variable."""
    env_api_key = "sk-envkey456"
    llm_synthesizer_mocks["mock_os_getenv"].return_value = env_api_key

    synthesizer = LLMSynthesizer(model_name="gpt-env-test")

    assert synthesizer.api_key == env_api_key
    assert synthesizer.model_name == "gpt-env-test"
    llm_synthesizer_mocks["mock_os_getenv"].assert_called_once_with("OPENAI_API_KEY")
    llm_synthesizer_mocks["logger"].info.assert_any_call("LLMSynthesizer initialized with model: gpt-env-test.")

def test_llmsynthesizer_init_no_api_key_raises_valueerror(llm_synthesizer_mocks):
    """Test ValueError is raised if API key is not provided and not in env."""
    llm_synthesizer_mocks["mock_os_getenv"].return_value = None # Ensure it's None

    with pytest.raises(ValueError, match="OpenAI API key is required for LLMSynthesizer."):
        LLMSynthesizer()
    llm_synthesizer_mocks["logger"].error.assert_any_call("OpenAI API key not provided and not found in OPENAI_API_KEY environment variable.")

def test_llmsynthesizer_init_uses_default_model(llm_synthesizer_mocks):
    """Test initialization uses the default LLM model if not specified."""
    api_key = "sk-defaultmodelkey"
    synthesizer = LLMSynthesizer(openai_api_key=api_key)
    assert synthesizer.model_name == DEFAULT_LLM_MODEL
    llm_synthesizer_mocks["logger"].info.assert_any_call(f"LLMSynthesizer initialized with model: {DEFAULT_LLM_MODEL}.")

# --- Tests for LLMSynthesizer.synthesize ---

def test_llmsynthesizer_synthesize_success(llm_synthesizer_mocks, mock_reranked_docs_fixture):
    """Test successful answer synthesis."""
    api_key = "sk-synthkey"
    synthesizer = LLMSynthesizer(openai_api_key=api_key)

    query = "Test query for LLM"
    top_n_llm = 2 # Use top 2 docs for context

    # The mock_openai_chatcompletion_create is already set up in llm_synthesizer_mocks
    # to return a mock response with "Mocked LLM answer."

    answer = synthesizer.synthesize(query, mock_reranked_docs_fixture, top_n_docs_for_llm=top_n_llm)

    assert answer == "Mocked LLM answer."
    llm_synthesizer_mocks["mock_openai_chatcompletion_create"].assert_called_once()
    args, _ = llm_synthesizer_mocks["mock_openai_chatcompletion_create"].call_args

    # Check model and messages passed to ChatCompletion.create
    assert args[0]['model'] == DEFAULT_LLM_MODEL
    messages = args[0]['messages']
    assert messages[0]['role'] == "system"
    # Check that the prompt contains the query and parts of the context documents
    user_prompt = messages[1]['content']
    assert query in user_prompt
    assert mock_reranked_docs_fixture[0][2] in user_prompt # Text of first doc
    assert mock_reranked_docs_fixture[1][2] in user_prompt # Text of second doc
    assert mock_reranked_docs_fixture[2][2] not in user_prompt # Third doc should not be in context if top_n_llm=2

    llm_synthesizer_mocks["logger"].info.assert_any_call(f"Sending request to OpenAI API (model: {DEFAULT_LLM_MODEL}). Using {top_n_llm} documents for context.")
    llm_synthesizer_mocks["logger"].info.assert_any_call("Answer successfully synthesized by LLM.")

def test_llmsynthesizer_synthesize_no_documents(llm_synthesizer_mocks):
    """Test synthesis when no documents are provided."""
    api_key = "sk-nodockey"
    synthesizer = LLMSynthesizer(openai_api_key=api_key)

    answer = synthesizer.synthesize("Test query", [], top_n_docs_for_llm=2)

    assert answer == "No documents were available to synthesize an answer."
    llm_synthesizer_mocks["mock_openai_chatcompletion_create"].assert_not_called()
    llm_synthesizer_mocks["logger"].warning.assert_any_call("No documents provided to LLM for context.")

@patch('openai.ChatCompletion.create') # Patching directly here to test specific errors
def test_llmsynthesizer_synthesize_openai_auth_error(mock_openai_create_direct, llm_synthesizer_mocks, mock_reranked_docs_fixture):
    """Test handling of OpenAI AuthenticationError."""
    # Need to import the actual error for side_effect
    from openai.error import AuthenticationError

    mock_openai_create_direct.side_effect = AuthenticationError("Invalid API key.")
    api_key = "sk-autherrorkey" # This key will be "invalid" due to the mock

    # Re-patch the logger inside LLMSynthesizer if it's a new instance or ensure the fixture's logger is used
    # For simplicity, assuming the fixture's logger mock is effective if LLMSynthesizer uses tiered_ranking_pipeline.logger

    synthesizer = LLMSynthesizer(openai_api_key=api_key)
    answer = synthesizer.synthesize("query", mock_reranked_docs_fixture)

    assert "OpenAI API Authentication Error: Invalid API key." in answer
    llm_synthesizer_mocks["logger"].error.assert_any_call("OpenAI API Authentication Error: Invalid API key. Check your API key.", exc_info=True)

@patch('openai.ChatCompletion.create')
def test_llmsynthesizer_synthesize_openai_api_error(mock_openai_create_direct, llm_synthesizer_mocks, mock_reranked_docs_fixture):
    """Test handling of general OpenAI APIError."""
    from openai.error import APIError

    mock_openai_create_direct.side_effect = APIError("OpenAI server error.")
    api_key = "sk-apierrorkey"

    synthesizer = LLMSynthesizer(openai_api_key=api_key)
    answer = synthesizer.synthesize("query", mock_reranked_docs_fixture)

    assert "OpenAI API Error: OpenAI server error." in answer
    llm_synthesizer_mocks["logger"].error.assert_any_call("OpenAI API Error: OpenAI server error.", exc_info=True)

@patch('openai.ChatCompletion.create')
def test_llmsynthesizer_synthesize_unexpected_error(mock_openai_create_direct, llm_synthesizer_mocks, mock_reranked_docs_fixture):
    """Test handling of other unexpected errors during synthesis."""
    mock_openai_create_direct.side_effect = Exception("Some weird network issue.")
    api_key = "sk-unexpectedkey"

    synthesizer = LLMSynthesizer(openai_api_key=api_key)
    answer = synthesizer.synthesize("query", mock_reranked_docs_fixture)

    assert "An unexpected error occurred during LLM synthesis: Some weird network issue." in answer
    llm_synthesizer_mocks["logger"].error.assert_any_call("Unexpected error during LLM synthesis: Some weird network issue.", exc_info=True)

def test_llmsynthesizer_synthesize_prompt_formatting(llm_synthesizer_mocks, mock_reranked_docs_fixture):
    """Test the prompt formatting more closely."""
    api_key = "sk-promptkey"
    synthesizer = LLMSynthesizer(openai_api_key=api_key)

    query = "Specific query for prompt content?"
    # Use only one document for easier assertion
    single_doc_list = [mock_reranked_docs_fixture[0]] # ("doc2", 0.9, "Relevant text from document 2.")

    synthesizer.synthesize(query, single_doc_list, top_n_docs_for_llm=1)

    args, _ = llm_synthesizer_mocks["mock_openai_chatcompletion_create"].call_args
    user_prompt = args[0]['messages'][1]['content']

    assert f"Query: \"{query}\"" in user_prompt
    assert f"Document 1 (ID: {single_doc_list[0][0]}, Relevance Score: {single_doc_list[0][1]:.4f}):\n{single_doc_list[0][2]}" in user_prompt
    assert "Based ONLY on the provided documents" in user_prompt # Part of system instruction in prompt

# Note: It's important that the patch path for openai.ChatCompletion.create is correct.
# If LLMSynthesizer uses `import openai` and then `openai.ChatCompletion.create`,
# then patching `openai.ChatCompletion.create` or `tiered_ranking_pipeline.openai.ChatCompletion.create`
# (if openai is imported in tiered_ranking_pipeline.py) should work.
# The fixture patches 'openai.ChatCompletion.create' which is usually correct if 'import openai' is used.
# If it was 'from openai import ChatCompletion', the patch path would be different.
# Given the current structure, 'openai.ChatCompletion.create' is the most robust.
# The `patch` directly in test methods is to show specific error types from `openai.error`.
# The fixture `llm_synthesizer_mocks` already patches `openai.ChatCompletion.create` globally for most tests.
# However, to test specific exceptions from `create`, you might need to manage the mock's side_effect per test.
# The tests `test_llmsynthesizer_synthesize_openai_auth_error` etc. demonstrate this by re-patching.
# This is fine for isolating specific error handling.
