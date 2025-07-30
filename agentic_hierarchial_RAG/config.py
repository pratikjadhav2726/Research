from pathlib import Path

# ------------ General paths ----------------
PACKAGE_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PACKAGE_ROOT / "data"
INDEX_DIR: Path = PACKAGE_ROOT / "indexes"
EVAL_DIR: Path = PACKAGE_ROOT / "evaluation"

# Create dirs if they don't exist
for _d in (DATA_DIR, INDEX_DIR, EVAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------ Embeddings & Models ----------------
# Choose a lightweight default embedding model that is widely available.
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# Summarization model (can be swapped for any text2text generation model)
SUMMARIZATION_MODEL_NAME: str = "facebook/bart-large-cnn"

# Generator LLM backend. Set to 'openai' or 'hf' for HuggingFace
GENERATOR_BACKEND: str = "openai"

# If using OpenAI, read env var OPENAI_API_KEY at runtime

# ------------- Hierarchy Hyperparameters ----------------
# Maximum children per cluster when building hierarchy
MAX_CHILDREN: int = 10
# Embedding dimensionality is inferred at runtime from model

# ------------- Retrieval ----------------
TOP_K_CANDIDATES: int = 5