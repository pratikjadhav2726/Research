"""Setup script for Agentic Hierarchical RAG."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="agentic-hierarchical-rag",
    version="0.1.0",
    author="AH-RAG Team",
    description="Agentic Hierarchical RAG - Advanced retrieval-augmented generation with intelligent navigation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/agentic-hierarchical-rag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "faiss-cpu>=1.7.4",
        "scipy>=1.10.0",
        "networkx>=3.0",
        "nltk>=3.8.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
        ],
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.5.0",
            "langchain>=0.1.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ah-rag-demo=examples.demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml"],
    },
)