"""
Setup script for CoRS: Collaborative Retrieval and Synthesis
"""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cors-multi-agent-rag",
    version="0.1.0",
    author="CoRS Research Team",
    author_email="research@cors-rag.org",
    description="A novel multi-agent RAG architecture for emergent consensus and coherent synthesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/CoRS_Multi_Agent_RAG",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cors-demo=examples.basic_example:main",
        ],
    },
    keywords="multi-agent, rag, retrieval-augmented-generation, consensus, ai, nlp, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/CoRS_Multi_Agent_RAG/issues",
        "Source": "https://github.com/your-repo/CoRS_Multi_Agent_RAG",
        "Documentation": "https://cors-multi-agent-rag.readthedocs.io/",
    },
)