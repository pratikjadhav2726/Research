# Tiered Document Ranking Pipeline for RAG

A multi-stage document retrieval pipeline that retrieves, re-ranks, and synthesizes responses using BM25, Cross-Encoder models, and an LLM. This implementation uses publicly available datasets and pre-trained models, requiring no additional training.

### Project Overview

This project implements a tiered document ranking pipeline for efficient Retrieval-Augmented Generation (RAG):
	1.	Phase 1 - Retrieve (BM25/Semantic Search)
	•	Retrieve top 500 documents using BM25 (keyword-based retrieval).
	2.	Phase 2 - Re-Rank (Cross-Encoder)
	•	Re-rank top 50 documents using a pre-trained cross-encoder model for better relevance.
	3.	Phase 3 - Synthesize (LLM Generation)
	•	Pass top 10 documents to a Large Language Model (LLM) to generate a final response.

### Tech Stack
	•	Python (Core programming language)
	•	BM25 via Pyserini
	•	Cross-Encoder via Hugging Face SentenceTransformers
	•	LLM for Answer Synthesis (GPT-4 or Open-Source mistralai/Mistral-7B-Instruct)
	•	Datasets:
	•	MS MARCO: Hugging Face Dataset (for passage retrieval)
	•	Alternative: Wikipedia from Hugging Face Datasets
	•	FAISS (for potential vector retrieval, optional)
	•	Hugging Face Transformers (for open-source LLM integration)
### Future Work
	•	Implement multi-query expansion for better retrieval.
	•	Explore multi-stage re-ranking (e.g., T5-based models).
	•	Use RAG architectures like LangChain for scalable retrieval.
