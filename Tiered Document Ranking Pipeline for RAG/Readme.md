Tiered Document Ranking Pipeline for RAG

A multi-stage document retrieval pipeline that retrieves, re-ranks, and synthesizes responses using BM25, Cross-Encoder models, and an LLM. This implementation uses publicly available datasets and pre-trained models, requiring no additional training.

🚀 Project Overview

This project implements a tiered document ranking pipeline for efficient Retrieval-Augmented Generation (RAG):
	1.	Phase 1 - Retrieve (BM25/Semantic Search)
	•	Retrieve top 500 documents using BM25 (keyword-based retrieval).
	2.	Phase 2 - Re-Rank (Cross-Encoder)
	•	Re-rank top 50 documents using a pre-trained cross-encoder model for better relevance.
	3.	Phase 3 - Synthesize (LLM Generation)
	•	Pass top 10 documents to a Large Language Model (LLM) to generate a final response.

✅ No additional training required!
✅ Uses pre-trained, publicly available models and datasets!
