"""
Retrieval Agent Implementation

This agent is responsible for retrieving relevant documents from vector
databases and other knowledge sources based on sub-queries.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from .base_agent import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """Represents a retrieved document with metadata."""
    doc_id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata or {}
        }


class VectorStoreInterface:
    """Abstract interface for vector stores."""
    
    def search(self, query: str, k: int = 5) -> List[RetrievedDocument]:
        """Search for documents similar to the query."""
        raise NotImplementedError
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector store."""
        raise NotImplementedError


class ChromaDBInterface(VectorStoreInterface):
    """ChromaDB implementation of vector store interface."""
    
    def __init__(self, collection_name: str = "cors_knowledge", persist_directory: str = "./data/chroma_db"):
        """Initialize ChromaDB interface."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Initialized ChromaDB with collection: {collection_name}")
            
        except ImportError:
            logger.error("ChromaDB not installed. Install with: pip install chromadb")
            raise
    
    def search(self, query: str, k: int = 5) -> List[RetrievedDocument]:
        """Search ChromaDB for similar documents."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    doc_id = results['ids'][0][i] if results['ids'] else f"doc_{i}"
                    score = 1.0 - results['distances'][0][i] if results['distances'] else 0.0
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    
                    documents.append(RetrievedDocument(
                        doc_id=doc_id,
                        content=doc,
                        score=score,
                        source="chromadb",
                        metadata=metadata
                    ))
            
            return documents
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to ChromaDB."""
        try:
            ids = [doc.get('id', f"doc_{i}") for i, doc in enumerate(documents)]
            texts = [doc['content'] for doc in documents]
            metadatas = [doc.get('metadata', {}) for doc in documents]
            
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            return False


class PineconeInterface(VectorStoreInterface):
    """Pinecone implementation of vector store interface."""
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        """Initialize Pinecone interface."""
        try:
            import pinecone
            
            pinecone.init(api_key=api_key, environment=environment)
            self.index = pinecone.Index(index_name)
            
            # Initialize sentence transformer for embeddings
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info(f"Initialized Pinecone with index: {index_name}")
            
        except ImportError:
            logger.error("Pinecone not installed. Install with: pip install pinecone-client")
            raise
    
    def search(self, query: str, k: int = 5) -> List[RetrievedDocument]:
        """Search Pinecone for similar documents."""
        try:
            # Generate embedding for query
            query_embedding = self.encoder.encode([query])[0].tolist()
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )
            
            documents = []
            for match in results['matches']:
                documents.append(RetrievedDocument(
                    doc_id=match['id'],
                    content=match['metadata'].get('content', ''),
                    score=match['score'],
                    source="pinecone",
                    metadata=match['metadata']
                ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to Pinecone."""
        try:
            # Generate embeddings
            texts = [doc['content'] for doc in documents]
            embeddings = self.encoder.encode(texts)
            
            # Prepare vectors for upsert
            vectors = []
            for i, doc in enumerate(documents):
                vectors.append({
                    'id': doc.get('id', f"doc_{i}"),
                    'values': embeddings[i].tolist(),
                    'metadata': {
                        'content': doc['content'],
                        **doc.get('metadata', {})
                    }
                })
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors)
            
            logger.info(f"Added {len(documents)} documents to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to Pinecone: {e}")
            return False


class RetrievalAgent(BaseAgent):
    """
    Agent responsible for document retrieval.
    
    The Retrieval Agent searches vector databases and other knowledge sources
    to find relevant documents for each sub-query.
    """
    
    SYSTEM_PROMPT = """You are a document retrieval expert. Your role is to analyze sub-queries and determine the best search strategy to find relevant information.

Your responsibilities include:
1. Understanding the intent and key concepts in sub-queries
2. Generating effective search terms and queries
3. Evaluating the relevance of retrieved documents
4. Providing metadata about the retrieval process

You should be thorough but efficient in your retrieval process."""
    
    def __init__(self, 
                 agent_id: str = None,
                 vector_store: VectorStoreInterface = None,
                 **kwargs):
        """
        Initialize the Retrieval Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            vector_store: Vector store interface for document retrieval
            **kwargs: Additional configuration parameters
        """
        if agent_id is None:
            agent_id = f"retrieval_{int(time.time())}"
        
        config = AgentConfig(
            agent_id=agent_id,
            agent_type="retrieval",
            temperature=0.0,  # Deterministic retrieval
            max_tokens=500,
            **kwargs
        )
        
        super().__init__(config)
        
        # Set up vector store
        self.vector_store = vector_store
        if self.vector_store is None:
            # Default to ChromaDB
            try:
                self.vector_store = ChromaDBInterface()
            except Exception as e:
                logger.warning(f"Failed to initialize default vector store: {e}")
        
        # Retrieval settings
        self.default_k = 5  # Number of documents to retrieve
        self.max_k = 10
        self.min_relevance_score = 0.3
        self.query_expansion_enabled = True
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Retrieval Agent."""
        return self.SYSTEM_PROMPT
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process a sub-query and retrieve relevant documents.
        
        Args:
            input_data: Dictionary containing 'sub_query' and optionally 'k'
            **kwargs: Additional processing parameters
            
        Returns:
            Dictionary with retrieved documents and metadata
        """
        self.log_processing_start(input_data)
        
        try:
            # Validate input
            self.validate_input(input_data, ['sub_query'])
            sub_query = input_data['sub_query']
            k = input_data.get('k', self.default_k)
            k = min(k, self.max_k)  # Enforce maximum
            
            if not self.vector_store:
                error_result = {
                    "documents": [],
                    "sub_query": sub_query,
                    "success": False,
                    "error": "No vector store configured"
                }
                self.log_processing_end(error_result, False)
                return error_result
            
            # Generate search queries
            search_queries = self._generate_search_queries(sub_query)
            
            # Retrieve documents for each search query
            all_documents = []
            retrieval_metadata = {
                "search_queries": search_queries,
                "total_searches": len(search_queries),
                "retrieval_times": []
            }
            
            for search_query in search_queries:
                start_time = time.time()
                docs = self.vector_store.search(search_query, k=k)
                retrieval_time = time.time() - start_time
                
                retrieval_metadata["retrieval_times"].append(retrieval_time)
                all_documents.extend(docs)
                
                logger.debug(f"Retrieved {len(docs)} documents for query: '{search_query}'")
            
            # Deduplicate and rank documents
            unique_documents = self._deduplicate_documents(all_documents)
            ranked_documents = self._rank_documents(unique_documents, sub_query)
            
            # Filter by relevance score
            filtered_documents = [
                doc for doc in ranked_documents 
                if doc.score >= self.min_relevance_score
            ]
            
            # Limit final results
            final_documents = filtered_documents[:k]
            
            # Prepare result
            result = {
                "documents": [doc.to_dict() for doc in final_documents],
                "sub_query": sub_query,
                "success": True,
                "metadata": {
                    **retrieval_metadata,
                    "total_documents_found": len(all_documents),
                    "unique_documents": len(unique_documents),
                    "filtered_documents": len(filtered_documents),
                    "final_documents": len(final_documents),
                    "avg_retrieval_time": sum(retrieval_metadata["retrieval_times"]) / len(retrieval_metadata["retrieval_times"]) if retrieval_metadata["retrieval_times"] else 0,
                    "min_score": min([doc.score for doc in final_documents]) if final_documents else 0,
                    "max_score": max([doc.score for doc in final_documents]) if final_documents else 0
                }
            }
            
            self.log_processing_end(result, True)
            return result
            
        except Exception as e:
            error_result = {
                "documents": [],
                "sub_query": input_data.get('sub_query', ''),
                "success": False,
                "error": str(e)
            }
            self.log_processing_end(error_result, False)
            return error_result
    
    def _generate_search_queries(self, sub_query: str) -> List[str]:
        """
        Generate multiple search queries from a sub-query.
        
        Args:
            sub_query: Original sub-query
            
        Returns:
            List of search query variations
        """
        queries = [sub_query]  # Always include original
        
        if self.query_expansion_enabled:
            # Simple query expansion techniques
            
            # Remove question words for keyword search
            question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'do', 'does']
            keywords = []
            for word in sub_query.lower().split():
                if word not in question_words and len(word) > 2:
                    keywords.append(word)
            
            if keywords:
                keyword_query = ' '.join(keywords)
                if keyword_query != sub_query:
                    queries.append(keyword_query)
            
            # Extract key phrases (simple noun phrase extraction)
            key_phrases = self._extract_key_phrases(sub_query)
            for phrase in key_phrases:
                if phrase not in queries and len(phrase) > 5:
                    queries.append(phrase)
        
        return queries[:3]  # Limit to 3 queries to avoid too many searches
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """
        Simple key phrase extraction.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted key phrases
        """
        # Simple approach: extract noun phrases (very basic)
        words = text.lower().replace('?', '').replace('.', '').split()
        
        # Look for patterns like "adjective noun" or "noun noun"
        phrases = []
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i+1]) > 3:
                phrase = f"{words[i]} {words[i+1]}"
                phrases.append(phrase)
        
        return phrases[:2]  # Return up to 2 key phrases
    
    def _deduplicate_documents(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        Remove duplicate documents based on content similarity.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            List of unique documents
        """
        if not documents:
            return []
        
        unique_docs = []
        seen_content = set()
        
        for doc in documents:
            # Simple deduplication based on content hash
            content_hash = hash(doc.content[:200])  # Use first 200 chars for hash
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
    
    def _rank_documents(self, documents: List[RetrievedDocument], query: str) -> List[RetrievedDocument]:
        """
        Rank documents by relevance to the query.
        
        Args:
            documents: List of documents to rank
            query: Original query
            
        Returns:
            List of documents sorted by relevance score
        """
        # For now, just sort by existing scores
        # In a more advanced implementation, you could:
        # - Use a reranker model
        # - Apply query-specific scoring
        # - Consider document freshness, authority, etc.
        
        return sorted(documents, key=lambda doc: doc.score, reverse=True)
    
    def add_documents_to_store(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            True if successful, False otherwise
        """
        if not self.vector_store:
            logger.error("No vector store configured")
            return False
        
        try:
            return self.vector_store.add_documents(documents)
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about retrieval performance.
        
        Returns:
            Dictionary with retrieval statistics
        """
        return {
            "agent_id": self.agent_id,
            "vector_store_type": type(self.vector_store).__name__ if self.vector_store else "None",
            "default_k": self.default_k,
            "min_relevance_score": self.min_relevance_score,
            "query_expansion_enabled": self.query_expansion_enabled,
            "total_retrievals": self.metrics.total_calls,
            "success_rate": self.metrics.success_rate,
            "avg_response_time": self.metrics.avg_response_time
        }
    
    def update_retrieval_settings(self, 
                                 default_k: int = None,
                                 min_relevance_score: float = None,
                                 query_expansion_enabled: bool = None):
        """
        Update retrieval settings.
        
        Args:
            default_k: Default number of documents to retrieve
            min_relevance_score: Minimum relevance score threshold
            query_expansion_enabled: Whether to enable query expansion
        """
        if default_k is not None:
            self.default_k = max(1, min(default_k, self.max_k))
        
        if min_relevance_score is not None:
            self.min_relevance_score = max(0.0, min(min_relevance_score, 1.0))
        
        if query_expansion_enabled is not None:
            self.query_expansion_enabled = query_expansion_enabled
        
        logger.info(f"Updated retrieval settings for agent {self.agent_id}")


def create_retrieval_agent(agent_id: str = None, 
                          vector_store_type: str = "chromadb",
                          **kwargs) -> RetrievalAgent:
    """
    Factory function to create a Retrieval Agent.
    
    Args:
        agent_id: Optional agent identifier
        vector_store_type: Type of vector store ("chromadb", "pinecone")
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured RetrievalAgent instance
    """
    vector_store = None
    
    if vector_store_type == "chromadb":
        try:
            vector_store = ChromaDBInterface()
        except Exception as e:
            logger.warning(f"Failed to create ChromaDB interface: {e}")
    
    elif vector_store_type == "pinecone":
        # Pinecone requires API credentials
        api_key = kwargs.get('pinecone_api_key')
        environment = kwargs.get('pinecone_environment')
        index_name = kwargs.get('pinecone_index_name', 'cors-knowledge-base')
        
        if api_key and environment:
            try:
                vector_store = PineconeInterface(api_key, environment, index_name)
            except Exception as e:
                logger.warning(f"Failed to create Pinecone interface: {e}")
    
    return RetrievalAgent(agent_id=agent_id, vector_store=vector_store, **kwargs)