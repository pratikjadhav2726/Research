"""
Shared Dynamic Context Workspace - The Heart of CoRS RAG

This module implements the novel collaborative cache that sits between agents and the 
knowledge base, enabling emergent consensus and massive efficiency gains through 
intelligent information sharing and verification.

The workspace acts like a digital whiteboard where agents post findings, draw connections,
and build upon each other's discoveries in real-time.
"""

import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import redis
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Rich metadata for each information chunk in the workspace"""
    chunk_id: str
    content: str
    saliency_score: float  # 0.0 to 1.0 - relevance to original query
    verification_count: int  # How many agents retrieved this independently
    retrieval_sources: List[Dict[str, str]]  # [{'agent_id': 'x', 'query': 'y', 'timestamp': 'z'}]
    synthesis_status: bool  # Has this been used in final synthesis?
    timestamp: float  # Last access time
    ttl_expires: float  # When this entry expires
    embedding: Optional[List[float]] = None  # For semantic search within workspace
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Redis storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChunkMetadata':
        """Create from dictionary loaded from Redis"""
        return cls(**data)

@dataclass
class WorkspaceStats:
    """Statistics about workspace performance and collaboration"""
    total_chunks: int
    cache_hit_rate: float
    avg_verification_count: float
    top_contributing_agents: List[Tuple[str, int]]
    synthesis_efficiency: float  # % of chunks actually used in synthesis
    collaboration_score: float  # Measure of inter-agent information sharing

class SharedDynamicWorkspace:
    """
    The Shared Dynamic Context Workspace - CoRS's intelligent collaborative cache
    
    This acts as a temporary, highly structured cache between agents and the knowledge base.
    It stores not just information, but the collaborative intelligence around that information:
    - Who found it and why (provenance)
    - How many agents independently verified it (consensus)
    - Whether it's been used in synthesis (efficiency)
    """
    
    def __init__(self, 
                 redis_client: redis.Redis,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 default_ttl: int = 3600,  # 1 hour default TTL
                 similarity_threshold: float = 0.85):
        """
        Initialize the Shared Dynamic Context Workspace
        
        Args:
            redis_client: Redis connection for storage
            embedding_model: Model for semantic similarity in workspace
            default_ttl: Default time-to-live for workspace entries (seconds)
            similarity_threshold: Threshold for considering chunks as duplicates
        """
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.similarity_threshold = similarity_threshold
        
        # Initialize embedding model for semantic search within workspace
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Redis key patterns
        self.CHUNK_KEY_PREFIX = "cors:chunk:"
        self.EMBEDDING_KEY_PREFIX = "cors:embedding:"
        self.STATS_KEY = "cors:workspace:stats"
        self.AGENT_STATS_KEY = "cors:agent:stats"
        
        logger.info(f"Initialized SharedDynamicWorkspace with TTL={default_ttl}s")
    
    def _generate_chunk_id(self, content: str) -> str:
        """Generate unique ID for content chunk"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _calculate_saliency_score(self, content: str, query: str) -> float:
        """
        Calculate how relevant this content is to the query that found it
        Uses semantic similarity between query and content
        """
        try:
            query_embedding = self.embedding_model.encode([query])
            content_embedding = self.embedding_model.encode([content])
            
            # Cosine similarity
            similarity = np.dot(query_embedding[0], content_embedding[0]) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(content_embedding[0])
            )
            
            # Normalize to 0-1 range
            return max(0.0, min(1.0, (similarity + 1) / 2))
        except Exception as e:
            logger.warning(f"Error calculating saliency score: {e}")
            return 0.5  # Default moderate relevance
    
    def search_workspace(self, query: str, top_k: int = 5, min_verification: int = 1) -> List[ChunkMetadata]:
        """
        Search the workspace for relevant information before hitting the main knowledge base
        This is Step 1 of the CoRS protocol
        
        Args:
            query: Search query from agent
            top_k: Maximum number of results to return
            min_verification: Minimum verification count to consider
            
        Returns:
            List of relevant chunks with metadata, sorted by relevance and verification
        """
        try:
            # Get all chunk keys from Redis
            chunk_keys = self.redis.keys(f"{self.CHUNK_KEY_PREFIX}*")
            
            if not chunk_keys:
                logger.info("Workspace is empty - cache miss")
                return []
            
            # Load all chunks and filter by verification count
            candidates = []
            query_embedding = self.embedding_model.encode([query])[0]
            
            for key in chunk_keys:
                try:
                    chunk_data = self.redis.hgetall(key)
                    if not chunk_data:
                        continue
                    
                    # Decode Redis data
                    chunk_dict = {k.decode(): v.decode() for k, v in chunk_data.items()}
                    
                    # Parse JSON fields
                    chunk_dict['retrieval_sources'] = json.loads(chunk_dict['retrieval_sources'])
                    chunk_dict['verification_count'] = int(chunk_dict['verification_count'])
                    chunk_dict['saliency_score'] = float(chunk_dict['saliency_score'])
                    chunk_dict['synthesis_status'] = chunk_dict['synthesis_status'].lower() == 'true'
                    chunk_dict['timestamp'] = float(chunk_dict['timestamp'])
                    chunk_dict['ttl_expires'] = float(chunk_dict['ttl_expires'])
                    
                    # Skip if below minimum verification threshold
                    if chunk_dict['verification_count'] < min_verification:
                        continue
                    
                    # Check TTL
                    if time.time() > chunk_dict['ttl_expires']:
                        self._expire_chunk(chunk_dict['chunk_id'])
                        continue
                    
                    # Calculate semantic similarity to query
                    chunk_embedding = self._get_chunk_embedding(chunk_dict['chunk_id'])
                    if chunk_embedding is not None:
                        similarity = np.dot(query_embedding, chunk_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                        )
                        
                        # Combine similarity with verification count and saliency for ranking
                        relevance_score = (
                            0.4 * similarity +
                            0.3 * (chunk_dict['verification_count'] / 10.0) +  # Normalize verification
                            0.3 * chunk_dict['saliency_score']
                        )
                        
                        chunk_metadata = ChunkMetadata.from_dict(chunk_dict)
                        candidates.append((relevance_score, chunk_metadata))
                
                except Exception as e:
                    logger.warning(f"Error processing chunk {key}: {e}")
                    continue
            
            # Sort by relevance score and return top_k
            candidates.sort(key=lambda x: x[0], reverse=True)
            results = [chunk for _, chunk in candidates[:top_k]]
            
            if results:
                logger.info(f"Workspace cache HIT: Found {len(results)} relevant chunks for query")
                self._update_cache_stats(hit=True)
            else:
                logger.info("Workspace cache MISS: No relevant chunks found")
                self._update_cache_stats(hit=False)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching workspace: {e}")
            self._update_cache_stats(hit=False)
            return []
    
    def contribute_to_workspace(self, 
                              chunks: List[str], 
                              agent_id: str, 
                              query: str,
                              source_db: str = "main_kb") -> Dict[str, Any]:
        """
        Contribute new findings to the workspace (Step 3 of CoRS protocol)
        This is where the collaborative magic happens - de-duplication and verification counting
        
        Args:
            chunks: List of text chunks retrieved from main knowledge base
            agent_id: ID of the agent contributing this information
            query: The query that led to these chunks
            source_db: Source database identifier
            
        Returns:
            Dictionary with contribution statistics
        """
        contribution_stats = {
            'new_chunks': 0,
            'verified_chunks': 0,
            'duplicate_chunks': 0,
            'total_contributed': len(chunks)
        }
        
        current_time = time.time()
        ttl_expires = current_time + self.default_ttl
        
        for content in chunks:
            try:
                chunk_id = self._generate_chunk_id(content)
                chunk_key = f"{self.CHUNK_KEY_PREFIX}{chunk_id}"
                
                # Check if this chunk already exists
                existing_chunk = self.redis.hgetall(chunk_key)
                
                if existing_chunk:
                    # Chunk exists - increment verification and update metadata
                    existing_data = {k.decode(): v.decode() for k, v in existing_chunk.items()}
                    existing_sources = json.loads(existing_data['retrieval_sources'])
                    
                    # Check if this agent already contributed this chunk for this query
                    already_contributed = any(
                        source['agent_id'] == agent_id and source['query'] == query 
                        for source in existing_sources
                    )
                    
                    if not already_contributed:
                        # New verification from different agent or different query
                        existing_sources.append({
                            'agent_id': agent_id,
                            'query': query,
                            'timestamp': datetime.now().isoformat(),
                            'source_db': source_db
                        })
                        
                        verification_count = int(existing_data['verification_count']) + 1
                        
                        # Update the chunk with new verification
                        self.redis.hset(chunk_key, mapping={
                            'verification_count': verification_count,
                            'retrieval_sources': json.dumps(existing_sources),
                            'timestamp': current_time,
                            'ttl_expires': ttl_expires
                        })
                        
                        # Extend TTL since chunk is still being accessed
                        self.redis.expire(chunk_key, self.default_ttl)
                        
                        contribution_stats['verified_chunks'] += 1
                        logger.info(f"Verified existing chunk {chunk_id} (verification_count: {verification_count})")
                    else:
                        contribution_stats['duplicate_chunks'] += 1
                        logger.debug(f"Duplicate contribution for chunk {chunk_id} by {agent_id}")
                
                else:
                    # New chunk - calculate saliency and store
                    saliency_score = self._calculate_saliency_score(content, query)
                    
                    chunk_metadata = ChunkMetadata(
                        chunk_id=chunk_id,
                        content=content,
                        saliency_score=saliency_score,
                        verification_count=1,
                        retrieval_sources=[{
                            'agent_id': agent_id,
                            'query': query,
                            'timestamp': datetime.now().isoformat(),
                            'source_db': source_db
                        }],
                        synthesis_status=False,
                        timestamp=current_time,
                        ttl_expires=ttl_expires
                    )
                    
                    # Store chunk metadata in Redis
                    chunk_dict = chunk_metadata.to_dict()
                    chunk_dict['retrieval_sources'] = json.dumps(chunk_dict['retrieval_sources'])
                    chunk_dict['synthesis_status'] = str(chunk_dict['synthesis_status']).lower()
                    
                    self.redis.hset(chunk_key, mapping=chunk_dict)
                    self.redis.expire(chunk_key, self.default_ttl)
                    
                    # Store embedding for semantic search
                    self._store_chunk_embedding(chunk_id, content)
                    
                    contribution_stats['new_chunks'] += 1
                    logger.info(f"Added new chunk {chunk_id} to workspace (saliency: {saliency_score:.3f})")
            
            except Exception as e:
                logger.error(f"Error contributing chunk to workspace: {e}")
                continue
        
        # Update agent statistics
        self._update_agent_stats(agent_id, contribution_stats)
        
        logger.info(f"Agent {agent_id} contribution complete: {contribution_stats}")
        return contribution_stats
    
    def get_synthesis_context(self, 
                            task_description: str, 
                            min_verification: int = 2,
                            prioritize_unused: bool = True) -> List[ChunkMetadata]:
        """
        Get prioritized context for synthesis agent
        Prioritizes highly verified, unused chunks to prevent redundancy
        
        Args:
            task_description: Description of the synthesis task
            min_verification: Minimum verification count to include
            prioritize_unused: Whether to prioritize unused chunks
            
        Returns:
            List of chunks sorted by priority for synthesis
        """
        try:
            # Search workspace for relevant content
            relevant_chunks = self.search_workspace(task_description, top_k=50, min_verification=min_verification)
            
            if prioritize_unused:
                # Separate unused and used chunks
                unused_chunks = [chunk for chunk in relevant_chunks if not chunk.synthesis_status]
                used_chunks = [chunk for chunk in relevant_chunks if chunk.synthesis_status]
                
                # Prioritize unused chunks, then used ones
                prioritized_chunks = unused_chunks + used_chunks
            else:
                prioritized_chunks = relevant_chunks
            
            # Sort by verification count and saliency
            prioritized_chunks.sort(
                key=lambda x: (x.verification_count * x.saliency_score), 
                reverse=True
            )
            
            logger.info(f"Retrieved {len(prioritized_chunks)} chunks for synthesis (unused: {len([c for c in prioritized_chunks if not c.synthesis_status])})")
            return prioritized_chunks
            
        except Exception as e:
            logger.error(f"Error getting synthesis context: {e}")
            return []
    
    def mark_chunks_synthesized(self, chunk_ids: List[str], synthesizer_agent_id: str):
        """
        Mark chunks as used in synthesis to prevent redundancy
        
        Args:
            chunk_ids: List of chunk IDs that were used
            synthesizer_agent_id: ID of the agent that used these chunks
        """
        for chunk_id in chunk_ids:
            try:
                chunk_key = f"{self.CHUNK_KEY_PREFIX}{chunk_id}"
                self.redis.hset(chunk_key, 'synthesis_status', 'true')
                self.redis.hset(chunk_key, 'synthesized_by', synthesizer_agent_id)
                self.redis.hset(chunk_key, 'synthesis_timestamp', time.time())
                
                logger.debug(f"Marked chunk {chunk_id} as synthesized by {synthesizer_agent_id}")
            except Exception as e:
                logger.error(f"Error marking chunk {chunk_id} as synthesized: {e}")
    
    def _store_chunk_embedding(self, chunk_id: str, content: str):
        """Store embedding for semantic search within workspace"""
        try:
            embedding = self.embedding_model.encode([content])[0]
            embedding_key = f"{self.EMBEDDING_KEY_PREFIX}{chunk_id}"
            
            # Store as binary data for efficiency
            embedding_bytes = embedding.astype(np.float32).tobytes()
            self.redis.set(embedding_key, embedding_bytes, ex=self.default_ttl)
            
        except Exception as e:
            logger.warning(f"Error storing embedding for chunk {chunk_id}: {e}")
    
    def _get_chunk_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Retrieve stored embedding for chunk"""
        try:
            embedding_key = f"{self.EMBEDDING_KEY_PREFIX}{chunk_id}"
            embedding_bytes = self.redis.get(embedding_key)
            
            if embedding_bytes:
                return np.frombuffer(embedding_bytes, dtype=np.float32)
            return None
            
        except Exception as e:
            logger.warning(f"Error retrieving embedding for chunk {chunk_id}: {e}")
            return None
    
    def _expire_chunk(self, chunk_id: str):
        """Remove expired chunk from workspace"""
        try:
            chunk_key = f"{self.CHUNK_KEY_PREFIX}{chunk_id}"
            embedding_key = f"{self.EMBEDDING_KEY_PREFIX}{chunk_id}"
            
            self.redis.delete(chunk_key, embedding_key)
            logger.debug(f"Expired chunk {chunk_id}")
            
        except Exception as e:
            logger.error(f"Error expiring chunk {chunk_id}: {e}")
    
    def _update_cache_stats(self, hit: bool):
        """Update cache hit/miss statistics"""
        try:
            stats_key = self.STATS_KEY
            if hit:
                self.redis.hincrby(stats_key, 'cache_hits', 1)
            else:
                self.redis.hincrby(stats_key, 'cache_misses', 1)
                
        except Exception as e:
            logger.warning(f"Error updating cache stats: {e}")
    
    def _update_agent_stats(self, agent_id: str, contribution_stats: Dict[str, Any]):
        """Update per-agent statistics"""
        try:
            agent_key = f"{self.AGENT_STATS_KEY}:{agent_id}"
            
            for stat_name, value in contribution_stats.items():
                self.redis.hincrby(agent_key, stat_name, value)
            
            self.redis.hincrby(agent_key, 'total_contributions', 1)
            self.redis.expire(agent_key, self.default_ttl * 24)  # Keep stats longer
            
        except Exception as e:
            logger.warning(f"Error updating agent stats for {agent_id}: {e}")
    
    def get_workspace_stats(self) -> WorkspaceStats:
        """Get comprehensive workspace statistics"""
        try:
            # Get cache statistics
            cache_stats = self.redis.hgetall(self.STATS_KEY)
            cache_hits = int(cache_stats.get(b'cache_hits', b'0'))
            cache_misses = int(cache_stats.get(b'cache_misses', b'0'))
            
            total_requests = cache_hits + cache_misses
            cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0
            
            # Get chunk statistics
            chunk_keys = self.redis.keys(f"{self.CHUNK_KEY_PREFIX}*")
            total_chunks = len(chunk_keys)
            
            verification_counts = []
            synthesized_count = 0
            
            for key in chunk_keys:
                chunk_data = self.redis.hgetall(key)
                if chunk_data:
                    verification_count = int(chunk_data.get(b'verification_count', b'1'))
                    verification_counts.append(verification_count)
                    
                    synthesis_status = chunk_data.get(b'synthesis_status', b'false').decode()
                    if synthesis_status.lower() == 'true':
                        synthesized_count += 1
            
            avg_verification = np.mean(verification_counts) if verification_counts else 0.0
            synthesis_efficiency = synthesized_count / total_chunks if total_chunks > 0 else 0.0
            
            # Get top contributing agents
            agent_keys = self.redis.keys(f"{self.AGENT_STATS_KEY}:*")
            agent_contributions = []
            
            for key in agent_keys:
                agent_stats = self.redis.hgetall(key)
                if agent_stats:
                    agent_id = key.decode().split(':')[-1]
                    total_contributions = int(agent_stats.get(b'total_contributions', b'0'))
                    agent_contributions.append((agent_id, total_contributions))
            
            agent_contributions.sort(key=lambda x: x[1], reverse=True)
            top_contributing_agents = agent_contributions[:5]
            
            # Calculate collaboration score (measure of information sharing)
            collaboration_score = min(1.0, avg_verification - 1.0) if avg_verification > 1.0 else 0.0
            
            return WorkspaceStats(
                total_chunks=total_chunks,
                cache_hit_rate=cache_hit_rate,
                avg_verification_count=avg_verification,
                top_contributing_agents=top_contributing_agents,
                synthesis_efficiency=synthesis_efficiency,
                collaboration_score=collaboration_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating workspace stats: {e}")
            return WorkspaceStats(0, 0.0, 0.0, [], 0.0, 0.0)
    
    def cleanup_expired_chunks(self) -> int:
        """Clean up expired chunks and return count of removed chunks"""
        try:
            current_time = time.time()
            chunk_keys = self.redis.keys(f"{self.CHUNK_KEY_PREFIX}*")
            removed_count = 0
            
            for key in chunk_keys:
                chunk_data = self.redis.hgetall(key)
                if chunk_data:
                    ttl_expires = float(chunk_data.get(b'ttl_expires', b'0'))
                    if current_time > ttl_expires:
                        chunk_id = chunk_data.get(b'chunk_id', b'').decode()
                        self._expire_chunk(chunk_id)
                        removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} expired chunks")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    def clear_workspace(self):
        """Clear all workspace data (for testing/reset)"""
        try:
            chunk_keys = self.redis.keys(f"{self.CHUNK_KEY_PREFIX}*")
            embedding_keys = self.redis.keys(f"{self.EMBEDDING_KEY_PREFIX}*")
            stats_keys = self.redis.keys(f"{self.STATS_KEY}*")
            agent_keys = self.redis.keys(f"{self.AGENT_STATS_KEY}*")
            
            all_keys = chunk_keys + embedding_keys + stats_keys + agent_keys
            
            if all_keys:
                self.redis.delete(*all_keys)
                logger.info(f"Cleared workspace: removed {len(all_keys)} keys")
            
        except Exception as e:
            logger.error(f"Error clearing workspace: {e}")