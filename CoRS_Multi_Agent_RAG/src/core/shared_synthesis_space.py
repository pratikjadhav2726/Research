"""
Shared Synthesis Space (SSS) Implementation

This module implements the stateful cognitive workspace where agents
collaboratively store and access intermediate results, syntheses, and
consensus decisions using Redis as the backend.
"""

import json
import time
import uuid
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import redis
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SubQueryStatus(Enum):
    """Status enum for sub-query processing."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    SYNTHESIZING = "synthesizing"
    CRITIQUING = "critiquing"
    CONSENSUS_REACHED = "consensus_reached"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubQueryState:
    """State object for individual sub-queries."""
    sub_query_id: str
    sub_query: str
    status: SubQueryStatus
    retrieved_docs: List[Dict[str, Any]]
    syntheses: Dict[str, Dict[str, Any]]  # agent_id -> synthesis data
    critiques: Dict[str, float]  # synthesis_id -> critique score
    consensus_output: Optional[str]
    consensus_agent_id: Optional[str]
    created_at: float
    updated_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubQueryState':
        """Create from dictionary retrieved from Redis."""
        data['status'] = SubQueryStatus(data['status'])
        return cls(**data)


@dataclass
class CoRSState:
    """Main state object for the entire CoRS workflow."""
    session_id: str
    original_query: str
    sub_queries: List[SubQueryState]
    final_answer: Optional[str]
    agent_reputations: Dict[str, float]
    created_at: float
    updated_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        data = asdict(self)
        data['sub_queries'] = [sq.to_dict() for sq in self.sub_queries]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoRSState':
        """Create from dictionary retrieved from Redis."""
        data['sub_queries'] = [SubQueryState.from_dict(sq) for sq in data['sub_queries']]
        return cls(**data)


class SharedSynthesisSpace:
    """
    Redis-based implementation of the Shared Synthesis Space.
    
    This class provides the cognitive workspace where agents can:
    - Store and retrieve sub-query states
    - Log synthesis attempts and critiques
    - Maintain agent reputation scores
    - Track workflow progress
    """
    
    def __init__(self, 
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 redis_password: Optional[str] = None):
        """
        Initialize the SSS with Redis connection.
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port  
            redis_db: Redis database number
            redis_password: Redis password (if required)
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True
        )
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def create_session(self, original_query: str) -> str:
        """
        Create a new CoRS session.
        
        Args:
            original_query: The original user query
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        current_time = time.time()
        
        state = CoRSState(
            session_id=session_id,
            original_query=original_query,
            sub_queries=[],
            final_answer=None,
            agent_reputations={},
            created_at=current_time,
            updated_at=current_time
        )
        
        self._store_session_state(state)
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session_state(self, session_id: str) -> Optional[CoRSState]:
        """
        Retrieve the complete session state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            CoRSState object or None if not found
        """
        key = f"session:{session_id}"
        data = self.redis_client.hgetall(key)
        
        if not data:
            return None
        
        # Parse JSON fields
        for field in ['sub_queries', 'agent_reputations']:
            if field in data:
                data[field] = json.loads(data[field])
        
        # Convert numeric fields
        for field in ['created_at', 'updated_at']:
            if field in data:
                data[field] = float(data[field])
        
        return CoRSState.from_dict(data)
    
    def update_sub_queries(self, session_id: str, sub_queries: List[str]) -> None:
        """
        Initialize sub-queries for a session.
        
        Args:
            session_id: Session identifier
            sub_queries: List of sub-query strings
        """
        current_time = time.time()
        sub_query_states = []
        
        for i, sub_query in enumerate(sub_queries):
            sub_query_id = f"{session_id}_sq_{i}"
            state = SubQueryState(
                sub_query_id=sub_query_id,
                sub_query=sub_query,
                status=SubQueryStatus.PENDING,
                retrieved_docs=[],
                syntheses={},
                critiques={},
                consensus_output=None,
                consensus_agent_id=None,
                created_at=current_time,
                updated_at=current_time
            )
            sub_query_states.append(state)
        
        # Update session state
        session_state = self.get_session_state(session_id)
        if session_state:
            session_state.sub_queries = sub_query_states
            session_state.updated_at = current_time
            self._store_session_state(session_state)
            logger.info(f"Updated sub-queries for session {session_id}: {len(sub_queries)} sub-queries")
    
    def update_sub_query_status(self, session_id: str, sub_query_id: str, status: SubQueryStatus) -> None:
        """
        Update the status of a specific sub-query.
        
        Args:
            session_id: Session identifier
            sub_query_id: Sub-query identifier
            status: New status
        """
        session_state = self.get_session_state(session_id)
        if not session_state:
            raise ValueError(f"Session {session_id} not found")
        
        for sq in session_state.sub_queries:
            if sq.sub_query_id == sub_query_id:
                sq.status = status
                sq.updated_at = time.time()
                break
        else:
            raise ValueError(f"Sub-query {sub_query_id} not found in session {session_id}")
        
        session_state.updated_at = time.time()
        self._store_session_state(session_state)
    
    def store_retrieved_docs(self, session_id: str, sub_query_id: str, docs: List[Dict[str, Any]]) -> None:
        """
        Store retrieved documents for a sub-query.
        
        Args:
            session_id: Session identifier
            sub_query_id: Sub-query identifier  
            docs: List of retrieved documents
        """
        session_state = self.get_session_state(session_id)
        if not session_state:
            raise ValueError(f"Session {session_id} not found")
        
        for sq in session_state.sub_queries:
            if sq.sub_query_id == sub_query_id:
                sq.retrieved_docs = docs
                sq.updated_at = time.time()
                break
        else:
            raise ValueError(f"Sub-query {sub_query_id} not found")
        
        session_state.updated_at = time.time()
        self._store_session_state(session_state)
        logger.info(f"Stored {len(docs)} documents for sub-query {sub_query_id}")
    
    def store_synthesis(self, session_id: str, sub_query_id: str, agent_id: str, 
                       synthesis_text: str, evidence_docs: List[str] = None) -> str:
        """
        Store a synthesis from an agent.
        
        Args:
            session_id: Session identifier
            sub_query_id: Sub-query identifier
            agent_id: Agent identifier
            synthesis_text: The synthesized response
            evidence_docs: List of document IDs used as evidence
            
        Returns:
            Synthesis ID
        """
        synthesis_id = f"{sub_query_id}_{agent_id}_{int(time.time())}"
        synthesis_data = {
            "synthesis_id": synthesis_id,
            "agent_id": agent_id,
            "text": synthesis_text,
            "evidence_docs": evidence_docs or [],
            "created_at": time.time()
        }
        
        session_state = self.get_session_state(session_id)
        if not session_state:
            raise ValueError(f"Session {session_id} not found")
        
        for sq in session_state.sub_queries:
            if sq.sub_query_id == sub_query_id:
                sq.syntheses[synthesis_id] = synthesis_data
                sq.updated_at = time.time()
                break
        else:
            raise ValueError(f"Sub-query {sub_query_id} not found")
        
        session_state.updated_at = time.time()
        self._store_session_state(session_state)
        
        # Log to Redis Stream for audit trail
        self._log_event(session_id, sub_query_id, "synthesis_created", {
            "synthesis_id": synthesis_id,
            "agent_id": agent_id,
            "text_length": len(synthesis_text)
        })
        
        logger.info(f"Stored synthesis {synthesis_id} from agent {agent_id}")
        return synthesis_id
    
    def store_critique(self, session_id: str, sub_query_id: str, synthesis_id: str, 
                      critic_agent_id: str, score: float, reasoning: str = None) -> None:
        """
        Store a critique for a synthesis.
        
        Args:
            session_id: Session identifier
            sub_query_id: Sub-query identifier
            synthesis_id: Synthesis identifier
            critic_agent_id: Critic agent identifier
            score: Critique score (0.0 to 1.0)
            reasoning: Optional reasoning for the score
        """
        session_state = self.get_session_state(session_id)
        if not session_state:
            raise ValueError(f"Session {session_id} not found")
        
        for sq in session_state.sub_queries:
            if sq.sub_query_id == sub_query_id:
                sq.critiques[synthesis_id] = score
                sq.updated_at = time.time()
                break
        else:
            raise ValueError(f"Sub-query {sub_query_id} not found")
        
        session_state.updated_at = time.time()
        self._store_session_state(session_state)
        
        # Log to Redis Stream
        self._log_event(session_id, sub_query_id, "critique_created", {
            "synthesis_id": synthesis_id,
            "critic_agent_id": critic_agent_id,
            "score": score,
            "reasoning": reasoning
        })
        
        logger.info(f"Stored critique for synthesis {synthesis_id}: score={score}")
    
    def store_consensus(self, session_id: str, sub_query_id: str, 
                       winning_synthesis_id: str, consensus_output: str) -> None:
        """
        Store the consensus decision for a sub-query.
        
        Args:
            session_id: Session identifier
            sub_query_id: Sub-query identifier
            winning_synthesis_id: ID of the winning synthesis
            consensus_output: Final consensus output
        """
        session_state = self.get_session_state(session_id)
        if not session_state:
            raise ValueError(f"Session {session_id} not found")
        
        for sq in session_state.sub_queries:
            if sq.sub_query_id == sub_query_id:
                # Get the agent ID from the winning synthesis
                winning_agent_id = None
                if winning_synthesis_id in sq.syntheses:
                    winning_agent_id = sq.syntheses[winning_synthesis_id]["agent_id"]
                
                sq.consensus_output = consensus_output
                sq.consensus_agent_id = winning_agent_id
                sq.status = SubQueryStatus.CONSENSUS_REACHED
                sq.updated_at = time.time()
                break
        else:
            raise ValueError(f"Sub-query {sub_query_id} not found")
        
        session_state.updated_at = time.time()
        self._store_session_state(session_state)
        
        # Log consensus decision
        self._log_event(session_id, sub_query_id, "consensus_reached", {
            "winning_synthesis_id": winning_synthesis_id,
            "winning_agent_id": winning_agent_id
        })
        
        logger.info(f"Stored consensus for sub-query {sub_query_id}")
    
    def get_agent_reputation(self, agent_id: str) -> float:
        """
        Get the reputation score for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Reputation score (default 0.5 for new agents)
        """
        score = self.redis_client.zscore("agent_reputations", agent_id)
        return score if score is not None else 0.5
    
    def update_agent_reputation(self, agent_id: str, new_score: float) -> None:
        """
        Update an agent's reputation score.
        
        Args:
            agent_id: Agent identifier
            new_score: New reputation score
        """
        self.redis_client.zadd("agent_reputations", {agent_id: new_score})
        logger.info(f"Updated reputation for agent {agent_id}: {new_score}")
    
    def get_top_agents(self, limit: int = 10) -> List[tuple]:
        """
        Get the top-rated agents by reputation.
        
        Args:
            limit: Maximum number of agents to return
            
        Returns:
            List of (agent_id, reputation_score) tuples
        """
        return self.redis_client.zrevrange("agent_reputations", 0, limit-1, withscores=True)
    
    def _store_session_state(self, state: CoRSState) -> None:
        """Store the complete session state in Redis."""
        key = f"session:{state.session_id}"
        data = state.to_dict()
        
        # Convert complex fields to JSON
        for field in ['sub_queries', 'agent_reputations']:
            data[field] = json.dumps(data[field])
        
        self.redis_client.hset(key, mapping=data)
    
    def _log_event(self, session_id: str, sub_query_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event to Redis Stream for audit trail."""
        stream_key = f"log:{sub_query_id}"
        event_data = {
            "session_id": session_id,
            "sub_query_id": sub_query_id,
            "event_type": event_type,
            "timestamp": time.time(),
            "data": json.dumps(data)
        }
        self.redis_client.xadd(stream_key, event_data)
    
    @contextmanager
    def session_context(self, session_id: str):
        """Context manager for session operations."""
        try:
            yield self.get_session_state(session_id)
        except Exception as e:
            logger.error(f"Error in session {session_id}: {e}")
            raise
    
    def cleanup_session(self, session_id: str) -> None:
        """
        Clean up all data for a session.
        
        Args:
            session_id: Session identifier
        """
        # Delete main session data
        self.redis_client.delete(f"session:{session_id}")
        
        # Delete log streams
        session_state = self.get_session_state(session_id)
        if session_state:
            for sq in session_state.sub_queries:
                self.redis_client.delete(f"log:{sq.sub_query_id}")
        
        logger.info(f"Cleaned up session {session_id}")
    
    def get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """
        Get metrics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary of session metrics
        """
        session_state = self.get_session_state(session_id)
        if not session_state:
            return {}
        
        total_sub_queries = len(session_state.sub_queries)
        completed_sub_queries = sum(1 for sq in session_state.sub_queries 
                                  if sq.status == SubQueryStatus.CONSENSUS_REACHED)
        
        total_syntheses = sum(len(sq.syntheses) for sq in session_state.sub_queries)
        total_critiques = sum(len(sq.critiques) for sq in session_state.sub_queries)
        
        return {
            "session_id": session_id,
            "total_sub_queries": total_sub_queries,
            "completed_sub_queries": completed_sub_queries,
            "completion_rate": completed_sub_queries / total_sub_queries if total_sub_queries > 0 else 0,
            "total_syntheses": total_syntheses,
            "total_critiques": total_critiques,
            "session_duration": time.time() - session_state.created_at,
            "final_answer_ready": session_state.final_answer is not None
        }