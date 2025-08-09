"""
CoRS specialized agents.
"""

from .base_agent import BaseAgent, AgentConfig, AgentMetrics, AgentPool
from .decomposer_agent import DecomposerAgent, create_decomposer_agent
from .retrieval_agent import (
    RetrievalAgent, 
    RetrievedDocument,
    VectorStoreInterface,
    ChromaDBInterface,
    PineconeInterface,
    create_retrieval_agent
)
from .synthesizer_agent import SynthesizerAgent, create_synthesizer_agent
from .critic_agent import CriticAgent, CritiqueResult, create_critic_agent

__all__ = [
    # Base agent
    "BaseAgent",
    "AgentConfig", 
    "AgentMetrics",
    "AgentPool",
    
    # Decomposer
    "DecomposerAgent",
    "create_decomposer_agent",
    
    # Retrieval
    "RetrievalAgent",
    "RetrievedDocument",
    "VectorStoreInterface",
    "ChromaDBInterface", 
    "PineconeInterface",
    "create_retrieval_agent",
    
    # Synthesizer
    "SynthesizerAgent",
    "create_synthesizer_agent",
    
    # Critic
    "CriticAgent",
    "CritiqueResult",
    "create_critic_agent",
]