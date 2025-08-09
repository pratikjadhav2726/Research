"""
CoRS: Collaborative Retrieval and Synthesis

A novel multi-agent RAG architecture for emergent consensus and coherent synthesis.
"""

__version__ = "0.1.0"
__author__ = "CoRS Research Team"
__email__ = "research@cors-rag.org"

from .core.cors_system import CoRSSystem, CoRSConfig, create_cors_system
from .core.reputation_weighted_consensus import (
    ReputationWeightedConsensus,
    ConsensusStrategy,
    SynthesisCandidate,
    ConsensusResult
)
from .core.shared_synthesis_space import (
    SharedSynthesisSpace,
    CoRSState,
    SubQueryState,
    SubQueryStatus
)

__all__ = [
    # Main system
    "CoRSSystem",
    "CoRSConfig", 
    "create_cors_system",
    
    # Consensus mechanism
    "ReputationWeightedConsensus",
    "ConsensusStrategy",
    "SynthesisCandidate",
    "ConsensusResult",
    
    # Shared synthesis space
    "SharedSynthesisSpace",
    "CoRSState",
    "SubQueryState", 
    "SubQueryStatus",
]