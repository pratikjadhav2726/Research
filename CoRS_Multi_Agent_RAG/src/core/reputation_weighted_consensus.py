"""
Reputation-Weighted Consensus (RWC) Implementation

This module implements the core consensus mechanism that allows the CoRS system
to adjudicate between conflicting syntheses based on agent reputation scores
and critique evaluations.
"""

import logging
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class ConsensusStrategy(Enum):
    """Different strategies for reaching consensus."""
    WEIGHTED_AVERAGE = "weighted_average"
    WINNER_TAKES_ALL = "winner_takes_all"
    THRESHOLD_BASED = "threshold_based"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


@dataclass
class SynthesisCandidate:
    """Represents a synthesis candidate with its metadata."""
    synthesis_id: str
    agent_id: str
    text: str
    critique_score: float
    agent_reputation: float
    evidence_docs: List[str]
    created_at: float


@dataclass
class ConsensusResult:
    """Result of the consensus process."""
    winning_synthesis_id: str
    winning_agent_id: str
    consensus_output: str
    confidence_score: float
    weighted_score: float
    all_candidates: List[SynthesisCandidate]
    consensus_strategy: ConsensusStrategy
    reasoning: str


class ReputationWeightedConsensus:
    """
    Implements the RWC mechanism for agent consensus.
    
    This class provides methods to:
    - Calculate weighted scores based on critique and reputation
    - Select winning synthesis using various strategies
    - Update agent reputations based on performance
    - Handle edge cases and conflicts
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 consensus_threshold: float = 0.8,
                 min_reputation: float = 0.1,
                 max_reputation: float = 1.0,
                 strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_AVERAGE):
        """
        Initialize the RWC mechanism.
        
        Args:
            learning_rate: Rate at which agent reputations are updated (alpha in EMA)
            consensus_threshold: Minimum score threshold for consensus
            min_reputation: Minimum allowed reputation score
            max_reputation: Maximum allowed reputation score
            strategy: Consensus strategy to use
        """
        self.learning_rate = learning_rate
        self.consensus_threshold = consensus_threshold
        self.min_reputation = min_reputation
        self.max_reputation = max_reputation
        self.strategy = strategy
        
        logger.info(f"Initialized RWC with strategy: {strategy.value}, "
                   f"learning_rate: {learning_rate}, threshold: {consensus_threshold}")
    
    def reach_consensus(self, 
                       candidates: List[SynthesisCandidate],
                       sub_query: str,
                       context_docs: List[Dict[str, Any]] = None) -> ConsensusResult:
        """
        Reach consensus among synthesis candidates.
        
        Args:
            candidates: List of synthesis candidates
            sub_query: The original sub-query
            context_docs: Context documents for reference
            
        Returns:
            ConsensusResult with the winning synthesis and metadata
        """
        if not candidates:
            raise ValueError("No synthesis candidates provided")
        
        if len(candidates) == 1:
            # Single candidate - automatic consensus
            candidate = candidates[0]
            return ConsensusResult(
                winning_synthesis_id=candidate.synthesis_id,
                winning_agent_id=candidate.agent_id,
                consensus_output=candidate.text,
                confidence_score=candidate.critique_score,
                weighted_score=self._calculate_weighted_score(candidate),
                all_candidates=candidates,
                consensus_strategy=self.strategy,
                reasoning="Single candidate - automatic consensus"
            )
        
        # Calculate weighted scores for all candidates
        weighted_candidates = []
        for candidate in candidates:
            weighted_score = self._calculate_weighted_score(candidate)
            weighted_candidates.append((candidate, weighted_score))
        
        # Sort by weighted score (descending)
        weighted_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Apply consensus strategy
        if self.strategy == ConsensusStrategy.WEIGHTED_AVERAGE:
            result = self._weighted_average_consensus(weighted_candidates, sub_query)
        elif self.strategy == ConsensusStrategy.WINNER_TAKES_ALL:
            result = self._winner_takes_all_consensus(weighted_candidates, sub_query)
        elif self.strategy == ConsensusStrategy.THRESHOLD_BASED:
            result = self._threshold_based_consensus(weighted_candidates, sub_query)
        elif self.strategy == ConsensusStrategy.CONFIDENCE_WEIGHTED:
            result = self._confidence_weighted_consensus(weighted_candidates, sub_query)
        else:
            raise ValueError(f"Unknown consensus strategy: {self.strategy}")
        
        logger.info(f"Consensus reached: {result.winning_agent_id} "
                   f"(score: {result.weighted_score:.3f}, "
                   f"confidence: {result.confidence_score:.3f})")
        
        return result
    
    def _calculate_weighted_score(self, candidate: SynthesisCandidate) -> float:
        """
        Calculate the weighted score for a synthesis candidate.
        
        Args:
            candidate: Synthesis candidate
            
        Returns:
            Weighted score combining critique score and agent reputation
        """
        # Basic weighted score: critique_score * agent_reputation
        base_score = candidate.critique_score * candidate.agent_reputation
        
        # Apply additional factors
        # 1. Evidence quality bonus (more evidence docs = slight bonus)
        evidence_bonus = min(0.1, len(candidate.evidence_docs) * 0.02)
        
        # 2. Reputation confidence factor (higher reputation = more weight)
        reputation_confidence = self._sigmoid(candidate.agent_reputation - 0.5)
        
        # 3. Critique confidence factor (extreme scores get slight penalty for overconfidence)
        critique_confidence = 1.0 - abs(candidate.critique_score - 0.5) * 0.2
        
        weighted_score = (base_score + evidence_bonus) * reputation_confidence * critique_confidence
        
        # Ensure score is within [0, 1] range
        return max(0.0, min(1.0, weighted_score))
    
    def _weighted_average_consensus(self, 
                                  weighted_candidates: List[Tuple[SynthesisCandidate, float]],
                                  sub_query: str) -> ConsensusResult:
        """
        Implement weighted average consensus strategy.
        
        This strategy selects the candidate with the highest weighted score
        but also considers the overall distribution of scores.
        """
        best_candidate, best_score = weighted_candidates[0]
        
        # Calculate confidence based on score distribution
        all_scores = [score for _, score in weighted_candidates]
        score_std = self._calculate_std(all_scores)
        confidence = best_score * (1.0 - score_std)  # Lower std = higher confidence
        
        reasoning = (f"Selected synthesis with highest weighted score ({best_score:.3f}). "
                    f"Score distribution std: {score_std:.3f}")
        
        return ConsensusResult(
            winning_synthesis_id=best_candidate.synthesis_id,
            winning_agent_id=best_candidate.agent_id,
            consensus_output=best_candidate.text,
            confidence_score=confidence,
            weighted_score=best_score,
            all_candidates=[c for c, _ in weighted_candidates],
            consensus_strategy=ConsensusStrategy.WEIGHTED_AVERAGE,
            reasoning=reasoning
        )
    
    def _winner_takes_all_consensus(self, 
                                  weighted_candidates: List[Tuple[SynthesisCandidate, float]],
                                  sub_query: str) -> ConsensusResult:
        """
        Implement winner-takes-all consensus strategy.
        
        Simply selects the candidate with the highest weighted score.
        """
        best_candidate, best_score = weighted_candidates[0]
        
        reasoning = f"Winner-takes-all: Selected synthesis with score {best_score:.3f}"
        
        return ConsensusResult(
            winning_synthesis_id=best_candidate.synthesis_id,
            winning_agent_id=best_candidate.agent_id,
            consensus_output=best_candidate.text,
            confidence_score=best_score,
            weighted_score=best_score,
            all_candidates=[c for c, _ in weighted_candidates],
            consensus_strategy=ConsensusStrategy.WINNER_TAKES_ALL,
            reasoning=reasoning
        )
    
    def _threshold_based_consensus(self, 
                                 weighted_candidates: List[Tuple[SynthesisCandidate, float]],
                                 sub_query: str) -> ConsensusResult:
        """
        Implement threshold-based consensus strategy.
        
        Only accepts a synthesis if it meets the consensus threshold.
        """
        best_candidate, best_score = weighted_candidates[0]
        
        if best_score >= self.consensus_threshold:
            reasoning = f"Threshold consensus: Score {best_score:.3f} meets threshold {self.consensus_threshold}"
            confidence = best_score
        else:
            # If no candidate meets threshold, fall back to best available
            reasoning = (f"No candidate meets threshold {self.consensus_threshold}. "
                        f"Selecting best available with score {best_score:.3f}")
            confidence = best_score * 0.5  # Reduced confidence
        
        return ConsensusResult(
            winning_synthesis_id=best_candidate.synthesis_id,
            winning_agent_id=best_candidate.agent_id,
            consensus_output=best_candidate.text,
            confidence_score=confidence,
            weighted_score=best_score,
            all_candidates=[c for c, _ in weighted_candidates],
            consensus_strategy=ConsensusStrategy.THRESHOLD_BASED,
            reasoning=reasoning
        )
    
    def _confidence_weighted_consensus(self, 
                                     weighted_candidates: List[Tuple[SynthesisCandidate, float]],
                                     sub_query: str) -> ConsensusResult:
        """
        Implement confidence-weighted consensus strategy.
        
        Considers both the score and the confidence in that score.
        """
        # Calculate confidence-adjusted scores
        confidence_scores = []
        for candidate, score in weighted_candidates:
            # Confidence based on reputation and critique alignment
            reputation_confidence = candidate.agent_reputation
            critique_confidence = 1.0 - abs(candidate.critique_score - 0.5) * 0.5
            
            overall_confidence = (reputation_confidence + critique_confidence) / 2
            confidence_adjusted_score = score * overall_confidence
            
            confidence_scores.append((candidate, score, overall_confidence, confidence_adjusted_score))
        
        # Sort by confidence-adjusted score
        confidence_scores.sort(key=lambda x: x[3], reverse=True)
        
        best_candidate, best_score, best_confidence, best_adjusted_score = confidence_scores[0]
        
        reasoning = (f"Confidence-weighted: Original score {best_score:.3f}, "
                    f"confidence {best_confidence:.3f}, "
                    f"adjusted score {best_adjusted_score:.3f}")
        
        return ConsensusResult(
            winning_synthesis_id=best_candidate.synthesis_id,
            winning_agent_id=best_candidate.agent_id,
            consensus_output=best_candidate.text,
            confidence_score=best_confidence,
            weighted_score=best_score,
            all_candidates=[c for c, _, _, _ in confidence_scores],
            consensus_strategy=ConsensusStrategy.CONFIDENCE_WEIGHTED,
            reasoning=reasoning
        )
    
    def update_agent_reputation(self, 
                               agent_id: str, 
                               current_reputation: float,
                               performance_score: float) -> float:
        """
        Update an agent's reputation using exponential moving average.
        
        Args:
            agent_id: Agent identifier
            current_reputation: Current reputation score
            performance_score: New performance score (typically critique score)
            
        Returns:
            Updated reputation score
        """
        # Exponential moving average update
        new_reputation = (self.learning_rate * performance_score + 
                         (1 - self.learning_rate) * current_reputation)
        
        # Clamp to valid range
        new_reputation = max(self.min_reputation, min(self.max_reputation, new_reputation))
        
        logger.info(f"Updated reputation for {agent_id}: "
                   f"{current_reputation:.3f} -> {new_reputation:.3f} "
                   f"(performance: {performance_score:.3f})")
        
        return new_reputation
    
    def calculate_consensus_metrics(self, result: ConsensusResult) -> Dict[str, float]:
        """
        Calculate metrics for the consensus process.
        
        Args:
            result: Consensus result
            
        Returns:
            Dictionary of consensus metrics
        """
        candidates = result.all_candidates
        scores = [self._calculate_weighted_score(c) for c in candidates]
        
        metrics = {
            "consensus_score": 1.0 - self._calculate_std(scores),  # Higher = more agreement
            "winning_score": result.weighted_score,
            "score_variance": self._calculate_variance(scores),
            "score_range": max(scores) - min(scores) if scores else 0.0,
            "num_candidates": len(candidates),
            "avg_reputation": sum(c.agent_reputation for c in candidates) / len(candidates),
            "avg_critique_score": sum(c.critique_score for c in candidates) / len(candidates),
            "confidence_score": result.confidence_score
        }
        
        return metrics
    
    def detect_consensus_issues(self, result: ConsensusResult) -> List[str]:
        """
        Detect potential issues with the consensus process.
        
        Args:
            result: Consensus result
            
        Returns:
            List of issue descriptions
        """
        issues = []
        candidates = result.all_candidates
        
        # Check for low confidence
        if result.confidence_score < 0.5:
            issues.append(f"Low confidence score: {result.confidence_score:.3f}")
        
        # Check for reputation bias
        reputations = [c.agent_reputation for c in candidates]
        if max(reputations) - min(reputations) > 0.5:
            issues.append("High reputation variance among candidates")
        
        # Check for critique disagreement
        critiques = [c.critique_score for c in candidates]
        if self._calculate_std(critiques) > 0.3:
            issues.append("High disagreement in critique scores")
        
        # Check for insufficient evidence
        avg_evidence = sum(len(c.evidence_docs) for c in candidates) / len(candidates)
        if avg_evidence < 1.0:
            issues.append("Low average evidence count per synthesis")
        
        return issues
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + math.exp(-x))
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)


class AdaptiveRWC(ReputationWeightedConsensus):
    """
    Adaptive version of RWC that can adjust its parameters based on performance.
    
    This extended version can:
    - Adjust learning rate based on consensus quality
    - Switch consensus strategies dynamically
    - Learn optimal thresholds over time
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.consensus_history = []
        self.performance_window = 10  # Number of recent consensus decisions to consider
    
    def reach_consensus_adaptive(self, 
                                candidates: List[SynthesisCandidate],
                                sub_query: str,
                                context_docs: List[Dict[str, Any]] = None) -> ConsensusResult:
        """
        Reach consensus with adaptive parameter adjustment.
        
        This method adjusts consensus parameters based on recent performance.
        """
        # Analyze recent performance
        if len(self.consensus_history) >= 3:
            self._adapt_parameters()
        
        # Reach consensus using current parameters
        result = self.reach_consensus(candidates, sub_query, context_docs)
        
        # Store result for future adaptation
        metrics = self.calculate_consensus_metrics(result)
        self.consensus_history.append({
            'result': result,
            'metrics': metrics,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.consensus_history) > self.performance_window:
            self.consensus_history = self.consensus_history[-self.performance_window:]
        
        return result
    
    def _adapt_parameters(self):
        """Adapt RWC parameters based on recent performance."""
        recent_metrics = [h['metrics'] for h in self.consensus_history[-5:]]
        
        # Calculate average confidence
        avg_confidence = sum(m['confidence_score'] for m in recent_metrics) / len(recent_metrics)
        
        # Adjust learning rate based on confidence
        if avg_confidence < 0.6:
            # Low confidence - increase learning rate for faster adaptation
            self.learning_rate = min(0.2, self.learning_rate * 1.1)
        elif avg_confidence > 0.8:
            # High confidence - decrease learning rate for stability
            self.learning_rate = max(0.05, self.learning_rate * 0.9)
        
        # Adjust consensus threshold based on score variance
        avg_variance = sum(m['score_variance'] for m in recent_metrics) / len(recent_metrics)
        if avg_variance > 0.1:
            # High variance - lower threshold to be more permissive
            self.consensus_threshold = max(0.6, self.consensus_threshold * 0.95)
        elif avg_variance < 0.05:
            # Low variance - raise threshold for higher quality
            self.consensus_threshold = min(0.9, self.consensus_threshold * 1.05)
        
        logger.info(f"Adapted RWC parameters: learning_rate={self.learning_rate:.3f}, "
                   f"threshold={self.consensus_threshold:.3f}")