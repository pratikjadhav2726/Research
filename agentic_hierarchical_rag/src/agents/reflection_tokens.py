"""Reflection tokens for self-aware RAG control."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


class ReflectionToken(Enum):
    """Types of reflection tokens used by the agentic controller."""
    
    # Retrieval decision tokens
    RETRIEVE_YES = "[Retrieve]"
    RETRIEVE_NO = "[No Retrieval]"
    RETRIEVE_CONTINUE = "[Continue Retrieval]"
    
    # Relevance critique tokens
    RELEVANT = "[Relevant]"
    PARTIALLY_RELEVANT = "[Partially Relevant]"
    IRRELEVANT = "[Irrelevant]"
    
    # Sufficiency tokens
    SUFFICIENT = "[Sufficient]"
    INSUFFICIENT = "[Insufficient]"
    PARTIALLY_SUFFICIENT = "[Partially Sufficient]"
    
    # Support tokens
    FULLY_SUPPORTED = "[Fully Supported]"
    PARTIALLY_SUPPORTED = "[Partially Supported]"
    NOT_SUPPORTED = "[Not Supported]"
    
    # Navigation tokens
    ZOOM_IN = "[Zoom In]"
    ZOOM_OUT = "[Zoom Out]"
    STAY_LEVEL = "[Stay Level]"
    EXPLORE_SIBLINGS = "[Explore Siblings]"
    
    # Confidence tokens
    HIGH_CONFIDENCE = "[High Confidence]"
    MEDIUM_CONFIDENCE = "[Medium Confidence]"
    LOW_CONFIDENCE = "[Low Confidence]"


@dataclass
class ReflectionTokens:
    """Container for reflection tokens generated during processing."""
    
    retrieval_decision: Optional[ReflectionToken] = None
    relevance_assessment: Optional[ReflectionToken] = None
    sufficiency_assessment: Optional[ReflectionToken] = None
    support_assessment: Optional[ReflectionToken] = None
    navigation_decision: Optional[ReflectionToken] = None
    confidence_level: Optional[ReflectionToken] = None
    reasoning: Optional[str] = None
    
    def requires_retrieval(self) -> bool:
        """Check if retrieval is needed."""
        return self.retrieval_decision in [
            ReflectionToken.RETRIEVE_YES,
            ReflectionToken.RETRIEVE_CONTINUE
        ]
    
    def is_sufficient(self) -> bool:
        """Check if current context is sufficient."""
        return self.sufficiency_assessment == ReflectionToken.SUFFICIENT
    
    def is_relevant(self) -> bool:
        """Check if retrieved content is relevant."""
        return self.relevance_assessment in [
            ReflectionToken.RELEVANT,
            ReflectionToken.PARTIALLY_RELEVANT
        ]
    
    def is_supported(self) -> bool:
        """Check if answer is supported by evidence."""
        return self.support_assessment in [
            ReflectionToken.FULLY_SUPPORTED,
            ReflectionToken.PARTIALLY_SUPPORTED
        ]
    
    def should_zoom_in(self) -> bool:
        """Check if should navigate to more detailed level."""
        return self.navigation_decision == ReflectionToken.ZOOM_IN
    
    def should_zoom_out(self) -> bool:
        """Check if should navigate to more abstract level."""
        return self.navigation_decision == ReflectionToken.ZOOM_OUT
    
    def should_explore_siblings(self) -> bool:
        """Check if should explore sibling nodes."""
        return self.navigation_decision == ReflectionToken.EXPLORE_SIBLINGS
    
    def get_confidence_score(self) -> float:
        """Convert confidence token to numeric score."""
        if self.confidence_level == ReflectionToken.HIGH_CONFIDENCE:
            return 0.9
        elif self.confidence_level == ReflectionToken.MEDIUM_CONFIDENCE:
            return 0.6
        elif self.confidence_level == ReflectionToken.LOW_CONFIDENCE:
            return 0.3
        else:
            return 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "retrieval_decision": self.retrieval_decision.value if self.retrieval_decision else None,
            "relevance_assessment": self.relevance_assessment.value if self.relevance_assessment else None,
            "sufficiency_assessment": self.sufficiency_assessment.value if self.sufficiency_assessment else None,
            "support_assessment": self.support_assessment.value if self.support_assessment else None,
            "navigation_decision": self.navigation_decision.value if self.navigation_decision else None,
            "confidence_level": self.confidence_level.value if self.confidence_level else None,
            "reasoning": self.reasoning,
            "confidence_score": self.get_confidence_score()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReflectionTokens":
        """Create from dictionary representation."""
        def get_token(key: str) -> Optional[ReflectionToken]:
            value = data.get(key)
            if value:
                for token in ReflectionToken:
                    if token.value == value:
                        return token
            return None
        
        return cls(
            retrieval_decision=get_token("retrieval_decision"),
            relevance_assessment=get_token("relevance_assessment"),
            sufficiency_assessment=get_token("sufficiency_assessment"),
            support_assessment=get_token("support_assessment"),
            navigation_decision=get_token("navigation_decision"),
            confidence_level=get_token("confidence_level"),
            reasoning=data.get("reasoning")
        )