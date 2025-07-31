"""Query analysis for determining query type and abstraction level."""

import re
from typing import Dict, Any, List, Optional, Callable
import logging

from ..core import Query, QueryType, QueryAnalysis

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Analyzes queries to determine type and optimal search strategy."""
    
    def __init__(
        self,
        llm_analyzer: Optional[Callable[[str], Dict[str, Any]]] = None
    ):
        """Initialize the query analyzer.
        
        Args:
            llm_analyzer: Optional LLM-based analyzer function
        """
        self.llm_analyzer = llm_analyzer
        
        # Keywords for different query types
        self.factual_keywords = [
            "when", "where", "who", "how many", "how much", "what year",
            "specific", "exact", "date", "number", "name", "list",
            "definition", "meaning", "formula", "equation"
        ]
        
        self.thematic_keywords = [
            "theme", "main idea", "summary", "overview", "general",
            "broad", "concept", "theory", "philosophy", "approach",
            "trend", "pattern", "significance", "importance"
        ]
        
        self.comparative_keywords = [
            "compare", "contrast", "difference", "similarity", "versus",
            "vs", "better", "worse", "advantage", "disadvantage",
            "pros and cons", "trade-off"
        ]
        
        self.analytical_keywords = [
            "analyze", "explain", "why", "how does", "reasoning",
            "cause", "effect", "relationship", "impact", "influence",
            "evaluate", "assess", "critique", "interpret"
        ]
        
        self.exploratory_keywords = [
            "explore", "investigate", "discover", "find out",
            "what if", "possibilities", "options", "alternatives",
            "brainstorm", "ideas", "creative", "innovative"
        ]
    
    def analyze_query(self, query: Query) -> QueryAnalysis:
        """Analyze a query to determine type and search strategy.
        
        Args:
            query: Query to analyze
            
        Returns:
            QueryAnalysis with recommendations
        """
        # Use LLM if available, otherwise fall back to rule-based
        if self.llm_analyzer:
            return self._llm_analyze(query)
        else:
            return self._rule_based_analyze(query)
    
    def _rule_based_analyze(self, query: Query) -> QueryAnalysis:
        """Rule-based query analysis.
        
        Args:
            query: Query to analyze
            
        Returns:
            QueryAnalysis with recommendations
        """
        text_lower = query.text.lower()
        
        # Determine query type based on keywords
        query_type = self._determine_query_type(text_lower)
        query.query_type = query_type
        
        # Determine abstraction level and search strategy
        if query_type == QueryType.FACTUAL:
            suggested_levels = [0, 1]  # Leaf and first intermediate
            search_strategy = "bottom_up"
            reasoning = "Factual query requires specific details from leaf nodes"
        elif query_type == QueryType.THEMATIC:
            max_level = 3  # Assume max 3 levels for now
            suggested_levels = list(range(max_level, 0, -1))  # Top-down
            search_strategy = "top_down"
            reasoning = "Thematic query benefits from high-level summaries"
        elif query_type == QueryType.COMPARATIVE:
            suggested_levels = [1, 2]  # Middle levels
            search_strategy = "targeted"
            reasoning = "Comparative query needs intermediate abstraction"
        elif query_type == QueryType.ANALYTICAL:
            suggested_levels = [0, 1, 2]  # Multiple levels
            search_strategy = "multi_level"
            reasoning = "Analytical query requires both details and context"
        else:  # EXPLORATORY
            suggested_levels = list(range(4))  # All levels
            search_strategy = "exploratory"
            reasoning = "Exploratory query benefits from comprehensive search"
        
        # Set confidence based on keyword matches
        confidence = self._calculate_confidence(text_lower, query_type)
        
        return QueryAnalysis(
            query=query,
            suggested_levels=suggested_levels,
            search_strategy=search_strategy,
            reasoning=reasoning,
            requires_retrieval=True,
            confidence=confidence
        )
    
    def _determine_query_type(self, text: str) -> QueryType:
        """Determine query type based on keywords.
        
        Args:
            text: Lowercase query text
            
        Returns:
            Determined QueryType
        """
        # Count keyword matches for each type
        scores = {
            QueryType.FACTUAL: sum(1 for kw in self.factual_keywords if kw in text),
            QueryType.THEMATIC: sum(1 for kw in self.thematic_keywords if kw in text),
            QueryType.COMPARATIVE: sum(1 for kw in self.comparative_keywords if kw in text),
            QueryType.ANALYTICAL: sum(1 for kw in self.analytical_keywords if kw in text),
            QueryType.EXPLORATORY: sum(1 for kw in self.exploratory_keywords if kw in text),
        }
        
        # Return type with highest score
        max_type = max(scores, key=scores.get)
        
        # Default to analytical if no clear winner
        if scores[max_type] == 0:
            return QueryType.ANALYTICAL
            
        return max_type
    
    def _calculate_confidence(self, text: str, query_type: QueryType) -> float:
        """Calculate confidence in query type determination.
        
        Args:
            text: Lowercase query text
            query_type: Determined query type
            
        Returns:
            Confidence score (0-1)
        """
        # Get keywords for the determined type
        if query_type == QueryType.FACTUAL:
            keywords = self.factual_keywords
        elif query_type == QueryType.THEMATIC:
            keywords = self.thematic_keywords
        elif query_type == QueryType.COMPARATIVE:
            keywords = self.comparative_keywords
        elif query_type == QueryType.ANALYTICAL:
            keywords = self.analytical_keywords
        else:
            keywords = self.exploratory_keywords
        
        # Count matches
        matches = sum(1 for kw in keywords if kw in text)
        
        # Calculate confidence based on matches
        if matches >= 3:
            return 0.9
        elif matches >= 2:
            return 0.7
        elif matches >= 1:
            return 0.5
        else:
            return 0.3
    
    def _llm_analyze(self, query: Query) -> QueryAnalysis:
        """LLM-based query analysis.
        
        Args:
            query: Query to analyze
            
        Returns:
            QueryAnalysis with recommendations
        """
        # Call LLM analyzer
        result = self.llm_analyzer(query.text)
        
        # Parse result and create QueryAnalysis
        query_type_str = result.get("query_type", "analytical")
        query_type_map = {
            "factual": QueryType.FACTUAL,
            "thematic": QueryType.THEMATIC,
            "comparative": QueryType.COMPARATIVE,
            "analytical": QueryType.ANALYTICAL,
            "exploratory": QueryType.EXPLORATORY,
        }
        
        query.query_type = query_type_map.get(query_type_str, QueryType.ANALYTICAL)
        
        return QueryAnalysis(
            query=query,
            suggested_levels=result.get("suggested_levels", [0, 1, 2]),
            search_strategy=result.get("search_strategy", "multi_level"),
            reasoning=result.get("reasoning", "LLM-based analysis"),
            requires_retrieval=result.get("requires_retrieval", True),
            confidence=result.get("confidence", 0.7)
        )
    
    def extract_temporal_constraints(self, query_text: str) -> Optional[Dict[str, Any]]:
        """Extract temporal constraints from query.
        
        Args:
            query_text: Query text
            
        Returns:
            Dictionary with temporal constraints or None
        """
        # Simple regex patterns for common temporal expressions
        patterns = {
            "year": r"\b(19|20)\d{2}\b",
            "month_year": r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(19|20)\d{2}\b",
            "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            "relative": r"\b(last|previous|next|this)\s+(year|month|week|day)\b",
            "range": r"\b(between|from)\s+.*\s+(to|and)\s+.*\b",
        }
        
        constraints = {}
        
        for constraint_type, pattern in patterns.items():
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            if matches:
                constraints[constraint_type] = matches
        
        return constraints if constraints else None