"""
Decomposer Agent Implementation

This agent is responsible for breaking down complex user queries into
simpler, independent sub-queries that can be processed in parallel.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent, AgentConfig
import time

logger = logging.getLogger(__name__)


class DecomposerAgent(BaseAgent):
    """
    Agent responsible for query decomposition.
    
    The Decomposer Agent analyzes complex queries and breaks them down into
    a series of simple, independent, fact-seeking sub-queries that can be
    researched and answered in parallel.
    """
    
    SYSTEM_PROMPT = """You are a task decomposition expert. Your role is to break down complex user queries into a series of simple, independent, and fact-based sub-queries that can be researched in parallel.

GUIDELINES:
1. Each sub-query should be self-contained and not depend on answers to other sub-queries
2. Sub-queries should be specific and fact-seeking (not opinion-based)
3. Aim for 2-5 sub-queries for most questions
4. Ensure sub-queries cover all aspects of the original question
5. Use clear, concise language
6. Avoid redundant or overlapping sub-queries

OUTPUT FORMAT:
Return your response as a JSON object with this structure:
{
    "sub_queries": [
        "First specific sub-query",
        "Second specific sub-query",
        ...
    ],
    "reasoning": "Brief explanation of your decomposition strategy"
}

EXAMPLES:

User Query: "What are the environmental and economic impacts of renewable energy adoption in developing countries?"
Response:
{
    "sub_queries": [
        "What are the main environmental benefits of renewable energy compared to fossil fuels?",
        "What are the economic costs and benefits of renewable energy projects in developing countries?",
        "What are the current renewable energy adoption rates in developing countries?",
        "What are the main barriers to renewable energy adoption in developing countries?"
    ],
    "reasoning": "Decomposed into environmental impacts, economic impacts, current status, and barriers to provide comprehensive coverage"
}

User Query: "How does machine learning work in healthcare applications?"
Response:
{
    "sub_queries": [
        "What are the main types of machine learning algorithms used in healthcare?",
        "What are specific examples of machine learning applications in medical diagnosis?",
        "What are the benefits of using machine learning in healthcare?",
        "What are the challenges and limitations of machine learning in healthcare?"
    ],
    "reasoning": "Separated into algorithm types, specific applications, benefits, and challenges for thorough analysis"
}"""

    def __init__(self, agent_id: str = None, **kwargs):
        """
        Initialize the Decomposer Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            **kwargs: Additional configuration parameters
        """
        if agent_id is None:
            agent_id = f"decomposer_{int(time.time())}"
        
        config = AgentConfig(
            agent_id=agent_id,
            agent_type="decomposer",
            temperature=0.1,  # Low temperature for consistent decomposition
            max_tokens=800,
            **kwargs
        )
        
        super().__init__(config)
        
        # Decomposer-specific settings
        self.min_sub_queries = 2
        self.max_sub_queries = 6
        self.complexity_threshold = 10  # Words in query to trigger decomposition
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Decomposer Agent."""
        return self.SYSTEM_PROMPT
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process a query and decompose it into sub-queries.
        
        Args:
            input_data: Dictionary containing 'original_query'
            **kwargs: Additional processing parameters
            
        Returns:
            Dictionary with decomposed sub-queries and metadata
        """
        self.log_processing_start(input_data)
        
        try:
            # Validate input
            self.validate_input(input_data, ['original_query'])
            original_query = input_data['original_query']
            
            # Check if decomposition is needed
            if not self._should_decompose(original_query):
                result = {
                    "sub_queries": [original_query],
                    "reasoning": "Query is simple enough to be processed as a single sub-query",
                    "decomposed": False,
                    "original_query": original_query,
                    "success": True
                }
                self.log_processing_end(result, True)
                return result
            
            # Create prompt for decomposition
            prompt = self.format_prompt(
                "User Query: {query}\n\nPlease decompose this query following the guidelines above.",
                query=original_query
            )
            
            # Invoke LLM
            llm_result = self.invoke_llm([self.create_human_message(prompt)])
            
            if not llm_result['success']:
                error_result = {
                    "sub_queries": [original_query],  # Fallback to original
                    "reasoning": "LLM call failed, using original query as fallback",
                    "decomposed": False,
                    "original_query": original_query,
                    "success": False,
                    "error": llm_result.get('error', 'Unknown error')
                }
                self.log_processing_end(error_result, False)
                return error_result
            
            # Parse LLM response
            parsed_result = self._parse_decomposition_response(llm_result['content'])
            
            if not parsed_result:
                # Fallback parsing
                sub_queries = self._fallback_decomposition(original_query)
                result = {
                    "sub_queries": sub_queries,
                    "reasoning": "Used fallback decomposition due to parsing error",
                    "decomposed": len(sub_queries) > 1,
                    "original_query": original_query,
                    "success": True
                }
            else:
                # Validate and clean sub-queries
                sub_queries = self._validate_sub_queries(parsed_result['sub_queries'])
                result = {
                    "sub_queries": sub_queries,
                    "reasoning": parsed_result.get('reasoning', ''),
                    "decomposed": len(sub_queries) > 1,
                    "original_query": original_query,
                    "success": True,
                    "llm_metadata": {
                        "tokens_used": llm_result['tokens_used'],
                        "response_time": llm_result['response_time']
                    }
                }
            
            self.log_processing_end(result, True)
            return result
            
        except Exception as e:
            error_result = {
                "sub_queries": [input_data.get('original_query', '')],
                "reasoning": f"Error during decomposition: {str(e)}",
                "decomposed": False,
                "original_query": input_data.get('original_query', ''),
                "success": False,
                "error": str(e)
            }
            self.log_processing_end(error_result, False)
            return error_result
    
    def _should_decompose(self, query: str) -> bool:
        """
        Determine if a query should be decomposed.
        
        Args:
            query: Original query string
            
        Returns:
            True if query should be decomposed
        """
        # Simple heuristics for decomposition decision
        word_count = len(query.split())
        
        # Don't decompose very short queries
        if word_count < self.complexity_threshold:
            return False
        
        # Look for complexity indicators
        complexity_indicators = [
            'and', 'or', 'what are', 'how does', 'why do', 'compare',
            'difference between', 'advantages and disadvantages',
            'benefits and drawbacks', 'pros and cons', 'impact of'
        ]
        
        query_lower = query.lower()
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in query_lower)
        
        # Decompose if multiple complexity indicators or long query
        return indicator_count >= 2 or word_count > 20
    
    def _parse_decomposition_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse the LLM response to extract sub-queries.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed result dictionary or None if parsing failed
        """
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                return None
            
            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)
            
            # Validate required fields
            if 'sub_queries' not in parsed or not isinstance(parsed['sub_queries'], list):
                return None
            
            return parsed
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse decomposition response: {e}")
            return None
    
    def _validate_sub_queries(self, sub_queries: List[str]) -> List[str]:
        """
        Validate and clean the sub-queries.
        
        Args:
            sub_queries: List of sub-query strings
            
        Returns:
            Cleaned and validated list of sub-queries
        """
        cleaned_queries = []
        
        for query in sub_queries:
            if isinstance(query, str):
                # Clean the query
                cleaned_query = query.strip()
                if cleaned_query and len(cleaned_query) > 5:  # Minimum length check
                    # Ensure it ends with a question mark if it's a question
                    if cleaned_query.startswith(('What', 'How', 'Why', 'When', 'Where', 'Who')):
                        if not cleaned_query.endswith('?'):
                            cleaned_query += '?'
                    cleaned_queries.append(cleaned_query)
        
        # Ensure we have at least one sub-query
        if not cleaned_queries:
            cleaned_queries = ["Please provide information about the requested topic."]
        
        # Limit number of sub-queries
        if len(cleaned_queries) > self.max_sub_queries:
            cleaned_queries = cleaned_queries[:self.max_sub_queries]
        
        return cleaned_queries
    
    def _fallback_decomposition(self, query: str) -> List[str]:
        """
        Provide fallback decomposition when LLM parsing fails.
        
        Args:
            query: Original query
            
        Returns:
            List of sub-queries (may just be the original query)
        """
        # Simple rule-based fallback
        if 'and' in query.lower():
            # Try to split on 'and'
            parts = query.split(' and ')
            if len(parts) == 2:
                return [part.strip() + '?' if not part.strip().endswith('?') else part.strip() 
                       for part in parts if part.strip()]
        
        # If no good decomposition found, return original
        return [query]
    
    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze the complexity characteristics of a query.
        
        Args:
            query: Query to analyze
            
        Returns:
            Dictionary with complexity metrics
        """
        word_count = len(query.split())
        char_count = len(query)
        sentence_count = len([s for s in query.split('.') if s.strip()])
        
        # Count question words
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        question_word_count = sum(1 for word in question_words if word in query.lower())
        
        # Count complexity indicators
        complexity_indicators = [
            'compare', 'contrast', 'analyze', 'evaluate', 'explain',
            'describe', 'discuss', 'advantages', 'disadvantages',
            'benefits', 'drawbacks', 'impact', 'effect', 'cause'
        ]
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in query.lower())
        
        return {
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "question_word_count": question_word_count,
            "complexity_score": complexity_score,
            "should_decompose": self._should_decompose(query),
            "estimated_sub_queries": min(max(2, complexity_score + 1), 5)
        }
    
    def get_decomposition_strategies(self) -> Dict[str, str]:
        """
        Get available decomposition strategies.
        
        Returns:
            Dictionary mapping strategy names to descriptions
        """
        return {
            "aspect_based": "Decompose by different aspects or dimensions of the topic",
            "temporal": "Decompose by time periods or chronological order",
            "categorical": "Decompose by categories or types",
            "causal": "Decompose by causes and effects",
            "comparative": "Decompose by different entities to compare",
            "process_based": "Decompose by steps in a process",
            "stakeholder_based": "Decompose by different stakeholders or perspectives"
        }


def create_decomposer_agent(agent_id: str = None, **kwargs) -> DecomposerAgent:
    """
    Factory function to create a Decomposer Agent.
    
    Args:
        agent_id: Optional agent identifier
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured DecomposerAgent instance
    """
    return DecomposerAgent(agent_id=agent_id, **kwargs)