"""
Synthesizer Agent Implementation

This agent is responsible for generating concise, evidence-based syntheses
from retrieved documents to answer sub-queries.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)


class SynthesizerAgent(BaseAgent):
    """
    Agent responsible for synthesizing information from retrieved documents.
    
    The Synthesizer Agent takes retrieved documents and generates concise,
    evidence-based answers that directly address sub-queries while maintaining
    faithfulness to the source material.
    """
    
    SYSTEM_PROMPT = """You are a meticulous and factual synthesizer. Your task is to answer sub-queries based *only* on the provided context documents. You must be extremely careful to stay faithful to the source material.

CRITICAL GUIDELINES:
1. Answer ONLY based on information explicitly stated in the provided context
2. If the context does not contain sufficient information, state this clearly
3. Be concise but comprehensive - include all relevant information from the context
4. Always cite which documents support your statements when possible
5. Do not add external knowledge or make inferences beyond what's explicitly stated
6. If there are contradictions in the sources, acknowledge them
7. Use clear, professional language

RESPONSE FORMAT:
Provide your response as a JSON object with this structure:
{
    "answer": "Your direct answer to the sub-query based on the context",
    "evidence_docs": ["doc_1", "doc_2"],
    "confidence": 0.85,
    "reasoning": "Brief explanation of how you derived the answer",
    "limitations": "Any limitations or gaps in the available information"
}

CONFIDENCE SCORING:
- 1.0: Complete information available, high certainty
- 0.8-0.9: Good information available, minor gaps
- 0.6-0.7: Partial information available, some uncertainty
- 0.4-0.5: Limited information available, low certainty
- 0.0-0.3: Insufficient information to provide a reliable answer

EXAMPLES:

Sub-query: "What are the main benefits of renewable energy?"
Context: [Document 1: "Renewable energy sources like solar and wind power produce no direct carbon emissions during operation, making them environmentally friendly alternatives to fossil fuels." Document 2: "Studies show that renewable energy can reduce electricity costs over time due to lower operational expenses."]

Response:
{
    "answer": "Based on the provided context, the main benefits of renewable energy include: 1) Environmental benefits - renewable energy sources like solar and wind produce no direct carbon emissions during operation, making them environmentally friendly alternatives to fossil fuels, and 2) Economic benefits - studies indicate that renewable energy can reduce electricity costs over time due to lower operational expenses.",
    "evidence_docs": ["Document 1", "Document 2"],
    "confidence": 0.8,
    "reasoning": "The context provides clear information about environmental and economic benefits from two different sources",
    "limitations": "The context may not cover all potential benefits of renewable energy"
}"""

    def __init__(self, agent_id: str = None, **kwargs):
        """
        Initialize the Synthesizer Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            **kwargs: Additional configuration parameters
        """
        if agent_id is None:
            agent_id = f"synthesizer_{int(time.time())}"
        
        config = AgentConfig(
            agent_id=agent_id,
            agent_type="synthesizer",
            temperature=0.2,  # Low temperature for consistent synthesis
            max_tokens=1200,
            **kwargs
        )
        
        super().__init__(config)
        
        # Synthesizer-specific settings
        self.max_context_length = 4000  # Max characters from all documents
        self.min_confidence_threshold = 0.3
        self.require_evidence_citation = True
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Synthesizer Agent."""
        return self.SYSTEM_PROMPT
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process a sub-query with retrieved documents to generate a synthesis.
        
        Args:
            input_data: Dictionary containing 'sub_query' and 'documents'
            **kwargs: Additional processing parameters
            
        Returns:
            Dictionary with synthesis and metadata
        """
        self.log_processing_start(input_data)
        
        try:
            # Validate input
            self.validate_input(input_data, ['sub_query', 'documents'])
            sub_query = input_data['sub_query']
            documents = input_data['documents']
            
            # Check if we have documents to work with
            if not documents:
                no_docs_result = {
                    "answer": "No context documents were provided to answer this sub-query.",
                    "evidence_docs": [],
                    "confidence": 0.0,
                    "reasoning": "No source documents available for synthesis",
                    "limitations": "Cannot provide an answer without source material",
                    "sub_query": sub_query,
                    "success": True,
                    "synthesis_metadata": {
                        "document_count": 0,
                        "total_context_length": 0
                    }
                }
                self.log_processing_end(no_docs_result, True)
                return no_docs_result
            
            # Prepare context from documents
            context_info = self._prepare_context(documents)
            context_text = context_info["context_text"]
            doc_mapping = context_info["doc_mapping"]
            
            # Check context length
            if len(context_text) > self.max_context_length:
                context_text = context_text[:self.max_context_length] + "..."
                logger.warning(f"Context truncated to {self.max_context_length} characters")
            
            # Create synthesis prompt
            prompt = self._create_synthesis_prompt(sub_query, context_text)
            
            # Invoke LLM
            llm_result = self.invoke_llm([self.create_human_message(prompt)])
            
            if not llm_result['success']:
                error_result = {
                    "answer": "Failed to generate synthesis due to LLM error.",
                    "evidence_docs": [],
                    "confidence": 0.0,
                    "reasoning": "LLM call failed",
                    "limitations": "Technical error prevented synthesis generation",
                    "sub_query": sub_query,
                    "success": False,
                    "error": llm_result.get('error', 'Unknown error')
                }
                self.log_processing_end(error_result, False)
                return error_result
            
            # Parse LLM response
            parsed_result = self._parse_synthesis_response(llm_result['content'])
            
            if not parsed_result:
                # Fallback: create basic synthesis
                fallback_synthesis = self._create_fallback_synthesis(sub_query, documents)
                result = {
                    **fallback_synthesis,
                    "sub_query": sub_query,
                    "success": True,
                    "synthesis_metadata": {
                        "document_count": len(documents),
                        "total_context_length": len(context_text),
                        "used_fallback": True,
                        "tokens_used": llm_result['tokens_used'],
                        "response_time": llm_result['response_time']
                    }
                }
            else:
                # Validate and enhance parsed result
                validated_result = self._validate_synthesis_result(parsed_result, doc_mapping)
                result = {
                    **validated_result,
                    "sub_query": sub_query,
                    "success": True,
                    "synthesis_metadata": {
                        "document_count": len(documents),
                        "total_context_length": len(context_text),
                        "used_fallback": False,
                        "tokens_used": llm_result['tokens_used'],
                        "response_time": llm_result['response_time'],
                        "original_confidence": parsed_result.get('confidence', 0.5)
                    }
                }
            
            self.log_processing_end(result, True)
            return result
            
        except Exception as e:
            error_result = {
                "answer": f"Error during synthesis: {str(e)}",
                "evidence_docs": [],
                "confidence": 0.0,
                "reasoning": "Exception occurred during processing",
                "limitations": "Technical error prevented synthesis",
                "sub_query": input_data.get('sub_query', ''),
                "success": False,
                "error": str(e)
            }
            self.log_processing_end(error_result, False)
            return error_result
    
    def _prepare_context(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare context text from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Dictionary with context text and document mapping
        """
        context_parts = []
        doc_mapping = {}
        
        for i, doc in enumerate(documents):
            doc_id = doc.get('doc_id', f'doc_{i}')
            content = doc.get('content', '')
            score = doc.get('score', 0.0)
            
            # Create a readable document reference
            doc_ref = f"Document {i+1}"
            doc_mapping[doc_ref] = {
                "doc_id": doc_id,
                "score": score,
                "source": doc.get('source', 'unknown')
            }
            
            # Add document to context
            context_parts.append(f"{doc_ref}: {content}")
        
        context_text = "\n\n".join(context_parts)
        
        return {
            "context_text": context_text,
            "doc_mapping": doc_mapping,
            "total_documents": len(documents),
            "total_length": len(context_text)
        }
    
    def _create_synthesis_prompt(self, sub_query: str, context: str) -> str:
        """
        Create the synthesis prompt for the LLM.
        
        Args:
            sub_query: The sub-query to answer
            context: Context text from documents
            
        Returns:
            Formatted prompt string
        """
        return self.format_prompt(
            """Sub-query: {sub_query}

Context Documents:
{context}

Please provide a synthesis following the guidelines and format specified in the system prompt.""",
            sub_query=sub_query,
            context=context
        )
    
    def _parse_synthesis_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse the LLM synthesis response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed synthesis dictionary or None if parsing failed
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
            required_fields = ['answer', 'confidence']
            if not all(field in parsed for field in required_fields):
                return None
            
            return parsed
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse synthesis response: {e}")
            return None
    
    def _validate_synthesis_result(self, result: Dict[str, Any], doc_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and enhance the synthesis result.
        
        Args:
            result: Parsed synthesis result
            doc_mapping: Document mapping for validation
            
        Returns:
            Validated and enhanced result
        """
        # Ensure all required fields are present with defaults
        validated = {
            "answer": result.get('answer', 'No answer provided'),
            "evidence_docs": result.get('evidence_docs', []),
            "confidence": max(0.0, min(1.0, result.get('confidence', 0.5))),
            "reasoning": result.get('reasoning', 'No reasoning provided'),
            "limitations": result.get('limitations', 'No limitations specified')
        }
        
        # Validate evidence docs
        valid_evidence_docs = []
        for doc_ref in validated['evidence_docs']:
            if doc_ref in doc_mapping:
                valid_evidence_docs.append(doc_ref)
        validated['evidence_docs'] = valid_evidence_docs
        
        # Adjust confidence based on evidence citation
        if self.require_evidence_citation and not valid_evidence_docs:
            validated['confidence'] *= 0.7  # Reduce confidence if no evidence cited
            validated['limitations'] += " No specific evidence documents cited."
        
        # Ensure minimum confidence threshold
        if validated['confidence'] < self.min_confidence_threshold:
            validated['answer'] = f"Low confidence answer: {validated['answer']}"
        
        return validated
    
    def _create_fallback_synthesis(self, sub_query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a basic fallback synthesis when LLM parsing fails.
        
        Args:
            sub_query: The sub-query
            documents: Retrieved documents
            
        Returns:
            Basic synthesis result
        """
        # Create a simple synthesis by concatenating key information
        content_snippets = []
        evidence_docs = []
        
        for i, doc in enumerate(documents[:3]):  # Use top 3 documents
            content = doc.get('content', '')[:200]  # First 200 chars
            if content:
                content_snippets.append(content)
                evidence_docs.append(f"Document {i+1}")
        
        if content_snippets:
            answer = f"Based on the available documents: {' '.join(content_snippets)}"
        else:
            answer = "No relevant information found in the provided documents."
        
        return {
            "answer": answer,
            "evidence_docs": evidence_docs,
            "confidence": 0.4,  # Low confidence for fallback
            "reasoning": "Fallback synthesis created due to parsing error",
            "limitations": "This is a basic synthesis due to processing limitations"
        }
    
    def evaluate_synthesis_quality(self, synthesis: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the quality of a synthesis.
        
        Args:
            synthesis: Synthesis result to evaluate
            
        Returns:
            Dictionary with quality metrics
        """
        answer = synthesis.get('answer', '')
        evidence_docs = synthesis.get('evidence_docs', [])
        confidence = synthesis.get('confidence', 0.0)
        
        # Calculate basic quality metrics
        metrics = {
            "answer_length_score": min(1.0, len(answer) / 100),  # Normalized by 100 chars
            "evidence_citation_score": min(1.0, len(evidence_docs) / 3),  # Normalized by 3 docs
            "confidence_score": confidence,
            "completeness_score": 1.0 if all(field in synthesis for field in ['answer', 'reasoning', 'limitations']) else 0.5
        }
        
        # Overall quality score (weighted average)
        weights = {
            "answer_length_score": 0.3,
            "evidence_citation_score": 0.3,
            "confidence_score": 0.3,
            "completeness_score": 0.1
        }
        
        overall_score = sum(metrics[key] * weights[key] for key in weights)
        metrics["overall_quality_score"] = overall_score
        
        return metrics
    
    def get_synthesis_stats(self) -> Dict[str, Any]:
        """
        Get statistics about synthesis performance.
        
        Returns:
            Dictionary with synthesis statistics
        """
        return {
            "agent_id": self.agent_id,
            "max_context_length": self.max_context_length,
            "min_confidence_threshold": self.min_confidence_threshold,
            "require_evidence_citation": self.require_evidence_citation,
            "total_syntheses": self.metrics.total_calls,
            "success_rate": self.metrics.success_rate,
            "avg_response_time": self.metrics.avg_response_time,
            "reputation_score": self.metrics.reputation_score
        }
    
    def update_synthesis_settings(self,
                                 max_context_length: int = None,
                                 min_confidence_threshold: float = None,
                                 require_evidence_citation: bool = None):
        """
        Update synthesis settings.
        
        Args:
            max_context_length: Maximum context length in characters
            min_confidence_threshold: Minimum confidence threshold
            require_evidence_citation: Whether to require evidence citation
        """
        if max_context_length is not None:
            self.max_context_length = max(1000, max_context_length)
        
        if min_confidence_threshold is not None:
            self.min_confidence_threshold = max(0.0, min(1.0, min_confidence_threshold))
        
        if require_evidence_citation is not None:
            self.require_evidence_citation = require_evidence_citation
        
        logger.info(f"Updated synthesis settings for agent {self.agent_id}")


def create_synthesizer_agent(agent_id: str = None, **kwargs) -> SynthesizerAgent:
    """
    Factory function to create a Synthesizer Agent.
    
    Args:
        agent_id: Optional agent identifier
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured SynthesizerAgent instance
    """
    return SynthesizerAgent(agent_id=agent_id, **kwargs)