"""Main agentic controller for Agentic Hierarchical RAG."""

import uuid
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
import logging
from dataclasses import dataclass

from ..core import (
    Query, QueryType, HierarchicalTree, HierarchicalNode,
    RetrievalResponse, GenerationResponse, CritiqueResponse
)
from ..retrieval import HierarchicalRetriever
from .query_analyzer import QueryAnalyzer
from .reflection_tokens import ReflectionToken, ReflectionTokens

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Tracks the state of the agentic controller during processing."""
    query: Query
    current_level: int
    visited_nodes: List[str]
    retrieved_nodes: List[HierarchicalNode]
    iteration_count: int
    max_iterations: int
    reflection_history: List[ReflectionTokens]
    
    def can_continue(self) -> bool:
        """Check if agent can continue iterating."""
        return self.iteration_count < self.max_iterations


class AgenticController:
    """Orchestrates the agentic hierarchical RAG process.
    
    This controller implements the key innovation of AH-RAG: intelligent,
    adaptive retrieval that dynamically navigates the hierarchical tree
    based on query requirements and self-reflection.
    """
    
    def __init__(
        self,
        tree: HierarchicalTree,
        retriever: HierarchicalRetriever,
        generator: Callable[[str, str], Dict[str, Any]],
        query_analyzer: Optional[QueryAnalyzer] = None,
        reflection_model: Optional[Callable[[str, str], ReflectionTokens]] = None,
        max_iterations: int = 5,
        confidence_threshold: float = 0.7,
    ):
        """Initialize the agentic controller.
        
        Args:
            tree: Hierarchical tree to navigate
            retriever: Retriever for finding relevant nodes
            generator: LLM generator function
            query_analyzer: Query analyzer (creates default if None)
            reflection_model: Model for generating reflection tokens
            max_iterations: Maximum retrieval iterations
            confidence_threshold: Minimum confidence for stopping
        """
        self.tree = tree
        self.retriever = retriever
        self.generator = generator
        self.query_analyzer = query_analyzer or QueryAnalyzer()
        self.reflection_model = reflection_model
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        
    def process_query(self, query_text: str) -> GenerationResponse:
        """Process a query through the agentic hierarchical RAG pipeline.
        
        Args:
            query_text: User query text
            
        Returns:
            GenerationResponse with answer and metadata
        """
        start_time = time.time()
        
        # Create query object
        query = Query(
            query_id=str(uuid.uuid4()),
            text=query_text
        )
        
        # Step 1: Analyze query
        logger.info(f"Analyzing query: {query_text}")
        query_analysis = self.query_analyzer.analyze_query(query)
        
        # Step 2: Generate initial reflection - do we need retrieval?
        initial_reflection = self._generate_reflection(
            query_text,
            "",
            "initial_retrieval_decision"
        )
        
        # If no retrieval needed, generate directly
        if not initial_reflection.requires_retrieval():
            logger.info("No retrieval needed, generating from internal knowledge")
            return self._generate_without_retrieval(query, start_time)
        
        # Step 3: Initialize agent state
        state = AgentState(
            query=query,
            current_level=query_analysis.suggested_levels[0],
            visited_nodes=[],
            retrieved_nodes=[],
            iteration_count=0,
            max_iterations=self.max_iterations,
            reflection_history=[initial_reflection]
        )
        
        # Step 4: Iterative retrieval with self-reflection
        logger.info("Starting iterative retrieval process")
        retrieval_response = self._iterative_retrieval(state, query_analysis)
        
        # Step 5: Generate final answer
        logger.info("Generating final answer")
        generation_response = self._generate_answer(
            query,
            retrieval_response,
            start_time
        )
        
        return generation_response
    
    def _iterative_retrieval(
        self,
        state: AgentState,
        query_analysis: Any
    ) -> RetrievalResponse:
        """Perform iterative retrieval with self-reflection.
        
        Args:
            state: Current agent state
            query_analysis: Initial query analysis
            
        Returns:
            RetrievalResponse with retrieved nodes
        """
        retrieval_start = time.time()
        levels_searched = []
        total_examined = 0
        
        while state.can_continue():
            state.iteration_count += 1
            logger.info(f"Iteration {state.iteration_count}, Level {state.current_level}")
            
            # Retrieve at current level
            level_nodes = self.retriever.retrieve_at_level(
                state.query,
                state.current_level,
                top_k=5
            )
            
            # Filter out already visited nodes
            new_nodes = [
                node for node in level_nodes
                if node.node_id not in state.visited_nodes
            ]
            
            if not new_nodes:
                logger.info("No new nodes found at current level")
                # Try to navigate to a different level
                state = self._navigate_to_new_level(state, query_analysis)
                continue
            
            # Add new nodes to retrieved set
            state.retrieved_nodes.extend(new_nodes)
            state.visited_nodes.extend([n.node_id for n in new_nodes])
            levels_searched.append(state.current_level)
            total_examined += len(new_nodes)
            
            # Generate context from retrieved nodes
            context = self._create_context(state.retrieved_nodes)
            
            # Self-reflect on retrieved content
            reflection = self._generate_reflection(
                state.query.text,
                context,
                "retrieval_assessment"
            )
            state.reflection_history.append(reflection)
            
            # Check if we have sufficient information
            if reflection.is_sufficient():
                logger.info("Sufficient information retrieved")
                break
            
            # Decide navigation based on reflection
            if reflection.should_zoom_in():
                logger.info("Zooming in to more detailed level")
                state.current_level = max(0, state.current_level - 1)
            elif reflection.should_zoom_out():
                logger.info("Zooming out to more abstract level")
                max_level = self.tree.get_max_depth()
                state.current_level = min(max_level, state.current_level + 1)
            elif reflection.should_explore_siblings():
                logger.info("Exploring sibling nodes")
                # Stay at same level but will get different nodes
                pass
            else:
                # Move to next suggested level
                state = self._navigate_to_new_level(state, query_analysis)
        
        # Create final retrieval response
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        return RetrievalResponse(
            query=state.query,
            retrieved_nodes=state.retrieved_nodes,
            retrieval_strategy=query_analysis.search_strategy,
            levels_searched=list(set(levels_searched)),
            total_nodes_examined=total_examined,
            retrieval_time_ms=retrieval_time,
            metadata={
                "iterations": state.iteration_count,
                "reflection_history": [r.to_dict() for r in state.reflection_history]
            }
        )
    
    def _navigate_to_new_level(
        self,
        state: AgentState,
        query_analysis: Any
    ) -> AgentState:
        """Navigate to a new level based on query analysis.
        
        Args:
            state: Current agent state
            query_analysis: Query analysis with suggested levels
            
        Returns:
            Updated agent state
        """
        # Find next unvisited suggested level
        for level in query_analysis.suggested_levels:
            if level != state.current_level:
                # Check if we have unvisited nodes at this level
                level_nodes = self.tree.get_nodes_at_level(level)
                unvisited = [
                    n for n in level_nodes
                    if n.node_id not in state.visited_nodes
                ]
                if unvisited:
                    state.current_level = level
                    break
        
        return state
    
    def _create_context(self, nodes: List[HierarchicalNode]) -> str:
        """Create context string from retrieved nodes.
        
        Args:
            nodes: Retrieved nodes
            
        Returns:
            Formatted context string
        """
        contexts = []
        
        for node in nodes:
            # Include level and type information
            header = f"[Level {node.level} - {node.node_type.value}]"
            content = node.get_text_for_retrieval()
            contexts.append(f"{header}\n{content}")
        
        return "\n\n---\n\n".join(contexts)
    
    def _generate_reflection(
        self,
        query: str,
        context: str,
        reflection_type: str
    ) -> ReflectionTokens:
        """Generate reflection tokens for current state.
        
        Args:
            query: Query text
            context: Current context
            reflection_type: Type of reflection needed
            
        Returns:
            ReflectionTokens
        """
        if self.reflection_model:
            return self.reflection_model(query, context)
        else:
            # Simple rule-based reflection
            return self._rule_based_reflection(query, context, reflection_type)
    
    def _rule_based_reflection(
        self,
        query: str,
        context: str,
        reflection_type: str
    ) -> ReflectionTokens:
        """Simple rule-based reflection generation.
        
        Args:
            query: Query text
            context: Current context
            reflection_type: Type of reflection
            
        Returns:
            ReflectionTokens
        """
        reflection = ReflectionTokens()
        
        if reflection_type == "initial_retrieval_decision":
            # Always retrieve for now in rule-based mode
            reflection.retrieval_decision = ReflectionToken.RETRIEVE_YES
            reflection.reasoning = "Retrieval needed to answer query"
        
        elif reflection_type == "retrieval_assessment":
            # Check context length and relevance
            if not context:
                reflection.sufficiency_assessment = ReflectionToken.INSUFFICIENT
                reflection.navigation_decision = ReflectionToken.EXPLORE_SIBLINGS
            elif len(context) < 500:
                reflection.sufficiency_assessment = ReflectionToken.PARTIALLY_SUFFICIENT
                reflection.navigation_decision = ReflectionToken.ZOOM_IN
            else:
                reflection.sufficiency_assessment = ReflectionToken.SUFFICIENT
                reflection.navigation_decision = ReflectionToken.STAY_LEVEL
            
            # Simple relevance check
            query_words = set(query.lower().split())
            context_words = set(context.lower().split())
            overlap = len(query_words & context_words) / len(query_words)
            
            if overlap > 0.5:
                reflection.relevance_assessment = ReflectionToken.RELEVANT
            elif overlap > 0.2:
                reflection.relevance_assessment = ReflectionToken.PARTIALLY_RELEVANT
            else:
                reflection.relevance_assessment = ReflectionToken.IRRELEVANT
        
        # Set confidence based on assessments
        if (reflection.sufficiency_assessment == ReflectionToken.SUFFICIENT and
            reflection.relevance_assessment == ReflectionToken.RELEVANT):
            reflection.confidence_level = ReflectionToken.HIGH_CONFIDENCE
        elif (reflection.sufficiency_assessment == ReflectionToken.INSUFFICIENT or
              reflection.relevance_assessment == ReflectionToken.IRRELEVANT):
            reflection.confidence_level = ReflectionToken.LOW_CONFIDENCE
        else:
            reflection.confidence_level = ReflectionToken.MEDIUM_CONFIDENCE
        
        return reflection
    
    def _generate_answer(
        self,
        query: Query,
        retrieval_response: RetrievalResponse,
        start_time: float
    ) -> GenerationResponse:
        """Generate final answer from retrieved context.
        
        Args:
            query: Original query
            retrieval_response: Retrieval results
            start_time: Process start time
            
        Returns:
            GenerationResponse
        """
        generation_start = time.time()
        
        # Create prompt with context
        context = retrieval_response.get_context()
        prompt = self._create_generation_prompt(query.text, context)
        
        # Generate answer
        generation_result = self.generator(prompt, query.text)
        
        # Extract answer and metadata
        answer = generation_result.get("answer", "")
        model_name = generation_result.get("model", "unknown")
        total_tokens = generation_result.get("total_tokens", 0)
        prompt_tokens = generation_result.get("prompt_tokens", 0)
        completion_tokens = generation_result.get("completion_tokens", 0)
        
        # Generate final critique
        critique = self._generate_final_critique(query.text, context, answer)
        
        # Extract citations from answer
        citations = self._extract_citations(answer, retrieval_response.retrieved_nodes)
        
        generation_time = (time.time() - generation_start) * 1000
        
        return GenerationResponse(
            query=query,
            answer=answer,
            retrieval_response=retrieval_response,
            model_name=model_name,
            generation_time_ms=generation_time,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            confidence_score=critique.confidence,
            citations=citations,
            metadata={
                "critique": critique.to_dict(),
                "total_time_ms": (time.time() - start_time) * 1000
            }
        )
    
    def _generate_without_retrieval(
        self,
        query: Query,
        start_time: float
    ) -> GenerationResponse:
        """Generate answer without retrieval.
        
        Args:
            query: Query object
            start_time: Process start time
            
        Returns:
            GenerationResponse
        """
        # Create empty retrieval response
        retrieval_response = RetrievalResponse(
            query=query,
            retrieved_nodes=[],
            retrieval_strategy="none",
            levels_searched=[],
            total_nodes_examined=0,
            retrieval_time_ms=0,
            metadata={"no_retrieval": True}
        )
        
        # Generate directly
        prompt = f"Answer the following question: {query.text}"
        generation_result = self.generator(prompt, query.text)
        
        return GenerationResponse(
            query=query,
            answer=generation_result.get("answer", ""),
            retrieval_response=retrieval_response,
            model_name=generation_result.get("model", "unknown"),
            generation_time_ms=(time.time() - start_time) * 1000,
            total_tokens=generation_result.get("total_tokens", 0),
            prompt_tokens=generation_result.get("prompt_tokens", 0),
            completion_tokens=generation_result.get("completion_tokens", 0),
            confidence_score=0.5,
            citations=[],
            metadata={"no_retrieval": True}
        )
    
    def _create_generation_prompt(self, query: str, context: str) -> str:
        """Create prompt for generation.
        
        Args:
            query: Query text
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are an AI assistant using a hierarchical knowledge base to answer questions.

Context from the knowledge base:
{context}

Question: {query}

Please provide a comprehensive answer based on the provided context. If the context contains information from different abstraction levels (marked with [Level X]), integrate them appropriately. Cite specific levels or sections when making claims.

Answer:"""
        
        return prompt
    
    def _generate_final_critique(
        self,
        query: str,
        context: str,
        answer: str
    ) -> CritiqueResponse:
        """Generate final critique of the answer.
        
        Args:
            query: Query text
            context: Retrieved context
            answer: Generated answer
            
        Returns:
            CritiqueResponse
        """
        # Simple rule-based critique for now
        critique = CritiqueResponse()
        
        # Check if answer addresses the query
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(query_words & answer_words) / len(query_words)
        
        if overlap > 0.7:
            critique.answer_quality = "supported"
            critique.confidence = 0.8
        elif overlap > 0.4:
            critique.answer_quality = "partial"
            critique.confidence = 0.5
        else:
            critique.answer_quality = "unsupported"
            critique.confidence = 0.2
        
        # Check context sufficiency
        if len(context) > 1000:
            critique.retrieval_quality = "sufficient"
        elif len(context) > 500:
            critique.retrieval_quality = "partial"
        else:
            critique.retrieval_quality = "insufficient"
        
        critique.reasoning = f"Answer overlap: {overlap:.2f}, Context length: {len(context)}"
        
        return critique
    
    def _extract_citations(
        self,
        answer: str,
        nodes: List[HierarchicalNode]
    ) -> List[str]:
        """Extract citations from answer.
        
        Args:
            answer: Generated answer
            nodes: Retrieved nodes
            
        Returns:
            List of citations
        """
        citations = []
        
        # Look for level references in answer
        import re
        level_pattern = r"\[Level (\d+)[^\]]*\]"
        matches = re.findall(level_pattern, answer)
        
        for match in matches:
            level = int(match)
            # Find nodes at this level
            level_nodes = [n for n in nodes if n.level == level]
            for node in level_nodes:
                citation = f"Level {level} - Node {node.node_id[:8]}"
                if citation not in citations:
                    citations.append(citation)
        
        return citations