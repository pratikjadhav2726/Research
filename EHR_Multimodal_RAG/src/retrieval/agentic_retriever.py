"""
Agentic Retriever for EHR Multimodal RAG System

This module implements an intelligent retrieval system that can reason about
its own retrieval process, adapt queries based on clinical context, and
perform multi-step reasoning using the AR-MCTS framework adapted for healthcare.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
from collections import defaultdict
import math
import random

from ..embedding.multimodal_encoder import MultimodalEncoder, MultimodalEmbedding
from ..utils.clinical_validators import validate_clinical_query
from .clinical_reasoner import ClinicalReasoner
from .context_synthesizer import ContextSynthesizer


class RetrievalAction(Enum):
    """Types of retrieval actions the agent can take."""
    QUERY_REFORMULATION = "query_reformulation"
    MODALITY_FOCUS = "modality_focus"
    TEMPORAL_EXPANSION = "temporal_expansion"
    CONCEPT_EXPANSION = "concept_expansion"
    CROSS_REFERENCE = "cross_reference"
    VALIDATION_CHECK = "validation_check"


@dataclass
class RetrievalState:
    """State representation for the retrieval process."""
    original_query: str
    current_query: str
    retrieved_items: List[Dict[str, Any]] = field(default_factory=list)
    clinical_concepts: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    iteration: int = 0
    actions_taken: List[RetrievalAction] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCTSNode:
    """Monte Carlo Tree Search node for retrieval planning."""
    state: RetrievalState
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    action: Optional[RetrievalAction] = None
    untried_actions: Set[RetrievalAction] = field(default_factory=lambda: set(RetrievalAction))
    
    @property
    def average_value(self) -> float:
        """Calculate average value of this node."""
        return self.value_sum / self.visits if self.visits > 0 else 0.0
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0
    
    def best_child(self, exploration_weight: float = 1.414) -> 'MCTSNode':
        """Select best child using UCB1."""
        if not self.children:
            return None
        
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                return child  # Prioritize unvisited children
            
            # UCB1 formula
            exploitation = child.average_value
            exploration = exploration_weight * math.sqrt(
                math.log(self.visits) / child.visits
            )
            score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child


class AgenticRetriever:
    """
    Intelligent retrieval agent using AR-MCTS framework for clinical reasoning.
    
    This retriever can:
    - Reason about retrieval quality and adapt strategies
    - Perform multi-step clinical reasoning
    - Actively seek additional knowledge when needed
    - Validate retrieved information for clinical accuracy
    """
    
    def __init__(self, config: Dict[str, Any], 
                 encoder: MultimodalEncoder,
                 vector_db: Any):
        self.config = config
        self.encoder = encoder
        self.vector_db = vector_db
        self.logger = logging.getLogger(__name__)
        
        # Initialize clinical reasoner
        self.clinical_reasoner = ClinicalReasoner(config.get('clinical_reasoning', {}))
        
        # Initialize context synthesizer
        self.context_synthesizer = ContextSynthesizer(config.get('context_synthesis', {}))
        
        # Retrieval parameters
        self.max_iterations = config.get('max_iterations', 3)
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.top_k = config.get('top_k', 20)
        
        # MCTS parameters
        self.mcts_simulations = config.get('mcts_simulations', 50)
        self.exploration_weight = config.get('exploration_weight', 1.414)
        
        # Clinical knowledge integration
        self.medical_ontologies = config.get('medical_ontologies', {})
        
    async def retrieve(self, query: str, 
                      query_type: str = "general",
                      patient_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main retrieval method using agentic reasoning.
        
        Args:
            query: User query
            query_type: Type of clinical query (diagnosis, treatment, etc.)
            patient_context: Optional patient context for personalization
            
        Returns:
            Dictionary containing retrieved items and reasoning trace
        """
        # Validate clinical query
        if not validate_clinical_query(query):
            self.logger.warning(f"Clinical query validation failed: {query}")
        
        # Initialize retrieval state
        initial_state = RetrievalState(
            original_query=query,
            current_query=query,
            metadata={
                'query_type': query_type,
                'patient_context': patient_context or {}
            }
        )
        
        # Perform initial retrieval
        initial_results = await self._perform_retrieval(initial_state)
        initial_state.retrieved_items = initial_results['items']
        initial_state.confidence_score = initial_results['confidence']
        
        # Check if initial retrieval is sufficient
        if initial_state.confidence_score >= self.confidence_threshold:
            return await self._format_results(initial_state)
        
        # Use MCTS for iterative improvement
        final_state = await self._mcts_retrieval(initial_state)
        
        return await self._format_results(final_state)
    
    async def _mcts_retrieval(self, initial_state: RetrievalState) -> RetrievalState:
        """
        Use Monte Carlo Tree Search for retrieval optimization.
        
        Args:
            initial_state: Initial retrieval state
            
        Returns:
            Optimized retrieval state
        """
        root = MCTSNode(state=initial_state)
        
        for _ in range(self.mcts_simulations):
            # Selection: Navigate to leaf node
            node = self._select_node(root)
            
            # Expansion: Add new child if not terminal
            if not self._is_terminal(node.state) and not node.is_fully_expanded():
                node = await self._expand_node(node)
            
            # Simulation: Evaluate the state
            value = await self._simulate(node.state)
            
            # Backpropagation: Update node values
            self._backpropagate(node, value)
        
        # Select best path
        best_path = self._select_best_path(root)
        return best_path[-1].state if best_path else initial_state
    
    def _select_node(self, root: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCB1."""
        current = root
        
        while current.children and current.is_fully_expanded():
            current = current.best_child(self.exploration_weight)
        
        return current
    
    async def _expand_node(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding a new child."""
        if not node.untried_actions:
            return node
        
        # Select random untried action
        action = random.choice(list(node.untried_actions))
        node.untried_actions.remove(action)
        
        # Apply action to create new state
        new_state = await self._apply_action(node.state, action)
        
        # Create new child node
        child = MCTSNode(
            state=new_state,
            parent=node,
            action=action
        )
        
        node.children.append(child)
        return child
    
    async def _apply_action(self, state: RetrievalState, 
                          action: RetrievalAction) -> RetrievalState:
        """
        Apply a retrieval action to create a new state.
        
        Args:
            state: Current retrieval state
            action: Action to apply
            
        Returns:
            New retrieval state
        """
        new_state = RetrievalState(
            original_query=state.original_query,
            current_query=state.current_query,
            retrieved_items=state.retrieved_items.copy(),
            clinical_concepts=state.clinical_concepts.copy(),
            confidence_score=state.confidence_score,
            iteration=state.iteration + 1,
            actions_taken=state.actions_taken + [action],
            reasoning_trace=state.reasoning_trace.copy(),
            metadata=state.metadata.copy()
        )
        
        if action == RetrievalAction.QUERY_REFORMULATION:
            new_state = await self._reformulate_query(new_state)
        
        elif action == RetrievalAction.MODALITY_FOCUS:
            new_state = await self._focus_modality(new_state)
        
        elif action == RetrievalAction.TEMPORAL_EXPANSION:
            new_state = await self._expand_temporal(new_state)
        
        elif action == RetrievalAction.CONCEPT_EXPANSION:
            new_state = await self._expand_concepts(new_state)
        
        elif action == RetrievalAction.CROSS_REFERENCE:
            new_state = await self._cross_reference(new_state)
        
        elif action == RetrievalAction.VALIDATION_CHECK:
            new_state = await self._validate_results(new_state)
        
        return new_state
    
    async def _reformulate_query(self, state: RetrievalState) -> RetrievalState:
        """Reformulate query based on clinical reasoning."""
        reasoning = await self.clinical_reasoner.reason_about_query(
            state.current_query,
            state.retrieved_items,
            state.metadata.get('patient_context', {})
        )
        
        if reasoning.get('reformulated_query'):
            state.current_query = reasoning['reformulated_query']
            state.reasoning_trace.append(
                f"Reformulated query: {reasoning.get('reasoning', 'No reasoning provided')}"
            )
            
            # Perform new retrieval
            results = await self._perform_retrieval(state)
            state.retrieved_items = results['items']
            state.confidence_score = results['confidence']
        
        return state
    
    async def _focus_modality(self, state: RetrievalState) -> RetrievalState:
        """Focus retrieval on specific modality based on query analysis."""
        # Analyze query to determine best modality
        modality_scores = await self._analyze_modality_relevance(state.current_query)
        best_modality = max(modality_scores.items(), key=lambda x: x[1])[0]
        
        # Filter results by modality
        filtered_items = [
            item for item in state.retrieved_items
            if item.get('modality', '').lower() == best_modality.lower()
        ]
        
        if filtered_items:
            state.retrieved_items = filtered_items
            state.reasoning_trace.append(f"Focused on {best_modality} modality")
            
            # Recalculate confidence
            state.confidence_score = await self._calculate_confidence(state)
        
        return state
    
    async def _expand_temporal(self, state: RetrievalState) -> RetrievalState:
        """Expand retrieval to include temporal context."""
        patient_context = state.metadata.get('patient_context', {})
        
        if 'patient_id' in patient_context:
            # Retrieve historical data for the patient
            temporal_query = f"patient:{patient_context['patient_id']} timeline"
            temporal_results = await self._perform_retrieval(
                RetrievalState(original_query=temporal_query, current_query=temporal_query)
            )
            
            # Merge with existing results
            state.retrieved_items.extend(temporal_results['items'])
            state.reasoning_trace.append("Expanded temporal context")
            
            # Recalculate confidence
            state.confidence_score = await self._calculate_confidence(state)
        
        return state
    
    async def _expand_concepts(self, state: RetrievalState) -> RetrievalState:
        """Expand query with related clinical concepts."""
        # Extract concepts from current query
        concepts = await self.clinical_reasoner.extract_clinical_concepts(
            state.current_query
        )
        
        # Find related concepts using medical ontologies
        expanded_concepts = await self._find_related_concepts(concepts)
        
        if expanded_concepts:
            # Create expanded query
            concept_terms = " OR ".join(expanded_concepts)
            expanded_query = f"{state.current_query} ({concept_terms})"
            
            # Perform expanded retrieval
            expanded_state = RetrievalState(
                original_query=state.original_query,
                current_query=expanded_query
            )
            results = await self._perform_retrieval(expanded_state)
            
            # Merge results
            state.retrieved_items.extend(results['items'])
            state.clinical_concepts.extend(expanded_concepts)
            state.reasoning_trace.append(f"Expanded with concepts: {expanded_concepts}")
            
            # Recalculate confidence
            state.confidence_score = await self._calculate_confidence(state)
        
        return state
    
    async def _cross_reference(self, state: RetrievalState) -> RetrievalState:
        """Cross-reference findings with medical literature."""
        # Extract key findings from retrieved items
        key_findings = await self._extract_key_findings(state.retrieved_items)
        
        if key_findings:
            # Search medical literature for validation
            literature_query = f"medical literature: {' '.join(key_findings)}"
            literature_results = await self._perform_retrieval(
                RetrievalState(original_query=literature_query, current_query=literature_query)
            )
            
            # Add literature references
            state.retrieved_items.extend(literature_results['items'])
            state.reasoning_trace.append("Added literature cross-references")
            
            # Recalculate confidence
            state.confidence_score = await self._calculate_confidence(state)
        
        return state
    
    async def _validate_results(self, state: RetrievalState) -> RetrievalState:
        """Validate retrieved results for clinical accuracy."""
        validation_results = await self.clinical_reasoner.validate_clinical_information(
            state.retrieved_items,
            state.current_query
        )
        
        # Filter out invalid results
        valid_items = [
            item for item, is_valid in zip(state.retrieved_items, validation_results)
            if is_valid
        ]
        
        if len(valid_items) != len(state.retrieved_items):
            state.retrieved_items = valid_items
            state.reasoning_trace.append("Filtered invalid clinical information")
            
            # Recalculate confidence
            state.confidence_score = await self._calculate_confidence(state)
        
        return state
    
    async def _simulate(self, state: RetrievalState) -> float:
        """Simulate the value of a retrieval state."""
        # Simple simulation based on confidence and completeness
        confidence_score = state.confidence_score
        completeness_score = min(len(state.retrieved_items) / self.top_k, 1.0)
        
        # Penalize too many iterations
        iteration_penalty = max(0, (state.iteration - self.max_iterations) * 0.1)
        
        # Reward clinical concept coverage
        concept_bonus = len(state.clinical_concepts) * 0.05
        
        return confidence_score * 0.6 + completeness_score * 0.3 + concept_bonus - iteration_penalty
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        current = node
        while current is not None:
            current.visits += 1
            current.value_sum += value
            current = current.parent
    
    def _select_best_path(self, root: MCTSNode) -> List[MCTSNode]:
        """Select the best path from root to leaf."""
        path = [root]
        current = root
        
        while current.children:
            current = max(current.children, key=lambda c: c.average_value)
            path.append(current)
        
        return path
    
    def _is_terminal(self, state: RetrievalState) -> bool:
        """Check if retrieval state is terminal."""
        return (
            state.confidence_score >= self.confidence_threshold or
            state.iteration >= self.max_iterations or
            not state.retrieved_items
        )
    
    async def _perform_retrieval(self, state: RetrievalState) -> Dict[str, Any]:
        """Perform actual retrieval using vector database."""
        # Encode query
        query_embedding = self.encoder.encode_clinical_text(state.current_query)
        
        # Search vector database
        results = await self.vector_db.search(
            query_vector=query_embedding.unified_embedding,
            top_k=self.top_k,
            filters=self._build_filters(state)
        )
        
        # Calculate confidence based on similarity scores
        if results:
            avg_similarity = np.mean([r.get('similarity', 0) for r in results])
            confidence = min(avg_similarity * 1.2, 1.0)  # Scale and cap at 1.0
        else:
            confidence = 0.0
        
        return {
            'items': results,
            'confidence': confidence
        }
    
    def _build_filters(self, state: RetrievalState) -> Dict[str, Any]:
        """Build filters for vector database search."""
        filters = {}
        
        # Patient-specific filters
        patient_context = state.metadata.get('patient_context', {})
        if 'patient_id' in patient_context:
            filters['patient_id'] = patient_context['patient_id']
        
        # Query type filters
        query_type = state.metadata.get('query_type')
        if query_type:
            filters['category'] = query_type
        
        return filters
    
    async def _analyze_modality_relevance(self, query: str) -> Dict[str, float]:
        """Analyze which modalities are most relevant for the query."""
        modality_keywords = {
            'image': ['image', 'scan', 'x-ray', 'mri', 'ct', 'ultrasound', 'visual'],
            'text': ['note', 'report', 'description', 'history', 'assessment'],
            'lab': ['lab', 'blood', 'test', 'result', 'value'],
            'vital': ['vital', 'temperature', 'pressure', 'heart rate', 'oxygen']
        }
        
        query_lower = query.lower()
        scores = {}
        
        for modality, keywords in modality_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            scores[modality] = score / len(keywords)
        
        return scores
    
    async def _find_related_concepts(self, concepts: List[str]) -> List[str]:
        """Find related clinical concepts using medical ontologies."""
        related_concepts = []
        
        for concept in concepts:
            # This would integrate with actual medical ontologies
            # For now, return placeholder related concepts
            if concept.lower() in ['diabetes', 'diabetic']:
                related_concepts.extend(['hyperglycemia', 'insulin', 'glucose'])
            elif concept.lower() in ['hypertension', 'high blood pressure']:
                related_concepts.extend(['systolic', 'diastolic', 'cardiovascular'])
        
        return list(set(related_concepts))
    
    async def _extract_key_findings(self, items: List[Dict[str, Any]]) -> List[str]:
        """Extract key clinical findings from retrieved items."""
        findings = []
        
        for item in items:
            if 'findings' in item:
                findings.append(item['findings'])
            elif 'diagnosis' in item:
                findings.append(item['diagnosis'])
            elif 'impression' in item:
                findings.append(item['impression'])
        
        return findings
    
    async def _calculate_confidence(self, state: RetrievalState) -> float:
        """Calculate confidence score for retrieval state."""
        if not state.retrieved_items:
            return 0.0
        
        # Base confidence on similarity scores
        similarities = [item.get('similarity', 0) for item in state.retrieved_items]
        base_confidence = np.mean(similarities) if similarities else 0.0
        
        # Adjust based on clinical concept coverage
        concept_bonus = min(len(state.clinical_concepts) * 0.1, 0.3)
        
        # Adjust based on result diversity
        modalities = set(item.get('modality', '') for item in state.retrieved_items)
        diversity_bonus = min(len(modalities) * 0.05, 0.2)
        
        return min(base_confidence + concept_bonus + diversity_bonus, 1.0)
    
    async def _format_results(self, state: RetrievalState) -> Dict[str, Any]:
        """Format final retrieval results."""
        # Synthesize context from retrieved items
        synthesized_context = await self.context_synthesizer.synthesize(
            state.retrieved_items,
            state.current_query,
            state.clinical_concepts
        )
        
        return {
            'retrieved_items': state.retrieved_items,
            'synthesized_context': synthesized_context,
            'confidence_score': state.confidence_score,
            'clinical_concepts': state.clinical_concepts,
            'reasoning_trace': state.reasoning_trace,
            'final_query': state.current_query,
            'iterations': state.iteration,
            'actions_taken': [action.value for action in state.actions_taken],
            'metadata': state.metadata
        }