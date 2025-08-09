"""
CoRS System Implementation

This module implements the main CoRS (Collaborative Retrieval and Synthesis)
system that orchestrates all agents using LangGraph for workflow management.
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver

from .shared_synthesis_space import SharedSynthesisSpace, CoRSState, SubQueryStatus
from .reputation_weighted_consensus import (
    ReputationWeightedConsensus, 
    SynthesisCandidate, 
    ConsensusStrategy
)
from ..agents.decomposer_agent import DecomposerAgent
from ..agents.retrieval_agent import RetrievalAgent
from ..agents.synthesizer_agent import SynthesizerAgent
from ..agents.critic_agent import CriticAgent
from ..agents.base_agent import AgentPool

logger = logging.getLogger(__name__)


class CoRSWorkflowState(TypedDict):
    """State object for the CoRS workflow."""
    session_id: str
    original_query: str
    current_sub_query_index: int
    sub_queries: List[str]
    current_sub_query: str
    retrieved_documents: List[Dict[str, Any]]
    syntheses: List[Dict[str, Any]]
    critiques: List[Dict[str, Any]]
    consensus_reached: bool
    consensus_output: str
    final_answer: str
    workflow_status: str
    error_message: str
    metadata: Dict[str, Any]


@dataclass
class CoRSConfig:
    """Configuration for the CoRS system."""
    # Agent configuration
    num_synthesizer_agents: int = 3
    num_retrieval_agents: int = 2
    
    # Consensus configuration
    consensus_strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_AVERAGE
    consensus_threshold: float = 0.8
    learning_rate: float = 0.1
    
    # System configuration
    max_sub_queries: int = 5
    max_documents_per_query: int = 5
    enable_parallel_processing: bool = True
    
    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Vector store configuration
    vector_store_type: str = "chromadb"
    
    # Quality thresholds
    min_synthesis_confidence: float = 0.3
    min_critique_score: float = 0.4


class CoRSSystem:
    """
    Main CoRS system that orchestrates collaborative retrieval and synthesis.
    
    This system implements the complete CoRS workflow using LangGraph for
    orchestration and Redis for state management.
    """
    
    def __init__(self, config: CoRSConfig = None):
        """
        Initialize the CoRS system.
        
        Args:
            config: System configuration
        """
        self.config = config or CoRSConfig()
        
        # Initialize core components
        self.sss = SharedSynthesisSpace(
            redis_host=self.config.redis_host,
            redis_port=self.config.redis_port,
            redis_db=self.config.redis_db
        )
        
        self.consensus_mechanism = ReputationWeightedConsensus(
            learning_rate=self.config.learning_rate,
            consensus_threshold=self.config.consensus_threshold,
            strategy=self.config.consensus_strategy
        )
        
        # Initialize agent pool
        self.agent_pool = AgentPool()
        self._initialize_agents()
        
        # Initialize LangGraph workflow
        self.workflow = self._build_workflow()
        
        logger.info("CoRS system initialized successfully")
    
    def _initialize_agents(self):
        """Initialize all agents and add them to the pool."""
        # Create Decomposer Agent
        decomposer = DecomposerAgent()
        self.agent_pool.add_agent(decomposer)
        
        # Create Retrieval Agents
        for i in range(self.config.num_retrieval_agents):
            retrieval_agent = RetrievalAgent(
                agent_id=f"retrieval_{i}",
                vector_store_type=self.config.vector_store_type
            )
            self.agent_pool.add_agent(retrieval_agent)
        
        # Create Synthesizer Agents
        for i in range(self.config.num_synthesizer_agents):
            synthesizer = SynthesizerAgent(agent_id=f"synthesizer_{i}")
            self.agent_pool.add_agent(synthesizer)
        
        # Create Critic Agent
        critic = CriticAgent()
        self.agent_pool.add_agent(critic)
        
        logger.info(f"Initialized {self.agent_pool.get_pool_stats()['total_agents']} agents")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for CoRS."""
        # Create the state graph
        workflow = StateGraph(CoRSWorkflowState)
        
        # Add nodes for each workflow step
        workflow.add_node("decompose_query", self._decompose_query_node)
        workflow.add_node("retrieve_documents", self._retrieve_documents_node)
        workflow.add_node("synthesize_responses", self._synthesize_responses_node)
        workflow.add_node("critique_syntheses", self._critique_syntheses_node)
        workflow.add_node("reach_consensus", self._reach_consensus_node)
        workflow.add_node("check_completion", self._check_completion_node)
        workflow.add_node("final_synthesis", self._final_synthesis_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Define the workflow edges
        workflow.set_entry_point("decompose_query")
        
        # Linear flow for each sub-query
        workflow.add_edge("decompose_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "synthesize_responses")
        workflow.add_edge("synthesize_responses", "critique_syntheses")
        workflow.add_edge("critique_syntheses", "reach_consensus")
        workflow.add_edge("reach_consensus", "check_completion")
        
        # Conditional edges from check_completion
        workflow.add_conditional_edges(
            "check_completion",
            self._route_after_completion_check,
            {
                "continue": "retrieve_documents",  # Process next sub-query
                "finalize": "final_synthesis",     # All sub-queries done
                "error": "handle_error"            # Error occurred
            }
        )
        
        # Terminal nodes
        workflow.add_edge("final_synthesis", END)
        workflow.add_edge("handle_error", END)
        
        # Compile the workflow
        return workflow.compile()
    
    def _decompose_query_node(self, state: CoRSWorkflowState) -> CoRSWorkflowState:
        """Node to decompose the original query into sub-queries."""
        try:
            logger.info(f"Decomposing query for session {state['session_id']}")
            
            # Get decomposer agent
            decomposer = self.agent_pool.get_best_agent("decomposer")
            if not decomposer:
                raise ValueError("No decomposer agent available")
            
            # Decompose the query
            result = decomposer.process({
                "original_query": state["original_query"]
            })
            
            if result["success"]:
                sub_queries = result["sub_queries"][:self.config.max_sub_queries]
                
                # Update SSS
                self.sss.update_sub_queries(state["session_id"], sub_queries)
                
                # Update state
                state.update({
                    "sub_queries": sub_queries,
                    "current_sub_query_index": 0,
                    "current_sub_query": sub_queries[0] if sub_queries else "",
                    "workflow_status": "decomposed",
                    "metadata": {
                        **state.get("metadata", {}),
                        "decomposition_result": result
                    }
                })
            else:
                state.update({
                    "workflow_status": "error",
                    "error_message": f"Query decomposition failed: {result.get('error', 'Unknown error')}"
                })
            
            return state
            
        except Exception as e:
            logger.error(f"Error in decompose_query_node: {e}")
            state.update({
                "workflow_status": "error",
                "error_message": f"Decomposition error: {str(e)}"
            })
            return state
    
    def _retrieve_documents_node(self, state: CoRSWorkflowState) -> CoRSWorkflowState:
        """Node to retrieve documents for the current sub-query."""
        try:
            current_sub_query = state["current_sub_query"]
            logger.info(f"Retrieving documents for sub-query: '{current_sub_query}'")
            
            # Get retrieval agent
            retrieval_agent = self.agent_pool.get_best_agent("retrieval")
            if not retrieval_agent:
                raise ValueError("No retrieval agent available")
            
            # Retrieve documents
            result = retrieval_agent.process({
                "sub_query": current_sub_query,
                "k": self.config.max_documents_per_query
            })
            
            if result["success"]:
                documents = result["documents"]
                
                # Store documents in SSS
                sub_query_id = f"{state['session_id']}_sq_{state['current_sub_query_index']}"
                self.sss.store_retrieved_docs(state["session_id"], sub_query_id, documents)
                
                # Update state
                state.update({
                    "retrieved_documents": documents,
                    "workflow_status": "retrieved",
                    "metadata": {
                        **state.get("metadata", {}),
                        f"retrieval_result_{state['current_sub_query_index']}": result["metadata"]
                    }
                })
            else:
                state.update({
                    "workflow_status": "error",
                    "error_message": f"Document retrieval failed: {result.get('error', 'Unknown error')}"
                })
            
            return state
            
        except Exception as e:
            logger.error(f"Error in retrieve_documents_node: {e}")
            state.update({
                "workflow_status": "error",
                "error_message": f"Retrieval error: {str(e)}"
            })
            return state
    
    def _synthesize_responses_node(self, state: CoRSWorkflowState) -> CoRSWorkflowState:
        """Node to synthesize responses from multiple agents."""
        try:
            current_sub_query = state["current_sub_query"]
            documents = state["retrieved_documents"]
            
            logger.info(f"Synthesizing responses for sub-query: '{current_sub_query}'")
            
            # Get synthesizer agents
            synthesizer_agents = self.agent_pool.get_agents_by_type("synthesizer")
            if not synthesizer_agents:
                raise ValueError("No synthesizer agents available")
            
            # Generate syntheses from multiple agents
            syntheses = []
            for agent in synthesizer_agents[:self.config.num_synthesizer_agents]:
                try:
                    result = agent.process({
                        "sub_query": current_sub_query,
                        "documents": documents
                    })
                    
                    if result["success"]:
                        synthesis_data = {
                            "agent_id": agent.agent_id,
                            "synthesis_id": f"{agent.agent_id}_{int(time.time())}",
                            "answer": result["answer"],
                            "confidence": result.get("confidence", 0.5),
                            "evidence_docs": result.get("evidence_docs", []),
                            "reasoning": result.get("reasoning", ""),
                            "limitations": result.get("limitations", "")
                        }
                        syntheses.append(synthesis_data)
                        
                        # Store synthesis in SSS
                        sub_query_id = f"{state['session_id']}_sq_{state['current_sub_query_index']}"
                        self.sss.store_synthesis(
                            state["session_id"], 
                            sub_query_id, 
                            agent.agent_id,
                            result["answer"],
                            result.get("evidence_docs", [])
                        )
                    
                except Exception as e:
                    logger.warning(f"Synthesis failed for agent {agent.agent_id}: {e}")
                    continue
            
            if not syntheses:
                state.update({
                    "workflow_status": "error",
                    "error_message": "No successful syntheses generated"
                })
            else:
                state.update({
                    "syntheses": syntheses,
                    "workflow_status": "synthesized"
                })
            
            return state
            
        except Exception as e:
            logger.error(f"Error in synthesize_responses_node: {e}")
            state.update({
                "workflow_status": "error",
                "error_message": f"Synthesis error: {str(e)}"
            })
            return state
    
    def _critique_syntheses_node(self, state: CoRSWorkflowState) -> CoRSWorkflowState:
        """Node to critique all syntheses."""
        try:
            syntheses = state["syntheses"]
            documents = state["retrieved_documents"]
            current_sub_query = state["current_sub_query"]
            
            logger.info(f"Critiquing {len(syntheses)} syntheses")
            
            # Get critic agent
            critic = self.agent_pool.get_best_agent("critic")
            if not critic:
                raise ValueError("No critic agent available")
            
            # Critique each synthesis
            critiques = []
            for synthesis in syntheses:
                try:
                    result = critic.process({
                        "synthesis": synthesis,
                        "context_documents": documents,
                        "sub_query": current_sub_query
                    })
                    
                    if result["success"]:
                        critique_data = result["critique"]
                        critiques.append(critique_data)
                        
                        # Store critique in SSS
                        sub_query_id = f"{state['session_id']}_sq_{state['current_sub_query_index']}"
                        self.sss.store_critique(
                            state["session_id"],
                            sub_query_id,
                            synthesis["synthesis_id"],
                            critic.agent_id,
                            critique_data["overall_score"],
                            critique_data.get("feedback", "")
                        )
                    
                except Exception as e:
                    logger.warning(f"Critique failed for synthesis {synthesis.get('synthesis_id', 'unknown')}: {e}")
                    continue
            
            if not critiques:
                state.update({
                    "workflow_status": "error",
                    "error_message": "No successful critiques generated"
                })
            else:
                state.update({
                    "critiques": critiques,
                    "workflow_status": "critiqued"
                })
            
            return state
            
        except Exception as e:
            logger.error(f"Error in critique_syntheses_node: {e}")
            state.update({
                "workflow_status": "error",
                "error_message": f"Critique error: {str(e)}"
            })
            return state
    
    def _reach_consensus_node(self, state: CoRSWorkflowState) -> CoRSWorkflowState:
        """Node to reach consensus among syntheses."""
        try:
            syntheses = state["syntheses"]
            critiques = state["critiques"]
            current_sub_query = state["current_sub_query"]
            
            logger.info("Reaching consensus among syntheses")
            
            # Prepare synthesis candidates for consensus
            candidates = []
            critique_map = {c["synthesis_id"]: c for c in critiques}
            
            for synthesis in syntheses:
                synthesis_id = synthesis["synthesis_id"]
                agent_id = synthesis["agent_id"]
                
                # Get critique score
                critique = critique_map.get(synthesis_id)
                critique_score = critique["overall_score"] if critique else 0.3
                
                # Get agent reputation
                agent_reputation = self.sss.get_agent_reputation(agent_id)
                
                # Create synthesis candidate
                candidate = SynthesisCandidate(
                    synthesis_id=synthesis_id,
                    agent_id=agent_id,
                    text=synthesis["answer"],
                    critique_score=critique_score,
                    agent_reputation=agent_reputation,
                    evidence_docs=synthesis.get("evidence_docs", []),
                    created_at=time.time()
                )
                candidates.append(candidate)
            
            if not candidates:
                state.update({
                    "workflow_status": "error",
                    "error_message": "No synthesis candidates available for consensus"
                })
                return state
            
            # Reach consensus
            consensus_result = self.consensus_mechanism.reach_consensus(
                candidates, current_sub_query
            )
            
            # Update agent reputations
            for candidate in candidates:
                current_reputation = self.sss.get_agent_reputation(candidate.agent_id)
                new_reputation = self.consensus_mechanism.update_agent_reputation(
                    candidate.agent_id,
                    current_reputation,
                    candidate.critique_score
                )
                self.sss.update_agent_reputation(candidate.agent_id, new_reputation)
                
                # Update agent object reputation
                agent = self.agent_pool.get_agent(candidate.agent_id)
                if agent:
                    agent.update_reputation(new_reputation)
            
            # Store consensus in SSS
            sub_query_id = f"{state['session_id']}_sq_{state['current_sub_query_index']}"
            self.sss.store_consensus(
                state["session_id"],
                sub_query_id,
                consensus_result.winning_synthesis_id,
                consensus_result.consensus_output
            )
            
            # Update state
            state.update({
                "consensus_reached": True,
                "consensus_output": consensus_result.consensus_output,
                "workflow_status": "consensus_reached",
                "metadata": {
                    **state.get("metadata", {}),
                    f"consensus_result_{state['current_sub_query_index']}": {
                        "winning_agent": consensus_result.winning_agent_id,
                        "confidence": consensus_result.confidence_score,
                        "strategy": consensus_result.consensus_strategy.value
                    }
                }
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error in reach_consensus_node: {e}")
            state.update({
                "workflow_status": "error",
                "error_message": f"Consensus error: {str(e)}"
            })
            return state
    
    def _check_completion_node(self, state: CoRSWorkflowState) -> CoRSWorkflowState:
        """Node to check if all sub-queries have been processed."""
        try:
            current_index = state["current_sub_query_index"]
            sub_queries = state["sub_queries"]
            
            # Check if we've processed all sub-queries
            if current_index >= len(sub_queries) - 1:
                state.update({
                    "workflow_status": "all_sub_queries_complete"
                })
            else:
                # Move to next sub-query
                next_index = current_index + 1
                state.update({
                    "current_sub_query_index": next_index,
                    "current_sub_query": sub_queries[next_index],
                    "workflow_status": "processing_next_sub_query",
                    # Reset per-sub-query state
                    "retrieved_documents": [],
                    "syntheses": [],
                    "critiques": [],
                    "consensus_reached": False,
                    "consensus_output": ""
                })
            
            return state
            
        except Exception as e:
            logger.error(f"Error in check_completion_node: {e}")
            state.update({
                "workflow_status": "error",
                "error_message": f"Completion check error: {str(e)}"
            })
            return state
    
    def _final_synthesis_node(self, state: CoRSWorkflowState) -> CoRSWorkflowState:
        """Node to create the final synthesis from all consensus outputs."""
        try:
            session_id = state["session_id"]
            original_query = state["original_query"]
            
            logger.info("Creating final synthesis")
            
            # Get session state from SSS
            session_state = self.sss.get_session_state(session_id)
            if not session_state:
                raise ValueError("Session state not found")
            
            # Collect all consensus outputs
            consensus_outputs = []
            for sub_query_state in session_state.sub_queries:
                if sub_query_state.consensus_output:
                    consensus_outputs.append({
                        "sub_query": sub_query_state.sub_query,
                        "consensus_output": sub_query_state.consensus_output
                    })
            
            if not consensus_outputs:
                raise ValueError("No consensus outputs available for final synthesis")
            
            # Get a synthesizer agent for final synthesis
            final_synthesizer = self.agent_pool.get_best_agent("synthesizer")
            if not final_synthesizer:
                raise ValueError("No synthesizer agent available for final synthesis")
            
            # Create final synthesis prompt
            consensus_text = "\n\n".join([
                f"Sub-query: {item['sub_query']}\nAnswer: {item['consensus_output']}"
                for item in consensus_outputs
            ])
            
            # Generate final synthesis
            final_result = final_synthesizer.process({
                "sub_query": f"Please synthesize the following verified facts into a comprehensive answer to: {original_query}",
                "documents": [{"content": consensus_text, "doc_id": "consensus_outputs"}]
            })
            
            if final_result["success"]:
                final_answer = final_result["answer"]
                
                # Update session state in SSS
                session_state.final_answer = final_answer
                session_state.updated_at = time.time()
                
                state.update({
                    "final_answer": final_answer,
                    "workflow_status": "completed",
                    "metadata": {
                        **state.get("metadata", {}),
                        "final_synthesis_metadata": final_result.get("synthesis_metadata", {})
                    }
                })
            else:
                # Fallback: concatenate consensus outputs
                final_answer = f"Based on the analysis of your question '{original_query}':\n\n"
                for i, item in enumerate(consensus_outputs, 1):
                    final_answer += f"{i}. {item['consensus_output']}\n\n"
                
                state.update({
                    "final_answer": final_answer,
                    "workflow_status": "completed_with_fallback",
                    "error_message": "Final synthesis failed, used fallback approach"
                })
            
            return state
            
        except Exception as e:
            logger.error(f"Error in final_synthesis_node: {e}")
            state.update({
                "workflow_status": "error",
                "error_message": f"Final synthesis error: {str(e)}"
            })
            return state
    
    def _handle_error_node(self, state: CoRSWorkflowState) -> CoRSWorkflowState:
        """Node to handle errors and provide fallback response."""
        try:
            error_message = state.get("error_message", "Unknown error occurred")
            logger.error(f"Handling workflow error: {error_message}")
            
            # Provide fallback response
            fallback_answer = (
                f"I apologize, but I encountered an error while processing your query: "
                f"'{state['original_query']}'. Error: {error_message}. "
                f"Please try rephrasing your question or contact support if the issue persists."
            )
            
            state.update({
                "final_answer": fallback_answer,
                "workflow_status": "error_handled"
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error in error handling node: {e}")
            state.update({
                "final_answer": "A critical error occurred during processing. Please contact support.",
                "workflow_status": "critical_error"
            })
            return state
    
    def _route_after_completion_check(self, state: CoRSWorkflowState) -> str:
        """Router function to determine next step after completion check."""
        status = state["workflow_status"]
        
        if status == "error":
            return "error"
        elif status == "all_sub_queries_complete":
            return "finalize"
        else:
            return "continue"
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the complete CoRS workflow.
        
        Args:
            query: User query to process
            
        Returns:
            Dictionary with final answer and metadata
        """
        try:
            # Create session in SSS
            session_id = self.sss.create_session(query)
            
            # Initialize workflow state
            initial_state = CoRSWorkflowState(
                session_id=session_id,
                original_query=query,
                current_sub_query_index=0,
                sub_queries=[],
                current_sub_query="",
                retrieved_documents=[],
                syntheses=[],
                critiques=[],
                consensus_reached=False,
                consensus_output="",
                final_answer="",
                workflow_status="initialized",
                error_message="",
                metadata={"start_time": time.time()}
            )
            
            # Execute workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Calculate metrics
            end_time = time.time()
            total_time = end_time - initial_state["metadata"]["start_time"]
            
            # Get session metrics from SSS
            session_metrics = self.sss.get_session_metrics(session_id)
            
            # Prepare result
            result = {
                "session_id": session_id,
                "query": query,
                "answer": final_state["final_answer"],
                "status": final_state["workflow_status"],
                "success": final_state["workflow_status"] in ["completed", "completed_with_fallback"],
                "processing_time": total_time,
                "sub_queries": final_state["sub_queries"],
                "metadata": {
                    **final_state["metadata"],
                    "session_metrics": session_metrics,
                    "agent_pool_stats": self.agent_pool.get_pool_stats()
                }
            }
            
            if final_state["error_message"]:
                result["error"] = final_state["error_message"]
            
            logger.info(f"Query processing completed for session {session_id} in {total_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in process_query: {e}")
            return {
                "query": query,
                "answer": f"An error occurred while processing your query: {str(e)}",
                "status": "error",
                "success": False,
                "error": str(e)
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        return {
            "config": {
                "num_synthesizer_agents": self.config.num_synthesizer_agents,
                "num_retrieval_agents": self.config.num_retrieval_agents,
                "consensus_strategy": self.config.consensus_strategy.value,
                "consensus_threshold": self.config.consensus_threshold
            },
            "agent_pool": self.agent_pool.get_pool_stats(),
            "top_agents": self.sss.get_top_agents(10),
            "consensus_mechanism": {
                "strategy": self.consensus_mechanism.strategy.value,
                "learning_rate": self.consensus_mechanism.learning_rate,
                "threshold": self.consensus_mechanism.consensus_threshold
            }
        }


def create_cors_system(config: CoRSConfig = None) -> CoRSSystem:
    """
    Factory function to create a CoRS system.
    
    Args:
        config: Optional system configuration
        
    Returns:
        Configured CoRSSystem instance
    """
    return CoRSSystem(config)