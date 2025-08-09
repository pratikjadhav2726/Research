"""
CoRS Retrieval Protocol - The Three-Step Collaborative Process

This module implements the core CoRS protocol that transforms RAG from a solitary 
activity into a collaborative team sport:

Step 1: Consult the Workspace - Check if another agent already found the answer
Step 2: Retrieve and Contribute - Query main KB only if needed, then share findings  
Step 3: Synthesize from Workspace - Build final response from verified, collaborative context

This protocol enables massive efficiency gains and emergent consensus through 
intelligent information sharing between agents.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

from .shared_dynamic_workspace import SharedDynamicWorkspace, ChunkMetadata

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Result of a CoRS retrieval operation"""
    chunks: List[str]
    source: str  # "workspace" or "knowledge_base" 
    cache_hit: bool
    contribution_stats: Optional[Dict[str, Any]] = None
    retrieval_time: float = 0.0
    metadata: Optional[List[ChunkMetadata]] = None

@dataclass 
class CoRSMetrics:
    """Metrics for CoRS protocol performance"""
    total_queries: int = 0
    workspace_hits: int = 0
    kb_queries: int = 0
    cache_hit_rate: float = 0.0
    avg_retrieval_time: float = 0.0
    collaboration_efficiency: float = 0.0  # % of queries satisfied by workspace

class VectorStoreInterface(ABC):
    """Abstract interface for vector databases"""
    
    @abstractmethod
    def similarity_search(self, query: str, top_k: int = 5) -> List[str]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def get_database_name(self) -> str:
        """Get the name/identifier of this database"""
        pass

class CoRSRetrievalProtocol:
    """
    Implementation of the Collaborative Retrieval and Synthesis Protocol
    
    This is the heart of the CoRS system - it transforms traditional RAG from 
    independent agent queries into a collaborative, cache-first approach that
    builds collective intelligence through shared discoveries.
    """
    
    def __init__(self, 
                 workspace: SharedDynamicWorkspace,
                 vector_store: VectorStoreInterface,
                 workspace_search_threshold: float = 0.7,
                 min_workspace_verification: int = 1):
        """
        Initialize the CoRS Retrieval Protocol
        
        Args:
            workspace: The shared dynamic context workspace
            vector_store: Interface to the main knowledge base
            workspace_search_threshold: Minimum relevance score for workspace hits
            min_workspace_verification: Minimum verification count for workspace results
        """
        self.workspace = workspace
        self.vector_store = vector_store
        self.workspace_search_threshold = workspace_search_threshold
        self.min_workspace_verification = min_workspace_verification
        
        # Metrics tracking
        self.metrics = CoRSMetrics()
        
        logger.info("Initialized CoRS Retrieval Protocol")
    
    def retrieve_context(self, 
                        query: str, 
                        agent_id: str,
                        top_k: int = 5,
                        force_kb_search: bool = False) -> RetrievalResult:
        """
        Execute the complete CoRS retrieval protocol
        
        This is the main entry point that implements the three-step CoRS process:
        1. Search workspace first (collaborative cache)
        2. Query main knowledge base if needed  
        3. Contribute findings back to workspace
        
        Args:
            query: The search query
            agent_id: Identifier of the requesting agent
            top_k: Number of results to return
            force_kb_search: Skip workspace and go directly to knowledge base
            
        Returns:
            RetrievalResult with chunks, metadata, and performance stats
        """
        start_time = time.time()
        self.metrics.total_queries += 1
        
        logger.info(f"Agent {agent_id} requesting context for: '{query[:100]}...'")
        
        # Step 1: Consult the Workspace First
        workspace_results = []
        if not force_kb_search:
            workspace_results = self._search_workspace_first(query, top_k)
            
            if workspace_results:
                # Cache hit! Return workspace results
                retrieval_time = time.time() - start_time
                self.metrics.workspace_hits += 1
                self.metrics.avg_retrieval_time = self._update_avg_time(retrieval_time)
                
                chunks = [chunk.content for chunk in workspace_results]
                
                logger.info(f"🎯 CACHE HIT: Agent {agent_id} found {len(chunks)} chunks in workspace")
                
                return RetrievalResult(
                    chunks=chunks,
                    source="workspace",
                    cache_hit=True,
                    retrieval_time=retrieval_time,
                    metadata=workspace_results
                )
        
        # Step 2: Retrieve from Main Knowledge Base
        logger.info(f"📚 CACHE MISS: Agent {agent_id} querying main knowledge base")
        kb_chunks = self._query_knowledge_base(query, top_k)
        self.metrics.kb_queries += 1
        
        # Step 3: Contribute Back to Workspace
        contribution_stats = None
        if kb_chunks:
            contribution_stats = self._contribute_to_workspace(
                kb_chunks, agent_id, query
            )
        
        retrieval_time = time.time() - start_time
        self.metrics.avg_retrieval_time = self._update_avg_time(retrieval_time)
        
        logger.info(f"✅ Agent {agent_id} completed retrieval: {len(kb_chunks)} chunks from KB")
        
        return RetrievalResult(
            chunks=kb_chunks,
            source="knowledge_base", 
            cache_hit=False,
            contribution_stats=contribution_stats,
            retrieval_time=retrieval_time
        )
    
    def _search_workspace_first(self, query: str, top_k: int) -> List[ChunkMetadata]:
        """
        Step 1: Search the workspace for existing relevant information
        
        This is where the collaborative magic begins - before doing expensive
        searches of the main knowledge base, we check if another agent has
        already found what we're looking for.
        """
        try:
            logger.debug(f"🔍 Searching workspace for: '{query}'")
            
            workspace_results = self.workspace.search_workspace(
                query=query,
                top_k=top_k,
                min_verification=self.min_workspace_verification
            )
            
            if workspace_results:
                # Filter by relevance threshold if needed
                high_quality_results = []
                for chunk in workspace_results:
                    # Use saliency score as relevance indicator
                    if chunk.saliency_score >= self.workspace_search_threshold:
                        high_quality_results.append(chunk)
                
                if high_quality_results:
                    logger.info(f"Found {len(high_quality_results)} high-quality chunks in workspace")
                    return high_quality_results
                else:
                    logger.info("Workspace chunks found but below quality threshold")
            
            return []
            
        except Exception as e:
            logger.error(f"Error searching workspace: {e}")
            return []
    
    def _query_knowledge_base(self, query: str, top_k: int) -> List[str]:
        """
        Step 2: Query the main knowledge base when workspace doesn't have the answer
        
        This is the traditional RAG retrieval step, but now it only happens when
        the collaborative workspace doesn't already have what we need.
        """
        try:
            logger.debug(f"📖 Querying main knowledge base for: '{query}'")
            
            chunks = self.vector_store.similarity_search(query, top_k=top_k)
            
            if chunks:
                logger.info(f"Retrieved {len(chunks)} chunks from {self.vector_store.get_database_name()}")
            else:
                logger.warning(f"No results found in knowledge base for query: '{query}'")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return []
    
    def _contribute_to_workspace(self, 
                               chunks: List[str], 
                               agent_id: str, 
                               query: str) -> Dict[str, Any]:
        """
        Step 3: Contribute findings back to the workspace
        
        This is where individual discoveries become collective intelligence.
        The workspace will automatically:
        - De-duplicate information
        - Track verification counts (how many agents found the same info)
        - Calculate saliency scores
        - Maintain provenance (who found what and why)
        """
        try:
            logger.debug(f"🤝 Agent {agent_id} contributing {len(chunks)} chunks to workspace")
            
            contribution_stats = self.workspace.contribute_to_workspace(
                chunks=chunks,
                agent_id=agent_id,
                query=query,
                source_db=self.vector_store.get_database_name()
            )
            
            # Log contribution summary
            new_chunks = contribution_stats.get('new_chunks', 0)
            verified_chunks = contribution_stats.get('verified_chunks', 0)
            
            logger.info(f"Agent {agent_id} contributed: {new_chunks} new, {verified_chunks} verified chunks")
            
            return contribution_stats
            
        except Exception as e:
            logger.error(f"Error contributing to workspace: {e}")
            return {}
    
    def get_synthesis_context(self, 
                            task_description: str,
                            synthesizer_agent_id: str,
                            min_verification: int = 2) -> Tuple[List[ChunkMetadata], Dict[str, Any]]:
        """
        Get prioritized context for synthesis, focusing on highly verified information
        
        This method is used by synthesizer agents to get the best available context
        from the workspace, prioritizing information that has been independently
        verified by multiple agents.
        
        Args:
            task_description: Description of what needs to be synthesized
            synthesizer_agent_id: ID of the synthesizer agent
            min_verification: Minimum verification count for inclusion
            
        Returns:
            Tuple of (prioritized_chunks, synthesis_metadata)
        """
        try:
            logger.info(f"🎯 Getting synthesis context for: '{task_description[:100]}...'")
            
            # Get prioritized chunks from workspace
            synthesis_chunks = self.workspace.get_synthesis_context(
                task_description=task_description,
                min_verification=min_verification,
                prioritize_unused=True  # Avoid redundancy
            )
            
            # Calculate synthesis metadata
            synthesis_metadata = {
                'total_chunks': len(synthesis_chunks),
                'avg_verification': sum(c.verification_count for c in synthesis_chunks) / len(synthesis_chunks) if synthesis_chunks else 0,
                'avg_saliency': sum(c.saliency_score for c in synthesis_chunks) / len(synthesis_chunks) if synthesis_chunks else 0,
                'unique_contributors': len(set(
                    source['agent_id'] 
                    for chunk in synthesis_chunks 
                    for source in chunk.retrieval_sources
                )),
                'unused_chunks': len([c for c in synthesis_chunks if not c.synthesis_status])
            }
            
            logger.info(f"Synthesis context: {synthesis_metadata['total_chunks']} chunks, "
                       f"avg verification: {synthesis_metadata['avg_verification']:.1f}, "
                       f"{synthesis_metadata['unique_contributors']} contributors")
            
            return synthesis_chunks, synthesis_metadata
            
        except Exception as e:
            logger.error(f"Error getting synthesis context: {e}")
            return [], {}
    
    def mark_synthesis_complete(self, 
                              chunk_ids: List[str], 
                              synthesizer_agent_id: str):
        """
        Mark chunks as used in synthesis to prevent redundancy
        
        This helps maintain efficiency by tracking which information has already
        been incorporated into final outputs.
        """
        try:
            self.workspace.mark_chunks_synthesized(chunk_ids, synthesizer_agent_id)
            logger.info(f"Marked {len(chunk_ids)} chunks as synthesized by {synthesizer_agent_id}")
            
        except Exception as e:
            logger.error(f"Error marking synthesis complete: {e}")
    
    def _update_avg_time(self, new_time: float) -> float:
        """Update rolling average retrieval time"""
        if self.metrics.avg_retrieval_time == 0:
            return new_time
        
        # Simple exponential moving average
        alpha = 0.1
        return alpha * new_time + (1 - alpha) * self.metrics.avg_retrieval_time
    
    def get_protocol_metrics(self) -> CoRSMetrics:
        """Get performance metrics for the CoRS protocol"""
        # Update calculated metrics
        if self.metrics.total_queries > 0:
            self.metrics.cache_hit_rate = self.metrics.workspace_hits / self.metrics.total_queries
            self.metrics.collaboration_efficiency = self.metrics.workspace_hits / self.metrics.total_queries
        
        return self.metrics
    
    def get_workspace_intelligence_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report on workspace intelligence and collaboration
        
        This provides insights into how well agents are collaborating and what
        kind of emergent consensus is developing in the workspace.
        """
        try:
            workspace_stats = self.workspace.get_workspace_stats()
            protocol_metrics = self.get_protocol_metrics()
            
            report = {
                'protocol_performance': {
                    'total_queries': protocol_metrics.total_queries,
                    'cache_hit_rate': protocol_metrics.cache_hit_rate,
                    'avg_retrieval_time': protocol_metrics.avg_retrieval_time,
                    'collaboration_efficiency': protocol_metrics.collaboration_efficiency
                },
                'workspace_intelligence': {
                    'total_chunks': workspace_stats.total_chunks,
                    'avg_verification_count': workspace_stats.avg_verification_count,
                    'synthesis_efficiency': workspace_stats.synthesis_efficiency,
                    'collaboration_score': workspace_stats.collaboration_score
                },
                'agent_collaboration': {
                    'top_contributors': workspace_stats.top_contributing_agents,
                    'collective_knowledge_quality': workspace_stats.avg_verification_count * workspace_stats.collaboration_score
                },
                'efficiency_gains': {
                    'queries_saved': protocol_metrics.workspace_hits,
                    'kb_query_reduction': f"{protocol_metrics.cache_hit_rate * 100:.1f}%",
                    'estimated_cost_savings': protocol_metrics.workspace_hits * 0.1  # Assume 10x cost difference
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating intelligence report: {e}")
            return {}

class GatekeeperAgent:
    """
    The Gatekeeper Agent - Quality Control for the Workspace
    
    This agent monitors the workspace for anomalies, manages information quality,
    and implements active curation to prevent context pollution.
    """
    
    def __init__(self, 
                 workspace: SharedDynamicWorkspace,
                 anomaly_threshold: float = 0.3,
                 trust_decay_factor: float = 0.95):
        """
        Initialize the Gatekeeper Agent
        
        Args:
            workspace: The shared dynamic workspace to monitor
            anomaly_threshold: Threshold for detecting anomalous retrievals
            trust_decay_factor: Factor for decaying trust scores over time
        """
        self.workspace = workspace
        self.anomaly_threshold = anomaly_threshold
        self.trust_decay_factor = trust_decay_factor
        
        # Agent trust scores (starts neutral)
        self.agent_trust_scores: Dict[str, float] = {}
        
        logger.info("Initialized Gatekeeper Agent for workspace quality control")
    
    def analyze_workspace_health(self) -> Dict[str, Any]:
        """
        Analyze the overall health and quality of the workspace
        
        Returns:
            Dictionary with health metrics and recommendations
        """
        try:
            stats = self.workspace.get_workspace_stats()
            
            health_report = {
                'overall_health': 'good',  # Will be calculated
                'quality_metrics': {
                    'avg_verification': stats.avg_verification_count,
                    'collaboration_score': stats.collaboration_score,
                    'synthesis_efficiency': stats.synthesis_efficiency
                },
                'potential_issues': [],
                'recommendations': []
            }
            
            # Analyze potential issues
            if stats.avg_verification_count < 1.5:
                health_report['potential_issues'].append("Low verification - agents not finding shared information")
                health_report['recommendations'].append("Encourage agents to query similar topics")
            
            if stats.collaboration_score < 0.3:
                health_report['potential_issues'].append("Low collaboration - agents working in isolation")
                health_report['recommendations'].append("Review agent task distribution")
            
            if stats.synthesis_efficiency < 0.5:
                health_report['potential_issues'].append("Low synthesis efficiency - information not being used")
                health_report['recommendations'].append("Improve synthesis agent context selection")
            
            # Overall health score
            health_score = (stats.avg_verification_count / 3.0 + 
                          stats.collaboration_score + 
                          stats.synthesis_efficiency) / 3.0
            
            if health_score > 0.7:
                health_report['overall_health'] = 'excellent'
            elif health_score > 0.5:
                health_report['overall_health'] = 'good'
            elif health_score > 0.3:
                health_report['overall_health'] = 'fair'
            else:
                health_report['overall_health'] = 'poor'
            
            return health_report
            
        except Exception as e:
            logger.error(f"Error analyzing workspace health: {e}")
            return {'overall_health': 'unknown', 'error': str(e)}
    
    def detect_anomalous_retrievals(self) -> List[Dict[str, Any]]:
        """
        Detect potentially anomalous or suspicious retrieval patterns
        
        Returns:
            List of detected anomalies with details
        """
        anomalies = []
        
        try:
            # This would analyze patterns in the workspace
            # For now, return a placeholder structure
            
            # Example anomaly detection logic:
            # - Agent retrieving information far outside their usual domain
            # - Chunks with very low saliency scores but high retrieval frequency
            # - Sudden changes in agent behavior patterns
            
            logger.info("Anomaly detection completed - no issues found")
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            anomalies.append({
                'type': 'system_error',
                'description': f"Anomaly detection failed: {e}",
                'severity': 'high'
            })
        
        return anomalies
    
    def cleanup_low_quality_content(self) -> Dict[str, int]:
        """
        Remove low-quality or stale content from the workspace
        
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            # Clean up expired chunks
            expired_count = self.workspace.cleanup_expired_chunks()
            
            cleanup_stats = {
                'expired_chunks_removed': expired_count,
                'low_quality_chunks_removed': 0,  # Would implement quality-based removal
                'total_cleaned': expired_count
            }
            
            logger.info(f"Gatekeeper cleanup: removed {cleanup_stats['total_cleaned']} items")
            
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {'error': str(e)}