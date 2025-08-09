"""
CoRS Collaborative Agents - True Multi-Agent Collaboration

This module implements agents that follow the CoRS protocol, transforming RAG
from independent agent queries into a collaborative team sport where agents
build upon each other's discoveries through the shared dynamic workspace.

Key agents:
- ResearcherAgent: Specialized information gathering with workspace-first approach
- PlannerAgent: Task decomposition and coordination using collaborative intelligence  
- SynthesizerAgent: Final synthesis from verified, collaborative context
"""

import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from ..core.cors_protocol import CoRSRetrievalProtocol, RetrievalResult
from ..core.shared_dynamic_workspace import ChunkMetadata

logger = logging.getLogger(__name__)

@dataclass
class AgentTask:
    """Task structure for collaborative agents"""
    task_id: str
    description: str
    assigned_agent: str
    priority: int = 1
    dependencies: List[str] = None
    status: str = "pending"  # pending, in_progress, completed
    result: Optional[Dict[str, Any]] = None

@dataclass
class CollaborationMetrics:
    """Metrics for measuring agent collaboration effectiveness"""
    cache_hit_rate: float = 0.0
    knowledge_reuse_rate: float = 0.0
    collaboration_efficiency: float = 0.0
    avg_task_completion_time: float = 0.0
    unique_information_contributed: int = 0

class CoRSBaseAgent(ABC):
    """
    Base class for all CoRS collaborative agents
    
    All CoRS agents follow the collaborative protocol:
    1. Check workspace first for existing relevant information
    2. Query knowledge base only if needed
    3. Contribute findings back to shared workspace
    4. Build upon other agents' verified discoveries
    """
    
    def __init__(self, 
                 agent_id: str,
                 cors_protocol: CoRSRetrievalProtocol,
                 llm: ChatOpenAI,
                 specialization: str = "general"):
        """
        Initialize a CoRS collaborative agent
        
        Args:
            agent_id: Unique identifier for this agent
            cors_protocol: The CoRS retrieval protocol instance
            llm: Language model for this agent
            specialization: Agent's area of specialization
        """
        self.agent_id = agent_id
        self.cors_protocol = cors_protocol
        self.llm = llm
        self.specialization = specialization
        
        # Performance tracking
        self.metrics = CollaborationMetrics()
        self.task_history: List[AgentTask] = []
        
        logger.info(f"Initialized {self.__class__.__name__} '{agent_id}' specialized in '{specialization}'")
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        pass
    
    def retrieve_context(self, 
                        query: str, 
                        top_k: int = 5,
                        force_kb_search: bool = False) -> RetrievalResult:
        """
        Retrieve context using the CoRS protocol
        
        This is the collaborative magic - agents first check what their teammates
        have already discovered before doing expensive knowledge base searches.
        """
        return self.cors_protocol.retrieve_context(
            query=query,
            agent_id=self.agent_id,
            top_k=top_k,
            force_kb_search=force_kb_search
        )
    
    def _invoke_llm(self, messages: List, **kwargs) -> str:
        """Invoke the LLM with error handling"""
        try:
            response = self.llm.invoke(messages, **kwargs)
            return response.content
        except Exception as e:
            logger.error(f"Error invoking LLM for agent {self.agent_id}: {e}")
            return f"Error: {str(e)}"
    
    def _update_metrics(self, retrieval_result: RetrievalResult, task_time: float):
        """Update collaboration metrics"""
        if retrieval_result.cache_hit:
            # Update cache hit rate using exponential moving average
            alpha = 0.1
            self.metrics.cache_hit_rate = alpha * 1.0 + (1 - alpha) * self.metrics.cache_hit_rate
        else:
            self.metrics.cache_hit_rate = 0.9 * self.metrics.cache_hit_rate
        
        # Update task completion time
        if self.metrics.avg_task_completion_time == 0:
            self.metrics.avg_task_completion_time = task_time
        else:
            self.metrics.avg_task_completion_time = 0.1 * task_time + 0.9 * self.metrics.avg_task_completion_time

class ResearcherAgent(CoRSBaseAgent):
    """
    Researcher Agent - Specialized Information Gathering
    
    This agent excels at finding and contributing high-quality information to the
    collaborative workspace. It follows the CoRS protocol to avoid redundant work
    and build upon other agents' discoveries.
    """
    
    def __init__(self, 
                 agent_id: str,
                 cors_protocol: CoRSRetrievalProtocol,
                 llm: ChatOpenAI,
                 research_domain: str = "general"):
        """
        Initialize a Researcher Agent
        
        Args:
            agent_id: Unique identifier for this agent
            cors_protocol: The CoRS retrieval protocol instance  
            llm: Language model for this agent
            research_domain: Domain of expertise (e.g., "technology", "finance", "science")
        """
        super().__init__(agent_id, cors_protocol, llm, research_domain)
        self.research_domain = research_domain
    
    def get_system_prompt(self) -> str:
        return f"""You are a specialized Researcher Agent working in a collaborative multi-agent system.

Your expertise: {self.research_domain}
Your role: Find and analyze relevant information to answer research questions.

COLLABORATIVE PROTOCOL:
- You work with other agents who may have already found relevant information
- Always build upon existing verified findings rather than duplicating work
- Focus on finding NEW information that complements what others have discovered
- Provide clear, factual analysis with proper attribution to sources

Your responses should be:
1. Factual and well-sourced
2. Complementary to existing team knowledge  
3. Clearly structured with key findings highlighted
4. Honest about limitations or uncertainties

Remember: You're part of a research TEAM. Your job is to contribute unique value, not work in isolation."""
    
    def research_topic(self, 
                      research_question: str,
                      focus_areas: List[str] = None,
                      depth: str = "comprehensive") -> Dict[str, Any]:
        """
        Research a topic using the collaborative CoRS approach
        
        Args:
            research_question: The question to research
            focus_areas: Specific areas to focus on (optional)
            depth: Research depth ("quick", "standard", "comprehensive")
            
        Returns:
            Dictionary with research findings and collaboration metadata
        """
        start_time = time.time()
        
        logger.info(f"Researcher {self.agent_id} starting research: '{research_question[:100]}...'")
        
        # Step 1: Check what teammates have already discovered
        retrieval_result = self.retrieve_context(research_question, top_k=8)
        
        # Prepare context for LLM
        if retrieval_result.cache_hit:
            context_source = "collaborative workspace (verified by teammates)"
            logger.info(f"🎯 Building on teammates' discoveries: {len(retrieval_result.chunks)} verified chunks")
        else:
            context_source = "knowledge base"
            logger.info(f"📚 Pioneering new research: {len(retrieval_result.chunks)} new chunks from KB")
        
        # Build research prompt
        context_text = "\n\n".join([
            f"Source {i+1}: {chunk}" 
            for i, chunk in enumerate(retrieval_result.chunks)
        ])
        
        focus_instruction = ""
        if focus_areas:
            focus_instruction = f"\nFocus particularly on these areas: {', '.join(focus_areas)}"
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=f"""Research Question: {research_question}

Available Context (from {context_source}):
{context_text}

Research Depth: {depth}{focus_instruction}

Provide a comprehensive research analysis that:
1. Synthesizes the available information
2. Identifies key findings and insights
3. Notes any gaps or areas needing further investigation
4. Highlights the most important discoveries

Format your response as a structured analysis with clear sections.""")
        ]
        
        # Generate research analysis
        analysis = self._invoke_llm(messages)
        
        # Prepare research result
        research_result = {
            'question': research_question,
            'analysis': analysis,
            'source_type': retrieval_result.source,
            'cache_hit': retrieval_result.cache_hit,
            'chunks_analyzed': len(retrieval_result.chunks),
            'research_time': time.time() - start_time,
            'researcher_id': self.agent_id,
            'domain': self.research_domain
        }
        
        # Update metrics
        self._update_metrics(retrieval_result, research_result['research_time'])
        
        # Add to task history
        task = AgentTask(
            task_id=f"research_{int(time.time())}",
            description=research_question,
            assigned_agent=self.agent_id,
            status="completed",
            result=research_result
        )
        self.task_history.append(task)
        
        logger.info(f"✅ Research completed by {self.agent_id}: "
                   f"{research_result['research_time']:.2f}s, "
                   f"cache_hit={retrieval_result.cache_hit}")
        
        return research_result

class PlannerAgent(CoRSBaseAgent):
    """
    Planner Agent - Task Decomposition and Coordination
    
    This agent breaks down complex tasks into manageable subtasks and coordinates
    the work of other agents. It uses the collaborative workspace to understand
    what information is already available and plan accordingly.
    """
    
    def __init__(self, 
                 agent_id: str,
                 cors_protocol: CoRSRetrievalProtocol,
                 llm: ChatOpenAI):
        super().__init__(agent_id, cors_protocol, llm, "planning_coordination")
    
    def get_system_prompt(self) -> str:
        return """You are a Planner Agent in a collaborative multi-agent research system.

Your role: Break down complex tasks and coordinate team efforts efficiently.

COLLABORATIVE INTELLIGENCE:
- You can see what information teammates have already gathered
- Plan tasks to minimize redundant work and maximize team efficiency  
- Identify gaps where new research is needed vs. areas already well-covered
- Coordinate parallel work streams for maximum efficiency

Your planning should:
1. Analyze existing team knowledge before assigning new work
2. Create clear, actionable subtasks with priorities
3. Identify dependencies between tasks
4. Optimize for team efficiency and knowledge reuse
5. Suggest which agents are best suited for each subtask

Remember: You're optimizing TEAM performance, not just task completion."""
    
    def create_research_plan(self, 
                           main_objective: str,
                           available_agents: List[str] = None,
                           time_constraint: str = "standard") -> Dict[str, Any]:
        """
        Create a comprehensive research plan using collaborative intelligence
        
        Args:
            main_objective: The main research objective
            available_agents: List of available agent IDs
            time_constraint: Time constraint ("urgent", "standard", "extended")
            
        Returns:
            Detailed research plan with subtasks and agent assignments
        """
        start_time = time.time()
        
        logger.info(f"Planner {self.agent_id} creating plan for: '{main_objective[:100]}...'")
        
        # Step 1: Check what the team already knows about this objective
        retrieval_result = self.retrieve_context(main_objective, top_k=10)
        
        # Analyze existing knowledge
        existing_knowledge = ""
        knowledge_gaps = []
        
        if retrieval_result.cache_hit and retrieval_result.metadata:
            # Analyze what areas are well-covered vs. gaps
            topics_covered = set()
            verification_counts = []
            
            for chunk in retrieval_result.metadata:
                verification_counts.append(chunk.verification_count)
                # Extract key topics (simplified - could use NLP for better topic extraction)
                topics_covered.add(chunk.chunk_id[:8])  # Simplified topic identification
            
            avg_verification = sum(verification_counts) / len(verification_counts)
            existing_knowledge = f"Team has {len(retrieval_result.chunks)} pieces of verified information (avg verification: {avg_verification:.1f})"
            
            logger.info(f"🧠 Existing team knowledge: {len(retrieval_result.chunks)} verified chunks")
        else:
            existing_knowledge = "This appears to be a new research area for the team."
            logger.info("🆕 New research area - planning from scratch")
        
        # Create planning prompt
        available_agents_text = f"Available agents: {', '.join(available_agents)}" if available_agents else "Agent assignments to be determined"
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=f"""Main Objective: {main_objective}

Existing Team Knowledge:
{existing_knowledge}

Time Constraint: {time_constraint}
{available_agents_text}

Create a detailed research plan that:
1. Breaks down the objective into 4-8 specific subtasks
2. Prioritizes subtasks (1=highest priority)  
3. Identifies which subtasks can be done in parallel
4. Suggests optimal agent assignments based on specialization
5. Estimates effort level for each subtask
6. Identifies dependencies between tasks

Format as a structured plan with clear sections:
- OBJECTIVE ANALYSIS
- SUBTASK BREAKDOWN  
- EXECUTION STRATEGY
- RESOURCE ALLOCATION
- SUCCESS CRITERIA""")
        ]
        
        # Generate the plan
        plan_text = self._invoke_llm(messages)
        
        # Create structured plan result
        research_plan = {
            'objective': main_objective,
            'plan_details': plan_text,
            'existing_knowledge_leveraged': retrieval_result.cache_hit,
            'knowledge_base_size': len(retrieval_result.chunks) if retrieval_result.chunks else 0,
            'planning_time': time.time() - start_time,
            'planner_id': self.agent_id,
            'time_constraint': time_constraint,
            'available_agents': available_agents or []
        }
        
        # Update metrics
        self._update_metrics(retrieval_result, research_plan['planning_time'])
        
        # Add to task history
        task = AgentTask(
            task_id=f"plan_{int(time.time())}",
            description=f"Plan: {main_objective}",
            assigned_agent=self.agent_id,
            status="completed",
            result=research_plan
        )
        self.task_history.append(task)
        
        logger.info(f"✅ Planning completed by {self.agent_id}: "
                   f"{research_plan['planning_time']:.2f}s, "
                   f"leveraged_existing_knowledge={retrieval_result.cache_hit}")
        
        return research_plan

class SynthesizerAgent(CoRSBaseAgent):
    """
    Synthesizer Agent - Collaborative Knowledge Integration
    
    This agent specializes in creating coherent, comprehensive syntheses from
    the verified information in the collaborative workspace. It prioritizes
    highly-verified information and avoids redundancy.
    """
    
    def __init__(self, 
                 agent_id: str,
                 cors_protocol: CoRSRetrievalProtocol,
                 llm: ChatOpenAI,
                 synthesis_style: str = "comprehensive"):
        """
        Initialize a Synthesizer Agent
        
        Args:
            agent_id: Unique identifier for this agent
            cors_protocol: The CoRS retrieval protocol instance
            llm: Language model for this agent
            synthesis_style: Style of synthesis ("comprehensive", "concise", "analytical")
        """
        super().__init__(agent_id, cors_protocol, llm, "synthesis")
        self.synthesis_style = synthesis_style
    
    def get_system_prompt(self) -> str:
        return f"""You are a Synthesizer Agent in a collaborative multi-agent research system.

Your expertise: Creating coherent, comprehensive syntheses from verified team knowledge
Synthesis style: {self.synthesis_style}

COLLABORATIVE SYNTHESIS:
- You work with information that has been verified by multiple team members
- Prioritize highly-verified information (found by multiple agents independently)
- Create coherent narratives that integrate diverse findings
- Avoid redundancy by building on established team knowledge
- Cite the collaborative verification when relevant

Your syntheses should be:
1. Coherent and well-structured
2. Based on the most verified and reliable team findings
3. Comprehensive yet concise
4. Clear about confidence levels and limitations
5. Properly attributed to the collaborative research process

Remember: You're creating the definitive team output that represents our collective intelligence."""
    
    def synthesize_research(self, 
                          synthesis_topic: str,
                          min_verification: int = 2,
                          include_metadata: bool = True) -> Dict[str, Any]:
        """
        Create a comprehensive synthesis from collaborative workspace knowledge
        
        Args:
            synthesis_topic: Topic to synthesize
            min_verification: Minimum verification count for included information
            include_metadata: Whether to include collaboration metadata
            
        Returns:
            Dictionary with synthesis and collaboration metadata
        """
        start_time = time.time()
        
        logger.info(f"Synthesizer {self.agent_id} creating synthesis: '{synthesis_topic[:100]}...'")
        
        # Get prioritized synthesis context from workspace
        synthesis_chunks, synthesis_metadata = self.cors_protocol.get_synthesis_context(
            task_description=synthesis_topic,
            synthesizer_agent_id=self.agent_id,
            min_verification=min_verification
        )
        
        if not synthesis_chunks:
            logger.warning(f"No verified information available for synthesis: {synthesis_topic}")
            return {
                'topic': synthesis_topic,
                'synthesis': "Insufficient verified information available for synthesis.",
                'collaboration_metadata': {'error': 'No verified chunks found'},
                'synthesis_time': time.time() - start_time
            }
        
        # Prepare synthesis context
        context_sections = []
        chunk_ids_used = []
        
        for i, chunk in enumerate(synthesis_chunks):
            verification_info = f"(Verified by {chunk.verification_count} agents"
            if chunk.retrieval_sources:
                contributors = [source['agent_id'] for source in chunk.retrieval_sources]
                verification_info += f": {', '.join(set(contributors))}"
            verification_info += ")"
            
            context_sections.append(f"""
Section {i+1} {verification_info}:
{chunk.content}
""")
            chunk_ids_used.append(chunk.chunk_id)
        
        context_text = "\n".join(context_sections)
        
        # Create synthesis prompt
        metadata_instruction = ""
        if include_metadata:
            metadata_instruction = f"""
            
COLLABORATION CONTEXT:
- Total verified sources: {synthesis_metadata['total_chunks']}
- Average verification level: {synthesis_metadata['avg_verification']:.1f} agents
- Contributing team members: {synthesis_metadata['unique_contributors']}
- Unused information available: {synthesis_metadata['unused_chunks']} chunks
"""
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=f"""Synthesis Topic: {synthesis_topic}

Verified Information from Team Research:
{context_text}{metadata_instruction}

Create a {self.synthesis_style} synthesis that:
1. Integrates all verified information into a coherent narrative
2. Highlights key insights and findings
3. Notes the collaborative verification levels where relevant
4. Identifies any remaining gaps or uncertainties
5. Provides clear conclusions based on the team's research

Structure your synthesis with clear sections and smooth transitions between ideas.""")
        ]
        
        # Generate synthesis
        synthesis_text = self._invoke_llm(messages)
        
        # Mark chunks as synthesized to prevent redundancy
        self.cors_protocol.mark_synthesis_complete(chunk_ids_used, self.agent_id)
        
        # Prepare synthesis result
        synthesis_result = {
            'topic': synthesis_topic,
            'synthesis': synthesis_text,
            'style': self.synthesis_style,
            'chunks_synthesized': len(chunk_ids_used),
            'collaboration_metadata': synthesis_metadata,
            'verification_quality': synthesis_metadata['avg_verification'],
            'team_contributors': synthesis_metadata['unique_contributors'],
            'synthesis_time': time.time() - start_time,
            'synthesizer_id': self.agent_id
        }
        
        # Update metrics (synthesis always uses workspace, so it's always a cache hit)
        fake_retrieval = RetrievalResult(chunks=[], source="workspace", cache_hit=True)
        self._update_metrics(fake_retrieval, synthesis_result['synthesis_time'])
        
        # Add to task history
        task = AgentTask(
            task_id=f"synthesis_{int(time.time())}",
            description=synthesis_topic,
            assigned_agent=self.agent_id,
            status="completed",
            result=synthesis_result
        )
        self.task_history.append(task)
        
        logger.info(f"✅ Synthesis completed by {self.agent_id}: "
                   f"{synthesis_result['synthesis_time']:.2f}s, "
                   f"{len(chunk_ids_used)} chunks, "
                   f"avg_verification={synthesis_metadata['avg_verification']:.1f}")
        
        return synthesis_result

class CollaborativeTeam:
    """
    Collaborative Team - Orchestrates Multi-Agent CoRS Workflows
    
    This class manages a team of CoRS agents working together on complex tasks,
    demonstrating the efficiency gains and emergent intelligence from the
    collaborative cache approach.
    """
    
    def __init__(self, 
                 cors_protocol: CoRSRetrievalProtocol,
                 llm: ChatOpenAI):
        """
        Initialize a collaborative team
        
        Args:
            cors_protocol: The shared CoRS retrieval protocol
            llm: Language model for agents
        """
        self.cors_protocol = cors_protocol
        self.llm = llm
        self.agents: Dict[str, CoRSBaseAgent] = {}
        self.team_metrics = CollaborationMetrics()
        
        logger.info("Initialized CoRS Collaborative Team")
    
    def add_researcher(self, agent_id: str, research_domain: str = "general") -> ResearcherAgent:
        """Add a researcher agent to the team"""
        researcher = ResearcherAgent(agent_id, self.cors_protocol, self.llm, research_domain)
        self.agents[agent_id] = researcher
        logger.info(f"Added researcher '{agent_id}' specialized in '{research_domain}'")
        return researcher
    
    def add_planner(self, agent_id: str) -> PlannerAgent:
        """Add a planner agent to the team"""
        planner = PlannerAgent(agent_id, self.cors_protocol, self.llm)
        self.agents[agent_id] = planner
        logger.info(f"Added planner '{agent_id}'")
        return planner
    
    def add_synthesizer(self, agent_id: str, synthesis_style: str = "comprehensive") -> SynthesizerAgent:
        """Add a synthesizer agent to the team"""
        synthesizer = SynthesizerAgent(agent_id, self.cors_protocol, self.llm, synthesis_style)
        self.agents[agent_id] = synthesizer
        logger.info(f"Added synthesizer '{agent_id}' with style '{synthesis_style}'")
        return synthesizer
    
    def execute_collaborative_research(self, 
                                     research_objective: str,
                                     research_areas: List[str] = None) -> Dict[str, Any]:
        """
        Execute a collaborative research workflow demonstrating CoRS benefits
        
        Args:
            research_objective: Main research objective
            research_areas: Specific areas to research (optional)
            
        Returns:
            Complete research results with collaboration metrics
        """
        start_time = time.time()
        
        logger.info(f"🚀 Starting collaborative research: '{research_objective}'")
        
        # Phase 1: Planning (if planner available)
        plan_result = None
        planner_agents = [agent for agent in self.agents.values() if isinstance(agent, PlannerAgent)]
        
        if planner_agents:
            planner = planner_agents[0]
            plan_result = planner.create_research_plan(
                research_objective,
                available_agents=list(self.agents.keys())
            )
            logger.info("📋 Planning phase completed")
        
        # Phase 2: Research (parallel execution)
        research_results = []
        researcher_agents = [agent for agent in self.agents.values() if isinstance(agent, ResearcherAgent)]
        
        if research_areas and researcher_agents:
            # Assign different research areas to different researchers
            for i, area in enumerate(research_areas):
                if i < len(researcher_agents):
                    researcher = researcher_agents[i]
                    research_question = f"{research_objective} - Focus on {area}"
                    result = researcher.research_topic(research_question)
                    research_results.append(result)
                    logger.info(f"🔬 Research completed for area: {area}")
        else:
            # Single comprehensive research
            if researcher_agents:
                researcher = researcher_agents[0]
                result = researcher.research_topic(research_objective)
                research_results.append(result)
                logger.info("🔬 Comprehensive research completed")
        
        # Phase 3: Synthesis
        synthesis_result = None
        synthesizer_agents = [agent for agent in self.agents.values() if isinstance(agent, SynthesizerAgent)]
        
        if synthesizer_agents:
            synthesizer = synthesizer_agents[0]
            synthesis_result = synthesizer.synthesize_research(research_objective)
            logger.info("📝 Synthesis phase completed")
        
        # Calculate team collaboration metrics
        total_time = time.time() - start_time
        
        # Get workspace intelligence report
        intelligence_report = self.cors_protocol.get_workspace_intelligence_report()
        
        # Aggregate agent metrics
        agent_metrics = {}
        total_cache_hits = 0
        total_queries = 0
        
        for agent_id, agent in self.agents.items():
            agent_metrics[agent_id] = {
                'cache_hit_rate': agent.metrics.cache_hit_rate,
                'avg_task_time': agent.metrics.avg_task_completion_time,
                'tasks_completed': len(agent.task_history)
            }
            
            # Estimate queries (simplified)
            if hasattr(agent, 'task_history'):
                total_queries += len(agent.task_history)
                total_cache_hits += int(agent.metrics.cache_hit_rate * len(agent.task_history))
        
        collaboration_result = {
            'objective': research_objective,
            'execution_time': total_time,
            'phases': {
                'planning': plan_result,
                'research': research_results,
                'synthesis': synthesis_result
            },
            'collaboration_metrics': {
                'team_cache_hit_rate': total_cache_hits / total_queries if total_queries > 0 else 0,
                'agents_participated': len([a for a in self.agents.values() if a.task_history]),
                'workspace_intelligence': intelligence_report,
                'agent_performance': agent_metrics
            },
            'efficiency_gains': {
                'estimated_queries_saved': intelligence_report.get('efficiency_gains', {}).get('queries_saved', 0),
                'collaboration_quality': intelligence_report.get('workspace_intelligence', {}).get('collaboration_score', 0),
                'knowledge_reuse_rate': intelligence_report.get('protocol_performance', {}).get('cache_hit_rate', 0)
            }
        }
        
        logger.info(f"🎉 Collaborative research completed in {total_time:.2f}s")
        logger.info(f"📊 Team cache hit rate: {collaboration_result['collaboration_metrics']['team_cache_hit_rate']:.2f}")
        
        return collaboration_result
    
    def get_team_status(self) -> Dict[str, Any]:
        """Get current status of the collaborative team"""
        return {
            'agents': {agent_id: {
                'type': agent.__class__.__name__,
                'specialization': agent.specialization,
                'tasks_completed': len(agent.task_history),
                'cache_hit_rate': agent.metrics.cache_hit_rate
            } for agent_id, agent in self.agents.items()},
            'workspace_stats': self.cors_protocol.workspace.get_workspace_stats(),
            'protocol_metrics': self.cors_protocol.get_protocol_metrics()
        }