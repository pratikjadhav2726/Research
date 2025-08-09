"""
Base Agent Implementation

This module provides the base class for all CoRS agents, defining common
interfaces, utilities, and shared functionality.
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    agent_id: str
    agent_type: str
    model_name: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: float = 30.0
    retry_attempts: int = 3
    custom_params: Dict[str, Any] = None


@dataclass
class AgentMetrics:
    """Metrics tracking for an agent."""
    agent_id: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens_used: int = 0
    avg_response_time: float = 0.0
    reputation_score: float = 0.5
    last_updated: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls
    
    def update_metrics(self, success: bool, tokens_used: int, response_time: float):
        """Update agent metrics after a call."""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        
        self.total_tokens_used += tokens_used
        
        # Update average response time using exponential moving average
        alpha = 0.1
        if self.total_calls == 1:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = alpha * response_time + (1 - alpha) * self.avg_response_time
        
        self.last_updated = time.time()


class BaseAgent(ABC):
    """
    Abstract base class for all CoRS agents.
    
    This class provides:
    - Common LLM interface
    - Metrics tracking
    - Error handling and retry logic
    - Logging and debugging utilities
    - Shared utility methods
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the base agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.agent_id = config.agent_id
        self.agent_type = config.agent_type
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout
        )
        
        # Initialize metrics
        self.metrics = AgentMetrics(agent_id=self.agent_id)
        
        # Agent state
        self.is_active = True
        self.created_at = time.time()
        
        logger.info(f"Initialized {self.agent_type} agent: {self.agent_id}")
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Main processing method that each agent must implement.
        
        Args:
            input_data: Input data for processing
            **kwargs: Additional keyword arguments
            
        Returns:
            Processing result as dictionary
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.
        
        Returns:
            System prompt string
        """
        pass
    
    def invoke_llm(self, 
                   messages: List[BaseMessage], 
                   **kwargs) -> Dict[str, Any]:
        """
        Invoke the LLM with retry logic and metrics tracking.
        
        Args:
            messages: List of messages to send to LLM
            **kwargs: Additional LLM parameters
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        tokens_used = 0
        
        for attempt in range(self.config.retry_attempts):
            try:
                # Add system message if not present
                if not messages or not isinstance(messages[0], SystemMessage):
                    system_msg = SystemMessage(content=self.get_system_prompt())
                    messages = [system_msg] + messages
                
                # Invoke LLM
                response = self.llm.invoke(messages, **kwargs)
                
                # Calculate metrics
                response_time = time.time() - start_time
                tokens_used = self._estimate_tokens(messages, response.content)
                
                # Update metrics
                self.metrics.update_metrics(True, tokens_used, response_time)
                
                result = {
                    "content": response.content,
                    "response_time": response_time,
                    "tokens_used": tokens_used,
                    "attempt": attempt + 1,
                    "success": True
                }
                
                logger.debug(f"Agent {self.agent_id} LLM call successful "
                           f"(attempt {attempt + 1}, {response_time:.2f}s, {tokens_used} tokens)")
                
                return result
                
            except Exception as e:
                logger.warning(f"Agent {self.agent_id} LLM call failed "
                             f"(attempt {attempt + 1}/{self.config.retry_attempts}): {e}")
                
                if attempt == self.config.retry_attempts - 1:
                    # Final attempt failed
                    response_time = time.time() - start_time
                    self.metrics.update_metrics(False, 0, response_time)
                    
                    return {
                        "content": None,
                        "error": str(e),
                        "response_time": response_time,
                        "tokens_used": 0,
                        "attempt": attempt + 1,
                        "success": False
                    }
                
                # Wait before retry
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def create_human_message(self, content: str) -> HumanMessage:
        """Create a human message."""
        return HumanMessage(content=content)
    
    def create_system_message(self, content: str) -> SystemMessage:
        """Create a system message."""
        return SystemMessage(content=content)
    
    def format_prompt(self, template: str, **kwargs) -> str:
        """
        Format a prompt template with provided arguments.
        
        Args:
            template: Prompt template with {variable} placeholders
            **kwargs: Variables to substitute
            
        Returns:
            Formatted prompt string
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            raise ValueError(f"Missing required template variable: {e}")
    
    def validate_input(self, input_data: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        Validate that input data contains required keys.
        
        Args:
            input_data: Input data to validate
            required_keys: List of required keys
            
        Returns:
            True if valid, raises ValueError if not
        """
        missing_keys = [key for key in required_keys if key not in input_data]
        if missing_keys:
            raise ValueError(f"Missing required input keys: {missing_keys}")
        return True
    
    def log_processing_start(self, input_data: Dict[str, Any]):
        """Log the start of processing."""
        logger.info(f"Agent {self.agent_id} starting processing: "
                   f"{self._truncate_dict(input_data, 100)}")
    
    def log_processing_end(self, result: Dict[str, Any], success: bool):
        """Log the end of processing."""
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"Agent {self.agent_id} finished processing: {status}")
        if not success and "error" in result:
            logger.error(f"Agent {self.agent_id} error: {result['error']}")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about this agent.
        
        Returns:
            Dictionary with agent information
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "config": {
                "model_name": self.config.model_name,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            },
            "metrics": {
                "total_calls": self.metrics.total_calls,
                "success_rate": self.metrics.success_rate,
                "avg_response_time": self.metrics.avg_response_time,
                "reputation_score": self.metrics.reputation_score,
                "total_tokens_used": self.metrics.total_tokens_used
            },
            "status": {
                "is_active": self.is_active,
                "created_at": self.created_at,
                "uptime": time.time() - self.created_at
            }
        }
    
    def update_reputation(self, new_score: float):
        """
        Update the agent's reputation score.
        
        Args:
            new_score: New reputation score
        """
        old_score = self.metrics.reputation_score
        self.metrics.reputation_score = max(0.0, min(1.0, new_score))
        
        logger.info(f"Agent {self.agent_id} reputation updated: "
                   f"{old_score:.3f} -> {self.metrics.reputation_score:.3f}")
    
    def deactivate(self):
        """Deactivate this agent."""
        self.is_active = False
        logger.info(f"Agent {self.agent_id} deactivated")
    
    def activate(self):
        """Activate this agent."""
        self.is_active = True
        logger.info(f"Agent {self.agent_id} activated")
    
    def _estimate_tokens(self, messages: List[BaseMessage], response: str) -> int:
        """
        Estimate token usage for a conversation.
        
        This is a rough estimation. In production, you'd want to use
        the actual token counts from the API response.
        
        Args:
            messages: Input messages
            response: Response content
            
        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token
        total_chars = sum(len(msg.content) for msg in messages) + len(response)
        return total_chars // 4
    
    def _truncate_dict(self, data: Dict[str, Any], max_length: int) -> str:
        """
        Create a truncated string representation of a dictionary.
        
        Args:
            data: Dictionary to truncate
            max_length: Maximum string length
            
        Returns:
            Truncated string representation
        """
        str_repr = str(data)
        if len(str_repr) <= max_length:
            return str_repr
        return str_repr[:max_length-3] + "..."
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.agent_type}({self.agent_id})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return (f"{self.agent_type}(id={self.agent_id}, "
                f"model={self.config.model_name}, "
                f"reputation={self.metrics.reputation_score:.3f})")


class AgentPool:
    """
    Manages a pool of agents for dynamic allocation and load balancing.
    """
    
    def __init__(self):
        """Initialize the agent pool."""
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_types: Dict[str, List[str]] = {}  # type -> list of agent_ids
        self.created_at = time.time()
    
    def add_agent(self, agent: BaseAgent):
        """
        Add an agent to the pool.
        
        Args:
            agent: Agent to add
        """
        self.agents[agent.agent_id] = agent
        
        if agent.agent_type not in self.agent_types:
            self.agent_types[agent.agent_type] = []
        self.agent_types[agent.agent_type].append(agent.agent_id)
        
        logger.info(f"Added agent {agent.agent_id} to pool")
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent instance or None if not found
        """
        return self.agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """
        Get all agents of a specific type.
        
        Args:
            agent_type: Type of agents to retrieve
            
        Returns:
            List of agents of the specified type
        """
        agent_ids = self.agent_types.get(agent_type, [])
        return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]
    
    def get_best_agent(self, agent_type: str) -> Optional[BaseAgent]:
        """
        Get the best agent of a specific type based on reputation.
        
        Args:
            agent_type: Type of agent needed
            
        Returns:
            Best agent of the specified type or None
        """
        agents = self.get_agents_by_type(agent_type)
        if not agents:
            return None
        
        # Filter active agents
        active_agents = [agent for agent in agents if agent.is_active]
        if not active_agents:
            return None
        
        # Return agent with highest reputation
        return max(active_agents, key=lambda a: a.metrics.reputation_score)
    
    def remove_agent(self, agent_id: str):
        """
        Remove an agent from the pool.
        
        Args:
            agent_id: Agent identifier
        """
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            del self.agents[agent_id]
            
            # Remove from type index
            if agent.agent_type in self.agent_types:
                self.agent_types[agent.agent_type] = [
                    aid for aid in self.agent_types[agent.agent_type] if aid != agent_id
                ]
            
            logger.info(f"Removed agent {agent_id} from pool")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the agent pool.
        
        Returns:
            Dictionary with pool statistics
        """
        total_agents = len(self.agents)
        active_agents = sum(1 for agent in self.agents.values() if agent.is_active)
        
        type_counts = {
            agent_type: len(agent_ids) 
            for agent_type, agent_ids in self.agent_types.items()
        }
        
        avg_reputation = 0.0
        if self.agents:
            avg_reputation = sum(agent.metrics.reputation_score for agent in self.agents.values()) / len(self.agents)
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "agent_types": type_counts,
            "avg_reputation": avg_reputation,
            "pool_uptime": time.time() - self.created_at
        }