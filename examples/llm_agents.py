"""
Example LLM-based agent implementation.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from control_plane.core import Agent, AgentInfo, AgentCapability, Task, AgentStatus
from control_plane.core.interfaces import ControlPlane

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM agent."""
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    system_prompt: str = "You are a helpful AI assistant."


class LLMAgent(Agent):
    """An agent that uses an LLM to execute tasks."""
    
    def __init__(
        self, 
        agent_info: AgentInfo, 
        config: Optional[LLMConfig] = None,
        control_plane: Optional[ControlPlane] = None
    ):
        super().__init__(agent_info)
        self.config = config or LLMConfig()
        self.control_plane = control_plane
        self._llm_client = None
    
    async def initialize(self) -> None:
        """Initialize the LLM client."""
        # In a real implementation, this would connect to an LLM API
        # For example: openai.AsyncOpenAI(), anthropic.AsyncAnthropic(), etc.
        logger.info(f"Initializing LLM agent {self.name} with model {self.config.model}")
        # self._llm_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        await asyncio.sleep(0.1)  # Simulate initialization
        logger.info(f"LLM agent {self.name} initialized")
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a task using the LLM."""
        logger.info(f"Agent {self.name} executing task: {task.name}")
        
        # Build prompt from task
        prompt = self._build_prompt(task)
        
        # Call LLM (simulated)
        result = await self._call_llm(prompt, task)
        
        logger.info(f"Agent {self.name} completed task: {task.name}")
        return result
    
    async def shutdown(self) -> None:
        """Shutdown the agent."""
        logger.info(f"Shutting down LLM agent {self.name}")
        if self._llm_client:
            # await self._llm_client.close()
            pass
    
    def _build_prompt(self, task: Task) -> str:
        """Build a prompt from the task."""
        parts = [self.config.system_prompt, ""]
        
        if task.description:
            parts.append(f"Task: {task.description}")
        
        if task.payload:
            parts.append("Input:")
            for key, value in task.payload.items():
                parts.append(f"  {key}: {value}")
        
        parts.append("\nProvide your response as JSON.")
        return "\n".join(parts)
    
    async def _call_llm(self, prompt: str, task: Task) -> Dict[str, Any]:
        """Call the LLM API."""
        # Simulated LLM response
        await asyncio.sleep(0.5)  # Simulate API call
        
        # Return mock result based on capability
        capability = task.required_capability
        
        if capability == "research":
            return {
                "findings": [
                    "Finding 1: Key information about the topic",
                    "Finding 2: Additional relevant data",
                    "Finding 3: Supporting evidence"
                ],
                "sources": ["source1.com", "source2.com"],
                "confidence": 0.85
            }
        elif capability == "summarize":
            text = task.payload.get("text", "")
            return {
                "summary": f"Summary of {len(text)} characters: Key points extracted...",
                "key_points": ["Point 1", "Point 2", "Point 3"],
                "compression_ratio": 0.15
            }
        elif capability == "code_generation":
            spec = task.payload.get("specification", "")
            return {
                "code": f"# Generated code for: {spec}\ndef main():\n    pass\n",
                "language": "python",
                "tests": ["test_case_1", "test_case_2"]
            }
        elif capability == "code_review":
            code = task.payload.get("code", "")
            return {
                "issues": [
                    {"severity": "warning", "message": "Consider using type hints", "line": 10},
                    {"severity": "info", "message": "Function could be simplified", "line": 25}
                ],
                "score": 8.5,
                "suggestions": ["Add docstrings", "Extract helper functions"]
            }
        elif capability == "task_planning":
            goal = task.payload.get("goal", "")
            return {
                "plan": [
                    {"step": 1, "action": "Research requirements", "capability": "research"},
                    {"step": 2, "action": "Design architecture", "capability": "workflow_design"},
                    {"step": 3, "action": "Implement core features", "capability": "code_generation"},
                    {"step": 4, "action": "Test and validate", "capability": "code_review"}
                ],
                "estimated_steps": 4
            }
        elif capability == "workflow_design":
            return {
                "workflow": {
                    "name": "Generated Workflow",
                    "tasks": [
                        {"name": "Task 1", "capability": "research", "depends_on": []},
                        {"name": "Task 2", "capability": "code_generation", "depends_on": ["Task 1"]},
                        {"name": "Task 3", "capability": "code_review", "depends_on": ["Task 2"]}
                    ]
                }
            }
        
        return {"result": f"Completed {capability} task", "status": "success"}


class ResearchAgent(LLMAgent):
    """Specialized agent for research tasks."""
    
    def __init__(self, agent_info: AgentInfo, config: Optional[LLMConfig] = None, control_plane: Optional[ControlPlane] = None):
        # Override capabilities for research
        if not agent_info.capabilities:
            agent_info.capabilities = [
                AgentCapability(
                    name="research",
                    description="Web research and information gathering",
                    input_schema={"topic": "string", "depth": "string"},
                    output_schema={"findings": "array", "sources": "array", "confidence": "number"}
                ),
                AgentCapability(
                    name="summarize",
                    description="Summarize long texts",
                    input_schema={"text": "string", "max_length": "integer"},
                    output_schema={"summary": "string", "key_points": "array"}
                )
            ]
        
        # Use research-optimized config
        research_config = config or LLMConfig(
            model="gpt-4",
            temperature=0.3,  # Lower temperature for factual tasks
            system_prompt="You are a research assistant. Provide accurate, well-sourced information."
        )
        
        super().__init__(agent_info, research_config, control_plane)


class CoderAgent(LLMAgent):
    """Specialized agent for coding tasks."""
    
    def __init__(self, agent_info: AgentInfo, config: Optional[LLMConfig] = None, control_plane: Optional[ControlPlane] = None):
        if not agent_info.capabilities:
            agent_info.capabilities = [
                AgentCapability(
                    name="code_generation",
                    description="Generate code from specifications",
                    input_schema={"specification": "string", "language": "string", "framework": "string"},
                    output_schema={"code": "string", "language": "string", "tests": "array"}
                ),
                AgentCapability(
                    name="code_review",
                    description="Review code for issues",
                    input_schema={"code": "string", "language": "string"},
                    output_schema={"issues": "array", "score": "number", "suggestions": "array"}
                )
            ]
        
        coder_config = config or LLMConfig(
            model="gpt-4",
            temperature=0.2,  # Low temperature for code
            system_prompt="You are a senior software engineer. Write clean, well-tested, maintainable code."
        )
        
        super().__init__(agent_info, coder_config, control_plane)


class PlannerAgent(LLMAgent):
    """Specialized agent for planning and workflow design."""
    
    def __init__(self, agent_info: AgentInfo, config: Optional[LLMConfig] = None, control_plane: Optional[ControlPlane] = None):
        if not agent_info.capabilities:
            agent_info.capabilities = [
                AgentCapability(
                    name="task_planning",
                    description="Break down goals into executable tasks",
                    input_schema={"goal": "string", "constraints": "object"},
                    output_schema={"plan": "array", "estimated_steps": "integer"}
                ),
                AgentCapability(
                    name="workflow_design",
                    description="Design multi-step workflows with dependencies",
                    input_schema={"objective": "string", "available_capabilities": "array"},
                    output_schema={"workflow": "object"}
                )
            ]
        
        planner_config = config or LLMConfig(
            model="gpt-4",
            temperature=0.5,
            system_prompt="You are a project planner. Break down complex goals into clear, actionable steps with proper dependencies."
        )
        
        super().__init__(agent_info, planner_config, control_plane)


# Factory functions for easy agent creation
def create_research_agent(name: str = "researcher", control_plane: Optional[ControlPlane] = None) -> ResearchAgent:
    """Create a research agent."""
    import uuid
    info = AgentInfo(
        id=str(uuid.uuid4()),
        name=name,
        type="research_agent",
        capabilities=[]
    )
    return ResearchAgent(info, control_plane=control_plane)


def create_coder_agent(name: str = "coder", control_plane: Optional[ControlPlane] = None) -> CoderAgent:
    """Create a coder agent."""
    import uuid
    info = AgentInfo(
        id=str(uuid.uuid4()),
        name=name,
        type="coder_agent",
        capabilities=[]
    )
    return CoderAgent(info, control_plane=control_plane)


def create_planner_agent(name: str = "planner", control_plane: Optional[ControlPlane] = None) -> PlannerAgent:
    """Create a planner agent."""
    import uuid
    info = AgentInfo(
        id=str(uuid.uuid4()),
        name=name,
        type="planner_agent",
        capabilities=[]
    )
    return PlannerAgent(info, control_plane=control_plane)