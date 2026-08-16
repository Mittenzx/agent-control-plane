"""
Core interfaces and base classes for the Agent Control Plane.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
import uuid


class AgentStatus(Enum):
    """Lifecycle states of an agent."""

    INITIALIZING = "initializing"
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TaskStatus(Enum):
    """Lifecycle states of a task."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageType(Enum):
    """Types of messages in the system."""

    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_PROGRESS = "task_progress"
    AGENT_EVENT = "agent_event"
    SYSTEM_EVENT = "system_event"
    HEARTBEAT = "heartbeat"
    CUSTOM = "custom"


@dataclass
class AgentCapability:
    """Describes a capability an agent provides."""

    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class AgentInfo:
    """Metadata about an agent."""

    id: str
    name: str
    type: str
    capabilities: List[AgentCapability]
    status: AgentStatus = AgentStatus.INITIALIZING
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Project:
    """A project is a goal-oriented container for a set of tasks.

    Tasks are the individual units of work that move a project toward its
    goal. A project tracks its own progress as its tasks complete.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    goal: str = ""
    status: TaskStatus = TaskStatus.PENDING
    task_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """A unit of work that can be executed by an agent.

    A task is the smallest piece of executable work. It may optionally belong
    to a :class:`Project` (via ``project_id``), in which case completing it
    advances the project's progress toward its goal.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    required_capability: str = ""
    project_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timeout_seconds: Optional[float] = None
    usage: Optional["UsageRecord"] = None


@dataclass
class UsageRecord:
    """Token + cost usage for a single task execution (from Hermes/OpenRouter).

    Mirrors the usage fields Hermes records per session in its session store.
    ``provider`` is the billing provider (e.g. ``openrouter``).
    """

    model: str = ""
    provider: str = ""
    session_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    actual_cost_usd: Optional[float] = None
    api_call_count: int = 0

    @property
    def prompt_tokens(self) -> int:
        """Alias: input tokens are the prompt tokens."""
        return self.input_tokens

    @property
    def completion_tokens(self) -> int:
        """Alias: output tokens are the completion tokens."""
        return self.output_tokens

    @property
    def cost_usd(self) -> float:
        return self.actual_cost_usd or self.estimated_cost_usd


@dataclass
class Message:
    """Inter-agent communication message."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.CUSTOM
    sender_id: str = ""
    recipient_id: Optional[str] = None  # None = broadcast
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None  # For request/response pairing
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Base class for all agents in the control plane."""

    def __init__(self, agent_info: AgentInfo):
        self.info = agent_info
        self._running = False
        self._message_handlers: Dict[MessageType, Callable] = {}

    @property
    def id(self) -> str:
        return self.info.id

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def capabilities(self) -> List[AgentCapability]:
        return self.info.capabilities

    @property
    def status(self) -> AgentStatus:
        return self.info.status

    @status.setter
    def status(self, value: AgentStatus):
        self.info.status = value
        self.info.updated_at = datetime.utcnow()

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent (load models, connect to services, etc.)."""
        pass

    @abstractmethod
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a task and return the result."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shut down the agent."""
        pass

    def register_message_handler(self, msg_type: MessageType, handler: Callable):
        """Register a handler for a specific message type."""
        self._message_handlers[msg_type] = handler

    async def handle_message(self, message: Message) -> Optional[Message]:
        """Handle an incoming message."""
        handler = self._message_handlers.get(message.type)
        if handler:
            return await handler(message)
        return None

    def can_handle_task(self, task: Task) -> bool:
        """Check if this agent can handle the given task."""
        return any(cap.name == task.required_capability for cap in self.capabilities)


class AgentPlugin(ABC):
    """Base class for agent plugins that extend functionality."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass

    @abstractmethod
    async def on_load(self, control_plane: "ControlPlane") -> None:
        """Called when plugin is loaded."""
        pass

    @abstractmethod
    async def on_unload(self) -> None:
        """Called when plugin is unloaded."""
        pass


class ControlPlane:
    """Main control plane interface - implemented in core/control_plane.py"""

    pass  # Forward reference
