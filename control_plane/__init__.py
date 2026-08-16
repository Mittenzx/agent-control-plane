"""
Agent Control Plane - A modular, customizable control plane for AI agents.

This package provides:
- Agent registry and lifecycle management
- Task orchestration and workflow execution
- Inter-agent messaging (request/response, pub/sub, events)
- Plugin system for extensibility
- Configuration management
"""

from .core import (
    ControlPlane,
    Agent,
    AgentInfo,
    AgentStatus,
    AgentCapability,
    Task,
    TaskStatus,
    Project,
    UsageRecord,
    Message,
    MessageType,
    Workflow,
    AgentPlugin,
)
from .config import (
    ConfigManager,
    ControlPlaneConfig,
    AgentConfig,
    DEFAULT_CONFIG,
)
from .agents import AgentRegistry, AgentLifecycleManager
from .orchestration import OrchestrationEngine, TaskScheduler, WorkflowEngine
from .messaging import MessageBus, RequestResponseBus, EventBus, SystemEvents
from .plugins import (
    PluginManager,
    CapabilityRegistry,
    AgentFactory,
    ExtensionPoint,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "ControlPlane",
    "Agent",
    "AgentInfo",
    "AgentStatus",
    "AgentCapability",
    "Task",
    "TaskStatus",
    "Project",
    "UsageRecord",
    "Message",
    "MessageType",
    "Workflow",
    "AgentPlugin",
    # Config
    "ConfigManager",
    "ControlPlaneConfig",
    "AgentConfig",
    "DEFAULT_CONFIG",
    # Agents
    "AgentRegistry",
    "AgentLifecycleManager",
    # Orchestration
    "OrchestrationEngine",
    "TaskScheduler",
    "WorkflowEngine",
    "Workflow",
    # Messaging
    "MessageBus",
    "RequestResponseBus",
    "EventBus",
    "SystemEvents",
    # Plugins
    "PluginManager",
    "CapabilityRegistry",
    "AgentFactory",
    "ExtensionPoint",
]
