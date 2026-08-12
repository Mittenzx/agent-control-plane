"""
Main Control Plane - ties all components together.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..core.interfaces import (
    Agent,
    AgentInfo,
    AgentStatus,
    AgentCapability,
    Task,
    TaskStatus,
    Message,
    MessageType,
    ControlPlane as ControlPlaneInterface,
)
from ..orchestration.engine import Workflow
from ..agents.registry import AgentRegistry, AgentLifecycleManager
from ..orchestration.engine import OrchestrationEngine
from ..messaging.bus import RequestResponseBus, EventBus, SystemEvents
from ..plugins.manager import PluginManager, CapabilityRegistry, AgentFactory
from ..config.manager import ConfigManager, ControlPlaneConfig, AgentConfig

logger = logging.getLogger(__name__)


class ControlPlane(ControlPlaneInterface):
    """
    Main control plane for managing AI agents.

    This is the central coordinator that brings together:
    - Agent registry and lifecycle management
    - Task orchestration and workflow execution
    - Inter-agent messaging
    - Plugin system for extensibility
    - Configuration management
    """

    def __init__(self, config: Optional[ControlPlaneConfig] = None):
        self.config = config or ControlPlaneConfig()
        self.config_manager = ConfigManager(self.config)

        # Core components
        self.registry = AgentRegistry()
        self.lifecycle = AgentLifecycleManager(self.registry)
        self.orchestration = OrchestrationEngine(self.registry, self.lifecycle)
        self.message_bus = RequestResponseBus()
        self.event_bus = EventBus()
        self.plugin_manager = PluginManager(self)
        self.capability_registry = CapabilityRegistry()
        self.agent_factory = AgentFactory(self)

        # State
        self._running = False
        self._started_at: Optional[datetime] = None
        self._tasks: Dict[str, Task] = {}
        self._workflows: Dict[str, Workflow] = {}

        # Wire up event handlers
        self._setup_event_handlers()

        logger.info(f"ControlPlane initialized: {self.config.name}")

    def _setup_event_handlers(self) -> None:
        """Set up internal event handlers."""
        # Agent status changes
        self.registry.add_status_listener(self._on_agent_status_change)

        # Task completion from orchestration
        self.orchestration.workflow_engine.add_workflow_callback(self._on_workflow_change)
        self.orchestration.scheduler.add_assignment_callback(self._on_task_assigned)

    async def start(self) -> None:
        """Start the control plane and all components."""
        if self._running:
            logger.warning("Control plane already running")
            return

        self._running = True
        self._started_at = datetime.utcnow()

        # Start orchestration engine
        await self.orchestration.start()

        # Load and initialize configured agents
        await self._initialize_agents()

        # Load plugins
        await self._load_plugins()

        # Emit startup event
        await self.event_bus.emit(
            SystemEvents.AGENT_REGISTERED,
            {"control_plane": self.config.name, "started_at": self._started_at.isoformat()},
        )

        logger.info(f"Control plane started: {self.config.name}")

    async def stop(self) -> None:
        """Stop the control plane and all components."""
        if not self._running:
            return

        self._running = False

        # Stop orchestration
        await self.orchestration.stop()

        # Shutdown all agents
        await self.lifecycle.shutdown_all(force=True)

        # Unload plugins
        await self.plugin_manager.unload_all()

        # Emit shutdown event
        await self.event_bus.emit(
            SystemEvents.AGENT_UNREGISTERED, {"control_plane": self.config.name}
        )

        logger.info(f"Control plane stopped: {self.config.name}")

    async def _initialize_agents(self) -> None:
        """Initialize agents from configuration."""
        agent_configs = self.config_manager.get_agent_configs()

        for agent_config in agent_configs:
            if not agent_config.auto_start:
                continue

            try:
                await self.create_and_start_agent(agent_config)
            except Exception as e:
                logger.error(f"Failed to create agent {agent_config.name}: {e}")

    async def _load_plugins(self) -> None:
        """Load plugins from configuration."""
        for plugin_config in self.config.plugins:
            try:
                module_path = plugin_config.get("module")
                class_name = plugin_config.get("class")
                config = plugin_config.get("config", {})

                if module_path and class_name:
                    await self.plugin_manager.load_plugin_from_module(
                        module_path, class_name, config
                    )
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_config}: {e}")

    async def create_and_start_agent(self, agent_config: AgentConfig) -> Agent:
        """Create and start an agent from configuration."""
        # Create agent info
        capabilities = [
            AgentCapability(
                name=cap["name"],
                description=cap.get("description", ""),
                input_schema=cap.get("input_schema", {}),
                output_schema=cap.get("output_schema", {}),
                tags=cap.get("tags", []),
            )
            for cap in agent_config.capabilities
        ]

        agent_info = AgentInfo(
            id=str(uuid.uuid4()),
            name=agent_config.name,
            type=agent_config.type,
            capabilities=capabilities,
            metadata=agent_config.config,
        )

        # Create agent instance
        agent = await self.agent_factory.create_agent(
            agent_config.type, agent_info, agent_config.config
        )

        # Register and initialize
        self.registry.register(agent)
        await self.lifecycle.initialize_agent(agent)

        # Register capabilities
        for cap in capabilities:
            self.capability_registry.register_capability(cap, agent.id)

        # Register message queue
        queue: asyncio.Queue = asyncio.Queue()
        self.message_bus.register_agent_queue(agent.id, queue)

        # Subscribe to messages
        self.message_bus.subscribe(agent.id, agent.handle_message)

        # Emit event
        await self.event_bus.emit(
            SystemEvents.AGENT_REGISTERED,
            {"agent_id": agent.id, "agent_name": agent.name, "agent_type": agent_config.type},
        )

        logger.info(f"Created and started agent: {agent.name} ({agent.id})")
        return agent

    async def stop_agent(self, agent_id: str, force: bool = False) -> None:
        """Stop and unregister an agent."""
        agent = self.registry.get(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        # Shutdown
        await self.lifecycle.shutdown_agent(agent, force=force)

        # Unregister capabilities
        for cap in agent.capabilities:
            self.capability_registry.unregister_capability(cap.name, agent.id)

        # Unregister message queue
        self.message_bus.unregister_agent_queue(agent.id)
        self.message_bus.unsubscribe(agent.id, agent.handle_message)

        # Unregister from registry
        self.registry.unregister(agent_id)

        # Emit event
        await self.event_bus.emit(
            SystemEvents.AGENT_UNREGISTERED, {"agent_id": agent_id, "agent_name": agent.name}
        )

        logger.info(f"Stopped agent: {agent.name} ({agent_id})")

    def submit_task(self, task: Task) -> str:
        """Submit a task for execution."""
        self._tasks[task.id] = task
        self.orchestration.submit_task(task)

        # Emit event
        self.event_bus.emit_sync(
            SystemEvents.TASK_SUBMITTED,
            {
                "task_id": task.id,
                "task_name": task.name,
                "required_capability": task.required_capability,
            },
        )

        return task.id

    def submit_workflow(self, name: str, tasks: List[Task]) -> Workflow:
        """Submit a workflow for execution."""
        workflow = self.orchestration.create_and_start_workflow(name, tasks)
        self._workflows[workflow.id] = workflow

        # Store tasks
        for task in tasks:
            self._tasks[task.id] = task

        return workflow

    async def send_message(self, message: Message) -> None:
        """Send a message through the message bus."""
        await self.message_bus.send(message)

    async def request(
        self,
        sender_id: str,
        recipient_id: str,
        payload: Dict[str, Any],
        msg_type: MessageType = MessageType.TASK_REQUEST,
        timeout: Optional[float] = None,
    ) -> Message:
        """Send a request and wait for response."""
        return await self.message_bus.request(sender_id, recipient_id, payload, msg_type, timeout)

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self.registry.get(agent_id)

    def get_agent_info(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent info by ID."""
        return self.registry.get_info(agent_id)

    def list_agents(self, status: Optional[AgentStatus] = None) -> List[AgentInfo]:
        """List all agents."""
        return self.registry.list_agents(status)

    def find_agents_by_capability(self, capability: str) -> List[Agent]:
        """Find agents with a specific capability."""
        return self.registry.find_by_capability(capability)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return self._workflows.get(workflow_id)

    def list_workflows(self) -> List[Workflow]:
        """List all workflows."""
        return list(self._workflows.values())

    # Event handlers
    def _on_agent_status_change(
        self, info: AgentInfo, old_status: AgentStatus, new_status: AgentStatus
    ):
        """Handle agent status change."""
        self.event_bus.emit_sync(
            SystemEvents.AGENT_STATUS_CHANGED,
            {
                "agent_id": info.id,
                "agent_name": info.name,
                "old_status": old_status.value,
                "new_status": new_status.value,
            },
        )

    def _on_task_assigned(self, task: Task, agent: Agent):
        """Handle task assignment."""
        self.event_bus.emit_sync(
            SystemEvents.TASK_STARTED,
            {
                "task_id": task.id,
                "task_name": task.name,
                "agent_id": agent.id,
                "agent_name": agent.name,
            },
        )

    def _on_workflow_change(self, workflow: Workflow):
        """Handle workflow status change."""
        event = (
            SystemEvents.WORKFLOW_COMPLETED
            if workflow.status == TaskStatus.COMPLETED
            else SystemEvents.WORKFLOW_FAILED
        )
        self.event_bus.emit_sync(
            event,
            {
                "workflow_id": workflow.id,
                "workflow_name": workflow.name,
                "status": workflow.status.value,
            },
        )

    # Properties
    @property
    def running(self) -> bool:
        return self._running

    @property
    def uptime(self) -> Optional[float]:
        if self._started_at:
            return (datetime.utcnow() - self._started_at).total_seconds()
        return None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
