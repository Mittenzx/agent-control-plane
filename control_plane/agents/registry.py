"""
Agent registry and lifecycle management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime

from ..core.interfaces import (
    Agent,
    AgentInfo,
    AgentStatus,
    Task,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Registry for managing agent discovery and metadata."""

    def __init__(self):
        self._agents: Dict[str, Agent] = {}
        self._agent_info: Dict[str, AgentInfo] = {}
        self._capability_index: Dict[str, List[str]] = {}  # capability -> agent_ids
        self._listeners: List[Callable[[AgentInfo, AgentStatus, AgentStatus], None]] = []

    def register(self, agent: Agent) -> None:
        """Register an agent."""
        agent_id = agent.id
        if agent_id in self._agents:
            raise ValueError(f"Agent {agent_id} already registered")

        self._agents[agent_id] = agent
        self._agent_info[agent_id] = agent.info

        # Update capability index
        for cap in agent.capabilities:
            if cap.name not in self._capability_index:
                self._capability_index[cap.name] = []
            self._capability_index[cap.name].append(agent_id)

        logger.info(f"Registered agent: {agent.name} ({agent_id})")
        self._notify_status_change(agent.info, AgentStatus.INITIALIZING, agent.status)

    def unregister(self, agent_id: str) -> Optional[Agent]:
        """Unregister an agent."""
        agent = self._agents.pop(agent_id, None)
        if agent:
            info = self._agent_info.pop(agent_id)
            # Update capability index
            for cap in agent.capabilities:
                if cap.name in self._capability_index:
                    self._capability_index[cap.name] = [
                        aid for aid in self._capability_index[cap.name] if aid != agent_id
                    ]
                    if not self._capability_index[cap.name]:
                        del self._capability_index[cap.name]

            logger.info(f"Unregistered agent: {agent.name} ({agent_id})")
            self._notify_status_change(info, info.status, AgentStatus.STOPPED)
            return agent
        return None

    def get(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def get_info(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent info by ID."""
        return self._agent_info.get(agent_id)

    def list_agents(self, status: Optional[AgentStatus] = None) -> List[AgentInfo]:
        """List all agents, optionally filtered by status."""
        agents = list(self._agent_info.values())
        if status:
            agents = [a for a in agents if a.status == status]
        return agents

    def find_by_capability(self, capability: str) -> List[Agent]:
        """Find agents that have a specific capability."""
        agent_ids = self._capability_index.get(capability, [])
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def find_available_by_capability(self, capability: str) -> List[Agent]:
        """Find idle agents that have a specific capability."""
        agents = self.find_by_capability(capability)
        return [a for a in agents if a.status == AgentStatus.IDLE]

    def add_status_listener(self, listener: Callable[[AgentInfo, AgentStatus, AgentStatus], None]):
        """Add a listener for agent status changes."""
        self._listeners.append(listener)

    def _notify_status_change(
        self, info: AgentInfo, old_status: AgentStatus, new_status: AgentStatus
    ):
        """Notify listeners of status change."""
        for listener in self._listeners:
            try:
                listener(info, old_status, new_status)
            except Exception as e:
                logger.error(f"Error in status listener: {e}")


class AgentLifecycleManager:
    """Manages agent lifecycle: initialization, execution, shutdown."""

    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._initializing: Dict[str, asyncio.Task] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}  # task_id -> task
        self._shutting_down: Dict[str, asyncio.Task] = {}

    async def initialize_agent(self, agent: Agent) -> None:
        """Initialize an agent."""
        agent_id = agent.id
        if agent_id in self._initializing:
            raise ValueError(f"Agent {agent_id} already initializing")

        agent.status = AgentStatus.INITIALIZING

        async def _init():
            try:
                await agent.initialize()
                agent.status = AgentStatus.IDLE
                logger.info(f"Agent {agent.name} initialized successfully")
            except Exception as e:
                agent.status = AgentStatus.FAILED
                logger.error(f"Agent {agent.name} initialization failed: {e}")
                raise
            finally:
                self._initializing.pop(agent_id, None)

        task = asyncio.create_task(_init())
        self._initializing[agent_id] = task
        await task

    async def execute_task(self, agent: Agent, task: Task) -> Dict[str, Any]:
        """Execute a task on an agent."""
        agent_id = agent.id
        task_id = task.id

        if agent.status not in (AgentStatus.IDLE, AgentStatus.WAITING):
            raise ValueError(f"Agent {agent.name} not available (status: {agent.status})")

        if not agent.can_handle_task(task):
            raise ValueError(
                f"Agent {agent.name} cannot handle task requiring '{task.required_capability}'"
            )

        agent.status = AgentStatus.RUNNING
        task.status = TaskStatus.RUNNING
        task.assigned_agent_id = agent_id
        task.started_at = datetime.utcnow()

        # Check timeout
        timeout = task.timeout_seconds

        async def _execute():
            try:
                if timeout:
                    result = await asyncio.wait_for(agent.execute_task(task), timeout=timeout)
                else:
                    result = await agent.execute_task(task)

                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow()
                agent.status = AgentStatus.IDLE
                logger.info(f"Task {task.name} completed on agent {agent.name}")
                return result
            except asyncio.TimeoutError:
                task.status = TaskStatus.FAILED
                task.error = f"Task timed out after {timeout}s"
                agent.status = AgentStatus.FAILED
                logger.error(f"Task {task.name} timed out on agent {agent.name}")
                raise
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                agent.status = AgentStatus.FAILED
                logger.error(f"Task {task.name} failed on agent {agent.name}: {e}")
                raise
            finally:
                self._running_tasks.pop(task_id, None)

        exec_task = asyncio.create_task(_execute())
        self._running_tasks[task_id] = exec_task
        return await exec_task

    async def shutdown_agent(self, agent: Agent, force: bool = False) -> None:
        """Shutdown an agent gracefully."""
        agent_id = agent.id

        if agent_id in self._shutting_down:
            return

        # Cancel running task if any
        running_task = None
        for task_id, task in self._running_tasks.items():
            if task.assigned_agent_id == agent_id:  # type: ignore[attr-defined]  # pre-existing: _running_tasks stores futures
                running_task = task
                break

        if running_task and not force:
            raise ValueError(
                f"Agent {agent.name} has running task {running_task.id}. Use force=True to cancel.",  # type: ignore[attr-defined]  # pre-existing
            )

        if running_task and force:
            running_task.cancel()
            try:
                await running_task
            except asyncio.CancelledError:
                pass

        agent.status = AgentStatus.STOPPED

        async def _shutdown():
            try:
                await agent.shutdown()
                logger.info(f"Agent {agent.name} shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down agent {agent.name}: {e}")
            finally:
                self._shutting_down.pop(agent_id, None)

        task = asyncio.create_task(_shutdown())
        self._shutting_down[agent_id] = task
        await task

    async def shutdown_all(self, force: bool = False) -> None:
        """Shutdown all agents."""
        agents = list(self.registry._agents.values())
        await asyncio.gather(*[self.shutdown_agent(agent, force=force) for agent in agents])
