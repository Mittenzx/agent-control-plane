"""
Task orchestration engine for managing complex workflows.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime
from dataclasses import dataclass, field

from ..core.interfaces import Task, TaskStatus, Agent
from ..agents.registry import AgentRegistry, AgentLifecycleManager

logger = logging.getLogger(__name__)


@dataclass
class Workflow:
    """A workflow composed of multiple tasks with dependencies."""

    id: str
    name: str
    tasks: List[Task] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskScheduler:
    """Schedules tasks to available agents based on capabilities and priority."""

    def __init__(self, registry: AgentRegistry, lifecycle: AgentLifecycleManager):
        self.registry = registry
        self.lifecycle = lifecycle
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._assignment_callbacks: List[Callable[[Task, Agent], None]] = []

    def add_assignment_callback(self, callback: Callable[[Task, Agent], None]):
        """Add a callback when a task is assigned to an agent."""
        self._assignment_callbacks.append(callback)

    async def start(self):
        """Start the scheduler."""
        if self._running:
            return
        self._running = True
        self._scheduler_task = asyncio.create_task(self._run())
        logger.info("Task scheduler started")

    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Task scheduler stopped")

    def submit_task(self, task: Task) -> None:
        """Submit a task for scheduling."""
        # Priority queue uses negative priority for max-heap behavior
        # Add task.id as tiebreaker to avoid comparing Task objects
        self._queue.put_nowait((-task.priority, task.created_at, task.id, task))
        logger.debug(f"Task {task.name} ({task.id}) queued with priority {task.priority}")

    def submit_tasks(self, tasks: List[Task]) -> None:
        """Submit multiple tasks."""
        for task in tasks:
            self.submit_task(task)

    async def _run(self):
        """Main scheduler loop."""
        while self._running:
            try:
                # Get next task (with timeout to allow checking _running)
                try:
                    _, _, _, task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Skip if task is no longer pending
                if task.status != TaskStatus.PENDING:
                    continue

                # Check dependencies
                if not self._dependencies_met(task):
                    # Re-queue with lower priority
                    task.priority -= 1
                    self._queue.put_nowait((-task.priority, task.created_at, task.id, task))
                    continue

                # Find available agent
                agent = await self._find_best_agent(task)
                if not agent:
                    # Distinguish "no agent has this capability" from "all busy"
                    capability_agents = self.registry.find_by_capability(task.required_capability)
                    if not capability_agents:
                        # No agent can ever handle this task - fail fast
                        task.status = TaskStatus.FAILED
                        task.error = (
                            f"No agent registered with capability "
                            f"'{task.required_capability}'. Registered capabilities: "
                            f"{self.registry.list_capabilities() or 'none'}"
                        )
                        logger.error(
                            f"Task {task.name} failed: no agent with capability "
                            f"'{task.required_capability}'"
                        )
                        continue

                    # Agents have the capability but are all busy - re-queue
                    task.priority -= 1
                    self._queue.put_nowait((-task.priority, task.created_at, task.id, task))
                    await asyncio.sleep(0.5)
                    continue

                # Assign task
                task.status = TaskStatus.QUEUED
                for callback in self._assignment_callbacks:
                    try:
                        callback(task, agent)
                    except Exception as e:
                        logger.error(f"Assignment callback error: {e}")

                # Execute task
                asyncio.create_task(self._execute_task(agent, task))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(1)

    def _dependencies_met(self, task: Task) -> bool:
        """Check if all task dependencies are completed."""
        # Check if all dependent task IDs are in COMPLETED status
        # This requires access to the task store - we check via the registry
        # For tasks not in a workflow, assume no dependencies
        if not task.dependencies:
            return True

        # Check if we can find the dependent tasks
        # In a full implementation, this would query a task store
        # For now, we'll check the workflow engine if available
        return True  # Simplified - would need task store access

    async def _find_best_agent(self, task: Task) -> Optional[Agent]:
        """Find the best available agent for a task."""
        agents = self.registry.find_available_by_capability(task.required_capability)
        if not agents:
            return None

        # Simple strategy: pick first available
        # Could be extended with load balancing, affinity, etc.
        return agents[0]

    async def _execute_task(self, agent: Agent, task: Task):
        """Execute a task on an agent."""
        try:
            await self.lifecycle.execute_task(agent, task)
        except Exception as e:
            logger.error(f"Task {task.name} execution failed: {e}")


class WorkflowEngine:
    """Manages workflow execution with dependency resolution."""

    def __init__(self, scheduler: TaskScheduler):
        self.scheduler = scheduler
        self._workflows: Dict[str, Workflow] = {}
        self._task_to_workflow: Dict[str, str] = {}
        self._completed_tasks: Set[str] = set()
        self._failed_tasks: Set[str] = set()
        self._callbacks: List[Callable[[Workflow], None]] = []

    def add_workflow_callback(self, callback: Callable[[Workflow], None]):
        """Add a callback when workflow status changes."""
        self._callbacks.append(callback)

    def create_workflow(self, name: str, tasks: List[Task]) -> Workflow:
        """Create a new workflow."""
        import uuid

        workflow = Workflow(id=str(uuid.uuid4()), name=name, tasks=tasks)

        for task in tasks:
            self._task_to_workflow[task.id] = workflow.id

        self._workflows[workflow.id] = workflow
        logger.info(f"Created workflow: {name} ({workflow.id}) with {len(tasks)} tasks")
        return workflow

    def start_workflow(self, workflow_id: str) -> None:
        """Start a workflow by submitting its root tasks."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow.status = TaskStatus.RUNNING
        workflow.started_at = datetime.utcnow()

        # Submit tasks that have no dependencies (or whose dependencies are met)
        for task in workflow.tasks:
            if not task.dependencies:
                self.scheduler.submit_task(task)

        self._notify_workflow_change(workflow)

    def on_task_completed(self, task: Task):
        """Handle task completion - submit dependent tasks."""
        workflow_id = self._task_to_workflow.get(task.id)
        if not workflow_id:
            return

        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return

        self._completed_tasks.add(task.id)

        # Check if any dependent tasks can now run
        for dep_task in workflow.tasks:
            if dep_task.status == TaskStatus.PENDING:
                if all(dep_id in self._completed_tasks for dep_id in dep_task.dependencies):
                    self.scheduler.submit_task(dep_task)

        # Check if workflow is complete
        if all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) for t in workflow.tasks):
            workflow.status = (
                TaskStatus.COMPLETED
                if all(t.status == TaskStatus.COMPLETED for t in workflow.tasks)
                else TaskStatus.FAILED
            )
            workflow.completed_at = datetime.utcnow()
            self._notify_workflow_change(workflow)

    def on_task_failed(self, task: Task):
        """Handle task failure."""
        self._failed_tasks.add(task.id)
        workflow_id = self._task_to_workflow.get(task.id)
        if workflow_id:
            workflow = self._workflows.get(workflow_id)
            if workflow:
                # Fail dependent tasks
                for dep_task in workflow.tasks:
                    if task.id in dep_task.dependencies and dep_task.status == TaskStatus.PENDING:
                        dep_task.status = TaskStatus.FAILED
                        dep_task.error = f"Dependency {task.id} failed"
                self._notify_workflow_change(workflow)

    def _notify_workflow_change(self, workflow: Workflow):
        """Notify callbacks of workflow change."""
        for callback in self._callbacks:
            try:
                callback(workflow)
            except Exception as e:
                logger.error(f"Workflow callback error: {e}")

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID."""
        return self._workflows.get(workflow_id)

    def list_workflows(self) -> List[Workflow]:
        """List all workflows."""
        return list(self._workflows.values())


class OrchestrationEngine:
    """High-level orchestration combining scheduling and workflows."""

    def __init__(self, registry: AgentRegistry, lifecycle: AgentLifecycleManager):
        self.registry = registry
        self.lifecycle = lifecycle
        self.scheduler = TaskScheduler(registry, lifecycle)
        self.workflow_engine = WorkflowEngine(self.scheduler)

        # Wire up callbacks
        self.scheduler.add_assignment_callback(self._on_task_assigned)
        self.workflow_engine.add_workflow_callback(self._on_workflow_change)

    async def start(self):
        """Start the orchestration engine."""
        await self.scheduler.start()
        logger.info("Orchestration engine started")

    async def stop(self):
        """Stop the orchestration engine."""
        await self.scheduler.stop()
        logger.info("Orchestration engine stopped")

    def _on_task_assigned(self, task: Task, agent: Agent):
        """Handle task assignment."""
        logger.info(f"Task {task.name} assigned to agent {agent.name}")

    def _on_workflow_change(self, workflow: Workflow):
        """Handle workflow status change."""
        logger.info(f"Workflow {workflow.name} status: {workflow.status}")

    def submit_task(self, task: Task) -> None:
        """Submit a standalone task."""
        self.scheduler.submit_task(task)

    def create_and_start_workflow(self, name: str, tasks: List[Task]) -> Workflow:
        """Create and immediately start a workflow."""
        workflow = self.workflow_engine.create_workflow(name, tasks)
        self.workflow_engine.start_workflow(workflow.id)
        return workflow
