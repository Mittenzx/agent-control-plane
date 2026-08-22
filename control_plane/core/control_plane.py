"""
Main Control Plane - ties all components together.
"""

import asyncio
import logging
import os
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
    Project,
    Message,
    MessageType,
    ControlPlane as ControlPlaneInterface,
)
from ..orchestration.engine import Workflow
from ..persistence.store import PersistenceStore
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

        # Register the Hermes agent type so configs can use
        # {"type": "hermes_agent", ...} to run agents on real Hermes processes
        self._register_hermes_agent_type()

        # State
        self._running = False
        self._started_at: Optional[datetime] = None
        self._tasks: Dict[str, Task] = {}
        self._workflows: Dict[str, Workflow] = {}
        self._projects: Dict[str, Project] = {}

        # Give the scheduler a live reference to the task store so
        # dependency resolution works across all tasks/subtasks.
        self.orchestration.scheduler._task_store = self._tasks

        # SQLite persistence (projects/tasks/usage survive restarts)
        self._store: Optional[PersistenceStore] = None
        if self.config.persistence_enabled:
            db_path = os.path.join(self.config.data_dir, "control-plane.db")
            try:
                self._store = PersistenceStore(db_path)
            except Exception as e:
                logger.warning(f"Persistence disabled: {e}")
                self._store = None

        # Wire up event handlers
        self._setup_event_handlers()

        logger.info(f"ControlPlane initialized: {self.config.name}")

    def _register_hermes_agent_type(self) -> None:
        """Register the hermes_agent type with the agent factory."""
        from ..agents.hermes_agent import HermesAgent

        def _builder(agent_info, config, cp):
            config = config or {}
            return HermesAgent(
                agent_info,
                hermes_command=config.get("hermes_command", "hermes"),
                model=config.get("model"),
                provider=config.get("provider"),
                timeout_seconds=config.get("timeout_seconds", 300.0),
                cwd=config.get("cwd"),
            )

        self.agent_factory.register_builder("hermes_agent", _builder)

    def _setup_event_handlers(self) -> None:
        """Set up internal event handlers."""
        # Agent status changes
        self.registry.add_status_listener(self._on_agent_status_change)

        # Task completion from orchestration
        self.orchestration.workflow_engine.add_workflow_callback(self._on_workflow_change)
        self.orchestration.scheduler.add_assignment_callback(self._on_task_assigned)

        # Task completion callback (for subtask aggregation + agent-driven spawn)
        self.lifecycle.add_task_completed_callback(self._on_task_completed)

        # Task status-change callback (keeps project status live as tasks transition)
        self.lifecycle.add_task_status_callback(self._on_task_status_change)

    async def start(self) -> None:
        """Start the control plane and all components."""
        if self._running:
            logger.warning("Control plane already running")
            return

        self._running = True
        self._started_at = datetime.utcnow()

        # Load persisted projects/tasks/usage from SQLite (if enabled)
        self._load_from_store()

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

        # Persist state to SQLite on shutdown
        self._persist()
        if self._store:
            try:
                self._store.close()
            except Exception:
                pass

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
        """Submit a task for execution.

        If the task has a ``project_id``, it is associated with that project
        and the project's progress is recomputed.
        """
        self._tasks[task.id] = task
        self.orchestration.submit_task(task)

        # Associate with project if applicable
        if task.project_id:
            self._associate_task_with_project(task)

        # Emit event
        self.event_bus.emit_sync(
            SystemEvents.TASK_SUBMITTED,
            {
                "task_id": task.id,
                "task_name": task.name,
                "required_capability": task.required_capability,
            },
        )

        self._persist()
        return task.id

    # ----- Project management -----

    def create_project(
        self,
        name: str,
        description: str = "",
        goal: str = "",
    ) -> Project:
        """Create a new project (a goal-oriented container for tasks)."""
        project = Project(name=name, description=description, goal=goal)
        self._projects[project.id] = project
        logger.info(f"Created project: {name} ({project.id})")
        self.event_bus.emit_sync(
            "project.created",
            {"project_id": project.id, "project_name": project.name},
        )
        self._persist()
        return project

    def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID."""
        return self._projects.get(project_id)

    def list_projects(self) -> List[Project]:
        """List all projects."""
        return list(self._projects.values())

    def get_project_tasks(self, project_id: str) -> List[Task]:
        """Get all tasks belonging to a project."""
        return [t for t in self._tasks.values() if t.project_id == project_id]

    def project_progress(self, project_id: str) -> Dict[str, Any]:
        """Compute progress for a project from its tasks.

        Returns counts and a 0-100 completion percentage plus an overall
        status derived from the task statuses.
        """
        tasks = self.get_project_tasks(project_id)
        total = len(tasks)
        done = len([t for t in tasks if t.status == TaskStatus.COMPLETED])
        failed = len([t for t in tasks if t.status == TaskStatus.FAILED])
        running = len([t for t in tasks if t.status == TaskStatus.RUNNING])
        pending = total - done - failed - running

        pct = round((done / total * 100) if total else 0)

        project = self.get_project(project_id)
        status = project.status if project else TaskStatus.PENDING
        if total and failed and not running and not pending:
            status = TaskStatus.FAILED
        elif total and done == total:
            status = TaskStatus.COMPLETED
        elif running or pending:
            status = TaskStatus.RUNNING

        # Per-project cost rollup (sum of UsageRecord cost for this project's tasks)
        cost = sum((t.usage.cost_usd or 0.0) for t in tasks if t.usage)

        return {
            "total": total,
            "completed": done,
            "failed": failed,
            "running": running,
            "pending": pending,
            "progress_pct": pct,
            "status": status.value,
            "cost_usd": round(cost, 6),
        }

    def _associate_task_with_project(self, task: Task) -> None:
        """Link a task to its project and refresh project state."""
        project = self._projects.get(task.project_id or "")
        if not project:
            logger.warning(f"Task {task.name} references unknown project {task.project_id}")
            return
        if task.id not in project.task_ids:
            project.task_ids.append(task.id)
        project.updated_at = datetime.utcnow()
        progress = self.project_progress(project.id)
        project.status = TaskStatus(progress["status"])

    def project_cost(self, project_id: str) -> float:
        """Total OpenRouter cost (USD) summed across a project's tasks."""
        return round(
            sum((t.usage.cost_usd or 0.0) for t in self.get_project_tasks(project_id) if t.usage),
            6,
        )

    def set_project_budget(self, project_id: str, budget_usd: Optional[float]) -> None:
        """Set (or clear) a project's spend budget used to trigger alerts."""
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")
        project.budget_usd = budget_usd
        project.updated_at = datetime.utcnow()
        self._persist()
        logger.info(f"Project {project.name} budget set to {budget_usd}")

    def cost_alerts(self) -> List[Dict[str, Any]]:
        """Return budget-exceeded alerts across all projects.

        A project with a ``budget_usd`` whose total cost has reached or passed
        that threshold produces an alert. Also flags projects that are close
        (>= 80% of budget).
        """
        alerts = []
        for project in self.list_projects():
            budget = project.budget_usd
            if not budget:
                continue
            cost = self.project_cost(project.id)
            pct = cost / budget if budget else 0.0
            if cost >= budget:
                alerts.append(
                    {
                        "level": "critical",
                        "project_id": project.id,
                        "project": project.name,
                        "cost_usd": cost,
                        "budget_usd": budget,
                        "message": f"Project '{project.name}' exceeded its ${budget:.2f} budget (spent ${cost:.4f})",
                    }
                )
            elif pct >= 0.8:
                alerts.append(
                    {
                        "level": "warning",
                        "project_id": project.id,
                        "project": project.name,
                        "cost_usd": cost,
                        "budget_usd": budget,
                        "message": f"Project '{project.name}' has used {pct:.0%} of its ${budget:.2f} budget",
                    }
                )
        return alerts

    # ----- Subtask / decomposition support -----

    def spawn_subtasks(self, parent_task: Task, subtask_specs: List[Dict[str, Any]]) -> List[Task]:
        """Create and schedule subtasks for a parent task.

        Subtasks inherit the parent's ``project_id`` and are linked back to the
        parent via ``parent_task_id``. The parent is marked with the
        ``has_subtasks`` metadata so aggregation logic knows to wait for them.
        Returns the created subtask Task objects.
        """
        subtasks = []
        for spec in subtask_specs:
            st = Task(
                name=spec.get("name", f"Subtask of {parent_task.name}"),
                description=spec.get("description", ""),
                required_capability=spec.get(
                    "required_capability", parent_task.required_capability
                ),
                project_id=parent_task.project_id,
                parent_task_id=parent_task.id,
                payload=spec.get("payload", {}),
                priority=spec.get("priority", parent_task.priority),
                dependencies=spec.get("dependencies", []),
            )
            st.metadata["is_subtask"] = True
            self._tasks[st.id] = st
            self.orchestration.submit_task(st)
            # Associate with parent's project
            if st.project_id:
                self._associate_task_with_project(st)
            # Link parent's dependencies: follow-up subtasks wait on their siblings
            subtasks.append(st)

        # Record parent as decomposed so it won't be marked complete itself
        parent_task.metadata["has_subtasks"] = True
        parent_task.metadata["subtask_ids"] = [st.id for st in subtasks]
        logger.info(f"Parent task {parent_task.name} decomposed into {len(subtasks)} subtask(s)")
        self._persist()
        return subtasks

    def _on_task_status_change(
        self, task: Task, old_status: TaskStatus, new_status: TaskStatus
    ) -> None:
        """Refresh project status whenever any task changes state.

        This keeps ``Project.status`` accurate in the backend as tasks move
        through pending -> running -> completed/failed/cancelled (not just on
        completion), so the stored status always reflects live progress.
        """
        if not task.project_id:
            return
        project = self.get_project(task.project_id)
        if not project:
            return
        progress = self.project_progress(project.id)
        project.status = TaskStatus(progress["status"])
        project.updated_at = datetime.utcnow()

    def _on_task_completed(self, task: Task, result: Optional[Dict[str, Any]]) -> None:
        """Handle task completion: refresh project + drive spawning/aggregation."""
        # Agent-driven spawn: if result contains a 'spawn' key, create subtasks.
        # The spawn directive can live directly in the result dict, or inside the
        # agent's textual output as a JSON block (Hermes agents work this way).
        specs = None
        if result and isinstance(result, dict):
            if result.get("spawn"):
                specs = result["spawn"]
            elif isinstance(result.get("output"), str):
                specs = self._extract_spawn_from_output(result["output"])

        if specs:
            try:
                if isinstance(specs, list):
                    # If this task was a planner for a parent, route subtasks to
                    # the original parent; otherwise spawn under this task.
                    parent_id = task.metadata.get("is_planner_for") or task.id
                    parent = self._tasks.get(parent_id) or task
                    self.spawn_subtasks(parent, specs)
                    # The parent's plan is ready; ensure it waits on subtasks
                    parent.metadata["has_subtasks"] = True
            except Exception as e:
                logger.error(f"Failed to spawn subtasks from result: {e}")

        # If this task was decomposed, run aggregation once all subtasks finish
        if task.parent_task_id:
            self._maybe_complete_parent(task.parent_task_id)

        # Refresh enclosing project progress
        if task.project_id:
            project = self.get_project(task.project_id)
            if project:
                progress = self.project_progress(project.id)
                project.status = TaskStatus(progress["status"])
                project.updated_at = datetime.utcnow()

        self._persist()

    @staticmethod
    def _extract_spawn_from_output(output: str) -> Optional[List[Dict[str, Any]]]:
        """Look for a JSON array (list of subtask specs) in an agent's output.

        An agent that was asked to decompose a task can emit a JSON array with
        keys ``name``/``description``/``required_capability``/``payload``. We find
        the first JSON array in the output and treat it as the subtask plan.
        Fallback: also accept a JSON object with a ``spawn`` list.
        """
        import json
        import re

        if not output:
            return None

        # 1) JSON object containing a "spawn": [...] list
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict) and isinstance(parsed.get("spawn"), list):
                return parsed["spawn"]
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        # 2) A JSON array embedded in surrounding text
        match = re.search(r"\[\s*\{[\s\S]*?\}\s*\]", output)
        if match:
            try:
                arr = json.loads(match.group(0))
                if isinstance(arr, list) and arr and isinstance(arr[0], dict):
                    return arr
            except Exception:
                pass
        return None

    def _maybe_complete_parent(self, parent_task_id: str) -> None:
        """Complete a parent task once all of its subtasks are done."""
        parent = self._tasks.get(parent_task_id)
        if not parent:
            return
        subtask_ids = [
            t.id
            for t in self._tasks.values()
            if t.parent_task_id == parent_task_id and t.metadata.get("is_subtask")
        ]
        if not subtask_ids:
            return

        remaining = [t for t in self._tasks.values() if t.id in subtask_ids]
        all_terminal = all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) for t in remaining)
        if not all_terminal:
            return  # some subtasks still running/pending

        failed = [t for t in remaining if t.status == TaskStatus.FAILED]
        # Mark parent complete (or failed) with aggregated results
        parent.result = {
            "subtasks_total": len(remaining),
            "subtasks_completed": len(remaining) - len(failed),
            "subtasks_failed": len(failed),
        }
        if failed:
            parent.status = TaskStatus.FAILED
            parent.error = f"{len(failed)} of {len(remaining)} subtasks failed"
        else:
            parent.status = TaskStatus.COMPLETED
        parent.completed_at = datetime.utcnow()
        logger.info(
            f"Parent task {parent.name} -> {parent.status.value} ({len(remaining)} subtasks)"
        )
        # Refresh any enclosing project
        if parent.project_id:
            self._associate_task_with_project(parent)
        self._persist()

    def flush(self) -> None:
        """Public hook to persist any in-memory state changes immediately."""
        self._persist()

    def _persist(self) -> None:
        """Write current projects/tasks (with usage) to the SQLite store."""
        if not self._store:
            return
        try:
            self._store.save_projects(list(self._projects.values()))
            self._store.save_tasks(list(self._tasks.values()))
        except Exception as e:
            logger.error(f"Persistence write failed: {e}")

    def _load_from_store(self) -> None:
        """Load projects/tasks saved in SQLite into memory on startup."""
        if not self._store:
            return
        try:
            for p in self._store.load_projects():
                self._projects[p.id] = p
            tasks = self._store.load_tasks()
            # Only restore tasks that were not terminal (active work continues
            # across restart); terminal ones are retained for history/usage.
            for t in tasks:
                if t.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                    t.status = TaskStatus.PENDING  # restart-eligible
                self._tasks[t.id] = t
            logger.info(
                f"Loaded {len(self._projects)} project(s), {len(tasks)} task(s) from {self.config.data_dir}"
            )
        except Exception as e:
            logger.error(f"Persistence read failed: {e}")

    def usage_totals(self) -> Dict[str, Any]:
        """Aggregate token/model usage across all completed tasks.

        Returns totals for input/output/cache/reasoning tokens, total cost,
        and per-model breakdowns.
        """
        total_input = 0
        total_output = 0
        total_cache_read = 0
        total_cache_write = 0
        total_reasoning = 0
        total_cost = 0.0
        total_api_calls = 0
        by_model: Dict[str, int] = {}
        session_count = 0

        for task in self._tasks.values():
            u = task.usage
            if not u:
                continue
            session_count += 1
            total_input += u.input_tokens or 0
            total_output += u.output_tokens or 0
            total_cache_read += u.cache_read_tokens or 0
            total_cache_write += u.cache_write_tokens or 0
            total_reasoning += u.reasoning_tokens or 0
            total_cost += u.cost_usd or 0.0
            total_api_calls += u.api_call_count or 0
            model = u.model or "unknown"
            model_tokens = (
                (u.input_tokens or 0)
                + (u.output_tokens or 0)
                + (u.cache_read_tokens or 0)
                + (u.cache_write_tokens or 0)
            )
            by_model[model] = by_model.get(model, 0) + model_tokens

        total = total_input + total_output + total_cache_read + total_cache_write
        return {
            "total_tokens": total,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cache_read_tokens": total_cache_read,
            "cache_write_tokens": total_cache_write,
            "reasoning_tokens": total_reasoning,
            "total_cost_usd": round(total_cost, 6),
            "api_call_count": total_api_calls,
            "session_count": session_count,
            "by_model": by_model,
        }

    def task_tree(self, task_id: str) -> Dict[str, Any]:
        """Return a task and its subtasks nested (for dashboards)."""
        """Return a task and its subtasks nested (for dashboards)."""
        task = self._tasks.get(task_id)
        if not task:
            return {}
        children = [
            self.task_tree(t.id) for t in self._tasks.values() if t.parent_task_id == task_id
        ]
        return {"task": task, "children": children}

    async def decompose_task(
        self,
        parent_task: Task,
        planner_capability: str = "task_planning",
        max_subtasks: int = 8,
    ) -> List[Task]:
        """Auto-decompose a task by asking a planner agent to split it.

        Submits a one-off planning task requiring ``planner_capability``, whose
        agent returns a JSON subtask plan. The plan is parsed into subtasks that
        inherit the parent's project and are linked via ``parent_task_id``.
        The parent is not marked complete until all subtasks finish.
        """
        # Build a planning prompt
        prompt = (
            f"Decompose this goal into at most {max_subtasks} concrete subtasks.\n"
            f"Goal: {parent_task.description or parent_task.name}\n"
            f"Context: {parent_task.payload}\n\n"
            "Return ONLY a JSON array of objects, each with: "
            "name, description, required_capability, payload. "
            "The required_capability values must match skills available to agents."
        )

        planner = Task(
            name=f"Plan: {parent_task.name}",
            description=prompt,
            required_capability=planner_capability,
            project_id=parent_task.project_id,
            parent_task_id=parent_task.id,
            payload={
                "goal": parent_task.description or parent_task.name,
                "plan_for": parent_task.id,
            },
        )
        # The planner's result will be turned into subtasks when it completes
        planner.metadata["is_planner_for"] = parent_task.id
        self._tasks[planner.id] = planner
        self.orchestration.submit_task(planner)
        logger.info(f"Submitted planner task for {parent_task.name}: {planner.id}")
        # Mark parent as decomposed / waiting on its plan
        parent_task.metadata["decomposing"] = True
        parent_task.metadata["planner_task_id"] = planner.id
        return []

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
