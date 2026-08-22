"""
FastAPI Web Server for Agent Control Plane Dashboard.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware

from ..core.control_plane import ControlPlane
from ..core.interfaces import (
    Task,
    TaskStatus,
    AgentStatus,
)
from ..config.manager import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, event: str, data: Dict[str, Any]):
        """Broadcast event to all connected clients."""
        message = json.dumps(
            {"event": event, "data": data, "timestamp": datetime.utcnow().isoformat()}
        )
        async with self._lock:
            disconnected = set()
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except Exception:
                    disconnected.add(connection)

            for conn in disconnected:
                self.active_connections.discard(conn)


# Global instances
control_plane: Optional[ControlPlane] = None
connection_manager = ConnectionManager()
dashboard_state = {
    "agents": [],
    "tasks": [],
    "workflows": [],
    "events": [],
    "metrics": {
        "total_tasks": 0,
        "completed_tasks": 0,
        "failed_tasks": 0,
        "active_agents": 0,
        "total_agents": 0,
    },
}


def serialize_agent(info) -> Dict[str, Any]:
    """Serialize agent info for JSON response."""
    return {
        "id": info.id,
        "name": info.name,
        "type": info.type,
        "status": info.status.value if hasattr(info.status, "value") else str(info.status),
        "capabilities": [c.name for c in info.capabilities]
        if hasattr(info, "capabilities")
        else [],
        "metadata": getattr(info, "metadata", {}),
    }


def serialize_usage(usage) -> Optional[Dict[str, Any]]:
    """Serialize a UsageRecord (or None) for JSON responses."""
    if not usage:
        return None
    return {
        "model": getattr(usage, "model", ""),
        "provider": getattr(usage, "provider", ""),
        "session_id": getattr(usage, "session_id", ""),
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
        "cache_read_tokens": getattr(usage, "cache_read_tokens", 0),
        "cache_write_tokens": getattr(usage, "cache_write_tokens", 0),
        "reasoning_tokens": getattr(usage, "reasoning_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
        "estimated_cost_usd": getattr(usage, "estimated_cost_usd", 0.0),
        "actual_cost_usd": getattr(usage, "actual_cost_usd", None),
        "cost_usd": getattr(usage, "cost_usd", 0.0),
        "api_call_count": getattr(usage, "api_call_count", 0),
    }


def serialize_task(task: Task) -> Dict[str, Any]:
    """Serialize task for JSON response."""
    return {
        "id": task.id,
        "name": task.name,
        "description": task.description,
        "status": task.status.value if hasattr(task.status, "value") else str(task.status),
        "required_capability": task.required_capability,
        "project_id": getattr(task, "project_id", None),
        "parent_task_id": getattr(task, "parent_task_id", None),
        "priority": task.priority,
        "payload": task.payload,
        "result": task.result,
        "error": task.error,
        "dependencies": list(task.dependencies) if hasattr(task, "dependencies") else [],
        "created_at": task.created_at.isoformat()
        if hasattr(task, "created_at") and task.created_at
        else None,
        "started_at": task.started_at.isoformat()
        if hasattr(task, "started_at") and task.started_at
        else None,
        "completed_at": task.completed_at.isoformat()
        if hasattr(task, "completed_at") and task.completed_at
        else None,
        "usage": serialize_usage(getattr(task, "usage", None)),
    }


def serialize_project(project) -> Dict[str, Any]:
    """Serialize a project with computed progress."""
    if not control_plane:
        return {"id": project.id, "name": project.name}
    progress = control_plane.project_progress(project.id)
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "goal": project.goal,
        "status": project.status.value if hasattr(project.status, "value") else str(project.status),
        "task_ids": list(project.task_ids),
        "budget_usd": getattr(project, "budget_usd", None),
        "created_at": project.created_at.isoformat()
        if hasattr(project, "created_at") and project.created_at
        else None,
        "progress": progress,
    }


async def update_dashboard_state():
    """Update dashboard state from control plane."""
    global dashboard_state
    if not control_plane:
        return

    try:
        # Get agents
        agents = control_plane.list_agents()
        dashboard_state["agents"] = [serialize_agent(a) for a in agents]

        # Get tasks
        tasks = list(control_plane._tasks.values())
        dashboard_state["tasks"] = [serialize_task(t) for t in tasks]

        # Get workflows
        # Get workflows
        workflows = control_plane.list_workflows()
        dashboard_state["workflows"] = [
            {
                "id": wf.id,
                "name": wf.name,
                "status": wf.status.value if hasattr(wf.status, "value") else str(wf.status),
                "tasks": [serialize_task(t) for t in wf.tasks],
            }
            for wf in workflows
        ]

        # Get projects (with computed progress)
        projects = control_plane.list_projects()
        dashboard_state["projects"] = [serialize_project(p) for p in projects]

        # Update usage totals
        dashboard_state["usage"] = control_plane.usage_totals()

        # Update metrics
        dashboard_state["metrics"] = {
            "total_tasks": len(tasks),
            "completed_tasks": len([t for t in tasks if t.status == TaskStatus.COMPLETED]),
            "failed_tasks": len([t for t in tasks if t.status == TaskStatus.FAILED]),
            "active_agents": len(
                [
                    a
                    for a in agents
                    if a.status in (AgentStatus.RUNNING, AgentStatus.INITIALIZING, AgentStatus.IDLE)
                ]
            ),
            "total_agents": len(agents),
            "total_projects": len(projects),
        }

        # Cost/usage alerts (budget thresholds)
        dashboard_state["alerts"] = control_plane.cost_alerts()

        # Broadcast update
        await connection_manager.broadcast("state_update", dashboard_state)
    except Exception as e:
        logger.error(f"Error updating dashboard state: {e}")


async def state_updater():
    """Background task to periodically update dashboard state."""
    while True:
        await asyncio.sleep(1)
        await update_dashboard_state()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global control_plane

    # Start control plane
    config = DEFAULT_CONFIG
    config.log_level = "INFO"
    control_plane = ControlPlane(config)
    await control_plane.start()

    # Start background state updater
    updater_task = asyncio.create_task(state_updater())

    logger.info("Dashboard server started")

    yield

    # Cleanup
    updater_task.cancel()
    try:
        await updater_task
    except asyncio.CancelledError:
        pass

    await control_plane.stop()
    logger.info("Dashboard server stopped")


app = FastAPI(
    title="Agent Control Plane Dashboard",
    description="Real-time monitoring and control for AI agent orchestration",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=_TEMPLATE_DIR)


# API Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse(request, "dashboard.html", {})


@app.get("/api/state")
async def get_state():
    """Get current dashboard state."""
    return JSONResponse(dashboard_state)


@app.get("/api/agents")
async def get_agents():
    """Get all agents."""
    if not control_plane:
        return JSONResponse({"agents": []})
    agents = control_plane.list_agents()
    return JSONResponse({"agents": [serialize_agent(a) for a in agents]})


@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get specific agent details."""
    if not control_plane:
        return JSONResponse({"error": "Control plane not running"}, status_code=503)
    agent = control_plane.get_agent(agent_id)
    if not agent:
        return JSONResponse({"error": "Agent not found"}, status_code=404)
    return JSONResponse(serialize_agent(agent))


@app.post("/api/agents/{agent_id}/shutdown")
async def shutdown_agent(agent_id: str):
    """Shutdown an agent."""
    if not control_plane:
        return JSONResponse({"error": "Control plane not running"}, status_code=503)
    agent = control_plane.get_agent(agent_id)
    if not agent:
        return JSONResponse({"error": "Agent not found"}, status_code=404)
    await control_plane.lifecycle.shutdown_agent(agent)
    await update_dashboard_state()
    return JSONResponse({"success": True})


@app.get("/api/tasks")
async def get_tasks():
    """Get all tasks."""
    if not control_plane:
        return JSONResponse({"tasks": []})
    tasks = list(control_plane._tasks.values())
    return JSONResponse({"tasks": [serialize_task(t) for t in tasks]})


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    """Get specific task details."""
    if not control_plane:
        return JSONResponse({"error": "Control plane not running"}, status_code=503)
    task = control_plane.get_task(task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse(serialize_task(task))


@app.post("/api/tasks")
async def create_task(task_data: Dict[str, Any]):
    """Create a new task."""
    if not control_plane:
        return JSONResponse({"error": "Control plane not running"}, status_code=503)

    task = Task(
        name=task_data.get("name", "Untitled Task"),
        description=task_data.get("description", ""),
        required_capability=task_data.get("required_capability", ""),
        project_id=task_data.get("project_id"),
        payload=task_data.get("payload", {}),
        priority=task_data.get("priority", 5),
        dependencies=task_data.get("dependencies", []),
    )

    task_id = control_plane.submit_task(task)
    await update_dashboard_state()
    return JSONResponse({"task_id": task_id, "task": serialize_task(task)})


# ----- Project endpoints -----
@app.get("/api/projects")
async def get_projects():
    """List all projects with computed progress."""
    if not control_plane:
        return JSONResponse({"projects": []})
    return JSONResponse({"projects": [serialize_project(p) for p in control_plane.list_projects()]})


@app.post("/api/projects")
async def create_project(project_data: Dict[str, Any]):
    """Create a new project."""
    if not control_plane:
        return JSONResponse({"error": "Control plane not running"}, status_code=503)
    project = control_plane.create_project(
        name=project_data.get("name", "Untitled Project"),
        description=project_data.get("description", ""),
        goal=project_data.get("goal", ""),
    )
    budget = project_data.get("budget_usd")
    if budget is not None:
        project.budget_usd = float(budget)
        control_plane.flush()
    await update_dashboard_state()
    return JSONResponse({"project_id": project.id, "project": serialize_project(project)})


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Get a project with its progress and tasks."""
    if not control_plane:
        return JSONResponse({"error": "Control plane not running"}, status_code=503)
    project = control_plane.get_project(project_id)
    if not project:
        return JSONResponse({"error": "Project not found"}, status_code=404)
    tasks = [serialize_task(t) for t in control_plane.get_project_tasks(project_id)]
    return JSONResponse({"project": serialize_project(project), "tasks": tasks})


@app.post("/api/projects/{project_id}/budget")
async def set_project_budget(project_id: str, body: Optional[Dict[str, Any]] = None):
    """Set (or clear) a project's spend budget for cost alerts."""
    if not control_plane:
        return JSONResponse({"error": "Control plane not running"}, status_code=503)
    body = body or {}
    try:
        control_plane.set_project_budget(project_id, body.get("budget_usd"))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    await update_dashboard_state()
    return JSONResponse({"success": True, "project_id": project_id})


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a task."""
    if not control_plane:
        return JSONResponse({"error": "Control plane not running"}, status_code=503)
    task = control_plane.get_task(task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    task.status = TaskStatus.CANCELLED
    await update_dashboard_state()
    return JSONResponse({"success": True})


@app.post("/api/tasks/{task_id}/decompose")
async def decompose_task(task_id: str, body: Optional[Dict[str, Any]] = None):
    """Decompose a task into subtasks (opt-in via a planner agent)."""
    if not control_plane:
        return JSONResponse({"error": "Control plane not running"}, status_code=503)
    task = control_plane.get_task(task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    body = body or {}
    planner_cap = body.get("planner_capability", "task_planning")
    await control_plane.decompose_task(task, planner_capability=planner_cap)
    await update_dashboard_state()
    return JSONResponse({"success": True, "task_id": task_id})


@app.get("/api/tasks/{task_id}/tree")
async def get_task_tree(task_id: str):
    """Return a task and its nested subtasks."""
    if not control_plane:
        return JSONResponse({"error": "Control plane not running"}, status_code=503)
    tree = control_plane.task_tree(task_id)
    if not tree:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    def _ser(node):
        return {
            "task": serialize_task(node["task"]),
            "children": [_ser(c) for c in node["children"]],
        }

    return JSONResponse(_ser(tree))


@app.get("/api/workflows")
async def get_workflows():
    """Get all workflows."""
    if not control_plane:
        return JSONResponse({"workflows": []})
    workflows = control_plane.list_workflows()
    return JSONResponse(
        {
            "workflows": [
                {
                    "id": wf.id,
                    "name": wf.name,
                    "status": wf.status.value if hasattr(wf.status, "value") else str(wf.status),
                    "tasks": [serialize_task(t) for t in wf.tasks],
                }
                for wf in workflows
            ]
        }
    )


@app.post("/api/workflows")
async def create_workflow(workflow_data: Dict[str, Any]):
    """Create a new workflow."""
    if not control_plane:
        return JSONResponse({"error": "Control plane not running"}, status_code=503)

    tasks = []
    for t_data in workflow_data.get("tasks", []):
        task = Task(
            name=t_data.get("name", "Untitled Task"),
            description=t_data.get("description", ""),
            required_capability=t_data.get("required_capability", ""),
            payload=t_data.get("payload", {}),
            priority=t_data.get("priority", 5),
            dependencies=t_data.get("dependencies", []),
        )
        tasks.append(task)

    workflow = control_plane.submit_workflow(workflow_data.get("name", "Untitled Workflow"), tasks)
    await update_dashboard_state()
    return JSONResponse({"workflow_id": workflow.id, "name": workflow.name})


@app.get("/api/events")
async def get_events(limit: int = 100):
    """Get recent events."""
    return JSONResponse({"events": dashboard_state["events"][-limit:]})  # type: ignore[index]


@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics."""
    return JSONResponse(dashboard_state["metrics"])


@app.get("/api/usage")
async def get_usage():
    """Get aggregated token/model/cost usage across all tasks."""
    if not control_plane:
        return JSONResponse({"error": "Control plane not running"}, status_code=503)
    return JSONResponse(control_plane.usage_totals())


@app.get("/api/capabilities")
async def get_capabilities():
    """Get all capabilities available across registered agents."""
    if not control_plane:
        return JSONResponse({"capabilities": []})
    return JSONResponse({"capabilities": control_plane.registry.list_capabilities()})


@app.post("/api/hermes/agents")
async def spawn_hermes_agent(agent_data: Dict[str, Any]):
    """Spawn a new Hermes-backed agent."""
    if not control_plane:
        return JSONResponse({"error": "Control plane not running"}, status_code=503)

    from ..core.interfaces import AgentCapability, AgentInfo

    name = agent_data.get("name", "hermes-agent")
    caps = agent_data.get("capabilities", ["reasoning"])
    agent_id = agent_data.get("id")

    caps_list = [AgentCapability(name=c, description=f"Hermes capability: {c}") for c in caps]
    info = AgentInfo(
        id=agent_id or f"hermes-{name}-{len(control_plane.list_agents())}",
        name=name,
        type="hermes_agent",
        capabilities=caps_list,
        metadata={"backend": "hermes"},
    )

    agent = await control_plane.agent_factory.create_agent(
        "hermes_agent", info, agent_data.get("config", {})
    )
    control_plane.registry.register(agent)
    await control_plane.lifecycle.initialize_agent(agent)
    for cap in caps_list:
        control_plane.capability_registry.register_capability(cap, agent.id)
    control_plane.message_bus.register_agent_queue(agent.id, asyncio.Queue())
    control_plane.message_bus.subscribe(agent.id, agent.handle_message)

    await update_dashboard_state()
    return JSONResponse({"agent_id": agent.id, "name": agent.name, "type": "hermes_agent"})


@app.get("/api/hermes/sessions")
async def list_hermes_sessions(limit: int = 20):
    """List recent Hermes sessions from the Hermes session store."""
    import subprocess

    try:
        proc = subprocess.run(
            ["hermes", "sessions", "list", "--limit", str(limit)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return JSONResponse({"sessions": proc.stdout, "exit_code": proc.returncode})
    except FileNotFoundError:
        return JSONResponse({"sessions": "", "error": "hermes CLI not found"})
    except Exception as e:
        return JSONResponse({"sessions": "", "error": str(e)})


@app.get("/api/agents/{agent_id}/session")
async def get_agent_session(agent_id: str):
    """Get the Hermes session ID + last output for a Hermes-backed agent."""
    if not control_plane:
        return JSONResponse({"error": "Control plane not running"}, status_code=503)
    agent = control_plane.registry.get(agent_id)
    if not agent:
        return JSONResponse({"error": "Agent not found"}, status_code=404)
    session_id = getattr(agent, "session_id", None)
    return JSONResponse(
        {
            "agent_id": agent_id,
            "name": agent.name,
            "session_id": session_id,
            "stdout": getattr(agent, "get_stdout", lambda: "")() or "",
        }
    )


# WebSocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await connection_manager.connect(websocket)
    try:
        # Send initial state
        await websocket.send_text(
            json.dumps(
                {
                    "event": "initial_state",
                    "data": dashboard_state,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        )

        while True:
            data = await websocket.receive_text()
            # Handle client messages if needed
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await connection_manager.disconnect(websocket)


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the dashboard server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
