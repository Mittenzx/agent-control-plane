"""
FastAPI Web Server for Agent Control Plane Dashboard.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from ..core.control_plane import ControlPlane
from ..core.interfaces import Task, TaskStatus, Agent, AgentInfo, AgentStatus, AgentCapability, Message, MessageType
from ..config.manager import ControlPlaneConfig, DEFAULT_CONFIG

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
        message = json.dumps({"event": event, "data": data, "timestamp": datetime.utcnow().isoformat()})
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
    }
}


def serialize_agent(info) -> Dict[str, Any]:
    """Serialize agent info for JSON response."""
    return {
        "id": info.id,
        "name": info.name,
        "type": info.type,
        "status": info.status.value if hasattr(info.status, 'value') else str(info.status),
        "capabilities": [c.name for c in info.capabilities] if hasattr(info, 'capabilities') else [],
        "metadata": getattr(info, 'metadata', {}),
    }


def serialize_task(task: Task) -> Dict[str, Any]:
    """Serialize task for JSON response."""
    return {
        "id": task.id,
        "name": task.name,
        "description": task.description,
        "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
        "required_capability": task.required_capability,
        "priority": task.priority,
        "payload": task.payload,
        "result": task.result,
        "error": task.error,
        "dependencies": list(task.dependencies) if hasattr(task, 'dependencies') else [],
        "created_at": task.created_at.isoformat() if hasattr(task, 'created_at') and task.created_at else None,
        "started_at": task.started_at.isoformat() if hasattr(task, 'started_at') and task.started_at else None,
        "completed_at": task.completed_at.isoformat() if hasattr(task, 'completed_at') and task.completed_at else None,
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
        workflows = control_plane.list_workflows()
        dashboard_state["workflows"] = [
            {
                "id": wf.id,
                "name": wf.name,
                "status": wf.status.value if hasattr(wf.status, 'value') else str(wf.status),
                "tasks": [serialize_task(t) for t in wf.tasks],
            }
            for wf in workflows
        ]
        
        # Update metrics
        dashboard_state["metrics"] = {
            "total_tasks": len(tasks),
            "completed_tasks": len([t for t in tasks if t.status == TaskStatus.COMPLETED]),
            "failed_tasks": len([t for t in tasks if t.status == TaskStatus.FAILED]),
            "active_agents": len([a for a in agents if a.status in (AgentStatus.RUNNING, AgentStatus.INITIALIZING, AgentStatus.IDLE)]),
            "total_agents": len(agents),
        }
        
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
    
    # Register example agents so the dashboard shows live data
    try:
        from examples.llm_agents import (
            create_research_agent, create_coder_agent, create_planner_agent
        )
        for factory in (create_research_agent, create_coder_agent, create_planner_agent):
            agent = factory(factory.__name__.replace("create_", "").replace("_agent", ""), control_plane)
            control_plane.registry.register(agent)
            await control_plane.lifecycle.initialize_agent(agent)
            for cap in agent.capabilities:
                control_plane.capability_registry.register_capability(cap, agent.id)
            control_plane.message_bus.register_agent_queue(agent.id, asyncio.Queue())
            control_plane.message_bus.subscribe(agent.id, agent.handle_message)
        logger.info("Registered example agents")
    except Exception as e:
        logger.warning(f"Could not register example agents: {e}")
    
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
import os
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
        payload=task_data.get("payload", {}),
        priority=task_data.get("priority", 5),
        dependencies=task_data.get("dependencies", []),
    )
    
    task_id = control_plane.submit_task(task)
    await update_dashboard_state()
    return JSONResponse({"task_id": task_id, "task": serialize_task(task)})


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


@app.get("/api/workflows")
async def get_workflows():
    """Get all workflows."""
    if not control_plane:
        return JSONResponse({"workflows": []})
    workflows = control_plane.list_workflows()
    return JSONResponse({"workflows": [
        {
            "id": wf.id,
            "name": wf.name,
            "status": wf.status.value if hasattr(wf.status, 'value') else str(wf.status),
            "tasks": [serialize_task(t) for t in wf.tasks],
        }
        for wf in workflows
    ]})


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
    return JSONResponse({"events": dashboard_state["events"][-limit:]})


@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics."""
    return JSONResponse(dashboard_state["metrics"])


# WebSocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await connection_manager.connect(websocket)
    try:
        # Send initial state
        await websocket.send_text(json.dumps({
            "event": "initial_state",
            "data": dashboard_state,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
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