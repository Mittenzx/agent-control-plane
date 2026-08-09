# Agent Control Plane

A modular, customizable control plane for AI agents with support for multi-agent systems, workflow orchestration, and plugin extensibility.

## Features

- **Agent Registry & Lifecycle Management** - Register, initialize, and manage agent lifecycles
- **Task Orchestration** - Submit tasks, create workflows with dependencies, priority-based scheduling
- **Inter-Agent Messaging** - Request/response, pub/sub, event-driven communication
- **Plugin System** - Extend with custom agents, schedulers, messaging transports
- **Configuration Management** - JSON/YAML config with hot-reloading and env var overrides
- **Multi-Agent Workflows** - Coordinate complex multi-step processes across agents

## Architecture

```
control_plane/
├── core/              # Core interfaces and main ControlPlane class
├── agents/            # Agent registry and lifecycle management
├── orchestration/     # Task scheduling and workflow engine
├── messaging/         # Message bus, request/response, event bus
├── plugins/           # Plugin manager, capability registry, agent factory
├── config/            # Configuration management
└── examples/          # Example agents and usage
```

## Quick Start

```python
import asyncio
from control_plane import ControlPlane, DEFAULT_CONFIG, Task

async def main():
    async with ControlPlane(DEFAULT_CONFIG) as cp:
        # Submit a task
        task = Task(
            name="Research AI trends",
            required_capability="research",
            payload={"topic": "AI agents 2024"}
        )
        task_id = cp.submit_task(task)
        
        # Wait for result
        await asyncio.sleep(2)
        result = cp.get_task(task_id)
        print(result.result)

asyncio.run(main())
```

## Configuration

Create a `config.json`:

```json
{
  "name": "my-control-plane",
  "agents": [
    {
      "type": "research_agent",
      "name": "researcher",
      "capabilities": [
        {"name": "research", "description": "Web research"}
      ],
      "config": {"model": "gpt-4", "temperature": 0.3}
    }
  ]
}
```

Load it:

```python
from control_plane import ControlPlane, ControlPlaneConfig

config = ControlPlaneConfig.from_file("config.json")
cp = ControlPlane(config)
```

## Built-in Agent Types

- **ResearchAgent** - Web research, summarization
- **CoderAgent** - Code generation, code review
- **PlannerAgent** - Task planning, workflow design

## Creating Custom Agents

```python
from control_plane import Agent, AgentInfo, AgentCapability, Task

class MyAgent(Agent):
    async def initialize(self):
        # Setup connections, load models
        pass
    
    async def execute_task(self, task: Task) -> dict:
        # Implement task logic
        return {"result": "done"}
    
    async def shutdown(self):
        # Cleanup
        pass

# Register with factory
cp.agent_factory.register_agent_type("my_agent", MyAgent)
```

## Plugin System

```python
from control_plane import AgentPlugin

class MyPlugin(AgentPlugin):
    @property
    def name(self) -> str:
        return "my-plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def on_load(self, control_plane):
        # Register custom agents, schedulers, etc.
        control_plane.agent_factory.register_agent_type("custom", CustomAgent)
    
    async def on_unload(self):
        pass

# Load plugin
await cp.plugin_manager.load_plugin(MyPlugin)
```

## Messaging Patterns

```python
# Request/Response
response = await cp.request(
    sender_id="me",
    recipient_id="agent-id",
    payload={"action": "process", "data": "..."}
)

# Publish/Subscribe
await cp.message_bus.subscribe("agent-id", handler)

# Events
cp.event_bus.subscribe("task.completed", handler)
await cp.event_bus.emit("task.completed", {"task_id": "..."})
```

## Workflows

```python
from control_plane import Task, TaskStatus

tasks = [
    Task(name="Step 1", required_capability="research", payload={}),
    Task(name="Step 2", required_capability="code_generation", 
         payload={}, dependencies=[tasks[0].id]),
    Task(name="Step 3", required_capability="code_review",
         payload={}, dependencies=[tasks[1].id])
]

workflow = cp.submit_workflow("My Workflow", tasks)
```

## Running Examples

```bash
# Install dependencies
pip install -r requirements.txt

# Run full example
python -m examples.full_example

# Run with custom config
python -c "
import asyncio
from control_plane import ControlPlane, ControlPlaneConfig
config = ControlPlaneConfig.from_file('config.json')
asyncio.run(ControlPlane(config).start())
"
```

## Extending the Control Plane

### Custom Scheduler
```python
from control_plane.orchestration import TaskScheduler

class PriorityScheduler(TaskScheduler):
    async def _find_best_agent(self, task):
        # Custom agent selection logic
        pass

cp.orchestration.scheduler = PriorityScheduler(cp.registry, cp.lifecycle)
```

### Custom Message Transport
```python
from control_plane.messaging import MessageBus

class RedisMessageBus(MessageBus):
    async def send(self, message):
        # Send via Redis pub/sub
        pass

cp.message_bus = RedisMessageBus()
```

## License

MIT