"""
Basic tests for the control plane.
"""
import asyncio
import pytest
from unittest.mock import Mock, AsyncMock

from control_plane.core.interfaces import (
    AgentInfo, AgentCapability, Task, TaskStatus, AgentStatus, MessageType
)
from control_plane.agents.registry import AgentRegistry, AgentLifecycleManager
from control_plane.orchestration.engine import TaskScheduler, WorkflowEngine, OrchestrationEngine
from control_plane.messaging.bus import MessageBus, RequestResponseBus, EventBus
from control_plane.plugins.manager import PluginManager, AgentFactory, CapabilityRegistry
from control_plane.config.manager import ConfigManager, ControlPlaneConfig, AgentConfig
from control_plane.core.control_plane import ControlPlane


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, agent_id: str, name: str, capabilities: list = None):
        self.info = AgentInfo(
            id=agent_id,
            name=name,
            type="mock",
            capabilities=capabilities or []
        )
        self._initialized = False
        self._shutdown = False
    
    @property
    def id(self):
        return self.info.id
    
    @property
    def name(self):
        return self.info.name
    
    @property
    def capabilities(self):
        return self.info.capabilities
    
    @property
    def status(self):
        return self.info.status
    
    @status.setter
    def status(self, value):
        self.info.status = value
    
    async def initialize(self):
        self._initialized = True
        self.info.status = AgentStatus.IDLE
    
    async def execute_task(self, task: Task):
        return {"result": f"completed {task.name}"}
    
    async def shutdown(self):
        self._shutdown = True
        self.info.status = AgentStatus.STOPPED
    
    def can_handle_task(self, task: Task):
        return any(cap.name == task.required_capability for cap in self.capabilities)
    
    async def handle_message(self, message):
        return None


@pytest.fixture
def mock_agent():
    cap = AgentCapability(name="test_cap", description="Test capability")
    return MockAgent("agent-1", "TestAgent", [cap])


@pytest.fixture
def registry():
    return AgentRegistry()


@pytest.fixture
def lifecycle(registry):
    return AgentLifecycleManager(registry)


class TestAgentRegistry:
    """Tests for AgentRegistry."""
    
    def test_register_agent(self, registry, mock_agent):
        registry.register(mock_agent)
        assert registry.get("agent-1") == mock_agent
        assert len(registry.list_agents()) == 1
    
    def test_unregister_agent(self, registry, mock_agent):
        registry.register(mock_agent)
        unregistered = registry.unregister("agent-1")
        assert unregistered == mock_agent
        assert registry.get("agent-1") is None
    
    def test_find_by_capability(self, registry, mock_agent):
        registry.register(mock_agent)
        agents = registry.find_by_capability("test_cap")
        assert len(agents) == 1
        assert agents[0] == mock_agent
    
    def test_find_available_by_capability(self, registry, mock_agent):
        registry.register(mock_agent)
        mock_agent.info.status = AgentStatus.IDLE
        agents = registry.find_available_by_capability("test_cap")
        assert len(agents) == 1
        
        mock_agent.info.status = AgentStatus.RUNNING
        agents = registry.find_available_by_capability("test_cap")
        assert len(agents) == 0


class TestAgentLifecycleManager:
    """Tests for AgentLifecycleManager."""
    
    @pytest.mark.asyncio
    async def test_initialize_agent(self, lifecycle, mock_agent):
        await lifecycle.initialize_agent(mock_agent)
        assert mock_agent._initialized
        assert mock_agent.status == AgentStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_execute_task(self, lifecycle, registry, mock_agent):
        registry.register(mock_agent)
        await lifecycle.initialize_agent(mock_agent)
        
        task = Task(
            name="Test Task",
            required_capability="test_cap",
            payload={"data": "test"}
        )
        
        result = await lifecycle.execute_task(mock_agent, task)
        assert result["result"] == "completed Test Task"
        assert task.status == TaskStatus.COMPLETED
        assert mock_agent.status == AgentStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_shutdown_agent(self, lifecycle, registry, mock_agent):
        registry.register(mock_agent)
        await lifecycle.initialize_agent(mock_agent)
        
        await lifecycle.shutdown_agent(mock_agent)
        assert mock_agent._shutdown
        assert mock_agent.status == AgentStatus.STOPPED


class TestTaskScheduler:
    """Tests for TaskScheduler."""
    
    @pytest.mark.asyncio
    async def test_submit_and_schedule(self, registry, lifecycle, mock_agent):
        registry.register(mock_agent)
        await lifecycle.initialize_agent(mock_agent)
        
        scheduler = TaskScheduler(registry, lifecycle)
        await scheduler.start()
        
        task = Task(
            name="Scheduled Task",
            required_capability="test_cap",
            priority=5
        )
        
        scheduler.submit_task(task)
        await asyncio.sleep(0.5)
        
        assert task.status in (TaskStatus.COMPLETED, TaskStatus.RUNNING, TaskStatus.QUEUED)
        
        await scheduler.stop()


class TestMessageBus:
    """Tests for MessageBus."""
    
    @pytest.mark.asyncio
    async def test_send_receive(self):
        bus = MessageBus()
        received = []
        
        def handler(msg):
            received.append(msg)
        
        bus.subscribe("agent-1", handler)
        
        from control_plane.core.interfaces import Message
        msg = Message(
            sender_id="sender",
            recipient_id="agent-1",
            payload={"data": "test"}
        )
        
        await bus.send(msg)
        await asyncio.sleep(0.1)
        
        assert len(received) == 1
        assert received[0].payload["data"] == "test"
    
    @pytest.mark.asyncio
    async def test_broadcast(self):
        bus = MessageBus()
        received = []
        
        def handler(msg):
            received.append(msg)
        
        bus.subscribe_broadcast(handler)
        
        from control_plane.core.interfaces import Message
        msg = Message(
            sender_id="sender",
            recipient_id=None,  # Broadcast
            payload={"broadcast": True}
        )
        
        await bus.send(msg)
        await asyncio.sleep(0.1)
        
        assert len(received) == 1
        assert received[0].payload["broadcast"] is True


class TestRequestResponseBus:
    """Tests for RequestResponseBus."""
    
    @pytest.mark.asyncio
    async def test_request_response(self):
        bus = RequestResponseBus()
        
        # Set up responder
        async def responder(msg):
            await bus.respond(msg, {"response": "ok"})
        
        bus.subscribe("responder", responder)
        
        # Make request
        response = await bus.request(
            sender_id="requester",
            recipient_id="responder",
            payload={"request": "data"}
        )
        
        assert response.payload["response"] == "ok"


class TestEventBus:
    """Tests for EventBus."""
    
    @pytest.mark.asyncio
    async def test_emit_subscribe(self):
        bus = EventBus()
        events = []
        
        def handler(data):
            events.append(data)
        
        bus.subscribe("test.event", handler)
        await bus.emit("test.event", {"key": "value"})
        
        assert len(events) == 1
        assert events[0]["key"] == "value"
    
    @pytest.mark.asyncio
    async def test_wildcard_subscribe(self):
        bus = EventBus()
        events = []
        
        def handler(event_type, data):
            events.append((event_type, data))
        
        bus.subscribe_all(handler)
        await bus.emit("event.a", {"a": 1})
        await bus.emit("event.b", {"b": 2})
        
        assert len(events) == 2


class TestConfigManager:
    """Tests for ConfigManager."""
    
    def test_default_config(self):
        config = ControlPlaneConfig()
        assert config.name == "agent-control-plane"
        assert config.max_concurrent_tasks == 10
    
    def test_agent_config(self):
        agent_config = AgentConfig(
            type="test",
            name="test-agent",
            capabilities=[{"name": "cap1"}]
        )
        assert agent_config.type == "test"
        assert agent_config.enabled is True
    
    def test_config_serialization(self):
        config = ControlPlaneConfig(
            name="test",
            agents=[AgentConfig(type="t", name="a", capabilities=[])]
        )
        data = config.to_dict()
        assert data["name"] == "test"
        assert len(data["agents"]) == 1
        
        restored = ControlPlaneConfig.from_dict(data)
        assert restored.name == "test"


class TestCapabilityRegistry:
    """Tests for CapabilityRegistry."""
    
    def test_register_capability(self):
        reg = CapabilityRegistry()
        cap = AgentCapability(name="cap1", description="Test")
        
        reg.register_capability(cap, "agent-1")
        
        assert reg.get_capability("cap1") == cap
        assert reg.find_providers("cap1") == ["agent-1"]
    
    def test_unregister_capability(self):
        reg = CapabilityRegistry()
        cap = AgentCapability(name="cap1", description="Test")
        
        reg.register_capability(cap, "agent-1")
        reg.unregister_capability("cap1", "agent-1")
        
        assert reg.get_capability("cap1") is None
        assert reg.find_providers("cap1") == []


class TestAgentFactory:
    """Tests for AgentFactory."""
    
    def test_register_and_create(self):
        from control_plane.core.interfaces import ControlPlane
        
        cp = Mock(spec=ControlPlane)
        factory = AgentFactory(cp)
        
        class TestAgent:
            def __init__(self, agent_info, config=None, control_plane=None):
                self.info = agent_info
                self.config = config
                self.control_plane = control_plane
        
        factory.register_agent_type("test", TestAgent)
        
        import asyncio
        async def create():
            info = AgentInfo(id="1", name="Test", type="test", capabilities=[])
            agent = await factory.create_agent("test", info, {"key": "value"})
            return agent
        
        agent = asyncio.run(create())
        assert isinstance(agent, TestAgent)
        assert agent.config["key"] == "value"


class TestControlPlane:
    """Integration tests for ControlPlane."""
    
    @pytest.mark.asyncio
    async def test_start_stop(self):
        config = ControlPlaneConfig(name="test-cp")
        cp = ControlPlane(config)
        
        await cp.start()
        assert cp.running
        assert cp.uptime is not None
        
        await cp.stop()
        assert not cp.running
    
    @pytest.mark.asyncio
    async def test_submit_task(self):
        config = ControlPlaneConfig(name="test-cp")
        cp = ControlPlane(config)
        
        await cp.start()
        
        # Register a mock agent manually
        from control_plane.core.interfaces import AgentCapability
        mock_agent = MockAgent("test-agent", "Test", [
            AgentCapability(name="test_cap", description="Test")
        ])
        cp.registry.register(mock_agent)
        await cp.lifecycle.initialize_agent(mock_agent)
        
        task = Task(
            name="Integration Task",
            required_capability="test_cap"
        )
        
        task_id = cp.submit_task(task)
        await asyncio.sleep(0.5)
        
        result_task = cp.get_task(task_id)
        assert result_task.status == TaskStatus.COMPLETED
        
        await cp.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])