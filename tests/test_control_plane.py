"""
Basic tests for the control plane.
"""
import asyncio
import pytest
from unittest.mock import Mock, AsyncMock

from control_plane.core.interfaces import (
    AgentInfo, AgentCapability, Task, TaskStatus, AgentStatus, MessageType, UsageRecord
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

    @pytest.mark.asyncio
    async def test_submit_task_no_capability_agent_fails_fast(self):
        """A task with a capability no agent has should fail fast, not hang."""
        config = ControlPlaneConfig(name="test-cp")
        cp = ControlPlane(config)

        await cp.start()

        # Register a mock agent with only 'test_cap'
        from control_plane.core.interfaces import AgentCapability

        mock_agent = MockAgent("test-agent", "Test", [
            AgentCapability(name="test_cap", description="Test")
        ])
        cp.registry.register(mock_agent)
        await cp.lifecycle.initialize_agent(mock_agent)

        # Submit a task requiring a capability no agent provides
        task = Task(
            name="Impossible Task",
            required_capability="nonexistent_capability"
        )

        task_id = cp.submit_task(task)
        # Give scheduler time to attempt (and fail) the task
        await asyncio.sleep(1.0)

        result_task = cp.get_task(task_id)
        assert result_task.status == TaskStatus.FAILED
        assert result_task.error is not None
        assert "nonexistent_capability" in result_task.error

        await cp.stop()


class TestHermesAgent:
    """Tests for the Hermes-backed agent adapter."""

    def test_hermes_agent_type_registered(self):
        """ControlPlane factory should register the hermes_agent type."""
        cp = ControlPlane()
        types = cp.agent_factory.list_agent_types()
        assert "hermes_agent" in types

    def test_parse_session_id(self):
        """Session ID should be parsed from Hermes quiet-mode stderr."""
        from control_plane.agents.hermes_agent import HermesAgent

        assert HermesAgent._parse_session_id(
            "some output\nsession_id: 20260812_123456_abc123\nmore"
        ) == "20260812_123456_abc123"
        assert HermesAgent._parse_session_id("no session here") is None

    @pytest.mark.asyncio
    async def test_execute_task_captures_result(self, mocker):
        """execute_task should return the Hermes stdout + session_id."""
        from control_plane.agents.hermes_agent import HermesAgent

        info = AgentInfo(
            id="hermes-1",
            name="hermes-agent",
            type="hermes_agent",
            capabilities=[AgentCapability(name="reasoning", description="r")],
        )
        agent = HermesAgent(info)

        # Mock the subprocess creation
        mock_proc = Mock()
        mock_proc.communicate = AsyncMock(
            return_value=(b"RESULT_TEXT", b"\nsession_id: 20260812_9999_abc\n")
        )
        mock_proc.returncode = 0

        mocker.patch.object(
            agent, "hermes_command", "hermes"
        )
        mocker.patch(
            "asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)
        )

        result = await agent.execute_task(
            Task(name="T", required_capability="reasoning")
        )
        assert result["output"] == "RESULT_TEXT"
        assert result["session_id"] == "20260812_9999_abc"
        assert result["exit_code"] == 0


class TestProjects:
    """Tests for project + task association and progress."""

    @pytest.mark.asyncio
    async def test_create_and_list_project(self):
        """Projects can be created and listed."""
        cp = ControlPlane()
        await cp.start()
        p = cp.create_project("Website", goal="Ship a marketing site")
        assert p.name == "Website"
        assert p.id in [x.id for x in cp.list_projects()]
        await cp.stop()

    @pytest.mark.asyncio
    async def test_task_associates_with_project(self):
        """A task with project_id is linked and project state updates."""
        cp = ControlPlane()
        await cp.start()
        p = cp.create_project("API", goal="Build a REST API")
        task = Task(name="Write endpoints", required_capability="code",
                    project_id=p.id)
        cp.submit_task(task)
        assert task.id in p.task_ids
        assert cp.get_project_tasks(p.id) == [task]
        await cp.stop()

    @pytest.mark.asyncio
    async def test_project_progress_computes_percent(self):
        """Progress percent reflects completed task ratio."""
        cp = ControlPlane()
        await cp.start()
        p = cp.create_project("Docs", goal="Document the system")
        t1 = Task(name="T1", required_capability="write", project_id=p.id)
        t2 = Task(name="T2", required_capability="write", project_id=p.id)
        cp.submit_task(t1)
        cp.submit_task(t2)
        # Mark one complete
        t1.status = TaskStatus.COMPLETED
        progress = cp.project_progress(p.id)
        assert progress["total"] == 2
        assert progress["completed"] == 1
        assert progress["progress_pct"] == 50
        await cp.stop()


class TestSubtaskDecomposition:
    """Tests for task decomposition and subtask aggregation."""

    @pytest.mark.asyncio
    async def test_spawn_subtasks_links_parent_and_project(self):
        """spawn_subtasks links parent_task_id and inherits project_id."""
        cp = ControlPlane()
        await cp.start()
        p = cp.create_project("Site", goal="Ship a site")
        parent = Task(name="Build site", required_capability="code",
                      project_id=p.id)
        cp.submit_task(parent)
        subs = cp.spawn_subtasks(parent, [
            {"name": "Design", "required_capability": "design"},
            {"name": "Build", "required_capability": "build"},
        ])
        assert len(subs) == 2
        assert all(s.parent_task_id == parent.id for s in subs)
        assert all(s.project_id == p.id for s in subs)
        assert parent.metadata.get("has_subtasks") is True
        assert len(cp._tasks) == 3
        await cp.stop()

    def test_extract_spawn_from_output_parses_json_array(self):
        """Decompose output with a JSON array is parsed into subtask specs."""
        cp = ControlPlane()
        output = ('Here is my plan:\n'
                  '[{"name": "T1", "required_capability": "c1"}, '
                  '{"name": "T2", "required_capability": "c2"}]')
        specs = cp._extract_spawn_from_output(output)
        assert specs is not None
        assert len(specs) == 2
        assert specs[0]["name"] == "T1"

    def test_extract_spawn_handles_wrapped_object(self):
        """Decompose output with a 'spawn' object is parsed."""
        cp = ControlPlane()
        output = '{"spawn": [{"name": "A"}, {"name": "B"}]}'
        specs = cp._extract_spawn_from_output(output)
        assert specs == [{"name": "A"}, {"name": "B"}]

    @pytest.mark.asyncio
    async def test_parent_completes_when_all_subtasks_done(self):
        """Parent task completes once every subtask is in a terminal state."""
        cp = ControlPlane()
        await cp.start()
        parent = Task(name="Parent", required_capability="x")
        cp.submit_task(parent)
        subs = cp.spawn_subtasks(parent, [
            {"name": "S1", "required_capability": "x"},
            {"name": "S2", "required_capability": "x"},
        ])
        # Simulate both subtasks completing
        for s in subs:
            s.status = TaskStatus.COMPLETED
        cp._on_task_completed(subs[0], {"ok": True})  # triggers aggregation
        assert parent.status == TaskStatus.COMPLETED
        assert parent.result["subtasks_completed"] == 2
        await cp.stop()

    @pytest.mark.asyncio
    async def test_parent_fails_when_subtask_fails(self):
        """Parent task fails if any subtask fails."""
        cp = ControlPlane()
        await cp.start()
        parent = Task(name="Parent", required_capability="x")
        cp.submit_task(parent)
        subs = cp.spawn_subtasks(parent, [
            {"name": "S1", "required_capability": "x"},
            {"name": "S2", "required_capability": "x"},
        ])
        subs[0].status = TaskStatus.COMPLETED
        subs[1].status = TaskStatus.FAILED
        cp._on_task_completed(subs[1], None)
        assert parent.status == TaskStatus.FAILED
        assert "1 of 2 subtasks failed" in parent.error
        await cp.stop()

    @pytest.mark.asyncio
    async def test_dependency_resolution_waits(self):
        """A task with an unmet dependency stays pending."""
        cp = ControlPlane()
        await cp.start()
        dep = Task(name="Dep", required_capability="x")
        task = Task(name="Depends", required_capability="x",
                    dependencies=[dep.id])
        cp.submit_task(dep)
        # Before dep completes, task store knows dep but it is PENDING
        assert cp.orchestration.scheduler._dependencies_met(task) is False
        # After dep completes, dependencies are met
        dep.status = TaskStatus.COMPLETED
        assert cp.orchestration.scheduler._dependencies_met(task) is True
        await cp.stop()


class TestUsageTracking:
    """Tests for OpenRouter token/model/cost usage tracking."""

    @pytest.mark.asyncio
    async def test_usage_totals_aggregates_across_tasks(self):
        """usage_totals sums tokens/cost across tasks with usage records."""
        cp = ControlPlane()
        await cp.start()
        t1 = Task(name="T1", required_capability="x")
        t2 = Task(name="T2", required_capability="x")
        t1.usage = UsageRecord(
            model="deepseek/deepseek-v4", provider="openrouter",
            input_tokens=100, output_tokens=50, cache_read_tokens=1000,
            estimated_cost_usd=0.01, api_call_count=2,
        )
        t2.usage = UsageRecord(
            model="deepseek/deepseek-v4", provider="openrouter",
            input_tokens=200, output_tokens=100, cache_read_tokens=500,
            estimated_cost_usd=0.02, api_call_count=1,
        )
        cp._tasks[t1.id] = t1
        cp._tasks[t2.id] = t2
        totals = cp.usage_totals()
        assert totals["input_tokens"] == 300
        assert totals["output_tokens"] == 150
        assert totals["cache_read_tokens"] == 1500
        assert totals["total_cost_usd"] == 0.03
        assert totals["api_call_count"] == 3
        assert totals["session_count"] == 2
        assert totals["by_model"]["deepseek/deepseek-v4"] == 1950
        await cp.stop()

    def test_usage_record_properties(self):
        """UsageRecord exposes prompt/completion/cost aliases."""
        u = UsageRecord(
            model="m", input_tokens=10, output_tokens=20,
            estimated_cost_usd=0.5, actual_cost_usd=0.4,
        )
        assert u.prompt_tokens == 10
        assert u.completion_tokens == 20
        assert u.cost_usd == 0.4  # actual wins over estimated
        u2 = UsageRecord(model="m", input_tokens=1, estimated_cost_usd=0.5)
        assert u2.cost_usd == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])