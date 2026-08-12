"""
Example demonstrating the full control plane with multiple agents and workflows.
"""

import asyncio
import logging

from control_plane import (
    ControlPlane,
    Task,
    TaskStatus,
    DEFAULT_CONFIG,
)
from examples.llm_agents import (
    create_research_agent,
    create_coder_agent,
    create_planner_agent,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def register_agents(cp):
    """Helper to create and register standard agents."""
    researcher = create_research_agent("researcher", cp)
    coder = create_coder_agent("coder", cp)
    planner = create_planner_agent("planner", cp)

    for agent in [researcher, coder, planner]:
        cp.registry.register(agent)
        await cp.lifecycle.initialize_agent(agent)
        for cap in agent.capabilities:
            cp.capability_registry.register_capability(cap, agent.id)
        cp.message_bus.register_agent_queue(agent.id, asyncio.Queue())
        cp.message_bus.subscribe(agent.id, agent.handle_message)

    return researcher, coder, planner


async def run_basic_example():
    """Run a basic example with the control plane."""
    logger.info("=== Starting Basic Control Plane Example ===")

    config = DEFAULT_CONFIG
    config.log_level = "INFO"

    async with ControlPlane(config) as cp:
        await asyncio.sleep(0.5)

        # Create and register agents manually
        researcher, coder, planner = await register_agents(cp)

        # List agents
        agents = cp.list_agents()
        logger.info(f"Active agents: {[a.name for a in agents]}")

        # Submit a simple task
        task = Task(
            name="Research AI trends",
            description="Research current trends in AI agent frameworks",
            required_capability="research",
            payload={"topic": "AI agent frameworks 2024", "depth": "comprehensive"},
            priority=5,
        )

        task_id = cp.submit_task(task)
        logger.info(f"Submitted task: {task_id}")

        # Wait for completion
        await asyncio.sleep(2)

        # Check result
        result_task = cp.get_task(task_id)
        if result_task and result_task.result:
            logger.info(f"Task result: {result_task.result}")

        # Verify agent exists
        researcher_agents = cp.find_agents_by_capability("research")
        if researcher_agents:
            researcher = researcher_agents[0]
            logger.info(f"Found researcher agent: {researcher.name}")

    logger.info("=== Basic Example Complete ===")


async def run_workflow_example():
    """Run a workflow example with multiple dependent tasks."""
    logger.info("=== Starting Workflow Example ===")

    config = DEFAULT_CONFIG
    async with ControlPlane(config) as cp:
        await asyncio.sleep(0.5)

        researcher, coder, planner = await register_agents(cp)

        # Create a workflow: Research -> Plan -> Code -> Review
        tasks = [
            Task(
                name="Research Requirements",
                description="Research best practices for building a REST API",
                required_capability="research",
                payload={"topic": "REST API best practices 2024", "depth": "comprehensive"},
                priority=10,
            ),
            Task(
                name="Design Architecture",
                description="Design the API architecture based on research",
                required_capability="workflow_design",
                payload={
                    "objective": "Design a scalable REST API architecture",
                    "available_capabilities": ["code_generation", "code_review"],
                },
                dependencies=[],
                priority=8,
            ),
            Task(
                name="Implement Core Endpoints",
                description="Generate code for core API endpoints",
                required_capability="code_generation",
                payload={
                    "specification": "REST API with users, posts, comments endpoints",
                    "language": "python",
                    "framework": "fastapi",
                },
                dependencies=[],
                priority=6,
            ),
            Task(
                name="Code Review",
                description="Review the generated code for issues",
                required_capability="code_review",
                payload={"code": "", "language": "python"},
                dependencies=[],
                priority=4,
            ),
        ]

        # Set up dependencies
        tasks[1].dependencies = [tasks[0].id]  # Design depends on Research
        tasks[2].dependencies = [tasks[1].id]  # Implement depends on Design
        tasks[3].dependencies = [tasks[2].id]  # Review depends on Implement

        # Submit workflow
        workflow = cp.submit_workflow("Build REST API", tasks)
        logger.info(f"Started workflow: {workflow.name} ({workflow.id})")

        # Monitor workflow with timeout
        max_wait = 30  # seconds
        start_time = asyncio.get_event_loop().time()

        while workflow.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            await asyncio.sleep(1)
            workflow = cp.get_workflow(workflow.id)

            if asyncio.get_event_loop().time() - start_time > max_wait:
                logger.warning("Workflow timeout")
                break

            # Log task statuses
            for task in workflow.tasks:
                logger.info(f"  Task {task.name}: {task.status}")

        logger.info(f"Workflow completed with status: {workflow.status}")

        # Print results
        for task in workflow.tasks:
            if task.result:
                logger.info(f"Task {task.name} result keys: {list(task.result.keys())}")

    logger.info("=== Workflow Example Complete ===")


async def run_multi_agent_collaboration():
    """Run an example of multi-agent collaboration."""
    logger.info("=== Starting Multi-Agent Collaboration Example ===")

    config = DEFAULT_CONFIG
    async with ControlPlane(config) as cp:
        await asyncio.sleep(0.5)

        researcher, coder, planner = await register_agents(cp)

        logger.info(f"Collaborating agents: {researcher.name}, {coder.name}, {planner.name}")

        # Step 1: Planner creates a plan
        plan_task = Task(
            name="Plan Feature Implementation",
            description="Create a plan for implementing a user authentication system",
            required_capability="task_planning",
            payload={
                "goal": "Implement JWT-based user authentication with login, register, and password reset",
                "constraints": {"framework": "fastapi", "database": "postgresql"},
            },
        )

        plan_task_id = cp.submit_task(plan_task)
        await asyncio.sleep(2)

        plan_result = cp.get_task(plan_task_id)
        if plan_result and plan_result.result:
            plan = plan_result.result.get("plan", [])
            logger.info(f"Plan created with {len(plan)} steps")

            # Step 2: For each step, assign to appropriate agent
            for i, step in enumerate(plan):
                capability = step.get("capability", "")
                if capability:
                    task = Task(
                        name=f"Step {i + 1}: {step.get('action', 'Execute')}",
                        description=step.get("action", ""),
                        required_capability=capability,
                        payload={"step": step, "plan_context": plan},
                    )
                    cp.submit_task(task)

            # Wait for all tasks
            await asyncio.sleep(3)

            # Check results
            all_tasks = [cp.get_task(plan_task_id)]

            for task in all_tasks:
                if task and task.result:
                    logger.info(f"Task {task.name} completed: {list(task.result.keys())}")

    logger.info("=== Multi-Agent Collaboration Example Complete ===")


async def run_event_driven_example():
    """Run an example showing event-driven architecture."""
    logger.info("=== Starting Event-Driven Example ===")

    config = DEFAULT_CONFIG
    cp = ControlPlane(config)

    # Track events
    events_received = []

    def event_handler(event_type: str, data: any):
        events_received.append((event_type, data))
        logger.info(f"Event received: {event_type} - {data}")

    # Subscribe to events
    cp.event_bus.subscribe_all(event_handler)

    await cp.start()
    await asyncio.sleep(0.5)

    # Create and register a research agent
    researcher = create_research_agent("researcher", cp)
    cp.registry.register(researcher)
    await cp.lifecycle.initialize_agent(researcher)
    for cap in researcher.capabilities:
        cp.capability_registry.register_capability(cap, researcher.id)
    cp.message_bus.register_agent_queue(researcher.id, asyncio.Queue())
    cp.message_bus.subscribe(researcher.id, researcher.handle_message)

    # Submit some tasks to generate events
    task = Task(
        name="Event Test Task",
        description="Task to generate events",
        required_capability="research",
        payload={"topic": "event-driven architecture"},
    )

    cp.submit_task(task)
    await asyncio.sleep(2)

    await cp.stop()

    logger.info(f"Total events received: {len(events_received)}")
    for event_type, data in events_received:
        logger.info(f"  {event_type}: {data}")

    logger.info("=== Event-Driven Example Complete ===")


async def main():
    """Run all examples."""
    # Run examples sequentially
    await run_basic_example()
    print()
    await run_workflow_example()
    print()
    await run_multi_agent_collaboration()
    print()
    await run_event_driven_example()

    logger.info("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
