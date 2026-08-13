"""
HermesAgent adapter — drives real Hermes processes as control-plane agents.

Each HermesAgent is backed by an actual `hermes chat` process (headless,
quiet mode). Tasks are executed by invoking Hermes with the task payload as a
prompt, and results + session IDs are captured back. Sessions persist in
Hermes's SQLite store and can be resumed via --resume for stateful agents.

This is the integration point that makes the control plane run *on Hermes* as
its execution core: agents get full Hermes tool access, skills, and persistent
memory, rather than running mock logic.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.interfaces import (
    Agent,
    AgentCapability,
    AgentInfo,
    Task,
)

logger = logging.getLogger(__name__)


class HermesAgent(Agent):
    """A control-plane agent backed by a real Hermes process."""

    def __init__(
        self,
        agent_info: AgentInfo,
        hermes_command: str = "hermes",
        model: Optional[str] = None,
        provider: Optional[str] = None,
        timeout_seconds: float = 300.0,
        cwd: Optional[str] = None,
    ):
        super().__init__(agent_info)
        self.hermes_command = hermes_command
        self.model = model
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.cwd = cwd
        self._session_id: Optional[str] = None
        self._process: Optional[asyncio.subprocess.Process] = None
        self._last_output: str = ""
        self._last_error: str = ""

    @property
    def session_id(self) -> Optional[str]:
        """The Hermes session ID backing this agent (persistent/stateful)."""
        return self._session_id

    async def initialize(self) -> None:
        """Verify Hermes is available and mark the agent ready."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.hermes_command,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            self.info.metadata["hermes_version"] = out.decode().strip()
            logger.info(
                f"Hermes agent {self.name} initialized "
                f"(hermes {self.info.metadata.get('hermes_version')})"
            )
        except Exception as e:
            logger.error(f"Hermes agent {self.name} init check failed: {e}")
            raise RuntimeError(
                f"Hermes not available for agent {self.name}: {e}. Ensure `hermes` is on PATH."
            ) from e

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a task by running it through a real Hermes process."""
        prompt = self._build_prompt(task)
        self._last_output = ""
        self._last_error = ""

        cmd = self._build_command(prompt)
        logger.info(
            f"Hermes agent {self.name} running task '{task.name}' "
            f"(capability={task.required_capability})"
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
            )
            self._process = proc

            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_seconds)
            except asyncio.TimeoutError:
                proc.kill()
                raise TimeoutError(
                    f"Task '{task.name}' timed out after {self.timeout_seconds}s"
                ) from None

            stdout = out.decode(errors="replace")
            stderr = err.decode(errors="replace")
            self._last_output = stdout
            self._last_error = stderr

            # Parse session_id (Hermes emits it on stderr in quiet mode)
            self._session_id = self._parse_session_id(stdout) or self._parse_session_id(stderr)

            return {
                "session_id": self._session_id,
                "output": stdout.strip(),
                "exit_code": proc.returncode,
                "agent_id": self.id,
                "agent_name": self.name,
                "completed_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Hermes agent {self.name} task failed: {e}")
            raise

    def _build_prompt(self, task: Task) -> str:
        """Compose a Hermes prompt from a task."""
        lines = []
        if task.name:
            lines.append(f"# Task: {task.name}")
        if task.description:
            lines.append(f"\n{task.description}")
        if task.payload:
            import json

            lines.append(f"\nContext/parameters:\n{json.dumps(task.payload, indent=2)}")
        lines.append("\nComplete this task and provide a clear, concise final result.")
        return "\n".join(lines)

    def _build_command(self, prompt: str) -> List[str]:
        """Build the `hermes chat` command for the given prompt."""
        cmd = [self.hermes_command, "chat", "-q", prompt, "-Q"]
        if self.model:
            cmd += ["-m", self.model]
        if self.provider:
            cmd += ["--provider", self.provider]
        if self._session_id:
            cmd += ["--resume", self._session_id]
        return cmd

    @staticmethod
    def _parse_session_id(output: str) -> Optional[str]:
        """Extract session_id from Hermes quiet-mode output."""
        for line in output.splitlines():
            if line.startswith("session_id:"):
                return line.split(":", 1)[1].strip()
        return None

    async def shutdown(self) -> None:
        """Terminate any running Hermes process."""
        if self._process and self._process.returncode is None:
            self._process.kill()
            try:
                await self._process.wait()
            except Exception:
                pass
        logger.info(f"Hermes agent {self.name} shut down")

    def get_stdout(self) -> str:
        """Return the last captured output (for dashboards / logs)."""
        return self._last_output

    def get_stderr(self) -> str:
        return self._last_error

    def __repr__(self) -> str:
        return f"<HermesAgent {self.name} session={self._session_id}>"


def create_hermes_agent(
    name: str,
    capabilities: List[str],
    control_plane: Any = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    hermes_command: str = "hermes",
    description: str = "Hermes-powered agent",
) -> HermesAgent:
    """Factory helper to build a HermesAgent from a capability list."""
    import uuid

    caps = [AgentCapability(name=c, description=f"Hermes capability: {c}") for c in capabilities]
    info = AgentInfo(
        id=str(uuid.uuid4()),
        name=name,
        type="hermes_agent",
        capabilities=caps,
        metadata={"backend": "hermes", "model": model or "default"},
    )
    return HermesAgent(
        info,
        hermes_command=hermes_command,
        model=model,
        provider=provider,
    )
