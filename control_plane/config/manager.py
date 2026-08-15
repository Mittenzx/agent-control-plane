"""
Configuration management for the control plane.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    type: str
    name: str
    capabilities: List[Dict[str, Any]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    auto_start: bool = True


@dataclass
class ControlPlaneConfig:
    """Main control plane configuration."""

    name: str = "agent-control-plane"
    version: str = "0.1.0"
    log_level: str = "INFO"
    data_dir: str = "./data"

    # Agent configurations
    agents: List[AgentConfig] = field(default_factory=list)

    # Plugin configurations
    plugins: List[Dict[str, Any]] = field(default_factory=list)

    # Orchestration settings
    max_concurrent_tasks: int = 10
    default_task_timeout: float = 300.0
    scheduler_interval: float = 1.0

    # Messaging settings
    message_history_size: int = 10000
    request_timeout: float = 30.0

    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str) -> "ControlPlaneConfig":
        """Load configuration from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ControlPlaneConfig":
        """Create configuration from a dictionary."""
        agents = [AgentConfig(**a) for a in data.get("agents", [])]
        return cls(
            name=data.get("name", "agent-control-plane"),
            version=data.get("version", "0.1.0"),
            log_level=data.get("log_level", "INFO"),
            data_dir=data.get("data_dir", "./data"),
            agents=agents,
            plugins=data.get("plugins", []),
            max_concurrent_tasks=data.get("max_concurrent_tasks", 10),
            default_task_timeout=data.get("default_task_timeout", 300.0),
            scheduler_interval=data.get("scheduler_interval", 1.0),
            message_history_size=data.get("message_history_size", 10000),
            request_timeout=data.get("request_timeout", 30.0),
            custom=data.get("custom", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "log_level": self.log_level,
            "data_dir": self.data_dir,
            "agents": [asdict(a) for a in self.agents],
            "plugins": self.plugins,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "default_task_timeout": self.default_task_timeout,
            "scheduler_interval": self.scheduler_interval,
            "message_history_size": self.message_history_size,
            "request_timeout": self.request_timeout,
            "custom": self.custom,
        }

    def to_file(self, path: str) -> None:
        """Save configuration to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ConfigManager:
    """Manages configuration loading, validation, and hot-reloading."""

    def __init__(self, config: Optional[ControlPlaneConfig] = None):
        self._config = config or ControlPlaneConfig()
        self._watchers: List[Callable[[ControlPlaneConfig], None]] = []
        self._config_path: Optional[str] = None

    @property
    def config(self) -> ControlPlaneConfig:
        return self._config

    def load_from_file(self, path: str) -> None:
        """Load configuration from file."""
        self._config = ControlPlaneConfig.from_file(path)
        self._config_path = path
        logger.info(f"Loaded configuration from {path}")
        self._notify_watchers()

    def load_from_env(self, prefix: str = "AGENT_CP_") -> None:
        """Load configuration from environment variables."""
        # Simple env var override for common settings
        env_mappings = {
            f"{prefix}LOG_LEVEL": "log_level",
            f"{prefix}DATA_DIR": "data_dir",
            f"{prefix}MAX_CONCURRENT_TASKS": "max_concurrent_tasks",
            f"{prefix}DEFAULT_TASK_TIMEOUT": "default_task_timeout",
        }

        for env_var, attr in env_mappings.items():
            raw_value = os.environ.get(env_var)
            if raw_value is not None:
                # Type conversion
                if attr in ("max_concurrent_tasks",):
                    value: object = int(raw_value)
                elif attr in ("default_task_timeout",):
                    value = float(raw_value)
                else:
                    value = raw_value
                setattr(self._config, attr, value)
                logger.info(f"Override {attr} from env: {value}")

        self._notify_watchers()

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                self._config.custom[key] = value
        self._notify_watchers()

    def add_watcher(self, callback: Callable[[ControlPlaneConfig], None]) -> None:
        """Add a configuration change watcher."""
        self._watchers.append(callback)

    def remove_watcher(self, callback: Callable[[ControlPlaneConfig], None]) -> None:
        """Remove a configuration change watcher."""
        if callback in self._watchers:
            self._watchers.remove(callback)

    def _notify_watchers(self) -> None:
        """Notify all watchers of configuration change."""
        for watcher in self._watchers:
            try:
                watcher(self._config)
            except Exception as e:
                logger.error(f"Error in config watcher: {e}")

    def save(self, path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        target_path = path or self._config_path
        if not target_path:
            raise ValueError("No config path specified")
        self._config.to_file(target_path)
        logger.info(f"Saved configuration to {target_path}")

    def get_agent_configs(self) -> List[AgentConfig]:
        """Get enabled agent configurations."""
        return [a for a in self._config.agents if a.enabled]


# Default configuration template
DEFAULT_CONFIG = ControlPlaneConfig(
    name="agent-control-plane",
    agents=[
        AgentConfig(
            type="hermes_agent",
            name="researcher",
            capabilities=[
                {"name": "research", "description": "Web research and information gathering"},
                {"name": "summarize", "description": "Summarize long texts"},
            ],
            config={"model": "gpt-4", "temperature": 0.7},
        ),
        AgentConfig(
            type="hermes_agent",
            name="coder",
            capabilities=[
                {"name": "code_generation", "description": "Generate code from specs"},
                {"name": "code_review", "description": "Review code for issues"},
            ],
            config={"model": "gpt-4", "temperature": 0.3},
        ),
        AgentConfig(
            type="hermes_agent",
            name="planner",
            capabilities=[
                {"name": "task_planning", "description": "Break down goals into tasks"},
                {"name": "workflow_design", "description": "Design multi-step workflows"},
            ],
            config={"model": "gpt-4", "temperature": 0.5},
        ),
    ],
    plugins=[],
    max_concurrent_tasks=10,
    default_task_timeout=300.0,
)
