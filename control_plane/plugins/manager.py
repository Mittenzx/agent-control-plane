"""
Plugin system for extending the control plane.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Type, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict
import importlib
import inspect

from ..core.interfaces import AgentPlugin, ControlPlane, Agent, AgentInfo, AgentCapability

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Information about a loaded plugin."""

    name: str
    version: str
    plugin_class: Type[AgentPlugin]
    instance: Optional[AgentPlugin] = None
    loaded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class PluginManager:
    """Manages loading, unloading, and lifecycle of plugins."""

    def __init__(self, control_plane: ControlPlane):
        self.control_plane = control_plane
        self._plugins: Dict[str, PluginInfo] = {}
        self._hooks: Dict[str, List[Callable]] = defaultdict(list)

    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a hook callback."""
        self._hooks[hook_name].append(callback)

    async def load_plugin(
        self, plugin_class: Type[AgentPlugin], config: Optional[Dict[str, Any]] = None
    ) -> AgentPlugin:
        """Load a plugin by class."""
        # Create instance
        plugin = plugin_class()

        # Check if already loaded
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin {plugin.name} already loaded")

        # Store info
        info = PluginInfo(
            name=plugin.name,
            version=plugin.version,
            plugin_class=plugin_class,
            instance=plugin,
            metadata=config or {},
        )
        self._plugins[plugin.name] = info

        # Call on_load
        try:
            await plugin.on_load(self.control_plane)
            info.loaded = True
            logger.info(f"Loaded plugin: {plugin.name} v{plugin.version}")
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin.name}: {e}")
            self._plugins.pop(plugin.name, None)
            raise

        # Trigger hook
        await self._trigger_hook("plugin_loaded", plugin)

        return plugin

    async def load_plugin_from_module(
        self, module_path: str, class_name: str, config: Optional[Dict[str, Any]] = None
    ) -> AgentPlugin:
        """Load a plugin from a module path."""
        module = importlib.import_module(module_path)
        plugin_class = getattr(module, class_name)
        return await self.load_plugin(plugin_class, config)

    async def unload_plugin(self, name: str) -> None:
        """Unload a plugin by name."""
        info = self._plugins.get(name)
        if not info:
            raise ValueError(f"Plugin {name} not loaded")

        if info.instance:
            try:
                await info.instance.on_unload()
            except Exception as e:
                logger.error(f"Error unloading plugin {name}: {e}")

        # Trigger hook
        await self._trigger_hook("plugin_unloaded", info.instance)

        self._plugins.pop(name, None)
        logger.info(f"Unloaded plugin: {name}")

    async def unload_all(self) -> None:
        """Unload all plugins."""
        for name in list(self._plugins.keys()):
            await self.unload_plugin(name)

    def get_plugin(self, name: str) -> Optional[AgentPlugin]:
        """Get a loaded plugin instance."""
        info = self._plugins.get(name)
        return info.instance if info else None

    def list_plugins(self) -> List[PluginInfo]:
        """List all loaded plugins."""
        return list(self._plugins.values())

    async def _trigger_hook(self, hook_name: str, *args, **kwargs) -> None:
        """Trigger all callbacks for a hook."""
        for callback in self._hooks.get(hook_name, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in hook {hook_name}: {e}")


class CapabilityRegistry:
    """Registry for dynamic capability registration."""

    def __init__(self):
        self._capabilities: Dict[str, AgentCapability] = {}
        self._providers: Dict[str, List[str]] = {}  # capability -> agent_ids

    def register_capability(self, capability: AgentCapability, agent_id: str) -> None:
        """Register a capability provided by an agent."""
        self._capabilities[capability.name] = capability
        if capability.name not in self._providers:
            self._providers[capability.name] = []
        if agent_id not in self._providers[capability.name]:
            self._providers[capability.name].append(agent_id)
        logger.debug(f"Registered capability: {capability.name} from agent {agent_id}")

    def unregister_capability(self, capability_name: str, agent_id: str) -> None:
        """Unregister a capability from an agent."""
        if capability_name in self._providers:
            self._providers[capability_name] = [
                aid for aid in self._providers[capability_name] if aid != agent_id
            ]
            if not self._providers[capability_name]:
                self._providers.pop(capability_name, None)
                self._capabilities.pop(capability_name, None)
        logger.debug(f"Unregistered capability: {capability_name} from agent {agent_id}")

    def get_capability(self, name: str) -> Optional[AgentCapability]:
        """Get capability definition."""
        return self._capabilities.get(name)

    def list_capabilities(self) -> List[AgentCapability]:
        """List all registered capabilities."""
        return list(self._capabilities.values())

    def find_providers(self, capability_name: str) -> List[str]:
        """Find agent IDs that provide a capability."""
        return self._providers.get(capability_name, [])


class ExtensionPoint(ABC):
    """Base class for extension points that plugins can implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Extension point name."""
        pass

    @abstractmethod
    def get_interface(self) -> Type:
        """Get the interface type for this extension point."""
        pass


class AgentFactory:
    """Factory for creating agents from plugins or configuration."""

    def __init__(self, control_plane: ControlPlane):
        self.control_plane = control_plane
        self._agent_types: Dict[str, Type[Agent]] = {}
        self._builders: Dict[str, Callable] = {}

    def register_agent_type(self, type_name: str, agent_class: Type[Agent]) -> None:
        """Register an agent type."""
        self._agent_types[type_name] = agent_class
        logger.info(f"Registered agent type: {type_name}")

    def register_builder(self, type_name: str, builder: Callable) -> None:
        """Register a custom builder function."""
        self._builders[type_name] = builder
        logger.info(f"Registered builder for: {type_name}")

    async def create_agent(
        self, type_name: str, agent_info: AgentInfo, config: Optional[Dict[str, Any]] = None
    ) -> Agent:
        """Create an agent instance."""
        # Try builder first
        if type_name in self._builders:
            builder = self._builders[type_name]
            if asyncio.iscoroutinefunction(builder):
                return await builder(agent_info, config, self.control_plane)
            return builder(agent_info, config, self.control_plane)

        # Try registered type
        if type_name in self._agent_types:
            agent_class = self._agent_types[type_name]
            # Inspect constructor
            sig = inspect.signature(agent_class.__init__)
            kwargs: Dict[str, Any] = {"agent_info": agent_info}
            if "config" in sig.parameters:
                kwargs["config"] = config or {}
            if "control_plane" in sig.parameters:
                kwargs["control_plane"] = self.control_plane
            return agent_class(**kwargs)

        raise ValueError(f"Unknown agent type: {type_name}")

    def list_agent_types(self) -> List[str]:
        """List available agent types."""
        return list(self._agent_types.keys()) + list(self._builders.keys())


# Built-in extension points
class AgentExtensionPoint(ExtensionPoint):
    """Extension point for custom agent types."""

    @property
    def name(self) -> str:
        return "agent"

    def get_interface(self) -> Type:
        return Agent


class SchedulerExtensionPoint(ExtensionPoint):
    """Extension point for custom scheduling strategies."""

    @property
    def name(self) -> str:
        return "scheduler"

    def get_interface(self) -> Type:
        from ..orchestration.engine import TaskScheduler

        return TaskScheduler


class MessengerExtensionPoint(ExtensionPoint):
    """Extension point for custom messaging transports."""

    @property
    def name(self) -> str:
        return "messenger"

    def get_interface(self) -> Type:
        from ..messaging.bus import MessageBus

        return MessageBus
