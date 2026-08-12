"""
Plugin system for extensibility.
"""

from .manager import (
    PluginManager,
    PluginInfo,
    CapabilityRegistry,
    AgentFactory,
    ExtensionPoint,
    AgentExtensionPoint,
    SchedulerExtensionPoint,
    MessengerExtensionPoint,
)

__all__ = [
    "PluginManager",
    "PluginInfo",
    "CapabilityRegistry",
    "AgentFactory",
    "ExtensionPoint",
    "AgentExtensionPoint",
    "SchedulerExtensionPoint",
    "MessengerExtensionPoint",
]
