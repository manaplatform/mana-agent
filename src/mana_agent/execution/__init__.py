"""Provider-neutral remote execution fabric."""

from mana_agent.execution.config import ExecutionConfig, build_provider_registry
from mana_agent.execution.manager import ExecutionManager


def build_execution_manager(settings, *, event_sink=None) -> ExecutionManager:
    config = ExecutionConfig.from_settings(settings)
    return ExecutionManager(build_provider_registry(config), config, event_sink=event_sink)


__all__ = ["ExecutionConfig", "ExecutionManager", "build_execution_manager"]
