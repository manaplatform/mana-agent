"""Provider-neutral coding backend contracts."""

from mana_agent.coding.backend import CodingAgentBackend
from mana_agent.coding.models import (
    AgentEvent,
    CodingBackendDecision,
    CodingTask,
    CodingTaskResult,
    WorkspaceContext,
)
from mana_agent.coding.orchestrator import CodingBackendOrchestrator
from mana_agent.coding.registry import CodingBackendRegistry
from mana_agent.coding.selection import (
    CodingBackendConfigurationError,
    CodingBackendSelection,
    resolve_coding_backend,
)

__all__ = [
    "AgentEvent",
    "CodingAgentBackend",
    "CodingBackendDecision",
    "CodingBackendOrchestrator",
    "CodingBackendRegistry",
    "CodingBackendConfigurationError",
    "CodingBackendSelection",
    "resolve_coding_backend",
    "CodingTask",
    "CodingTaskResult",
    "WorkspaceContext",
]
