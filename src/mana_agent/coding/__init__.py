"""Provider-neutral coding backend contracts."""

from typing import Any

from mana_agent.coding.backend import CodingAgentBackend
from mana_agent.coding.models import (
    AgentEvent,
    CodingBackendDecision,
    CodingEngineMode,
    CodingEngineSettings,
    CodingStatus,
    CodingTask,
    CodingTaskResult,
    ExternalCodingEngineConfig,
    InternalCodingEngineConfig,
    WorkspaceContext,
)
from mana_agent.coding.orchestrator import CodingBackendOrchestrator
from mana_agent.coding.registry import (
    CodingBackendDecisionError,
    CodingBackendRegistry,
)
from mana_agent.coding.selection import (
    CodingBackendConfigurationError,
    CodingBackendName,
    CodingBackendSelection,
    resolve_coding_backend,
)


def __getattr__(name: str) -> Any:
    if name == "UnifiedCodingAgentFacade":
        from mana_agent.coding.facade import UnifiedCodingAgentFacade

        return UnifiedCodingAgentFacade
    if name == "CodingEngine":
        from mana_agent.coding.engine import CodingEngine

        return CodingEngine
    if name in {"CodexProviderAdapter", "ExternalCodingEngine"}:
        from mana_agent.coding.external_engine import CodexProviderAdapter, ExternalCodingEngine

        return CodexProviderAdapter if name == "CodexProviderAdapter" else ExternalCodingEngine
    if name == "InternalCodingEngine":
        from mana_agent.coding.internal_engine import InternalCodingEngine

        return InternalCodingEngine
    if name in {
        "ProgrammaticExecutionSession",
        "ProgrammaticExecutionState",
        "ProgrammaticToolCall",
        "ToolExecutionDispatcher",
        "build_programmatic_tool_definitions",
    }:
        import mana_agent.coding.programmatic as prog

        return getattr(prog, name)
    if name == "INTERNAL_CODING_SYSTEM_INSTRUCTION":
        from mana_agent.coding.prompts import INTERNAL_CODING_SYSTEM_INSTRUCTION

        return INTERNAL_CODING_SYSTEM_INSTRUCTION
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AgentEvent",
    "CodexProviderAdapter",
    "CodingAgentBackend",
    "CodingBackendConfigurationError",
    "CodingBackendDecision",
    "CodingBackendDecisionError",
    "CodingBackendName",
    "CodingBackendOrchestrator",
    "CodingBackendRegistry",
    "CodingBackendSelection",
    "CodingEngine",
    "CodingEngineMode",
    "CodingEngineSettings",
    "CodingStatus",
    "CodingTask",
    "CodingTaskResult",
    "ExternalCodingEngine",
    "ExternalCodingEngineConfig",
    "INTERNAL_CODING_SYSTEM_INSTRUCTION",
    "InternalCodingEngine",
    "InternalCodingEngineConfig",
    "ProgrammaticExecutionSession",
    "ProgrammaticExecutionState",
    "ProgrammaticToolCall",
    "ToolExecutionDispatcher",
    "UnifiedCodingAgentFacade",
    "WorkspaceContext",
    "build_programmatic_tool_definitions",
    "resolve_coding_backend",
]
