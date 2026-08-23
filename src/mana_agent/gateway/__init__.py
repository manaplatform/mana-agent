"""Gateway layer for unified access to Mana-Agent multi-agent chat and runtime.

All frontends (TUI chat, Telegram, Dashboard/API, CLI) should connect through
the gateway to reach agents. The gateway centralizes construction of
AskService / ChatService / CodingAgent stacks, auto-chat, and turn orchestration.

The gateway package intentionally uses lazy exports.

Why:
    Importing ``mana_agent.gateway.feature_integration`` first initializes the
    ``mana_agent.gateway`` package. Eager imports from this ``__init__`` can
    therefore pull in ``chat_gateway`` / ``stack`` / ``turn_engine`` before the
    caller's module graph has finished initializing.

    In particular, the CLI composition path can otherwise form a cycle similar
    to:

        commands.cli_internal
            -> multi_agent.MainAgent
            -> gateway.feature_integration
            -> gateway.__init__
            -> gateway.chat_gateway
            -> gateway.stack
            -> commands.cli_internal

    Keeping package-level exports lazy preserves the public API without making
    importing one gateway submodule initialize the entire gateway runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AgentChatGateway",
    "RichChatContext",
    "ChatGatewayConfig",
    "EntryRouteContext",
    "EntryRouteRegistry",
    "EntryRouter",
    "EntryRoutingDecision",
    "EntryRoutingError",
    "RouteAvailability",
    "RouteRegistration",
    "ChatStack",
    "build_chat_stack",
    "LaneCoordinator",
    "GatewayRoutingAuthority",
    "GatewayRoutingError",
    "LaneContract",
    "LaneId",
    "LanePriority",
    "LaneTaskState",
    "LockMode",
    "MultiTaskError",
    "MultiTaskItem",
    "MultiTaskOrchestrator",
    "MultiTaskPlan",
    "ChatTurnResult",
    "process_chat_turn",
    "should_use_coding_agent_turn",
    "is_auto_chat_connector_turn",
    "ApprovalState",
    "ConversationContextAvailability",
    "ExecutionRecoveryState",
    "IdentitySessionRelationship",
    "MemoryAvailability",
    "ModelCandidateCapacity",
    "PreviousTurnPointers",
    "RoutingExecutionEnvelope",
    "build_routing_execution_envelope",
    "IntegrationAuthority",
]


# ---------------------------------------------------------------------------
# Static type-checking imports
#
# These imports are visible to type checkers but are never executed during
# normal runtime package initialization.
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from .chat_gateway import AgentChatGateway, RichChatContext
    from .config import ChatGatewayConfig
    from .entry_routing import (
        EntryRouteContext,
        EntryRouteRegistry,
        EntryRouter,
        EntryRoutingDecision,
        EntryRoutingError,
        RouteAvailability,
        RouteRegistration,
    )
    from .envelope import (
        ApprovalState,
        ConversationContextAvailability,
        ExecutionRecoveryState,
        IdentitySessionRelationship,
        MemoryAvailability,
        ModelCandidateCapacity,
        PreviousTurnPointers,
        RoutingExecutionEnvelope,
        build_routing_execution_envelope,
    )
    from .lane_coordinator import LaneCoordinator
    from .routing import GatewayRoutingAuthority, GatewayRoutingError
    from .stack import ChatStack, build_chat_stack
    from .turn_engine import (
        ChatTurnResult,
        is_auto_chat_connector_turn,
        process_chat_turn,
        should_use_coding_agent_turn,
    )

    # Keep these paths synchronized with their actual definitions.
    #
    # If LaneContract / LaneId / LanePriority / LaneTaskState / LockMode live
    # in a different module in the repository, update only this mapping and
    # _LAZY_EXPORTS below.
    from .lane_contract import (
        LaneContract,
        LaneId,
        LanePriority,
        LaneTaskState,
        LockMode,
    )

    from mana_agent.multi_agent.runtime.multi_task_orchestrator import (
        MultiTaskError,
        MultiTaskItem,
        MultiTaskOrchestrator,
        MultiTaskPlan,
    )


# ---------------------------------------------------------------------------
# Lazy export registry
#
# Format:
#     public_name: ("module.path", "attribute_name")
#
# Relative paths are resolved against ``mana_agent.gateway``.
# ---------------------------------------------------------------------------

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # chat_gateway
    "AgentChatGateway": (".chat_gateway", "AgentChatGateway"),
    "RichChatContext": (".chat_gateway", "RichChatContext"),

    # config
    "ChatGatewayConfig": (".config", "ChatGatewayConfig"),

    # entry_routing
    "EntryRouteContext": (".entry_routing", "EntryRouteContext"),
    "EntryRouteRegistry": (".entry_routing", "EntryRouteRegistry"),
    "EntryRouter": (".entry_routing", "EntryRouter"),
    "EntryRoutingDecision": (".entry_routing", "EntryRoutingDecision"),
    "EntryRoutingError": (".entry_routing", "EntryRoutingError"),
    "RouteAvailability": (".entry_routing", "RouteAvailability"),
    "RouteRegistration": (".entry_routing", "RouteRegistration"),

    # stack
    "ChatStack": (".stack", "ChatStack"),
    "build_chat_stack": (".stack", "build_chat_stack"),

    # lane coordinator
    "LaneCoordinator": (".lane_coordinator", "LaneCoordinator"),

    # routing
    "GatewayRoutingAuthority": (".routing", "GatewayRoutingAuthority"),
    "GatewayRoutingError": (".routing", "GatewayRoutingError"),

    # lane contract / lane models
    #
    # NOTE:
    # Your original __all__ exported these names but the supplied file did not
    # import them at all. This mapping assumes they are defined in
    # ``gateway/lane_contract.py``.
    #
    # If they actually live elsewhere, change ".lane_contract" to the real
    # module path.
    "LaneContract": (".lane_contract", "LaneContract"),
    "LaneId": (".lane_contract", "LaneId"),
    "LanePriority": (".lane_contract", "LanePriority"),
    "LaneTaskState": (".lane_contract", "LaneTaskState"),
    "LockMode": (".lane_contract", "LockMode"),

    # multi-task runtime
    "MultiTaskError": (
        "mana_agent.multi_agent.runtime.multi_task_orchestrator",
        "MultiTaskError",
    ),
    "MultiTaskItem": (
        "mana_agent.multi_agent.runtime.multi_task_orchestrator",
        "MultiTaskItem",
    ),
    "MultiTaskOrchestrator": (
        "mana_agent.multi_agent.runtime.multi_task_orchestrator",
        "MultiTaskOrchestrator",
    ),
    "MultiTaskPlan": (
        "mana_agent.multi_agent.runtime.multi_task_orchestrator",
        "MultiTaskPlan",
    ),

    # turn engine
    "ChatTurnResult": (".turn_engine", "ChatTurnResult"),
    "process_chat_turn": (".turn_engine", "process_chat_turn"),
    "should_use_coding_agent_turn": (
        ".turn_engine",
        "should_use_coding_agent_turn",
    ),
    "is_auto_chat_connector_turn": (
        ".turn_engine",
        "is_auto_chat_connector_turn",
    ),

    # routing execution envelope
    "ApprovalState": (".envelope", "ApprovalState"),
    "ConversationContextAvailability": (
        ".envelope",
        "ConversationContextAvailability",
    ),
    "ExecutionRecoveryState": (".envelope", "ExecutionRecoveryState"),
    "IdentitySessionRelationship": (
        ".envelope",
        "IdentitySessionRelationship",
    ),
    "MemoryAvailability": (".envelope", "MemoryAvailability"),
    "ModelCandidateCapacity": (".envelope", "ModelCandidateCapacity"),
    "PreviousTurnPointers": (".envelope", "PreviousTurnPointers"),
    "RoutingExecutionEnvelope": (".envelope", "RoutingExecutionEnvelope"),
    "build_routing_execution_envelope": (
        ".envelope",
        "build_routing_execution_envelope",
    ),
    "IntegrationAuthority": (".feature_integration", "IntegrationAuthority"),
}


def __getattr__(name: str) -> Any:
    """Resolve public gateway exports lazily.

    This prevents package initialization from eagerly importing the complete
    chat/runtime stack while preserving imports such as::

        from mana_agent.gateway import AgentChatGateway
        from mana_agent.gateway import build_chat_stack

    Resolved values are cached in ``globals()`` so subsequent attribute access
    has the same practical behavior as a normal module-level import.
    """

    target = _LAZY_EXPORTS.get(name)

    if target is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    module_name, attribute_name = target

    if module_name.startswith("."):
        module = import_module(module_name, package=__name__)
    else:
        module = import_module(module_name)

    value = getattr(module, attribute_name)

    # Cache the resolved export. This avoids repeated __getattr__ calls and
    # preserves normal package attribute semantics after first access.
    globals()[name] = value

    return value


def __dir__() -> list[str]:
    """Expose lazy public symbols to dir(), IDEs, and interactive shells."""

    return sorted(set(globals()) | set(__all__))
