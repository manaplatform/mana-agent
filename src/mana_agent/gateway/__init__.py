"""Gateway layer for unified access to Mana-Agent multi-agent chat and runtime.

All frontends (TUI chat, Telegram, Dashboard/API, CLI) should connect through
the gateway to reach agents. The gateway centralizes construction of
AskService / ChatService / CodingAgent stacks, auto-chat, and turn orchestration.

See chat_gateway.py for the main implementation.
"""

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
from .stack import ChatStack, build_chat_stack
from .lane_coordinator import LaneCoordinator
from .routing import GatewayRoutingAuthority, GatewayRoutingError
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
from .turn_engine import (
    ChatTurnResult,
    is_auto_chat_connector_turn,
    process_chat_turn,
    should_use_coding_agent_turn,
)
from mana_agent.multi_agent.runtime.multi_task_orchestrator import (
    MultiTaskError,
    MultiTaskItem,
    MultiTaskOrchestrator,
    MultiTaskPlan,
)

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
]

