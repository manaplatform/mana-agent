"""Canonical execution envelope for model routing and turn orchestration.

Routing receives this structured envelope containing current request, identity,
recovery state, Phase-0 accounting snapshot, model capacities, route availability,
and turn pointers. Raw chat transcripts, raw private memory, and credentials
are strictly excluded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from mana_agent.context_cost.models import AccountingSnapshot


@dataclass(frozen=True, slots=True)
class IdentitySessionRelationship:
    """Identity and session context for the current turn."""

    authenticated_user_id: str
    session_id: str
    conversation_id: str
    turn_id: str
    task_id: str = ""
    parent_task_id: str | None = None
    workspace_id: str = ""
    repository_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutionRecoveryState:
    """Active execution, recovery, and lane lifecycle state."""

    active_flow_id: str | None = None
    active_route: str = ""
    lane_id: str = ""
    lane_states: dict[str, Any] = field(default_factory=dict)
    recoverable_task_candidates: tuple[dict[str, Any], ...] = ()
    all_recovery_candidates: tuple[dict[str, Any], ...] = ()
    pending_required_work: bool = False
    pending_checkpoint_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_flow_id": self.active_flow_id,
            "active_route": self.active_route,
            "lane_id": self.lane_id,
            "lane_states": dict(self.lane_states),
            "recoverable_task_candidates": list(self.recoverable_task_candidates),
            "all_recovery_candidates": list(self.all_recovery_candidates),
            "pending_required_work": self.pending_required_work,
            "pending_checkpoint_id": self.pending_checkpoint_id,
        }


@dataclass(frozen=True, slots=True)
class ModelCandidateCapacity:
    """Summary of one candidate model's context capacity and capabilities."""

    model_id: str
    provider: str
    context_window: int
    max_output_tokens: int
    supported_roles: tuple[str, ...]
    supported_tools: tuple[str, ...] = ()
    available: bool = True
    latency_class: str = "standard"
    can_patch: bool = True
    can_verify: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ApprovalState:
    """Pending user, server, and action approvals."""

    pending_server_approvals: tuple[dict[str, Any], ...] = ()
    pending_action_approvals: tuple[dict[str, Any], ...] = ()
    pending_user_approvals: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pending_server_approvals": list(self.pending_server_approvals),
            "pending_action_approvals": list(self.pending_action_approvals),
            "pending_user_approvals": list(self.pending_user_approvals),
        }


@dataclass(frozen=True, slots=True)
class PreviousTurnPointers:
    """Opaque pointers and identifiers for previous turns and related tasks."""

    previous_turn_id: str = ""
    previous_route: str = ""
    previous_task_id: str = ""
    related_task_ids: tuple[str, ...] = ()
    retrieval_hints: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous_turn_id": self.previous_turn_id,
            "previous_route": self.previous_route,
            "previous_task_id": self.previous_task_id,
            "related_task_ids": list(self.related_task_ids),
            "retrieval_hints": list(self.retrieval_hints),
        }


@dataclass(frozen=True, slots=True)
class ConversationContextAvailability:
    """Availability of episodic conversation retrieval for this session."""

    has_history: bool
    available_turns: int
    last_turn_id: str = ""
    retrieval_tool_available: bool = True
    retrieval_token_budget: int = 12000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MemoryAvailability:
    """Availability of durable memory capsules for this session."""

    memory_capsules_enabled: bool
    memory_task_candidates: tuple[dict[str, str], ...] = ()
    available_scopes: tuple[str, ...] = ("private", "project")
    retrieval_tool_available: bool = True
    retrieval_token_budget: int = 4000

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_capsules_enabled": self.memory_capsules_enabled,
            "memory_task_candidates": list(self.memory_task_candidates),
            "available_scopes": list(self.available_scopes),
            "retrieval_tool_available": self.retrieval_tool_available,
            "retrieval_token_budget": self.retrieval_token_budget,
        }


@dataclass(frozen=True, slots=True)
class RoutingExecutionEnvelope:
    """Single canonical routing envelope passed across gateway routing boundaries."""

    user_request: str
    identity: IdentitySessionRelationship
    execution_state: ExecutionRecoveryState
    accounting_snapshot: AccountingSnapshot
    model_candidates: tuple[ModelCandidateCapacity, ...]
    route_availability: tuple[dict[str, Any], ...]
    capabilities_and_tools: tuple[dict[str, Any], ...]
    approval_state: ApprovalState
    artifact_metadata: dict[str, Any]
    previous_turn_pointers: PreviousTurnPointers
    conversation_context_availability: ConversationContextAvailability
    memory_availability: MemoryAvailability

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_request": self.user_request,
            "identity": self.identity.to_dict(),
            "execution_state": self.execution_state.to_dict(),
            "accounting_snapshot": self.accounting_snapshot.as_dict(),
            "model_candidates": [c.to_dict() for c in self.model_candidates],
            "route_availability": [
                r.to_dict() if hasattr(r, "to_dict") else dict(r)
                for r in self.route_availability
            ],
            "capabilities_and_tools": [
                item.to_dict() if hasattr(item, "to_dict") else dict(item)
                for item in self.capabilities_and_tools
            ],
            "approval_state": self.approval_state.to_dict(),
            "artifact_metadata": dict(self.artifact_metadata),
            "previous_turn_pointers": self.previous_turn_pointers.to_dict(),
            "conversation_context_availability": self.conversation_context_availability.to_dict(),
            "memory_availability": self.memory_availability.to_dict(),
        }


def build_routing_execution_envelope(
    *,
    user_request: str,
    identity: IdentitySessionRelationship,
    execution_state: ExecutionRecoveryState,
    accounting_snapshot: AccountingSnapshot,
    model_candidates: tuple[ModelCandidateCapacity, ...] = (),
    route_availability: tuple[dict[str, Any], ...] = (),
    capabilities_and_tools: tuple[dict[str, Any], ...] = (),
    approval_state: ApprovalState | None = None,
    artifact_metadata: dict[str, Any] | None = None,
    previous_turn_pointers: PreviousTurnPointers | None = None,
    conversation_context_availability: ConversationContextAvailability | None = None,
    memory_availability: MemoryAvailability | None = None,
) -> RoutingExecutionEnvelope:
    """Factory helper to construct a valid RoutingExecutionEnvelope."""
    return RoutingExecutionEnvelope(
        user_request=str(user_request or "").strip(),
        identity=identity,
        execution_state=execution_state,
        accounting_snapshot=accounting_snapshot,
        model_candidates=model_candidates,
        route_availability=route_availability,
        capabilities_and_tools=capabilities_and_tools,
        approval_state=approval_state or ApprovalState(),
        artifact_metadata=dict(artifact_metadata or {}),
        previous_turn_pointers=previous_turn_pointers or PreviousTurnPointers(),
        conversation_context_availability=conversation_context_availability
        or ConversationContextAvailability(has_history=False, available_turns=0),
        memory_availability=memory_availability
        or MemoryAvailability(memory_capsules_enabled=False),
    )


__all__ = [
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
