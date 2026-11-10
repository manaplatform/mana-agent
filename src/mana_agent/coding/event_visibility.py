"""Protocol-based visibility policy for coding execution events.

Correctness depends on normalized event type and execution phase, not on
matching provider-specific free-form text. Raw model drafts remain available
for internal traces; only typed progress and validated terminal results may
reach user-facing chat surfaces.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class EventVisibility(str, Enum):
    """Who may observe a coding event on user-facing surfaces."""

    # Full fidelity for audit/telemetry only (not chat/response stream).
    INTERNAL = "internal"
    # Safe typed progress (no model prose): commands, files, phase markers.
    PROGRESS = "progress"
    # Validated terminal outcome after execution evidence is collected.
    TERMINAL = "terminal"


class EventSemanticKind(str, Enum):
    """Normalized semantic category at the coding protocol boundary."""

    LIFECYCLE = "lifecycle"
    ASSISTANT_GENERATION = "assistant_generation"
    REASONING = "reasoning"
    TOOL_EXECUTION = "tool_execution"
    REPOSITORY_MUTATION = "repository_mutation"
    COMMAND = "command"
    VERIFICATION = "verification"
    PLAN = "plan"
    USAGE = "usage"
    WARNING = "warning"
    ERROR = "error"
    PROVIDER = "provider"
    CONTEXT = "context"
    UNKNOWN = "unknown"


# Event types that must never stream model-generated prose to chat.
_ASSISTANT_GENERATION_TYPES = frozenset(
    {
        "assistant.delta",
        "assistant.started",
        "assistant.message",
        "assistant.completed",
    }
)

_REASONING_PREFIXES = ("reasoning.",)

_PROGRESS_TYPES = frozenset(
    {
        "backend.selected",
        "turn.starting",
        "turn.started",
        "turn.finalizing",
        "turn.completed",
        "turn.finished",
        "turn.cancelled",
        "turn.resume_requested",
        "server.waiting_approval",
        "server.approval_decided",
        "api.waiting_approval",
        "api.approval_decided",
        "command.started",
        "command.completed",
        "command.output",
        "command.delta",
        "file.changed",
        "patch.applied",
        "test.started",
        "test.completed",
        "plan.created",
        "plan.step.started",
        "plan.step.completed",
        "tool.call.started",
        "tool.call.completed",
        "context.budget",
    }
)

_TERMINAL_TYPES = frozenset(
    {
        "turn.completed",
        "turn.finished",
        "turn.cancelled",
        "error",
        "coding.terminal",
    }
)

_MUTATION_ITEM_MARKERS = (
    "filechange",
    "file_change",
    "applypatch",
    "apply_patch",
    "patchapplication",
    "patch_application",
    "patch",
)

_COMMAND_ITEM_MARKERS = (
    "command",
    "shell",
    "exec",
)

_PLAN_ITEM_MARKERS = ("plan",)

_TEST_ITEM_MARKERS = ("test",)


def semantic_kind_for_event_type(event_type: str, *, tool_name: str = "") -> EventSemanticKind:
    """Map a normalized Mana coding event type to a semantic category."""
    et = str(event_type or "").strip().lower()
    tool = str(tool_name or "").strip().lower()

    if et == "agent.summary":
        return EventSemanticKind.LIFECYCLE
    if et.startswith(("agent.", "loop.", "task.", "subagent.", "worker.", "step.", "investigation.", "discovery.", "recovery.", "activity.", "model.")):
        return EventSemanticKind.PLAN if et == "agent.planning" else EventSemanticKind.LIFECYCLE
    if et.startswith(("tool.", "search.")) and not et.startswith("tool.call."):
        return EventSemanticKind.TOOL_EXECUTION
    if et.startswith(("verification.", "review.")):
        return EventSemanticKind.VERIFICATION
    if et.startswith(("action.approval.", "approval.", "inbox.", "computer.", "server.", "api.")):
        return EventSemanticKind.LIFECYCLE

    if (
        et in {"user.message", "turn.input", "user.input", "message.accepted"}
        or "usermessage" in tool
        or "user_message" in tool
    ):
        return EventSemanticKind.LIFECYCLE
    if (
        et in {"error", "systemerror", "system_error"}
        or "systemerror" in tool
        or "system_error" in tool
    ):
        return EventSemanticKind.ERROR
    if et in _ASSISTANT_GENERATION_TYPES or "agentmessage" in tool or "agent_message" in tool:
        return EventSemanticKind.ASSISTANT_GENERATION
    if et.startswith(_REASONING_PREFIXES) or "reasoning" in tool:
        return EventSemanticKind.REASONING
    if et in {"warning", "provider.warning"}:
        return EventSemanticKind.WARNING
    if et.startswith("steering."):
        return EventSemanticKind.LIFECYCLE
    if et.startswith("usage.") or et == "usage.update":
        return EventSemanticKind.USAGE
    if et.startswith("context.") or et.startswith("budget.") or et.startswith("cost."):
        return EventSemanticKind.CONTEXT
    if et.startswith("plan.") or any(marker in tool for marker in _PLAN_ITEM_MARKERS):
        return EventSemanticKind.PLAN
    if et.startswith("test.") or any(marker in tool for marker in _TEST_ITEM_MARKERS):
        return EventSemanticKind.VERIFICATION
    if et in {"file.changed", "patch.applied"} or any(
        marker in tool for marker in _MUTATION_ITEM_MARKERS
    ):
        return EventSemanticKind.REPOSITORY_MUTATION
    if et.startswith("command.") or any(marker in tool for marker in _COMMAND_ITEM_MARKERS):
        return EventSemanticKind.COMMAND
    if et.startswith("tool.call."):
        # Only treat as tool execution when the item is not a message/reasoning item.
        if "usermessage" in tool or "user_message" in tool:
            return EventSemanticKind.LIFECYCLE
        if "agentmessage" in tool or "agent_message" in tool:
            return EventSemanticKind.ASSISTANT_GENERATION
        if "systemerror" in tool or "system_error" in tool:
            return EventSemanticKind.ERROR
        if "reasoning" in tool:
            return EventSemanticKind.REASONING
        if any(marker in tool for marker in _MUTATION_ITEM_MARKERS):
            return EventSemanticKind.REPOSITORY_MUTATION
        if any(marker in tool for marker in _COMMAND_ITEM_MARKERS):
            return EventSemanticKind.COMMAND
        return EventSemanticKind.TOOL_EXECUTION
    if et in {
        "backend.selected",
        "turn.starting",
        "turn.started",
        "turn.finalizing",
        "turn.completed",
        "turn.finished",
        "turn.cancelled",
        "turn.resume_requested",
        "server.waiting_approval",
        "server.approval_decided",
        "api.waiting_approval",
        "api.approval_decided",
        "coding.terminal",
        "task.created",
        "task.scheduled",
        "worker.claimed",
        "task.completed",
        "task.finished",
    }:
        return EventSemanticKind.LIFECYCLE
    if et.startswith("provider."):
        return EventSemanticKind.PROVIDER
    return EventSemanticKind.UNKNOWN


def visibility_for_semantic_kind(
    kind: EventSemanticKind,
    *,
    event_type: str = "",
    requires_repository_write: bool = True,
) -> EventVisibility:
    """Decide user-surface visibility from semantic kind (not text content)."""
    et = str(event_type or "").strip().lower()
    if kind is EventSemanticKind.ASSISTANT_GENERATION:
        return EventVisibility.INTERNAL
    if kind is EventSemanticKind.REASONING:
        return EventVisibility.INTERNAL
    if kind is EventSemanticKind.USAGE:
        return EventVisibility.INTERNAL
    if kind is EventSemanticKind.PROVIDER and et not in {"provider.warning"}:
        return EventVisibility.INTERNAL
    if et in _TERMINAL_TYPES and et != "turn.started":
        # turn.completed is both progress marker and contributes to terminal assembly;
        # progress is enough for live UI — final answer is separate.
        return EventVisibility.PROGRESS
    if et in _PROGRESS_TYPES or kind in {
        EventSemanticKind.LIFECYCLE,
        EventSemanticKind.TOOL_EXECUTION,
        EventSemanticKind.REPOSITORY_MUTATION,
        EventSemanticKind.COMMAND,
        EventSemanticKind.VERIFICATION,
        EventSemanticKind.PLAN,
        EventSemanticKind.CONTEXT,
        EventSemanticKind.WARNING,
        EventSemanticKind.ERROR,
    }:
        return EventVisibility.PROGRESS
    # Plan/read-only: still never stream assistant generation mid-turn.
    _ = requires_repository_write
    return EventVisibility.INTERNAL


def classify_coding_event(
    event_type: str,
    *,
    tool_name: str = "",
    requires_repository_write: bool = True,
) -> tuple[EventSemanticKind, EventVisibility]:
    kind = semantic_kind_for_event_type(event_type, tool_name=tool_name)
    visibility = visibility_for_semantic_kind(
        kind,
        event_type=event_type,
        requires_repository_write=requires_repository_write,
    )
    return kind, visibility


def is_user_publishable(visibility: EventVisibility | str) -> bool:
    value = visibility.value if isinstance(visibility, EventVisibility) else str(visibility)
    return value in {EventVisibility.PROGRESS.value, EventVisibility.TERMINAL.value}


def progress_event_payload(event: Any) -> dict[str, Any]:
    """Return a safe user-facing progress projection (no raw model prose)."""
    from mana_agent.chat.activity import project_activity, normalize_activity

    projected = project_activity(event)
    if projected is not None:
        return projected
    raw_event = normalize_activity(event)
    return {
        "event_id": raw_event.get("event_id"), "event_type": raw_event["type"],
        "type": raw_event["type"], "visibility": "internal", "summary": "",
        "output_preview": "", "title": "", "status": raw_event.get("status", "running"),
    }


__all__ = [
    "EventSemanticKind",
    "EventVisibility",
    "classify_coding_event",
    "is_user_publishable",
    "progress_event_payload",
    "semantic_kind_for_event_type",
    "visibility_for_semantic_kind",
]
