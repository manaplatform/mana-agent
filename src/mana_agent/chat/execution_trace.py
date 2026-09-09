"""Normalized runtime execution trace model for turn-level execution stats.

Provides a unified step-by-step trace representation shared by TUI, Dashboard,
and testing suites. Each chat turn tracks its chronological progression
(routing, context preparation, coding, tools, model calls, completion, failure)
with idempotent step updates and clean running-state transitions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExecutionStep:
    """A single execution phase or operation within a chat turn."""

    step_id: str
    turn_id: str
    phase: str  # e.g. "routing", "context", "coding", "tool", "model", "completion", "failure", etc.
    title: str
    status: str = "running"  # "running", "completed", "failed", "cancelled"
    detail: str = ""
    started_at: str = field(default_factory=_utc_now_iso)
    ended_at: str | None = None
    duration_ms: int | None = None
    sub_events: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(self, *, status: str = "completed", detail: str = "") -> None:
        if status in {"success", "completed", "done"}:
            self.status = "completed"
        elif status in {"failed", "error"}:
            self.status = "failed"
        elif status in {"cancelled", "interrupted"}:
            self.status = "cancelled"
        else:
            self.status = status

        if detail:
            self.detail = detail
        self.ended_at = _utc_now_iso()
        if self.duration_ms is None:
            try:
                started = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
                ended = datetime.fromisoformat(self.ended_at.replace("Z", "+00:00"))
                self.duration_ms = max(0, int((ended - started).total_seconds() * 1000))
            except Exception:
                self.duration_ms = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "turn_id": self.turn_id,
            "phase": self.phase,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "sub_events": list(self.sub_events),
            "metadata": dict(self.metadata),
        }


SEARCH_TOOLS = {
    "web_search",
    "github_search",
    "repo_search",
    "code_search",
    "find_by_name",
    "grep_search",
    "search_files",
    "search",
    "document_query",
}


def classify_runtime_phase(event_type: str, metadata: dict[str, Any] | None = None) -> tuple[str, str]:
    """Map a runtime event type to (phase, title).

    Generic fallback derives phase and title cleanly for future phases.
    """
    et = str(event_type or "").strip().lower()
    meta = dict(metadata or {})
    tool_name = str(meta.get("tool_name") or meta.get("name") or "").strip()

    # Routing
    if et in {
        "routing_started",
        "routing_envelope_created",
        "agent.routing",
        "agent.decision",
        "entry_route_decided",
        "routing_completed",
        "routing_failed",
        "gateway.entry_route",
        "followup_classified",
        "route_selected",
    }:
        return "routing", "Routing"

    # Context Preparation
    if et in {
        "context_preparation_started",
        "context_retrieval_started",
        "context_retrieval_completed",
        "context_retrieval",
        "context.budget",
        "context.compacted",
        "context.capabilities_loaded",
        "context.capabilities_unloaded",
        "workspace.repository_initialized",
    } or et.startswith(("context.", "budget.", "cost.")):
        return "context", "Context preparation"

    # Search
    if (
        et in {"search_started", "search_completed", "search_failed", "search"}
        or et.startswith("search.")
        or tool_name in SEARCH_TOOLS
        or (bool(tool_name) and "search" in tool_name.lower())
        or meta.get("route") in {"search", "repository", "github"}
    ):
        return "search", "Searching"

    # Coding / Codex Backend
    if (
        et in {
            "coding_started",
            "coding.terminal",
            "coding.progress",
            "coding",
        }
        or et.startswith(("command.", "patch.", "file."))
        or (
            meta.get("backend") == "codex"
            and not et.startswith(("turn.", "error", "routing", "tool", "search", "model"))
        )
    ):
        backend = str(meta.get("backend") or "coding")
        title = "Codex" if backend == "codex" else "Coding"
        return "coding", title

    # Tool Execution (non-search)
    if et.startswith("tool.") or et in {"tool_started", "tool_finished", "tool_failed", "tool_cancelled"}:
        title = f"Tool: {tool_name}" if tool_name else "Tool execution"
        return "tool", title

    # Model Execution
    if et in {
        "model_execution_started",
        "model.started",
        "model.completed",
        "assistant.started",
        "assistant.delta",
        "thinking_started",
    }:
        return "model", "Model execution"

    # Completion
    if et in {
        "turn.completed",
        "turn.finished",
        "conversation_response_created",
        "assistant.completed",
    }:
        return "completion", "Completed"

    # Failure / Cancellation
    if et in {"error", "turn.cancelled", "turn.timeout", "tool.timeout", "cancelled"}:
        return "failure", "Failed" if "cancelled" not in et else "Cancelled"

    # Generic Fallback: supports future runtime phases automatically
    parts = et.replace("_", ".").split(".")
    phase = parts[0] if parts else "activity"
    title = phase.replace("_", " ").title()
    return phase, title


class ExecutionTrace:
    """Maintains the chronological execution steps for a single chat turn."""

    def __init__(self, turn_id: str) -> None:
        self.turn_id = str(turn_id or "").strip()
        self.steps: list[ExecutionStep] = []
        self._step_by_id: dict[str, ExecutionStep] = {}
        self._step_ids_by_phase: dict[str, str] = {}
        self.active_step_id: str | None = None
        self.is_completed: bool = False
        self.is_failed: bool = False
        self.is_cancelled: bool = False
        self._seen_sub_event_ids: set[str] = set()

    @property
    def active_step(self) -> ExecutionStep | None:
        if self.active_step_id:
            return self._step_by_id.get(self.active_step_id)
        return None

    def get_step(self, step_id: str) -> ExecutionStep | None:
        return self._step_by_id.get(step_id)

    def apply_event(self, event: dict[str, Any]) -> ExecutionStep | None:
        """Apply an event to this turn's trace idempotently."""
        ev = dict(event)
        event_type = str(ev.get("event_type") or ev.get("type") or "").strip()
        raw_status = str(ev.get("status") or "running").strip().lower()
        title = str(ev.get("title") or "").strip()
        detail = str(ev.get("detail") or ev.get("summary") or ev.get("output_preview") or ev.get("message") or ev.get("error") or "").strip()
        meta = dict(ev.get("metadata") or ev.get("details") or ev.get("payload") or {})
        duration_ms = ev.get("duration_ms")
        event_id = str(ev.get("event_id") or ev.get("id") or "").strip()

        phase, default_title = classify_runtime_phase(event_type, meta)
        step_title = title or default_title

        # Determine step ID
        if phase == "tool":
            tool_call_id = str(ev.get("tool_call_id") or meta.get("tool_call_id") or meta.get("call_id") or event_id or "").strip()
            step_id = f"{self.turn_id}:tool:{tool_call_id}" if tool_call_id else f"{self.turn_id}:tool:{event_type}"
        elif phase in {"routing", "context", "coding", "model", "completion", "failure"}:
            step_id = f"{self.turn_id}:{phase}"
        else:
            step_id = f"{self.turn_id}:{phase}"

        # Status normalization
        is_terminal_step = raw_status in {"success", "completed", "done", "failed", "error", "cancelled"}

        # If turn finished or failed, mark previous running steps
        if phase in {"completion", "failure"}:
            if phase == "completion":
                self.is_completed = True
            elif raw_status in {"cancelled", "interrupted"}:
                self.is_cancelled = True
            else:
                self.is_failed = True

            # Clean up any stale running step
            for st in self.steps:
                if st.status == "running":
                    st.finish(status="completed" if self.is_completed else ("cancelled" if self.is_cancelled else "failed"))
            self.active_step_id = None

        existing = self._step_by_id.get(step_id)

        if existing is not None:
            # Update existing step in place
            if is_terminal_step:
                existing.finish(status=raw_status, detail=detail or existing.detail)
                if duration_ms is not None:
                    existing.duration_ms = int(duration_ms)
                if self.active_step_id == step_id:
                    self.active_step_id = None
            else:
                if detail:
                    existing.detail = detail
                if step_title and existing.title == default_title:
                    existing.title = step_title

            # Collect sub-event for coding or tool
            if event_id and event_id not in self._seen_sub_event_ids:
                self._seen_sub_event_ids.add(event_id)
                existing.sub_events.append(ev)
            elif not event_id and ev:
                existing.sub_events.append(ev)

            return existing

        # If a previous step of a different phase was still marked running, complete it when new phase starts
        if self.active_step_id and self.active_step_id in self._step_by_id:
            active_step = self._step_by_id[self.active_step_id]
            if active_step.phase != phase and active_step.status == "running":
                active_step.finish(status="completed")

        # Create new step
        step = ExecutionStep(
            step_id=step_id,
            turn_id=self.turn_id,
            phase=phase,
            title=step_title,
            status="completed" if is_terminal_step else "running",
            detail=detail,
            metadata=meta,
        )
        if is_terminal_step:
            step.finish(status=raw_status, detail=detail)
            if duration_ms is not None:
                step.duration_ms = int(duration_ms)
        else:
            self.active_step_id = step_id

        if event_id:
            self._seen_sub_event_ids.add(event_id)
        if ev:
            step.sub_events.append(ev)

        self.steps.append(step)
        self._step_by_id[step_id] = step
        self._step_ids_by_phase[phase] = step_id
        return step

    def mark_failed(self, error: str = "Execution failed", *, cancelled: bool = False) -> None:
        """Clean up active running steps on failure, timeout, or cancellation."""
        status = "cancelled" if cancelled else "failed"
        if cancelled:
            self.is_cancelled = True
        else:
            self.is_failed = True

        for step in self.steps:
            if step.status == "running":
                step.finish(status=status, detail=error)
        self.active_step_id = None


__all__ = [
    "ExecutionStep",
    "ExecutionTrace",
    "classify_runtime_phase",
]
