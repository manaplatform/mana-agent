from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mana_agent.telemetry.tokens import TokenUsage


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ChatEvent:
    event_id: str = field(default_factory=lambda: f"evt-{uuid.uuid4().hex}")
    parent_event_id: str | None = None
    session_id: str = ""
    turn_id: str = ""
    agent_id: str | None = "main"
    subagent_id: str | None = None
    step_id: str | None = None
    type: str = "step.updated"
    status: str = "running"
    title: str = ""
    summary: str | None = None
    started_at: str = field(default_factory=utc_now_iso)
    ended_at: str | None = None
    duration_ms: int | None = None
    token_usage: TokenUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.event_id

    @id.setter
    def id(self, value: str) -> None:
        self.event_id = str(value or "").strip() or self.event_id

    @property
    def parent_id(self) -> str | None:
        return self.parent_event_id

    @parent_id.setter
    def parent_id(self, value: str | None) -> None:
        self.parent_event_id = str(value or "").strip() or None

    @property
    def timestamp(self) -> str:
        return self.started_at

    @timestamp.setter
    def timestamp(self, value: str) -> None:
        self.started_at = str(value or "").strip() or utc_now_iso()

    @property
    def kind(self) -> str:
        return normalize_event_kind(self.metadata.get("kind") or self.type)

    @kind.setter
    def kind(self, value: str) -> None:
        kind = str(value or "").strip()
        if kind:
            self.metadata["kind"] = kind
            self.type = _kind_to_event_type(kind)

    @property
    def details(self) -> dict[str, Any]:
        return self.metadata

    @details.setter
    def details(self, value: dict[str, Any] | None) -> None:
        self.metadata = dict(value or {})

    @property
    def message(self) -> str:
        return self.summary or ""

    @message.setter
    def message(self, value: str | None) -> None:
        self.summary = str(value or "") or None

    def finish(self, *, status: str = "success", message: str | None = None) -> "ChatEvent":
        self.status = normalize_event_status(status)
        if message is not None:
            self.summary = message
        self.ended_at = utc_now_iso()
        try:
            started = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
            ended = datetime.fromisoformat(self.ended_at.replace("Z", "+00:00"))
            self.duration_ms = int(max(0.0, (ended - started).total_seconds() * 1000))
        except Exception:
            self.duration_ms = 0
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "id": self.event_id,
            "parent_event_id": self.parent_event_id,
            "parent_id": self.parent_event_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "agent_id": self.agent_id,
            "subagent_id": self.subagent_id,
            "step_id": self.step_id,
            "type": self.type,
            "kind": self.kind,
            "status": self.status,
            "title": self.title,
            "summary": self.summary,
            "started_at": self.started_at,
            "timestamp": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "token_usage": self.token_usage.as_dict() if self.token_usage is not None else None,
            "metadata": dict(self.metadata),
            "details": dict(self.metadata),
        }


AgentEvent = ChatEvent


_TYPE_TO_KIND = {
    "session.started": "session",
    "session.ready": "session",
    "turn.started": "user_request",
    "message.accepted": "user_request",
    "turn.finished": "response",
    "turn.cancelled": "response",
    "assistant.started": "response",
    "assistant.delta": "response",
    "agent.decision": "reasoning",
    "agent.routing": "routing",
    "agent.planning": "plan_step",
    "step.started": "plan_step",
    "step.finished": "plan_step",
    "plan.created": "plan_step",
    "tool.started": "tool",
    "tool.finished": "tool",
    "tool.failed": "tool",
    "tool.cancelled": "tool",
    "tool.timeout": "tool",
    "tool.progress": "tool",
    "tool.stdout": "tool",
    "tool.stderr": "tool",
    "subagent.started": "subagent",
    "subagent.created": "subagent",
    "subagent.finished": "subagent",
    "subagent.delta": "subagent",
    "task.started": "plan_step",
    "task.progress": "plan_step",
    "task.finished": "plan_step",
    "verification.started": "plan_step",
    "verification.finished": "plan_step",
    "review.started": "plan_step",
    "review.finished": "plan_step",
    "log": "reasoning",
    "log.info": "reasoning",
    "log.warning": "error",
    "log.error": "error",
    "file.read": "tool",
    "file.changed": "tool",
    "patch.applied": "tool",
    "test.started": "tool",
    "test.finished": "tool",
    "approval.required": "routing",
    "warning": "error",
    "error": "error",
}

_RAW_KIND_TO_TYPE = {
    "session_started": "session.started",
    "session_ready": "session.ready",
    "user_message": "turn.started",
    "assistant_message_done": "turn.finished",
    "assistant_message_start": "assistant.started",
    "assistant_delta": "assistant.delta",
    "thinking_summary": "agent.decision",
    "plan_step_started": "step.started",
    "plan_step_done": "step.finished",
    "plan_created": "plan.created",
    "tool_started": "tool.started",
    "tool_done": "tool.finished",
    "tool_stdout": "tool.stdout",
    "tool_stderr": "tool.stderr",
    "subagent_started": "subagent.started",
    "subagent_created": "subagent.created",
    "subagent_done": "subagent.finished",
    "subagent_delta": "subagent.delta",
    "file_read": "file.read",
    "file_changed": "file.changed",
    "patch_applied": "patch.applied",
    "test_started": "test.started",
    "test_done": "test.finished",
    "approval_required": "approval.required",
    "SessionStarted": "session.started",
    "UserMessageQueued": "turn.started",
    "RoutingStarted": "agent.routing",
    "RoutingCompleted": "agent.routing",
    "PlanStarted": "step.started",
    "PlanCompleted": "step.finished",
    "ReasoningStarted": "agent.decision",
    "ToolStarted": "tool.started",
    "ToolCompleted": "tool.finished",
    "ToolFailed": "tool.failed",
    "SubagentCreated": "subagent.created",
    "SubagentStarted": "subagent.started",
    "SubagentCompleted": "subagent.finished",
    "AssistantDelta": "assistant.delta",
    "AssistantFinal": "turn.finished",
    "ResponseRendered": "turn.finished",
}

_KIND_TO_TYPE = {value: key for key, value in _TYPE_TO_KIND.items()}
_KIND_TO_TYPE.update(
    {
        **_RAW_KIND_TO_TYPE,
        "session": "session.started",
        "user_request": "turn.started",
        "routing": "agent.routing",
        "plan_step": "step.started",
        "reasoning": "agent.decision",
        "tool": "tool.started",
        "subagent": "subagent.started",
        "response": "turn.finished",
        "error": "error",
    }
)

_NORMALIZED_KINDS = {
    "session",
    "user_request",
    "routing",
    "plan_step",
    "reasoning",
    "tool",
    "subagent",
    "response",
    "error",
}

_STATUS_ALIASES = {
    "done": "success",
    "complete": "success",
    "completed": "success",
    "ok": "success",
    "failure": "failed",
    "error": "failed",
    "warning": "failed",
    "pending": "queued",
    "blocked": "waiting",
}


def _event_type_to_kind(event_type: str) -> str:
    return normalize_event_kind(event_type)


def normalize_event_kind(value: str) -> str:
    normalized = str(value or "").strip()
    if normalized in _NORMALIZED_KINDS:
        return normalized
    if normalized in _TYPE_TO_KIND:
        return _TYPE_TO_KIND[normalized]
    if normalized in _RAW_KIND_TO_TYPE:
        return _TYPE_TO_KIND.get(_RAW_KIND_TO_TYPE[normalized], "reasoning")
    if normalized.startswith(("tool.", "command.", "file.", "patch.", "test.")):
        return "tool"
    if normalized.startswith(("assistant.", "turn.")):
        return "response"
    if normalized.startswith(("agent.", "reasoning.")):
        return "reasoning"
    if normalized.startswith("subagent."):
        return "subagent"
    if normalized.startswith(("task.", "plan.", "verification.", "review.", "step.")):
        return "plan_step"
    if normalized.startswith("session."):
        return "session"
    if normalized.startswith("log.") or normalized in {"warning", "error"}:
        return "error" if normalized in {"warning", "error", "log.warning", "log.error"} else "reasoning"
    return _TYPE_TO_KIND.get(normalized.replace("_", "."), "reasoning")


def normalize_event_status(value: str) -> str:
    normalized = str(value or "running").strip().lower().replace(" ", "_")
    normalized = _STATUS_ALIASES.get(normalized, normalized)
    return normalized if normalized in {
        "queued",
        "running",
        "success",
        "failed",
        "cancelled",
        "timed_out",
        "skipped",
        "waiting",
    } else "running"


def _kind_to_event_type(kind: str) -> str:
    normalized = str(kind or "").strip()
    if normalized in _KIND_TO_TYPE:
        return _KIND_TO_TYPE[normalized]
    return normalized.replace("_", ".") or "step.updated"


def make_event(
    event_type: str,
    *,
    title: str,
    message: str = "",
    status: str = "running",
    session_id: str = "",
    turn_id: str = "",
    agent_id: str | None = "main",
    subagent_id: str | None = None,
    step_id: str | None = None,
    parent_event_id: str | None = None,
    token_usage: TokenUsage | None = None,
    metadata: dict[str, Any] | None = None,
) -> ChatEvent:
    event_type_text = str(event_type or "")
    normalized_type = _kind_to_event_type(event_type_text) if "." not in event_type_text else event_type_text
    normalized_kind = normalize_event_kind(normalized_type)
    event_metadata = dict(metadata or {})
    if event_type_text and event_type_text != normalized_kind:
        event_metadata.setdefault("raw_kind", event_type_text.replace(".", "_"))
    event_metadata.setdefault("kind", normalized_kind)
    return ChatEvent(
        parent_event_id=parent_event_id,
        session_id=session_id,
        turn_id=turn_id,
        agent_id=agent_id,
        subagent_id=subagent_id,
        step_id=step_id,
        type=normalized_type,
        status=normalize_event_status(status),
        title=title,
        summary=message or None,
        token_usage=token_usage,
        metadata=event_metadata,
    )


def derive_tool_action_summary(
    tool_name: str,
    args: Any = None,
    *,
    result: Any = None,
    max_len: int = 96,
) -> str:
    """Return a concise, human-readable action summary for a tool invocation.

    Pure function. Inspects common arg shapes (dict or JSON str) for path/query/command etc.
    Falls back to a compact tool + arg preview. Used for immediate live display.
    """
    name = str(tool_name or "tool").strip() or "tool"
    text_parts: list[str] = []

    def _add_candidate(val: Any) -> None:
        if val is None:
            return
        s = str(val).strip()
        if not s:
            return
        # Redact obvious secrets quickly (best effort, non-crypto)
        low = s.lower()
        if any(k in low for k in ("password", "secret", "token", "apikey", "api_key")) and len(s) > 8:
            s = s[:4] + "…[redacted]"
        text_parts.append(s)

    if isinstance(args, dict):
        for key in ("path", "file_path", "file", "target", "query", "q", "command", "cmd", "pattern", "url", "glob", "name"):
            if key in args:
                _add_candidate(args[key])
                break
        else:
            # fallback to short json of first few items
            try:
                import json as _json
                short = _json.dumps({k: args[k] for k in list(args)[:2]}, ensure_ascii=False, separators=(",", ":"))
                _add_candidate(short)
            except Exception:
                pass
    elif isinstance(args, str):
        s = args.strip().replace("\n", " ")
        if s:
            # try parse json for structured
            parsed = None
            try:
                import json as _json
                parsed = _json.loads(s)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                for key in ("path", "file_path", "query", "command", "url"):
                    if key in parsed:
                        _add_candidate(parsed[key])
                        break
                else:
                    _add_candidate(s[:72])
            else:
                _add_candidate(s[:72])
    else:
        # try to pull from result or generic
        pass

    if not text_parts and result is not None:
        try:
            r = str(result).strip()[:60]
            if r:
                text_parts.append(r)
        except Exception:
            pass

    summary = (name + (": " + " ".join(text_parts) if text_parts else "")).strip()
    if len(summary) > max_len:
        summary = summary[: max_len - 1].rstrip() + "…"
    return summary or name


def make_tool_event(
    phase: str,
    tool_name: str,
    *,
    action_summary: str = "",
    args: Any = None,
    status: str = "running",
    session_id: str = "",
    turn_id: str = "",
    agent_id: str | None = "main",
    subagent_id: str | None = None,
    step_id: str | None = None,
    parent_event_id: str | None = None,
    event_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    duration_ms: int | None = None,
    result_summary: str = "",
) -> ChatEvent:
    """Create a normalized tool ChatEvent with stable id for start + update.

    phase: "start" | "end" | "error" (maps to tool.started / tool.finished / tool.failed)
    The caller is responsible for supplying the same event_id for start and its terminal update.
    A concise action_summary (or auto-derived) is placed in summary and metadata.
    """
    phase = str(phase or "").strip().lower()
    if phase in {"start", "started"}:
        etype = "tool.started"
        default_status = "running"
        default_msg = action_summary or "Tool started."
    elif phase in {"end", "finished", "complete", "success"}:
        etype = "tool.finished"
        default_status = "success"
        default_msg = result_summary or action_summary or "Tool finished."
    else:
        etype = "tool.failed"
        default_status = "failed"
        default_msg = result_summary or action_summary or "Tool failed."

    base_meta: dict[str, Any] = {
        "tool_name": str(tool_name or "tool"),
        "args_summary": action_summary or (derive_tool_action_summary(tool_name, args) if args is not None else ""),
    }
    if result_summary:
        base_meta["result_summary"] = result_summary
    if metadata:
        base_meta.update(metadata)

    ev = make_event(
        etype,
        title=str(tool_name or "tool"),
        message=default_msg,
        status=status or default_status,
        session_id=session_id,
        turn_id=turn_id,
        agent_id=agent_id,
        subagent_id=subagent_id,
        step_id=step_id,
        parent_event_id=parent_event_id,
        metadata=base_meta,
    )
    if event_id:
        ev.event_id = event_id
        ev.id = event_id
    if duration_ms is not None:
        ev.duration_ms = int(duration_ms)
        if ev.status in {"success", "failed"} and not ev.ended_at:
            ev.ended_at = utc_now_iso()
    # ensure summary carries the human action for renderers
    if action_summary and not ev.summary:
        ev.summary = action_summary
    return ev
