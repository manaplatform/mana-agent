"""Public activity projection. This module neither decides nor executes work.

Provider reasoning is private unless a trusted adapter explicitly emits a
completed public summary. In particular, a Responses-shaped bridge is not
evidence that its ``reasoning_content`` is safe to disclose.
"""
from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from urllib.parse import urlsplit
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

from mana_agent.utils.redaction import redact_secrets


class ActivityVisibility(str, Enum):
    MINIMAL = "minimal"
    NORMAL = "normal"
    VERBOSE = "verbose"


class ActivityMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    task_id: str = ""
    root_task_id: str = ""
    parent_task_id: str = ""
    parent_agent_id: str = ""
    attempt_id: str = ""
    attempt_generation: int = Field(default=0, ge=0)
    operation_id: str = ""
    producer_sequence: int = Field(default=0, ge=0)
    execution_mode: str = ""
    summary_source: Literal["observable", "agent", "provider"] = "observable"
    evidence_status: Literal["attempted", "executed", "observed", "verified", "unverified"] = "unverified"


TERMINAL = frozenset({"success", "completed", "done", "failed", "error", "cancelled", "interrupted", "rejected", "superseded", "unverified", "failure", "timed_out", "timeout", "skipped", "not_run"})
_PRIVATE_PREFIXES = ("reasoning.", "provider.", "usage.")
_LEGACY_PUBLIC = frozenset({
    "routing_started", "routing_completed", "routing_failed", "routing_envelope_created",
    "entry_route_decided", "gateway.entry_route", "followup_classified", "route_selected",
    "coding_started", "user_turn_received", "conversation_response_created",
    "context_preparation_started", "context_retrieval_started", "context_retrieval_completed",
    "context_retrieval", "model_execution_started", "search_started", "search_completed", "search_failed",
})
_PUBLIC_PREFIXES = (
    "agent.", "task.", "subagent.", "worker.", "loop.", "step.", "plan.",
    "tool.", "command.", "file.", "patch.", "test.", "verification.",
    "review.", "approval.", "action.", "steering.", "turn.",
    "backend.", "context.", "cost.", "budget.", "batch.", "parallel.",
    "model.", "workspace.", "search.", "investigation.", "discovery.", "recovery.", "activity.",
    "api.", "server.", "computer.", "inbox.", "routing.", "execution.", "log.",
)
_IDENTITY_FIELDS = (
    "event_id", "id", "parent_event_id", "parent_id", "conversation_id", "session_id",
    "repository_id", "workspace_id", "execution_id", "turn_id", "task_id", "root_task_id",
    "parent_task_id", "agent_id", "subagent_id", "parent_agent_id", "agent_role", "step_id",
    "attempt_id", "attempt_generation", "operation_id", "producer_sequence", "sequence",
    "backend", "model", "thread_id", "execution_mode", "started_at", "timestamp",
    "created_at", "ended_at", "duration_ms", "tool_call_id", "call_id", "request_id",
)
_PUBLIC_METADATA = frozenset(_IDENTITY_FIELDS) | frozenset({
    "phase", "tool_name", "name", "path", "file_path", "filePath", "command", "changed_files",
    "status", "kind", "category", "visibility", "semantic_kind", "summary_source", "evidence_status",
    "decision_id", "public_summary", "reason", "progress", "result_summary", "verification_status",
    "goal_satisfied", "verification_passed", "commands_run", "tests_run", "tests_passed",
    "permission_request_id", "approval_request_id", "inbox_item_id", "action_id", "decision",
    "expires_at", "permission_scope", "preview", "approval_display", "server_approval", "api_approval",
    "transactional_action_approval", "auto_approved", "action_digest", "preview_digest",
    "scope", "target_id", "instruction", "text", "steering_sequence", "applied_at", "rejection_reason",
    "sub_operation", "indent", "group", "operation_group", "active_concurrency", "count",
    "batch_id", "total_requests", "completed_requests", "completed", "total", "retry_count",
    "reason_code", "recovery_reason", "resume_checkpoint_id", "checkpoint_id", "last_checkpoint_id",
    "provider", "transport", "summary_trusted", "summary_complete", "summary_index", "item_id",
    "role", "message_id", "client_message_id", "content", "attachments", "summary", "resumed",
    "used_tokens", "context_window", "utilization_ratio", "breakdown", "cumulative_cost",
    "remaining_cost", "estimated", "input_tokens", "output_tokens", "total_tokens", "token_usage",
    "model_level", "resolved_model", "requested_by_agent_id", "queue_job_id", "delegation_path",
    "replayable", "error", "output_preview", "dropped", "source_sequence", "execution_supervisor",
})
_FORBIDDEN = frozenset({
    "payload", "raw", "raw_payload", "instructions", "system_prompt", "developer_prompt", "prompt",
    "messages", "reasoning", "reasoning_content", "reasoning_text", "encrypted_content",
    "lease_token", "lease_token_hash", "protected_context", "environment", "env", "arguments", "args",
    "raw_provider_payload", "response_body", "reasoning_details",
})
_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_CREDENTIAL = re.compile(r"(?i)\b(password|api[_-]?key|access[_-]?token|refresh[_-]?token|secret|authorization|cookie)\s*[:=]\s*([^\s,;]+)")


def safe_text(value: Any, limit: int = 4000, *, protected: Iterable[str] = ()) -> str:
    text = _CONTROL.sub("", _ANSI.sub("", str(value or "")))
    for secret in protected:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = _CREDENTIAL.sub(r"\1=[REDACTED]", str(redact_secrets(text)))
    return text if len(text) <= limit else text[:limit] + "…"


def _clean(value: Any, *, protected: Iterable[str] = (), depth: int = 0) -> Any:
    if depth > 6:
        return "[omitted]"
    if isinstance(value, dict):
        return {str(k): _clean(v, protected=protected, depth=depth + 1)
                for k, v in list(value.items())[:100] if str(k).lower() not in _FORBIDDEN}
    if isinstance(value, (list, tuple)):
        return [_clean(v, protected=protected, depth=depth + 1) for v in value[:100]]
    if isinstance(value, str):
        return safe_text(value, protected=protected)
    return value


def sanitize_legacy_envelope(event: dict[str, Any], *, protected: Iterable[str] = ()) -> dict[str, Any]:
    """Retain non-activity message/canvas aliases while removing private fields."""
    return dict(redact_secrets(_clean(event, protected=protected)))


def normalize_activity(event: Any) -> dict[str, Any]:
    raw = event.model_dump(mode="json") if hasattr(event, "model_dump") else event.as_dict() if hasattr(event, "as_dict") else dict(event)
    metadata: dict[str, Any] = {}
    for key in ("payload", "details", "metadata"):
        if isinstance(raw.get(key), dict):
            metadata.update(raw[key])
    raw["metadata"] = metadata
    event_type = str(raw.get("type") or raw.get("event_type") or "step.updated")
    raw["type"] = raw["event_type"] = event_type
    for key in (*_IDENTITY_FIELDS, "tool_name", "path", "command", "semantic_kind", "visibility"):
        if raw.get(key) in (None, "") and metadata.get(key) not in (None, ""):
            raw[key] = metadata[key]
    return raw


def is_public_activity(event: Any) -> bool:
    raw = normalize_activity(event)
    et = raw["type"]
    if raw.get("visibility") == "internal":
        return False
    if et == "agent.summary":
        meta = raw["metadata"]
        return meta.get("summary_source") == "agent" or (
            meta.get("summary_source") == "provider" and meta.get("summary_trusted") is True
            and meta.get("summary_complete") is True and meta.get("provider") == "openai"
            and meta.get("transport") == "direct_responses"
        )
    if raw.get("semantic_kind") in {"assistant_generation", "reasoning", "provider", "usage"}:
        return False
    if (et.startswith(_PRIVATE_PREFIXES) and et != "provider.warning") or et in {"assistant.delta", "assistant.started", "assistant.message", "assistant.completed", "subagent.delta"}:
        return False
    return et.startswith(_PUBLIC_PREFIXES) or et in _LEGACY_PUBLIC or et in {"error", "warning", "provider.warning", "coding.progress", "coding.terminal", "log.warning", "log.error"}


def project_activity(event: Any, *, protected: Iterable[str] = ()) -> dict[str, Any] | None:
    raw = normalize_activity(event)
    if not is_public_activity(raw):
        return None
    meta = raw["metadata"]
    safe = {k: raw[k] for k in _IDENTITY_FIELDS if raw.get(k) is not None}
    for key in ("title", "tool_name", "path", "command", "status", "error", "output_preview", "token_usage", "changed_files", "category", "semantic_kind"):
        if key in raw:
            safe[key] = raw[key]
    public_meta = {k: v for k, v in meta.items() if k in _PUBLIC_METADATA and k != "reason"}
    if not raw["type"].startswith("steering."):
        for key in ("text", "instruction"):
            public_meta.pop(key, None)
    if raw["type"] != "turn.finished":
        for key in ("content", "attachments"):
            public_meta.pop(key, None)
    for key in _PUBLIC_METADATA - set(_IDENTITY_FIELDS) - {"reason", "content", "text", "instruction", "summary"}:
        if key in raw:
            public_meta[key] = raw[key]
    try:
        typed = ActivityMetadata.model_validate({k: v for k, v in {**public_meta, **safe}.items() if v is not None})
    except ValueError:
        return None
    safe.update({key: value for key, value in typed.model_dump().items() if value not in ("", 0)})
    safe["metadata"] = public_meta
    safe["details"] = public_meta
    safe["type"] = safe["event_type"] = raw["type"]
    safe["visibility"] = "terminal" if raw.get("visibility") == "terminal" else "progress"
    public_meta["visibility"] = safe["visibility"]
    summary = raw.get("public_summary") or meta.get("public_summary") or raw.get("summary") or raw.get("detail") or raw.get("message") or ""
    safe["summary"] = safe_text(summary, 240, protected=protected)
    safe["detail"] = safe_text(raw.get("detail") or summary, protected=protected)
    safe["summary_source"] = meta.get("summary_source", "observable")
    # Tool success is an observation, never independent goal verification.
    safe["evidence_status"] = ("unverified" if safe["summary_source"] != "observable" else
        meta.get("evidence_status", "observed" if raw.get("status") in TERMINAL else "attempted"))
    public_meta["summary_source"] = safe["summary_source"]
    public_meta["evidence_status"] = safe["evidence_status"]
    operation = raw.get("operation_id") or raw.get("tool_call_id") or raw.get("call_id") or meta.get("tool_call_id") or meta.get("call_id") or raw.get("step_id") or raw.get("request_id") or meta.get("request_id")
    if operation:
        safe["operation_id"] = str(operation)
    cleaned = dict(redact_secrets(_clean(safe, protected=protected)))
    # The final conversation message is not an activity preview. Keep the
    # existing response-content contract while bounding ordinary activity text.
    if raw["type"] == "turn.finished" and isinstance(meta.get("content"), str):
        content = safe_text(meta["content"], 100000, protected=protected)
        cleaned["metadata"]["content"] = content
        cleaned["details"]["content"] = content
    return cleaned


def delivery_key(event: dict[str, Any]) -> str:
    """Exact delivery identity; lifecycle revisions and repeated stream chunks differ."""
    raw = normalize_activity(event)
    comparable = {k: v for k, v in raw.items() if k not in {"sequence", "timestamp", "started_at", "created_at", "details"}}
    return hashlib.sha256(json.dumps(comparable, sort_keys=True, default=str).encode()).hexdigest()


def operation_key(event: dict[str, Any]) -> str:
    raw = normalize_activity(event)
    meta = raw["metadata"]
    if raw["type"].startswith("steering."):
        request_id = raw.get("request_id") or meta.get("request_id") or raw.get("operation_id")
        if request_id:
            return f"steering:{raw.get('conversation_id') or raw.get('session_id') or ''}:{request_id}"
    approval_id = (
        raw.get("inbox_item_id")
        or meta.get("inbox_item_id")
        or raw.get("permission_request_id")
        or meta.get("permission_request_id")
        or raw.get("approval_request_id")
        or meta.get("approval_request_id")
        or raw.get("action_id")
        or meta.get("action_id")
    )
    if approval_id:
        return f"approval:{raw.get('conversation_id') or raw.get('session_id') or ''}:{approval_id}"
    op = (
        raw.get("operation_id")
        or raw.get("tool_call_id")
        or raw.get("call_id")
        or meta.get("call_id")
        or meta.get("tool_call_id")
        or raw.get("step_id")
        or raw.get("event_id")
        or raw.get("id")
        or raw.get("request_id")
        or meta.get("request_id")
        or (raw.get("title") if (raw.get("sub_operation") or meta.get("sub_operation")) else None)
    )
    return ":".join(str(raw.get(k) or "") for k in ("task_id", "attempt_id", "agent_id", "subagent_id")) + ":" + str(op or raw["type"])


def accepts_activity_update(previous: dict[str, Any], event: dict[str, Any]) -> bool:
    old_sequence, new_sequence = int(previous.get("sequence") or 0), int(event.get("sequence") or 0)
    if old_sequence and new_sequence and new_sequence < old_sequence:
        return False
    if str(previous.get("status")) in TERMINAL and str(event.get("status", "running")) not in TERMINAL:
        return False
    old_type = str(previous.get("type") or previous.get("event_type") or "")
    new_type = str(event.get("type") or event.get("event_type") or "")
    return (old_type not in {"steering.applied", "steering.rejected", "steering.superseded"}
            or old_type == new_type or not old_sequence and new_sequence > 0)


def is_mutation(event: dict[str, Any]) -> bool:
    raw = normalize_activity(event)
    tool = str(raw.get("tool_name") or raw.get("title") or "").lower()
    type_matches = raw["type"] in {"file.changed", "patch.applied"}
    tool_matches = any(m in tool for m in ("edit_file", "write_file", "apply_patch", "modify_file", "create_file"))
    is_mut = type_matches or tool_matches
    return is_mut and raw.get("status", "success") in {"success", "completed", "done"} and raw.get("execution_mode", raw["metadata"].get("phase")) not in {"plan", "plan_only"}


def visible_activity(event: dict[str, Any], visibility: str = "normal") -> bool:
    et = str(event.get("type") or event.get("event_type") or "")
    if visibility == "verbose":
        return True
    if visibility == "minimal":
        return str(event.get("status")) not in TERMINAL or et.startswith(("turn.", "approval.", "action.approval.", "steering.", "recovery.")) or str(event.get("status")) in {"failed", "error", "cancelled"}
    return et not in {"command.output", "command.delta", "tool.stdout", "tool.stderr", "tool.progress"}


def bound_operations(events: list[dict[str, Any]], recent: int = 200) -> list[dict[str, Any]]:
    active = [e for e in events if str(e.get("status", "running")) not in TERMINAL]
    completed = [e for e in events if str(e.get("status", "running")) in TERMINAL]
    retained = {id(e) for e in active + completed[-recent:]}
    return [e for e in events if id(e) in retained]


def native_summary_endpoint(base_url: str) -> bool:
    """An OpenAI label alone cannot establish native summary provenance."""
    parsed = urlsplit(str(base_url or ""))
    return parsed.scheme == "https" and parsed.hostname == "api.openai.com" and not parsed.username and not parsed.password


class ProviderSummaryAssembler:
    """Bounded private assembly; only a completed native summary becomes public."""
    def __init__(self, *, provider: str, transport: str, supported: bool) -> None:
        self.trusted = supported and provider == "openai" and transport == "direct_responses"
        self._parts: dict[tuple[str, int], str | None] = {}

    def append(self, item_id: str, index: int, text: str) -> None:
        key = (item_id, index)
        if self.trusted and (key in self._parts or len(self._parts) < 64):
            previous = self._parts.get(key, "")
            if previous is not None:
                assembled = previous + text
                # Truncation could expose part of a protected secret. Discard
                # the entire oversized section rather than project a fragment.
                self._parts[key] = assembled if len(assembled) <= 8000 else None

    def finish(self, item_id: str, index: int) -> dict[str, Any] | None:
        text = self._parts.pop((item_id, index), "")
        if not self.trusted or not text:
            return None
        return {"event_type": "agent.summary", "title": "Reasoning summary", "summary": safe_text(text, 8000),
                "status": "success", "metadata": {"summary_source": "provider", "summary_trusted": True,
                "summary_complete": True, "provider": "openai", "transport": "direct_responses",
                "item_id": item_id, "summary_index": index}}

    def finish_item(self, item_id: str) -> list[dict[str, Any]]:
        completed = []
        for key in list(self._parts):
            if key[0] == item_id:
                summary = self.finish(*key)
                if summary is not None:
                    completed.append(summary)
        return completed
