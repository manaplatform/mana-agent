"""Normalize Codex app-server notifications at the integration boundary."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from mana_agent.coding.models import AgentEvent


_METHOD_TYPES = {
    "thread/started": "backend.selected",
    "turn/started": "turn.started",
    "turn/completed": "turn.completed",
    "turn/failed": "error",
    "turn/cancelled": "turn.cancelled",
    "item/reasoning/summaryTextDelta": "reasoning.update",
    "item/reasoning/summaryPartAdded": "reasoning.update",
    "item/agentMessage/delta": "assistant.delta",
    "item/commandExecution/outputDelta": "command.output",
    "turn/plan/updated": "plan.created",
    "plan/updated": "plan.created",
    "tokenUsage/updated": "usage.update",
    "turn/tokenUsage/updated": "usage.update",
    "account/rateLimits/updated": "usage.update",
    "approval/requestApproval": "warning",
    "item/commandExecution/requestApproval": "warning",
    "item/fileChange/requestApproval": "warning",
    "item/permissions/requestApproval": "warning",
    "execCommandApproval": "warning",
    "applyPatchApproval": "warning",
}


def adapt_codex_event(
    task_id: str,
    notification: dict[str, Any],
    *,
    sequence: int = 0,
    model: str = "",
) -> AgentEvent:
    method = str(notification.get("method") or "")
    params = notification.get("params")
    payload = dict(params) if isinstance(params, dict) else {}
    item = payload.get("item") if isinstance(payload.get("item"), dict) else {}
    item_type = str(item.get("type") or payload.get("itemType") or "")
    event_type = _METHOD_TYPES.get(method, _item_event_type(method, item_type))
    thread_id = str(payload.get("threadId") or payload.get("thread_id") or "")
    turn = payload.get("turn")
    turn_id = str(payload.get("turnId") or (turn.get("id") if isinstance(turn, dict) else "") or "")
    status = _status(method, event_type, item)
    command = _first_text(item, "command", "cmd") or _first_text(payload, "command")
    path = _first_text(item, "path", "filePath") or _first_text(payload, "path", "filePath")
    output = _first_text(payload, "delta", "output", "text") or _first_text(item, "output", "text")
    usage = _usage(payload)
    error = _error(payload) if status == "failed" or event_type == "error" else ""
    event_id = _event_id(notification)
    return AgentEvent(
        event_id=event_id,
        event_type=event_type,
        task_id=task_id,
        parent_event_id=str(payload.get("parentEventId") or item.get("id") or "") or None,
        backend="codex",
        sequence=sequence,
        status=status,
        title=_title(event_type, item_type, command, path),
        summary=_summary(payload, item),
        thread_id=thread_id,
        turn_id=turn_id,
        tool_name=item_type or _first_text(payload, "toolName"),
        command=command,
        path=path,
        duration_ms=_duration_ms(payload, item),
        token_usage=usage,
        cost=_number(payload.get("cost")),
        model=str(payload.get("model") or model or ""),
        error=error,
        output_preview=output,
        payload=payload,
    )


def _item_event_type(method: str, item_type: str) -> str:
    phase = "started" if method == "item/started" else "completed" if method == "item/completed" else "update"
    lowered = item_type.lower()
    if "command" in lowered:
        return f"command.{phase}"
    if "filechange" in lowered or "patch" in lowered:
        return "patch.applied" if phase == "completed" else "file.changed"
    if "reasoning" in lowered:
        return "reasoning.started" if phase == "started" else "reasoning.update"
    if "plan" in lowered:
        return f"plan.step.{phase}"
    if "test" in lowered:
        return f"test.{phase}"
    if item_type:
        return f"tool.call.{phase}"
    return f"provider.{method.replace('/', '.') or 'notification'}"


def _status(method: str, event_type: str, item: dict[str, Any]) -> str:
    raw = str(item.get("status") or "").lower()
    if method.endswith("/cancelled") or raw == "cancelled":
        return "cancelled"
    if method.endswith("/failed") or raw in {"failed", "error"}:
        return "failed"
    if method.endswith("/completed") or event_type.endswith(".completed") or raw in {"completed", "success"}:
        return "success"
    return "running"


def _event_id(notification: dict[str, Any]) -> str:
    params = notification.get("params") if isinstance(notification.get("params"), dict) else {}
    direct = notification.get("id") or params.get("eventId") or params.get("notificationId")
    if direct:
        return f"codex-{direct}"
    canonical = json.dumps(notification, sort_keys=True, ensure_ascii=False, default=str)
    return f"codex-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:24]}"


def _usage(payload: dict[str, Any]) -> dict[str, Any] | None:
    source = payload.get("usage") or payload.get("tokenUsage")
    if not isinstance(source, dict):
        return None
    aliases = {
        "inputTokens": "input_tokens",
        "outputTokens": "output_tokens",
        "cachedInputTokens": "cached_tokens",
        "cachedTokens": "cached_tokens",
        "reasoningTokens": "reasoning_tokens",
        "totalTokens": "total_tokens",
        "contextWindow": "context_window",
    }
    return {aliases.get(str(key), str(key)): value for key, value in source.items() if value is not None}


def _summary(payload: dict[str, Any], item: dict[str, Any]) -> str:
    return (_first_text(payload, "message", "summary", "delta", "text") or _first_text(item, "summary", "text", "message", "command"))[:1000]


def _error(payload: dict[str, Any]) -> str:
    value = payload.get("error")
    if isinstance(value, dict):
        return _first_text(value, "message", "detail", "code")
    return str(value or payload.get("message") or "")


def _first_text(source: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            return "\n".join(str(item) for item in value)[:8000]
    return ""


def _duration_ms(payload: dict[str, Any], item: dict[str, Any]) -> int | None:
    value = payload.get("durationMs", item.get("durationMs"))
    number = _number(value)
    return int(number) if number is not None and number >= 0 else None


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _title(event_type: str, item_type: str, command: str, path: str) -> str:
    if command:
        return command[:160]
    if path:
        return path
    return item_type or event_type.replace(".", " ").title()


__all__ = ["adapt_codex_event"]
