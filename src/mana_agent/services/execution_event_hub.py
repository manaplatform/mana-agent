"""Shared execution event hub for CLI, API, and dashboard.

All surfaces publish and consume the same normalized ``ChatEvent`` envelope
from ``mana_agent.cli.events``. This module is the process-local fan-out and
durable per-conversation event log used by WebSocket clients and REST recovery.
"""

from __future__ import annotations

import json
import logging
import hashlib
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mana_agent.cli.events import ChatEvent, make_event, normalize_event_kind, utc_now_iso
from mana_agent.utils.redaction import redact_json_line, redact_secrets
from mana_agent.workspaces.paths import repository_dir, repository_id_for_path, session_dir

logger = logging.getLogger(__name__)

Subscriber = Callable[[dict[str, Any]], None]


def conversations_root(repository_id: str) -> Path:
    return repository_dir(repository_id) / "dashboard" / "conversations"


def conversation_events_path(repository_id: str, conversation_id: str) -> Path:
    if str(conversation_id).startswith("session_"):
        return session_dir(conversation_id) / "events.jsonl"
    return conversations_root(repository_id) / conversation_id / "events.jsonl"


def normalize_execution_event(
    event: ChatEvent | dict[str, Any],
    *,
    conversation_id: str = "",
    execution_id: str = "",
    repository_id: str = "",
) -> dict[str, Any]:
    """Normalize a ChatEvent (or dict) into the dashboard/API wire payload."""
    if isinstance(event, ChatEvent):
        payload = event.as_dict()
    else:
        payload = dict(event)
    meta = dict(payload.get("metadata") or payload.get("details") or {})
    conversation = str(
        conversation_id
        or payload.get("conversation_id")
        or meta.get("conversation_id")
        or ""
    ).strip()
    execution = str(
        execution_id
        or payload.get("execution_id")
        or meta.get("execution_id")
        or payload.get("turn_id")
        or ""
    ).strip()
    repo = str(
        repository_id
        or payload.get("repository_id")
        or meta.get("repository_id")
        or ""
    ).strip()
    if conversation:
        meta["conversation_id"] = conversation
        payload["conversation_id"] = conversation
    if execution:
        meta["execution_id"] = execution
        payload["execution_id"] = execution
        if not payload.get("turn_id"):
            payload["turn_id"] = execution
    if repo:
        meta["repository_id"] = repo
        payload["repository_id"] = repo
    payload["metadata"] = meta
    payload["details"] = dict(meta)
    payload.setdefault("event_id", payload.get("id") or f"evt-{uuid.uuid4().hex}")
    payload.setdefault("id", payload["event_id"])
    payload.setdefault("started_at", payload.get("timestamp") or utc_now_iso())
    payload.setdefault("timestamp", payload["started_at"])
    payload.setdefault("status", "running")
    payload.setdefault("type", "step.updated")
    payload.setdefault(
        "kind",
        meta.get("kind") or normalize_event_kind(str(payload.get("type") or "")),
    )
    structurally_redacted = redact_secrets(payload)
    return dict(
        json.loads(
            redact_json_line(
                json.dumps(structurally_redacted, ensure_ascii=False, default=str)
            )
        )
    )


def _event_fingerprint(payload: dict[str, Any]) -> str:
    """Identify an exact delivery while allowing correlated lifecycle updates."""
    metadata = payload.get("metadata") or payload.get("details") or {}
    comparable = {
        "conversation_id": payload.get("conversation_id"),
        "event_id": payload.get("event_id") or payload.get("id"),
        "type": payload.get("type"),
        "status": payload.get("status"),
        "summary": payload.get("summary") or payload.get("message"),
        "output_preview": payload.get("output_preview"),
        "delta": payload.get("delta") or metadata.get("delta"),
        "progress": metadata.get("progress"),
        "result_summary": metadata.get("result_summary"),
        "error": payload.get("error") or metadata.get("error"),
    }
    encoded = json.dumps(comparable, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class ExecutionEventHub:
    """Thread-safe pub/sub + durable JSONL event log for conversation executions."""

    keep_memory: int = 4000
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _memory: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _seen_fingerprints: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _sequences: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _subscribers: dict[str, list[Subscriber]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )
    _global_subscribers: list[Subscriber] = field(default_factory=list, init=False, repr=False)

    def publish(
        self,
        event: ChatEvent | dict[str, Any],
        *,
        conversation_id: str = "",
        execution_id: str = "",
        repository_id: str = "",
        persist: bool = True,
    ) -> dict[str, Any]:
        payload = normalize_execution_event(
            event,
            conversation_id=conversation_id,
            execution_id=execution_id,
            repository_id=repository_id,
        )
        conversation = str(payload.get("conversation_id") or "").strip()
        repository = str(payload.get("repository_id") or "").strip()
        fingerprint = _event_fingerprint(payload)

        with self._lock:
            duplicate = self._seen_fingerprints.get(fingerprint)
            if duplicate is not None:
                return dict(duplicate)
            requested_sequence = int(payload.get("sequence") or 0)
            current_sequence = self._sequence_base(repository, conversation)
            sequence = requested_sequence if requested_sequence > current_sequence else current_sequence + 1
            if conversation:
                self._sequences[conversation] = sequence
            payload["sequence"] = sequence
            self._seen_fingerprints[fingerprint] = dict(payload)
            self._memory.append(payload)
            if len(self._memory) > self.keep_memory:
                removed = self._memory[: len(self._memory) - self.keep_memory]
                del self._memory[: len(self._memory) - self.keep_memory]
                for row in removed:
                    self._seen_fingerprints.pop(_event_fingerprint(row), None)
            subscribers = list(self._global_subscribers)
            if conversation:
                subscribers.extend(self._subscribers.get(conversation, []))

        if persist and conversation and repository:
            try:
                self._append_durable(repository, conversation, payload)
            except Exception:  # durability must never block execution
                logger.debug("failed to persist execution event", exc_info=True)

        for callback in subscribers:
            try:
                callback(payload)
            except Exception:
                logger.debug("execution event subscriber raised", exc_info=True)
        return payload

    def _sequence_base(self, repository_id: str, conversation_id: str) -> int:
        if not conversation_id:
            return 0
        if conversation_id in self._sequences:
            return self._sequences[conversation_id]
        maximum = 0
        if repository_id:
            for row in self.load_durable(repository_id, conversation_id, limit=5000):
                try:
                    maximum = max(maximum, int(row.get("sequence") or 0))
                except (TypeError, ValueError):
                    continue
        self._sequences[conversation_id] = maximum
        return maximum

    def emit(
        self,
        event_type: str,
        *,
        title: str,
        conversation_id: str,
        execution_id: str = "",
        repository_id: str = "",
        message: str = "",
        status: str = "running",
        agent_id: str | None = "main",
        subagent_id: str | None = None,
        step_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str = "",
        event_id: str | None = None,
        parent_event_id: str | None = None,
        duration_ms: int | None = None,
        ended_at: str | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        event_metadata = {
            **(metadata or {}),
            "conversation_id": conversation_id,
            "execution_id": execution_id,
            "repository_id": repository_id,
        }
        if event_id and event_type.startswith("tool."):
            event_metadata.setdefault("tool_call_id", event_id)
        event = make_event(
            event_type,
            title=title,
            message=message,
            status=status,
            session_id=session_id or conversation_id,
            turn_id=execution_id,
            agent_id=agent_id,
            subagent_id=subagent_id,
            step_id=step_id,
            parent_event_id=parent_event_id,
            metadata=event_metadata,
        )
        if event_id:
            event.event_id = str(event_id)
        if duration_ms is not None:
            event.duration_ms = max(0, int(duration_ms))
        if ended_at:
            event.ended_at = str(ended_at)
        return self.publish(
            event,
            conversation_id=conversation_id,
            execution_id=execution_id,
            repository_id=repository_id,
            persist=persist,
        )

    def subscribe(self, conversation_id: str, callback: Subscriber) -> Callable[[], None]:
        key = str(conversation_id or "").strip()
        if not key:
            raise ValueError("conversation_id is required for subscription")
        with self._lock:
            self._subscribers[key].append(callback)

        def _unsubscribe() -> None:
            with self._lock:
                rows = self._subscribers.get(key) or []
                if callback in rows:
                    rows.remove(callback)
                if not rows and key in self._subscribers:
                    del self._subscribers[key]

        return _unsubscribe

    def subscribe_all(self, callback: Subscriber) -> Callable[[], None]:
        with self._lock:
            self._global_subscribers.append(callback)

        def _unsubscribe() -> None:
            with self._lock:
                if callback in self._global_subscribers:
                    self._global_subscribers.remove(callback)

        return _unsubscribe

    def history(
        self,
        *,
        conversation_id: str = "",
        execution_id: str = "",
        limit: int = 200,
        repository_id: str = "",
        after_sequence: int = 0,
    ) -> list[dict[str, Any]]:
        conversation = str(conversation_id or "").strip()
        execution = str(execution_id or "").strip()
        limit = max(1, min(int(limit or 200), 5000))

        durable: list[dict[str, Any]] = []
        if conversation and repository_id:
            durable = self.load_durable(repository_id, conversation, limit=limit * 2)

        with self._lock:
            memory = list(self._memory)

        merged: dict[str, dict[str, Any]] = {}
        for row in durable + memory:
            if conversation and str(row.get("conversation_id") or "") != conversation:
                continue
            if execution and str(row.get("execution_id") or row.get("turn_id") or "") != execution:
                continue
            try:
                sequence = int(row.get("sequence") or 0)
            except (TypeError, ValueError):
                sequence = 0
            if sequence <= max(0, int(after_sequence or 0)):
                continue
            event_id = str(row.get("event_id") or row.get("id") or "")
            merge_key = f"seq:{sequence}" if sequence else f"id:{event_id}"
            if not event_id:
                continue
            merged[merge_key] = row
        ordered = sorted(
            merged.values(),
            key=lambda item: (
                (
                    int(item.get("sequence") or 0)
                    if str(item.get("sequence") or "0").lstrip("-").isdigit()
                    else 0
                ),
                str(item.get("started_at") or item.get("timestamp") or ""),
            ),
        )
        return ordered[-limit:]

    def load_durable(self, repository_id: str, conversation_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
        path = conversation_events_path(repository_id, conversation_id)
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines[-max(1, limit) :]:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def _append_durable(self, repository_id: str, conversation_id: str, payload: dict[str, Any]) -> None:
        path = conversation_events_path(repository_id, conversation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(redact_json_line(json.dumps(payload, ensure_ascii=False, default=str)) + "\n")


_HUB: ExecutionEventHub | None = None
_HUB_LOCK = threading.Lock()


def get_execution_event_hub() -> ExecutionEventHub:
    global _HUB
    with _HUB_LOCK:
        if _HUB is None:
            _HUB = ExecutionEventHub()
        return _HUB


def reset_execution_event_hub_for_tests() -> ExecutionEventHub:
    """Replace the process hub (tests only)."""
    global _HUB
    with _HUB_LOCK:
        _HUB = ExecutionEventHub()
        return _HUB


def repository_id_for_root(root: str | Path) -> str:
    return repository_id_for_path(root)
