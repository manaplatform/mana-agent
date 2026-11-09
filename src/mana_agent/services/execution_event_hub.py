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
import queue
from contextlib import contextmanager, nullcontext
from collections import deque, OrderedDict
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mana_agent.cli.events import ChatEvent, make_event, normalize_event_kind, utc_now_iso
from mana_agent.chat.activity import normalize_activity, project_activity, is_public_activity, delivery_key, sanitize_legacy_envelope
from mana_agent.utils.redaction import redact_json_line, redact_secrets
from mana_agent.workspaces.paths import repository_dir, repository_id_for_path, session_dir

logger = logging.getLogger(__name__)

Subscriber = Callable[[dict[str, Any]], None]


def conversations_root(repository_id: str) -> Path:
    return repository_dir(repository_id) / "dashboard" / "conversations"


def conversation_events_path(repository_id: str, conversation_id: str) -> Path:
    for identifier in (repository_id, conversation_id):
        if not identifier or identifier in {".", ".."} or any(character in identifier for character in ("/", "\\", "\x00")):
            raise ValueError("Invalid activity journal scope")
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
    payload = normalize_activity(event)
    meta = dict(payload.get("metadata") or {})
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
    payload["event_id"] = str(payload.get("event_id") or payload.get("id") or f"evt-{uuid.uuid4().hex}")
    payload["id"] = payload["event_id"]
    payload.setdefault("started_at", payload.get("timestamp") or utc_now_iso())
    payload.setdefault("timestamp", payload["started_at"])
    payload.setdefault("status", "running")
    payload["type"] = payload["event_type"] = str(payload.get("type") or payload.get("event_type") or "step.updated")
    payload.setdefault(
        "kind",
        meta.get("kind") or normalize_event_kind(str(payload.get("type") or "")),
    )
    from mana_agent.chat.execution_trace import classify_activity_category

    tool_name = str(meta.get("tool_name") or meta.get("name") or payload.get("tool_name") or "")
    payload.setdefault(
        "category",
        classify_activity_category(
            str(payload.get("type") or ""),
            tool_name=tool_name,
            metadata=meta,
        ),
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
    return delivery_key(payload)


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

    _delivery_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _repositories: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _journal_offsets: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _journal_tails: dict[str, deque] = field(default_factory=dict, init=False, repr=False)
    _journal_index: dict[str, list[tuple[int, int]]] = field(default_factory=dict, init=False, repr=False)
    _journal_seen: dict[str, OrderedDict] = field(default_factory=dict, init=False, repr=False)
    _journal_max: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _attempts: dict[str, tuple[int, str]] = field(default_factory=dict, init=False, repr=False)
    _poll_cursors: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _watcher: threading.Thread | None = field(default=None, init=False, repr=False)
    degraded: bool = field(default=False, init=False)
    _pending_delivery: deque = field(default_factory=deque, init=False, repr=False)
    _overflow_delivery: dict[str, list[Subscriber]] = field(default_factory=dict, init=False, repr=False)
    _delivering: bool = field(default=False, init=False, repr=False)

    @contextmanager
    def _journal_lock(self, repository: str, conversation: str):
        # Share the existing portable file-lock implementation, including Windows.
        from mana_agent.execution_supervisor.store import (
            _acquire_file_lock, _release_file_lock, _thread_lock_for_store_path,
        )
        path = conversation_events_path(repository, conversation).with_suffix(".lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        with _thread_lock_for_store_path(path):
            with path.open("a+b") as handle:
                _acquire_file_lock(handle)
                try:
                    yield
                finally:
                    _release_file_lock(handle)

    def _refresh_journal(self, repository: str, conversation: str) -> deque:
        path = conversation_events_path(repository, conversation)
        key = str(path)
        tail = self._journal_tails.setdefault(key, deque(maxlen=self.keep_memory))
        index = self._journal_index.setdefault(key, [])
        seen = self._journal_seen.setdefault(key, OrderedDict())
        if not path.exists():
            return tail
        size = path.stat().st_size
        offset = self._journal_offsets.get(key, 0)
        if size < offset:
            tail.clear()
            index.clear()
            seen.clear()
            self._journal_max[key] = 0
            offset = 0
        with path.open("rb") as handle:
            handle.seek(offset)
            while True:
                position = handle.tell()
                line = handle.readline()
                if not line or not line.endswith(b"\n"):
                    break  # A writer has not finished this record yet.
                offset = handle.tell()
                try:
                    row = json.loads(line)
                    sequence = int(row.get("sequence") or 0)
                except (ValueError, TypeError, AttributeError):
                    continue
                if sequence and (not index or sequence // 256 > index[-1][0] // 256):
                    index.append((sequence, position))
                tail.append(row)
                seen[_event_fingerprint(row)] = row
                if len(seen) > self.keep_memory:
                    seen.popitem(last=False)
                self._journal_max[key] = max(self._journal_max.get(key, 0), sequence)
        self._journal_offsets[key] = offset
        return tail

    def publish(
        self, event: ChatEvent | dict[str, Any], *, conversation_id: str = "",
        execution_id: str = "", repository_id: str = "", persist: bool = True,
        authoritative: bool = False,
    ) -> dict[str, Any]:
        from mana_agent.services.activity_context import current_activity_context
        ambient = current_activity_context()
        context = None if authoritative else ambient
        protected = ambient.protected_text if ambient else ()
        raw = normalize_activity(event)
        if raw.get("backend") and raw.get("sequence") and not raw.get("producer_sequence"):
            raw["producer_sequence"] = raw["sequence"]
        if context is not None:
            for key, value in context.identity().items():
                if key in {"agent_id", "subagent_id", "parent_agent_id", "task_id", "root_task_id", "parent_task_id"} and raw.get(key):
                    continue
                raw[key] = value
            conversation_id = context.conversation_id
            execution_id = context.execution_id
            repository_id = context.repository_id
        payload = normalize_execution_event(raw, conversation_id=conversation_id,
                                            execution_id=execution_id, repository_id=repository_id)
        et = payload["type"]
        if payload.get("visibility") == "internal" or et.startswith(("reasoning.", "provider.", "usage.")) and et != "provider.warning":
            return {"event_id": payload["event_id"], "visibility": "internal", "sequence": 0}
        projected = project_activity(payload, protected=protected)
        if projected is not None:
            payload = normalize_execution_event(projected, conversation_id=conversation_id,
                                                execution_id=execution_id, repository_id=repository_id)
        elif (is_public_activity(payload) or et == "agent.summary"
              or payload.get("semantic_kind") in {"reasoning", "assistant_generation", "provider", "usage"}
              or et.startswith(("assistant.", "subagent.delta")) and payload.get("backend")):
            return {"event_id": payload["event_id"], "visibility": "internal", "sequence": 0}
        if projected is None:
            payload = sanitize_legacy_envelope(payload, protected=protected)
        conversation = str(payload.get("conversation_id") or "")
        repository = str(payload.get("repository_id") or "")
        task_id = str(payload.get("task_id") or "")
        generation = int(payload.get("attempt_generation") or 0)
        if authoritative and (payload.get("metadata") or {}).get("execution_supervisor") and task_id:
            known = self._attempts.get(task_id)
            if known is None or generation >= known[0]:
                self._attempts[task_id] = (generation, str(payload.get("attempt_id") or ""))
        elif task_id in self._attempts and not authoritative:
            expected = self._attempts[task_id]
            if (generation, str(payload.get("attempt_id") or "")) != expected or context is None or context.supervisor is None:
                return {"event_id": payload["event_id"], "visibility": "internal", "sequence": 0}
        fingerprint = _event_fingerprint(payload)
        fence = context.publication() if context else nullcontext()
        try:
            with fence, self._delivery_lock:
                with self._lock:
                    duplicate = self._seen_fingerprints.get(fingerprint)
                    if duplicate is not None:
                        return deepcopy(duplicate)
                durable = persist and bool(conversation and repository)
                with self._journal_lock(repository, conversation) if durable else nullcontext():
                    if durable:
                        self._repositories[conversation] = repository
                        self._repair_journal_tail(repository, conversation)
                        self._refresh_journal(repository, conversation)
                        journal_key = str(conversation_events_path(repository, conversation))
                        duplicate = self._journal_seen[journal_key].get(fingerprint)
                        if duplicate is not None:
                            return deepcopy(duplicate)
                        base = max(self._sequences.get(conversation, 0), self._journal_max.get(journal_key, 0))
                    else:
                        base = self._sequences.get(conversation, 0)
                    # Provider sequences do not reserve holes in the public cursor.
                    payload["sequence"] = base + 1
                    if durable:
                        try:
                            self._append_durable(repository, conversation, payload)
                            self._refresh_journal(repository, conversation)
                        except OSError:
                            self.degraded = True
                            payload["metadata"]["replayable"] = False
                            payload["sequence"] = 0
                            logger.warning("Activity persistence unavailable; recovery may be incomplete")
                            self._queue_delivery({"type": "activity.degraded", "event_type": "activity.degraded",
                                "event_id": f"degraded-{uuid.uuid4().hex}", "sequence": 0, "status": "waiting",
                                "conversation_id": conversation, "execution_id": str(payload.get("execution_id") or ""),
                                "title": "Activity history unavailable", "summary": "Live updates may not be replayable."})
                    self._sequences[conversation] = max(base, payload["sequence"])
                if durable and payload["sequence"] and conversation in self._subscribers:
                    cursor = self._poll_cursors.setdefault(conversation, base)
                    while cursor < payload["sequence"]:
                        pending = self.history(conversation_id=conversation, repository_id=repository,
                                               after_sequence=cursor, forward=True, limit=1024)
                        if not pending:
                            break
                        for row in pending:
                            if int(row.get("sequence") or 0) > payload["sequence"]:
                                break
                            cursor = int(row.get("sequence") or 0)
                            if _event_fingerprint(row) not in self._seen_fingerprints:
                                self._remember(row)
                                self._queue_delivery(row)
                        self._poll_cursors[conversation] = cursor
                else:
                    self._remember(payload)
                    self._queue_delivery(payload)
            if authoritative or context is not None and context.supervisor is not None:
                self._start_watcher()
            else:
                self._drain_deliveries()
            return deepcopy(payload)
        except Exception as exc:
            # A stale publisher cannot affect execution or become current activity.
            from mana_agent.execution_supervisor.errors import ExecutionSupervisorError
            if isinstance(exc, OSError):
                self.degraded = True
                logger.warning("Activity stream unavailable; execution state is preserved")
                warning = {"event_id": f"degraded-{uuid.uuid4().hex}", "type": "activity.degraded",
                           "event_type": "activity.degraded", "title": "Activity history unavailable",
                           "summary": "Execution state is preserved; activity replay may be incomplete.",
                           "status": "waiting", "sequence": 0, "conversation_id": conversation,
                           "execution_id": str(payload.get("execution_id") or "")}
                with self._delivery_lock:
                    self._queue_delivery(warning)
                if authoritative or context is not None and context.supervisor is not None:
                    self._start_watcher()
                else:
                    self._drain_deliveries()
                return warning
            if isinstance(exc, (ExecutionSupervisorError, ValueError)) and context is not None:
                logger.debug("Rejected activity from an invalid execution context: %s", type(exc).__name__)
                return {"event_id": payload["event_id"], "visibility": "internal", "sequence": 0}
            raise

    def _remember(self, payload: dict[str, Any]) -> None:
        with self._lock:
            key = _event_fingerprint(payload)
            self._seen_fingerprints[key] = dict(payload)
            self._memory.append(payload)
            while len(self._memory) > self.keep_memory:
                removed = self._memory.pop(0)
                self._seen_fingerprints.pop(_event_fingerprint(removed), None)

    def _queue_delivery(self, payload: dict[str, Any]) -> None:
        # Enqueue while sequence allocation is serialized; never invoke user code
        # under a supervisor fence, journal lock, or delivery lock.
        with self._lock:
            subscribers = list(self._global_subscribers) + list(self._subscribers.get(str(payload.get("conversation_id") or ""), []))
            if subscribers:
                conversation = str(payload.get("conversation_id") or "")
                if conversation in self._overflow_delivery or len(self._pending_delivery) >= 1024:
                    targets = self._overflow_delivery.setdefault(conversation, [])
                    for callback in subscribers:
                        if callback not in targets:
                            targets.append(callback)
                    return
                self._pending_delivery.append((payload, subscribers))

    def _drain_deliveries(self) -> None:
        with self._lock:
            if self._delivering:
                return
            self._delivering = True
        while True:
            with self._lock:
                if not self._pending_delivery:
                    if not self._overflow_delivery:
                        self._delivering = False
                        return
                    conversation = next(iter(self._overflow_delivery))
                    subscribers = self._overflow_delivery.pop(conversation)
                    current = {"type": "activity.backpressure", "event_type": "activity.backpressure",
                               "event_id": f"backpressure-{uuid.uuid4().hex}", "sequence": 0,
                               "status": "waiting", "conversation_id": conversation,
                               "title": "Recovering activity history",
                               "summary": "Delivery exceeded its buffer; durable replay is required."}
                else:
                    current, subscribers = self._pending_delivery.popleft()
            for callback in subscribers:
                try:
                    callback(deepcopy(current))
                except Exception:
                    logger.debug("execution event subscriber raised", exc_info=True)

    def _start_watcher(self) -> None:
        with self._lock:
            if self._watcher and self._watcher.is_alive():
                return
            self._stop.clear()
            self._watcher = threading.Thread(target=self._watch, name="mana-activity-journal", daemon=True)
            self._watcher.start()

    def _watch(self) -> None:
        while not self._stop.wait(0.1):
            self._drain_deliveries()
            with self._lock:
                watched = [(sid, self._repositories.get(sid, "")) for sid in self._subscribers]
            for sid, repository in watched:
                if not repository:
                    continue
                try:
                    with self._delivery_lock:
                        rows = self.history(conversation_id=sid, repository_id=repository,
                                            after_sequence=self._poll_cursors.get(sid, 0), limit=1024,
                                            forward=True)
                        for row in rows:
                            self._poll_cursors[sid] = int(row.get("sequence") or 0)
                            if _event_fingerprint(row) not in self._seen_fingerprints:
                                self._remember(row)
                                self._queue_delivery(row)
                    self._drain_deliveries()
                except OSError:
                    if not self.degraded:
                        self._queue_delivery({"type": "activity.degraded", "title": "Activity recovery unavailable",
                                              "status": "waiting", "sequence": 0, "conversation_id": sid})
                        self._drain_deliveries()
                    self.degraded = True
                    logger.warning("Activity journal temporarily unavailable")

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            callbacks = [c for rows in self._subscribers.values() for c in rows] + self._global_subscribers
            self._subscribers.clear()
            self._global_subscribers.clear()
        if self._watcher is not None and threading.current_thread() is not self._watcher:
            self._watcher.join(timeout=0.5)
        for callback in callbacks:
            close = getattr(callback, "close", None)
            if close:
                close()

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

    def subscribe(self, conversation_id: str, callback: Subscriber, *, repository_id: str = "", buffered: bool = False) -> Callable[[], None]:
        key = str(conversation_id or "").strip()
        if not key:
            raise ValueError("conversation_id is required for subscription")
        with self._delivery_lock:
            baseline = self._sequences.get(key, 0)
            if repository_id:
                previous = self.history(conversation_id=key, repository_id=repository_id, limit=1)
                baseline = max(baseline, int(previous[-1].get("sequence") or 0) if previous else 0)
            if buffered:
                callback = _BufferedSubscriber(self, key, callback, repository_id, baseline)
            with self._lock:
                self._subscribers[key].append(callback)
                self._poll_cursors.setdefault(key, baseline)
                if repository_id:
                    self._repositories[key] = repository_id
        self._start_watcher()

        def _unsubscribe() -> None:
            with self._lock:
                rows = self._subscribers.get(key) or []
                if callback in rows:
                    rows.remove(callback)
                if not rows:
                    self._subscribers.pop(key, None)
            close = getattr(callback, "close", None)
            if close:
                close()
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
        self, *, conversation_id: str = "", execution_id: str = "", limit: int = 200,
        repository_id: str = "", after_sequence: int = 0, before_sequence: int = 0,
        forward: bool = False,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 200), 5000))
        repository_id = repository_id or self._repositories.get(conversation_id, "")
        if after_sequence and before_sequence:
            raise ValueError("use either after_sequence or before_sequence")
        rows: dict[str, dict[str, Any]] = {}
        if repository_id and conversation_id:
            with self._delivery_lock:
                self._repositories[conversation_id] = repository_id
                for row in self.load_durable(repository_id, conversation_id, limit=limit,
                                              after_sequence=after_sequence, before_sequence=before_sequence,
                                              execution_id=execution_id, forward=forward or after_sequence > 0):
                    rows[str(row.get("sequence") or row.get("event_id"))] = row
        with self._lock:
            memory = list(self._memory)
        for row in memory:
            if conversation_id and row.get("conversation_id") != conversation_id:
                continue
            if execution_id and str(row.get("execution_id") or row.get("turn_id")) != execution_id:
                continue
            seq = int(row.get("sequence") or 0)
            if seq <= after_sequence or before_sequence and seq >= before_sequence:
                continue
            rows[str(seq or row.get("event_id"))] = row
        ordered = sorted(rows.values(), key=lambda row: int(row.get("sequence") or 0))
        page = ordered[:limit] if forward or after_sequence else ordered[-limit:]
        return [self._public_history_row(row) for row in page]

    @staticmethod
    def _public_history_row(row: dict[str, Any]) -> dict[str, Any]:
        public = project_activity(row)
        if public is not None:
            return normalize_execution_event(public)
        et = str(row.get("type") or row.get("event_type") or "")
        if is_public_activity(row) or et == "agent.summary" or et.startswith(("reasoning.", "provider.", "usage.")) or row.get("semantic_kind") in {"reasoning", "assistant_generation", "provider", "usage"} or row.get("visibility") == "internal":
            # Preserve the sequence cursor without retaining protected content.
            return {"type": "activity.omitted", "event_type": "activity.omitted", "event_id": row.get("event_id"),
                    "sequence": row.get("sequence"), "visibility": "internal", "summary": "",
                    "conversation_id": row.get("conversation_id"), "execution_id": row.get("execution_id")}
        return normalize_execution_event(sanitize_legacy_envelope(row))

    def load_durable(
        self, repository_id: str, conversation_id: str, *, limit: int = 500,
        after_sequence: int = 0, before_sequence: int = 0, execution_id: str = "", forward: bool = False,
    ) -> list[dict[str, Any]]:
        path = conversation_events_path(repository_id, conversation_id)
        if not path.exists():
            return []
        self._refresh_journal(repository_id, conversation_id)
        result: deque = deque(maxlen=max(1, limit))
        offset = 0
        if after_sequence:
            for sequence, position in self._journal_index.get(str(path), []):
                if sequence > after_sequence:
                    break
                offset = position
        with path.open("rb") as handle:
            handle.seek(offset)
            for line in handle:
                if not line.endswith(b"\n"):
                    break
                try:
                    row = json.loads(line)
                    seq = int(row.get("sequence") or 0)
                except (ValueError, TypeError, AttributeError):
                    continue
                if before_sequence and seq >= before_sequence:
                    break
                if seq <= after_sequence or execution_id and str(row.get("execution_id") or row.get("turn_id")) != execution_id:
                    continue
                result.append(row)
                if forward and len(result) >= limit:
                    break
        return list(result)

    def _repair_journal_tail(self, repository_id: str, conversation_id: str) -> None:
        path = conversation_events_path(repository_id, conversation_id)
        if path.exists() and path.stat().st_size:
            with path.open("rb") as previous:
                previous.seek(-1, 2)
                incomplete = previous.read(1) != b"\n"
            if incomplete:
                with path.open("ab") as repair:
                    repair.write(b"\n")

    def _append_durable(self, repository_id: str, conversation_id: str, payload: dict[str, Any]) -> None:
        path = conversation_events_path(repository_id, conversation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(redact_json_line(json.dumps(payload, ensure_ascii=False, default=str)) + "\n")


class _BufferedSubscriber:
    """Bounded frontend delivery; overflow resumes from the durable cursor."""
    def __init__(self, hub: ExecutionEventHub, conversation: str, callback: Subscriber, repository: str, cursor: int = 0) -> None:
        self.hub, self.conversation, self.callback, self.repository = hub, conversation, callback, repository
        self.queue: queue.Queue = queue.Queue(maxsize=1024)
        self.stop = threading.Event()
        self.cursor = cursor
        self.recover = False
        self._recovery_revision = 0
        self._recovery_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="mana-activity-subscriber", daemon=True)
        self.thread.start()

    def __call__(self, payload: dict[str, Any]) -> None:
        if self.stop.is_set():
            return
        if payload.get("type") == "activity.backpressure":
            self._request_recovery()
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            self._request_recovery()

    def _request_recovery(self) -> None:
        with self._recovery_lock:
            self._recovery_revision += 1
            self.recover = True

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                payload = self.queue.get(timeout=0.1)
            except queue.Empty:
                if not self.recover:
                    continue
                payload = {}
            if int(payload.get("sequence") or 0) > self.cursor + 1:
                self.recover = True
            if self.recover:
                with self._recovery_lock:
                    recovery_revision = self._recovery_revision
                if not self.repository:
                    self._send({"type": "activity.degraded", "title": "Activity history unavailable",
                                "status": "waiting", "conversation_id": self.conversation})
                    self.recover = False
                else:
                    while not self.stop.is_set():
                        previous = self.cursor
                        try:
                            rows = self.hub.history(conversation_id=self.conversation, repository_id=self.repository,
                                                    after_sequence=self.cursor, forward=True, limit=1024)
                        except (OSError, ValueError):
                            self.hub.degraded = True
                            self._send({"type": "activity.degraded", "title": "Activity recovery unavailable",
                                        "status": "waiting", "conversation_id": self.conversation})
                            self.recover = False
                            break
                        for row in rows:
                            if not self._send(row):
                                self.hub.degraded = True
                                self.stop.set()
                                break
                        if len(rows) < 1024:
                            with self._recovery_lock:
                                self.recover = self._recovery_revision != recovery_revision
                            break
                        if self.cursor <= previous:
                            # A broken consumer cannot make recovery spin forever.
                            self.hub.degraded = True
                            self.stop.set()
                            break
            if payload.get("type") == "activity.backpressure":
                payload = {}  # A recovered delivery control record is not a running operation.
            if payload and (not payload.get("sequence") or int(payload.get("sequence") or 0) > self.cursor):
                self._send(payload)

    def _send(self, payload: dict[str, Any]) -> bool:
        try:
            sequence = int(payload.get("sequence") or 0)
            if sequence > self.cursor + 1:
                self.callback({"type": "activity.degraded", "title": "Activity history contains a gap",
                               "status": "waiting", "sequence": 0, "conversation_id": self.conversation})
                self.recover = True
                return False
            self.callback(deepcopy(payload))
            if sequence == self.cursor + 1:
                self.cursor = sequence
            return True
        except Exception:
            self.recover = True
            logger.debug("Activity subscriber needs recovery", exc_info=True)
            return False

    def close(self) -> None:
        self.stop.set()
        if threading.current_thread() is not self.thread:
            self.thread.join(timeout=0.25)


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
        if _HUB is not None:
            _HUB.close()
        _HUB = ExecutionEventHub()
        return _HUB


def repository_id_for_root(root: str | Path) -> str:
    return repository_id_for_path(root)
