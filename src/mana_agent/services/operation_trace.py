"""Read retained tool details through owned conversation events, never raw paths."""
from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field
import threading
from mana_agent.chat.activity import accepts_activity_update, operation_key
from mana_agent.chat.operations import merge_operation
from mana_agent.context_cost.artifact_store import ContextArtifactStore


@dataclass
class TraceBudget:
    used: int = 0
    lock: Any = field(default_factory=threading.RLock)

    def claim(self, requested: int, limit: int) -> int:
        with self.lock:
            permitted = min(max(0, requested), max(0, limit - self.used))
            self.used += permitted
            return permitted


def execution_trace_budget(context) -> TraceBudget:
    """A journal-derived accounting cache, not another execution state store."""
    if context is None or not context.conversation_id or not context.repository_id:
        return TraceBudget()
    from mana_agent.services.execution_event_hub import get_execution_event_hub
    hub = get_execution_event_hub()
    key = (context.repository_id, context.conversation_id, context.execution_id)
    with hub._lock:
        if not hasattr(hub, "_trace_budgets"):
            import weakref
            hub._trace_budgets = weakref.WeakValueDictionary()
        cached = hub._trace_budgets.get(key)
        if cached is not None:
            return cached
    tail = hub.history(conversation_id=context.conversation_id, repository_id=context.repository_id,
        execution_id=context.execution_id, limit=1)
    high_water = max((int(event.get("sequence") or 0) for event in tail), default=0)
    used, cursor = 0, 0
    while cursor < high_water:
        page = hub.history(conversation_id=context.conversation_id, repository_id=context.repository_id,
            execution_id=context.execution_id, after_sequence=cursor, limit=1000, forward=True)
        if not page:
            break
        for event in page:
            if int(event.get("sequence") or 0) > high_water:
                break
            meta = event.get("metadata") or {}
            used += int(meta.get("content_bytes") or 0)
        next_cursor = max(int(event.get("sequence") or 0) for event in page)
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    with hub._lock:
        return hub._trace_budgets.setdefault(key, TraceBudget(used=used))


class BackendTraceRecorder:
    """Retain provider observations without impersonating a native ToolRunner."""
    def __init__(self):
        from mana_agent.config.settings import Settings
        self.settings = Settings()
        from mana_agent.services.activity_context import current_activity_context
        self.budget = execution_trace_budget(current_activity_context())
        self.pending: dict[tuple[str, str], str] = {}
        self.key_filters: dict[tuple[str, str], Any] = {}
        self.offsets: dict[tuple[str, str], int] = {}
        self.bytes: dict[str, int] = {}
        self.omitted: dict[str, int] = {}
        self.total = 0
        self.suppressing: set[tuple[str, str]] = set()
        self.seen_deltas: set[tuple[str, str]] = set()
        self.degraded: set[str] = set()

    def observe(self, event, notification):
        from mana_agent.chat.activity import safe_text
        from mana_agent.services.activity_context import current_activity_context
        context = current_activity_context()
        if event.semantic_kind != "tool_execution" or not event.operation_id:
            return event
        params = notification.get("params") or {}
        item = params.get("item") or {}
        channel = event.payload.get("stream_channel", "combined")
        provider_diff = item.get("changes")
        if isinstance(provider_diff, list) and event.status in {"success", "failed"}:
            channel = "diff"
        key = (event.operation_id, channel)
        is_delta = notification.get("method") == "item/commandExecution/outputDelta"
        body = str(params.get("delta") or "") if is_delta else "\n".join(str(change.get("diff") or "") for change in provider_diff if isinstance(change, dict)) if isinstance(provider_diff, list) else ""
        if is_delta:
            self.seen_deltas.add(key)
        if key in self.suppressing:
            boundary = body.find("\n")
            self.omitted[event.operation_id] = self.omitted.get(event.operation_id, 0) + len((body[:boundary + 1] if boundary >= 0 else body).encode())
            body = "[oversized provider output token omitted]\n" + body[boundary + 1:] if boundary >= 0 else ""
            if boundary >= 0:
                self.suppressing.discard(key)
        pending = self.pending.pop(key, "") + body
        if not is_delta and event.status in {"success", "failed", "cancelled"}:
            if not pending and key not in self.seen_deltas:
                pending = str(item.get("aggregatedOutput") or "")
            boundary = len(pending)
        else:
            boundary = pending.rfind("\n") + 1
        from mana_agent.utils.redaction import PrivateKeyStreamFilter
        filtered = self.key_filters.setdefault(key, PrivateKeyStreamFilter()).filter(pending[:boundary])
        safe = safe_text(filtered, None, protected=getattr(context, "protected_text", ()))
        remaining = pending[boundary:]
        if len(remaining) > 65536:
            self.omitted[event.operation_id] = self.omitted.get(event.operation_id, 0) + len(remaining.encode())
            remaining = ""
            self.suppressing.add(key)
        self.pending[key] = remaining
        metadata = {**event.payload, "provenance": "codex_backend_owned",
            "trace_complete": not self.omitted.get(event.operation_id) and event.operation_id not in self.degraded,
            "omitted_bytes": self.omitted.get(event.operation_id, 0)}
        if not safe:
            return event.model_copy(update={"summary": "Command output" if is_delta else event.summary,
                "output_preview": "", "payload": metadata})
        room = self.budget.claim(min(len(safe.encode()), max(0, 10_485_760 - self.bytes.get(event.operation_id, 0))), self.settings.mana_execution_trace_max_bytes)
        retained = safe.encode()[:room].decode(errors="ignore")
        reference = ""
        if context and retained:
            try:
                reference = ContextArtifactStore().put(retained, session_id=context.conversation_id,
                    repository_id=context.repository_id, workspace_id="", content_type=f"execution/{channel}").artifact_id
            except Exception:
                self.degraded.add(event.operation_id)
        offset = self.offsets.get(key, 0)
        self.offsets[key] = offset + len(retained)
        size = len(retained.encode())
        self.omitted[event.operation_id] = self.omitted.get(event.operation_id, 0) + len(safe.encode()) - size
        self.bytes[event.operation_id] = self.bytes.get(event.operation_id, 0) + size
        self.total += size
        metadata.update(stream_channel=channel, stream_offset=offset, content_length=len(retained),
            content_reference=reference, content_bytes=size, workspace_id="",
            trace_complete=self.omitted[event.operation_id] == 0 and bool(reference) and event.operation_id not in self.degraded,
            preview_truncated=len(retained) > 3000,
            omitted_bytes=self.omitted[event.operation_id])
        return event.model_copy(update={"payload": metadata, "output_preview": retained[:3000],
            "summary": "Command output" if is_delta else event.summary})


def operation_records(service, conversation_id: str, execution_id: str, operation_id: str,
                      attempt_id: str, operation_attempt_id: str = "") -> list[dict[str, Any]]:
    service.get_or_raise(conversation_id)
    tail = service.list_events(conversation_id, execution_id=execution_id, limit=1)
    high_water = max((int(e.get("sequence") or 0) for e in tail), default=0)
    cursor = 0
    records = []
    while cursor < high_water:
        page = service.list_events(conversation_id, execution_id=execution_id, after_sequence=cursor, limit=1000, forward=True)
        if not page:
            break
        for event in page:
            sequence = int(event.get("sequence") or 0)
            if sequence > high_water:
                break
            cursor = max(cursor, sequence)
            meta = event.get("metadata") or {}
            if (str(event.get("operation_id") or meta.get("operation_id") or "") == operation_id
                    and str(event.get("attempt_id") or meta.get("attempt_id") or "") == attempt_id
                    and (not operation_attempt_id or str(event.get("operation_attempt_id") or meta.get("operation_attempt_id") or "") == operation_attempt_id)):
                records.append(event)
        if int(page[-1].get("sequence") or 0) >= high_water:
            break
    identities = {operation_key(e) for e in records}
    if len(identities) != 1:
        raise ValueError("Operation is missing or ambiguous in this execution attempt")
    return records


def operation_detail(service, conversation_id: str, execution_id: str, operation_id: str,
                     attempt_id: str, operation_attempt_id: str = "") -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for event in operation_records(service, conversation_id, execution_id, operation_id, attempt_id, operation_attempt_id):
        if not merged or accepts_activity_update(merged, event):
            merged = merge_operation(merged, event)
    return merged


def operation_content(service, conversation_id: str, execution_id: str, operation_id: str,
                      attempt_id: str, *, channel: str, offset: int = 0, limit: int = 16000,
                      operation_attempt_id: str = "") -> dict[str, Any]:
    if channel not in {"stdout", "stderr", "combined", "diff", "result"} or offset < 0 or not 1 <= limit <= 64000:
        raise ValueError("Use a supported output channel and a bounded nonnegative content range")
    records = operation_records(service, conversation_id, execution_id, operation_id, attempt_id, operation_attempt_id)
    fragments = []
    complete = True
    end = offset
    available = 0
    seen = set()
    for event in records:
        meta = event.get("metadata") or {}
        complete = complete and meta.get("trace_complete", True)
        if meta.get("stream_channel") != channel:
            continue
        start = int(meta.get("stream_offset") or 0)
        length = int(meta.get("content_length") or 0)
        identity = (start, meta.get("content_reference"))
        if identity in seen:
            continue
        seen.add(identity)
        available = max(available, start + length)
        if start + length <= offset or start >= offset + limit:
            continue
        reference = meta.get("content_reference")
        if not reference:
            complete = False
            continue
        if sum(len(f) for f in fragments) >= limit:
            continue
        body = ContextArtifactStore().read(reference, session_id=conversation_id,
            repository_id=service.repository_id, workspace_id=str(event.get("workspace_id") or meta.get("workspace_id") or ""),
            offset=max(0, offset - start), limit=min(limit - sum(len(f) for f in fragments), start + length - max(offset, start)))
        fragments.append(body)
        end = max(end, max(offset, start) + len(body))
    return {"content": "".join(fragments), "channel": channel, "offset": offset,
            "next_offset": end, "available_length": available, "has_more": end < available,
            "trace_complete": complete}
