"""Live Agent Work Queue.

This module is the orchestration spine the coding agent uses to stay *on top*
of tool execution instead of handing a frozen plan to the tools-manager and
waiting for a final answer.

    coding agent  <---->  AgentWorkQueue  <---->  tools_* (executor / manager)

Pieces
------
* ``WorkItem``        -- one unit of work (a job): a tool call with a gate,
                         priority, dependencies and a content fingerprint.
* ``AgentWorkQueue``  -- the planner emits jobs into it; the executor claims the
                         next *runnable* job (dependencies satisfied, highest
                         priority). Fingerprint dedup prevents the duplicate
                         searches / double reads the legacy pass-loop produced.
* ``EventBus``        -- every status transition is broadcast to subscribers.
* ``TaskBoard``       -- a live, renderable view of progress built from events.
* ``WorkQueueRunner`` -- drives the loop: claim -> execute -> classify ->
                         broadcast -> let the coding agent *sniff* the result
                         and emit follow-up jobs.
* ``JobSniffer``      -- the coding-agent hook: given a finished job and its
                         result, decide what new jobs (reads, edits, verify)
                         should be enqueued. This is where the agent lives at
                         the top of the hierarchy and steers the run.

The runner is deliberately transport-agnostic: it executes a job through an
injected ``execute`` callable, so it works with the local worker, the redis
executor, or a fake in tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Literal, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

WorkKind = Literal["discover", "search", "read", "edit", "verify", "summarize"]
WorkStatus = Literal[
    "pending",   # submitted, dependencies not yet satisfied
    "ready",     # runnable now
    "running",   # claimed and executing
    "done",      # completed successfully
    "failed",    # exhausted attempts / hard error
    "blocked",   # a dependency failed; will never run
    "skipped",   # duplicate or suppressed before execution
]

# Tools that should never run twice for the same fingerprint once they succeed.
_IDEMPOTENT_KINDS = {"discover", "search", "read"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def compute_fingerprint(*, kind: str, tool_name: str, tool_args: dict[str, Any], question: str = "") -> str:
    """Stable content key for a job.

    Two jobs that would perform the same observable action collapse to the same
    fingerprint, which is what the queue uses to refuse duplicates. Reads key on
    their path; searches on their normalized query; everything else on the tool
    name plus a normalized question/arg blob.
    """
    tool = _normalize_text(tool_name)
    args = tool_args if isinstance(tool_args, dict) else {}

    def _norm_path(value: Any) -> str:
        return _normalize_text(value).replace("\\", "/").lstrip("./").strip("/")

    if tool == "read_file":
        path = _norm_path(args.get("path") or args.get("file") or args.get("file_path"))
        payload = f"read_file:{path}" if path else f"read_file:{_normalize_text(question)[:160]}"
    elif tool in {"repo_search", "semantic_search", "list_files"}:
        query = _normalize_text(args.get("query") or args.get("q") or args.get("pattern") or question)
        payload = f"{tool}:{query}"
    elif tool in {"apply_patch", "write_file", "create_file"}:
        path = _norm_path(args.get("path") or args.get("file") or args.get("target_file"))
        payload = f"{tool}:{path or _normalize_text(question)[:160]}"
    else:
        blob = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
        payload = f"{tool or 'request'}:{_normalize_text(question)[:160]}:{blob}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


class WorkItem(BaseModel):
    """A single job on the queue."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    kind: WorkKind
    title: str = ""
    gate: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = Field(default_factory=dict)
    question: str = ""
    priority: int = 50  # lower runs first
    dependencies: list[str] = Field(default_factory=list)
    fingerprint: str = ""
    status: WorkStatus = "pending"
    attempts: int = 0
    max_attempts: int = 2
    created_by: str = "planner"
    result_summary: str = ""
    error: str = ""
    files_read: list[str] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
    files_discovered: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)

    def model_post_init(self, _ctx: Any) -> None:  # noqa: D401 - pydantic hook
        if not self.fingerprint:
            self.fingerprint = compute_fingerprint(
                kind=self.kind,
                tool_name=self.tool_name,
                tool_args=self.tool_args,
                question=self.question,
            )
        if not self.title:
            self.title = (self.question or f"{self.tool_name or self.kind}").strip()[:120]

    @property
    def terminal(self) -> bool:
        return self.status in {"done", "failed", "blocked", "skipped"}


class WorkResult(BaseModel):
    """Outcome of executing a single job."""

    ok: bool
    summary: str = ""
    error: str = ""
    files_read: list[str] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
    files_discovered: list[str] = Field(default_factory=list)
    answer: str = ""
    sources: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    duration_ms: float = 0.0


class WorkEvent(BaseModel):
    type: str
    item_id: str = ""
    kind: str = ""
    status: str = ""
    title: str = ""
    at: str = Field(default_factory=_utc_now)
    data: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# EventBus
# --------------------------------------------------------------------------- #
class EventBus:
    """Thread-safe publish/subscribe channel for work-queue status updates."""

    def __init__(self, *, keep_history: int = 2000) -> None:
        self._subscribers: list[Callable[[WorkEvent], None]] = []
        self._history: list[WorkEvent] = []
        self._keep = max(0, int(keep_history))
        self._lock = threading.RLock()

    def subscribe(self, callback: Callable[[WorkEvent], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def _unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return _unsubscribe

    def publish(self, event: WorkEvent) -> None:
        with self._lock:
            if self._keep:
                self._history.append(event)
                if len(self._history) > self._keep:
                    del self._history[: len(self._history) - self._keep]
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:  # subscribers must never break the run
                logger.debug("event subscriber raised", exc_info=True)

    def history(self) -> list[WorkEvent]:
        with self._lock:
            return list(self._history)


# --------------------------------------------------------------------------- #
# AgentWorkQueue
# --------------------------------------------------------------------------- #
class AgentWorkQueue:
    """Dependency-aware, fingerprint-deduplicated job queue.

    The planner (and the live sniffer) ``submit`` jobs; the executor ``claim``s
    the next runnable one. Jobs only become runnable when every dependency has
    completed successfully; if a dependency fails the dependents are marked
    ``blocked`` rather than silently retried forever.
    """

    def __init__(self, *, bus: EventBus | None = None) -> None:
        self._bus = bus or EventBus()
        self._items: dict[str, WorkItem] = {}
        self._order: list[str] = []
        self._fingerprints: dict[str, str] = {}  # fingerprint -> item id
        self._lock = threading.RLock()

    @property
    def bus(self) -> EventBus:
        return self._bus

    # -- submission -------------------------------------------------------- #
    def submit(self, item: WorkItem) -> bool:
        """Add a job. Returns ``False`` if a duplicate was suppressed."""
        with self._lock:
            existing_id = self._fingerprints.get(item.fingerprint)
            if existing_id is not None:
                existing = self._items[existing_id]
                # A finished idempotent action never needs to run again.
                if existing.kind in _IDEMPOTENT_KINDS and existing.status in {"done", "running", "ready", "pending"}:
                    self._emit("job_skipped_duplicate", item, status="skipped")
                    return False
                if not existing.terminal:
                    self._emit("job_skipped_duplicate", item, status="skipped")
                    return False
            self._items[item.id] = item
            self._order.append(item.id)
            self._fingerprints[item.fingerprint] = item.id
            self._recompute_readiness_locked()
            self._emit("job_submitted", item, status=item.status)
            return True

    def submit_many(self, items: Iterable[WorkItem]) -> int:
        return sum(1 for item in items if self.submit(item))

    # -- claiming ---------------------------------------------------------- #
    def claim(self) -> WorkItem | None:
        """Return the highest-priority runnable job, marking it ``running``."""
        with self._lock:
            self._recompute_readiness_locked()
            ready = [self._items[i] for i in self._order if self._items[i].status == "ready"]
            if not ready:
                return None
            ready.sort(key=lambda it: (it.priority, it.created_at, it.id))
            item = ready[0]
            item.status = "running"
            item.attempts += 1
            item.updated_at = _utc_now()
            self._emit("job_running", item, status="running")
            return item

    # -- completion -------------------------------------------------------- #
    def complete(
        self,
        item_id: str,
        *,
        status: WorkStatus,
        result: WorkResult | None = None,
    ) -> WorkItem:
        with self._lock:
            item = self._items[item_id]
            item.status = status
            item.updated_at = _utc_now()
            if result is not None:
                item.result_summary = result.summary
                item.error = result.error
                item.files_read = list(result.files_read)
                item.files_changed = list(result.files_changed)
                item.files_discovered = list(result.files_discovered)
            self._recompute_readiness_locked()
            self._emit(f"job_{status}", item, status=status)
            return item

    def requeue(self, item_id: str) -> None:
        """Put a transiently-failed job back into contention for retry."""
        with self._lock:
            item = self._items[item_id]
            if item.attempts < item.max_attempts:
                item.status = "pending"
                item.updated_at = _utc_now()
                self._recompute_readiness_locked()
                self._emit("job_requeued", item, status=item.status)

    # -- introspection ----------------------------------------------------- #
    def get(self, item_id: str) -> WorkItem | None:
        with self._lock:
            return self._items.get(item_id)

    def items(self) -> list[WorkItem]:
        with self._lock:
            return [self._items[i] for i in self._order]

    def has_fingerprint(self, fingerprint: str) -> bool:
        with self._lock:
            return fingerprint in self._fingerprints

    def is_drained(self) -> bool:
        """True when no job is runnable or in flight."""
        with self._lock:
            self._recompute_readiness_locked()
            for item in self._items.values():
                if item.status in {"pending", "ready", "running"}:
                    return False
            return True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            for item in self._items.values():
                counts[item.status] = counts.get(item.status, 0) + 1
            return {
                "total": len(self._items),
                "counts": counts,
                "done": counts.get("done", 0),
                "failed": counts.get("failed", 0),
                "blocked": counts.get("blocked", 0),
                "skipped": counts.get("skipped", 0),
                "remaining": counts.get("pending", 0) + counts.get("ready", 0) + counts.get("running", 0),
            }

    # -- internals --------------------------------------------------------- #
    def _recompute_readiness_locked(self) -> None:
        for item in self._items.values():
            if item.status not in {"pending", "ready"}:
                continue
            dep_states = [self._items.get(dep) for dep in item.dependencies]
            if any(dep is not None and dep.status in {"failed", "blocked"} for dep in dep_states):
                item.status = "blocked"
                self._emit("job_blocked", item, status="blocked")
                continue
            satisfied = all(dep is not None and dep.status == "done" for dep in dep_states) if item.dependencies else True
            item.status = "ready" if satisfied else "pending"

    def _emit(self, event_type: str, item: WorkItem, *, status: str) -> None:
        self._bus.publish(
            WorkEvent(
                type=event_type,
                item_id=item.id,
                kind=item.kind,
                status=status,
                title=item.title,
                data={
                    "gate": item.gate,
                    "tool_name": item.tool_name,
                    "attempts": item.attempts,
                    "fingerprint": item.fingerprint,
                    "created_by": item.created_by,
                    "error": item.error,
                },
            )
        )


# --------------------------------------------------------------------------- #
# TaskBoard
# --------------------------------------------------------------------------- #
class TaskBoard:
    """Live, renderable progress view fed by the EventBus."""

    def __init__(self, *, queue: AgentWorkQueue) -> None:
        self._queue = queue
        self._events: list[WorkEvent] = []
        self._lock = threading.RLock()
        self._unsub = queue.bus.subscribe(self._on_event)

    def _on_event(self, event: WorkEvent) -> None:
        with self._lock:
            self._events.append(event)

    def close(self) -> None:
        self._unsub()

    def recent_events(self, limit: int = 20) -> list[WorkEvent]:
        with self._lock:
            return self._events[-limit:]

    def snapshot(self) -> dict[str, Any]:
        return self._queue.snapshot()

    def render(self) -> str:
        items = self._queue.items()
        snap = self._queue.snapshot()
        glyph = {
            "done": "[x]",
            "running": "[~]",
            "ready": "[ ]",
            "pending": "[.]",
            "failed": "[!]",
            "blocked": "[#]",
            "skipped": "[-]",
        }
        lines = [
            f"Work board: {snap['done']} done / {snap['remaining']} remaining"
            f" / {snap['failed']} failed / {snap['blocked']} blocked / {snap['skipped']} skipped",
        ]
        for item in sorted(items, key=lambda it: (it.priority, it.created_at)):
            mark = glyph.get(item.status, "[?]")
            origin = "" if item.created_by == "planner" else f" <-{item.created_by}"
            lines.append(f"  {mark} {item.kind:9} {item.title[:64]}{origin}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# JobSniffer -- the coding agent's live steering hook
# --------------------------------------------------------------------------- #
class JobSniffer(Protocol):
    """The coding agent implements this to live-analyze results and emit jobs.

    Called after every job finishes. Returning new :class:`WorkItem` objects
    enqueues them; the queue's fingerprint dedup makes it safe to over-emit.
    """

    def on_result(self, item: WorkItem, result: WorkResult, *, board: TaskBoard) -> list[WorkItem]:
        ...


# --------------------------------------------------------------------------- #
# WorkQueueRunner -- the live loop
# --------------------------------------------------------------------------- #
class RunReport(BaseModel):
    steps: int = 0
    done: int = 0
    failed: int = 0
    blocked: int = 0
    skipped: int = 0
    emitted_by_sniffer: int = 0
    duration_ms: float = 0.0
    terminal_reason: str = ""
    items: list[WorkItem] = Field(default_factory=list)


class WorkQueueRunner:
    """Drives the queue: claim -> execute -> classify -> broadcast -> sniff."""

    def __init__(
        self,
        *,
        queue: AgentWorkQueue,
        execute: Callable[[WorkItem], WorkResult],
        sniffer: JobSniffer | None = None,
        board: TaskBoard | None = None,
        max_steps: int = 60,
    ) -> None:
        self._queue = queue
        self._execute = execute
        self._sniffer = sniffer
        self._board = board or TaskBoard(queue=queue)
        self._max_steps = max(1, int(max_steps))

    @property
    def board(self) -> TaskBoard:
        return self._board

    def run(self) -> RunReport:
        t0 = time.perf_counter()
        steps = 0
        emitted = 0
        terminal_reason = "drained"
        while steps < self._max_steps:
            item = self._queue.claim()
            if item is None:
                terminal_reason = "drained" if self._queue.is_drained() else "no_runnable_jobs"
                break
            steps += 1
            result = self._safe_execute(item)
            status = self._classify(item, result)
            if status == "retry":
                self._queue.requeue(item.id)
            else:
                self._queue.complete(item.id, status=status, result=result)
                if self._sniffer is not None and status == "done":
                    emitted += self._run_sniffer(item, result)
        else:
            terminal_reason = "step_budget_exhausted"

        snap = self._queue.snapshot()
        return RunReport(
            steps=steps,
            done=snap["done"],
            failed=snap["failed"],
            blocked=snap["blocked"],
            skipped=snap["skipped"],
            emitted_by_sniffer=emitted,
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            terminal_reason=terminal_reason,
            items=self._queue.items(),
        )

    def _safe_execute(self, item: WorkItem) -> WorkResult:
        try:
            return self._execute(item)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("work item execution raised: %s", item.id)
            return WorkResult(ok=False, error=f"executor_exception: {exc}")

    def _classify(self, item: WorkItem, result: WorkResult) -> WorkStatus | Literal["retry"]:
        if result.ok:
            return "done"
        if item.attempts < item.max_attempts:
            return "retry"
        return "failed"

    def _run_sniffer(self, item: WorkItem, result: WorkResult) -> int:
        try:
            new_items = self._sniffer.on_result(item, result, board=self._board) or []
        except Exception:  # a bad sniffer must not crash the run
            logger.debug("sniffer raised", exc_info=True)
            return 0
        return self._queue.submit_many(new_items)


__all__ = [
    "AgentWorkQueue",
    "EventBus",
    "JobSniffer",
    "RunReport",
    "TaskBoard",
    "WorkEvent",
    "WorkItem",
    "WorkKind",
    "WorkQueueRunner",
    "WorkResult",
    "WorkStatus",
    "compute_fingerprint",
]
