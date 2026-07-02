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
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Protocol, Sequence

from pydantic import BaseModel, Field

from mana_agent.llm.goal_profiles import active_goal_profile
from mana_agent.llm.tool_worker_process import ToolWorkerClient
from mana_agent.llm.tools_executor import ToolsExecutionConfig, ToolsExecutor
from mana_agent.llm.tools_manager import (
    AutoExecuteResult,
    RunStateStore,
    _cleanup_stray_deliverables,
    _compose_final_answer,
    _extract_changed_files_from_value,
    _failed_tool_calls_from_trace,
    _forced_mutation_prompt,
    _latest_useful_answer,
    _looks_like_stub,
    _missing_required_files,
    _mutation_required_from_policy,
    _mutation_required_from_text,
    _mutation_state_from_trace,
    _mutation_tool_stats,
    _resolve_mutation_target_path,
    _resolve_required_deliverables,
    _salvage_misplaced_deliverables,
    _verification_summary_from_trace,
)
from mana_agent.services.coding_memory_service import CodingMemoryService
from mana_agent.services.coding_todo_service import TodoService

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
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            return str(Path(text).resolve())
        except Exception:
            return _normalize_text(value).replace("\\", "/").lstrip("./").strip("/")

    if tool == "read_file":
        path = _norm_path(args.get("path") or args.get("file") or args.get("file_path"))
        payload = f"read_file:{path}" if path else f"read_file:{_normalize_text(question)[:160]}"
    elif tool in {"repo_search", "semantic_search", "list_files"}:
        query = _normalize_text(args.get("query") or args.get("q") or args.get("pattern") or question)
        payload = f"{tool}:{query}"
    elif tool in {"edit_file", "multi_edit_file", "apply_patch", "write_file", "create_file", "delete_file"}:
        path = _norm_path(args.get("path") or args.get("file") or args.get("target_file"))
        payload = f"{tool}:{path or _normalize_text(question)[:160]}"
    else:
        blob = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
        payload = f"{tool or 'request'}:{_normalize_text(question)[:160]}:{blob}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


_DEFAULT_SKILL_TERMS = ("default skill", "default skills", "built-in skill", "builtin skill", "skill registry")


def _looks_like_default_skill_registry_request(request: str) -> bool:
    text = str(request or "").lower()
    if not any(term in text for term in _DEFAULT_SKILL_TERMS):
        return False
    return any(name in text for name in ("nestjs", "nextjs", "reactjs", "fastapi"))


def _default_skill_target_files(request: str) -> list[str]:
    text = str(request or "").lower()
    targets = ["src/mana_agent/skills/manager.py"]
    for name in ("nestjs", "nextjs", "reactjs", "fastapi"):
        if name in text:
            targets.append(f"src/mana_agent/default_skills/{name}.md")
    return targets


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


# --------------------------------------------------------------------------- #
# QueueManager
# --------------------------------------------------------------------------- #
class QueueManager:
    """Live Agent Work Queue manager (replaces the legacy planner pass-loop).

    Seeds a discovery job from the request, then lets the ``AgentWorkQueue`` and
    the coding agent's ``CodingAgentSniffer`` drive tool execution:
    claim -> execute -> broadcast -> sniff -> emit follow-up jobs. Returns an
    :class:`AutoExecuteResult` so existing chat/CLI callers keep working.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "",
        worker_client: ToolWorkerClient,
        repo_root: Path,
        base_url: str | None = None,
        execution_config: ToolsExecutionConfig | None = None,
        executor: ToolsExecutor | None = None,
        coding_memory_service: CodingMemoryService | None = None,
        todo_service: TodoService | None = None,
        decision_provider: Any = None,
    ) -> None:
        _ = (api_key, model, base_url, decision_provider)
        self.worker_client = worker_client
        self.repo_root = Path(repo_root).resolve()
        self.execution_config = execution_config or ToolsExecutionConfig()
        self.executor = executor
        self.coding_memory_service = coding_memory_service
        # The todo ledger is backed by the same flow-memory store. When a memory
        # service is present we always have a ledger, so the prechecklist can be
        # materialized into durable, status-tracked todos.
        self.todo_service = todo_service or (
            TodoService(memory=coding_memory_service) if coding_memory_service is not None else None
        )
        self._decision_provider = decision_provider

    def attach_decision_provider(self, provider: Any) -> None:
        # The provider (typically the CodingAgent) supplies the LLM planner used
        # by ``preview_plan`` for accurate checklists. The execution loop itself
        # stays deterministic; only pre-execution planning consults the provider.
        self._decision_provider = provider

    def update_model(self, new_model: str) -> None:
        logger.info("Ignoring model update; QueueManager is deterministic-only.")

    def preview_plan(
        self,
        *,
        request: str,
        flow_context: str | None = None,
        flow_id: str | None = None,
        target_files: Sequence[str] = (),
        requires_edit: bool | None = None,
        pass_cap: int = 4,
        **_ignored: Any,
    ) -> dict[str, Any]:
        """Build a pre-execution checklist and materialize it as durable todos.

        Planning strategy (accuracy first, cost-bounded):

        1. **Delegate to the attached decision provider** (the CodingAgent) when
           it exposes ``preview_execution_checklist``. That path runs the LLM
           planner *and* persists flow memory (ensure_flow / build_flow_context /
           persist_preview_checklist) in one place, giving the most accurate
           plan. Memory keeps it cheap: prior turns are reused as context and the
           preview is cached on the flow.
        2. **Deterministic fallback** when no provider is attached — still
           memory-backed (the flow is ensured and the checklist persisted), so
           continuity holds even without an LLM.

        Either way the resulting prechecklist is synced into the todo ledger via
        :class:`TodoService`, so the plan becomes status-tracked todos tied to
        the flow rather than throwaway UI text.
        """
        _ = pass_cap
        provider = self._decision_provider
        if provider is not None and hasattr(provider, "preview_execution_checklist"):
            try:
                payload = provider.preview_execution_checklist(
                    request,
                    flow_id=flow_id,
                    flow_context=flow_context,
                )
            except Exception as exc:  # pragma: no cover - provider failure -> deterministic
                logger.warning("decision provider preview failed; using deterministic preview: %s", exc)
                payload = self._deterministic_preview(
                    request,
                    flow_id=flow_id,
                    flow_context=flow_context,
                    target_files=target_files,
                    requires_edit=requires_edit,
                    warnings=[f"decision provider preview failed: {exc}"],
                )
        else:
            payload = self._deterministic_preview(
                request,
                flow_id=flow_id,
                flow_context=flow_context,
                target_files=target_files,
                requires_edit=requires_edit,
            )

        self._sync_preview_todos(payload)
        return payload

    def _sync_preview_todos(self, payload: dict[str, Any]) -> None:
        """Connect the preview's prechecklist to the durable todo ledger."""
        if self.todo_service is None:
            return
        flow_id = str(payload.get("flow_id") or "").strip()
        prechecklist = payload.get("prechecklist")
        if not flow_id or not isinstance(prechecklist, dict):
            return
        try:
            payload["todos"] = self.todo_service.sync_from_preview(
                flow_id=flow_id,
                prechecklist=prechecklist,
                source=str(payload.get("prechecklist_source", "") or ""),
            )
        except Exception as exc:  # pragma: no cover - ledger is best-effort
            logger.warning("todo sync from preview failed: %s", exc)
            payload.setdefault("warnings", [])
            if isinstance(payload["warnings"], list):
                payload["warnings"].append(f"todo sync failed: {exc}")

    def _deterministic_preview(
        self,
        request: str,
        *,
        flow_id: str | None,
        flow_context: str | None,
        target_files: Sequence[str] = (),
        requires_edit: bool | None = None,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build a memory-backed prechecklist without an LLM.

        Reuses the same deterministic deliverable/mutation analysis the run loop
        relies on, so the preview and the execution agree on intent. The flow is
        ensured and the checklist persisted through the memory service so the UI
        and flow views stay consistent with the LLM-backed path.
        """
        warnings = list(warnings or [])
        active_flow_id = flow_id
        effective_flow_context = flow_context
        memory = self.coding_memory_service

        if memory is not None:
            try:
                active_flow_id = memory.ensure_flow(flow_id=flow_id, request=request)
                if effective_flow_context is None:
                    effective_flow_context = memory.build_flow_context(active_flow_id, [])
            except Exception as exc:
                warnings.append(f"coding memory setup failed: {exc}")

        deliverables = _resolve_required_deliverables(request, self.repo_root, tuple(target_files))
        mutation_required = (
            bool(requires_edit)
            if requires_edit is not None
            else _mutation_required_from_text(request) or bool(deliverables)
        )

        steps: list[dict[str, str]] = [
            {"id": "discover", "title": f"Locate files relevant to: {request[:120]}", "status": "pending"},
            {"id": "read", "title": "Read the candidate files to ground the change", "status": "pending"},
        ]
        if mutation_required:
            targets_label = ", ".join(deliverables) if deliverables else "the target file(s)"
            steps.append(
                {
                    "id": "edit",
                    "title": (
                        "Apply the requested change and update all related imports, exports, "
                        "registries, routers, commands, call sites, tests, and docs required "
                        f"for {targets_label} to remain working"
                    ),
                    "status": "pending",
                    "requires_tools": ["edit_file", "multi_edit_file", "apply_patch", "write_file", "create_file", "delete_file"],
                    "checks": [
                        "target file changed/created/deleted",
                        "related imports/usages updated",
                        "integration path updated",
                        "stale references removed",
                        "verification selected and executed when possible",
                    ],
                }
            )
            steps.append(
                {
                    "id": "verify",
                    "title": "Verify the change (build/tests/static checks)",
                    "status": "pending",
                    "requires_tools": ["verify_project"],
                }
            )
        else:
            steps.append({"id": "answer", "title": "Compose the grounded answer", "status": "pending"})

        prechecklist = {
            "objective": request.strip()[:200],
            "requires_edit": mutation_required,
            "target_files": list(deliverables),
            "steps": steps,
            "source": "deterministic",
        }

        if memory is not None and active_flow_id:
            try:
                memory.persist_preview_checklist(
                    flow_id=active_flow_id,
                    user_request=request,
                    checklist=prechecklist,
                    source="deterministic",
                )
            except Exception as exc:
                warnings.append(f"coding memory preview persistence failed: {exc}")

        return {
            "flow_id": active_flow_id,
            "flow_context": effective_flow_context,
            "prechecklist": prechecklist,
            "prechecklist_source": "deterministic",
            "prechecklist_warning": "",
            "requires_edit": mutation_required,
            "target_files": list(deliverables),
            "warnings": warnings,
        }

    def _persist_turn_and_todos(
        self,
        *,
        request: str,
        flow_id: str | None,
        flow_context: str | None,
        answer: str,
        changed_files: list[str],
        warnings: list[str],
        mutation_succeeded: bool,
        verification_passed: bool,
        run_blocked: bool,
    ) -> None:
        """Record the completed turn to flow memory and reconcile todos."""
        memory = self.coding_memory_service
        if memory is None:
            return
        try:
            active_flow_id = memory.ensure_flow(flow_id=flow_id, request=request)
        except Exception as exc:  # pragma: no cover - memory is best-effort
            logger.warning("ensure_flow after run failed: %s", exc)
            return
        try:
            memory.record_turn(
                flow_id=active_flow_id,
                user_request=request,
                effective_prompt=(flow_context or "")[:4000],
                agent_answer=answer or "",
                changed_files=list(changed_files),
                warnings=list(warnings),
                static_findings=[],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("record_turn after run failed: %s", exc)
        if self.todo_service is not None:
            try:
                self.todo_service.reconcile_after_run(
                    flow_id=active_flow_id,
                    changed_files=list(changed_files),
                    mutation_succeeded=mutation_succeeded,
                    verification_passed=verification_passed,
                    run_blocked=run_blocked,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("todo reconcile after run failed: %s", exc)

    def run(
        self,
        *,
        request: str,
        flow_context: str | None = None,
        index_dir: str | Path | None = None,
        index_dirs: Sequence[str | Path] | None = None,
        k: int = 8,
        max_steps: int = 6,
        timeout_seconds: int = 60,
        tool_policy: dict[str, Any] | None = None,
        pass_cap: int = 4,
        on_event: Callable[[Any], None] | None = None,
        flow_id: str | None = None,
        run_id: str | None = None,
        max_no_progress_passes: int = 2,
        requires_edit: bool | None = None,
        target_files: Sequence[str] = (),
    ) -> AutoExecuteResult:
        from mana_agent.llm.agent_work_queue_adapters import (
            CodingAgentSniffer,
            make_worker_executor,
        )

        _ = (index_dirs, max_no_progress_passes, flow_context)
        store = RunStateStore(repo_root=self.repo_root, run_id=run_id)
        resolved_tool_policy = dict(tool_policy or {})
        default_skill_registry_request = _looks_like_default_skill_registry_request(request)
        if default_skill_registry_request:
            requires_edit = True if requires_edit is None else requires_edit
            if not target_files:
                target_files = _default_skill_target_files(request)
            resolved_tool_policy.setdefault("search_budget", 3)
            resolved_tool_policy.setdefault("read_budget", 3)
            resolved_tool_policy.setdefault("search_repeat_limit", 1)
        mutation_required = (
            _mutation_required_from_policy(resolved_tool_policy, requires_edit)
            or _mutation_required_from_text(request)
            or default_skill_registry_request
        )
        if mutation_required:
            resolved_tool_policy["mutation_required"] = True
        required_files = _resolve_required_deliverables(request, self.repo_root, target_files)
        resolved_target_path = (
            required_files[0]
            if required_files
            else _resolve_mutation_target_path(request, self.repo_root, target_files)
        )
        queue = AgentWorkQueue()
        board = TaskBoard(queue=queue)
        profile = active_goal_profile(request)
        if profile is not None:
            def _relevant(path: str) -> bool:
                return profile.is_relevant(path, self.repo_root)
        else:
            def _relevant(path: str) -> bool:
                return True

        if default_skill_registry_request:
            queue.submit(
                WorkItem(
                    kind="discover",
                    tool_name="repo_search",
                    tool_args={"query": "DEFAULT_SKILL_NAMES"},
                    question=(
                        "Locate the built-in skill registry by searching DEFAULT_SKILL_NAMES. "
                        "Prefer src/mana_agent/skills/manager.py and ignore dependency detection files."
                    ),
                    gate="locate_skill_registry",
                    priority=10,
                )
            )
            queue.submit(
                WorkItem(
                    kind="discover",
                    tool_name="list_files",
                    tool_args={"glob": "src/mana_agent/default_skills/*.md"},
                    question="List flat built-in skill markdown files under src/mana_agent/default_skills/*.md.",
                    gate="locate_builtin_skill_files",
                    priority=11,
                )
            )
        else:
            queue.submit(
                WorkItem(
                    kind="discover",
                    tool_name="repo_search",
                    tool_args={"query": request},
                    question=f"Locate files relevant to: {request}",
                    gate="locate_candidates",
                    priority=10,
                )
            )

        answers: list[str] = []
        sources: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        changed_files: list[str] = []
        mutation_state: dict[str, Any] = {
            "mutation_attempted": False,
            "mutation_succeeded": False,
            "changed_files": [],
            "no_op_reason": "",
            "verify_requires_mutation": False,
        }
        base_execute = make_worker_executor(
            worker_client=self.worker_client,
            repo_root=self.repo_root,
            on_event=on_event,
            default_timeout=int(timeout_seconds),
            default_k=int(k),
            default_max_steps=int(max_steps),
            tool_policy=resolved_tool_policy,
            index_dir=str(index_dir) if index_dir else None,
            flow_id=flow_id,
            run_id=store.run_id,
        )

        def execute(item: "WorkItem"):  # noqa: F821 - imported above
            nonlocal mutation_state
            if mutation_required and item.kind == "verify" and not mutation_state.get("mutation_succeeded"):
                blocked_trace = [
                    {
                        "tool_name": item.tool_name or "verify_project",
                        "status": "verify_project_blocked_until_mutation",
                        "error": "verify_project_blocked_until_mutation",
                    }
                ]
                trace.extend(blocked_trace)
                mutation_state = _mutation_state_from_trace(trace, changed_files)
                return WorkResult(
                    ok=False,
                    summary="verify_project_blocked_until_mutation",
                    error="verify_project_blocked_until_mutation",
                    trace=blocked_trace,
                )
            result = base_execute(item)
            if result.answer:
                answers.append(result.answer)
            sources.extend(result.sources)
            trace.extend(result.trace)
            changed_files.extend(result.files_changed)
            changed_files.extend(_extract_changed_files_from_value(result.trace))
            changed_files[:] = sorted(dict.fromkeys(path for path in changed_files if path))
            mutation_state = _mutation_state_from_trace(trace, changed_files)
            return result

        # Whether to finalize with an edit + verify is recognized upstream by the
        # coding agent's planner (the LLM checklist) and passed in as requires_edit.
        sniffer = CodingAgentSniffer(
            repo_root=self.repo_root,
            request=request,
            emit_edit=requires_edit,
            target_files=[str(item).strip() for item in target_files if str(item).strip()],
            relevant=_relevant,
        )
        runner = WorkQueueRunner(
            queue=queue,
            execute=execute,
            sniffer=sniffer,
            board=board,
            max_steps=max(12, int(pass_cap) * 8),
        )
        report = runner.run()
        forced_retry_ran = False
        forced_retry_mutation_attempted = False
        forced_retry_changed_files = False
        warnings: list[str] = []

        def _refresh_mutation_state() -> None:
            nonlocal mutation_state
            changed_files[:] = sorted(dict.fromkeys(path for path in changed_files if path))
            mutation_state = _mutation_state_from_trace(trace, changed_files)

        # The deliverables the request demands. The run is held to ALL of them,
        # not just the first, so "create A and B in docs" cannot finish with only
        # one file written. Falls back to the single primary target when the
        # request named no explicit files (e.g. generic "fix the bug").
        deliverables = list(required_files) or ([resolved_target_path] if resolved_target_path else [])

        # --- Forced mutation retry: one file at a time, mutation tool required. ---
        # This is the AGENTIC content path and runs *before* the deterministic
        # template fallback so the worker authors real, file-specific content
        # (a genuine overview vs. a genuine installation guide), not boilerplate.
        # For any deliverable still missing after the main pass (or, when no files
        # were named, when nothing was mutated) re-run the worker under a strict
        # MUTATION_REQUIRED prompt that forbids natural-language-only answers.
        if mutation_required and not mutation_state.get("no_op_reason"):
            if deliverables:
                forced_targets = _missing_required_files(
                    self.repo_root, deliverables, changed=set(mutation_state.get("changed_files") or [])
                )
            elif not mutation_state.get("mutation_succeeded"):
                forced_targets = [resolved_target_path]
            else:
                forced_targets = []
            for target_file in forced_targets:
                forced_retry_ran = True
                forced_policy = {
                    **resolved_tool_policy,
                    "mutation_required": True,
                    "mutation_strict": True,
                    # Agentic authoring: the worker may inspect the repo to ground
                    # the file, then must end with a mutation. The executor's edit
                    # branch enforces the same toolset; this keeps them aligned.
                    "allowed_tools": [
                        "read_file",
                        "repo_search",
                        "semantic_search",
                        "list_files",
                        "ls",
                        "find_symbols",
                        "edit_file",
                        "multi_edit_file",
                        "apply_patch",
                        "write_file",
                        "create_file",
                        "delete_file",
                        "git_diff",
                        "git_status",
                    ],
                    "verify_requires_mutation": True,
                }
                forced_execute = make_worker_executor(
                    worker_client=self.worker_client,
                    repo_root=self.repo_root,
                    on_event=on_event,
                    default_timeout=int(timeout_seconds),
                    default_k=int(k),
                    default_max_steps=max(1, int(max_steps)),
                    tool_policy=forced_policy,
                    index_dir=str(index_dir) if index_dir else None,
                    flow_id=flow_id,
                    run_id=store.run_id,
                )
                forced_item = WorkItem(
                    kind="edit",
                    tool_name="write_file" if target_file and (self.repo_root / target_file).exists() else ("create_file" if target_file else ""),
                    tool_args={"path": target_file} if target_file else {},
                    question=_forced_mutation_prompt(request, target_file),
                    gate="apply_changes",
                    priority=1,
                    created_by="forced_mutation_retry",
                )
                forced_result = forced_execute(forced_item)
                if forced_result.answer:
                    answers.append(forced_result.answer)
                sources.extend(forced_result.sources)
                trace.extend(forced_result.trace)
                changed_files.extend(forced_result.files_changed)
                changed_files.extend(_extract_changed_files_from_value(forced_result.trace))
                _refresh_mutation_state()
            if forced_retry_ran:
                forced_retry_mutation_attempted = bool(mutation_state.get("mutation_attempted"))
                forced_retry_changed_files = bool(mutation_state.get("changed_files"))
                if not forced_retry_mutation_attempted:
                    warnings.append("forced_mutation_retry_no_mutation_tool_attempted")
                elif not forced_retry_changed_files:
                    warnings.append("forced_mutation_retry_no_changed_files")

        # --- Path reconciliation: salvage content written to the wrong path. ---
        # The worker sometimes writes "01-overview.md" at the repo root instead of
        # the requested "docs/01-overview.md". Move substantial misplaced content
        # into the required path before deciding what still needs generating.
        removed_strays: list[str] = []
        if mutation_required and not mutation_state.get("no_op_reason") and deliverables:
            relocated = _salvage_misplaced_deliverables(self.repo_root, deliverables, changed_files, trace)
            removed_strays.extend(relocated)
            if relocated:
                _refresh_mutation_state()

        # --- Honest-failure accounting: NO fabricated content. ---
        # There is deliberately no template/boilerplate generator behind the
        # agentic pass. Deliverable content must be authored by the worker after
        # it analyzes the project; if a deliverable is still missing or is only a
        # placeholder stub the worker left behind, we surface that as a failure
        # (the verification gate below blocks the run) rather than inventing
        # content. This keeps every produced file genuinely project-grounded.
        if mutation_required and not mutation_state.get("no_op_reason") and deliverables:
            for path in deliverables:
                target_abs = self.repo_root / path
                stub_left_behind = False
                if target_abs.is_file():
                    try:
                        existing = target_abs.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        existing = ""
                    stub_left_behind = _looks_like_stub(existing)
                if (not target_abs.is_file()) or stub_left_behind:
                    warnings.append(f"deliverable_not_authored:{path}")

        # --- Cleanup: remove leftover wrong-path duplicates now satisfied. ---
        if mutation_required and not mutation_state.get("no_op_reason") and deliverables:
            removed_strays.extend(
                _cleanup_stray_deliverables(self.repo_root, deliverables, changed_files, trace)
            )

        # --- Verification gate: every required deliverable must exist on disk. ---
        missing_required_files = _missing_required_files(
            self.repo_root, deliverables, changed=set(mutation_state.get("changed_files") or [])
        )
        terminal_reason = report.terminal_reason
        run_status = "completed"
        if mutation_required and not mutation_state.get("no_op_reason"):
            if not mutation_state.get("mutation_attempted"):
                run_status = "blocked"
                terminal_reason = "mutation_required_but_no_mutation_tool_attempted"
            elif not mutation_state.get("changed_files"):
                run_status = "blocked"
                terminal_reason = "mutation_required_but_no_changed_files"
            elif missing_required_files:
                run_status = "blocked"
                terminal_reason = "mutation_required_but_missing_files"
        # Relocated/cleaned-up strays no longer exist on disk, so drop them from
        # the reported changed files regardless of what the trace remembers.
        _removed = set(removed_strays)
        changed_files = [p for p in (mutation_state.get("changed_files") or changed_files) if p not in _removed]
        # Final answer is rebuilt from authoritative execution state (trace,
        # changed_files, mutation_state, verification), never from the last
        # natural-language worker answer, so an intermediate "I could not edit"
        # cannot contradict a trace that proves a mutation landed.
        verification = _verification_summary_from_trace(trace)
        failed_calls = _failed_tool_calls_from_trace(trace)
        for failure in failed_calls:
            warning = f"tool_call_failed:{failure['tool']}"
            if warning not in warnings:
                warnings.append(warning)
        final_answer = _compose_final_answer(
            mutation_required=mutation_required,
            mutation_state=mutation_state,
            changed_files=changed_files,
            verification=verification,
            run_status=run_status,
            terminal_reason=terminal_reason,
            worker_answer=_latest_useful_answer(answers),
            fallback=board.render(),
            missing_required_files=missing_required_files,
        )
        mutation_tool_stats = _mutation_tool_stats(trace)
        verification_passed = bool(
            (not mutation_required or not missing_required_files)
            and (not verification.get("ran") or verification.get("passed"))
        )
        # Write-side flow continuity: persist this turn and reconcile the todo
        # ledger from authoritative results so the next turn (and the UI) reflect
        # what actually happened. Best-effort: memory failures never fail a run.
        self._persist_turn_and_todos(
            request=request,
            flow_id=flow_id,
            flow_context=flow_context,
            answer=final_answer,
            changed_files=changed_files,
            warnings=warnings,
            mutation_succeeded=bool(mutation_state.get("mutation_succeeded")),
            verification_passed=verification_passed,
            run_blocked=(run_status == "blocked"),
        )
        return AutoExecuteResult(
            answer=final_answer,
            sources=sources,
            trace=trace,
            warnings=warnings,
            changed_files=changed_files,
            passes=report.steps,
            terminal_reason=terminal_reason,
            toolsmanager_requests_count=report.steps,
            execution_backend="work_queue",
            execution_run_id=store.run_id,
            execution_duration_ms=report.duration_ms,
            execution_requests_ok=report.done,
            execution_requests_failed=report.failed,
            pass_logs=[{"made_progress": report.done > 0, "board": board.render()}],
            planner_decisions=[
                {
                    "mutation_required": mutation_required,
                    "mutation_attempted": bool(mutation_state.get("mutation_attempted")),
                    "mutation_succeeded": bool(mutation_state.get("mutation_succeeded")),
                    "changed_files": changed_files,
                    "no_op_reason": str(mutation_state.get("no_op_reason") or ""),
                    "verify_requires_mutation": bool(mutation_state.get("verify_requires_mutation")),
                    "forced_mutation_retry_ran": forced_retry_ran,
                    "forced_retry_mutation_attempted": forced_retry_mutation_attempted,
                    "forced_retry_changed_files": forced_retry_changed_files,
                    "verification_ran": bool(verification.get("ran")),
                    "verification_passed": bool(verification.get("passed")),
                    "verification_failed": bool(verification.get("failed")),
                    "verification_failing_checks": list(verification.get("failing", [])),
                    "mutation_tool_attempted": bool(mutation_state.get("mutation_attempted")),
                    "mutation_tool_successful": bool(mutation_state.get("mutation_succeeded")),
                    "mutation_fallback_count": int(bool(forced_retry_ran)),
                    "required_files": list(deliverables),
                    "missing_required_files": list(missing_required_files),
                    "verification_passed": verification_passed,
                    "mutation_tools_called": mutation_tool_stats["mutation_tools_called"],
                    "successful_mutations": mutation_tool_stats["successful_mutations"],
                    "failed_mutations": mutation_tool_stats["failed_mutations"],
                }
            ],
            run_id=store.run_id,
            run_dir=str(store.run_dir),
            run_status=run_status,
            next_action="",
        )

    def resume_run(
        self,
        *,
        run_id: str,
        index_dir: str | Path | None = None,
        index_dirs: Sequence[str | Path] | None = None,
        k: int = 8,
        max_steps: int = 6,
        timeout_seconds: int = 60,
        tool_policy: dict[str, Any] | None = None,
        pass_cap: int = 4,
        on_event: Callable[[Any], None] | None = None,
        flow_id: str | None = None,
        max_no_progress_passes: int = 2,
    ) -> AutoExecuteResult:
        store = RunStateStore(repo_root=self.repo_root, run_id=run_id)
        state = store.read_json("state.json", {})
        if not state:
            raise FileNotFoundError(f"Run state not found for run_id={run_id}")
        request = str(state.get("goal", state.get("original_user_task", "")) or "").strip() or (
            f"Resume mana-agent run {run_id}"
        )
        return self.run(
            request=request,
            flow_context=f"Resuming run_id={run_id}",
            index_dir=index_dir,
            index_dirs=index_dirs,
            k=k,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            tool_policy=tool_policy or {},
            pass_cap=pass_cap,
            on_event=on_event,
            flow_id=flow_id or str(state.get("flow_id", "") or ""),
            run_id=run_id,
            max_no_progress_passes=max_no_progress_passes,
        )


__all__ = [
    "AgentWorkQueue",
    "EventBus",
    "JobSniffer",
    "QueueManager",
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
