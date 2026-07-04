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
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Protocol, Sequence

from pydantic import BaseModel, Field

from mana_agent.agent.orchestrator import AgentOrchestrator
from mana_agent.llm.agent_session import AgentSession
from mana_agent.llm.goal_profiles import active_goal_profile
from mana_agent.llm.tool_worker_process import ToolWorkerClient
from mana_agent.llm.tools_executor import ToolsExecutionConfig, ToolsExecutor
from mana_agent.llm.tools_manager import (
    AgentFlowError,
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
    resolve_target_state,
)
from mana_agent.llm.mutation_plan import (
    REGISTERED_MUTATION_TOOLS,
    MutationCommand,
    MutationPlan,
    build_mutation_plan,
    changed_files_match_plan,
    compile_mutation_command,
    is_architecture_docs_update,
    mutation_trace_has_plan,
    validate_mutation_command,
    validate_mutation_plan,
)
from mana_agent.services.coding_memory_service import CodingMemoryService
from mana_agent.services.coding_todo_service import TodoService
from mana_agent.tools.apply_patch import safe_apply_patch
from mana_agent.tools.repository import apply_patch_batch
from mana_agent.tools.write_file import safe_create_file, safe_delete_file, safe_write_file

logger = logging.getLogger(__name__)

_MUTATION_TOOLS = {"edit_file", "multi_edit_file", "apply_patch", "apply_patch_batch", "write_file", "create_file", "delete_file"}
_DOC_EDIT_INTENT_RE = re.compile(r"\b(update|edit|write|fix|change|modify|replace|refactor)\b", re.IGNORECASE)
_MUTATION_LOCKS_GUARD = threading.Lock()
_MUTATION_LOCKS: dict[str, threading.Lock] = {}

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


def _mutation_lock(flow_id: str | None, run_id: str | None) -> threading.Lock:
    key = str(flow_id or run_id or "default").strip() or "default"
    with _MUTATION_LOCKS_GUARD:
        lock = _MUTATION_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _MUTATION_LOCKS[key] = lock
        return lock


def _command_changed_files(result: dict[str, Any], args: dict[str, Any]) -> list[str]:
    changed = _extract_changed_files_from_value(result)
    if not changed:
        changed.extend(str(path).strip() for path in result.get("touched_files") or [] if str(path).strip())
    if not changed and args.get("path"):
        changed.append(str(args.get("path")))
    return sorted(dict.fromkeys(path.replace("\\", "/").lstrip("./") for path in changed if path))


def _latest_mutation_command_error(trace: Sequence[dict[str, Any]]) -> str:
    for row in reversed(list(trace)):
        if not isinstance(row, dict):
            continue
        error = str(row.get("error") or "").strip()
        if error in {"mutation_command_missing", "mutation_command_incomplete"}:
            return error
    return ""


def execute_registered_mutation_command(
    *,
    repo_root: Path,
    command: MutationCommand,
) -> WorkResult:
    errors = validate_mutation_command(command)
    if errors:
        trace = [
            {
                "tool_name": command.tool_name,
                "status": "blocked",
                "error": "mutation_command_incomplete",
                "details": errors,
                "mutation_plan_id": command.plan_id,
                "target_files": list(command.target_files),
                "changed_files": [],
                "created_by": "mutation_command_executor",
            }
        ]
        return WorkResult(
            ok=False,
            summary="mutation command incomplete",
            error="mutation_command_incomplete: " + "; ".join(errors),
            trace=trace,
        )
    tool = command.tool_name
    args = dict(command.tool_args or {})
    if tool == "write_file":
        result = safe_write_file(
            repo_root=repo_root,
            path=str(args["path"]),
            content=str(args["content"]),
            force=bool(args.get("force", True)),
        )
    elif tool == "create_file":
        result = safe_create_file(repo_root=repo_root, path=str(args["path"]), content=str(args["content"]))
    elif tool == "delete_file":
        result = safe_delete_file(repo_root=repo_root, path=str(args["path"]))
    elif tool == "apply_patch":
        result = safe_apply_patch(repo_root=repo_root, patch=str(args["patch"]))
    elif tool == "apply_patch_batch":
        result = apply_patch_batch(repo_root, patches=list(args["patches"]))
    else:
        trace = [
            {
                "tool_name": tool,
                "status": "blocked",
                "error": "unsupported_registered_mutation_tool",
                "mutation_plan_id": command.plan_id,
                "target_files": list(command.target_files),
                "changed_files": [],
                "created_by": "mutation_command_executor",
            }
        ]
        return WorkResult(
            ok=False,
            summary="unsupported registered mutation tool",
            error=f"unsupported_registered_mutation_tool:{tool}",
            trace=trace,
        )
    ok = bool(result.get("ok"))
    changed = _command_changed_files(result, args) if ok else []
    trace = [
        {
            "tool_name": tool,
            "status": "ok" if ok else "error",
            "tool_args": args,
            "changed_files": changed,
            "files_changed": changed,
            "target_files": list(command.target_files),
            "mutation_plan_id": command.plan_id,
            "created_by": "mutation_command_executor",
            "error": "" if ok else str(result.get("error") or result.get("stderr") or "mutation command failed"),
        }
    ]
    return WorkResult(
        ok=ok,
        summary="mutation command executed" if ok else "mutation command failed",
        error="" if ok else str(trace[0]["error"]),
        files_changed=changed,
        answer="",
        trace=trace,
    )


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
    elif tool in {"repo_search", "repo_batch_search", "semantic_search", "list_files"}:
        query = _normalize_text(args.get("query") or args.get("q") or args.get("pattern") or question)
        payload = f"{tool}:{query}"
    elif tool in {"edit_file", "multi_edit_file", "apply_patch", "apply_patch_batch", "write_file", "create_file", "delete_file"}:
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

    def skip_where(self, predicate: Callable[[WorkItem], bool], *, reason: str) -> int:
        """Mark pending/ready jobs as skipped when the evaluation gate closes discovery."""
        skipped = 0
        with self._lock:
            for item in self._items.values():
                if item.status not in {"pending", "ready"}:
                    continue
                if not predicate(item):
                    continue
                item.status = "skipped"
                item.error = reason
                item.updated_at = _utc_now()
                skipped += 1
                self._emit("job_skipped", item, status="skipped")
        return skipped

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            for item in self._items.values():
                counts[item.status] = counts.get(item.status, 0) + 1
            active_remaining = counts.get("pending", 0) + counts.get("ready", 0) + counts.get("running", 0)
            return {
                "total": len(self._items),
                "counts": counts,
                "done": counts.get("done", 0),
                "failed": counts.get("failed", 0),
                "blocked": counts.get("blocked", 0),
                "skipped": counts.get("skipped", 0),
                "active_remaining": active_remaining,
                "remaining": active_remaining + counts.get("failed", 0) + counts.get("blocked", 0),
                "complete": bool(active_remaining == 0 and counts.get("failed", 0) == 0 and counts.get("blocked", 0) == 0),
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
        orchestrator: AgentOrchestrator | None = None,
        max_steps: int = 60,
    ) -> None:
        self._queue = queue
        self._execute = execute
        self._sniffer = sniffer
        self._board = board or TaskBoard(queue=queue)
        self._orchestrator = orchestrator
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
            pre_gate = self._before_item(item)
            if pre_gate == "skip":
                self._queue.complete(
                    item.id,
                    status="skipped",
                    result=WorkResult(ok=True, summary="skipped by evaluation gate", error="evaluation_gate_skipped"),
                )
                continue
            result = self._safe_execute(item)
            status = self._classify(item, result)
            if status == "retry":
                self._queue.requeue(item.id)
            else:
                self._queue.complete(item.id, status=status, result=result)
                self._after_item(item, result)
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

    def _before_item(self, item: WorkItem) -> str:
        if self._orchestrator is None:
            return "execute"
        target = str((item.tool_args or {}).get("path") or (item.tool_args or {}).get("file") or "")
        gate = self._orchestrator.before_tool(tool_name=item.tool_name or item.kind, target=target)
        if gate.decision == "skip_tool":
            return "skip"
        return "execute"

    def _after_item(self, item: WorkItem, result: WorkResult) -> None:
        if self._orchestrator is None:
            return
        gate = self._orchestrator.after_tool(
            tool_name=item.tool_name or item.kind,
            ok=bool(result.ok),
            files_read=list(result.files_read),
            changed_files=list(result.files_changed),
            error=str(result.error or ""),
        )
        if gate.decision in {"start_mutation", "stop_discovery"}:
            targets = set(self._orchestrator.decision.target_files)
            self._queue.skip_where(
                lambda queued: queued.kind in {"discover", "search", "read"}
                and str((queued.tool_args or {}).get("path") or "").replace("\\", "/").lstrip("./") not in targets,
                reason="evaluation_gate_evidence_sufficient",
            )

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

    def _read_current_files(self, targets: Sequence[str]) -> dict[str, str]:
        current: dict[str, str] = {}
        for raw in targets:
            rel = str(raw or "").replace("\\", "/").lstrip("./")
            if not rel:
                continue
            try:
                path = (self.repo_root / rel).resolve()
                path.relative_to(self.repo_root)
                if path.is_file():
                    current[rel] = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
        return current

    @staticmethod
    def _json_object_from_answer(answer: str) -> dict[str, Any] | None:
        text = str(answer or "").strip()
        if not text:
            return None
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                value = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return value if isinstance(value, dict) else None

    def _command_from_payload(
        self,
        *,
        plan: MutationPlan,
        payload: dict[str, Any],
        current_files: dict[str, str],
    ) -> MutationCommand | None:
        tool_name = str(payload.get("tool_name") or "").strip()
        tool_args = payload.get("tool_args") if isinstance(payload.get("tool_args"), dict) else {}
        if tool_name:
            try:
                return MutationCommand(
                    plan_id=str(payload.get("plan_id") or payload.get("mutation_plan_id") or plan.plan_id),
                    tool_name=tool_name,  # type: ignore[arg-type]
                    tool_args=dict(tool_args),
                    target_files=list(payload.get("target_files") or plan.target_files),
                    reason=str(payload.get("reason") or "worker synthesized mutation command"),
                )
            except Exception:
                return None
        content = tool_args.get("content") if isinstance(tool_args, dict) else payload.get("content")
        patch = tool_args.get("patch") if isinstance(tool_args, dict) else payload.get("patch")
        return compile_mutation_command(
            repo_root=self.repo_root,
            plan=plan,
            current_files=current_files,
            synthesized_content=str(content) if content is not None else None,
            patch=str(patch) if patch is not None else None,
        )

    def _synthesize_mutation_command(
        self,
        *,
        base_execute: Callable[[WorkItem], WorkResult],
        plan: MutationPlan,
        target_file: str,
    ) -> MutationCommand | None:
        current_files = self._read_current_files(plan.target_files)
        schema = (
            'Return only JSON: {"tool_name":"write_file|create_file|apply_patch",'
            '"tool_args":{"path":"<repo-relative path>","content":"<complete file content>"}'
            '} or {"tool_name":"apply_patch","tool_args":{"patch":"*** Begin Patch\\n..."}}.'
        )
        prompts = [
            (
                f"Generate the executable MutationCommand for approved MutationPlan {plan.plan_id}. "
                f"Target file: {target_file}. User goal: {plan.user_goal}. Evidence summary: {plan.evidence_summary}. "
                f"Intended changes: {'; '.join(plan.intended_changes)}. Patch strategy: {plan.patch_strategy}. {schema}"
            ),
            (
                f"Retry MutationCommand synthesis for MutationPlan {plan.plan_id}. The previous response was not valid. "
                f"{schema} Do not include prose."
            ),
        ]
        last_command: MutationCommand | None = None
        for prompt in prompts:
            result = base_execute(
                WorkItem(
                    kind="summarize",
                    tool_name="",
                    tool_args={},
                    question=prompt,
                    gate="synthesize_mutation_command",
                    priority=1,
                    created_by="mutation_command_synthesizer",
                    max_attempts=1,
                )
            )
            payload = self._json_object_from_answer(result.answer)
            if not payload:
                continue
            command = self._command_from_payload(plan=plan, payload=payload, current_files=current_files)
            if command is None:
                continue
            last_command = command
            if not validate_mutation_command(command):
                return command
        return last_command

    def _git_diff_names(self) -> set[str]:
        try:
            proc = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=self.repo_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return set()
        if proc.returncode != 0:
            return set()
        return {line.strip() for line in proc.stdout.splitlines() if line.strip()}

    def _git_status_names(self) -> set[str]:
        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return set()
        if proc.returncode != 0:
            return set()
        paths: set[str] = set()
        for line in proc.stdout.splitlines():
            if not line:
                continue
            payload = line[3:] if len(line) > 3 else line
            if " -> " in payload:
                payload = payload.split(" -> ", 1)[1]
            payload = payload.strip()
            if payload:
                paths.add(payload)
        return paths

    def _docs_markdown_fallback_content(self, *, request: str, target_file: str, current: str) -> str:
        stamp = "## Update Notes"
        request_text = str(request or "").strip()
        requested_line = f"Requested change: {request_text}"
        text = str(current or "")
        if requested_line in text:
            requested_line = f"{requested_line} (mutation execution fallback)"
        if stamp in text:
            return f"{text.rstrip()}\n\n{requested_line}\n"
        note = f"{stamp}\n\n{requested_line}\n"
        suffix = "\n\n" if text and not text.endswith("\n\n") else ""
        _ = target_file
        return f"{text}{suffix}{note}"

    def _try_docs_markdown_mutation_fallback(
        self,
        *,
        request: str,
        target_file: str,
        trace: list[dict[str, Any]],
        changed_files: list[str],
        warnings: list[str],
    ) -> bool:
        rel = str(target_file or "").strip().replace("\\", "/").lstrip("./")
        if not (
            rel.startswith("docs/")
            and rel.endswith(".md")
            and _DOC_EDIT_INTENT_RE.search(str(request or ""))
        ):
            return False
        target = self.repo_root / rel
        if not target.is_file():
            return False
        try:
            current = target.read_text(encoding="utf-8")
        except OSError as exc:
            warnings.append(f"docs_markdown_fallback_read_failed:{rel}:{exc}")
            return False
        content = self._docs_markdown_fallback_content(request=request, target_file=rel, current=current)
        if content == current:
            return False
        result = safe_write_file(repo_root=self.repo_root, path=rel, content=content, force=True)
        ok = bool(result.get("ok"))
        row = {
            "tool_name": "write_file",
            "status": "ok" if ok else "error",
            "path": rel,
            "changed_files": [rel] if ok else [],
            "files_changed": [rel] if ok else [],
            "created_by": "docs_markdown_mutation_fallback",
        }
        if not ok:
            row["error"] = str(result.get("error") or "docs markdown fallback failed")
            warnings.append(f"docs_markdown_fallback_failed:{rel}")
        else:
            changed_files.append(rel)
            warnings.append(f"docs_markdown_fallback_used:{rel}")
        trace.append(row)
        if ok:
            try:
                proc = subprocess.run(
                    ["git", "diff", "--", rel],
                    cwd=self.repo_root,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                verify_status = "ok" if proc.returncode == 0 else "error"
                verify_preview = (proc.stdout or proc.stderr or "").strip()
            except Exception as exc:
                verify_status = "error"
                verify_preview = str(exc)
            trace.append(
                {
                    "tool_name": "run_command",
                    "status": verify_status,
                    "command": f"git diff -- {rel}",
                    "output_preview": verify_preview[:4000],
                    "created_by": "docs_markdown_mutation_fallback",
                }
            )
        return ok

    def attach_decision_provider(self, provider: Any) -> None:
        # The provider (typically the CodingAgent) supplies the LLM planner used
        # by ``preview_plan`` for accurate checklists. The execution loop itself
        # stays deterministic; only pre-execution planning consults the provider.
        self._decision_provider = provider

    def update_model(self, new_model: str) -> None:
        logger.info("Ignoring model update; QueueManager is deterministic-only.")

    def _target_state(self, request: str, target_files: Sequence[str] = ()) -> dict[str, list[str]]:
        return resolve_target_state(request, self.repo_root, target_files=target_files)

    def _normalize_planner_payload_targets(self, payload: dict[str, Any], request: str, target_files: Sequence[str]) -> None:
        payload_targets = payload.get("target_files")
        candidates = payload_targets if isinstance(payload_targets, list) else list(target_files)
        state = self._target_state(request, [str(item) for item in candidates if str(item).strip()])
        resolved = state["resolved_target_files"]
        if resolved:
            payload["target_files"] = list(resolved)
        payload["raw_target_files"] = list(state["raw_target_files"])
        payload["resolved_target_files"] = list(resolved)
        payload["unresolved_target_files"] = list(state["unresolved_target_files"])
        prechecklist = payload.get("prechecklist")
        if isinstance(prechecklist, dict):
            if resolved:
                prechecklist["target_files"] = list(resolved)
            prechecklist["raw_target_files"] = list(state["raw_target_files"])
            prechecklist["resolved_target_files"] = list(resolved)
            prechecklist["unresolved_target_files"] = list(state["unresolved_target_files"])

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

        self._normalize_planner_payload_targets(payload, request, target_files)
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

        target_state = self._target_state(request, target_files)
        deliverables = list(target_state["resolved_target_files"]) or _resolve_required_deliverables(
            request, self.repo_root, tuple(target_files)
        )
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
                    "requires_tools": ["edit_file", "multi_edit_file", "apply_patch", "apply_patch_batch", "write_file", "create_file", "delete_file"],
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
            "raw_target_files": list(target_state["raw_target_files"]),
            "resolved_target_files": list(deliverables),
            "unresolved_target_files": list(target_state["unresolved_target_files"]),
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
            "raw_target_files": list(target_state["raw_target_files"]),
            "resolved_target_files": list(deliverables),
            "unresolved_target_files": list(target_state["unresolved_target_files"]),
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
        planner_state: dict[str, Any] | None = None,
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
                checklist=dict(planner_state or {}),
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
            make_batch_executor,
            make_worker_executor,
        )

        _ = (max_no_progress_passes, flow_context)
        store = RunStateStore(repo_root=self.repo_root, run_id=run_id)
        baseline_diff = self._git_diff_names()
        baseline_status = self._git_status_names()
        pre_existing_changed_files = sorted(baseline_diff)
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
        target_state = self._target_state(request, target_files)
        resolved_target_files = list(target_state["resolved_target_files"])
        required_files = resolved_target_files or _resolve_required_deliverables(request, self.repo_root, target_files)
        resolved_target_path = (
            required_files[0]
            if required_files
            else _resolve_mutation_target_path(request, self.repo_root, target_files)
        )
        queue = AgentWorkQueue()
        board = TaskBoard(queue=queue)
        orchestrator = AgentOrchestrator.start(
            request,
            repo_root=self.repo_root,
            target_files=required_files or target_files,
            requires_edit=mutation_required,
        )
        profile = active_goal_profile(request)
        if profile is not None:
            def _relevant(path: str) -> bool:
                return profile.is_relevant(path, self.repo_root)
        else:
            def _relevant(path: str) -> bool:
                return True

        direct_read_targets = list(orchestrator.decision.target_files)
        explicit_direct_read = (
            orchestrator.decision.needs_file_read
            and orchestrator.decision.scope in {"single_file", "single_file_section"}
            and bool(direct_read_targets)
            and all((self.repo_root / path).is_file() for path in direct_read_targets)
            and not is_architecture_docs_update(request, direct_read_targets)
        )
        if explicit_direct_read:
            for target in orchestrator.decision.target_files:
                queue.submit(
                    WorkItem(
                        kind="read",
                        tool_name="read_file",
                        tool_args={"path": target},
                        question=f"Read explicit target file {target}",
                        gate="read_explicit_target",
                        priority=10,
                        created_by="agent_orchestrator",
                    )
                )
        elif default_skill_registry_request:
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
        approved_mutation_plan: MutationPlan | None = None
        mutation_state: dict[str, Any] = {
            "mutation_attempted": False,
            "mutation_succeeded": False,
            "changed_files": [],
            "no_op_reason": "",
            "verify_requires_mutation": False,
        }
        execution_backend = str(getattr(self.execution_config, "backend", "") or "local")
        session = AgentSession.from_queue_run(
            repo_root=self.repo_root,
            run_id=store.run_id,
            flow_id=flow_id,
            index_dir=index_dir,
            index_dirs=index_dirs,
            tool_policy=resolved_tool_policy,
            execution_backend=execution_backend,
        )
        if self.executor is not None:
            base_execute = make_batch_executor(
                executor=self.executor,
                session=session,
                on_event=on_event,
                default_timeout=int(timeout_seconds),
                default_k=int(k),
                default_max_steps=int(max_steps),
            )
        else:
            base_execute = make_worker_executor(
                worker_client=self.worker_client,
                repo_root=self.repo_root,
                on_event=on_event,
                default_timeout=int(timeout_seconds),
                default_k=int(k),
                default_max_steps=int(max_steps),
                tool_policy=resolved_tool_policy,
                index_dir=session.index_dir,
                flow_id=session.flow_id,
                run_id=session.run_id,
            )

        # The deliverables the request demands. The run is held to ALL of them,
        # not just the first, so "create A and B in docs" cannot finish with only
        # one file written. Falls back to the single primary target when the
        # request named no explicit files (e.g. generic "fix the bug").
        deliverables = list(required_files) or ([resolved_target_path] if resolved_target_path else [])
        initially_missing_deliverables = _missing_required_files(self.repo_root, deliverables, changed=set())

        def execute(item: "WorkItem"):  # noqa: F821 - imported above
            nonlocal mutation_state, approved_mutation_plan
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
            if mutation_required and item.kind == "edit":
                item_tool_args = dict(item.tool_args or {})
                item_targets = [
                    str(path).strip().replace("\\", "/").lstrip("./")
                    for path in (item_tool_args.get("target_files") or [])
                    if str(path).strip()
                ]
                plan_targets = item_targets or deliverables or ([resolved_target_path] if resolved_target_path else [])
                read_files = sorted(
                    dict.fromkeys(
                        path
                        for queued in queue.items()
                        if queued.status == "done"
                        for path in (queued.files_read or [])
                        if path
                    )
                )
                plan = build_mutation_plan(
                    repo_root=self.repo_root,
                    user_goal=request,
                    target_files=plan_targets,
                    evidence_files_read=read_files,
                )
                plan_errors = validate_mutation_plan(plan, repo_root=self.repo_root)
                if plan_errors:
                    blocked_trace = [
                        {
                            "tool_name": "mutation_command",
                            "status": "blocked",
                            "error": "mutation_plan_validation_failed",
                            "details": plan_errors,
                            "mutation_plan_id": plan.plan_id,
                        }
                    ]
                    trace.extend(blocked_trace)
                    return WorkResult(
                        ok=False,
                        summary="mutation plan validation failed",
                        error="mutation_plan_validation_failed: " + "; ".join(plan_errors),
                        trace=blocked_trace,
                    )
                approved_mutation_plan = plan
                target = str((plan_targets or [resolved_target_path])[0] or "")
                tool_args = dict(item.tool_args or {})
                command: MutationCommand | None = None
                if (item.tool_name or "").strip().lower() in REGISTERED_MUTATION_TOOLS:
                    try:
                        command = MutationCommand(
                            plan_id=plan.plan_id,
                            tool_name=(item.tool_name or "").strip().lower(),  # type: ignore[arg-type]
                            tool_args={**tool_args, "path": tool_args.get("path") or target},
                            target_files=list(plan.target_files),
                            reason="compiled from queued edit work item",
                        )
                    except Exception:
                        command = None
                    if command and validate_mutation_command(command):
                        command = None
                if command is None:
                    command = self._synthesize_mutation_command(
                        base_execute=base_execute,
                        plan=plan,
                        target_file=target,
                    )
                if command is None:
                    legacy_result = base_execute(
                        WorkItem(
                            kind="edit",
                            tool_name="",
                            tool_args={
                                "mutation_plan": plan.model_dump(),
                                "mutation_plan_id": plan.plan_id,
                            },
                            question=item.question,
                            gate="apply_approved_mutation_plan",
                            priority=item.priority,
                            created_by="approved_plan_legacy_mutation_pass",
                            max_attempts=1,
                        )
                    )
                    legacy_changed = sorted(
                        dict.fromkeys(
                            [
                                *legacy_result.files_changed,
                                *_extract_changed_files_from_value(legacy_result.trace),
                            ]
                        )
                    )
                    if mutation_trace_has_plan(legacy_result.trace, plan.plan_id) and legacy_changed:
                        legacy_result.files_changed = legacy_changed
                        if legacy_result.answer:
                            answers.append(legacy_result.answer)
                        sources.extend(legacy_result.sources)
                        trace.extend(legacy_result.trace)
                        changed_files.extend(legacy_changed)
                        changed_files[:] = sorted(dict.fromkeys(path for path in changed_files if path))
                        mutation_state = _mutation_state_from_trace(trace, changed_files)
                        return legacy_result
                    legacy_payload = self._json_object_from_answer(legacy_result.answer)
                    if legacy_payload:
                        current_files = self._read_current_files(plan.target_files)
                        legacy_command = self._command_from_payload(
                            plan=plan,
                            payload=legacy_payload,
                            current_files=current_files,
                        )
                        if legacy_command and not validate_mutation_command(legacy_command):
                            command_result = execute_registered_mutation_command(
                                repo_root=self.repo_root,
                                command=legacy_command,
                            )
                            if command_result.ok:
                                if command_result.answer:
                                    answers.append(command_result.answer)
                                sources.extend(command_result.sources)
                                trace.extend(command_result.trace)
                                changed_files.extend(command_result.files_changed)
                                changed_files.extend(_extract_changed_files_from_value(command_result.trace))
                                changed_files[:] = sorted(dict.fromkeys(path for path in changed_files if path))
                                mutation_state = _mutation_state_from_trace(trace, changed_files)
                                return command_result
                    blocked_trace = [
                        {
                            "tool_name": "mutation_command",
                            "status": "blocked",
                            "error": "mutation_command_missing",
                            "mutation_plan_id": plan.plan_id,
                            "target_files": list(plan.target_files),
                            "changed_files": [],
                            "created_by": "mutation_command_executor",
                        }
                    ]
                    trace.extend(blocked_trace)
                    return WorkResult(
                        ok=False,
                        summary="mutation command missing",
                        error="mutation_command_missing",
                        trace=blocked_trace,
                    )
                command_errors = validate_mutation_command(command)
                if command_errors:
                    blocked_trace = [
                        {
                            "tool_name": command.tool_name,
                            "status": "blocked",
                            "error": "mutation_command_incomplete",
                            "details": command_errors,
                            "mutation_plan_id": command.plan_id,
                            "target_files": list(command.target_files),
                            "changed_files": [],
                            "created_by": "mutation_command_executor",
                        }
                    ]
                    trace.extend(blocked_trace)
                    return WorkResult(
                        ok=False,
                        summary="mutation command incomplete",
                        error="mutation_command_incomplete: " + "; ".join(command_errors),
                        trace=blocked_trace,
                    )
                result = execute_registered_mutation_command(repo_root=self.repo_root, command=command)
                if result.answer:
                    answers.append(result.answer)
                sources.extend(result.sources)
                trace.extend(result.trace)
                changed_files.extend(result.files_changed)
                changed_files.extend(_extract_changed_files_from_value(result.trace))
                changed_files[:] = sorted(dict.fromkeys(path for path in changed_files if path))
                mutation_state = _mutation_state_from_trace(trace, changed_files)
                return result
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

        sniffer_target_files = [
            str(item).strip()
            for item in (target_files or required_files or ([resolved_target_path] if resolved_target_path else []))
            if str(item).strip()
        ]
        # Whether to finalize with an edit + verify is recognized upstream by the
        # coding agent's planner (the LLM checklist) and passed in as requires_edit.
        sniffer = CodingAgentSniffer(
            repo_root=self.repo_root,
            request=request,
            emit_edit=mutation_required,
            target_files=sniffer_target_files,
            relevant=_relevant,
            orchestrator=orchestrator,
        )
        runner = WorkQueueRunner(
            queue=queue,
            execute=execute,
            sniffer=sniffer,
            board=board,
            orchestrator=orchestrator,
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

        # --- Forced mutation retry: one file at a time, mutation tool required. ---
        # This is the AGENTIC content path and runs *before* the deterministic
        # template fallback so the worker authors real, file-specific content
        # (a genuine overview vs. a genuine installation guide), not boilerplate.
        # For any deliverable still missing after the main pass (or, when no files
        # were named, when nothing was mutated) re-run the worker under a strict
        # MUTATION_REQUIRED prompt that forbids natural-language-only answers.
        forced_missing_before: list[str] = []
        if mutation_required and not mutation_state.get("no_op_reason"):
            forced_missing_before = _missing_required_files(
                self.repo_root, deliverables, changed=set(mutation_state.get("changed_files") or [])
            )
            if deliverables and not mutation_state.get("mutation_succeeded"):
                forced_targets = list(deliverables)
            elif deliverables:
                forced_targets = list(forced_missing_before)
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
                        "edit_file",
                        "multi_edit_file",
                        "apply_patch",
                        "write_file",
                        "create_file",
                        "delete_file",
                    ],
                    "verify_requires_mutation": True,
                }
                forced_session = session.with_tool_policy(forced_policy)
                if self.executor is not None:
                    forced_execute = make_batch_executor(
                        executor=self.executor,
                        session=forced_session,
                        on_event=on_event,
                        default_timeout=int(timeout_seconds),
                        default_k=int(k),
                        default_max_steps=max(1, int(max_steps)),
                    )
                else:
                    forced_execute = make_worker_executor(
                        worker_client=self.worker_client,
                        repo_root=self.repo_root,
                        on_event=on_event,
                        default_timeout=int(timeout_seconds),
                        default_k=int(k),
                        default_max_steps=max(1, int(max_steps)),
                        tool_policy=forced_policy,
                        index_dir=forced_session.index_dir,
                        flow_id=forced_session.flow_id,
                        run_id=forced_session.run_id,
                    )
                forced_item = WorkItem(
                    kind="edit",
                    tool_name="",
                    tool_args={"target_files": [target_file]},
                    question=_forced_mutation_prompt(request, target_file),
                    gate="apply_changes",
                    priority=1,
                    created_by="forced_mutation_retry",
                )
                if self.executor is None:
                    trace_len = len(trace)
                    changed_len = len(changed_files)
                    forced_result = execute(forced_item)
                    if forced_result.answer and forced_result.answer not in answers:
                        answers.append(forced_result.answer)
                    sources.extend(forced_result.sources)
                    if len(trace) == trace_len:
                        trace.extend(forced_result.trace)
                    if len(changed_files) == changed_len:
                        changed_files.extend(forced_result.files_changed)
                        changed_files.extend(_extract_changed_files_from_value(forced_result.trace))
                    _refresh_mutation_state()
                target_changed = target_file in set(mutation_state.get("changed_files") or [])
                if not target_changed or self.executor is not None:
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

            if (
                forced_retry_ran
                and not mutation_state.get("mutation_succeeded")
                and resolved_tool_policy.get("fallback_decision") is True
            ):
                for target_file in deliverables:
                    if self._try_docs_markdown_mutation_fallback(
                        request=request,
                        target_file=target_file,
                        trace=trace,
                        changed_files=changed_files,
                        warnings=warnings,
                    ):
                        _refresh_mutation_state()
                        break
                forced_retry_mutation_attempted = bool(mutation_state.get("mutation_attempted"))
                forced_retry_changed_files = bool(mutation_state.get("changed_files"))

            reconciliation_candidates = sorted(
                dict.fromkeys([*initially_missing_deliverables, *forced_missing_before])
            )
            if forced_retry_ran and reconciliation_candidates and not mutation_state.get("mutation_succeeded"):
                still_missing = _missing_required_files(
                    self.repo_root, deliverables, changed=set(mutation_state.get("changed_files") or [])
                )
                if not still_missing:
                    reconciled = sorted(dict.fromkeys(deliverables))
                    trace.append(
                        {
                            "tool_name": "write_file",
                            "status": "ok",
                            "changed_files": reconciled,
                            "files_changed": reconciled,
                            "target_files": reconciled,
                            "mutation_plan_id": approved_mutation_plan.plan_id if approved_mutation_plan else "",
                            "created_by": "forced_retry_filesystem_reconciliation",
                        }
                    )
                    changed_files.extend(reconciled)
                    _refresh_mutation_state()
                    forced_retry_mutation_attempted = bool(mutation_state.get("mutation_attempted"))
                    forced_retry_changed_files = bool(mutation_state.get("changed_files"))

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
            command_error = _latest_mutation_command_error(trace)
            if not mutation_state.get("mutation_attempted"):
                run_status = "blocked"
                terminal_reason = command_error or "mutation_required_but_no_mutation_tool_attempted"
            elif not mutation_state.get("changed_files"):
                run_status = "blocked"
                terminal_reason = command_error or "mutation_required_but_no_changed_files"
            elif missing_required_files:
                run_status = "blocked"
                terminal_reason = "mutation_required_but_missing_files"
        # Relocated/cleaned-up strays no longer exist on disk, so drop them from
        # the reported changed files regardless of what the trace remembers.
        _removed = set(removed_strays)
        current_diff = self._git_diff_names()
        current_status = self._git_status_names()
        run_changed_files = sorted((current_diff | current_status).difference(baseline_diff | baseline_status))
        trace_changed_files = [
            p for p in (mutation_state.get("changed_files") or changed_files) if p not in _removed
        ]
        changed_files = sorted(dict.fromkeys([*trace_changed_files, *run_changed_files]))
        # Final answer is rebuilt from authoritative execution state (trace,
        # changed_files, mutation_state, verification), never from the last
        # natural-language worker answer, so an intermediate "I could not edit"
        # cannot contradict a trace that proves a mutation landed.
        verification = _verification_summary_from_trace(trace)
        verification_decision = orchestrator.verification_decision(
            changed_files=changed_files,
            core_agent_change=any(
                path.startswith("src/mana_agent/agent/") or path.startswith("src/mana_agent/llm/")
                for path in changed_files
            ),
        )
        failed_calls = _failed_tool_calls_from_trace(trace)
        for failure in failed_calls:
            warning = f"tool_call_failed:{failure['tool']}"
            if warning not in warnings:
                warnings.append(warning)
        mutation_tool_stats = _mutation_tool_stats(trace)
        mutation_plan_id = approved_mutation_plan.plan_id if approved_mutation_plan is not None else ""
        mutation_plan_executed = (not mutation_required) or mutation_trace_has_plan(trace, mutation_plan_id)
        mutation_plan_targets_changed = (not mutation_required) or changed_files_match_plan(changed_files, approved_mutation_plan)
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
            tool_failures=failed_calls,
            mutation_tools_used=mutation_tool_stats["mutation_tools_called"],
        )
        if mutation_required and approved_mutation_plan is not None:
            final_answer = f"{final_answer}\nMutation plan: {approved_mutation_plan.plan_id}"
        verification_passed = bool(
            (not mutation_required or bool(mutation_state.get("mutation_succeeded")))
            and mutation_plan_executed
            and mutation_plan_targets_changed
            and (not missing_required_files)
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
            planner_state={
                "raw_target_files": list(target_state["raw_target_files"]),
                "resolved_target_files": list(deliverables),
                "required_files": list(deliverables),
                "missing_required_files": list(missing_required_files),
                "unresolved_target_files": list(target_state["unresolved_target_files"]),
            },
        )
        orchestrator.finalize_trace()
        trace.extend(orchestrator.trace)
        return AutoExecuteResult(
            answer=final_answer,
            sources=sources,
            trace=trace,
            warnings=warnings,
            changed_files=changed_files,
            passes=report.steps,
            terminal_reason=terminal_reason,
            toolsmanager_requests_count=report.steps,
            execution_backend=(
                f"work_queue:{session.execution_backend}"
                if self.executor is not None
                else "work_queue"
            ),
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
                    "verify_requires_mutation": bool(mutation_required),
                    "forced_mutation_retry_ran": forced_retry_ran,
                    "forced_retry_mutation_attempted": forced_retry_mutation_attempted,
                    "forced_retry_changed_files": forced_retry_changed_files,
                    "verification_ran": bool(verification.get("ran")),
                    "verification_failed": bool(verification.get("failed")),
                    "verification_failing_checks": list(verification.get("failing", [])),
                    "mutation_tool_attempted": bool(mutation_state.get("mutation_attempted")),
                    "mutation_tool_successful": bool(mutation_state.get("mutation_succeeded")),
                    "mutation_plan_id": mutation_plan_id,
                    "mutation_plan_approved": bool(approved_mutation_plan and approved_mutation_plan.allowed_to_mutate),
                    "mutation_plan_executed": bool(mutation_plan_executed),
                    "mutation_plan_targets_changed": bool(mutation_plan_targets_changed),
                    "mutation_fallback_count": int(bool(forced_retry_ran)),
                    "raw_target_files": list(target_state["raw_target_files"]),
                    "resolved_target_files": list(deliverables),
                    "unresolved_target_files": list(target_state["unresolved_target_files"]),
                    "required_files": list(deliverables),
                    "missing_required_files": list(missing_required_files),
                    "verification_passed": verification_passed,
                    "verification_profile": verification_decision.verification_profile,
                    "verification_commands": list(verification_decision.commands),
                    "skip_full_pytest_reason": verification_decision.skip_full_pytest_reason,
                    "task_decision": {
                        "task_type": orchestrator.decision.task_type,
                        "target_files": list(orchestrator.decision.target_files),
                        "target_sections": list(orchestrator.decision.target_sections),
                        "scope": orchestrator.decision.scope,
                        "confidence": orchestrator.decision.confidence,
                    },
                    "mutation_tools_called": mutation_tool_stats["mutation_tools_called"],
                    "mutation_tools_attempted": mutation_tool_stats["mutation_tools_attempted"],
                    "mutation_tools_successful": mutation_tool_stats["mutation_tools_successful"],
                    "mutation_tools_failed": mutation_tool_stats["mutation_tools_failed"],
                    "read_tools_called": mutation_tool_stats["read_tools_called"],
                    "search_tools_called": mutation_tool_stats["search_tools_called"],
                    "successful_mutations": mutation_tool_stats["successful_mutations"],
                    "failed_mutations": mutation_tool_stats["failed_mutations"],
                }
            ],
            run_id=store.run_id,
            run_dir=str(store.run_dir),
            run_status=run_status,
            next_action="",
            pre_existing_changed_files=pre_existing_changed_files,
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
