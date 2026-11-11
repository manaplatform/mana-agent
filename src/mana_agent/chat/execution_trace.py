"""Normalized runtime execution trace model for turn-level execution stats.

Provides a unified step-by-step trace representation shared by TUI, Dashboard,
and testing suites. Each chat turn tracks its chronological progression
(routing, context preparation, coding, tools, model calls, completion, failure)
with idempotent step updates and clean running-state transitions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        val_str = value.strip()
        if (val_str.startswith("{") and val_str.endswith("}")) or (val_str.startswith("[") and val_str.endswith("]")):
            try:
                parsed = json.loads(val_str)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
    return {}


def event_classification_metadata(event: dict[str, Any]) -> dict[str, Any]:
    """Collect event context used to classify execution steps and activity boxes.

    Runtime events may place backend/tool context at the top level while other
    producers nest it in metadata, details, or payload. Keep those contracts
    equivalent so a coding approval stays in Coding Activity instead of being
    misrouted to the general Activity stream.
    """
    metadata = dict(
        _safe_dict(event.get("metadata") or event.get("details") or event.get("payload"))
    )
    nested_sources = [
        _safe_dict(event.get("metadata")),
        _safe_dict(event.get("details")),
        _safe_dict(event.get("payload")),
    ]
    for field_name in ("backend", "tool_name", "name", "task_id", "root_task_id", "parent_task_id", "agent_id", "subagent_id", "attempt_id", "attempt_generation", "operation_id", "execution_mode"):
        value = event.get(field_name)
        if value is None or not str(value).strip():
            value = next(
                (source[field_name] for source in nested_sources if source.get(field_name)),
                None,
            )
        if value is not None and str(value).strip():
            metadata[field_name] = value
    return metadata


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

CODING_TOOLS: frozenset[str] = frozenset(
    {
        # File editing & mutations
        "edit_file",
        "multi_edit_file",
        "apply_patch",
        "apply_patch_batch",
        "write_file",
        "create_file",
        "delete_file",
        "patch",
        # Repository code reading & AST
        "read_file",
        "repo_batch_read",
        "list_files",
        "ls",
        "find_symbols",
        "call_graph",
        "chunk_file",
        "list_tools",
        "tool_contracts",
        "read_skill",
        # Git operations
        "git",
        "git_status",
        "git_diff",
        "git_log",
        "git_show",
        "git_branch",
        "git_switch",
        "git_checkout",
        "git_create_branch",
        "git_add",
        "git_restore",
        "git_stash",
        "git_commit",
        "git_push",
        "git_pull",
        "git_fetch",
        "git_remote",
        "git_tag",
        "git_merge",
        "git_rebase",
        "git_revert",
        "git_reset",
        "git_clean",
        "git_config",
        "git_generic",
        "git_help",
        # Shell / execution / verification
        "shell",
        "run_command",
        "run_script_once",
        "verify_project",
        "exec_command",
        "bash",
        "terminal",
        "local_shell",
    }
)


def is_coding_tool(tool_name: str | None) -> bool:
    """Return True if tool_name belongs to coding/repository/mutation/execution tools."""
    name = str(tool_name or "").strip().lower()
    if not name:
        return False
    # Search tools take precedence and are not coding tools
    if "search" in name or name in SEARCH_TOOLS:
        return False
    # Explicit exclusions for connectors, workflows, and approvals
    if name.startswith(("email_", "server_", "api_", "canvas_", "browser_", "media_", "generate_")):
        return False
    if name in {"approval_request", "request_user_approval", "wait_for_approval"}:
        return False
    if name in CODING_TOOLS:
        return True
    if name.startswith(("git.", "git_", "patch.", "patch_")):
        return True
    if name.startswith(("edit_", "write_")):
        return True
    return False



def classify_runtime_phase(event_type: str, metadata: dict[str, Any] | None = None) -> tuple[str, str]:
    """Map a runtime event type to (phase, title).

    Generic fallback derives phase and title cleanly for future phases.
    """
    et = str(event_type or "").strip().lower()
    meta = dict(metadata or {})
    tool_name = str(meta.get("tool_name") or meta.get("name") or "").strip()

    # Autonomous Loop events
    if et.startswith("loop."):
        if et == "loop.turn_received":
            return "loop", "Turn received"
        if et == "loop.initialized":
            return "loop", "Autonomous Loop initialized"
        if et == "loop.iteration.started":
            phase_label = str(meta.get("phase") or "Processing")
            return "loop", f"Iteration: {phase_label}"
        if et == "loop.strategy.selected":
            strat = str(meta.get("strategy") or "selected")
            return "routing", f"Strategy: {strat}"
        if et == "loop.taskboard.updated":
            return "taskboard", "TaskBoard updated"
        if et in {"loop.tool.running", "loop.tool.completed"}:
            t_name = str(meta.get("tool_name") or tool_name or "Tool")
            return "tool", f"Tool: {t_name}"
        if et == "loop.operations.completed":
            op_type = str(meta.get("type") or "operations")
            return "loop", f"{op_type.capitalize()} operations completed"
        if et == "loop.verification.completed":
            return "verification", "Verification"
        if et == "loop.completed":
            return "completion", "Execution completed"
        return "loop", "Autonomous Loop"

    # Lifecycle / Engine selection markers (handled as metadata, not separate user task steps)
    if et in {
        "backend.selected",
        "turn.starting",
        "turn.started",
        "turn.finalizing",
    }:
        return "lifecycle", ""

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

    # Coding research shares the active coding box, even when its underlying
    # tool is web_search. General search remains in Main Activity.
    if et.startswith("investigation.coding_research."):
        return "coding", "Coding research"

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
            meta.get("backend") in {"codex", "internal", "coding"}
            and not et.startswith(("turn.", "error", "routing", "search", "model"))
            and (not et.startswith("tool") or is_coding_tool(tool_name))
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

    # Approval
    if (
        et in {
            "action.approval.required",
            "server.waiting_approval",
            "api.waiting_approval",
            "computer.waiting_permission",
            "remote_execution.waiting_permission",
        }
        or tool_name in {"approval_request", "request_user_approval", "wait_for_approval"}
    ):
        return "approval", "Approval required"

    if et in {
        "action.approval.granted",
        "action.approval.denied",
        "server.approval_decided",
        "api.approval_decided",
        "computer.permission_decided",
    }:
        decision = str(meta.get("decision") or "")
        is_denied = "denied" in et or decision == "deny"
        return "approval", "Approval denied" if is_denied else "Approval granted"

    # Generic Fallback: supports future runtime phases automatically
    parts = et.replace("_", ".").split(".")
    phase = parts[0] if parts else "activity"
    title = phase.replace("_", " ").title()
    return phase, title


def classify_activity_category(
    event_type: str,
    tool_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Classify an event into 'coding' or 'main' activity box category.

    - 'coding': file mutations (edit, write, patch), repo reads, terminal commands,
      test execution, git operations, and coding engine / Codex events.
    - 'main': all general agent activities including non-coding tools (search,
      email, api, mcp), reasoning summaries, delegation, retries, and errors.
    """
    meta = dict(metadata or {})
    et = str(event_type or "").strip().lower()
    tool = str(tool_name or meta.get("tool_name") or meta.get("name") or "").strip()
    backend = str(meta.get("backend") or "").strip().lower()

    if et.startswith("investigation.coding_research."):
        return "coding"

    if backend in {"codex", "internal", "coding"} and not et.startswith(("turn.", "routing", "model")):
        return "coding"

    if tool and is_coding_tool(tool):
        return "coding"

    if (
        et.startswith(("command.", "patch.", "file."))
        or et in {"coding_started", "coding.terminal", "coding.progress", "coding"}
    ):
        return "coding"

    return "main"


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
        self._attempt_generations: dict[str, int] = {}
        # Persistent activity box event streams
        self.main_events: list[dict[str, Any]] = []
        self.coding_events: list[dict[str, Any]] = []
        self.changed_files: list[str] = []
        self._seen_main_ids: set[str] = set()
        self._seen_coding_ids: set[str] = set()

    @property
    def active_step(self) -> ExecutionStep | None:
        if self.active_step_id:
            return self._step_by_id.get(self.active_step_id)
        return None

    def get_step(self, step_id: str) -> ExecutionStep | None:
        return self._step_by_id.get(step_id)

    @property
    def active_steps(self) -> list[ExecutionStep]:
        return [step for step in self.steps if step.status in {"running", "waiting_approval", "waiting", "queued"}]

    def apply_event(self, event: dict[str, Any]) -> ExecutionStep | None:
        """Reduce observed operations without inventing completion of sibling work."""
        from mana_agent.chat.activity import normalize_activity, operation_key, delivery_key, is_mutation, bound_operations, accepts_activity_update
        ev = normalize_activity(event)
        event_type = ev["type"]
        meta = event_classification_metadata(ev)
        task_id = str(ev.get("task_id") or "")
        generation = int(ev.get("attempt_generation") or 0)
        known = self._attempt_generations.get(task_id, 0)
        if generation and known and generation < known:
            return None
        if task_id and generation and meta.get("execution_supervisor"):
            self._attempt_generations[task_id] = max(known, generation)
            if generation > known and event_type == "task.started" and task_id == ev.get("root_task_id"):
                self.is_completed = self.is_failed = self.is_cancelled = False
        phase, default_title = classify_runtime_phase(event_type, meta)
        status = str(ev.get("status") or ("failed" if event_type.endswith((".failed", ".error")) else "running")).lower()
        detail = str(ev.get("detail") or ev.get("summary") or ev.get("message") or ev.get("error") or "")
        key = delivery_key(ev)
        if key in self._seen_sub_event_ids:
            return self._step_by_id.get(str(ev.get("operation_id") or ""))
        self._seen_sub_event_ids.add(key)
        if len(self._seen_sub_event_ids) > 4000:
            self._seen_sub_event_ids = {key}
        category = classify_activity_category(event_type, ev.get("tool_name"), meta)
        ev["category"] = category
        collection = self.coding_events if category == "coding" else self.main_events
        op_key = operation_key(ev)
        for i, row in enumerate(collection):
            if operation_key(row) == op_key:
                if not accepts_activity_update(row, ev):
                    return None
                collection[i] = {**row, **ev}
                break
        else:
            collection.append(ev)
        collection[:] = bound_operations(collection)
        if is_mutation(ev):
            for path in [ev.get("path") or meta.get("path"), *(ev.get("changed_files") or meta.get("changed_files") or [])]:
                if path and str(path) not in self.changed_files:
                    self.changed_files.append(str(path))
        if phase == "lifecycle":
            return None

        approval = meta.get("inbox_item_id") or meta.get("permission_request_id") or meta.get("action_id")
        operation = ev.get("operation_id") or ev.get("tool_call_id") or meta.get("tool_call_id") or meta.get("call_id") or ev.get("call_id")
        actor = ev.get("subagent_id") or ev.get("agent_id") or ""
        owner = ":".join(str(ev.get(k) or "") for k in ("task_id", "attempt_id")) + ":" + str(actor)
        if phase == "approval" and approval:
            step_id = f"{self.turn_id}:approval:{approval}"
        elif operation:
            step_id = f"{self.turn_id}:{owner}:{operation}" if owner.strip(":") else f"{self.turn_id}:{phase}:{operation}"
        elif phase == "tool":
            legacy = str(ev.get("event_id") or event_type)
            for suffix in ("-started", "-completed", "-failed", "-success", "-running"):
                if legacy.endswith(suffix):
                    legacy = legacy[:-len(suffix)]
                    break
            step_id = f"{self.turn_id}:tool:{owner}:{legacy}"
        else:
            step_id = f"{self.turn_id}:{phase}" + (f":{owner}" if owner.strip(":") else "")

        # Backend turn completion is distinct from the root goal's terminal result.
        root_terminal = event_type in {"turn.finished", "turn.cancelled", "turn.timeout", "turn.failed", "loop.completed", "execution.failed", "execution.cancelled", "execution.timeout"} and not ev.get("subagent_id")
        if event_type == "turn.completed" and not ev.get("backend"):
            root_terminal = True
        if (event_type in {"task.completed", "task.failed", "task.cancelled"} and ev.get("task_id") == ev.get("root_task_id")
                and meta.get("execution_supervisor")):
            root_terminal = True
        if event_type == "error" and not ev.get("backend") and not ev.get("subagent_id") and not operation:
            root_terminal = True
        if root_terminal:
            if self.is_failed or self.is_cancelled:
                return None
            failed = status in {"failed", "error", "cancelled", "interrupted"}
            self.is_cancelled = status in {"cancelled", "interrupted"}
            self.is_failed = failed and not self.is_cancelled
            self.is_completed = not failed and status in {"success", "completed", "done"}
            if status in {"waiting", "waiting_approval", "blocked"}:
                self.is_completed = False
            for pending in self.active_steps:
                # A finished root does not prove an unobserved tool outcome.
                if failed:
                    has_operation = pending.metadata.get("operation_id") or pending.metadata.get("tool_call_id") or pending.metadata.get("call_id")
                    pending.finish(status="unverified" if has_operation else "cancelled" if self.is_cancelled else "failed")
                elif self.is_completed:
                    pending.finish(status="unverified", detail=pending.detail or "No terminal operation event observed")
            self.active_step_id = None

        terminal = status in {"success", "completed", "done", "failed", "error", "cancelled", "interrupted", "rejected", "superseded", "unverified"}
        waiting = status in {"waiting_approval", "pending_approval", "approval_required", "waiting_permission"}
        initial = "waiting_approval" if waiting else status
        existing = self._step_by_id.get(step_id)
        if existing is not None:
            # Late start/progress cannot resurrect a terminal operation.
            if existing.status in {"completed", "success", "failed", "error", "cancelled", "unverified"} and not terminal:
                return existing
            if terminal:
                existing.finish(status=status, detail=detail or existing.detail)
            else:
                existing.status = initial
                existing.detail = detail or existing.detail
            existing.title = str(ev.get("title") or existing.title)
            existing.metadata.update(meta)
            existing.sub_events = (existing.sub_events + [ev])[-40:]
            if ev.get("duration_ms") is not None:
                existing.duration_ms = int(ev["duration_ms"])
            if self.active_step_id == step_id and terminal:
                self.active_step_id = next((step.step_id for step in reversed(self.active_steps)), None)
            return existing

        step = ExecutionStep(step_id=step_id, turn_id=self.turn_id, phase=phase,
                             title=str(ev.get("title") or default_title), status=initial,
                             detail=detail, metadata=meta)
        if terminal:
            step.finish(status=status, detail=detail)
            if ev.get("duration_ms") is not None:
                step.duration_ms = int(ev["duration_ms"])
        else:
            self.active_step_id = step_id
        step.sub_events.append(ev)
        self.steps.append(step)
        self._step_by_id[step_id] = step
        self._step_ids_by_phase[phase] = step_id
        active_ids = {st.step_id for st in self.active_steps}
        recent = {st.step_id for st in self.steps[-200:]}
        self.steps[:] = [st for st in self.steps if st.step_id in active_ids | recent]
        self._step_by_id = {st.step_id: st for st in self.steps}
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
    "classify_activity_category",
    "classify_runtime_phase",
]
