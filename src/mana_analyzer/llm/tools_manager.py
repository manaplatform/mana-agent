from __future__ import annotations
import logging
import ast
import hashlib
import json
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence, TypeVar

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from mana_analyzer.llm.gate_command import (
    GATE_POLICIES,
    GateCommand,
    PolicyDecision,
    ProofResult,
    build_gate_command,
    can_run_final_report,
    can_run_verify,
    preflight_tool_policy,
    reconcile_gate_pointer,
    tool_fingerprint,
    validate_gate_proof,
)
from mana_analyzer.llm.goal_profiles import GoalProfile, active_goal_profile
from mana_analyzer.llm.tool_worker_process import ToolRunRequest, ToolRunResponse, ToolWorkerClient
from mana_analyzer.llm.tools_executor import (
    BatchToolRequest,
    BatchExecutionResult,
    LocalToolsExecutor,
    ToolsExecutionConfig,
    ToolsExecutor,
)
from mana_analyzer.services.coding_memory_service import CodingMemoryService

logger = logging.getLogger(__name__)

PlanDecision = Literal["continue", "revise", "finalize", "stop"]
StepStatus = Literal["pending", "in_progress", "done", "blocked"]
TodoKind = Literal["discover", "read", "edit", "verify", "test", "summarize"]
TodoStatus = Literal["pending", "in_progress", "worker_done", "agent_confirmed", "failed", "blocked"]
RunPhase = Literal[
    "DISCOVERY",
    "READING",
    "EXTRACTION",
    "PATCHING",
    "VERIFYING",
    "FINAL",
]
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class ToolsPlanStep(BaseModel):
    id: str
    title: str
    tool_intent: Literal["inspect", "search", "edit", "verify", "answer"]
    args_hint: str = ""
    success_signal: str = ""
    fallback: str = ""
    status: StepStatus = "pending"


class ToolsPlan(BaseModel):
    objective: str
    steps: list[ToolsPlanStep] = Field(default_factory=list)
    current_step_id: str = ""
    decision: PlanDecision = "continue"
    decision_reason: str = ""
    stop_conditions: list[str] = Field(default_factory=list)
    finalize_action: str = ""


class ToolsManagerRequest(BaseModel):
    question: str
    tool_policy_override: dict[str, Any] | None = None
    timeout_seconds: int | None = None
    tool_name: str = ""
    tool_args: dict[str, Any] = Field(default_factory=dict)
    mutating: bool = False
    strategy_hint: str = ""


class ToolsManagerBatch(BaseModel):
    planner_step_id: str = ""
    batch_reason: str = ""
    requests: list[ToolsManagerRequest] = Field(default_factory=list)
    continue_after: bool = True
    expected_progress: str = ""


class AutoExecuteResult(BaseModel):
    answer: str = ""
    sources: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    plan: dict[str, Any] | None = None
    passes: int = 0
    terminal_reason: str = ""
    toolsmanager_requests_count: int = 0
    pass_logs: list[dict[str, Any]] = Field(default_factory=list)
    planner_decisions: list[dict[str, Any]] = Field(default_factory=list)
    prechecklist: dict[str, Any] | None = None
    prechecklist_source: str = ""
    prechecklist_warning: str = ""
    execution_backend: str = "local"
    execution_run_id: str = ""
    execution_duration_ms: float = 0.0
    execution_requests_ok: int = 0
    execution_requests_failed: int = 0
    duplicate_request_skips: int = 0
    duplicate_semantic_search_skips: int = 0
    duplicate_tool_execution_blocks: int = 0
    request_retry_attempts: int = 0
    request_retry_exhausted: int = 0
    edit_retry_mode_activations: int = 0
    persisted_fingerprint_counts: dict[str, int] = Field(default_factory=dict)
    run_id: str = ""
    run_dir: str = ""
    run_status: str = ""
    resume_command: str = ""
    next_action: str = ""


class TodoLedgerItem(BaseModel):
    id: str
    gate: str = ""
    title: str
    kind: TodoKind
    status: TodoStatus = "pending"
    target_files: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    required_tool: str = ""
    dependencies: list[str] = Field(default_factory=list)
    done_condition: str = ""
    max_attempts: int = 3
    attempt_count: int = 0
    worker_checked: bool = False
    agent_confirmed: bool = False
    proof: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

    @property
    def is_complete(self) -> bool:
        return (
            self.status == "agent_confirmed"
            and self.worker_checked
            and self.agent_confirmed
            and bool(self.proof)
        )


class RunStateStore:
    """Persistent checkpoint state for resumable tools-manager runs."""

    phases: list[RunPhase] = [
        "DISCOVERY",
        "READING",
        "EXTRACTION",
        "PATCHING",
        "VERIFYING",
        "FINAL",
    ]
    gates = [
        "locate_candidates",
        "read_candidates",
        "classify_evidence",
        "plan_patch",
        "apply_changes",
        "verify_changes",
        "final_report",
    ]
    mutation_tools = {"apply_patch", "write_file", "create_file"}
    verification_tools = {"run_command", "verify_project"}
    # Internal placeholder recorded when a worker request does not resolve to a
    # concrete enumerated tool (the agentic ask path). It is never a real tool.
    toolsmanager_request_sentinel = "toolsmanager_request"
    # Tools that may complete a todo even when they fall outside the gate's
    # allow-list, because they are framework-injected rather than a worker tool
    # violation: the sentinel above is not a tool at all, and ``read_file`` is
    # forced by the pending-read queue and is side-effect-free. The kind-specific
    # checks in ``confirm_or_reject_todo`` still enforce the real invariants
    # (edits must use a mutation tool, reads must clear the pending queue).
    confirm_disallowed_tool_exemptions = {toolsmanager_request_sentinel, "read_file"}
    _gate_to_phase: dict[str, RunPhase] = {
        "locate_candidates": "DISCOVERY",
        "read_candidates": "READING",
        "classify_evidence": "EXTRACTION",
        "plan_patch": "PATCHING",
        "apply_changes": "PATCHING",
        "verify_changes": "VERIFYING",
        "final_report": "FINAL",
    }

    def __init__(self, *, repo_root: Path, run_id: str | None = None) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.run_id = str(run_id or uuid.uuid4().hex[:12]).strip()
        self.run_dir = self.repo_root / ".mana" / "runs" / self.run_id

    @staticmethod
    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def ensure(self, *, goal: str, flow_id: str = "") -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        defaults: dict[str, Any] = {
            "state.json": self._default_state(goal=goal, flow_id=flow_id),
            "todo.json": {
                "pending_file_reads": [],
                "pending_edits": [],
                "verification_status": "pending",
                "todos": [],
            },
            "visited_files.json": {"files": []},
        }
        for name, payload in defaults.items():
            path = self.run_dir / name
            if not path.exists():
                self._write_json(path, payload)
        for name in ("evidence.jsonl", "tool_calls.jsonl", "summary.md", "resume_prompt.md"):
            path = self.run_dir / name
            if not path.exists():
                path.write_text("", encoding="utf-8")
        checkpoint_path = self.run_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            self.write_checkpoint(
                status="running",
                completed_gates=[],
                pending_gates=list(self.gates),
                files_changed=[],
                verification_status="pending",
            )

    def _default_state(self, *, goal: str, flow_id: str) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": "running",
            "goal": str(goal or "").strip(),
            "original_user_task": str(goal or "").strip(),
            "flow_id": str(flow_id or "").strip(),
            "root_dir": str(self.repo_root),
            "repo_root": str(self.repo_root),
            "current_gate": self.gates[0],
            "current_phase": "DISCOVERY",
            "completed_gates": [],
            "completed_phases": [],
            "pending_gates": list(self.gates),
            "pending_phases": list(self.phases),
            "blocking_reason": "",
            "required_evidence": [],
            "pending_file_reads": [],
            "pending_edits": [],
            "blocked_files": [],
            "verification_status": "pending",
            "next_action": "locate candidate files",
            "next_exact_action": "locate candidate files",
            "progress_counters": {
                "passes": 0,
                "tool_calls": 0,
                "candidate_files": 0,
                "files_read": 0,
                "pending_files": 0,
                "blocked_files": 0,
                "new_findings": 0,
                "successful_patches": 0,
                "verification_commands": 0,
                "no_progress_count": 0,
            },
            "last_error": "",
            "created_at": self.utc_now(),
            "updated_at": self.utc_now(),
        }

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def read_json(self, name: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
        path = self.run_dir / name
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else dict(default or {})
        except Exception:
            return dict(default or {})

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        self._write_json(self.run_dir / name, payload)

    def append_jsonl(self, name: str, payload: dict[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def read_jsonl(self, name: str) -> list[dict[str, Any]]:
        path = self.run_dir / name
        rows: list[dict[str, Any]] = []
        if not path.exists():
            return rows
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    @staticmethod
    def _normalize_args(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): RunStateStore._normalize_args(value[key]) for key in sorted(value)}
        if isinstance(value, list):
            return [RunStateStore._normalize_args(item) for item in value]
        if isinstance(value, str):
            return re.sub(r"\s+", " ", value.strip())
        return value

    @staticmethod
    def _normalize_action_text(value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip().lower())
        text = re.sub(r"\bpass\s+\d+\b", "", text)
        text = re.sub(r"\bplanner pass\s+\d+\b", "", text)
        text = re.sub(r"\bfallback request\b", "", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _extract_arg_text(cls, args: dict[str, Any], keys: Sequence[str], question: str = "") -> str:
        for key in keys:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return cls._normalize_action_text(value)
        for nested_key in ("tool_args", "args", "arguments"):
            nested = args.get(nested_key)
            if isinstance(nested, dict):
                nested_value = cls._extract_arg_text(nested, keys, "")
                if nested_value:
                    return nested_value
        text = str(question or "")
        for key in keys:
            match = re.search(rf"\b{re.escape(key)}\s*[:=]\s*['\"]?([^'\"\n]+)", text, flags=re.IGNORECASE)
            if match:
                return cls._normalize_action_text(match.group(1))
        return ""

    @classmethod
    def normalized_action_key(cls, *, tool_name: str, args: dict[str, Any], question: str = "") -> str:
        tool = str(tool_name or "").strip().lower()
        payload = args if isinstance(args, dict) else {}
        if not tool:
            normalized_question = cls._normalize_action_text(question)
            for known in (
                "repo_search",
                "semantic_search",
                "read_file",
                "run_command",
                "find_symbols",
                "call_graph",
                "apply_patch",
                "write_file",
                "verify_project",
            ):
                if known in normalized_question:
                    tool = known
                    break
        if tool == "read_file":
            path = cls._extract_arg_text(payload, ("path", "file", "file_path", "target_file"), question)
            if not path:
                match = re.search(r"\bread(?:_file)?\s+([^\s`'\";]+)", str(question or ""), flags=re.IGNORECASE)
                path = cls._normalize_action_text(match.group(1)) if match else ""
            return f"read_file:{path}" if path else "read_file"
        if tool in {"repo_search", "semantic_search"}:
            query = cls._extract_arg_text(payload, ("query", "q", "pattern"), question)
            glob = cls._extract_arg_text(payload, ("glob", "path_glob", "include"), question)
            if not query:
                text = cls._normalize_action_text(question)
                text = re.sub(r"\b(repo_search|semantic_search|search|find|locate|grep)\b", " ", text)
                text = re.sub(r"\b(k|max_steps|timeout_seconds)\s*[:=]\s*\d+\b", " ", text)
                query = re.sub(r"\s+", " ", text).strip()
            prefix = "repo_search" if tool == "repo_search" else "semantic_search"
            return f"{prefix}:{query}:{glob}"
        if tool == "run_command":
            command = cls._extract_arg_text(payload, ("cmd", "command", "shell_command"), question)
            if not command:
                command = cls._normalize_action_text(question)
                command = re.sub(r"^(run_command|run command|execute command)\s*[:=]?\s*", "", command).strip()
            return f"run_command:{command}"
        if tool in {"find_symbols", "call_graph"}:
            path = cls._extract_arg_text(payload, ("path", "file", "file_path"), question)
            symbol = cls._extract_arg_text(payload, ("symbol", "name", "query"), question)
            return f"{tool}:{path}:{symbol}"
        if tool in {"apply_patch", "write_file"}:
            path = cls._extract_arg_text(payload, ("path", "file", "file_path", "target_file"), question)
            return f"{tool}:{path or cls._normalize_action_text(question)[:160]}"
        return f"{tool or 'toolsmanager_request'}:{cls._normalize_action_text(question)[:220]}"

    def fingerprint(
        self,
        *,
        gate: str,
        tool_name: str,
        args: dict[str, Any],
        filters: dict[str, Any] | None = None,
    ) -> str:
        action_key = self.normalized_action_key(
            tool_name=tool_name,
            args=args,
            question=str(args.get("question", "") if isinstance(args, dict) else ""),
        )
        payload = {
            "action_key": action_key,
            "repo_root": str(self.repo_root),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def successful_tool_call(self, fingerprint: str) -> dict[str, Any] | None:
        target = str(fingerprint or "").strip()
        if not target:
            return None
        for row in reversed(self.read_jsonl("tool_calls.jsonl")):
            if row.get("fingerprint") == target and row.get("status") == "ok":
                return row
        return None

    def gate_tool_fingerprint(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        gate: str,
        target_file: str = "",
    ) -> str:
        """Fingerprint a concrete tool call (name + args + gate + target file)."""
        return tool_fingerprint(
            tool_name=tool_name,
            args=args,
            gate=gate,
            target_file=target_file,
        )

    def duplicate_decision(self, fingerprint: str) -> str:
        """Decide whether a concrete tool call may run again.

        - ``allow``: fingerprint never seen.
        - ``skip_completed``: fingerprint already succeeded, skip it.
        - ``retry_once``: fingerprint returned no_progress once; allow exactly
          one explicit retry.
        - ``block_duplicate``: fingerprint already retried after no_progress;
          caller must increment ``duplicate_tool_calls_blocked`` and skip.
        """
        target = str(fingerprint or "").strip()
        if not target:
            return "allow"
        no_progress = 0
        for row in self.read_jsonl("tool_calls.jsonl"):
            if row.get("fingerprint") != target:
                continue
            status = str(row.get("status", "") or "")
            if status == "ok":
                return "skip_completed"
            if status.startswith("skipped_duplicate"):
                return "block_duplicate"
            if status in {"no_progress", "skipped_no_progress"} or status.startswith("no_progress"):
                no_progress += 1
        if no_progress == 0:
            return "allow"
        if no_progress == 1:
            return "retry_once"
        return "block_duplicate"

    @staticmethod
    def _gate_from_step(step: "ToolsPlanStep | None") -> str:
        if step is None:
            return "locate_candidates"
        intent = str(step.tool_intent or "").strip()
        title = str(step.title or "").lower()
        if intent in {"inspect", "search"}:
            if "read" in title:
                return "read_candidates"
            return "locate_candidates"
        if intent == "edit":
            return "apply_changes"
        if intent == "verify":
            return "verify_changes"
        return "final_report"

    @classmethod
    def _phase_from_gate(cls, gate: str) -> RunPhase:
        return cls._gate_to_phase.get(str(gate or "").strip(), "DISCOVERY")

    def required_gate_for_step(self, step: "ToolsPlanStep | None") -> str:
        """Resolve the executable gate from persisted state before planner phase hints."""
        state = self.read_json("state.json", {})
        pending_gates = [
            str(item).strip()
            for item in (state.get("pending_gates") if isinstance(state.get("pending_gates"), list) else [])
            if str(item).strip()
        ]
        if pending_gates:
            pending_gate = pending_gates[0]
            if pending_gate in {"apply_changes", "verify_changes", "final_report"}:
                return pending_gate
        return self._gate_from_step(step)

    # ------------------------------------------------------------------ #
    # Coding-agent gate authority (GateCommand + policy + proof)
    # ------------------------------------------------------------------ #

    def pending_files_count(self) -> int:
        """Number of still-unread candidate files."""
        todo = self.read_json("todo.json", {})
        goal = str(self.read_json("state.json", {}).get("goal", "") or "")
        pending = self._canonical_pending_reads(
            todo.get("pending_file_reads") if isinstance(todo.get("pending_file_reads"), list) else [],
            goal=goal,
        )
        return len(pending)

    def reconcile_gate_pointer(self) -> dict[str, Any]:
        """Reconcile current_gate/current_phase from completed + pending gates.

        This is the authoritative resume-time reconciliation: the current gate
        is always the first genuinely pending gate, and the phase is derived
        from that gate (never kept as stale free-text). Only the coding agent
        (through this store) may move the pointer.
        """
        state = self.read_json("state.json", self._default_state(goal="", flow_id=""))
        completed = [
            str(item).strip()
            for item in (state.get("completed_gates") if isinstance(state.get("completed_gates"), list) else [])
            if str(item).strip()
        ]
        pending = [
            str(item).strip()
            for item in (state.get("pending_gates") if isinstance(state.get("pending_gates"), list) else [])
            if str(item).strip()
        ]
        if not pending and not completed:
            pending = list(self.gates)
        gate, phase = reconcile_gate_pointer(
            completed_gates=completed,
            pending_gates=pending,
            pending_files=self.pending_files_count(),
        )
        state["current_gate"] = gate
        state["current_phase"] = phase
        state["pending_gates"] = [item for item in self.gates if item not in set(completed)]
        completed_phases = [self._phase_from_gate(item) for item in completed if item in self.gates]
        state["completed_phases"] = list(dict.fromkeys(completed_phases))
        state["pending_phases"] = [item for item in self.phases if item not in state["completed_phases"]]
        state["updated_at"] = self.utc_now()
        self.write_json("state.json", state)
        return state

    def build_gate_command(self, gate: str, *, goal: str = "", plan_id: str = "") -> GateCommand:
        """Issue the authoritative GateCommand for a gate (worker contract)."""
        todo = self.current_todo_for_gate(gate, goal=goal)
        return build_gate_command(
            gate=gate,
            run_id=self.run_id,
            plan_id=plan_id,
            step_id=gate,
            todo_id=todo.id,
            target_files=todo.target_files,
            done_condition=todo.done_condition,
            max_attempts=int(todo.max_attempts or 3),
        )

    def preflight_tool(
        self,
        command: GateCommand,
        *,
        tool_name: str,
        inner_tool: str = "",
    ) -> PolicyDecision:
        """Preflight a worker-selected tool against its GateCommand policy."""
        return preflight_tool_policy(command, tool_name=tool_name, inner_tool=inner_tool)

    def validate_gate_proof(self, gate: str, proof: dict[str, Any]) -> ProofResult:
        return validate_gate_proof(gate, proof)

    @staticmethod
    def _public_phase_name(phase: str) -> str:
        mapping = {
            "DISCOVER_FILES": "DISCOVERY",
            "READ_FILES": "READING",
            "ANALYZE_EVIDENCE": "EXTRACTION",
            "PLAN_PATCH": "PATCHING",
            "APPLY_PATCH": "PATCHING",
            "VERIFY": "VERIFYING",
            "FINAL_REPORT": "FINAL",
        }
        return mapping.get(str(phase or "").strip(), str(phase or "").strip() or "DISCOVERY")

    @staticmethod
    def active_goal_profile(goal: str) -> GoalProfile | None:
        return active_goal_profile(goal)

    def _profile_candidate_priority(self, path: str, *, profile: GoalProfile | None = None) -> int:
        if profile is None:
            return 50
        return profile.priority(path, self.repo_root)

    def _is_relevant_candidate(self, path: str, *, profile: GoalProfile | None = None) -> bool:
        if profile is None:
            return True
        return profile.is_relevant(path, self.repo_root)

    def _sort_pending_reads(self, pending_reads: Sequence[Any], *, goal: str = "") -> list[str]:
        sanitized = self._sanitize_pending_reads(pending_reads)
        profile = self.active_goal_profile(goal)
        if profile is None:
            return sanitized
        relevant = [path for path in sanitized if self._is_relevant_candidate(path, profile=profile)]
        return sorted(dict.fromkeys(relevant), key=lambda item: (self._profile_candidate_priority(item, profile=profile), item))

    def seed_goal_profile_queue(self, profile: GoalProfile | None = None) -> None:
        state = self.read_json("state.json", self._default_state(goal="", flow_id=""))
        goal = str(state.get("goal", state.get("original_user_task", "")) or "")
        profile = profile or self.active_goal_profile(goal)
        if profile is None:
            return
        todo = self.read_json("todo.json", {})
        existing = self._known_evidence_paths()
        pending = self._sanitize_pending_reads(
            todo.get("pending_file_reads") if isinstance(todo.get("pending_file_reads"), list) else []
        )
        discovered: list[str] = []
        try:
            for pattern in profile.discovery_globs:
                for path in self.repo_root.glob(pattern):
                    if not path.is_file():
                        continue
                    try:
                        rel = path.resolve().relative_to(self.repo_root).as_posix()
                    except Exception:
                        continue
                    normalized = self._normalize_candidate_path(rel)
                    if normalized and self._is_relevant_candidate(normalized, profile=profile):
                        discovered.append(normalized)
        except Exception:
            return
        for rel in sorted(dict.fromkeys(discovered), key=lambda item: (self._profile_candidate_priority(item, profile=profile), item)):
            if rel not in existing:
                self.append_jsonl(
                    "evidence.jsonl",
                    {
                        "timestamp": self.utc_now(),
                        "file_path": rel,
                        "evidence_type": "candidate_file",
                        "reason_discovered": f"deterministic_profile_queue:{profile.id}",
                        "status": "located_not_read",
                        "source_tool": "run_command",
                        "next_action": "read_file",
                        "confidence": 0.95,
                    },
                )
                existing.add(rel)
            if rel not in pending and rel not in self.read_files():
                pending.append(rel)
        todo["pending_file_reads"] = self._sort_pending_reads(pending, goal=goal)
        self.write_json("todo.json", todo)

    def seed_candidate_queue(self) -> None:
        self.seed_goal_profile_queue()

    def update_state(
        self,
        *,
        plan: "ToolsPlan | None" = None,
        step: "ToolsPlanStep | None" = None,
        status: str | None = None,
        blocking_reason: str = "",
        next_action: str = "",
        changed_files: Sequence[str] = (),
    ) -> dict[str, Any]:
        state = self.read_json("state.json", self._default_state(goal="", flow_id=""))
        todo = self.read_json("todo.json", {})
        gate = self.required_gate_for_step(step)
        if plan is not None and str(plan.decision or "").strip().lower() in {"finalize", "stop"}:
            gate = "final_report"
        has_edit_step = bool(
            plan is not None
            and any(str(item.tool_intent or "").strip() == "edit" for item in getattr(plan, "steps", []) or [])
        )
        if (
            has_edit_step
            and gate in self.gates
            and self.gates.index(gate) > self.gates.index("apply_changes")
            and not self._has_applied_changes(changed_files)
        ):
            gate = "apply_changes"
            if not blocking_reason:
                blocking_reason = "no changed files; docs/models.md not created/updated"
        completed = list(state.get("completed_gates") if isinstance(state.get("completed_gates"), list) else [])
        if not self._has_applied_changes(changed_files):
            completed = [item for item in completed if item != "apply_changes"]
        if not self._has_mutation_payload() and not self._has_applied_changes(changed_files):
            completed = [item for item in completed if item != "plan_patch"]
            if gate == "apply_changes":
                blocking_reason = "missing_edit_payload"
        if changed_files and "apply_changes" not in completed:
            completed.append("apply_changes")
        if gate in self.gates:
            prior = self.gates[: self.gates.index(gate)]
            for item in prior:
                if item == "plan_patch" and not self._has_mutation_payload() and not self._has_applied_changes(changed_files):
                    continue
                if item == "apply_changes" and not self._has_applied_changes(changed_files):
                    continue
                if item not in completed:
                    completed.append(item)
        pending = [item for item in self.gates if item not in completed]
        if status:
            state["status"] = status
        state["current_gate"] = gate
        state["completed_gates"] = completed
        state["pending_gates"] = pending
        state["blocking_reason"] = blocking_reason
        pending_reads = self._canonical_pending_reads(
            todo.get("pending_file_reads") if isinstance(todo.get("pending_file_reads"), list) else [],
            goal=str(state.get("goal", state.get("original_user_task", "")) or ""),
        )
        if pending_reads != (todo.get("pending_file_reads") if isinstance(todo.get("pending_file_reads"), list) else []):
            todo["pending_file_reads"] = pending_reads
            self.write_json("todo.json", todo)
        state["pending_file_reads"] = pending_reads
        state["pending_edits"] = list(todo.get("pending_edits", []))
        state["verification_status"] = str(todo.get("verification_status", "pending") or "pending")
        state["next_action"] = next_action or self.next_action()
        state["next_exact_action"] = state["next_action"]
        phase = self._phase_from_gate(gate)
        if pending_reads and phase == "DISCOVERY":
            phase = "READING"
            gate = "read_candidates"
            state["current_gate"] = gate
        state["current_phase"] = phase
        completed_phases = [self._phase_from_gate(item) for item in completed if item in self.gates]
        state["completed_phases"] = list(dict.fromkeys(completed_phases))
        state["pending_phases"] = [item for item in self.phases if item not in state["completed_phases"]]
        state["updated_at"] = self.utc_now()
        self.write_json("state.json", state)
        return state

    def read_files(self) -> set[str]:
        return {path for path, status in self._latest_file_statuses().items() if status == "read"}

    @staticmethod
    def _row_sort_key(index: int, row: dict[str, Any]) -> tuple[str, int]:
        return (str(row.get("timestamp", "") or ""), index)

    def _latest_file_statuses(self) -> dict[str, str]:
        events: list[tuple[tuple[str, int], str, str]] = []
        for index, row in enumerate(self.read_jsonl("evidence.jsonl")):
            path = self._normalize_candidate_path(str(row.get("file_path", "") or ""))
            status = str(row.get("status", "") or "").strip()
            if path and status:
                events.append((self._row_sort_key(index, row), path, status))
        for index, row in enumerate(self.read_jsonl("tool_calls.jsonl")):
            if row.get("status") != "ok":
                continue
            for item in row.get("files_read") if isinstance(row.get("files_read"), list) else []:
                path = self._normalize_candidate_path(str(item))
                if path:
                    events.append((self._row_sort_key(index, row), path, "read"))
        latest: dict[str, str] = {}
        for _sort_key, path, status in sorted(events, key=lambda item: item[0]):
            if status == "located_not_read" and latest.get(path) == "read":
                continue
            latest[path] = status
        return latest

    def _canonical_pending_reads(self, pending_reads: Sequence[Any], *, goal: str = "") -> list[str]:
        read_files = self.read_files()
        pending = [path for path in self._sanitize_pending_reads(pending_reads) if path not in read_files]
        return self._sort_pending_reads(pending, goal=goal)

    def _reconcile_read_paths(self, paths: Sequence[Any], *, gate: str, source_tool: str) -> tuple[int, int]:
        normalized_paths = [
            path
            for path in dict.fromkeys(self._normalize_candidate_path(str(item)) for item in paths)
            if path
        ]
        if not normalized_paths:
            return 0, 0
        before_read = self.read_files()
        evidence_read = {
            self._normalize_candidate_path(str(row.get("file_path", "") or ""))
            for row in self.read_jsonl("evidence.jsonl")
            if str(row.get("status", "") or "").strip() == "read"
        }
        appended = 0
        for path in normalized_paths:
            if path not in evidence_read:
                self.append_jsonl(
                    "evidence.jsonl",
                    {
                        "timestamp": self.utc_now(),
                        "file_path": path,
                        "evidence_type": "candidate_file",
                        "reason_discovered": f"{gate}:{source_tool}:read_reconciled",
                        "status": "read",
                        "source_tool": source_tool,
                        "next_action": "classify evidence",
                        "confidence": 0.8,
                    },
                )
                appended += 1
        todo = self.read_json("todo.json", {})
        state = self.read_json("state.json", {})
        goal = str(state.get("goal", state.get("original_user_task", "")) or "")
        todo["pending_file_reads"] = self._canonical_pending_reads(
            todo.get("pending_file_reads") if isinstance(todo.get("pending_file_reads"), list) else [],
            goal=goal,
        )
        self.write_json("todo.json", todo)
        visited = self.read_json("visited_files.json", {"files": []})
        visited_files = set(visited.get("files") if isinstance(visited.get("files"), list) else [])
        visited_files.update(normalized_paths)
        self.write_json("visited_files.json", {"files": sorted(visited_files)})
        newly_read = len([path for path in normalized_paths if path not in before_read])
        return newly_read, appended

    def _has_applied_changes(self, changed_files: Sequence[str] = ()) -> bool:
        if any(str(path).strip() for path in changed_files):
            return True
        for row in self.read_jsonl("tool_calls.jsonl"):
            if row.get("status") != "ok":
                continue
            tool = str(row.get("tool_name", "") or "").strip().lower()
            if tool in {"apply_patch", "write_file", "create_file"} and any(
                str(path).strip()
                for path in (row.get("files_changed") if isinstance(row.get("files_changed"), list) else [])
            ):
                return True
        return False

    def _has_mutation_payload(self) -> bool:
        todo = self.read_json("todo.json", {})
        state = self.read_json("state.json", {})
        pending_edits = todo.get("pending_edits") if isinstance(todo.get("pending_edits"), list) else []
        if any(pending_edits):
            return True
        for key in ("patch_plan", "write_file_payload", "create_file_payload"):
            value = todo.get(key)
            if value:
                return True
            value = state.get(key)
            if value:
                return True
        return False

    def completed_fingerprints(self) -> set[str]:
        return {
            str(row.get("fingerprint", ""))
            for row in self.read_jsonl("tool_calls.jsonl")
            if row.get("status") == "ok" and str(row.get("fingerprint", "")).strip()
        }

    def _known_evidence_paths(self) -> set[str]:
        return {str(row.get("file_path", "") or "") for row in self.read_jsonl("evidence.jsonl")}

    _ignored_candidate_parts = {
        ".git",
        ".hg",
        ".svn",
        ".mana",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "site-packages",
        "dist-packages",
    }
    _ignored_candidate_names = {
        "checkpoint.json",
        "state.json",
        "summary.md",
        "todo.json",
        "tool_calls.json",
        "tool_calls.jsonl",
        "visited_files.json",
        "work_ledger.json",
        "resume_prompt.md",
        "evidence.json",
        "evidence.jsonl",
    }

    def _normalize_candidate_path(self, value: str) -> str:
        text = str(value or "").strip().replace("\\", "/")
        if not text or text.startswith(("http://", "https://")) or "\n" in text or len(text) > 500:
            return ""
        try:
            path = Path(text)
            if path.is_absolute():
                resolved = path.resolve()
                try:
                    rel = resolved.relative_to(self.repo_root)
                except ValueError:
                    return ""
                text = rel.as_posix()
        except Exception:
            return ""
        text = text.lstrip("./")
        parts = [part for part in Path(text).parts if part not in {"", "."}]
        if not parts or any(part in self._ignored_candidate_parts for part in parts):
            return ""
        if parts[-1] in self._ignored_candidate_names:
            return ""
        if any(part.endswith(".egg-info") or part.endswith(".dist-info") for part in parts):
            return ""
        if not re.search(r"(^|/)[\w.\-]+(\.py|\.md|\.txt|\.toml|\.yaml|\.yml|\.json|models\.py)$", text):
            return ""
        return text

    def _validated_repo_file_path(self, value: str) -> str:
        normalized = self._normalize_candidate_path(value)
        if not normalized:
            return ""
        try:
            resolved = (self.repo_root / normalized).resolve()
            resolved.relative_to(self.repo_root)
        except Exception:
            return ""
        if not resolved.is_file():
            return ""
        return normalized

    def _extract_paths(self, value: Any) -> set[str]:
        paths: set[str] = set()
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = str(key).lower()
                if lowered in {"file", "path", "file_path", "filepath", "relative_path"} and isinstance(item, str):
                    normalized = self._validated_repo_file_path(item)
                    if normalized:
                        paths.add(normalized)
                else:
                    paths.update(self._extract_paths(item))
        elif isinstance(value, list):
            for item in value:
                paths.update(self._extract_paths(item))
        elif isinstance(value, str):
            for match in re.findall(r"(?:(?:^|\s|['\"])([\w./-]*(?:models\.py|[\w.-]+\.(?:py|md|txt|toml|yaml|yml|json))))", value):
                normalized = self._validated_repo_file_path(match)
                if normalized:
                    paths.add(normalized)
        return paths

    def _sanitize_pending_reads(self, pending_reads: Sequence[Any]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in pending_reads:
            normalized = self._validated_repo_file_path(str(item))
            if normalized and normalized not in seen:
                seen.add(normalized)
                out.append(normalized)
        return out

    def _read_file_category(self, path: str) -> str:
        normalized = self._normalize_candidate_path(path)
        if not normalized:
            return "artifact"
        name = Path(normalized).name
        if name in self._ignored_candidate_names or normalized.startswith(".mana/runs/"):
            return "artifact"
        state = self.read_json("state.json", {})
        goal = str(state.get("goal", state.get("original_user_task", "")) or "")
        target_paths = {
            item
            for item in (
                self._normalize_candidate_path(match)
                for match in re.findall(r"(?:^|\s)([\w./-]+\.(?:md|py|txt|toml|yaml|yml|json))", goal)
            )
            if item
        }
        if normalized in target_paths:
            return "target"
        return "source"

    def record_evidence_from_response(
        self,
        *,
        gate: str,
        source_tool: str,
        response: ToolRunResponse,
    ) -> dict[str, int]:
        existing = self._known_evidence_paths()
        todo = self.read_json("todo.json", {})
        state = self.read_json("state.json", {})
        goal = str(state.get("goal", state.get("original_user_task", "")) or "")
        pending_reads = self._canonical_pending_reads(
            todo.get("pending_file_reads") if isinstance(todo.get("pending_file_reads"), list) else [],
            goal=goal,
        )
        visited = self.read_json("visited_files.json", {"files": []})
        visited_files = set(visited.get("files") if isinstance(visited.get("files"), list) else [])
        discovered = 0
        read = 0
        tool = str(source_tool or "").strip().lower()
        payload = {"trace": response.trace, "sources": response.sources, "answer": response.answer}
        paths = sorted(self._extract_paths(payload))
        profile = self.active_goal_profile(goal)
        if profile is not None:
            paths = [path for path in paths if self._is_relevant_candidate(path, profile=profile)]
        for path in paths:
            is_read = tool == "read_file" or any(
                str(row.get("tool_name", "")).strip().lower() == "read_file"
                and path in self._extract_paths(row)
                for row in response.trace
                if isinstance(row, dict)
            )
            status = "read" if is_read else "located_not_read"
            if path not in existing:
                self.append_jsonl(
                    "evidence.jsonl",
                    {
                        "timestamp": self.utc_now(),
                        "file_path": path,
                        "evidence_type": "candidate_file",
                        "reason_discovered": f"{gate}:{source_tool}",
                        "status": status,
                        "source_tool": source_tool,
                        "next_action": "classify evidence" if is_read else "read_file",
                        "confidence": 0.7 if is_read else 0.5,
                    },
                )
                existing.add(path)
                discovered += 1
            elif is_read and path not in self.read_files():
                self.append_jsonl(
                    "evidence.jsonl",
                    {
                        "timestamp": self.utc_now(),
                        "file_path": path,
                        "evidence_type": "candidate_file",
                        "reason_discovered": f"{gate}:{source_tool}:read_reconciled",
                        "status": "read",
                        "source_tool": source_tool,
                        "next_action": "classify evidence",
                        "confidence": 0.8,
                    },
                )
            if is_read:
                read += 1
                visited_files.add(path)
                pending_reads = [item for item in pending_reads if item != path]
            elif path not in pending_reads:
                pending_reads.append(path)
        todo["pending_file_reads"] = self._canonical_pending_reads(pending_reads, goal=goal)
        self.write_json("todo.json", todo)
        self.write_json("visited_files.json", {"files": sorted(visited_files)})
        return {"discovered": discovered, "read": read, "pending_reads": len(pending_reads)}

    def mark_read_skipped(self, path: str, *, reason: str) -> None:
        normalized = self._normalize_candidate_path(path)
        if not normalized:
            return
        state = self.read_json("state.json", self._default_state(goal="", flow_id=""))
        goal = str(state.get("goal", state.get("original_user_task", "")) or "")
        todo = self.read_json("todo.json", {})
        pending_reads = self._canonical_pending_reads(
            todo.get("pending_file_reads") if isinstance(todo.get("pending_file_reads"), list) else [],
            goal=goal,
        )
        if normalized in pending_reads:
            todo["pending_file_reads"] = [item for item in pending_reads if item != normalized]
            self.write_json("todo.json", todo)
        self.append_jsonl(
            "evidence.jsonl",
            {
                "timestamp": self.utc_now(),
                "file_path": normalized,
                "evidence_type": "candidate_file",
                "reason_discovered": "read_failed",
                "status": "skipped_no_progress",
                "source_tool": "read_file",
                "next_action": self.next_action(),
                "confidence": 0.0,
                "error": reason[:500],
            },
        )

    def record_tool_call(
        self,
        *,
        gate: str,
        tool_name: str,
        normalized_args: dict[str, Any],
        fingerprint: str,
        status: str,
        result_summary: str = "",
        files_discovered: Sequence[str] = (),
        files_read: Sequence[str] = (),
        files_changed: Sequence[str] = (),
        error: str = "",
    ) -> None:
        phase = self._public_phase_name(self._phase_from_gate(gate))
        is_duplicate = str(status or "").startswith("skipped_duplicate")
        next_action = self.next_action()
        self.append_jsonl(
            "tool_calls.jsonl",
            {
                "timestamp": self.utc_now(),
                "gate": gate,
                "phase": phase,
                "tool_name": tool_name,
                "normalized_key": fingerprint,
                "purpose": gate,
                "normalized_args": self._normalize_args(normalized_args),
                "fingerprint": fingerprint,
                "status": status,
                "is_duplicate": is_duplicate,
                "duplicate_of": fingerprint if is_duplicate else "",
                "produced_new_evidence": bool(files_discovered or files_read or files_changed),
                "files_found": list(files_discovered),
                "result_summary": result_summary[:500],
                "files_discovered": list(files_discovered),
                "files_read": list(files_read),
                "files_changed": list(files_changed),
                "next_action": next_action,
                "ledger_checkpoint_path": str(self.run_dir / "work_ledger.json"),
                "error": error,
            },
        )
        if status == "ok" and files_read:
            self._reconcile_read_paths(files_read, gate=gate, source_tool=tool_name)
        self.write_work_ledger(
            status="running",
            checkpoint_reason="tool_call_recorded",
            last_error=error,
        )

    def next_action(self) -> str:
        todo = self.read_json("todo.json", {})
        state = self.read_json("state.json", {})
        pending_gates = [
            str(item).strip()
            for item in (state.get("pending_gates") if isinstance(state.get("pending_gates"), list) else [])
            if str(item).strip()
        ]
        first_gate = pending_gates[0] if pending_gates else ""
        pending_edits = todo.get("pending_edits") if isinstance(todo.get("pending_edits"), list) else []
        if first_gate == "apply_changes":
            if pending_edits:
                return f"apply pending edit {pending_edits[0]}"
            return "apply_changes requires mutation payload"
        if first_gate == "verify_changes":
            return "verify changed files"
        if first_gate == "final_report":
            return "final_report"
        pending_reads = self._canonical_pending_reads(
            todo.get("pending_file_reads") if isinstance(todo.get("pending_file_reads"), list) else [],
            goal=str(state.get("goal", state.get("original_user_task", "")) or ""),
        )
        if pending_reads != (todo.get("pending_file_reads") if isinstance(todo.get("pending_file_reads"), list) else []):
            todo["pending_file_reads"] = pending_reads
            self.write_json("todo.json", todo)
        if pending_reads:
            return f"read_file {pending_reads[0]}"
        if pending_edits:
            return f"apply pending edit {pending_edits[0]}"
        return f"continue gate {pending_gates[0]}" if pending_gates else "final_report"

    def progress_counters(self, *, pass_logs: Sequence[dict[str, Any]] = (), tool_calls: int = 0) -> dict[str, int]:
        evidence_rows = self.read_jsonl("evidence.jsonl")
        todo = self.read_json("todo.json", {})
        state = self.read_json("state.json", {})
        candidate_files = {str(row.get("file_path", "")) for row in evidence_rows if row.get("file_path")}
        read_files = self.read_files()
        read_categories = {path: self._read_file_category(path) for path in read_files}
        blocked_files = state.get("blocked_files") if isinstance(state.get("blocked_files"), list) else []
        verification_commands = 0
        successful_patches = 0
        for row in self.read_jsonl("tool_calls.jsonl"):
            tool = str(row.get("tool_name", "") or "").strip().lower()
            if row.get("status") == "ok" and tool in {"apply_patch", "write_file", "create_file"}:
                successful_patches += 1
            if row.get("status") == "ok" and tool in {"run_command", "verify_project"}:
                verification_commands += 1
        return {
            "passes": len(list(pass_logs)),
            "tool_calls": int(tool_calls),
            "candidate_files": len(candidate_files),
            "files_read": len(read_files),
            "source_files_read": len([path for path, category in read_categories.items() if category == "source"]),
            "artifact_files_read": len([path for path, category in read_categories.items() if category == "artifact"]),
            "target_files_read": len([path for path, category in read_categories.items() if category == "target"]),
            "total_files_read": len(read_files),
            "pending_files": len(
                self._canonical_pending_reads(
                    todo.get("pending_file_reads") if isinstance(todo.get("pending_file_reads"), list) else [],
                    goal=str(state.get("goal", state.get("original_user_task", "")) or ""),
                )
            ),
            "blocked_files": len(blocked_files),
            "new_findings": len(evidence_rows),
            "successful_patches": successful_patches,
            "verification_commands": verification_commands,
            "no_progress_count": int(state.get("no_progress_count", 0) or 0),
        }

    def write_checkpoint(
        self,
        *,
        status: str,
        completed_gates: Sequence[str],
        pending_gates: Sequence[str],
        files_changed: Sequence[str],
        verification_status: str,
        blocker: str = "",
        pass_logs: Sequence[dict[str, Any]] = (),
        tool_calls: int = 0,
        plan: dict[str, Any] | None = None,
        last_error: str = "",
    ) -> None:
        evidence_rows = self.read_jsonl("evidence.jsonl")
        read_files = sorted(self.read_files())
        located_files = sorted({str(row.get("file_path", "")) for row in evidence_rows if row.get("file_path")})
        completed_searches = [
            row
            for row in self.read_jsonl("tool_calls.jsonl")
            if row.get("status") == "ok" and str(row.get("tool_name", "")).strip().lower() in {"repo_search", "semantic_search", "list_files"}
        ]
        todo = self.read_json("todo.json", {})
        state = self.read_json("state.json", self._default_state(goal="", flow_id=""))
        pending_reads = self._canonical_pending_reads(
            todo.get("pending_file_reads") if isinstance(todo.get("pending_file_reads"), list) else [],
            goal=str(state.get("goal", state.get("original_user_task", "")) or ""),
        )
        if pending_reads != (todo.get("pending_file_reads") if isinstance(todo.get("pending_file_reads"), list) else []):
            todo["pending_file_reads"] = pending_reads
            self.write_json("todo.json", todo)
        next_action = self.next_action()
        current_phase = str(state.get("current_phase", "") or self._phase_from_gate(str(state.get("current_gate", "") or "")))
        counters = self.progress_counters(pass_logs=pass_logs, tool_calls=tool_calls)
        counters["no_progress_count"] = int(state.get("no_progress_count", counters.get("no_progress_count", 0)) or 0)
        checkpoint = {
            "run_id": self.run_id,
            "root_dir": str(self.repo_root),
            "current_phase": current_phase,
            "original_user_task": str(state.get("original_user_task", state.get("goal", "")) or ""),
            "completed_searches": completed_searches,
            "candidate_files": located_files,
            "read_files": read_files,
            "pending_files": pending_reads,
            "blocked_files": list(state.get("blocked_files", []) if isinstance(state.get("blocked_files"), list) else []),
            "evidence_collected": evidence_rows,
            "patch_plan": plan or {},
            "applied_patches": [
                row
                for row in self.read_jsonl("tool_calls.jsonl")
                if row.get("status") == "ok" and str(row.get("tool_name", "")).strip().lower() in {"apply_patch", "write_file", "create_file"}
                and any(str(path).strip() for path in (row.get("files_changed") if isinstance(row.get("files_changed"), list) else []))
            ],
            "verification_commands": [
                row
                for row in self.read_jsonl("tool_calls.jsonl")
                if row.get("status") == "ok" and str(row.get("tool_name", "")).strip().lower() in {"run_command", "verify_project"}
            ],
            "next_exact_action": next_action,
            "next_action": next_action,
            "progress_counters": counters,
            "last_error": last_error or str(state.get("last_error", "") or ""),
            "status": status,
            "updated_at": self.utc_now(),
        }
        self.write_json("checkpoint.json", checkpoint)
        state["progress_counters"] = counters
        state["next_action"] = next_action
        state["next_exact_action"] = next_action
        state["last_error"] = checkpoint["last_error"]
        self.write_json("state.json", state)
        summary_lines = [
            f"# Run {self.run_id}",
            "",
            f"- status: {status}",
            f"- current_phase: {current_phase}",
            f"- completed_gates: {', '.join(completed_gates) or '-'}",
            f"- pending_gates: {', '.join(pending_gates) or '-'}",
            f"- completed_work_count: {len(completed_gates)}",
            f"- pending_work_count: {len(pending_gates) + len(pending_reads)}",
            f"- files_located: {len(located_files)}",
            f"- files_read: {len(read_files)}",
            f"- pending_files: {len(pending_reads)}",
            f"- files_changed: {', '.join(files_changed) or '-'}",
            f"- verification_status: {verification_status}",
            f"- blocker: {blocker or '-'}",
            f"- next_action: {next_action}",
            "",
            self.todo_board(),
        ]
        self.run_dir.joinpath("summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        prompt_lines = [
            f"Resume mana-analyzer run {self.run_id}.",
            f"Goal: {self.read_json('state.json', {}).get('goal', '')}",
            f"Current phase: {current_phase}",
            f"Completed gates: {', '.join(completed_gates) or '-'}",
            f"Pending gates: {', '.join(pending_gates) or '-'}",
            f"Pending file reads: {', '.join(str(x) for x in pending_reads) or '-'}",
            f"Next action: {next_action}",
            "Skip successful tool calls already recorded in tool_calls.jsonl.",
        ]
        self.run_dir.joinpath("resume_prompt.md").write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
        self.write_work_ledger(
            status=status,
            checkpoint_reason=blocker or ("pass_budget_reached" if status == "needs_resume" else "checkpoint_saved"),
            plan=plan,
            last_error=last_error,
        )

    @staticmethod
    def _todo_defaults_for_gate(gate: str, *, model_docs: bool) -> dict[str, Any]:
        if model_docs:
            mapping: dict[str, dict[str, Any]] = {
                "locate_candidates": {
                    "id": "discover_models_sources",
                    "title": "Discover model source files",
                    "kind": "discover",
                    "allowed_tools": ["repo_search", "semantic_search", "list_files", "run_command"],
                    "done_condition": "candidate model source files are located or no candidates remain",
                },
                "read_candidates": {
                    "id": "read_model_sources",
                    "title": "Read model source files",
                    "kind": "read",
                    "allowed_tools": ["read_file"],
                    "required_tool": "read_file",
                    "dependencies": ["discover_models_sources"],
                    "done_condition": "pending_files == 0",
                },
                "classify_evidence": {
                    "id": "classify_model_evidence",
                    "title": "Classify model evidence",
                    "kind": "summarize",
                    "allowed_tools": ["run_command"],
                    "dependencies": ["read_model_sources"],
                    "done_condition": "model evidence is sufficient for docs update",
                },
                "plan_patch": {
                    "id": "plan_docs_models_update",
                    "title": "Plan docs/models.md update",
                    "kind": "summarize",
                    "allowed_tools": ["run_command"],
                    "dependencies": ["classify_model_evidence"],
                    "done_condition": "mutation payload or edit target is prepared",
                },
                "apply_changes": {
                    "id": "update_docs_models_md",
                    "title": "Update docs/models.md",
                    "kind": "edit",
                    "target_files": ["docs/models.md"],
                    "allowed_tools": ["apply_patch", "write_file", "create_file"],
                    "required_tool": "apply_patch|write_file|create_file",
                    "dependencies": ["plan_docs_models_update"],
                    "done_condition": "docs/models.md is modified by a mutation tool",
                },
                "verify_changes": {
                    "id": "verify_docs_models_md",
                    "title": "Verify docs/models.md",
                    "kind": "verify",
                    "target_files": ["docs/models.md"],
                    "allowed_tools": ["run_command", "verify_project"],
                    "required_tool": "run_command|verify_project",
                    "dependencies": ["update_docs_models_md"],
                    "done_condition": "verification command succeeds after mutation",
                },
                "final_report": {
                    "id": "summarize_result",
                    "title": "Summarize result",
                    "kind": "summarize",
                    "allowed_tools": [],
                    "dependencies": ["verify_docs_models_md"],
                    "done_condition": "final answer includes proof and remaining risk",
                },
            }
            return mapping.get(gate, mapping["final_report"])
        generic_mapping: dict[str, dict[str, Any]] = {
            "locate_candidates": {
                "kind": "discover",
                "allowed_tools": ["repo_search", "semantic_search", "list_files", "run_command"],
                "done_condition": "candidate files are located",
            },
            "read_candidates": {
                "kind": "read",
                "allowed_tools": ["read_file"],
                "required_tool": "read_file",
                "done_condition": "target files are read",
            },
            "apply_changes": {
                "kind": "edit",
                "allowed_tools": ["apply_patch", "write_file", "create_file"],
                "required_tool": "apply_patch|write_file|create_file",
                "done_condition": "target file is modified by a mutation tool",
            },
            "verify_changes": {
                "kind": "verify",
                "allowed_tools": ["run_command", "verify_project"],
                "required_tool": "run_command|verify_project",
                "done_condition": "verification command succeeds",
            },
            "final_report": {
                "kind": "summarize",
                "allowed_tools": [],
                "done_condition": "final answer is prepared",
            },
        }
        item = generic_mapping.get(gate, {"kind": "summarize", "allowed_tools": [], "done_condition": "step complete"})
        title = gate.replace("_", " ")
        return {"id": gate, "title": title, **item}

    def ensure_todo_ledger(self, *, goal: str = "") -> list[dict[str, Any]]:
        todo = self.read_json("todo.json", {})
        existing = todo.get("todos") if isinstance(todo.get("todos"), list) else []
        state = self.read_json("state.json", self._default_state(goal=goal, flow_id=""))
        active_profile = self.active_goal_profile(str(goal or state.get("goal", "") or ""))
        model_docs = active_profile is not None and active_profile.id == "model_docs"
        by_id = {
            str(item.get("id", "")): item
            for item in existing
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        }
        ordered: list[dict[str, Any]] = []
        previous_id = ""
        for gate in self.gates:
            defaults = self._todo_defaults_for_gate(gate, model_docs=model_docs)
            todo_id = str(defaults["id"])
            current = dict(by_id.get(todo_id, {}))
            merged = {
                "id": todo_id,
                "title": defaults.get("title", gate.replace("_", " ")),
                "kind": defaults.get("kind", "summarize"),
                "status": "pending",
                "target_files": defaults.get("target_files", []),
                "allowed_tools": defaults.get("allowed_tools", []),
                "required_tool": defaults.get("required_tool", ""),
                "dependencies": defaults.get("dependencies", [previous_id] if previous_id else []),
                "done_condition": defaults.get("done_condition", ""),
                "max_attempts": 3,
                "attempt_count": 0,
                "worker_checked": False,
                "agent_confirmed": False,
                "proof": {},
                "reason": "",
                "gate": gate,
            }
            merged.update(current)
            merged["gate"] = gate
            if not merged.get("dependencies") and previous_id:
                merged["dependencies"] = [previous_id]
            ordered.append(TodoLedgerItem.model_validate(merged).model_dump())
            previous_id = todo_id
        todo["todos"] = ordered
        self.write_json("todo.json", todo)
        return ordered

    def current_todo_for_gate(self, gate: str, *, goal: str = "") -> TodoLedgerItem:
        todos = self.ensure_todo_ledger(goal=goal)
        for item in todos:
            if str(item.get("gate", "") or "") == str(gate or ""):
                return TodoLedgerItem.model_validate(item)
        return TodoLedgerItem.model_validate(todos[-1])

    def _write_todo_item(self, updated: TodoLedgerItem) -> None:
        todo = self.read_json("todo.json", {})
        rows = todo.get("todos") if isinstance(todo.get("todos"), list) else []
        out: list[dict[str, Any]] = []
        replaced = False
        for row in rows:
            if isinstance(row, dict) and str(row.get("id", "")) == updated.id:
                out.append(updated.model_dump())
                replaced = True
            else:
                out.append(row if isinstance(row, dict) else {})
        if not replaced:
            out.append(updated.model_dump())
        todo["todos"] = out
        self.write_json("todo.json", todo)

    def mark_todo_worker_failure(
        self,
        todo_item: TodoLedgerItem,
        *,
        reason: str,
        proof: dict[str, Any] | None = None,
    ) -> TodoLedgerItem:
        cap = max(1, int(todo_item.max_attempts or 3))
        attempts = min(cap, int(todo_item.attempt_count or 0) + 1)
        status: TodoStatus = "blocked" if attempts >= cap else "failed"
        updated = todo_item.model_copy(
            update={
                "status": status,
                "attempt_count": attempts,
                "worker_checked": False,
                "agent_confirmed": False,
                "reason": reason,
                "proof": dict(proof or {}),
            }
        )
        self._write_todo_item(updated)
        return updated

    def mark_todo_worker_done(
        self,
        todo_item: TodoLedgerItem,
        *,
        tool_name: str,
        files_read: Sequence[str] = (),
        files_changed: Sequence[str] = (),
        command_output: str = "",
        verification_result: str = "",
        tool_call_id: str = "",
    ) -> TodoLedgerItem:
        proof = {
            "tool_name": str(tool_name or ""),
            "tool_call_id": str(tool_call_id or ""),
            "files_read": [str(path) for path in files_read if str(path).strip()],
            "modified_files": [str(path) for path in files_changed if str(path).strip()],
            "command_output": str(command_output or "")[:1000],
            "verification_result": str(verification_result or "")[:1000],
        }
        updated = todo_item.model_copy(
            update={
                "status": "worker_done",
                "attempt_count": min(
                    max(1, int(todo_item.max_attempts or 3)),
                    int(todo_item.attempt_count or 0) + 1,
                ),
                "worker_checked": True,
                "agent_confirmed": False,
                "proof": proof,
                "reason": "",
            }
        )
        self._write_todo_item(updated)
        return updated

    def confirm_or_reject_todo(
        self,
        todo_item: TodoLedgerItem,
        *,
        files_changed: Sequence[str] = (),
        pending_files: Sequence[str] = (),
        reason: str = "",
    ) -> TodoLedgerItem:
        proof = dict(todo_item.proof or {})
        tool_name = str(proof.get("tool_name", "") or "")
        modified = {str(path) for path in proof.get("modified_files", []) if str(path).strip()}
        modified.update(str(path) for path in files_changed if str(path).strip())
        allowed = {str(item).strip() for item in todo_item.allowed_tools if str(item).strip()}
        if (
            allowed
            and tool_name
            and tool_name not in allowed
            and tool_name not in self.confirm_disallowed_tool_exemptions
        ):
            return self.mark_todo_worker_failure(
                todo_item,
                reason=f"Worker used disallowed tool: {tool_name}",
                proof=proof,
            )
        if todo_item.kind == "edit":
            target = {str(path) for path in todo_item.target_files if str(path).strip()}
            if not tool_name or tool_name not in self.mutation_tools:
                return self.mark_todo_worker_failure(
                    todo_item,
                    reason="Edit todo claimed complete without required mutation tool",
                    proof=proof,
                )
            if target and not target.intersection(modified):
                return self.mark_todo_worker_failure(
                    todo_item,
                    reason="Edit todo claimed complete but no target file was modified",
                    proof={**proof, "modified_files": sorted(modified)},
                )
            if not modified:
                return self.mark_todo_worker_failure(
                    todo_item,
                    reason="Edit todo claimed complete but modified_files was empty",
                    proof=proof,
                )
        if todo_item.kind == "read" and pending_files:
            return self.mark_todo_worker_failure(
                todo_item,
                reason=f"Read todo still has pending files: {len(pending_files)}",
                proof=proof,
            )
        updated = todo_item.model_copy(
            update={
                "status": "agent_confirmed",
                "worker_checked": True,
                "agent_confirmed": True,
                "proof": {**proof, "modified_files": sorted(modified)},
                "reason": reason,
            }
        )
        self._write_todo_item(updated)
        return updated

    def validate_planner_todo_claims(self, *, changed_files: Sequence[str] = ()) -> list[str]:
        todos = self.ensure_todo_ledger(goal="")
        warnings: list[str] = []
        changed = {str(path) for path in changed_files if str(path).strip()}
        for row in todos:
            item = TodoLedgerItem.model_validate(row)
            if item.kind != "edit":
                continue
            target = {str(path) for path in item.target_files if str(path).strip()}
            proof_modified = {
                str(path)
                for path in (item.proof.get("modified_files") if isinstance(item.proof, dict) else []) or []
                if str(path).strip()
            }
            if item.status == "agent_confirmed" and target and not target.intersection(changed.union(proof_modified)):
                self.mark_todo_worker_failure(
                    item,
                    reason="Planner contradiction: edit todo confirmed without target modification",
                    proof=item.proof,
                )
                warnings.append("planner_contradiction_edit_without_modified_target")
        return warnings

    def todo_board(self) -> str:
        rows = self.ensure_todo_ledger(goal="")
        lines = ["Todo Board:"]
        for row in rows:
            item = TodoLedgerItem.model_validate(row)
            left = "!" if item.status in {"failed", "blocked"} else ("x" if item.worker_checked else " ")
            right = "x" if item.agent_confirmed else " "
            reason = str(item.reason or item.done_condition or "").strip()
            suffix = f" - {reason}" if reason else ""
            lines.append(f"[{left}][{right}] {item.id} - {item.status}{suffix}")
        return "\n".join(lines)

    def write_work_ledger(
        self,
        *,
        status: str,
        checkpoint_reason: str = "",
        plan: dict[str, Any] | None = None,
        last_error: str = "",
    ) -> None:
        state = self.read_json("state.json", self._default_state(goal="", flow_id=""))
        todo = self.read_json("todo.json", {})
        checkpoint = self.read_json("checkpoint.json", {})
        tool_history = self.read_jsonl("tool_calls.jsonl")
        evidence_rows = self.read_jsonl("evidence.jsonl")
        candidate_files = sorted({str(row.get("file_path", "")) for row in evidence_rows if row.get("file_path")})
        read_files = sorted(self.read_files())
        modified_files = sorted(
            {
                str(path)
                for row in tool_history
                for path in (
                    row.get("files_changed") if isinstance(row.get("files_changed"), list) else []
                )
                if str(path).strip()
            }
        )
        searched_queries = [
            row.get("normalized_args", {})
            for row in tool_history
            if str(row.get("tool_name", "")).strip().lower() in {"repo_search", "semantic_search", "list_files", "run_command"}
        ]
        verification_commands = [
            row.get("normalized_args", {})
            for row in tool_history
            if str(row.get("tool_name", "")).strip().lower() in {"run_command", "verify_project"}
        ]
        last_success = next((row for row in reversed(tool_history) if row.get("status") == "ok"), {})
        phase = self._public_phase_name(str(state.get("current_phase", checkpoint.get("current_phase", "DISCOVERY"))))
        ledger = {
            "run_id": self.run_id,
            "objective": str(state.get("goal", state.get("original_user_task", "")) or ""),
            "current_phase": phase,
            "completed_steps": list(state.get("completed_gates", []) if isinstance(state.get("completed_gates"), list) else []),
            "pending_steps": list(state.get("pending_gates", []) if isinstance(state.get("pending_gates"), list) else []),
            "searched_queries": searched_queries,
            "read_files": read_files,
            "candidate_files": candidate_files,
            "modified_files": modified_files,
            "verification_commands": verification_commands,
            "tool_call_history": tool_history,
            "duplicate_tool_calls_blocked": len([row for row in tool_history if row.get("is_duplicate")]),
            "last_successful_action": {
                "tool_name": last_success.get("tool_name", ""),
                "normalized_key": last_success.get("normalized_key", last_success.get("fingerprint", "")),
                "summary": last_success.get("result_summary", ""),
            },
            "next_action": self.next_action(),
            "checkpoint_reason": checkpoint_reason,
            "status": status,
            "pending_work": {
                "pending_file_reads": self._canonical_pending_reads(
                    todo.get("pending_file_reads") if isinstance(todo.get("pending_file_reads"), list) else [],
                    goal=str(state.get("goal", state.get("original_user_task", "")) or ""),
                ),
                "pending_edits": list(todo.get("pending_edits", []) if isinstance(todo.get("pending_edits"), list) else []),
            },
            "todo_board": self.todo_board(),
            "todos": todo.get("todos") if isinstance(todo.get("todos"), list) else [],
            "checkpoint_path": str(self.run_dir / "work_ledger.json"),
            "checkpoint_json_path": str(self.run_dir / "checkpoint.json"),
            "last_error": last_error or str(state.get("last_error", "") or ""),
            "updated_at": self.utc_now(),
        }
        if plan is not None:
            ledger["plan"] = plan
        self.write_json("work_ledger.json", ledger)



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
        decision_provider: Any = None,
    ) -> None:
        _ = (api_key, model, base_url, decision_provider)
        self.worker_client = worker_client
        self.repo_root = Path(repo_root).resolve()
        self.execution_config = execution_config or ToolsExecutionConfig()
        self.executor = executor
        self.coding_memory_service = coding_memory_service
        self._decision_provider = decision_provider

    def attach_decision_provider(self, provider: Any) -> None:
        # The queue manager is deterministic and does not use an LLM planner;
        # the provider is retained only for API compatibility.
        self._decision_provider = provider

    def update_model(self, new_model: str) -> None:
        logger.info("Ignoring model update; QueueManager is deterministic-only.")

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
    ) -> AutoExecuteResult:
        from mana_analyzer.llm.agent_work_queue import (
            AgentWorkQueue,
            TaskBoard,
            WorkItem,
            WorkQueueRunner,
        )
        from mana_analyzer.llm.agent_work_queue_adapters import (
            CodingAgentSniffer,
            make_worker_executor,
        )

        _ = (index_dirs, max_no_progress_passes, flow_context)
        queue = AgentWorkQueue()
        board = TaskBoard(queue=queue)
        profile = active_goal_profile(request)
        if profile is not None:
            def _relevant(path: str) -> bool:
                return profile.is_relevant(path, self.repo_root)
        else:
            def _relevant(path: str) -> bool:
                return True

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
        base_execute = make_worker_executor(
            worker_client=self.worker_client,
            repo_root=self.repo_root,
            on_event=on_event,
            default_timeout=int(timeout_seconds),
            default_k=int(k),
            default_max_steps=int(max_steps),
            tool_policy=tool_policy,
            index_dir=str(index_dir) if index_dir else None,
            flow_id=flow_id,
        )

        def execute(item: "WorkItem"):  # noqa: F821 - imported above
            result = base_execute(item)
            if result.answer:
                answers.append(result.answer)
            sources.extend(result.sources)
            trace.extend(result.trace)
            return result

        sniffer = CodingAgentSniffer(repo_root=self.repo_root, relevant=_relevant)
        runner = WorkQueueRunner(
            queue=queue,
            execute=execute,
            sniffer=sniffer,
            board=board,
            max_steps=max(12, int(pass_cap) * 8),
        )
        report = runner.run()
        store = RunStateStore(repo_root=self.repo_root, run_id=run_id)
        return AutoExecuteResult(
            answer="\n\n".join(a for a in answers if a).strip() or board.render(),
            sources=sources,
            trace=trace,
            warnings=[],
            passes=report.steps,
            terminal_reason=report.terminal_reason,
            toolsmanager_requests_count=report.steps,
            execution_backend="work_queue",
            execution_run_id=store.run_id,
            execution_duration_ms=report.duration_ms,
            execution_requests_ok=report.done,
            execution_requests_failed=report.failed,
            pass_logs=[{"made_progress": report.done > 0, "board": board.render()}],
            run_id=store.run_id,
            run_dir=str(store.run_dir),
            run_status="completed",
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
            f"Resume mana-analyzer run {run_id}"
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
    "ToolsPlan",
    "ToolsPlanStep",
    "ToolsManagerRequest",
    "ToolsManagerBatch",
    "AutoExecuteResult",
    "QueueManager",
]
