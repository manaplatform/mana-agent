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
        if allowed and tool_name and tool_name not in allowed:
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


class ToolsManagerDecisionProvider(Protocol):
    def plan_with_source(
        self,
        *,
        request: str,
        flow_context: str | None,
        pass_index: int,
        pass_cap: int,
        previous_plan: ToolsPlan | None,
        pass_logs: Sequence[dict[str, Any]],
        warnings: Sequence[str],
        changed_files: Sequence[str],
        latest_answer: str,
    ) -> tuple[ToolsPlan, list[str], str]:
        ...

    def build_batch(
        self,
        *,
        request: str,
        flow_context: str | None,
        plan: ToolsPlan,
        pass_index: int,
        pass_cap: int,
        pass_logs: Sequence[dict[str, Any]],
        warnings: Sequence[str],
        changed_files: Sequence[str],
        latest_answer: str,
    ) -> tuple[ToolsManagerBatch | None, list[str]]:
        ...



class _InternalDecisionProvider:
    """Fallback decision provider for tests and deterministic CLI resume paths."""

    def __init__(self, orchestrator: "ToolsManagerOrchestrator") -> None:
        self._orchestrator = orchestrator

    def plan_with_source(
        self,
        *,
        request: str,
        flow_context: str | None,
        pass_index: int = 0,
        pass_cap: int = 4,
        previous_plan: "ToolsPlan | None" = None,
        pass_logs: "Sequence[dict[str, Any]]" = (),
        warnings: "Sequence[str]" = (),
        changed_files: "Sequence[str]" = (),
        latest_answer: str = "",
    ) -> "tuple[ToolsPlan, list[str], str]":
        import json as _json
        issues: list[str] = []
        payload = {
            "request": request,
            "flow_context": (flow_context or "none").strip(),
            "pass_index": int(pass_index),
            "pass_cap": int(pass_cap),
            "previous_plan": previous_plan.model_dump() if previous_plan is not None else None,
            "pass_logs": list(pass_logs)[-4:],
            "warnings": list(warnings)[-12:],
            "changed_files": list(changed_files),
            "latest_answer": str(latest_answer or "")[:1500],
        }
        human_prompt = _json.dumps(payload, ensure_ascii=False, indent=2)
        try:
            raw = self._orchestrator._invoke_model(
                system_prompt="tools_planner",
                human_prompt=human_prompt,
            )
        except RuntimeError as exc:
            if "no longer supports model invocation" not in str(exc):
                raise
            issues.append("head_tools_planner unavailable; using deterministic fallback")
            fallback = self._orchestrator._deterministic_fallback_plan(
                request=request,
                flow_context=flow_context,
                previous_plan=previous_plan,
                reason="planner_unavailable",
            )
            return fallback, issues, "deterministic_fallback"
        parsed = self._orchestrator.parse_tools_plan(raw, request=request, previous_plan=previous_plan)
        if parsed is not None:
            return parsed, issues, "planner"

        issues.append("head_tools_planner parse failed; attempting repair")
        repair_raw = self._orchestrator._invoke_model(
            system_prompt="tools_planner_repair",
            human_prompt=(
                "Repair this planner output to strict JSON schema.\n"
                "Do not add markdown. Return only one JSON object.\n\n"
                f"Broken output:\n{raw}\n\n"
                f"Execution payload:\n{human_prompt}"
            ),
        )
        repaired = self._orchestrator.parse_repair(
            repair_raw, "plan", request=request, previous_plan=previous_plan
        )
        if isinstance(repaired, ToolsPlan):
            return repaired, issues, "planner_repair"

        issues.append("head_tools_planner repair failed; using deterministic fallback")
        fallback = self._orchestrator._deterministic_fallback_plan(
            request=request,
            flow_context=flow_context,
            previous_plan=previous_plan,
            reason="planner_parse_failed",
        )
        return fallback, issues, "deterministic_fallback"

    def build_batch(
        self,
        *,
        request: str,
        flow_context: str | None,
        plan: "ToolsPlan",
        pass_index: int,
        pass_cap: int,
        pass_logs: "Sequence[dict[str, Any]]" = (),
        warnings: "Sequence[str]" = (),
        changed_files: "Sequence[str]" = (),
        latest_answer: str = "",
    ) -> "tuple[ToolsManagerBatch | None, list[str]]":
        import json as _json
        issues: list[str] = []
        payload = {
            "request": request,
            "flow_context": (flow_context or "").strip(),
            "planner": plan.model_dump(),
            "pass_index": int(pass_index),
            "pass_cap": int(pass_cap),
            "pass_logs": list(pass_logs)[-4:],
            "warnings": list(warnings)[-10:],
            "changed_files": list(changed_files),
            "latest_answer": str(latest_answer or "")[:1500],
        }
        human_prompt = _json.dumps(payload, ensure_ascii=False, indent=2)
        try:
            raw = self._orchestrator._invoke_model(
                system_prompt="toolsmanager",
                human_prompt=human_prompt,
            )
        except RuntimeError as exc:
            if "no longer supports model invocation" not in str(exc):
                raise
            issues.append("toolsmanager unavailable; using deterministic fallback request")
            step = self._orchestrator._resolve_step(plan)
            fallback = self._orchestrator._deterministic_fallback_request(
                request=request,
                flow_context=flow_context,
                plan=plan,
                step=step,
                pass_index=pass_index,
            )
            if fallback is None:
                return None, issues
            return (
                ToolsManagerBatch(
                    planner_step_id=str(plan.current_step_id or ""),
                    batch_reason="deterministic_unavailable_batch_fallback",
                    requests=[fallback],
                    continue_after=True,
                    expected_progress="Execute deterministic fallback request for current planner step.",
                ),
                issues,
            )
        batch = self._orchestrator.parse_tools_batch(raw, planner_step_id=plan.current_step_id)
        if batch is not None:
            return batch, issues

        issues.append("toolsmanager batch invalid; attempting repair")
        repair_raw = self._orchestrator._invoke_model(
            system_prompt="toolsmanager_repair",
            human_prompt=(
                "Repair this tools-manager output to strict JSON schema.\n"
                "Do not add markdown. Return only one JSON object.\n\n"
                f"Broken output:\n{raw}\n\n"
                f"Execution payload:\n{human_prompt}"
            ),
        )
        repaired = self._orchestrator.parse_repair(
            repair_raw, "batch", request=request, previous_plan=plan,
            planner_step_id=plan.current_step_id,
        )
        if isinstance(repaired, ToolsManagerBatch):
            return repaired, issues

        issues.append("toolsmanager repair failed")
        return None, issues


class ToolsManagerOrchestrator:
    """Planner-driven auto-execution loop for agent-tools chat turns."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        worker_client: ToolWorkerClient,
        repo_root: Path,
        base_url: str | None = None,
        execution_config: ToolsExecutionConfig | None = None,
        executor: ToolsExecutor | None = None,
        coding_memory_service: CodingMemoryService | None = None,
        decision_provider: ToolsManagerDecisionProvider | None = None,
    ) -> None:
        _ = (api_key, model, base_url)
        self.worker_client = worker_client
        self.repo_root = repo_root.resolve()
        self.execution_config = execution_config or ToolsExecutionConfig()
        self.executor = executor or LocalToolsExecutor(worker_client=worker_client)
        self.coding_memory_service = coding_memory_service
        self._decision_provider: ToolsManagerDecisionProvider | None = decision_provider

    def _setup_llm(self) -> None:
        """Deprecated: tools manager is deterministic and does not use an LLM."""
        return None

    @staticmethod
    def _normalize_request(req: ToolsManagerRequest) -> ToolsManagerRequest:
        normalized_tool_name = str(req.tool_name or "").strip()
        normalized_tool_args = dict(req.tool_args or {})
        normalized_strategy = str(req.strategy_hint or "").strip().lower()
        if normalized_tool_name and not req.question:
            inferred = f"run tool {normalized_tool_name}"
        else:
            inferred = str(req.question or "")
        return ToolsManagerRequest(
            question=inferred,
            tool_policy_override=dict(req.tool_policy_override or {}),
            timeout_seconds=req.timeout_seconds,
            tool_name=normalized_tool_name,
            tool_args=normalized_tool_args,
            mutating=bool(req.mutating),
            strategy_hint=normalized_strategy,
        )

    @staticmethod
    def _should_force_write_fallback(
        *,
        request: ToolsManagerRequest,
        patch_attempts: int,
        saw_no_change: bool,
        failed: bool,
    ) -> bool:
        if str(request.tool_name or "") != "apply_patch":
            return False
        if str(request.strategy_hint or "") not in ("", "auto"):
            return False
        return failed or saw_no_change or patch_attempts >= 2

    def update_model(self, new_model: str) -> None:
        """No-op: tools manager has no model dependency."""
        logger.info("Ignoring model update; ToolsManagerOrchestrator is deterministic-only.")
        _ = new_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True
    )
    def _call_llm_with_retry(self, messages: list[Any]) -> Any:
        """
        متد مرکزی برای تمام فراخوانی‌های LLM.
        این متد مجهز به Retry با Exponential Backoff است.
        """
        _ = messages
        raise RuntimeError("ToolsManagerOrchestrator no longer supports LLM calls")

    @staticmethod
    def _strip_code_fence(raw: str) -> str:
        text = str(raw or "").strip()
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return text

    @staticmethod
    def _parse_json_or_literal(raw: str) -> Any | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        try:
            return ast.literal_eval(text)
        except Exception:
            return None

    @staticmethod
    def _extract_json_object_text(text: str) -> str | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        if raw.startswith("{") and raw.endswith("}"):
            return raw
        start = raw.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(raw)):
            ch = raw[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start : idx + 1].strip()
        return None

    @classmethod
    def _collect_candidates(cls, raw_text: str) -> list[Any]:
        pending: list[Any] = [raw_text]
        candidates: list[Any] = []
        seen_text: set[str] = set()
        seen_ids: set[int] = set()

        while pending:
            item = pending.pop(0)
            if isinstance(item, str):
                text = item.strip()
                if not text or text in seen_text:
                    continue
                seen_text.add(text)
                candidates.append(text)

                unwrapped = cls._strip_code_fence(text)
                if unwrapped and unwrapped not in seen_text:
                    pending.append(unwrapped)

                obj_text = cls._extract_json_object_text(text)
                if obj_text and obj_text not in seen_text:
                    pending.append(obj_text)

                parsed = cls._parse_json_or_literal(text)
                if parsed is not None:
                    pending.append(parsed)
                continue

            if isinstance(item, dict):
                marker = id(item)
                if marker in seen_ids:
                    continue
                seen_ids.add(marker)
                candidates.append(item)

                for key in ("answer", "content", "text", "message", "output", "payload", "data", "raw"):
                    if key in item:
                        pending.append(item.get(key))
                for value in item.values():
                    if isinstance(value, (dict, list)):
                        pending.append(value)
                    elif isinstance(value, str) and len(value) <= 20000:
                        pending.append(value)
                continue

            if isinstance(item, list):
                marker = id(item)
                if marker in seen_ids:
                    continue
                seen_ids.add(marker)
                candidates.append(item)
                pending.extend(item)

        return candidates

    @classmethod
    def _parse_model(cls, text: str, model_cls: type[_ModelT]) -> _ModelT:
        last_error: Exception | None = None
        for candidate in cls._collect_candidates(text):
            if isinstance(candidate, dict):
                try:
                    return model_cls.model_validate(candidate)
                except Exception as exc:
                    last_error = exc
            elif isinstance(candidate, str):
                parsed = cls._parse_json_or_literal(candidate)
                if isinstance(parsed, dict):
                    try:
                        return model_cls.model_validate(parsed)
                    except Exception as exc:
                        last_error = exc
        if last_error is not None:
            raise ValueError(str(last_error)) from last_error
        raise ValueError("No valid JSON object found")

    @staticmethod
    def _status_from_text(text: str) -> StepStatus:
        lowered = str(text or "").strip().lower()
        if lowered in {"pending", "in_progress", "done", "blocked"}:
            return lowered  # type: ignore[return-value]
        return "pending"

    def _normalize_plan(self, plan: ToolsPlan, *, previous_plan: ToolsPlan | None = None) -> ToolsPlan:
        steps: list[ToolsPlanStep] = []
        seen_ids: set[str] = set()
        for idx, step in enumerate(plan.steps, start=1):
            base_id = str(step.id or "").strip() or f"s{idx}"
            step_id = base_id
            suffix = 1
            while step_id in seen_ids:
                suffix += 1
                step_id = f"{base_id}_{suffix}"
            seen_ids.add(step_id)
            steps.append(
                ToolsPlanStep(
                    id=step_id,
                    title=str(step.title or "").strip() or f"Step {idx}",
                    tool_intent=step.tool_intent,
                    args_hint=str(step.args_hint or "").strip(),
                    success_signal=str(step.success_signal or "").strip(),
                    fallback=str(step.fallback or "").strip(),
                    status=self._status_from_text(step.status),
                )
            )

        if not steps and previous_plan is not None and previous_plan.steps:
            steps = [ToolsPlanStep.model_validate(item.model_dump()) for item in previous_plan.steps]

        if not steps:
            steps = [
                ToolsPlanStep(
                    id="s1",
                    title="Inspect target files",
                    tool_intent="inspect",
                    args_hint="Choose repo_search, semantic_search, read_file, find_symbols, or call_graph for the task.",
                    success_signal="relevant file context gathered",
                    fallback="If unknown files, run targeted search once.",
                    status="in_progress",
                ),
                ToolsPlanStep(
                    id="s2",
                    title="Apply requested changes",
                    tool_intent="edit",
                    args_hint="Use apply_patch first, write_file fallback if needed.",
                    success_signal="requested edits applied",
                    fallback="Use write_file if patch loop fails twice.",
                    status="pending",
                ),
                ToolsPlanStep(
                    id="s3",
                    title="Verify and finalize",
                    tool_intent="verify",
                    args_hint="Run targeted verification and summarize.",
                    success_signal="verification complete",
                    fallback="If verification tooling unavailable, state limits and remaining risk.",
                    status="pending",
                ),
            ]

        objective = str(plan.objective or "").strip() or "Execute requested plan"
        decision: PlanDecision = str(plan.decision or "continue").strip().lower()  # type: ignore[assignment]
        if decision not in {"continue", "revise", "finalize", "stop"}:
            decision = "continue"

        current_step_id = str(plan.current_step_id or "").strip()
        if current_step_id not in {step.id for step in steps}:
            active = next((step for step in steps if step.status not in {"done", "blocked"}), None)
            current_step_id = active.id if active is not None else steps[0].id

        if decision in {"finalize", "stop"} and not str(plan.decision_reason or "").strip():
            decision_reason = "Planner marked terminal decision."
        else:
            decision_reason = str(plan.decision_reason or "").strip()

        stop_conditions = [str(item).strip() for item in plan.stop_conditions if str(item).strip()]
        if not stop_conditions:
            stop_conditions = [
                "Planner chooses finalize/stop",
                "Two consecutive non-actionable passes",
                "Pass cap reached",
            ]

        finalize_action = str(plan.finalize_action or "").strip() or "Return final answer with completed work and verification."

        return ToolsPlan(
            objective=objective,
            steps=steps,
            current_step_id=current_step_id,
            decision=decision,
            decision_reason=decision_reason,
            stop_conditions=stop_conditions,
            finalize_action=finalize_action,
        )

    def _deterministic_fallback_plan(
        self,
        *,
        request: str,
        flow_context: str | None,
        previous_plan: ToolsPlan | None,
        reason: str,
    ) -> ToolsPlan:
        if previous_plan is not None:
            base = self._normalize_plan(previous_plan, previous_plan=previous_plan)
            return ToolsPlan(
                objective=base.objective,
                steps=base.steps,
                current_step_id=base.current_step_id,
                decision="continue",
                decision_reason=f"Deterministic fallback: {reason}",
                stop_conditions=base.stop_conditions,
                finalize_action=base.finalize_action,
            )

        context_hint = ""
        if flow_context:
            for line in str(flow_context).splitlines():
                text = line.strip()
                if text.lower().startswith("current objective:"):
                    context_hint = text.split(":", 1)[1].strip()
                    break

        objective = context_hint or (" ".join((request or "").strip().split())[:220] or "Execute requested plan")
        plan = ToolsPlan(
            objective=objective,
            steps=[],
            current_step_id="",
            decision="continue",
            decision_reason=f"Deterministic fallback: {reason}",
            stop_conditions=[],
            finalize_action="Return final answer with completed work.",
        )
        return self._normalize_plan(plan, previous_plan=None)

    def parse_tools_plan(
        self,
        raw_text: str,
        *,
        request: str,
        previous_plan: ToolsPlan | None = None,
    ) -> ToolsPlan | None:
        try:
            parsed = self._parse_model(raw_text, ToolsPlan)
            return self._normalize_plan(parsed, previous_plan=previous_plan)
        except Exception:
            return None

    def _normalize_batch(self, batch: ToolsManagerBatch, *, planner_step_id: str) -> ToolsManagerBatch:
        requests: list[ToolsManagerRequest] = []
        for item in batch.requests[:8]:
            question = str(item.question or "").strip()
            if not question:
                continue
            override = item.tool_policy_override if isinstance(item.tool_policy_override, dict) else None
            timeout = item.timeout_seconds if isinstance(item.timeout_seconds, int) else None
            requests.append(
                ToolsManagerRequest(
                    question=question,
                    tool_policy_override=override,
                    timeout_seconds=timeout,
                    tool_name=str(item.tool_name or "").strip(),
                    tool_args=dict(item.tool_args or {}),
                    mutating=bool(item.mutating),
                    strategy_hint=str(item.strategy_hint or "").strip().lower(),
                )
            )

        resolved_step = str(batch.planner_step_id or "").strip() or planner_step_id
        return ToolsManagerBatch(
            planner_step_id=resolved_step,
            batch_reason=str(batch.batch_reason or "").strip() or "toolsmanager_batch",
            requests=requests,
            continue_after=bool(batch.continue_after),
            expected_progress=str(batch.expected_progress or "").strip(),
        )

    def parse_tools_batch(
        self,
        raw_text: str,
        *,
        planner_step_id: str,
    ) -> ToolsManagerBatch | None:
        try:
            parsed = self._parse_model(raw_text, ToolsManagerBatch)
            batch = self._normalize_batch(parsed, planner_step_id=planner_step_id)
            self._validate_batch(batch)
            return batch
        except Exception:
            return None

    def parse_repair(
        self,
        raw_text: str,
        schema_kind: Literal["plan", "batch"],
        *,
        request: str,
        previous_plan: ToolsPlan | None = None,
        planner_step_id: str = "",
    ) -> ToolsPlan | ToolsManagerBatch | None:
        if schema_kind == "plan":
            return self.parse_tools_plan(raw_text, request=request, previous_plan=previous_plan)
        return self.parse_tools_batch(raw_text, planner_step_id=planner_step_id)

    @staticmethod
    def _validate_batch(batch: ToolsManagerBatch) -> None:
        for idx, req in enumerate(batch.requests):
            if not str(req.question or "").strip():
                raise ValueError(f"request[{idx}] question must not be empty")

    def _invoke_model(self, *, system_prompt: str, human_prompt: str) -> str:
        _ = (system_prompt, human_prompt)
        raise RuntimeError("ToolsManagerOrchestrator no longer supports model invocation")

    def attach_decision_provider(self, provider: ToolsManagerDecisionProvider) -> None:
        self._decision_provider = provider

    def _decision_provider_or_raise(self) -> ToolsManagerDecisionProvider:
        provider = getattr(self, "_decision_provider", None)
        if provider is None:
            return _InternalDecisionProvider(self)
        return provider

    def _plan(
        self,
        *,
        request: str,
        flow_context: str | None,
        pass_index: int = 0,
        pass_cap: int = 4,
        previous_plan: ToolsPlan | None = None,
        pass_logs: Sequence[dict[str, Any]] = (),
        warnings: Sequence[str] = (),
        changed_files: Sequence[str] = (),
        latest_answer: str = "",
    ) -> tuple[ToolsPlan, list[str]]:
        plan, issues, _source = self._plan_with_source(
            request=request,
            flow_context=flow_context,
            pass_index=pass_index,
            pass_cap=pass_cap,
            previous_plan=previous_plan,
            pass_logs=pass_logs,
            warnings=warnings,
            changed_files=changed_files,
            latest_answer=latest_answer,
        )
        return plan, issues

    def _build_batch(
        self,
        *,
        request: str,
        flow_context: str | None,
        plan: ToolsPlan,
        pass_index: int,
        pass_cap: int,
        pass_logs: Sequence[dict[str, Any]],
        warnings: Sequence[str],
        changed_files: Sequence[str],
        latest_answer: str = "",
    ) -> tuple[ToolsManagerBatch | None, list[str]]:
        provider = self._decision_provider_or_raise()
        return provider.build_batch(
            request=request,
            flow_context=flow_context,
            plan=plan,
            pass_index=pass_index,
            pass_cap=pass_cap,
            pass_logs=pass_logs,
            warnings=warnings,
            changed_files=changed_files,
            latest_answer=latest_answer,
        )

    @staticmethod
    def _merge_policy(base_policy: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(base_policy)
        if isinstance(override, dict):
            for key, value in override.items():
                merged[key] = value
        return merged

    @staticmethod
    def _repair_tool_policy_for_action(policy: dict[str, Any], tool_name: str) -> tuple[dict[str, Any], str]:
        tool = str(tool_name or "").strip().lower()
        if not tool:
            return dict(policy), ""
        required_by_tool = {
            "read_file": {"read_file"},
            "repo_search": {"repo_search"},
            "semantic_search": {"semantic_search"},
            "list_files": {"list_files", "repo_search"},
            "apply_patch": {"apply_patch"},
            "write_file": {"write_file"},
            "create_file": {"create_file"},
        }
        required = required_by_tool.get(tool)
        if not required:
            return dict(policy), ""
        repaired = dict(policy)
        raw_allowed = repaired.get("allowed_tools")
        if not isinstance(raw_allowed, list):
            repaired["allowed_tools"] = sorted(required)
            return repaired, f"tool_policy_repaired_for_action:{tool}:set_allowed_tools"
        allowed = [str(item).strip() for item in raw_allowed if str(item).strip()]
        missing = sorted(required.difference(allowed))
        if not missing:
            return repaired, ""
        repaired["allowed_tools"] = [*allowed, *missing]
        return repaired, f"tool_policy_repaired_for_action:{tool}:added={','.join(missing)}"

    @staticmethod
    def _clip_timeout(value: int | None, *, session_timeout: int) -> int:
        base = int(value or session_timeout)
        return max(5, min(base, max(5, int(session_timeout))))

    @staticmethod
    def _fingerprint_request(question: str, policy: dict[str, Any], timeout_seconds: int) -> str:
        normalized_question = re.sub(r"\s+", " ", str(question or "").strip()).lower()
        raw = json.dumps(
            {
                "question": normalized_question,
                "policy": policy,
                "timeout_seconds": timeout_seconds,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _extract_tool_path(tool_name: str, tool_args: dict[str, Any], question: str) -> str:
        args = tool_args if isinstance(tool_args, dict) else {}
        for key in ("path", "file", "file_path", "target_file"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().replace("\\", "/")
        lowered_tool = str(tool_name or "").strip().lower()
        if lowered_tool in {"read_file", "apply_patch", "write_file"}:
            match = re.search(r"\b(?:read_file|write_file|apply_patch)\s+([^\s]+)", str(question or ""))
            if match:
                return str(match.group(1) or "").strip().replace("\\", "/")
        return ""

    @classmethod
    def _request_tool_name(cls, request: ToolsManagerRequest) -> str:
        explicit = str(request.tool_name or "").strip().lower()
        if explicit:
            return explicit
        return cls._canonical_tool_name_from_question(str(request.question or "")).strip().lower()

    @classmethod
    def _batch_has_tool_family(cls, batch: ToolsManagerBatch, tools: set[str]) -> bool:
        return any(cls._request_tool_name(item) in tools for item in batch.requests)

    @classmethod
    def _batch_has_mutation_payload(cls, batch: ToolsManagerBatch, pending_edits: Sequence[Any]) -> bool:
        if any(pending_edits):
            return True
        return cls._batch_has_tool_family(batch, RunStateStore.mutation_tools)

    @staticmethod
    def _mark_structured_failure(
        run_store: RunStateStore,
        *,
        code: str,
        gate: str,
        required_next_tool: str,
        why_not_executed: str,
    ) -> None:
        state = run_store.read_json("state.json", run_store._default_state(goal="", flow_id=""))
        state["status"] = "failed_no_progress"
        state["blocking_reason"] = code
        state["failure_stage"] = run_store._public_phase_name(run_store._phase_from_gate(gate))
        state["failure_gate"] = gate
        state["required_next_tool"] = required_next_tool
        state["why_not_executed"] = why_not_executed
        state["last_error"] = f"{code}: {why_not_executed}"
        state["next_action"] = f"{code}: {required_next_tool}"
        state["next_exact_action"] = state["next_action"]
        run_store.write_json("state.json", state)

    @staticmethod
    def _successful_tool_status(
        *,
        tool_name: str,
        evidence_counts: dict[str, int],
        response_paths: Sequence[str],
        response: ToolRunResponse,
        changed_files: Sequence[str] = (),
    ) -> tuple[bool, str]:
        tool = str(tool_name or "").strip().lower()
        discovered = int(evidence_counts.get("discovered", 0) or 0)
        read = int(evidence_counts.get("read", 0) or 0)
        has_trace = bool(response.trace)
        has_answer = bool(str(response.answer or "").strip())
        if tool == "read_file":
            if read >= 1:
                return True, ""
            return False, "read_file_no_files_read"
        if tool in {"repo_search", "semantic_search", "list_files"}:
            if discovered >= 1 or response_paths:
                return True, ""
            if str(response.answer or "").strip() and any(
                str(row.get("result", "") or row.get("status", "") or "").strip()
                for row in response.trace
                if isinstance(row, dict)
            ):
                return True, ""
            return False, "search_no_candidates_or_evidence"
        if tool in {"apply_patch", "write_file", "create_file"}:
            if changed_files:
                return True, ""
            for row in response.trace:
                if not isinstance(row, dict):
                    continue
                status = str(row.get("status", "") or "").strip().lower()
                preview = str(row.get("output_preview", "") or row.get("error", "") or "").strip().lower()
                if status in {"error", "timeout", "failed"} or '"ok": false' in preview or "'ok': false" in preview:
                    return False, "mutation_no_modified_files"
            if response.trace or str(response.answer or "").strip():
                return True, ""
            return False, "mutation_no_modified_files"
        if tool in {"run_command", "verify_project"}:
            return (has_trace or has_answer), "" if (has_trace or has_answer) else "verify_result_missing"
        return (has_trace or has_answer), "" if (has_trace or has_answer) else "tool_result_missing"

    @staticmethod
    def _force_refresh_requested(tool_args: dict[str, Any], question: str) -> bool:
        args = tool_args if isinstance(tool_args, dict) else {}
        value = args.get("force_refresh")
        if isinstance(value, bool):
            return value
        return "force_refresh=true" in str(question or "").replace(" ", "").lower()

    @staticmethod
    def _pass_progress_counts(pass_log: dict[str, Any]) -> tuple[int, int, int, int]:
        if not isinstance(pass_log, dict):
            return (0, 0, 0, 0)
        return (
            int(pass_log.get("new_files_read", 0) or 0),
            int(pass_log.get("new_findings", 0) or 0),
            int(pass_log.get("successful_patches", 0) or 0),
            int(pass_log.get("verification_commands", 0) or 0),
        )

    @staticmethod
    def _enrich_trace_rows(
        rows: Sequence[dict[str, Any]],
        *,
        normalized_key: str,
        purpose: str,
        phase: str,
        duplicate_of: str = "",
        files_found: Sequence[str] = (),
        files_read: Sequence[str] = (),
        next_action: str = "",
        ledger_checkpoint_path: str = "",
    ) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item.setdefault("normalized_key", normalized_key)
            item.setdefault("purpose", purpose)
            item.setdefault("phase", phase)
            item.setdefault("is_duplicate", bool(duplicate_of))
            item.setdefault("duplicate_of", duplicate_of)
            item.setdefault("produced_new_evidence", bool(files_found or files_read))
            item.setdefault("files_found", list(files_found))
            item.setdefault("files_read", list(files_read))
            item.setdefault("next_action", next_action)
            item.setdefault("ledger_checkpoint_path", ledger_checkpoint_path)
            enriched.append(item)
        return enriched

    @staticmethod
    def _normalize_fingerprint_key(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip()).lower()

    @classmethod
    def _semantic_request_fingerprint(cls, question: str) -> str | None:
        normalized = cls._normalize_fingerprint_key(question)
        if not normalized:
            return None
        if not any(token in normalized for token in ("semantic_search", "search", "find", "locate", "grep")):
            return None
        semantic_key = cls._normalize_semantic_key(normalized)
        return hashlib.sha1(semantic_key.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def _normalize_semantic_key(cls, text: str) -> str:
        normalized = cls._normalize_fingerprint_key(text)
        if not normalized:
            return ""
        query_match = re.search(r"query\s*[:=]\s*['\"]?([^'\"\n]+)['\"]?", normalized)
        k_match = re.search(r"\bk\s*[:=]\s*(\d+)", normalized)
        query = cls._normalize_fingerprint_key(query_match.group(1)) if query_match else ""
        if query:
            query = re.sub(r"\bk\s*[:=]\s*\d+.*$", "", query).strip()
        k_val = str(k_match.group(1)).strip() if k_match else ""
        if query or k_val:
            return f"query={query}|k={k_val or '0'}"
        return normalized

    @classmethod
    def _semantic_trace_fingerprints(cls, rows: Sequence[dict[str, Any]]) -> set[str]:
        out: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("tool_name", "")).strip().lower() != "semantic_search":
                continue
            args_summary = cls._normalize_semantic_key(str(row.get("args_summary", "") or ""))
            if not args_summary:
                continue
            out.add(hashlib.sha1(args_summary.encode("utf-8")).hexdigest()[:12])
        return out

    @staticmethod
    def _looks_like_search_request_text(question: str) -> bool:
        lowered = str(question or "").strip().lower()
        if not lowered:
            return False
        patterns = (
            "semantic_search",
            "search",
            "find",
            "locate",
            "grep",
            "inspect repository",
        )
        return any(token in lowered for token in patterns)

    @staticmethod
    def _looks_like_edit_request_text(question: str) -> bool:
        lowered = str(question or "").strip().lower()
        if not lowered:
            return False
        patterns = (
            "apply_patch",
            "write_file",
            "edit",
            "modify",
            "mutation",
            "change file",
            "update file",
            "create",
            "generate",
            "add file",
            "write a",
            "write the",
        )
        return any(token in lowered for token in patterns)

    def _is_edit_task(self, plan: "ToolsPlan", request: str) -> bool:
        """A task is an edit task if the plan has an edit step or the request reads as one."""
        if any(step.tool_intent == "edit" for step in getattr(plan, "steps", []) or []):
            return True
        return self._looks_like_edit_request_text(request)

    @staticmethod
    def _looks_like_model_docs_request(*values: str) -> bool:
        combined = " ".join(str(value or "") for value in values).lower()
        if "docs/models.md" in combined:
            return True
        return bool(
            "model" in combined
            and any(token in combined for token in ("doc", "document", "documentation", "update"))
        )

    @staticmethod
    def _extract_model_class_names_from_text(text: str) -> set[str]:
        names: set[str] = set()
        for match in re.finditer(
            r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*(?:models\.Model|AbstractUser|AbstractBaseUser)[^)]*\)",
            str(text or ""),
        ):
            names.add(str(match.group(1)))
        return names

    @classmethod
    def _extract_model_class_names_from_response(cls, response: ToolRunResponse) -> set[str]:
        names: set[str] = set()
        payloads = [response.answer]
        payloads.extend(str(row.get("output_preview", "") or "") for row in response.trace if isinstance(row, dict))
        payloads.extend(json.dumps(row, ensure_ascii=False) for row in response.sources if isinstance(row, dict))
        for payload in payloads:
            names.update(cls._extract_model_class_names_from_text(str(payload or "")))
        return names

    def _model_docs_evidence_complete(self, run_store: RunStateStore) -> bool:
        state = run_store.read_json("state.json", {})
        goal = str(state.get("goal", state.get("original_user_task", "")) or "")
        profile = run_store.active_goal_profile(goal)
        if profile is None or profile.id != "model_docs":
            return True
        read_files = run_store.read_files()
        has_model_inventory = any(
            path.endswith("/models.py") or path == "models.py"
            for path in read_files
        )
        docs_read = "docs/models.md" in read_files or not (run_store.repo_root / "docs" / "models.md").exists()
        tool_rows = run_store.read_jsonl("tool_calls.jsonl")
        planned_sections = any(
            str(row.get("tool_name", "")).strip().lower() in {"run_command", "find_symbols", "repo_search"}
            and (
                "class " in str(row.get("result_summary", ""))
                or "model" in json.dumps(row.get("normalized_args", {}), ensure_ascii=False).lower()
            )
            for row in tool_rows
            if row.get("status") == "ok"
        )
        return bool(has_model_inventory and docs_read and (planned_sections or len(read_files) >= 2))

    @staticmethod
    def _response_read_failed(response: ToolRunResponse) -> str:
        for row in response.trace:
            if not isinstance(row, dict):
                continue
            if str(row.get("tool_name", "")).strip().lower() != "read_file":
                continue
            status = str(row.get("status", "") or "").strip().lower()
            preview = str(row.get("output_preview", "") or row.get("error", "") or "").strip()
            if status in {"error", "failed", "timeout"}:
                return preview or status
        return ""

    @staticmethod
    def _verified_noop_reason(latest_answer: str, warnings: Sequence[str]) -> str:
        combined = " ".join([str(latest_answer or ""), *(str(item or "") for item in warnings)]).lower()
        if any(token in combined for token in ("verified no-op", "verified noop", "no changes needed", "already up to date")):
            return "verified_noop"
        return ""

    def _completion_blocker(
        self,
        *,
        run_store: RunStateStore,
        plan: "ToolsPlan",
        final_state: dict[str, Any],
        changed_files: Sequence[str],
        is_edit_task: bool,
        latest_answer: str,
        warnings: Sequence[str],
        terminal_reason: str,
    ) -> str:
        current_phase = str(final_state.get("current_phase", "") or "")
        verification_status = str(final_state.get("verification_status", "pending") or "pending")
        completed = {
            str(item)
            for item in (final_state.get("completed_gates", []) if isinstance(final_state.get("completed_gates"), list) else [])
            if str(item).strip()
        }
        if str(final_state.get("current_gate", "") or "") == "final_report" and current_phase == "FINAL":
            completed.add("final_report")
        pending = [
            item
            for item in run_store.gates
            if item not in completed
        ]
        pending.extend(
            str(item)
            for item in (final_state.get("pending_gates", []) if isinstance(final_state.get("pending_gates"), list) else [])
            if str(item).strip()
        )
        pending = list(dict.fromkeys(pending))
        final_report_verified = "final_report" in completed and self._verified_noop_reason(latest_answer, warnings)
        if is_edit_task and not run_store._has_applied_changes(changed_files):
            return "no changed files; docs/models.md not created/updated"
        if pending and not final_report_verified:
            return f"pending required gates: {', '.join(pending)}"
        if current_phase != "FINAL":
            return f"current_phase is {current_phase or 'unknown'}, not FINAL"
        if verification_status != "passed":
            return f"verification_status is {verification_status}, not passed"
        if is_edit_task:
            created_target_exists = False
            goal = str(final_state.get("goal", final_state.get("original_user_task", "")) or "")
            for match in re.findall(r"(?:^|\s)([\w./-]+\.(?:md|py|txt|toml|yaml|yml|json))", goal):
                normalized = run_store._normalize_candidate_path(match)
                if normalized and (run_store.repo_root / normalized).exists():
                    created_target_exists = True
                    break
            if not changed_files and not created_target_exists and not self._verified_noop_reason(latest_answer, warnings):
                return "edit task produced no modified files, target file, or verified no-op reason"
        if terminal_reason == "stalled_no_actionable_requests":
            return "no progress after configured pass limit"
        return ""

    @staticmethod
    def _has_pending_plan_work(plan: "ToolsPlan") -> bool:
        if str(getattr(plan, "decision", "") or "").strip().lower() in {"continue", "revise"}:
            return True
        for step in getattr(plan, "steps", []) or []:
            status = str(getattr(step, "status", "") or "").strip().lower()
            if status in {"pending", "in_progress"}:
                return True
        return False

    def _compute_effective_pass_cap(
        self,
        *,
        configured_pass_cap: int,
        plan: "ToolsPlan",
        request: str,
        max_allowed_passes: int = 12,
    ) -> int:
        """Ensure edit tasks reserve enough passes to reach edit + verify.

        Inspect/search steps are compressed into a single pass (they can run as
        one batch), then at least one edit and one verify pass are reserved so
        an 8-step checklist with a low configured cap still reaches the
        create/write/verify stages.
        """
        configured = max(1, min(int(configured_pass_cap), max_allowed_passes))
        if not self._is_edit_task(plan, request):
            return configured

        intents = [step.tool_intent for step in getattr(plan, "steps", []) or []]
        edit_steps = sum(1 for intent in intents if intent == "edit")
        verify_steps = sum(1 for intent in intents if intent == "verify")
        inspect_like = sum(1 for intent in intents if intent in ("inspect", "search"))

        inspect_passes = 1 if inspect_like else 0
        # Always reserve >=1 edit and >=1 verify pass for edit tasks.
        minimum_passes = inspect_passes + max(1, edit_steps) + max(1, verify_steps)
        return max(configured, min(max_allowed_passes, minimum_passes + 2))

    @staticmethod
    def _looks_like_apply_patch_failure_trace(row: dict[str, Any]) -> bool:
        if str(row.get("tool_name", "")).strip().lower() != "apply_patch":
            return False
        status = str(row.get("status", "")).strip().lower()
        preview = str(row.get("output_preview", "")).strip().lower()
        if status in {"error", "timeout", "failed"}:
            return True
        if status == "ok" and not preview:
            return True
        if '"ok": false' in preview or "'ok': false" in preview or '"error"' in preview:
            return True
        return False

    @staticmethod
    def _build_failure_signature(*, question: str, code: str, detail: str) -> str:
        normalized = {
            "question": re.sub(r"\s+", " ", str(question or "").strip()).lower()[:320],
            "code": str(code or "").strip().lower(),
            "detail": re.sub(r"\s+", " ", str(detail or "").strip()).lower()[:320],
        }
        raw = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _mutation_retry_request(
        self,
        *,
        request: str,
        flow_context: str | None,
        plan: ToolsPlan,
        step: ToolsPlanStep | None,
        pass_index: int,
    ) -> ToolsManagerRequest:
        base = self._deterministic_fallback_request(
            request=request,
            flow_context=flow_context,
            plan=plan,
            step=step,
            pass_index=pass_index,
        )
        lines = [
            str(base.question if base is not None else "").strip(),
            "Mutation retry lock is active because prior apply_patch attempt failed or no-oped.",
            "Do not start new broad semantic search.",
            "Execute a direct mutation fallback now: use write_file with full content if apply_patch fails.",
            "Verify changed_files evidence before any terminal response.",
        ]
        return ToolsManagerRequest(question="\n".join(line for line in lines if line.strip()))

    def _forced_mutation_tool_request(self, *, todo_item: TodoLedgerItem) -> ToolsManagerRequest:
        target = todo_item.target_files[0] if todo_item.target_files else "docs/models.md"
        question = (
            "You must call exactly one required mutation tool now.\n"
            "Do not answer in prose.\n"
            "Do not perform discovery.\n"
            f"Call apply_patch, write_file, or create_file for the target file: {target}."
        )
        return ToolsManagerRequest(
            question=question,
            tool_name="apply_patch",
            tool_args={"path": target},
            strategy_hint="forced_required_mutation_tool",
        )

    def _git_status_paths(self) -> set[str]:
        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                return set()
            paths: set[str] = set()
            for line in proc.stdout.splitlines():
                if len(line) >= 4:
                    path = line[3:].strip()
                    if path:
                        paths.add(path.replace("\\", "/"))
            return paths
        except Exception:
            return set()

    def _resolve_step(self, plan: ToolsPlan) -> ToolsPlanStep | None:
        for step in plan.steps:
            if step.id == plan.current_step_id:
                return step
        return plan.steps[0] if plan.steps else None

    @staticmethod
    def _planner_decision_row(plan: ToolsPlan, pass_index: int) -> dict[str, Any]:
        step = next((item for item in plan.steps if item.id == plan.current_step_id), None)
        return {
            "pass_index": int(pass_index),
            "current_step_id": str(plan.current_step_id or ""),
            "current_step_title": str(getattr(step, "title", "") or ""),
            "decision": str(plan.decision or "continue"),
            "decision_reason": str(plan.decision_reason or ""),
        }

    @staticmethod
    def _planner_task_fingerprint(step: ToolsPlanStep | None) -> str:
        if step is None:
            return ""
        raw = json.dumps(
            {
                "id": str(step.id or "").strip().lower(),
                "title": re.sub(r"\s+", " ", str(step.title or "").strip()).lower(),
                "tool_intent": str(step.tool_intent or "").strip().lower(),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def _recent_step_repeat_count(
        cls,
        pass_logs: Sequence[dict[str, Any]],
        *,
        step_id: str,
    ) -> int:
        target = str(step_id or "").strip()
        if not target:
            return 0
        count = 0
        for row in reversed(list(pass_logs)):
            if not isinstance(row, dict):
                continue
            if str(row.get("planner_step_id", "")).strip() != target:
                break
            if int(row.get("requests_count", 0) or 0) <= 0 and int(row.get("tool_steps", 0) or 0) <= 0:
                break
            count += 1
        return count

    @staticmethod
    def _advance_plan_to_next_unfinished_step(plan: ToolsPlan) -> ToolsPlan | None:
        current_id = str(plan.current_step_id or "").strip()
        if not current_id:
            return None
        candidate: ToolsPlanStep | None = None
        seen_current = False
        for item in plan.steps:
            if str(item.id or "").strip() == current_id:
                seen_current = True
                continue
            if str(item.status or "").strip().lower() in {"done", "blocked"}:
                continue
            if seen_current:
                candidate = item
                break
            if candidate is None:
                candidate = item
        if candidate is None:
            return None
        updated_steps: list[ToolsPlanStep] = []
        for item in plan.steps:
            if str(item.id or "").strip() == current_id and str(item.status or "").strip().lower() == "in_progress":
                updated_steps.append(item.model_copy(update={"status": "done"}))
            elif str(item.id or "").strip() == str(candidate.id or "").strip():
                updated_steps.append(item.model_copy(update={"status": "in_progress"}))
            else:
                updated_steps.append(item)
        return plan.model_copy(
            update={
                "steps": updated_steps,
                "current_step_id": str(candidate.id or "").strip(),
                "decision": "continue",
                "decision_reason": (
                    f"Auto-advanced from duplicate task {current_id} to next unresolved step {str(candidate.id or '').strip()}"
                ),
            }
        )

    @staticmethod
    def _truncate_line(value: str, *, limit: int = 220) -> str:
        text = " ".join(str(value or "").strip().split())
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"

    def _deterministic_fallback_request(
        self,
        *,
        request: str,
        flow_context: str | None,
        plan: ToolsPlan,
        step: ToolsPlanStep | None,
        pass_index: int,
    ) -> ToolsManagerRequest | None:
        if step is None:
            return None

        step_title = self._truncate_line(step.title or "current step")
        step_hint = self._truncate_line(step.args_hint or step.fallback or step.success_signal or "")
        objective = self._truncate_line(plan.objective or request or "Execute requested plan")
        flow = self._truncate_line(flow_context or "", limit=320)
        user_request = self._truncate_line(request or "", limit=320)
        model_docs_request = self._looks_like_model_docs_request(
            request,
            plan.objective,
            step.title,
            step.args_hint,
            step.fallback,
        )

        if step.tool_intent == "inspect":
            if model_docs_request:
                directive = (
                    "Use repository-local evidence only. First read docs/models.md if it exists, then use run_command "
                    "or repo_search to enumerate model definitions with commands such as "
                    "`rg -n \"class .*\\(models\\.(Model|AbstractUser)\" back src .` and "
                    "`find . -path '*/models.py' -o -name '*models*.py'`. "
                    "Do not ask the user to paste files that can be read from the repository."
                )
            else:
                directive = (
                    "Inspect repository files for this step using repo_search, semantic_search, read_file, "
                    "find_symbols, call_graph, or run_command as appropriate. "
                    "Gather concrete evidence with file paths and line ranges."
                )
        elif step.tool_intent == "search":
            if model_docs_request:
                directive = (
                    "Run deterministic repository searches for model definitions and docs coverage. "
                    "Prefer concrete commands/searches for `models.py`, `models.Model`, `AbstractUser`, "
                    "`db_table`, and `docs/models.md`; report file paths and line ranges. "
                    "Do not request pasted file contents from the user."
                )
            else:
                directive = (
                    "Run targeted repository search for the requested behavior and gather concrete file evidence "
                    "before proposing edits."
                )
        elif step.tool_intent == "edit":
            if model_docs_request:
                directive = (
                    "Update docs/models.md from repository-local evidence gathered in prior/current steps. "
                    "If evidence is incomplete, run targeted read_file/repo_search/run_command calls now; "
                    "do not ask the user to paste repository files. Prefer apply_patch; if patching no-ops, "
                    "use write_file with the full updated document content."
                )
            else:
                directive = (
                    "Apply concrete repository edits for this step. Prefer apply_patch first; if patch chain fails or no-ops, "
                    "force write_file full-content fallback, then verify changed_files evidence before terminal response. "
                    "Do not emit conversational terminal text for unresolved edit-intent work."
                )
        elif step.tool_intent == "verify":
            directive = (
                "Verify relevant changes with targeted checks (tests/lint/type checks or focused run_command checks), "
                "then summarize verification evidence."
            )
        else:
            directive = (
                "Summarize current status with concrete repository evidence and identify the next actionable step."
            )

        lines = [
            f"Deterministic fallback request for planner pass {int(pass_index)}.",
            f"Objective: {objective}",
            f"Planner step: {step_title}",
            f"Intent: {step.tool_intent}",
            f"Original request: {user_request or '-'}",
        ]
        if step_hint:
            lines.append(f"Step hint: {step_hint}")
        if flow:
            lines.append(f"Flow context: {flow}")
        lines.append(f"Action: {directive}")
        return ToolsManagerRequest(question="\n".join(lines))

    @staticmethod
    def _looks_like_conversational_terminal(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        patterns = (
            "if you want",
            "reply \"yes",
            "reply 'yes",
            "let me know if you want",
            "if you want, i can",
            "if you want i can",
            "say yes",
            "type yes",
        )
        return any(token in lowered for token in patterns)

    @classmethod
    def _looks_like_hard_blocker_prompt(cls, text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        patterns = (
            "missing credential",
            "missing credentials",
            "missing api key",
            "missing token",
            "missing secret",
            "permission denied",
            "insufficient permission",
            "unauthorized",
            "forbidden",
            "access denied",
            "missing target identifier",
            "target identifier required",
            "missing identifier",
            "identifier is required",
            "missing file path",
            "path is required",
            "target path required",
            "provide file path",
            "unavailable",
        )
        return any(token in lowered for token in patterns)

    @classmethod
    def _looks_like_non_hard_blocker_prompt(cls, text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        if cls._looks_like_hard_blocker_prompt(lowered):
            return False
        patterns = (
            "blocker decision",
            "need one blocker decision",
            "scope choice",
            "scope decision",
            "need scope",
            "which scope",
            "choose scope",
            "choose option",
            "which option",
            "option 1",
            "option 2",
            "1 or 2",
            "one or two",
            "pick one",
            "awaiting scope decision",
            "awaiting your decision",
            "i'm blocked on",
            "i am blocked on",
            "blocked on making",
            "need to read",
            "need to inspect",
            "need to review",
            "before patching",
            "before editing",
            "before making changes",
            "share permission to proceed",
            "permission to proceed",
            "requires explicit tool execution",
            "tool access is available",
            "once tool access is available",
        )
        return any(token in lowered for token in patterns)

    @classmethod
    def _is_blocked_terminal_plan(cls, plan: ToolsPlan) -> bool:
        fields = [str(plan.decision_reason or ""), str(plan.finalize_action or "")]
        combined = " ".join(fields)
        if str(plan.decision or "").strip().lower() == "stop":
            return cls._looks_like_hard_blocker_prompt(combined)
        return cls._looks_like_hard_blocker_prompt(combined)

    @staticmethod
    def _synthesize_terminal_answer(
        *,
        terminal_reason: str,
        pass_logs: Sequence[dict[str, Any]],
        planner_decisions: Sequence[dict[str, Any]],
        toolsmanager_requests_count: int,
    ) -> str:
        reason = str(terminal_reason or "unknown").strip() or "unknown"
        passes = len(pass_logs)
        last_pass = pass_logs[-1] if pass_logs else {}
        if not isinstance(last_pass, dict):
            last_pass = {}
        last_step = str(last_pass.get("planner_step_title", "") or "").strip()
        decision_reason = ""
        if planner_decisions:
            tail = planner_decisions[-1]
            if isinstance(tail, dict):
                decision_reason = str(tail.get("decision_reason", "") or "").strip()
        if not decision_reason:
            decision_reason = str(last_pass.get("planner_decision_reason", "") or "").strip()

        lines = [
            "Auto-execute ended without a direct answer from tool runs.",
            f"terminal_reason={reason}",
            f"passes={passes}",
            f"toolsmanager_requests={int(toolsmanager_requests_count)}",
        ]
        if last_step:
            lines.append(f"last_step={last_step}")
        if decision_reason:
            lines.append(f"planner_reason={decision_reason}")
        return "\n".join(lines)

    @staticmethod
    def _synthesize_resumable_pass_cap_answer(
        *,
        pass_logs: Sequence[dict[str, Any]],
        planner_decisions: Sequence[dict[str, Any]],
        toolsmanager_requests_count: int,
    ) -> str:
        passes = len(pass_logs)
        last_pass = pass_logs[-1] if pass_logs else {}
        if not isinstance(last_pass, dict):
            last_pass = {}
        last_step = str(last_pass.get("planner_step_title", "") or "").strip()
        decision_reason = ""
        if planner_decisions:
            tail = planner_decisions[-1]
            if isinstance(tail, dict):
                decision_reason = str(tail.get("decision_reason", "") or "").strip()
        if not decision_reason:
            decision_reason = str(last_pass.get("planner_decision_reason", "") or "").strip()

        lines = [
            "Auto-execute reached the pass cap with pending work and should continue.",
            f"passes={passes}",
            f"toolsmanager_requests={int(toolsmanager_requests_count)}",
        ]
        if last_step:
            lines.append(f"last_step={last_step}")
        if decision_reason:
            lines.append(f"planner_reason={decision_reason}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_prechecklist(plan: ToolsPlan, *, source: str) -> dict[str, Any]:
        steps: list[dict[str, str]] = []
        for item in plan.steps[:20]:
            steps.append(
                {
                    "id": str(item.id or "").strip() or "step",
                    "title": str(item.title or "").strip() or "step",
                    "status": str(item.status or "pending"),
                }
            )
        return {
            "objective": str(plan.objective or "").strip(),
            "steps": steps,
            "source": str(source or ""),
        }

    def preview_plan(
        self,
        *,
        request: str,
        flow_context: str | None,
        pass_cap: int,
    ) -> dict[str, Any]:
        plan, warnings, source = self._plan_with_source(
            request=request,
            flow_context=flow_context,
            pass_index=0,
            pass_cap=pass_cap,
            previous_plan=None,
            pass_logs=[],
            warnings=[],
            changed_files=[],
            latest_answer="",
        )
        warning_text = ""
        if source == "deterministic_fallback":
            warning_text = "Planner parse failed; using deterministic fallback checklist."
        return {
            "prechecklist": self._normalize_prechecklist(plan, source=source),
            "prechecklist_source": source,
            "prechecklist_warning": warning_text,
            "warnings": warnings,
        }

    def _plan_with_source(
        self,
        *,
        request: str,
        flow_context: str | None,
        pass_index: int = 0,
        pass_cap: int = 4,
        previous_plan: ToolsPlan | None = None,
        pass_logs: Sequence[dict[str, Any]] = (),
        warnings: Sequence[str] = (),
        changed_files: Sequence[str] = (),
        latest_answer: str = "",
    ) -> tuple[ToolsPlan, list[str], str]:
        provider = self._decision_provider_or_raise()
        return provider.plan_with_source(
            request=request,
            flow_context=flow_context,
            pass_index=pass_index,
            pass_cap=pass_cap,
            previous_plan=previous_plan,
            pass_logs=pass_logs,
            warnings=warnings,
            changed_files=changed_files,
            latest_answer=latest_answer,
        )

    def run(
        self,
        *,
        request: str,
        flow_context: str | None,
        index_dir: str | Path | None,
        index_dirs: Sequence[str | Path] | None,
        k: int,
        max_steps: int,
        timeout_seconds: int,
        tool_policy: dict[str, Any],
        pass_cap: int,
        on_event: Callable[[Any], None] | None = None,
        flow_id: str | None = None,
        run_id: str | None = None,
        max_no_progress_passes: int = 2,
    ) -> AutoExecuteResult:
        pass_cap = max(1, min(int(pass_cap), 12))
        max_no_progress_passes = max(1, int(max_no_progress_passes or 2))
        all_warnings: list[str] = []
        all_trace: list[dict[str, Any]] = []
        all_sources: list[dict[str, Any]] = []
        all_pass_logs: list[dict[str, Any]] = []
        planner_decisions: list[dict[str, Any]] = []
        terminal_reason = "pass_cap_reached"
        toolsmanager_requests_count = 0
        latest_answer = ""
        run_store = RunStateStore(repo_root=self.repo_root, run_id=run_id)
        run_store.ensure(goal=request, flow_id=str(flow_id or ""))
        run_store.seed_candidate_queue()
        run_store.ensure_todo_ledger(goal=request)
        # Authoritative resume reconciliation: derive current_gate/current_phase
        # from completed + pending gates (and pending files) before any worker is
        # launched, so a stale current_gate can never strand the run on a
        # completed discovery gate.
        run_store.reconcile_gate_pointer()
        active_profile = run_store.active_goal_profile(request)
        model_docs_task = active_profile is not None and active_profile.id == "model_docs"
        initial_state = run_store.read_json("state.json", {})
        stalled_passes = int(initial_state.get("no_progress_count", 0) or 0)
        execution_run_id = run_store.run_id
        execution_started = time.perf_counter()
        execution_requests_ok = 0
        execution_requests_failed = 0
        duplicate_request_skips = 0
        duplicate_semantic_search_skips = 0
        duplicate_tool_execution_blocks = 0
        request_retry_attempts = 0
        request_retry_exhausted = 0
        edit_retry_mode_activations = 0
        edit_retry_mode_pending = False
        persisted_fingerprint_counts: dict[str, int] = {}
        recent_failure_summaries: list[str] = []
        seen_planner_task_fingerprints: set[str] = set()
        seen_model_classes: set[str] = set()
        execution_backend = str(getattr(getattr(self, "execution_config", None), "backend", "local") or "local")
        memory_service = getattr(self, "coding_memory_service", None)
        flow_key = str(flow_id or "").strip()

        seen_request_fingerprints: set[str] = set()
        seen_semantic_search_fingerprints: set[str] = set()
        executed_tools_this_turn: set[str] = set()
        seen_failure_signatures: set[str] = set()

        if memory_service is not None and flow_key:
            try:
                seen_request_fingerprints = set(
                    memory_service.get_tool_fingerprints(
                        flow_id=flow_key,
                        kind="request_fingerprint",
                    )
                )
                seen_semantic_search_fingerprints = set(
                    memory_service.get_tool_fingerprints(
                        flow_id=flow_key,
                        kind="semantic_search_fingerprint",
                    )
                )
                seen_failure_signatures = set(
                    memory_service.get_tool_fingerprints(
                        flow_id=flow_key,
                        kind="mutation_failure_signature",
                    )
                )
                seen_planner_task_fingerprints = set(
                    memory_service.get_tool_fingerprints(
                        flow_id=flow_key,
                        kind="planner_task_fingerprint",
                    )
                )
            except Exception as exc:
                all_warnings.append(f"toolsmanager persistent fingerprint load failed: {exc}")
            persisted_fingerprint_counts = {
                "request_fingerprint": len(seen_request_fingerprints),
                "semantic_search_fingerprint": len(seen_semantic_search_fingerprints),
                "mutation_failure_signature": len(seen_failure_signatures),
                "planner_task_fingerprint": len(seen_planner_task_fingerprints),
            }

        plan, plan_warnings, _source = self._plan_with_source(
            request=request,
            flow_context=flow_context,
            pass_index=0,
            pass_cap=pass_cap,
            previous_plan=None,
            pass_logs=[],
            warnings=[],
            changed_files=[],
            latest_answer="",
        )
        all_warnings.extend(plan_warnings)

        # Edit/create/verify workflows need enough passes to reach the write and
        # verify stages; a low configured cap would otherwise terminate early.
        is_edit_task = self._is_edit_task(plan, request)
        effective_pass_cap = self._compute_effective_pass_cap(
            configured_pass_cap=pass_cap,
            plan=plan,
            request=request,
        )
        if effective_pass_cap != pass_cap:
            all_warnings.append(
                f"pass_cap_raised_for_edit_task: {pass_cap} -> {effective_pass_cap}"
            )
            pass_cap = effective_pass_cap

        before = self._git_status_paths()
        changed_files: list[str] = []

        for pass_index in range(1, pass_cap + 1):
            step = self._resolve_step(plan)
            gate = run_store.required_gate_for_step(step)
            persisted_gate_before_pass = ""
            state_before_pass = run_store.read_json("state.json", {})
            pending_before_pass = (
                state_before_pass.get("pending_gates")
                if isinstance(state_before_pass.get("pending_gates"), list)
                else []
            )
            if pending_before_pass:
                persisted_gate_before_pass = str(pending_before_pass[0] or "").strip()
            run_store.update_state(
                plan=plan,
                step=step,
                status="running",
                next_action="execute current gate",
                changed_files=changed_files,
            )
            current_todo = run_store.current_todo_for_gate(gate, goal=request)
            if current_todo.status in {"pending", "failed"}:
                current_todo = current_todo.model_copy(update={"status": "in_progress"})
                run_store._write_todo_item(current_todo)
            step_repeat_count = self._recent_step_repeat_count(
                all_pass_logs,
                step_id=str(getattr(step, "id", "") or ""),
            )
            task_fingerprint = self._planner_task_fingerprint(step)
            if (
                step is not None
                and not changed_files
                and not edit_retry_mode_pending
                and (step_repeat_count >= 1 or (task_fingerprint and task_fingerprint in seen_planner_task_fingerprints))
            ):
                advanced_plan = self._advance_plan_to_next_unfinished_step(plan)
                if advanced_plan is not None:
                    all_warnings.append("planner_duplicate_task_advanced")
                    plan = advanced_plan
                    step = self._resolve_step(plan)
            planner_row = self._planner_decision_row(plan, pass_index)
            planner_decisions.append(planner_row)

            if plan.decision in {"finalize", "stop"}:
                decision_text = " ".join(
                    [
                        str(plan.decision_reason or "").strip(),
                        str(plan.finalize_action or "").strip(),
                    ]
                ).strip()
                has_hard_blocker = self._looks_like_hard_blocker_prompt(decision_text)
                has_non_hard_blocker = self._looks_like_non_hard_blocker_prompt(decision_text)
                has_conversational_terminal = (
                    self._looks_like_conversational_terminal(plan.finalize_action)
                    or self._looks_like_conversational_terminal(plan.decision_reason)
                )
                should_retry_terminal = bool(
                    (not changed_files)
                    and pass_index < pass_cap
                    and (
                        has_conversational_terminal
                        or has_non_hard_blocker
                        or (plan.decision == "stop" and not has_hard_blocker)
                    )
                )
                if should_retry_terminal:
                    if has_conversational_terminal:
                        all_warnings.append(
                            "planner_finalize_conversational_without_edits; forcing another execution pass"
                        )
                    if has_non_hard_blocker or (plan.decision == "stop" and not has_hard_blocker):
                        all_warnings.append(
                            "planner_terminal_nonhard_blocker_retry; forcing another execution pass"
                        )
                    plan = self._deterministic_fallback_plan(
                        request=(
                            f"{request}\n\n"
                            "Full-auto continuation directive:\n"
                            "- Do not ask for confirmation.\n"
                            "- Do not pause for non-hard scope/option choices.\n"
                            "- Continue with concrete file inspection/edits/verification.\n"
                            "- Return blocked only for true blockers."
                        ),
                        flow_context=flow_context,
                        previous_plan=plan,
                        reason="terminal_nonhard_blocker_retry",
                    )
                    continue

                if (
                    plan.decision == "finalize"
                    and not changed_files
                    and not self._is_blocked_terminal_plan(plan)
                    and (
                        self._looks_like_conversational_terminal(plan.finalize_action)
                        or self._looks_like_conversational_terminal(plan.decision_reason)
                    )
                    and pass_index < pass_cap
                ):
                    all_warnings.append(
                        "planner_finalize_conversational_without_edits; forcing another execution pass"
                    )
                    plan = self._deterministic_fallback_plan(
                        request=(
                            f"{request}\n\n"
                            "Full-auto continuation directive:\n"
                            "- Do not ask for confirmation.\n"
                            "- Continue with concrete file inspection/edits/verification.\n"
                            "- Return blocked only for true blockers."
                        ),
                        flow_context=flow_context,
                        previous_plan=plan,
                        reason="conversational_terminal_retry",
                    )
                    continue

                # Never let an edit task finalize as "success" without file-change
                # evidence; force another pass to actually write + verify.
                if (
                    plan.decision == "finalize"
                    and is_edit_task
                    and not changed_files
                    and not self._is_blocked_terminal_plan(plan)
                    and pass_index < pass_cap
                ):
                    all_warnings.append(
                        "planner_finalize_edit_without_changed_files; forcing edit/verify pass"
                    )
                    plan = self._deterministic_fallback_plan(
                        request=(
                            f"{request}\n\n"
                            "Full-auto continuation directive:\n"
                            "- This is a file create/edit task but no files have changed yet.\n"
                            "- Use write_file (full content) or apply_patch to make the change now.\n"
                            "- Then verify the file exists before any terminal response.\n"
                            "- Do not claim success without changed_files evidence."
                        ),
                        flow_context=flow_context,
                        previous_plan=plan,
                        reason="edit_finalize_without_changes",
                    )
                    edit_retry_mode_pending = True
                    continue
                if plan.decision == "stop":
                    if has_hard_blocker:
                        all_warnings.append("planner_terminal_hard_blocker_stop")
                    else:
                        all_warnings.append(
                            "planner_terminal_nonhard_blocker_retry_exhausted; pass cap reached before retry"
                        )
                        terminal_reason = "pass_cap_reached"
                        if not latest_answer:
                            latest_answer = str(plan.decision_reason or plan.finalize_action or "").strip()
                        all_pass_logs.append(
                            {
                                "pass_index": pass_index,
                                "planner_step_id": plan.current_step_id,
                                "planner_step_title": str(getattr(step, "title", "") or ""),
                                "planner_decision": "continue",
                                "planner_decision_reason": "non-hard blocker stop downgraded to pass_cap_reached",
                                "batch_reason": "planner_terminal_nonhard_exhausted",
                                "expected_progress": "",
                                "requests_count": 0,
                                "request_fingerprints": [],
                                "tool_steps": 0,
                                "warnings_delta": 0,
                            }
                        )
                        break
                terminal_reason = "planner_finalize" if plan.decision == "finalize" else "planner_stop"
                if not latest_answer:
                    latest_answer = str(plan.finalize_action or "").strip()
                all_pass_logs.append(
                    {
                        "pass_index": pass_index,
                        "planner_step_id": plan.current_step_id,
                        "planner_step_title": str(getattr(step, "title", "") or ""),
                        "planner_decision": plan.decision,
                        "planner_decision_reason": plan.decision_reason,
                        "batch_reason": "planner_terminal",
                        "expected_progress": "",
                        "requests_count": 0,
                        "request_fingerprints": [],
                        "tool_steps": 0,
                        "warnings_delta": 0,
                    }
                )
                break

            batch_warning_context = list(all_warnings)
            if recent_failure_summaries:
                batch_warning_context.append(
                    "recent_request_failures: " + " | ".join(recent_failure_summaries[-3:])
                )

            batch, batch_warnings = self._build_batch(
                request=request,
                flow_context=flow_context,
                plan=plan,
                pass_index=pass_index,
                pass_cap=pass_cap,
                pass_logs=all_pass_logs,
                warnings=batch_warning_context,
                changed_files=changed_files,
                latest_answer=latest_answer,
            )
            all_warnings.extend(batch_warnings)

            if batch is None:
                terminal_reason = "invalid_request_batch"
                break

            edit_retry_mode_active = bool(edit_retry_mode_pending)
            if edit_retry_mode_active:
                edit_retry_mode_pending = False

            if plan.decision == "continue" and not batch.requests:
                fallback_request = (
                    self._mutation_retry_request(
                        request=request,
                        flow_context=flow_context,
                        plan=plan,
                        step=step,
                        pass_index=pass_index,
                    )
                    if edit_retry_mode_active
                    else self._deterministic_fallback_request(
                        request=request,
                        flow_context=flow_context,
                        plan=plan,
                        step=step,
                        pass_index=pass_index,
                    )
                )
                if fallback_request is not None:
                    batch = ToolsManagerBatch(
                        planner_step_id=str(batch.planner_step_id or plan.current_step_id or ""),
                        batch_reason=(
                            "edit_retry_mode_forced_mutation"
                            if edit_retry_mode_active
                            else "deterministic_empty_batch_fallback"
                        ),
                        requests=[fallback_request],
                        continue_after=bool(batch.continue_after),
                        expected_progress=(
                            str(batch.expected_progress or "").strip()
                            or (
                                "Execute deterministic mutation retry for current planner step."
                                if edit_retry_mode_active
                                else "Execute deterministic fallback request for current planner step."
                            )
                        ),
                    )
                    all_warnings.append(
                        (
                            f"toolsmanager emitted empty request batch on pass {int(pass_index)}; "
                            "edit_retry_mode active so forcing mutation retry request"
                            if edit_retry_mode_active
                            else f"toolsmanager emitted empty request batch on pass {int(pass_index)}; using deterministic fallback request"
                        )
                    )
                else:
                    all_warnings.append(
                        f"toolsmanager emitted empty request batch on pass {int(pass_index)} and no deterministic fallback could be derived"
                    )

            todo_for_gate = run_store.read_json("todo.json", {})
            pending_edits_for_gate = (
                todo_for_gate.get("pending_edits")
                if isinstance(todo_for_gate.get("pending_edits"), list)
                else []
            )
            if (
                gate == "apply_changes"
                and persisted_gate_before_pass == "apply_changes"
                and not self._batch_has_mutation_payload(batch, pending_edits_for_gate)
            ):
                terminal_reason = "missing_edit_payload"
                all_warnings.append("missing_edit_payload")
                self._mark_structured_failure(
                    run_store,
                    code="missing_edit_payload",
                    gate=gate,
                    required_next_tool="apply_patch|write_file|create_file",
                    why_not_executed="apply_changes gate had no pending_edits and the tools batch contained no mutation tool",
                )
                break

            if (
                gate == "verify_changes"
                and persisted_gate_before_pass == "verify_changes"
                and not self._batch_has_tool_family(batch, RunStateStore.verification_tools)
            ):
                terminal_reason = "missing_verification_payload"
                all_warnings.append("missing_verification_payload")
                self._mark_structured_failure(
                    run_store,
                    code="missing_verification_payload",
                    gate=gate,
                    required_next_tool="run_command|verify_project",
                    why_not_executed="verify_changes gate had no verification command in the tools batch",
                )
                break

            request_fingerprints: list[str] = []
            tool_steps_this_pass = 0
            warnings_before = len(all_warnings)
            executed_requests = 0
            retries_this_pass = 0
            retries_exhausted_this_pass = 0
            duplicate_skips_this_pass = 0
            batch_requests: list[BatchToolRequest] = []
            request_lookup: dict[int, BatchToolRequest] = {}
            request_fingerprint_lookup: dict[int, str] = {}
            request_gate_lookup: dict[int, str] = {}
            request_args_lookup: dict[int, dict[str, Any]] = {}
            request_tool_lookup: dict[int, str] = {}
            pass_trace_rows: list[dict[str, Any]] = []
            new_files_read_this_pass = 0
            new_findings_this_pass = 0
            new_model_classes_this_pass = 0
            successful_patches_this_pass = 0
            verification_commands_this_pass = 0

            for request_index, item in enumerate(batch.requests):
                merged_policy = self._merge_policy(tool_policy, item.tool_policy_override)
                clipped_timeout = self._clip_timeout(item.timeout_seconds, session_timeout=timeout_seconds)
                question = str(item.question or "").strip()
                if not question:
                    continue
                explicit_tool_name = str(item.tool_name or "").strip()
                canonical_tool_name = explicit_tool_name or self._canonical_tool_name_from_question(question)
                if (
                    model_docs_task
                    and canonical_tool_name in {"apply_patch", "write_file"}
                    and not (persisted_gate_before_pass == "apply_changes" and gate == "apply_changes")
                ):
                    if not self._model_docs_evidence_complete(run_store):
                        duplicate_request_skips += 1
                        duplicate_skips_this_pass += 1
                        all_warnings.append("mutation_blocked_until_model_docs_evidence_complete")
                        run_store.record_tool_call(
                            gate=gate,
                            tool_name=canonical_tool_name,
                            normalized_args={"question": question, "tool_args": dict(item.tool_args or {})},
                            fingerprint=run_store.fingerprint(
                                gate=gate,
                                tool_name=canonical_tool_name,
                                args={"question": question, "tool_args": dict(item.tool_args or {})},
                                filters={},
                            ),
                            status="skipped_evidence_incomplete",
                            result_summary="model docs evidence gate blocked mutation",
                        )
                        continue
                todo_state = run_store.read_json("todo.json", {})
                pending_reads = run_store._sanitize_pending_reads(
                    todo_state.get("pending_file_reads")
                    if isinstance(todo_state.get("pending_file_reads"), list)
                    else []
                )
                pending_reads = run_store._sort_pending_reads(pending_reads, goal=request)
                if pending_reads != (
                    todo_state.get("pending_file_reads")
                    if isinstance(todo_state.get("pending_file_reads"), list)
                    else []
                ):
                    todo_state["pending_file_reads"] = pending_reads
                    run_store.write_json("todo.json", todo_state)
                if (
                    canonical_tool_name == "read_file"
                    and pending_reads
                    and not dict(item.tool_args or {}).get("path")
                ):
                    item = item.model_copy(
                        update={
                            "tool_name": "read_file",
                            "tool_args": {"path": pending_reads[0]},
                            "strategy_hint": "forced_pending_read_queue",
                        }
                    )
                    question = f"Read pending candidate file before any more broad search: {pending_reads[0]}"
                requested_read_path = run_store._normalize_candidate_path(
                    self._extract_tool_path(str(item.tool_name or ""), dict(item.tool_args or {}), question)
                )
                is_exact_pending_read = (
                    str(item.tool_name or "").strip().lower() == "read_file"
                    and bool(pending_reads)
                    and requested_read_path == pending_reads[0]
                )
                if pending_reads and not is_exact_pending_read and not (
                    persisted_gate_before_pass == "apply_changes" and gate == "apply_changes"
                ):
                    next_read = str(pending_reads[0])
                    question = f"Read pending candidate file before any more broad search: {next_read}"
                    item = item.model_copy(
                        update={
                            "question": question,
                            "tool_name": "read_file",
                            "tool_args": {"path": next_read},
                            "strategy_hint": "forced_pending_read_queue",
                        }
                    )
                    all_warnings.append("pending_read_queue_forced_progress")
                    explicit_tool_name = "read_file"
                    canonical_tool_name = "read_file"

                merged_policy, policy_repair_warning = self._repair_tool_policy_for_action(
                    merged_policy,
                    canonical_tool_name,
                )
                if policy_repair_warning:
                    all_warnings.append(policy_repair_warning)

                if (
                    model_docs_task
                    and canonical_tool_name == "read_file"
                    and self._extract_tool_path(canonical_tool_name, dict(item.tool_args or {}), question).endswith("package-lock.json")
                ):
                    duplicate_request_skips += 1
                    duplicate_skips_this_pass += 1
                    all_warnings.append("irrelevant_model_docs_read_skipped")
                    continue

                if (
                    model_docs_task
                    and canonical_tool_name in {"repo_search", "semantic_search", "list_files"}
                    and any(path.endswith("/models.py") or path == "models.py" for path in run_store._known_evidence_paths())
                    and not pending_reads
                ):
                    duplicate_semantic_search_skips += 1
                    duplicate_skips_this_pass += 1
                    all_warnings.append("broad_discovery_blocked_after_model_sources_known")
                    continue

                if (
                    model_docs_task
                    and not pending_reads
                    and canonical_tool_name in {"repo_search", "semantic_search", "list_files"}
                ):
                    duplicate_semantic_search_skips += 1
                    duplicate_skips_this_pass += 1
                    all_warnings.append("model_docs_discovery_blocked_after_pending_files_zero")
                    continue

                allowed_tools = {str(tool).strip() for tool in current_todo.allowed_tools if str(tool).strip()}
                if allowed_tools and canonical_tool_name and canonical_tool_name not in allowed_tools:
                    duplicate_tool_execution_blocks += 1
                    duplicate_skips_this_pass += 1
                    all_warnings.append(f"todo_disallowed_tool_blocked: {current_todo.id}:{canonical_tool_name}")
                    run_store.mark_todo_worker_failure(
                        current_todo,
                        reason=f"Worker attempted disallowed tool {canonical_tool_name}",
                        proof={"tool_name": canonical_tool_name},
                    )
                    continue

                if edit_retry_mode_active and self._looks_like_search_request_text(question):
                    if not self._looks_like_edit_request_text(question):
                        duplicate_semantic_search_skips += 1
                        duplicate_skips_this_pass += 1
                        all_warnings.append("duplicate_semantic_search_skipped")
                        continue

                semantic_fingerprint = self._semantic_request_fingerprint(question)
                if semantic_fingerprint and semantic_fingerprint in seen_semantic_search_fingerprints:
                    duplicate_semantic_search_skips += 1
                    duplicate_skips_this_pass += 1
                    all_warnings.append("duplicate_semantic_search_skipped")
                    continue

                candidate_tool_path = run_store._normalize_candidate_path(
                    self._extract_tool_path(canonical_tool_name, dict(item.tool_args or {}), question)
                )
                if canonical_tool_name == "read_file" and candidate_tool_path and not dict(item.tool_args or {}).get("path"):
                    item = item.model_copy(
                        update={
                            "tool_name": "read_file",
                            "tool_args": {**dict(item.tool_args or {}), "path": candidate_tool_path},
                        }
                    )
                if (
                    canonical_tool_name == "read_file"
                    and candidate_tool_path
                    and candidate_tool_path in run_store.read_files()
                    and not self._force_refresh_requested(dict(item.tool_args or {}), question)
                ):
                    duplicate_request_skips += 1
                    duplicate_skips_this_pass += 1
                    all_warnings.append(f"read_file_skipped_already_read: {candidate_tool_path}")
                    run_store.record_tool_call(
                        gate=gate,
                        tool_name="read_file",
                        normalized_args={"path": candidate_tool_path, "force_refresh": False},
                        fingerprint=run_store.fingerprint(
                            gate=gate,
                            tool_name="read_file",
                            args={"path": candidate_tool_path, "force_refresh": False},
                            filters={},
                        ),
                        status="skipped_duplicate",
                        result_summary="file already marked read in checkpoint",
                    )
                    continue
                normalized_args = {
                    "question": question,
                    "tool_args": dict(item.tool_args or {}),
                    "policy": merged_policy,
                    "timeout_seconds": clipped_timeout,
                    "index_dir": str(Path(index_dir).resolve()) if index_dir is not None else "",
                    "index_dirs": [str(Path(p).resolve()) for p in (index_dirs or []) if str(p).strip()],
                }
                normalized_args["normalized_action_key"] = run_store.normalized_action_key(
                    tool_name=canonical_tool_name or "toolsmanager_request",
                    args=normalized_args,
                    question=question,
                )
                legacy_request_fingerprint = self._fingerprint_request(question, merged_policy, clipped_timeout)
                if legacy_request_fingerprint in seen_request_fingerprints:
                    duplicate_request_skips += 1
                    duplicate_skips_this_pass += 1
                    all_warnings.append("duplicate_request_skipped")
                    continue

                request_fingerprint = run_store.fingerprint(
                    gate=gate,
                    tool_name=canonical_tool_name or "toolsmanager_request",
                    args=normalized_args,
                    filters={"k": int(k), "max_steps": int(max_steps)},
                )
                cached = run_store.successful_tool_call(request_fingerprint)
                if cached is not None:
                    duplicate_request_skips += 1
                    duplicate_skips_this_pass += 1
                    all_warnings.append("duplicate_request_skipped")
                    all_warnings.append(
                        f"duplicate_tool_call_skipped: fingerprint={request_fingerprint}"
                    )
                    run_store.record_tool_call(
                        gate=gate,
                        tool_name=canonical_tool_name or "toolsmanager_request",
                        normalized_args=normalized_args,
                        fingerprint=request_fingerprint,
                        status="skipped_duplicate",
                        result_summary=str(cached.get("result_summary", "") or "reused previous successful call"),
                    )
                    continue
                if request_fingerprint in seen_request_fingerprints:
                    duplicate_request_skips += 1
                    duplicate_skips_this_pass += 1
                    all_warnings.append("duplicate_request_skipped")
                    continue

                if canonical_tool_name:
                    allow_repeated_tool = canonical_tool_name in {"read_file", "repo_search", "list_files", "run_command"}
                    if not allow_repeated_tool and canonical_tool_name in executed_tools_this_turn:
                        duplicate_tool_execution_blocks += 1
                        duplicate_skips_this_pass += 1
                        all_warnings.append(
                            f"duplicate_tool_execution_blocked: tool={canonical_tool_name} turn={execution_run_id}"
                        )
                        continue
                    if not allow_repeated_tool:
                        executed_tools_this_turn.add(canonical_tool_name)
                    all_warnings.append(
                        f"tool_execution_registered: tool={canonical_tool_name} turn={execution_run_id}"
                    )

                seen_request_fingerprints.add(request_fingerprint)
                seen_request_fingerprints.add(legacy_request_fingerprint)
                request_fingerprints.append(request_fingerprint)
                if semantic_fingerprint:
                    seen_semantic_search_fingerprints.add(semantic_fingerprint)

                toolsmanager_requests_count += 1
                req = BatchToolRequest(
                    request_index=request_index,
                    request=ToolRunRequest(
                        question=question,
                        index_dir=str(Path(index_dir).resolve()) if index_dir is not None else None,
                        index_dirs=[str(Path(p).resolve()) for p in (index_dirs or []) if str(p).strip()] or None,
                        flow_id=flow_key or None,
                        k=int(k),
                        max_steps=int(max_steps),
                        timeout_seconds=clipped_timeout,
                        tool_policy=merged_policy,
                        system_prompt=None,
                        tool_name=canonical_tool_name or explicit_tool_name,
                        tool_args=dict(item.tool_args or {}),
                    ),
                )
                batch_requests.append(req)
                request_lookup[int(request_index)] = req
                request_fingerprint_lookup[int(request_index)] = request_fingerprint
                request_gate_lookup[int(request_index)] = gate
                request_args_lookup[int(request_index)] = normalized_args
                request_tool_lookup[int(request_index)] = canonical_tool_name or "toolsmanager_request"

            if plan.decision == "continue" and batch.requests and not batch_requests:
                pending_after_suppression = run_store._sort_pending_reads(
                    run_store.read_json("todo.json", {}).get("pending_file_reads", []),
                    goal=request,
                )
                if model_docs_task and pending_after_suppression:
                    fallback_request = ToolsManagerRequest(
                        question=f"Read pending candidate file before mutation: {pending_after_suppression[0]}",
                        tool_name="read_file",
                        tool_args={"path": pending_after_suppression[0]},
                        strategy_hint="forced_pending_read_queue",
                    )
                else:
                    fallback_request = self._mutation_retry_request(
                        request=request,
                        flow_context=flow_context,
                        plan=plan,
                        step=step,
                        pass_index=pass_index,
                    )
                merged_policy = self._merge_policy(tool_policy, fallback_request.tool_policy_override)
                fallback_tool_name = str(fallback_request.tool_name or "") or self._canonical_tool_name_from_question(
                    fallback_request.question
                )
                fallback_tool_path = run_store._normalize_candidate_path(
                    self._extract_tool_path(
                        fallback_tool_name,
                        dict(fallback_request.tool_args or {}),
                        fallback_request.question,
                    )
                )
                if fallback_tool_name == "read_file" and fallback_tool_path and not dict(fallback_request.tool_args or {}).get("path"):
                    fallback_request = fallback_request.model_copy(
                        update={"tool_name": "read_file", "tool_args": {"path": fallback_tool_path}}
                    )
                merged_policy, policy_repair_warning = self._repair_tool_policy_for_action(
                    merged_policy,
                    fallback_tool_name,
                )
                if policy_repair_warning:
                    all_warnings.append(policy_repair_warning)
                clipped_timeout = self._clip_timeout(fallback_request.timeout_seconds, session_timeout=timeout_seconds)
                legacy_fallback_fp = self._fingerprint_request(fallback_request.question, merged_policy, clipped_timeout)
                fallback_fp = run_store.fingerprint(
                    gate=gate,
                    tool_name=fallback_tool_name or "toolsmanager_request",
                    args={
                        "question": fallback_request.question,
                        "tool_args": dict(fallback_request.tool_args or {}),
                        "policy": merged_policy,
                        "timeout_seconds": clipped_timeout,
                    },
                    filters={"k": int(k), "max_steps": int(max_steps)},
                )
                if fallback_fp not in seen_request_fingerprints and legacy_fallback_fp not in seen_request_fingerprints:
                    seen_request_fingerprints.add(fallback_fp)
                    seen_request_fingerprints.add(legacy_fallback_fp)
                    request_fingerprints.append(fallback_fp)
                    toolsmanager_requests_count += 1
                    batch_requests = [
                        BatchToolRequest(
                            request_index=0,
                            request=ToolRunRequest(
                                question=fallback_request.question,
                                index_dir=str(Path(index_dir).resolve()) if index_dir is not None else None,
                                index_dirs=[str(Path(p).resolve()) for p in (index_dirs or []) if str(p).strip()] or None,
                                flow_id=flow_key or None,
                                k=int(k),
                                max_steps=int(max_steps),
                                timeout_seconds=clipped_timeout,
                                tool_policy=merged_policy,
                                system_prompt=None,
                                tool_name=fallback_tool_name,
                                tool_args=dict(fallback_request.tool_args or {}),
                            ),
                        )
                    ]
                    request_lookup = {0: batch_requests[0]}
                    request_fingerprint_lookup = {0: fallback_fp}
                    request_gate_lookup = {0: gate}
                    request_args_lookup = {
                        0: {
                            "question": fallback_request.question,
                            "tool_args": dict(fallback_request.tool_args or {}),
                            "policy": merged_policy,
                            "timeout_seconds": clipped_timeout,
                        }
                    }
                    request_tool_lookup = {0: fallback_tool_name or "toolsmanager_request"}
                    batch = ToolsManagerBatch(
                        planner_step_id=str(batch.planner_step_id or plan.current_step_id or ""),
                        batch_reason="deterministic_duplicate_suppression_fallback",
                        requests=[fallback_request],
                        continue_after=bool(batch.continue_after),
                        expected_progress="Execute deterministic fallback after duplicate suppression.",
                    )
                    all_warnings.append(
                        "toolsmanager duplicate suppression removed entire batch; forcing deterministic fallback request"
                    )
                else:
                    all_warnings.append(
                        "toolsmanager duplicate suppression removed entire batch and fallback was also duplicate"
                    )

            executor = getattr(self, "executor", LocalToolsExecutor(worker_client=self.worker_client))
            executor_failed = False
            batch_results: list[BatchExecutionResult] = []
            if batch_requests:
                try:
                    batch_results = executor.run_batch(
                        run_id=execution_run_id,
                        requests=batch_requests,
                        on_event=on_event,
                    )
                except Exception as exc:  # pragma: no cover - executor guardrail
                    batch_results = []
                    executor_failed = True
                    all_warnings.append(f"toolsmanager executor error: job_failed: {exc}")
                    execution_requests_failed += len(batch_requests)
                    for req in batch_requests:
                        idx = int(req.request_index)
                        run_store.record_tool_call(
                            gate=request_gate_lookup.get(idx, gate),
                            tool_name=request_tool_lookup.get(idx, "toolsmanager_request"),
                            normalized_args=request_args_lookup.get(idx, {}),
                            fingerprint=request_fingerprint_lookup.get(idx, ""),
                            status="error",
                            error=str(exc),
                        )

            failed_request_reasons: dict[int, tuple[str, str]] = {}
            if not executor_failed:
                seen_indexes = {int(item.request_index) for item in batch_results}
                for req in batch_requests:
                    if int(req.request_index) not in seen_indexes:
                        execution_requests_failed += 1
                        msg = (
                            "result_decode_failed: missing result for request index "
                            f"{int(req.request_index)}"
                        )
                        all_warnings.append(f"toolsmanager executor error: {msg}")
                        failed_request_reasons[int(req.request_index)] = ("result_decode_failed", msg)
                        run_store.record_tool_call(
                            gate=request_gate_lookup.get(int(req.request_index), gate),
                            tool_name=request_tool_lookup.get(int(req.request_index), "toolsmanager_request"),
                            normalized_args=request_args_lookup.get(int(req.request_index), {}),
                            fingerprint=request_fingerprint_lookup.get(int(req.request_index), ""),
                            status="error",
                            error=msg,
                        )

            for item in sorted(batch_results, key=lambda row: int(row.request_index)):
                idx = int(item.request_index)
                if idx not in request_lookup:
                    execution_requests_failed += 1
                    all_warnings.append(
                        f"toolsmanager executor error: result_decode_failed: unexpected request index {idx}"
                    )
                    run_store.record_tool_call(
                        gate=gate,
                        tool_name="toolsmanager_request",
                        normalized_args={},
                        fingerprint="",
                        status="error",
                        error=f"unexpected request index {idx}",
                    )
                    continue
                if not item.ok:
                    execution_requests_failed += 1
                    code = str(item.error_code or "job_failed")
                    detail = str(item.error_message or "request failed")
                    all_warnings.append(f"toolsmanager executor error: {code}: {detail}")
                    failed_request_reasons[idx] = (code, detail)
                    failed_todo = run_store.mark_todo_worker_failure(
                        current_todo,
                        reason=f"{code}: {detail}",
                        proof={"tool_name": request_tool_lookup.get(idx, ""), "error": detail},
                    )
                    current_todo = failed_todo
                    if code == "tools_only_violation":
                        all_warnings.append("tools_only_violation_current_todo_failed")
                        if current_todo.kind == "edit" and current_todo.status != "blocked":
                            edit_retry_mode_pending = True
                    run_store.record_tool_call(
                        gate=request_gate_lookup.get(idx, gate),
                        tool_name=request_tool_lookup.get(idx, "toolsmanager_request"),
                        normalized_args=request_args_lookup.get(idx, {}),
                        fingerprint=request_fingerprint_lookup.get(idx, ""),
                        status="error",
                        error=f"{code}: {detail}",
                    )
                    continue
                if not isinstance(item.response, dict):
                    execution_requests_failed += 1
                    msg = "result_decode_failed: response payload missing"
                    all_warnings.append(f"toolsmanager executor error: {msg}")
                    failed_request_reasons[idx] = ("result_decode_failed", msg)
                    run_store.record_tool_call(
                        gate=request_gate_lookup.get(idx, gate),
                        tool_name=request_tool_lookup.get(idx, "toolsmanager_request"),
                        normalized_args=request_args_lookup.get(idx, {}),
                        fingerprint=request_fingerprint_lookup.get(idx, ""),
                        status="error",
                        error=msg,
                    )
                    continue
                try:
                    response = ToolRunResponse.model_validate(item.response)
                except Exception as exc:
                    execution_requests_failed += 1
                    msg = f"result_decode_failed: {exc}"
                    all_warnings.append(f"toolsmanager executor error: {msg}")
                    failed_request_reasons[idx] = ("result_decode_failed", msg)
                    run_store.record_tool_call(
                        gate=request_gate_lookup.get(idx, gate),
                        tool_name=request_tool_lookup.get(idx, "toolsmanager_request"),
                        normalized_args=request_args_lookup.get(idx, {}),
                        fingerprint=request_fingerprint_lookup.get(idx, ""),
                        status="error",
                        error=msg,
                    )
                    continue

                executed_requests += 1
                evidence_counts = run_store.record_evidence_from_response(
                    gate=request_gate_lookup.get(idx, gate),
                    source_tool=request_tool_lookup.get(idx, "toolsmanager_request"),
                    response=response,
                )
                response_paths = sorted(
                    run_store._extract_paths(
                        {"trace": response.trace, "sources": response.sources, "answer": response.answer}
                    )
                )
                response_tool_name = request_tool_lookup.get(idx, "toolsmanager_request")
                response_read_paths = response_paths if str(response_tool_name).strip().lower() == "read_file" else []
                read_failure_reason = self._response_read_failed(response)
                success, no_progress_reason = self._successful_tool_status(
                    tool_name=response_tool_name,
                    evidence_counts=evidence_counts,
                    response_paths=response_paths,
                    response=response,
                )
                if (
                    str(response_tool_name).strip().lower() == "read_file"
                    and not int(evidence_counts.get("read", 0) or 0)
                ):
                    request_args = request_args_lookup.get(idx, {})
                    requested_path = run_store._normalize_candidate_path(
                        self._extract_tool_path(
                            response_tool_name,
                            request_args.get("tool_args", {}) if isinstance(request_args, dict) else {},
                            str(request_args.get("question", "") if isinstance(request_args, dict) else ""),
                        )
                    )
                    if requested_path:
                        run_store.mark_read_skipped(
                            requested_path,
                            reason=read_failure_reason or no_progress_reason or "read_file_no_files_read",
                        )
                if success:
                    failed_request_reasons.pop(idx, None)
                    execution_requests_ok += 1
                    current_todo = run_store.mark_todo_worker_done(
                        current_todo,
                        tool_name=response_tool_name,
                        files_read=response_read_paths,
                        files_changed=[],
                        command_output=str(response.answer or ""),
                        verification_result=str(response.answer or "") if response_tool_name in RunStateStore.verification_tools else "",
                        tool_call_id=request_fingerprint_lookup.get(idx, ""),
                    )
                else:
                    execution_requests_failed += 1
                    failed_request_reasons[idx] = ("no_progress", no_progress_reason or "tool produced no useful progress")
                    current_todo = run_store.mark_todo_worker_failure(
                        current_todo,
                        reason=no_progress_reason or "tool produced no useful progress",
                        proof={"tool_name": response_tool_name, "files_read": response_read_paths},
                    )
                run_store.record_tool_call(
                    gate=request_gate_lookup.get(idx, gate),
                    tool_name=response_tool_name,
                    normalized_args=request_args_lookup.get(idx, {}),
                    fingerprint=request_fingerprint_lookup.get(idx, ""),
                    status="ok" if success else "no_progress",
                    result_summary=(
                        f"answer_chars={len(str(response.answer or ''))} "
                        f"trace={len(response.trace)} sources={len(response.sources)} "
                        f"discovered={evidence_counts.get('discovered', 0)} "
                        f"read={evidence_counts.get('read', 0)}"
                        + (f" reason={no_progress_reason}" if no_progress_reason else "")
                    ),
                    files_discovered=response_paths,
                    files_read=response_read_paths,
                    error="" if success else no_progress_reason,
                )
                if success and response.answer:
                    latest_answer = str(response.answer)
                if isinstance(response.warnings, list):
                    for warning in response.warnings:
                        text = str(warning).strip()
                        if text == "Tool already executed in this turn.":
                            text = "duplicate tool execution converted to internal no-progress state"
                        if text:
                            all_warnings.append(text)
                if isinstance(response.trace, list):
                    rows = [row for row in response.trace if isinstance(row, dict)]
                    enriched_rows = self._enrich_trace_rows(
                        rows,
                        normalized_key=request_fingerprint_lookup.get(idx, ""),
                        purpose=request_gate_lookup.get(idx, gate),
                        phase=run_store._public_phase_name(run_store._phase_from_gate(request_gate_lookup.get(idx, gate))),
                        files_found=response_paths,
                        files_read=response_read_paths,
                        next_action=run_store.next_action(),
                        ledger_checkpoint_path=str(run_store.run_dir / "work_ledger.json"),
                    )
                    all_trace.extend(enriched_rows)
                    pass_trace_rows.extend(enriched_rows)
                    tool_steps_this_pass += len(enriched_rows)
                if isinstance(response.sources, list):
                    all_sources.extend([row for row in response.sources if isinstance(row, dict)])
                if success:
                    new_files_read_this_pass += int(evidence_counts.get("read", 0) or 0)
                    new_findings_this_pass += int(evidence_counts.get("discovered", 0) or 0)
                    model_classes = self._extract_model_class_names_from_response(response)
                    new_classes = model_classes.difference(seen_model_classes)
                    if new_classes:
                        seen_model_classes.update(new_classes)
                        new_model_classes_this_pass += len(new_classes)
                    executed_tool = request_tool_lookup.get(idx, "").strip().lower()
                    if executed_tool in {"apply_patch", "write_file", "create_file"}:
                        successful_patches_this_pass += 1
                    if executed_tool in {"run_command", "verify_project"}:
                        verification_commands_this_pass += 1

            if failed_request_reasons:
                retry_requests = []
                for idx in sorted(failed_request_reasons):
                    original = request_lookup.get(idx)
                    if original is None:
                        continue
                    retry_req = original.request.model_copy(
                        update={"retry_attempt": int(original.request.retry_attempt or 0) + 1}
                    )
                    code, _detail = failed_request_reasons.get(idx, ("", ""))
                    if code == "tools_only_violation":
                        forced = self._forced_mutation_tool_request(todo_item=current_todo)
                        retry_req = original.request.model_copy(
                            update={
                                "question": forced.question,
                                "tool_name": forced.tool_name,
                                "tool_args": forced.tool_args,
                                "retry_attempt": int(original.request.retry_attempt or 0) + 1,
                            }
                        )
                        request_tool_lookup[int(idx)] = forced.tool_name
                        request_args_lookup[int(idx)] = {
                            **request_args_lookup.get(int(idx), {}),
                            "question": forced.question,
                            "tool_args": forced.tool_args,
                        }
                    retry_requests.append(
                        BatchToolRequest(request_index=original.request_index, request=retry_req)
                    )
                if retry_requests:
                    retry_lookup = {int(req.request_index): req for req in retry_requests}
                    retries_this_pass = len(retry_requests)
                    request_retry_attempts += retries_this_pass
                    all_warnings.append(
                        f"toolsmanager_request_retry_once; retrying {len(retry_requests)} failed request(s)"
                    )
                    retry_results: list[BatchExecutionResult] = []
                    retry_executor_failed = False
                    try:
                        retry_results = executor.run_batch(
                            run_id=f"{execution_run_id}:retry:{pass_index}",
                            requests=retry_requests,
                            on_event=on_event,
                        )
                    except Exception as exc:  # pragma: no cover - executor guardrail
                        retry_results = []
                        retry_executor_failed = True
                        all_warnings.append(f"toolsmanager executor retry error: job_failed: {exc}")
                        execution_requests_failed += len(retry_requests)

                    retry_failures = dict(failed_request_reasons)
                    if not retry_executor_failed:
                        seen_retry_indexes = {int(item.request_index) for item in retry_results}
                        for req in retry_requests:
                            idx = int(req.request_index)
                            if idx not in seen_retry_indexes:
                                execution_requests_failed += 1
                                msg = f"result_decode_failed: missing retry result for request index {idx}"
                                all_warnings.append(f"toolsmanager executor retry error: {msg}")
                                retry_failures[idx] = ("result_decode_failed", msg)

                        for item in sorted(retry_results, key=lambda row: int(row.request_index)):
                            idx = int(item.request_index)
                            if idx not in retry_lookup:
                                execution_requests_failed += 1
                                msg = f"result_decode_failed: unexpected retry result index {idx}"
                                all_warnings.append(f"toolsmanager executor retry error: {msg}")
                                continue
                            if not item.ok:
                                execution_requests_failed += 1
                                code = str(item.error_code or "job_failed")
                                detail = str(item.error_message or "request failed")
                                all_warnings.append(f"toolsmanager executor retry error: {code}: {detail}")
                                retry_failures[idx] = (code, detail)
                                continue
                            if not isinstance(item.response, dict):
                                execution_requests_failed += 1
                                msg = "result_decode_failed: retry response payload missing"
                                all_warnings.append(f"toolsmanager executor retry error: {msg}")
                                retry_failures[idx] = ("result_decode_failed", msg)
                                continue
                            try:
                                response = ToolRunResponse.model_validate(item.response)
                            except Exception as exc:
                                execution_requests_failed += 1
                                msg = f"result_decode_failed: {exc}"
                                all_warnings.append(f"toolsmanager executor retry error: {msg}")
                                retry_failures[idx] = ("result_decode_failed", msg)
                                continue

                            executed_requests += 1
                            retry_evidence_counts = run_store.record_evidence_from_response(
                                gate=request_gate_lookup.get(idx, gate),
                                source_tool=request_tool_lookup.get(idx, "toolsmanager_request"),
                                response=response,
                            )
                            retry_response_paths = sorted(
                                run_store._extract_paths(
                                    {"trace": response.trace, "sources": response.sources, "answer": response.answer}
                                )
                            )
                            retry_tool_name = request_tool_lookup.get(idx, "toolsmanager_request")
                            retry_read_paths = (
                                retry_response_paths if str(retry_tool_name).strip().lower() == "read_file" else []
                            )
                            retry_read_failure_reason = self._response_read_failed(response)
                            retry_success, retry_no_progress_reason = self._successful_tool_status(
                                tool_name=retry_tool_name,
                                evidence_counts=retry_evidence_counts,
                                response_paths=retry_response_paths,
                                response=response,
                            )
                            if (
                                str(retry_tool_name).strip().lower() == "read_file"
                                and not int(retry_evidence_counts.get("read", 0) or 0)
                            ):
                                request_args = request_args_lookup.get(idx, {})
                                requested_path = run_store._normalize_candidate_path(
                                    self._extract_tool_path(
                                        retry_tool_name,
                                        request_args.get("tool_args", {}) if isinstance(request_args, dict) else {},
                                        str(request_args.get("question", "") if isinstance(request_args, dict) else ""),
                                    )
                                )
                                if requested_path:
                                    run_store.mark_read_skipped(
                                        requested_path,
                                        reason=retry_read_failure_reason
                                        or retry_no_progress_reason
                                        or "read_file_no_files_read",
                                    )
                            if retry_success:
                                retry_failures.pop(idx, None)
                                execution_requests_ok += 1
                                current_todo = run_store.mark_todo_worker_done(
                                    current_todo,
                                    tool_name=retry_tool_name,
                                    files_read=retry_read_paths,
                                    files_changed=[],
                                    command_output=str(response.answer or ""),
                                    verification_result=str(response.answer or "") if retry_tool_name in RunStateStore.verification_tools else "",
                                    tool_call_id=request_fingerprint_lookup.get(idx, ""),
                                )
                            else:
                                execution_requests_failed += 1
                                retry_failures[idx] = (
                                    "no_progress",
                                    retry_no_progress_reason or "retry produced no useful progress",
                                )
                                current_todo = run_store.mark_todo_worker_failure(
                                    current_todo,
                                    reason=retry_no_progress_reason or "retry produced no useful progress",
                                    proof={"tool_name": retry_tool_name, "files_read": retry_read_paths},
                                )
                            run_store.record_tool_call(
                                gate=request_gate_lookup.get(idx, gate),
                                tool_name=retry_tool_name,
                                normalized_args=request_args_lookup.get(idx, {}),
                                fingerprint=request_fingerprint_lookup.get(idx, ""),
                                status="ok" if retry_success else "no_progress",
                                result_summary=(
                                    f"retry answer_chars={len(str(response.answer or ''))} "
                                    f"trace={len(response.trace)} sources={len(response.sources)} "
                                    f"discovered={retry_evidence_counts.get('discovered', 0)} "
                                    f"read={retry_evidence_counts.get('read', 0)}"
                                    + (
                                        f" reason={retry_no_progress_reason}"
                                        if retry_no_progress_reason
                                        else ""
                                    )
                                ),
                                files_discovered=retry_response_paths,
                                files_read=retry_read_paths,
                                error="" if retry_success else retry_no_progress_reason,
                            )
                            if retry_success:
                                new_files_read_this_pass += int(retry_evidence_counts.get("read", 0) or 0)
                                new_findings_this_pass += int(retry_evidence_counts.get("discovered", 0) or 0)
                                model_classes = self._extract_model_class_names_from_response(response)
                                new_classes = model_classes.difference(seen_model_classes)
                                if new_classes:
                                    seen_model_classes.update(new_classes)
                                    new_model_classes_this_pass += len(new_classes)
                                retried_tool = request_tool_lookup.get(idx, "").strip().lower()
                                if retried_tool in {"apply_patch", "write_file", "create_file"}:
                                    successful_patches_this_pass += 1
                                if retried_tool in {"run_command", "verify_project"}:
                                    verification_commands_this_pass += 1
                            if retry_success and response.answer:
                                latest_answer = str(response.answer)
                            if isinstance(response.warnings, list):
                                for warning in response.warnings:
                                    text = str(warning).strip()
                                    if text == "Tool already executed in this turn.":
                                        text = "duplicate tool execution converted to internal no-progress state"
                                    if text:
                                        all_warnings.append(text)
                            if isinstance(response.trace, list):
                                rows = [row for row in response.trace if isinstance(row, dict)]
                                enriched_rows = self._enrich_trace_rows(
                                    rows,
                                    normalized_key=request_fingerprint_lookup.get(idx, ""),
                                    purpose=request_gate_lookup.get(idx, gate),
                                    phase=run_store._public_phase_name(
                                        run_store._phase_from_gate(request_gate_lookup.get(idx, gate))
                                    ),
                                    files_found=retry_response_paths,
                                    files_read=retry_read_paths,
                                    next_action=run_store.next_action(),
                                    ledger_checkpoint_path=str(run_store.run_dir / "work_ledger.json"),
                                )
                                all_trace.extend(enriched_rows)
                                pass_trace_rows.extend(enriched_rows)
                                tool_steps_this_pass += len(enriched_rows)
                            if isinstance(response.sources, list):
                                all_sources.extend([row for row in response.sources if isinstance(row, dict)])

                    if retry_failures:
                        retries_exhausted_this_pass = len(retry_failures)
                        request_retry_exhausted += retries_exhausted_this_pass
                        for idx, (code, detail) in retry_failures.items():
                            req = request_lookup.get(int(idx))
                            question = str(req.request.question if req is not None else "")
                            signature = self._build_failure_signature(
                                question=question,
                                code=code,
                                detail=detail,
                            )
                            if signature not in seen_failure_signatures:
                                seen_failure_signatures.add(signature)
                                summary = f"{code}: {self._truncate_line(detail, limit=140)}"
                                recent_failure_summaries.append(summary)
                                all_warnings.append(f"toolsmanager_retry_exhausted_signature={signature}:{summary}")
                            if memory_service is not None and flow_key:
                                try:
                                    memory_service.record_tool_fingerprint(
                                        flow_id=flow_key,
                                        kind="mutation_failure_signature",
                                        fingerprint=signature,
                                    )
                                except Exception:
                                    pass

            changed_now = sorted(self._git_status_paths().difference(before))
            changed_files = changed_now
            docs_models_changed_this_pass = "docs/models.md" in changed_now
            pending_after_pass = run_store._canonical_pending_reads(
                run_store.read_json("todo.json", {}).get("pending_file_reads", []),
                goal=request,
            )
            current_todo = run_store.current_todo_for_gate(gate, goal=request)
            if current_todo.status == "worker_done":
                current_todo = run_store.confirm_or_reject_todo(
                    current_todo,
                    files_changed=changed_files,
                    pending_files=pending_after_pass,
                )
                if current_todo.status == "failed" and current_todo.kind == "edit":
                    edit_retry_mode_pending = True
                    all_warnings.append("todo_edit_agent_confirmation_failed")
            all_warnings.extend(run_store.validate_planner_todo_claims(changed_files=changed_files))
            run_store.update_state(
                plan=plan,
                step=step,
                status="running",
                next_action=run_store.next_action(),
                changed_files=changed_files,
            )

            semantic_trace_fps = self._semantic_trace_fingerprints(pass_trace_rows)
            if semantic_trace_fps:
                seen_semantic_search_fingerprints.update(semantic_trace_fps)

            if memory_service is not None and flow_key:
                try:
                    if request_fingerprints:
                        memory_service.record_tool_fingerprints(
                            flow_id=flow_key,
                            kind="request_fingerprint",
                            fingerprints=request_fingerprints,
                        )
                    if semantic_trace_fps:
                        memory_service.record_tool_fingerprints(
                            flow_id=flow_key,
                            kind="semantic_search_fingerprint",
                            fingerprints=sorted(semantic_trace_fps),
                        )
                    step_fingerprint = self._planner_task_fingerprint(step)
                    if step_fingerprint and (executed_requests > 0 or tool_steps_this_pass > 0):
                        seen_planner_task_fingerprints.add(step_fingerprint)
                        memory_service.record_tool_fingerprint(
                            flow_id=flow_key,
                            kind="planner_task_fingerprint",
                            fingerprint=step_fingerprint,
                        )
                    memory_service.prune_tool_fingerprints(
                        flow_id=flow_key,
                        kind="request_fingerprint",
                    )
                    memory_service.prune_tool_fingerprints(
                        flow_id=flow_key,
                        kind="semantic_search_fingerprint",
                    )
                    memory_service.prune_tool_fingerprints(
                        flow_id=flow_key,
                        kind="mutation_failure_signature",
                    )
                    memory_service.prune_tool_fingerprints(
                        flow_id=flow_key,
                        kind="planner_task_fingerprint",
                    )
                except Exception as exc:
                    all_warnings.append(f"toolsmanager persistent fingerprint write failed: {exc}")

            apply_patch_attempted = any(
                str(row.get("tool_name", "")).strip().lower() == "apply_patch"
                for row in pass_trace_rows
            )
            apply_patch_failed = any(self._looks_like_apply_patch_failure_trace(row) for row in pass_trace_rows)
            if changed_now:
                edit_retry_mode_pending = False
            elif apply_patch_attempted and (apply_patch_failed or not changed_now):
                if model_docs_task and not self._model_docs_evidence_complete(run_store):
                    all_warnings.append("edit_retry_mode_blocked_until_model_docs_evidence_complete")
                else:
                    edit_retry_mode_pending = True
                    edit_retry_mode_activations += 1
                    all_warnings.append("edit_retry_mode_activated")

            warnings_delta = max(0, len(all_warnings) - warnings_before)
            made_progress_this_pass = bool(
                new_files_read_this_pass
                or new_model_classes_this_pass
                or docs_models_changed_this_pass
                or (successful_patches_this_pass and changed_now)
                or verification_commands_this_pass
            )
            all_pass_logs.append(
                {
                    "pass_index": pass_index,
                    "planner_step_id": plan.current_step_id,
                    "planner_step_title": str(getattr(step, "title", "") or ""),
                    "planner_decision": plan.decision,
                    "planner_decision_reason": plan.decision_reason,
                    "batch_reason": str(batch.batch_reason or ""),
                    "expected_progress": str(batch.expected_progress or ""),
                    "requests_count": len(batch.requests),
                    "request_fingerprints": request_fingerprints,
                    "tool_steps": tool_steps_this_pass,
                    "warnings_delta": warnings_delta,
                    "continue_after": bool(batch.continue_after),
                    "execution_backend": execution_backend,
                    "duplicate_skips": duplicate_skips_this_pass,
                    "request_retry_attempts": retries_this_pass,
                    "request_retry_exhausted": retries_exhausted_this_pass,
                    "edit_retry_mode_active": bool(edit_retry_mode_active),
                    "new_files_read": new_files_read_this_pass,
                    "new_findings": new_findings_this_pass,
                    "new_model_classes": new_model_classes_this_pass,
                    "docs_models_changed": docs_models_changed_this_pass,
                    "successful_patches": successful_patches_this_pass,
                    "verification_commands": verification_commands_this_pass,
                    "made_progress": made_progress_this_pass,
                }
            )

            state_after_pass = run_store.read_json("state.json", {})
            state_after_pass["progress_counters"] = run_store.progress_counters(
                pass_logs=all_pass_logs,
                tool_calls=toolsmanager_requests_count,
            )
            if not made_progress_this_pass:
                stalled_passes += 1
            else:
                stalled_passes = 0
            state_after_pass["no_progress_count"] = stalled_passes
            state_after_pass["progress_counters"]["no_progress_count"] = stalled_passes
            run_store.write_json("state.json", state_after_pass)

            if stalled_passes >= max_no_progress_passes:
                terminal_reason = "stalled_no_actionable_requests"
                break

            if pass_index >= pass_cap:
                terminal_reason = "pass_cap_reached"
                break

            plan_warning_context = list(all_warnings)
            if recent_failure_summaries:
                plan_warning_context.append(
                    "recent_request_failures: " + " | ".join(recent_failure_summaries[-3:])
                )

            plan, new_plan_warnings, _source = self._plan_with_source(
                request=request,
                flow_context=flow_context,
                pass_index=pass_index,
                pass_cap=pass_cap,
                previous_plan=plan,
                pass_logs=all_pass_logs,
                warnings=plan_warning_context,
                changed_files=changed_files,
                latest_answer=latest_answer,
            )
            all_warnings.extend(new_plan_warnings)

        if terminal_reason == "pass_cap_reached" and self._has_pending_plan_work(plan):
            latest_answer = self._synthesize_resumable_pass_cap_answer(
                pass_logs=all_pass_logs,
                planner_decisions=planner_decisions,
                toolsmanager_requests_count=toolsmanager_requests_count,
            )
            all_warnings.append("pass_cap_reached_with_pending_work")
            if is_edit_task and not changed_files:
                all_warnings.append("edit_task_pass_cap_without_changed_files")

        if not str(latest_answer or "").strip():
            latest_answer = self._synthesize_terminal_answer(
                terminal_reason=terminal_reason,
                pass_logs=all_pass_logs,
                planner_decisions=planner_decisions,
                toolsmanager_requests_count=toolsmanager_requests_count,
            )

        structured_failure_reasons = {
            "missing_edit_payload",
            "missing_verification_payload",
            "tools_only_no_tool_call",
            "mutation_noop",
            "verification_no_changed_files",
        }
        provisional_status = (
            "failed_no_progress"
            if terminal_reason in structured_failure_reasons
            else (
                "needs_resume"
                if terminal_reason == "pass_cap_reached" and self._has_pending_plan_work(plan)
                else "running"
            )
        )
        final_state = run_store.update_state(
            plan=plan,
            step=self._resolve_step(plan),
            status=provisional_status,
            blocking_reason=("pass budget reached" if provisional_status == "needs_resume" else ""),
            next_action=run_store.next_action(),
            changed_files=changed_files,
        )
        completion_blocker = self._completion_blocker(
            run_store=run_store,
            plan=plan,
            final_state=final_state,
            changed_files=changed_files,
            is_edit_task=is_edit_task,
            latest_answer=latest_answer,
            warnings=all_warnings,
            terminal_reason=terminal_reason,
        )
        if terminal_reason in structured_failure_reasons:
            final_status = "failed_no_progress"
        elif terminal_reason in {"stalled_no_actionable_requests", "invalid_request_batch"}:
            final_status = "failed_no_progress"
        elif provisional_status == "needs_resume" or self._has_pending_plan_work(plan):
            final_status = "needs_resume"
        elif completion_blocker:
            final_status = "needs_resume"
        else:
            final_status = "completed"
        if final_status == "completed":
            completed_gates = list(
                final_state.get("completed_gates", [])
                if isinstance(final_state.get("completed_gates"), list)
                else []
            )
            if "final_report" not in completed_gates:
                completed_gates.append("final_report")
            final_state["status"] = "completed"
            final_state["current_gate"] = "final_report"
            final_state["current_phase"] = "FINAL"
            final_state["completed_gates"] = completed_gates
            final_state["pending_gates"] = [item for item in run_store.gates if item not in completed_gates]
            final_state["blocking_reason"] = ""
        else:
            final_state["status"] = final_status
            final_state["blocking_reason"] = completion_blocker or str(final_state.get("blocking_reason", "") or "")
            if terminal_reason in structured_failure_reasons:
                final_state["blocking_reason"] = terminal_reason
                final_state.setdefault("failure_stage", run_store._public_phase_name(run_store._phase_from_gate(str(final_state.get("current_gate", "") or ""))))
                final_state.setdefault("failure_gate", str(final_state.get("current_gate", "") or ""))
                final_state.setdefault("required_next_tool", "")
                final_state.setdefault("why_not_executed", str(final_state.get("last_error", "") or terminal_reason))
        run_store.write_json("state.json", final_state)
        run_store.write_checkpoint(
            status=final_status,
            completed_gates=[
                str(item)
                for item in final_state.get("completed_gates", [])
                if str(item).strip()
            ],
            pending_gates=[
                str(item)
                for item in final_state.get("pending_gates", [])
                if str(item).strip()
            ],
            files_changed=changed_files,
            verification_status=str(final_state.get("verification_status", "pending") or "pending"),
            blocker=str(final_state.get("blocking_reason", "") or ""),
            pass_logs=all_pass_logs,
            tool_calls=toolsmanager_requests_count,
            plan=plan.model_dump(),
            last_error="; ".join(str(item) for item in all_warnings[-3:]),
        )
        resume_command = f"mana-analyzer continue --root-dir {run_store.repo_root} --run-id {run_store.run_id}"
        if final_status in {"needs_resume", "failed_no_progress"}:
            evidence_rows = run_store.read_jsonl("evidence.jsonl")
            located_count = len({str(row.get("file_path", "")) for row in evidence_rows if row.get("file_path")})
            read_count = len({str(row.get("file_path", "")) for row in evidence_rows if row.get("status") == "read"})
            pending_reads = final_state.get("pending_file_reads") if isinstance(final_state.get("pending_file_reads"), list) else []
            phase = str(final_state.get("current_phase", "") or "")
            completed_count = len(final_state.get("completed_gates", []) if isinstance(final_state.get("completed_gates"), list) else [])
            pending_count = len(final_state.get("pending_gates", []) if isinstance(final_state.get("pending_gates"), list) else []) + len(pending_reads)
            auto_continue_command = (
                f"{resume_command} --auto-continue --max-passes 12 --max-total-tool-calls 80 --max-no-progress-passes 2"
            )
            heading = (
                "No progress made; run needs inspection before resuming."
                if final_status == "failed_no_progress"
                else "Pass budget reached with pending work and should continue."
            )
            latest_answer = (
                f"{heading}\n"
                f"Current phase: {phase or 'unknown'}\n"
                f"terminal_reason={terminal_reason}\n"
                f"Completed work: {completed_count}; pending work: {pending_count}\n"
                f"Candidate files: located {located_count}, read {read_count}, pending {len(pending_reads)}\n"
                f"Reason: {final_state.get('blocking_reason', '') or terminal_reason}\n"
                f"Next exact action: {final_state.get('next_action', run_store.next_action())}\n"
                f"Resume command: {resume_command}\n"
                f"Auto-continue command: {auto_continue_command}\n\n"
                f"{run_store.todo_board()}"
            )
        elif final_status == "completed":
            latest_answer = f"{latest_answer}\n\n{run_store.todo_board()}".strip()

        persisted_fingerprint_counts = {
            "request_fingerprint": len(seen_request_fingerprints),
            "semantic_search_fingerprint": len(seen_semantic_search_fingerprints),
            "mutation_failure_signature": len(seen_failure_signatures),
            "planner_task_fingerprint": len(seen_planner_task_fingerprints),
        }

        return AutoExecuteResult(
            answer=latest_answer,
            sources=all_sources,
            trace=all_trace,
            warnings=all_warnings,
            changed_files=changed_files,
            plan=plan.model_dump(),
            passes=len(all_pass_logs),
            terminal_reason=terminal_reason,
            toolsmanager_requests_count=toolsmanager_requests_count,
            pass_logs=all_pass_logs,
            planner_decisions=planner_decisions,
            execution_backend=execution_backend,
            execution_run_id=execution_run_id,
            execution_duration_ms=round((time.perf_counter() - execution_started) * 1000.0, 3),
            execution_requests_ok=execution_requests_ok,
            execution_requests_failed=execution_requests_failed,
            duplicate_request_skips=duplicate_request_skips,
            duplicate_semantic_search_skips=duplicate_semantic_search_skips,
            duplicate_tool_execution_blocks=duplicate_tool_execution_blocks,
            request_retry_attempts=request_retry_attempts,
            request_retry_exhausted=request_retry_exhausted,
            edit_retry_mode_activations=edit_retry_mode_activations,
            persisted_fingerprint_counts=persisted_fingerprint_counts,
            run_id=run_store.run_id,
            run_dir=str(run_store.run_dir),
            run_status=final_status,
            resume_command=resume_command,
            next_action=str(final_state.get("next_action", "") or ""),
        )

    def resume_run(
        self,
        *,
        run_id: str,
        index_dir: str | Path | None,
        index_dirs: Sequence[str | Path] | None,
        k: int,
        max_steps: int,
        timeout_seconds: int,
        tool_policy: dict[str, Any],
        pass_cap: int,
        on_event: Callable[[Any], None] | None = None,
        flow_id: str | None = None,
        max_no_progress_passes: int = 2,
    ) -> AutoExecuteResult:
        store = RunStateStore(repo_root=self.repo_root, run_id=run_id)
        state = store.read_json("state.json", {})
        if not state:
            raise FileNotFoundError(f"Run state not found for run_id={run_id}")
        checkpoint = store.read_json("checkpoint.json", {})
        resume_prompt_path = store.run_dir / "resume_prompt.md"
        resume_prompt = resume_prompt_path.read_text(encoding="utf-8") if resume_prompt_path.exists() else ""
        next_exact_action = str(
            checkpoint.get("next_exact_action")
            or checkpoint.get("next_action")
            or state.get("next_exact_action")
            or state.get("next_action")
            or ""
        ).strip()
        request = (
            (f"Resume exact next action: {next_exact_action}\n{resume_prompt}".strip() if next_exact_action else "")
            or str(resume_prompt or "").strip()
            or str(state.get("goal", "") or "").strip()
            or f"Resume mana-analyzer run {run_id}"
        )
        return self.run(
            request=request,
            flow_context=(
                f"Resuming run_id={run_id}\n"
                f"Current phase: {checkpoint.get('current_phase', state.get('current_phase', ''))}\n"
                f"Current gate: {state.get('current_gate', '')}\n"
                f"Completed gates: {', '.join(state.get('completed_gates', []) if isinstance(state.get('completed_gates'), list) else [])}\n"
                f"Pending gates: {', '.join(state.get('pending_gates', []) if isinstance(state.get('pending_gates'), list) else [])}\n"
                f"Next action: {next_exact_action}"
            ),
            index_dir=index_dir,
            index_dirs=index_dirs,
            k=k,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            tool_policy=tool_policy,
            pass_cap=pass_cap,
            on_event=on_event,
            flow_id=flow_id or str(state.get("flow_id", "") or ""),
            run_id=run_id,
            max_no_progress_passes=max_no_progress_passes,
        )
        
    @staticmethod
    def _canonical_tool_name_from_question(question: str) -> str:
        normalized = re.sub(r"\s+", " ", str(question or "").strip()).lower()
        if not normalized:
            return ""
        match = re.search(r"\btool\s*[:=]\s*['\"]?([a-z0-9_\-]+)", normalized)
        if match:
            return str(match.group(1) or "").strip().lower()
        known = (
            "semantic_search",
            "repo_search",
            "read_file",
            "find_symbols",
            "call_graph",
            "run_command",
            "apply_patch",
            "write_file",
            "search_internet",
            "github_search",
        )
        for tool_name in known:
            if tool_name in normalized:
                return tool_name
        return ""


__all__ = [
    "ToolsPlan",
    "ToolsPlanStep",
    "ToolsManagerRequest",
    "ToolsManagerBatch",
    "AutoExecuteResult",
    "ToolsManagerOrchestrator",
]
