from __future__ import annotations
import logging
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Container, Literal, Sequence, TypeVar

from pydantic import BaseModel, Field

from mana_agent.llm.gate_command import (
    GateCommand,
    PolicyDecision,
    ProofResult,
    build_gate_command,
    preflight_tool_policy,
    reconcile_gate_pointer,
    tool_fingerprint,
    validate_gate_proof,
)
from mana_agent.llm.goal_profiles import GoalProfile, active_goal_profile
from mana_agent.llm.tool_worker_process import ToolRunResponse, ToolWorkerClient
from mana_agent.llm.tools_executor import (
    ToolsExecutionConfig,
    ToolsExecutor,
)
from mana_agent.services.coding_memory_service import CodingMemoryService
from mana_agent.services.coding_todo_service import TodoService
from mana_agent.tools.write_file import safe_create_file, safe_write_file

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
    mutation_tools = {"edit_file", "multi_edit_file", "apply_patch", "write_file", "create_file", "delete_file"}
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
            if tool in {"edit_file", "multi_edit_file", "apply_patch", "write_file", "create_file", "delete_file"} and any(
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
            if row.get("status") == "ok" and tool in {"edit_file", "multi_edit_file", "apply_patch", "write_file", "create_file", "delete_file"}:
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
                if row.get("status") == "ok" and str(row.get("tool_name", "")).strip().lower() in {"edit_file", "multi_edit_file", "apply_patch", "write_file", "create_file", "delete_file"}
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
            f"Resume mana-agent run {self.run_id}.",
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
                    "allowed_tools": ["edit_file", "multi_edit_file", "apply_patch", "write_file", "create_file", "delete_file"],
                    "required_tool": "edit_file|multi_edit_file|apply_patch|write_file|create_file|delete_file",
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
                "allowed_tools": ["edit_file", "multi_edit_file", "apply_patch", "write_file", "create_file", "delete_file"],
                "required_tool": "edit_file|multi_edit_file|apply_patch|write_file|create_file|delete_file",
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


_MUTATION_TOOLS = {"edit_file", "multi_edit_file", "apply_patch", "write_file", "create_file", "delete_file"}
_NON_PROGRESS_STATUSES = {
    "blocked",
    "skipped",
    "duplicate_blocked",
    "not_allowed",
    "verify_project_blocked_until_mutation",
    "no_progress",
    "skipped_no_progress",
}


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip().replace("\\", "/").lstrip("./") for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip().replace("\\", "/").lstrip("./")]
    return []


def _extract_changed_files_from_value(value: Any) -> list[str]:
    changed: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).strip().lower()
            if lowered in {"files_changed", "changed_files", "modified_files"}:
                changed.extend(_as_str_list(item))
                continue
            if lowered == "proof" and isinstance(item, dict):
                changed.extend(_as_str_list(item.get("modified_files")))
                changed.extend(_as_str_list(item.get("changed_files")))
                continue
            changed.extend(_extract_changed_files_from_value(item))
    elif isinstance(value, list):
        for item in value:
            changed.extend(_extract_changed_files_from_value(item))
    return sorted(dict.fromkeys(path for path in changed if path))


def _trace_tool_name(row: dict[str, Any]) -> str:
    return str(row.get("tool_name") or row.get("tool") or row.get("name") or row.get("action") or "").strip().lower()


def _trace_status(row: dict[str, Any]) -> str:
    return str(row.get("status") or row.get("result") or "").strip().lower()


_MUTATION_INTENT_RE = re.compile(
    r"\b(create|update|edit|modify|delete|remove|write|generate\s+file|add\s+file|patch|fix|refactor|rename)\b",
    re.IGNORECASE,
)
_EXPLICIT_FILE_RE = re.compile(r"(?P<path>(?:[\w.-]+/)*[\w.-]+\.(?:md|txt|py|json|toml|yaml|yml|ini|cfg))\b")
_CREATE_FILE_IN_DIR_RE = re.compile(
    r"\b(?:create|write|generate|add)\b(?:\s+(?:a|an|the|file))?\s+"
    r"(?P<file>[\w.-]+\.(?:md|txt|py|json|toml|yaml|yml|ini|cfg))\s+"
    r"(?:in|under|inside)\s+(?P<dir>[\w./-]+)",
    re.IGNORECASE,
)
_MUTATION_FALLBACK_ALLOWED_TOOLS = frozenset(
    {"edit_file", "multi_edit_file", "apply_patch", "write_file", "create_file", "delete_file", "git_status", "git_diff", "verify_project"}
)
_MUTATION_FALLBACK_BLOCKED_TOOLS = frozenset(
    {
        "repo_search",
        "semantic_search",
        "read_file",
        "list_files",
        "ls",
        "chunk_file",
        "find_symbols",
        "call_graph",
        "list_tools",
    }
)
_CREATE_ARTIFACT_INTENT_RE = re.compile(r"\b(create|write|generate|add(?:\s+file)?)\b", re.IGNORECASE)
_ANALYSIS_ARTIFACT_INTENT_RE = re.compile(r"\b(analy[sz]e|analysis|report|document|summarize)\b", re.IGNORECASE)
_DETERMINISTIC_ARTIFACT_SUFFIXES = (".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".ini", ".cfg")
_README_ATTACH_RE = re.compile(r"\b(?:attach|add|link|include|update)\b.*\breadme(?:\.md)?\b", re.IGNORECASE)


def _mutation_required_from_policy(tool_policy: dict[str, Any] | None, requires_edit: bool | None) -> bool:
    if bool(requires_edit):
        return True
    if not isinstance(tool_policy, dict):
        return False
    if bool(tool_policy.get("mutation_required")):
        return True
    nested = tool_policy.get("tool_policy")
    return isinstance(nested, dict) and bool(nested.get("mutation_required"))


def _mutation_required_from_text(text: str) -> bool:
    return bool(_MUTATION_INTENT_RE.search(str(text or "")))


def _safe_relative_path(repo_root: Path, raw: str) -> str:
    cleaned = str(raw or "").strip().replace("\\", "/").strip("`'\" ")
    cleaned = re.sub(r"[,.):;\]]+$", "", cleaned).lstrip("./")
    if not cleaned or cleaned.startswith("/") or "\x00" in cleaned:
        return ""
    if cleaned.lower() == "readme.md" and (repo_root / "README.md").exists():
        cleaned = "README.md"
    parts = [part for part in cleaned.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        return ""
    target = (repo_root / Path(*parts)).resolve()
    try:
        return target.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return ""


def _resolve_mutation_target_path(task: str, repo_root: Path, target_files: Sequence[str] = ()) -> str:
    for item in target_files:
        rel = _safe_relative_path(repo_root, item)
        if rel:
            return rel

    text = str(task or "")
    match = _CREATE_FILE_IN_DIR_RE.search(text)
    if match:
        directory = match.group("dir").strip().strip("`'\" ")
        filename = match.group("file").strip().strip("`'\" ")
        rel = _safe_relative_path(repo_root, f"{directory.rstrip('/')}/{filename}")
        if rel:
            return rel

    candidates: list[str] = []
    for match in _EXPLICIT_FILE_RE.finditer(text):
        rel = _safe_relative_path(repo_root, match.group("path"))
        if rel:
            candidates.append(rel)
    if candidates:
        if _README_ATTACH_RE.search(text):
            for candidate in candidates:
                if Path(candidate).name.lower() != "readme.md":
                    return candidate
        return candidates[-1]
    return ""


# Minimum size (bytes) a deliverable must reach to count as a real write rather
# than an empty placeholder. Verification rejects anything smaller.
_MIN_DELIVERABLE_SIZE = 1

# "in/under/inside <dir>" trailing clause used to normalize bare filenames under
# a requested directory (e.g. "create A and B in docs" -> docs/A, docs/B).
_DIRECTORY_CLAUSE_RE = re.compile(
    r"\b(?:in|under|inside|into|within)\s+(?:the\s+)?"
    r"(?P<dir>[\w./-]+?)\s*(?:directory|folder|dir)?\b",
    re.IGNORECASE,
)


def _shared_directory_clause(text: str) -> str:
    """Return the last ``in/under/inside <dir>`` directory mentioned, or ''.

    Bare filenames in a multi-file request inherit this directory so that
    "create 01-overview.md & 02-installation.md in docs" lands both files under
    ``docs/`` instead of the repository root.
    """
    directory = ""
    for match in _DIRECTORY_CLAUSE_RE.finditer(str(text or "")):
        candidate = match.group("dir").strip().strip("`'\"/ ")
        # Skip filenames accidentally captured as the directory token.
        if candidate and not candidate.lower().endswith(_DETERMINISTIC_ARTIFACT_SUFFIXES):
            directory = candidate
    return directory


def _resolve_required_deliverables(
    request: str, repo_root: Path, target_files: Sequence[str] = ()
) -> list[str]:
    """Extract every required output file from the request, normalized under repo.

    Unlike :func:`_resolve_mutation_target_path` (which returns a single primary
    target), this returns the full ordered list of deliverables the request
    demands. The run is held to *all* of them by the verification gate.
    """
    explicit: list[str] = []
    for item in target_files:
        rel = _safe_relative_path(repo_root, item)
        if rel and rel not in explicit:
            explicit.append(rel)
    if explicit:
        return explicit

    text = str(request or "")
    # README-attach requests retain single-target semantics: the primary
    # deliverable is the non-README artifact and README is updated as a side
    # effect by the deterministic fallback, not tracked as its own deliverable.
    if _README_ATTACH_RE.search(text):
        single = _resolve_mutation_target_path(text, repo_root, target_files)
        return [single] if single else []

    directory = _shared_directory_clause(text)
    normalized: list[str] = []
    for match in _EXPLICIT_FILE_RE.finditer(text):
        raw = match.group("path")
        candidate = f"{directory}/{raw}" if directory and "/" not in raw else raw
        rel = _safe_relative_path(repo_root, candidate)
        if rel and rel not in normalized:
            normalized.append(rel)
    if normalized:
        return normalized

    single = _resolve_mutation_target_path(text, repo_root, target_files)
    return [single] if single else []


def _required_file_satisfied(
    repo_root: Path,
    rel: str,
    *,
    min_size: int = _MIN_DELIVERABLE_SIZE,
    changed: Container[str] = (),
) -> bool:
    """True when ``rel`` is a real, substantive deliverable.

    The file must exist on disk with at least ``min_size`` bytes AND not be a
    placeholder stub. Being in ``changed`` (the trace-proven changed-files set)
    is necessary but not sufficient: a successful-but-empty/stub write still
    fails here, because there is no template fallback to rescue it — the worker
    must author genuine content. This is the on-disk truth the verification gate
    relies on to surface honest failure.
    """
    if not rel:
        return False
    target = (repo_root / rel)
    try:
        if not (target.is_file() and target.stat().st_size >= int(min_size)):
            return False
    except OSError:
        return False
    try:
        if _looks_like_stub(target.read_text(encoding="utf-8", errors="replace")):
            return False
    except OSError:
        return False
    # A non-stub file that exists on disk is satisfied. The ``changed`` set is
    # retained for callers that pass it but is no longer a bypass of the on-disk
    # content check above.
    _ = changed
    return True


def _missing_required_files(
    repo_root: Path, required_files: Sequence[str], *, changed: Container[str] = ()
) -> list[str]:
    """Required deliverables that are neither on disk nor proven changed."""
    return [path for path in required_files if not _required_file_satisfied(repo_root, path, changed=changed)]


def _mutation_tool_stats(trace: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-tool mutation accounting derived from the authoritative trace."""
    called: list[str] = []
    successful = 0
    failed = 0
    for row in trace:
        if not isinstance(row, dict):
            continue
        tool = _trace_tool_name(row)
        if tool not in _MUTATION_TOOLS:
            continue
        called.append(tool)
        status = _trace_status(row)
        changed = _extract_changed_files_from_value(row)
        if changed and status not in _NON_PROGRESS_STATUSES and status not in {"error", "failed", "timeout"}:
            successful += 1
        elif status in {"error", "failed", "timeout"}:
            failed += 1
    return {
        "mutation_tools_called": called,
        "successful_mutations": successful,
        "failed_mutations": failed,
    }


def _forced_mutation_prompt(request: str, target_file: str) -> str:
    """The MUTATION_REQUIRED prompt: analyze the project, then author the file.

    This drives an agentic pass, not a one-shot write. The worker is expected to
    inspect the repository (README, packaging metadata, source layout) so the
    file it produces is specific to *this* project, then finish with a single
    mutation tool call. Placeholder/stub content is explicitly forbidden — there
    is no template fallback behind this, so a stub is a failed deliverable.
    """
    lines = [
        "You must create the requested file by ANALYZING THIS PROJECT, then writing it.",
        "",
        "Work like an agent:",
        "1. Inspect the repository to understand it — read the README, packaging "
        "metadata (pyproject.toml/setup.py/package.json), and the relevant source "
        "files/directories. Use read_file, repo_search, list_files, and ls.",
        "2. Decide what THIS specific file should contain from its name/path "
        "(e.g. an overview summarizes the project; an installation guide gives the "
        "real setup and install commands; usage/commands document the actual CLI).",
        "3. Finish by writing real, project-specific content with exactly one "
        "mutation tool (edit_file, multi_edit_file, apply_patch, create_file, write_file, or delete_file).",
        "",
        "Hard requirements:",
        "- Ground every claim in what you actually found in the repository.",
        "- Do NOT write placeholders, 'TBD', 'TODO', or generic filler. If you do "
        "not know something, inspect the repo to find it.",
        "- The run is not complete until the target file exists with substantive content.",
    ]
    if target_file:
        lines.append(f"Target file: {target_file}")
    lines.append(f"User request: {request}")
    return "\n".join(lines)


def _mutation_fallback_tool_allowed(tool_name: str, *, target_exists: bool, prior_target_evidence: bool) -> bool:
    tool = str(tool_name or "").strip()
    if tool in _MUTATION_FALLBACK_ALLOWED_TOOLS:
        return True
    if tool == "read_file" and target_exists and not prior_target_evidence:
        return True
    if tool in _MUTATION_FALLBACK_BLOCKED_TOOLS:
        return False
    return False


def _looks_like_stub(content: str) -> bool:
    """True if ``content`` is a placeholder the agent left behind (TODO/empty)."""
    text = str(content or "").strip()
    if len(text) < 120:
        return True
    lowered = text.lower()
    return "todo: add" in lowered or "todo: write" in lowered or lowered.count("todo") >= 2


def _stray_misplacements(deliverable: str, changed_files: Sequence[str]) -> list[str]:
    """Top-level files matching a sub-directory deliverable's basename.

    When the user asks for ``docs/01-overview.md`` but the worker writes
    ``01-overview.md`` at the repository root, the root file is a misplacement
    of that deliverable. We only treat *top-level* same-basename files as strays
    so unrelated files in other directories are never touched.
    """
    if "/" not in deliverable:
        return []
    base = Path(deliverable).name
    strays: list[str] = []
    for path in dict.fromkeys(changed_files):
        if path == deliverable or "/" in path:
            continue
        if Path(path).name == base and path not in strays:
            strays.append(path)
    return strays


def _salvage_misplaced_deliverables(
    repo_root: Path,
    deliverables: Sequence[str],
    changed_files: list[str],
    trace: list[dict[str, Any]],
) -> list[str]:
    """Move substantial content written to the wrong path into the required path.

    Returns the stray paths that were relocated (and removed from disk) so the
    caller can drop them from the reported ``changed_files``.
    """
    relocated: list[str] = []
    for deliverable in deliverables:
        target_abs = repo_root / deliverable
        if target_abs.is_file():
            try:
                if not _looks_like_stub(target_abs.read_text(encoding="utf-8", errors="replace")):
                    continue
            except OSError:
                continue
        for stray in _stray_misplacements(deliverable, changed_files):
            stray_abs = repo_root / stray
            if not stray_abs.is_file():
                continue
            try:
                content = stray_abs.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if _looks_like_stub(content):
                continue  # not worth keeping; deterministic generation will replace it
            if target_abs.exists():
                result = safe_write_file(
                    repo_root=repo_root,
                    path=deliverable,
                    content=content,
                    allowed_prefixes=None,
                    force=True,
                )
            else:
                result = safe_create_file(repo_root=repo_root, path=deliverable, content=content, allowed_prefixes=None)
            if not bool(result.get("ok")):
                continue
            try:
                stray_abs.unlink()
            except OSError:
                pass
            trace.append(
                {
                    "tool_name": "write_file",
                    "status": "ok",
                    "path": deliverable,
                    "changed_files": [deliverable],
                    "files_changed": [deliverable],
                    "created_by": "deliverable_relocation",
                }
            )
            changed_files.append(deliverable)
            relocated.append(stray)
            break
    return relocated


def _cleanup_stray_deliverables(
    repo_root: Path,
    deliverables: Sequence[str],
    changed_files: Sequence[str],
    trace: list[dict[str, Any]],
) -> list[str]:
    """Delete leftover wrong-path duplicates once the real deliverable exists."""
    removed: list[str] = []
    for deliverable in deliverables:
        if not _required_file_satisfied(repo_root, deliverable):
            continue
        for stray in _stray_misplacements(deliverable, changed_files):
            stray_abs = repo_root / stray
            if not stray_abs.is_file():
                continue
            try:
                stray_abs.unlink()
            except OSError:
                continue
            trace.append(
                {
                    "tool_name": "cleanup",
                    "status": "ok",
                    "path": stray,
                    "created_by": "stray_deliverable_cleanup",
                }
            )
            removed.append(stray)
    return removed


def _no_op_reason_from_trace(trace: Sequence[dict[str, Any]]) -> str:
    for row in reversed([item for item in trace if isinstance(item, dict)]):
        for key in ("no_op_reason", "noop_reason", "safe_no_op_reason"):
            value = str(row.get(key, "") or "").strip()
            if value:
                return value
    return ""


def _mutation_state_from_trace(trace: Sequence[dict[str, Any]], changed_files: Sequence[str] = ()) -> dict[str, Any]:
    changed = set(str(path).strip().replace("\\", "/").lstrip("./") for path in changed_files if str(path).strip())
    mutation_attempted = False
    mutation_succeeded = False
    blocked_verify = False
    for row in trace:
        if not isinstance(row, dict):
            continue
        tool = _trace_tool_name(row)
        status = _trace_status(row)
        if status == "verify_project_blocked_until_mutation":
            blocked_verify = True
        row_changed = set(_extract_changed_files_from_value(row))
        changed.update(row_changed)
        if tool in _MUTATION_TOOLS:
            mutation_attempted = True
            if row_changed and status not in _NON_PROGRESS_STATUSES and status not in {"error", "failed", "timeout"}:
                mutation_succeeded = True
    if changed and mutation_attempted:
        mutation_succeeded = True
    no_op_reason = _no_op_reason_from_trace(trace)
    return {
        "mutation_attempted": mutation_attempted,
        "mutation_succeeded": mutation_succeeded,
        "changed_files": sorted(changed),
        "no_op_reason": no_op_reason,
        "verify_requires_mutation": blocked_verify,
    }


def _latest_useful_answer(answers: Sequence[str]) -> str:
    for answer in reversed([str(item or "").strip() for item in answers]):
        if answer:
            return answer
    return ""


_VERIFICATION_TOOLS = {"verify", "verify_project", "run_command", "n"}

# Phrases a worker may emit that directly contradict an authoritative execution
# state showing an edit landed. When the trace proves a mutation happened, an
# intermediate worker answer containing any of these is obsolete and must never
# be surfaced as the final answer.
_CONTRADICTION_PATTERNS = (
    "no edit tool",
    "edit tool was",
    "edit tool is",
    "edit tools were",
    "no edit tools",
    "could not edit",
    "couldn't edit",
    "unable to edit",
    "cannot edit",
    "no changes were made",
    "no changes made",
    "no file change",
    "no files changed",
    "no files were changed",
    "did not make any changes",
    "didn't make any changes",
    "no mutation",
    "nothing was changed",
    "nothing changed",
)


def _answer_contradicts_state(answer: str, *, mutated: bool) -> bool:
    """True if ``answer`` claims no edit/change while the trace proves otherwise."""
    if not mutated:
        return False
    low = str(answer or "").lower()
    return any(pattern in low for pattern in _CONTRADICTION_PATTERNS)


def _extract_verification_checks(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Best-effort: pull a list of structured verify checks out of a trace row.

    ``verify_project`` returns ``{"ok", "checks": [...], "summary": {...}}``. That
    payload may live on the row directly or be JSON-serialized into a string field
    (``output_preview``/``result``/``answer``). We look in both places.
    """
    checks = row.get("checks")
    if isinstance(checks, list):
        return [item for item in checks if isinstance(item, dict)]
    for key in ("output_preview", "result", "answer", "output"):
        raw = row.get(key)
        if not isinstance(raw, str) or '"checks"' not in raw:
            continue
        data: Any = None
        try:
            data = json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw, re.S)
            if match:
                try:
                    data = json.loads(match.group(0))
                except Exception:
                    data = None
        if isinstance(data, dict) and isinstance(data.get("checks"), list):
            return [item for item in data["checks"] if isinstance(item, dict)]
    return []


def _verification_summary_from_trace(trace: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Authoritative verify state derived from the trace (not worker prose)."""
    ran = False
    failed = False
    passed_any = False
    failing: list[dict[str, str]] = []
    for row in trace:
        if not isinstance(row, dict):
            continue
        tool = _trace_tool_name(row)
        if tool not in _VERIFICATION_TOOLS:
            continue
        ran = True
        status = _trace_status(row)
        if status == "verify_project_blocked_until_mutation":
            # The verify never actually ran; do not treat it as executed.
            ran = ran and bool(passed_any or failing)
            continue
        checks = _extract_verification_checks(row)
        if checks:
            for chk in checks:
                cstatus = str(chk.get("status", "")).strip().lower()
                if cstatus == "failed":
                    failed = True
                    detail = (
                        chk.get("reason")
                        or chk.get("stderr")
                        or chk.get("error")
                        or chk.get("stdout")
                        or ""
                    )
                    failing.append(
                        {"name": str(chk.get("name") or "check"), "detail": str(detail).strip()}
                    )
                elif cstatus == "passed":
                    passed_any = True
            continue
        if status in {"error", "failed", "timeout"}:
            failed = True
            failing.append(
                {
                    "name": tool,
                    "detail": str(row.get("error") or row.get("output_preview") or status).strip(),
                }
            )
        elif status in {"ok", "success", "passed"}:
            passed_any = True
    return {
        "ran": ran,
        "failed": failed,
        "passed": bool(ran and passed_any and not failed),
        "failing": failing,
    }


def _failed_tool_calls_from_trace(trace: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Hard tool failures in the trace (for surfacing in the final answer)."""
    failures: list[dict[str, str]] = []
    for row in trace:
        if not isinstance(row, dict):
            continue
        status = _trace_status(row)
        if status not in {"error", "failed", "timeout"}:
            continue
        failures.append(
            {
                "tool": _trace_tool_name(row) or "tool",
                "detail": str(row.get("error") or row.get("output_preview") or status).strip(),
            }
        )
    return failures


def _compose_final_answer(
    *,
    mutation_required: bool,
    mutation_state: dict[str, Any],
    changed_files: Sequence[str],
    verification: dict[str, Any],
    run_status: str,
    terminal_reason: str,
    worker_answer: str,
    fallback: str,
    missing_required_files: Sequence[str] = (),
) -> str:
    """Rebuild the final answer from authoritative execution state.

    The last natural-language worker answer is *never* the source of truth: it is
    only appended when it does not contradict what the trace proves happened.
    """
    changed = [str(path).strip() for path in (changed_files or []) if str(path).strip()]
    missing = [str(path).strip() for path in (missing_required_files or []) if str(path).strip()]
    mutated = bool(mutation_state.get("mutation_succeeded")) or bool(changed)
    no_op_reason = str(mutation_state.get("no_op_reason") or "").strip()
    worker_answer = str(worker_answer or "").strip()

    if run_status == "blocked":
        reason = str(terminal_reason or "").strip() or "blocked"
        if reason == "mutation_required_but_missing_files":
            # Partial success: surface what landed and what is still missing so
            # the caller (or a resumed run) knows exactly which files to finish.
            lines = []
            if changed:
                lines.append(f"Applied changes to {len(changed)} file(s):")
                lines.extend(f"- {path}" for path in changed)
                lines.append("")
            lines.append("Required files still missing:")
            lines.extend(f"- {path}" for path in (missing or ["(unknown)"]))
            return "\n".join(lines).strip()
        lines = ["The edit could not be completed."]
        if reason == "mutation_required_but_no_mutation_tool_attempted":
            lines.append(
                "No edit tool (edit_file/multi_edit_file/apply_patch/write_file/create_file/delete_file) was executed, so no changes were made."
            )
        elif reason == "mutation_required_but_no_changed_files":
            lines.append(
                "An edit tool ran but produced no file changes. Retry with a corrected edit payload."
            )
        else:
            lines.append(f"Reason: {reason}")
        return "\n".join(lines)

    if mutated:
        lines: list[str] = []
        if changed:
            lines.append(f"Applied changes to {len(changed)} file(s):")
            lines.extend(f"- {path}" for path in changed)
        else:
            lines.append("Applied changes.")
        if verification.get("ran"):
            if verification.get("failed"):
                lines.append("")
                lines.append("Verification: FAILED")
                for item in verification.get("failing", []):
                    name = str(item.get("name") or "check")
                    detail = str(item.get("detail") or "").strip()
                    lines.append(f"- {name}: {detail}" if detail else f"- {name}")
            else:
                lines.append("")
                lines.append("Verification: passed")
        else:
            lines.append("")
            lines.append("Verification: not run")
        if worker_answer and not _answer_contradicts_state(worker_answer, mutated=True):
            lines.append("")
            lines.append(worker_answer)
        return "\n".join(lines).strip()

    if mutation_required and no_op_reason:
        return f"No file changes were required. Reason: {no_op_reason}"

    # Non-mutating request (read/search/Q&A): the worker answer is authoritative.
    return worker_answer or fallback


__all__ = [
    "ToolsPlan",
    "ToolsPlanStep",
    "ToolsManagerRequest",
    "ToolsManagerBatch",
    "AutoExecuteResult",
]
