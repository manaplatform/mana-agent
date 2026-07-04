from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from mana_agent.agent.evidence_queue import EvidenceQueue
from mana_agent.agent.task_classifier import TaskDecision


class AgentRunPhase(str, Enum):
    RECEIVED = "RECEIVED"
    CLASSIFIED = "CLASSIFIED"
    PLANNED = "PLANNED"
    GATHERING_EVIDENCE = "GATHERING_EVIDENCE"
    READY_TO_MUTATE = "READY_TO_MUTATE"
    MUTATING = "MUTATING"
    VERIFYING = "VERIFYING"
    READY_TO_FINALIZE = "READY_TO_FINALIZE"
    DONE = "DONE"
    FAILED_RECOVERABLE = "FAILED_RECOVERABLE"
    FAILED_FINAL = "FAILED_FINAL"


ExecutionDecision = Literal[
    "execute_next_tool",
    "skip_tool",
    "stop_discovery",
    "start_mutation",
    "run_verification",
    "ask_user",
    "finalize",
    "abort_due_to_error",
]


@dataclass(slots=True)
class AgentRunState:
    phase: AgentRunPhase = AgentRunPhase.RECEIVED
    files_read: set[str] = field(default_factory=set)
    changed_files: set[str] = field(default_factory=set)
    failed_tool_calls: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    verification_status: str = ""
    loop_counters: dict[str, int] = field(default_factory=dict)
    evidence_sufficient: bool = False
    mutation_succeeded: bool = False


@dataclass(frozen=True, slots=True)
class GateResult:
    decision: ExecutionDecision
    reason: str
    next_action: str = ""
    confidence: float = 0.8
    warnings: tuple[str, ...] = ()

    def trace_row(self, *, changed_files: list[str] | None = None) -> dict[str, Any]:
        return {
            "layer": "evaluation_gate",
            "decision": self.decision,
            "reason": self.reason,
            "next_action": self.next_action,
            "confidence": self.confidence,
            "changed_files": list(changed_files or []),
            "warnings": list(self.warnings),
        }


class EvaluationGate:
    def __init__(self, *, repo_root: str | Path, max_repeated_tool: int = 1, max_repeated_error: int = 1) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.max_repeated_tool = max(1, int(max_repeated_tool))
        self.max_repeated_error = max(1, int(max_repeated_error))

    def target_section_found(self, decision: TaskDecision, *, path: str) -> bool:
        if not decision.target_sections:
            return True
        target = self.repo_root / path
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        headings = {
            " ".join(line.lstrip("#").strip().split()).lower()
            for line in text.splitlines()
            if line.lstrip().startswith("#")
        }
        return any(section.lower() in headings for section in decision.target_sections)

    def evidence_is_sufficient(
        self,
        *,
        decision: TaskDecision,
        state: AgentRunState,
    ) -> bool:
        if decision.scope == "single_file_section" and decision.target_files:
            target = decision.target_files[0]
            return target in state.files_read and self.target_section_found(decision, path=target)
        if decision.scope == "single_file" and decision.target_files:
            return decision.target_files[0] in state.files_read
        return False

    def after_tool(
        self,
        *,
        decision: TaskDecision,
        state: AgentRunState,
        evidence_queue: EvidenceQueue,
        tool_name: str,
        ok: bool,
        files_read: list[str],
        changed_files: list[str],
        error: str = "",
    ) -> GateResult:
        key = f"{tool_name}:{','.join(files_read or changed_files)}"
        state.loop_counters[key] = state.loop_counters.get(key, 0) + 1
        state.files_read.update(files_read)
        state.changed_files.update(changed_files)
        if error:
            state.failed_tool_calls.append(error)
        if not ok and error and state.failed_tool_calls.count(error) > self.max_repeated_error:
            state.phase = AgentRunPhase.FAILED_RECOVERABLE
            return GateResult("abort_due_to_error", f"repeated tool error stopped: {error}", confidence=0.9)
        if state.loop_counters[key] > self.max_repeated_tool and tool_name:
            return GateResult("skip_tool", f"repeated tool call stopped: {tool_name}", confidence=0.9)
        for path in files_read:
            evidence_queue.mark_done("read", path, evidence={"source": "tool"})
        state.evidence_sufficient = self.evidence_is_sufficient(decision=decision, state=state)
        if state.evidence_sufficient and decision.needs_mutation:
            skipped = evidence_queue.skip_where(
                lambda item: item.action_type in {"search", "read"} and item.target not in decision.target_files,
                reason="target evidence is sufficient",
            )
            state.phase = AgentRunPhase.READY_TO_MUTATE
            return GateResult(
                "start_mutation",
                "target file was read and requested section exists; unrelated discovery is optional",
                next_action="apply_patch",
                confidence=0.91,
                warnings=(f"skipped_pending_discovery:{skipped}",) if skipped else (),
            )
        if changed_files and decision.needs_verification:
            state.phase = AgentRunPhase.VERIFYING
            return GateResult("run_verification", "mutation changed files; verification is now possible")
        if changed_files:
            state.phase = AgentRunPhase.READY_TO_FINALIZE
            return GateResult("finalize", "mutation changed files and no verification was required")
        return GateResult("execute_next_tool", "continue with the next approved tool")

    def before_tool(
        self,
        *,
        decision: TaskDecision,
        state: AgentRunState,
        tool_name: str,
        target: str = "",
    ) -> GateResult:
        if state.evidence_sufficient and tool_name in {"repo_search", "repo_batch_search", "semantic_search", "list_files"}:
            return GateResult("skip_tool", "evidence is already sufficient; broad discovery is skipped")
        if (
            state.evidence_sufficient
            and tool_name == "read_file"
            and decision.target_files
            and target not in set(decision.target_files)
        ):
            return GateResult("skip_tool", "target evidence is already sufficient; unrelated read is skipped")
        return GateResult("execute_next_tool", "tool call is necessary for the current state")


__all__ = ["AgentRunPhase", "AgentRunState", "EvaluationGate", "ExecutionDecision", "GateResult"]
