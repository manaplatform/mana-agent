from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from mana_agent.agent.evaluation_gate import AgentRunPhase, AgentRunState, EvaluationGate, GateResult
from mana_agent.agent.evidence_queue import EvidenceQueue, EvidenceQueueItem
from mana_agent.agent.task_classifier import TaskDecision, classify_task
from mana_agent.agent.verification_planner import VerificationDecision, plan_verification


@dataclass(frozen=True, slots=True)
class TaskPlan:
    decision: TaskDecision
    evidence_queue: EvidenceQueue
    trace: tuple[dict[str, Any], ...] = ()


class PlanBuilder:
    def build(self, decision: TaskDecision) -> TaskPlan:
        queue = EvidenceQueue()
        if decision.needs_file_read:
            for path in decision.target_files:
                queue.add(
                    EvidenceQueueItem(
                        action_type="read",
                        target=path,
                        reason="explicit target file named by the user",
                        priority=10,
                        required=True,
                    )
                )
        if decision.needs_repo_search:
            queue.add(
                EvidenceQueueItem(
                    action_type="search",
                    target="repo",
                    reason="no explicit target file was available",
                    priority=20,
                    required=True,
                )
            )
        if decision.needs_mutation:
            queue.add(
                EvidenceQueueItem(
                    action_type="patch",
                    target=",".join(decision.target_files) or "resolved target",
                    reason="mutation is required after sufficient evidence",
                    priority=80,
                    required=True,
                )
            )
        if decision.needs_verification:
            queue.add(
                EvidenceQueueItem(
                    action_type="verify",
                    target="changed files",
                    reason="verify the requested mutation",
                    priority=90,
                    required=True,
                )
            )
        trace = (
            {
                "layer": "plan_builder",
                "decision": "minimal_plan_built",
                "reason": "explicit targets are prioritized before repository-wide discovery",
                "next_action": "read_file" if decision.needs_file_read else "repo_search",
                "confidence": decision.confidence,
            },
            queue.trace_row(),
        )
        return TaskPlan(decision=decision, evidence_queue=queue, trace=trace)


@dataclass(slots=True)
class CriticDecision:
    progressed: bool
    enough_evidence: bool
    should_finalize: bool
    reason: str

    def trace_row(self) -> dict[str, Any]:
        return {
            "layer": "post_tool_critic",
            "decision": "finalize" if self.should_finalize else "continue",
            "reason": self.reason,
            "progressed": self.progressed,
            "enough_evidence": self.enough_evidence,
        }


class PostToolCritic:
    def review(self, *, gate: GateResult, state: AgentRunState) -> CriticDecision:
        should_finalize = gate.decision == "finalize" or state.phase == AgentRunPhase.DONE
        progressed = gate.decision in {"start_mutation", "run_verification", "finalize", "execute_next_tool"}
        reason = gate.reason
        return CriticDecision(
            progressed=progressed,
            enough_evidence=state.evidence_sufficient,
            should_finalize=should_finalize,
            reason=reason,
        )


@dataclass(slots=True)
class AgentOrchestrator:
    repo_root: Path
    decision: TaskDecision
    plan: TaskPlan
    state: AgentRunState = field(default_factory=AgentRunState)
    gate: EvaluationGate = field(init=False)
    critic: PostToolCritic = field(default_factory=PostToolCritic)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve()
        self.gate = EvaluationGate(repo_root=self.repo_root)
        self.state.phase = AgentRunPhase.PLANNED
        self.trace.append(self.decision.as_trace_row())
        self.trace.extend(self.plan.trace)

    @classmethod
    def start(
        cls,
        request: str,
        *,
        repo_root: str | Path,
        target_files: Sequence[str] = (),
        requires_edit: bool | None = None,
    ) -> "AgentOrchestrator":
        root = Path(repo_root).resolve()
        decision = classify_task(request, repo_root=root, target_files=target_files, requires_edit=requires_edit)
        plan = PlanBuilder().build(decision)
        return cls(repo_root=root, decision=decision, plan=plan)

    def before_tool(self, *, tool_name: str, target: str = "") -> GateResult:
        result = self.gate.before_tool(decision=self.decision, state=self.state, tool_name=tool_name, target=target)
        self.trace.append(result.trace_row(changed_files=sorted(self.state.changed_files)))
        return result

    def after_tool(
        self,
        *,
        tool_name: str,
        ok: bool,
        files_read: list[str],
        changed_files: list[str],
        error: str = "",
    ) -> GateResult:
        result = self.gate.after_tool(
            decision=self.decision,
            state=self.state,
            evidence_queue=self.plan.evidence_queue,
            tool_name=tool_name,
            ok=ok,
            files_read=files_read,
            changed_files=changed_files,
            error=error,
        )
        self.trace.append(result.trace_row(changed_files=sorted(self.state.changed_files)))
        self.trace.append(self.critic.review(gate=result, state=self.state).trace_row())
        return result

    def verification_decision(self, *, changed_files: Sequence[str], core_agent_change: bool = False) -> VerificationDecision:
        decision = plan_verification(changed_files=changed_files, core_agent_change=core_agent_change)
        self.trace.append(decision.trace_row())
        return decision

    def finalize_trace(self) -> None:
        self.trace.append(
            {
                "layer": "final_response_builder",
                "decision": "finalize",
                "reason": "authoritative queue state is ready for the final response",
                "changed_files": sorted(self.state.changed_files),
                "warnings": list(self.state.warnings),
            }
        )


__all__ = ["AgentOrchestrator", "AgentRunState", "PlanBuilder", "PostToolCritic", "TaskPlan"]
