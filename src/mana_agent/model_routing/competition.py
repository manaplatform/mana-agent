from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mana_agent.model_routing.models import RoutingDecision, RoutingFailure


@dataclass(frozen=True, slots=True)
class CandidateWorkspace:
    candidate_id: str
    root: Path
    model: str


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    candidate_id: str
    model: str
    diff: str
    checks: tuple[dict[str, Any], ...]
    diagnostics: tuple[str, ...] = ()
    changed_files: tuple[str, ...] = ()
    cost: float = 0.0
    latency_seconds: float = 0.0

    def normalized(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model": self.model,
            "diff": self.diff,
            "checks": [
                {
                    "name": str(item.get("name") or ""),
                    "passed": bool(item.get("passed")),
                    "exit_code": item.get("exit_code"),
                    "diagnostic": str(item.get("diagnostic") or "")[:4000],
                }
                for item in self.checks
            ],
            "diagnostics": list(self.diagnostics),
            "changed_files": list(self.changed_files),
            "patch_bytes": len(self.diff.encode("utf-8")),
            "cost": self.cost,
            "latency_seconds": self.latency_seconds,
        }


@dataclass(frozen=True, slots=True)
class CompetitionJudgment:
    winner_id: str
    scores: dict[str, dict[str, float]]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompetitionResult:
    winner: CandidateEvidence
    judgment: CompetitionJudgment
    cleaned_candidates: tuple[str, ...]


class CandidateExecutor(Protocol):
    active_repository_root: Path

    def create_isolated(self, *, candidate_id: str, model: str) -> CandidateWorkspace: ...
    def execute(self, workspace: CandidateWorkspace) -> CandidateEvidence: ...
    def promote(self, workspace: CandidateWorkspace, evidence: CandidateEvidence) -> None: ...
    def cleanup(self, workspace: CandidateWorkspace) -> None: ...


class EvidenceJudge(Protocol):
    def judge(self, evidence: tuple[dict[str, Any], ...], *, verifier_model: str | None) -> CompetitionJudgment: ...


class CandidateCompetition:
    """Runs author configurations without ever granting access to the active checkout."""

    def __init__(self, executor: CandidateExecutor, judge: EvidenceJudge) -> None:
        self.executor = executor
        self.judge = judge

    def run(self, decision: RoutingDecision) -> CompetitionResult:
        if not decision.candidate_competition or len(decision.competition_candidates) < 2:
            raise RoutingFailure("Candidate competition was not selected by the validated routing decision.")
        active = self.executor.active_repository_root.resolve()
        workspaces: list[CandidateWorkspace] = []
        evidence_rows: list[CandidateEvidence] = []
        cleaned: list[str] = []
        try:
            for index, model in enumerate(decision.competition_candidates):
                workspace = self.executor.create_isolated(candidate_id=f"candidate-{index + 1}", model=model)
                root = workspace.root.resolve()
                if root == active or active in root.parents or root in active.parents:
                    raise RoutingFailure(f"Candidate {workspace.candidate_id} was not isolated from the active repository.")
                if any(root == item.root.resolve() for item in workspaces):
                    raise RoutingFailure("Candidate workspaces must be distinct.")
                workspaces.append(workspace)
                evidence = self.executor.execute(workspace)
                if not evidence.diff.strip() or not evidence.checks:
                    raise RoutingFailure(f"Candidate {workspace.candidate_id} lacks diff or verification evidence; judging claims alone is forbidden.")
                evidence_rows.append(evidence)
            judgment = self.judge.judge(tuple(item.normalized() for item in evidence_rows), verifier_model=decision.verifier_model)
            by_id = {item.candidate_id: item for item in evidence_rows}
            winner = by_id.get(judgment.winner_id)
            if winner is None or winner.candidate_id not in judgment.scores:
                raise RoutingFailure("Verifier returned an invalid competition judgment. No candidate was promoted.")
            required_criteria = {
                "correctness", "test_results", "regression_risk", "security", "scope_discipline",
                "maintainability", "repository_conventions", "patch_size", "verification_completeness",
                "cost_latency",
            }
            for candidate_id in by_id:
                missing = required_criteria - set(judgment.scores.get(candidate_id, {}))
                if missing:
                    raise RoutingFailure(
                        f"Verifier judgment for {candidate_id} omitted required evidence criteria: {', '.join(sorted(missing))}."
                    )
            winner_workspace = next(item for item in workspaces if item.candidate_id == winner.candidate_id)
            for workspace in workspaces:
                if workspace.candidate_id != winner.candidate_id:
                    self.executor.cleanup(workspace)
                    cleaned.append(workspace.candidate_id)
            self.executor.promote(winner_workspace, winner)
            return CompetitionResult(winner, judgment, tuple(cleaned))
        except Exception:
            for workspace in workspaces:
                if workspace.candidate_id not in cleaned:
                    self.executor.cleanup(workspace)
            raise
