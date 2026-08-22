from __future__ import annotations

from typing import Any

from mana_agent.multi_agent.agents.base_agent import BaseAgent
from mana_agent.evals.recorder import record_current


class ReviewerAgent(BaseAgent):
    def verify_runtime_reachability(
        self,
        task_id: str,
        path: list[str],
        *,
        summary: str = "",
        source_references: list[str] | None = None,
        observable_result: str = "",
        verification_source: str = "",
    ) -> bool:
        """Persist a concrete production path with an observable result."""
        try:
            self.taskboard.record_integration_evidence(
                task_id,
                path,
                summary=summary,
                source_references=source_references,
                observable_result=observable_result,
                verification_source=verification_source,
                reviewer=self.agent_id,
            )
        except ValueError as exc:
            self.reject_weak_evidence(task_id, f"INCOMPLETE_FEATURE_WIRING: {exc}")
            return False
        self.record_evidence(task_id, "Reviewer verified runtime path: " + " → ".join(path))
        return True

    def review(self, task_id: str, risk_summary: str) -> None:
        self.record_evidence(task_id, f"Reviewer assessment: {risk_summary}")
        record_current("review.finished", {"task_id": task_id, "reviewer": self.agent_id, "summary": risk_summary})

    def reject_weak_evidence(self, task_id: str, reason: str) -> None:
        self.taskboard.add_blocker(task_id, f"Reviewer rejected weak evidence: {reason}")
        self.record_evidence(task_id, f"Reviewer rejection: {reason}")

    def review_managed_branch(
        self,
        task_id: str,
        *,
        workspace_manager: Any,
        verification_passed: bool | None = None,
    ) -> dict[str, Any]:
        """Review the managed task branch against its recorded base revision.

        Successful reviews produce a merge candidate only; they never merge.
        """

        from mana_agent.multi_agent.worktrees.review import review_task_branch

        task = self.taskboard.get_task(task_id)
        result = review_task_branch(
            workspace_manager,
            task_id,
            reviewer_agent_id=self.agent_id,
            verification_passed=verification_passed,
            hierarchy_ok=not bool(task.hierarchy_violations),
            extra_blockers=list(task.blockers),
        )
        if result.get("approved"):
            task.reviewed_by_agent_id = self.agent_id
            self.record_evidence(task_id, result.get("summary") or "Managed branch review approved.")
        else:
            self.reject_weak_evidence(task_id, result.get("summary") or "Managed branch review rejected.")
        record_current("review.finished", {"task_id": task_id, "reviewer": self.agent_id, "result": result})
        return result

    def review_evidence(self, task_id: str, *, route_name: str, requires_verification: bool) -> bool:
        task = self.taskboard.get_task(task_id)
        if task.hierarchy_violations:
            self.reject_weak_evidence(task_id, "hierarchy violations were recorded")
            return False
        if route_name in {"coding", "tool", "high_risk_tool"} and not task.queue_job_ids:
            self.reject_weak_evidence(task_id, "tool-heavy route has no queue_job_ids")
            return False
        if any(event.get("agent_id") == "main" or str(event.get("agent_id", "")).startswith("agent_main_") for event in task.actual_tool_events):
            self.reject_weak_evidence(task_id, "MainAgent appeared in actual tool execution events")
            return False
        if task.integration_role == "wiring":
            if task.wiring_outcome not in {"mutation_applied", "already_integrated"}:
                self.reject_weak_evidence(task_id, "INCOMPLETE_FEATURE_WIRING: wiring outcome is unproven")
                return False
            if (
                not task.implementation_verified
                or not task.integration_verified
                or not task.runtime_reachability_verified
                or not task.verification_provenance
                or not task.integration_evidence_records
                or not all(
                    record.get("source_references") and record.get("observable_result")
                    for record in task.integration_evidence_records
                )
            ):
                self.reject_weak_evidence(
                    task_id,
                    "INCOMPLETE_FEATURE_WIRING: wiring child lacks complete verified provenance",
                )
                return False
        if requires_verification:
            latest = task.verification_results[-1] if task.verification_results else None
            if latest is None or not latest.passed or not task.verification_queue_job_ids:
                self.reject_weak_evidence(task_id, "verification lacks executed queue job evidence")
                return False
        if task.wiring_required:
            if not task.required_wiring_task_ids:
                self.reject_weak_evidence(task_id, "INCOMPLETE_FEATURE_WIRING: planner supplied no integration task")
                return False
            incomplete = [
                dependency_id for dependency_id in task.required_wiring_task_ids
                if dependency_id not in self.taskboard.tasks
                or self.taskboard.get_task(dependency_id).status.value != "done"
            ]
            if incomplete:
                self.reject_weak_evidence(
                    task_id,
                    "INCOMPLETE_FEATURE_WIRING: integration tasks are incomplete: " + ", ".join(incomplete),
                )
                return False
            if (
                not task.integration_verified
                or not task.runtime_reachability_verified
                or len(task.integration_evidence) < 3
                or not task.integration_evidence_records
                or not all(record.get("source_references") for record in task.integration_evidence_records)
            ):
                self.reject_weak_evidence(
                    task_id,
                    "INCOMPLETE_FEATURE_WIRING: no verified production entrypoint-to-capability path",
                )
                return False
            task.implementation_verified = True
        if route_name == "high_risk_tool" and any(str(item).startswith("git_") for item in task.required_capabilities):
            if not self._git_evidence_is_complete(task_id):
                return False
        task.reviewed_by_agent_id = self.agent_id
        self.record_evidence(task_id, "Reviewer approved hierarchy and verification evidence.")
        return True

    def _git_evidence_is_complete(self, task_id: str) -> bool:
        task = self.taskboard.get_task(task_id)
        commands = [_git_event_command(event) for event in task.actual_tool_events]
        blockers = " ".join(task.blockers).lower()
        if not any(command[:3] == ["status", "--short", "--branch"] for command in commands):
            self.reject_weak_evidence(task_id, "git task lacks git status --short --branch tool evidence")
            return False
        if not any(command[:2] == ["diff", "--stat"] for command in commands):
            self.reject_weak_evidence(task_id, "git task lacks git diff --stat tool evidence")
            return False
        if "git_commit" in task.required_capabilities:
            committed = any(command and command[0] == "commit" for command in commands)
            blocked = "commit" in blockers or "no changes to commit" in blockers
            if not committed and not blocked:
                self.reject_weak_evidence(task_id, "requested commit but no git commit evidence or blocker was recorded")
                return False
        if "git_push" in task.required_capabilities:
            pushed = any(command and command[0] == "push" for command in commands)
            blocked = "push" in blockers or "remote" in blockers or "branch" in blockers or "diverged" in blockers or "behind" in blockers
            if not pushed and not blocked:
                self.reject_weak_evidence(task_id, "requested push but no git push evidence or blocker was recorded")
                return False
        return True


def _git_event_command(event: dict) -> list[str]:
    if str(event.get("tool_name") or "") != "git":
        return []
    args = event.get("tool_args") if isinstance(event.get("tool_args"), dict) else {}
    nested = args.get("args") if isinstance(args.get("args"), dict) else {}
    raw = nested.get("args") if isinstance(nested, dict) else None
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []
