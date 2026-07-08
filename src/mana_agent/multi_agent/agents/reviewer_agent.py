from mana_agent.multi_agent.agents.base_agent import BaseAgent


class ReviewerAgent(BaseAgent):
    def review(self, task_id: str, risk_summary: str) -> None:
        self.record_evidence(task_id, f"Reviewer assessment: {risk_summary}")

    def reject_weak_evidence(self, task_id: str, reason: str) -> None:
        self.taskboard.add_blocker(task_id, f"Reviewer rejected weak evidence: {reason}")
        self.record_evidence(task_id, f"Reviewer rejection: {reason}")

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
        if requires_verification:
            latest = task.verification_results[-1] if task.verification_results else None
            if latest is None or not latest.passed or not task.verification_queue_job_ids:
                self.reject_weak_evidence(task_id, "verification lacks executed queue job evidence")
                return False
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
