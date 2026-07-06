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
        task.reviewed_by_agent_id = self.agent_id
        self.record_evidence(task_id, "Reviewer approved hierarchy and verification evidence.")
        return True
