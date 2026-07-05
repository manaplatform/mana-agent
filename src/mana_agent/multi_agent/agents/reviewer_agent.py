from mana_agent.multi_agent.agents.base_agent import BaseAgent


class ReviewerAgent(BaseAgent):
    def review(self, task_id: str, risk_summary: str) -> None:
        self.record_evidence(task_id, f"Reviewer assessment: {risk_summary}")

    def reject_weak_evidence(self, task_id: str, reason: str) -> None:
        self.taskboard.add_blocker(task_id, f"Reviewer rejected weak evidence: {reason}")
        self.record_evidence(task_id, f"Reviewer rejection: {reason}")
