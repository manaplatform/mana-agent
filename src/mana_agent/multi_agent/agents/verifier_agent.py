from __future__ import annotations

from mana_agent.multi_agent.agents.base_agent import BaseAgent
from mana_agent.multi_agent.core.ids import new_decision_id
from mana_agent.multi_agent.core.types import VerificationResult


class VerifierAgent(BaseAgent):
    def record_failed_verification(self, task_id: str, summary: str, failures: list[str] | None = None) -> VerificationResult:
        result = VerificationResult(
            verification_id=new_decision_id().replace("decision", "verification", 1),
            task_id=task_id,
            verified_by_agent_id=self.agent_id,
            commands_run=[],
            passed=False,
            summary=summary,
            failures=failures or [summary],
        )
        self.taskboard.add_verification_result(task_id, result)
        return result

    def verify_no_mutation(self, task_id: str, commands: list[str]) -> VerificationResult:
        result = VerificationResult(
            verification_id=new_decision_id().replace("decision", "verification", 1),
            task_id=task_id,
            verified_by_agent_id=self.agent_id,
            commands_run=commands,
            passed=False,
            summary="Verification plan recorded; commands have not been executed by this verifier.",
            risks=["planned_verification_not_executed"],
        )
        self.taskboard.add_verification_result(task_id, result)
        return result
