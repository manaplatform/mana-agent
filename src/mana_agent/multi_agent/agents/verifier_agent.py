from __future__ import annotations

from mana_agent.multi_agent.agents.base_agent import BaseAgent
from mana_agent.multi_agent.core.ids import new_decision_id
from mana_agent.multi_agent.core.types import QueueJobType, VerificationResult
from mana_agent.multi_agent.queue.queue_manager import QueueManager


class VerifierAgent(BaseAgent):
    def __init__(self, *args, queue_manager: QueueManager | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.queue_manager = queue_manager

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

    def execute_verification(self, task_id: str, commands: list[str]) -> VerificationResult:
        if self.queue_manager is None:
            return self.record_failed_verification(
                task_id,
                "Verification blocked: QueueManager unavailable.",
                ["queue_manager_unavailable"],
            )
        executed: list[str] = []
        failures: list[str] = []
        queue_job_ids: list[str] = []
        for command in commands:
            text = str(command or "").strip()
            if not text:
                continue
            job = self.queue_manager.enqueue(
                task_id=task_id,
                requested_by_agent_id=self.agent_id,
                approved_by_agent_id="agent_main_0001",
                job_type=QueueJobType.SHELL,
                payload={"command": text},
                purpose=f"Execute verification command: {text}",
                priority=80,
            )
            queue_job_ids.append(job.job_id)
            ran = self.queue_manager.run_next(worker_agent_id=job.assigned_worker_agent_id)
            if ran is None:
                failures.append(f"{text}: verification job did not run")
                continue
            executed.append(text)
            if ran.status.value != "done":
                failures.append(f"{text}: {ran.error or ran.result_summary or 'failed'}")
        passed = bool(executed) and not failures
        summary = (
            f"Executed {len(executed)} verification command(s) through queue jobs: {', '.join(queue_job_ids)}."
            if passed
            else f"Verification blocked or failed after queue execution: {'; '.join(failures) or 'no commands executed'}"
        )
        result = VerificationResult(
            verification_id=new_decision_id().replace("decision", "verification", 1),
            task_id=task_id,
            verified_by_agent_id=self.agent_id,
            commands_run=executed,
            passed=passed,
            summary=summary,
            failures=failures,
            risks=[] if passed else ["verification_failed_or_blocked"],
        )
        self.taskboard.add_verification_result(task_id, result)
        return result
