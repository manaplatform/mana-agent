from __future__ import annotations

from mana_agent.multi_agent.agents.base_agent import BaseAgent


class SummarizerAgent(BaseAgent):
    def summarize(self, task_id: str) -> str:
        task = self.taskboard.get_task(task_id)
        files = ", ".join(task.files_touched) if task.files_touched else "none recorded"
        verification = task.verification_results[-1].summary if task.verification_results else "not run yet"
        route = next((item for item in task.evidence if item.startswith("Route `")), "Route `unknown`")
        summary = (
            f"{route}. Agents: {', '.join(task.assigned_agent_ids) or 'none'}. "
            f"Subagents: {', '.join(task.assigned_subagent_ids) or 'none'}. "
            f"Queue jobs: {', '.join(task.queue_job_ids) or 'none'}. "
            f"Files touched: {files}. Verification: {verification}. "
            f"Budget used: {task.budget_used_tokens}/{task.budget_reserved_tokens} tokens."
        )
        self.record_evidence(task_id, "summary.created")
        return summary
