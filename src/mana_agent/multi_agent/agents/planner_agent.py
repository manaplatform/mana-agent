from __future__ import annotations

from mana_agent.multi_agent.agents.base_agent import BaseAgent
from mana_agent.multi_agent.core.types import PlanResult


class PlannerAgent(BaseAgent):
    def plan(self, task_id: str, user_request: str, route_name: str) -> PlanResult:
        commands = ["python3 -m compileall src"]
        if route_name in {"coding", "tool", "high_risk_tool"}:
            commands.append("pytest")
        result = PlanResult(
            task_id=task_id,
            plan_steps=[
                "Inspect the relevant repository context.",
                "Route work to the responsible specialist agents.",
                "Execute approved queue jobs through QueueManager.",
                "Verify outcomes before final summary.",
            ],
            acceptance_criteria=[
                "The request is represented on the TaskBoard.",
                "Important decisions, assumptions, and evidence are recorded.",
                "Mutations, when any, are verified by VerifierAgent.",
            ],
            files_to_inspect=[],
            verification_commands=commands,
            risks=[] if route_name != "coding" else ["Code mutation can affect multiple CLI paths."],
            assumptions=["Existing public command names remain compatible."],
        )
        task = self.taskboard.get_task(task_id)
        task.plan = result.plan_steps
        task.acceptance_criteria = result.acceptance_criteria
        task.verification_commands = result.verification_commands
        for assumption in result.assumptions:
            self.taskboard.add_assumption(task_id, assumption)
        return result
