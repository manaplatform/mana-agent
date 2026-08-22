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
            implementation_targets=["Selected implementation files and their downstream callers."],
            wiring_targets=["Production construction, registration, routing, and entrypoint wiring."],
            registration_points=["Relevant registry, factory, dependency-injection, or router."],
            runtime_entrypoints=["A production CLI, API, gateway, lifecycle, or supervisor entrypoint."],
            configuration_targets=["Configuration that enables or selects the capability."],
            export_targets=["Public exports required by the production import path."],
            integration_verification=["Reviewer traces a production entrypoint to the implementation and records the observable result."],
            wiring_required=route_name in {"coding", "tool", "high_risk_tool"},
            wiring_reason=(
                "The routed task may change runtime behavior; construction, registration, and reachability must be verified."
                if route_name in {"coding", "tool", "high_risk_tool"}
                else "The routed task is not an implementation route with a runtime capability."
            ),
        )
        task = self.taskboard.get_task(task_id)
        task.plan = result.plan_steps
        task.acceptance_criteria = result.acceptance_criteria
        task.verification_commands = result.verification_commands
        for name in (
            "implementation_targets", "wiring_targets", "registration_points",
            "runtime_entrypoints", "configuration_targets", "export_targets",
            "integration_verification", "wiring_required", "wiring_reason",
        ):
            setattr(task, name, getattr(result, name))
        if result.wiring_required:
            child = self.taskboard.create_child_task(
                task_id,
                title="Wire capability into the production runtime and verify reachability",
                user_request="Implement and verify the planner integration contract for this capability.",
                owner_agent_id=self.agent_id,
                acceptance_criteria=list(result.integration_verification),
                plan=[
                    "Discover upstream callers and downstream integration points.",
                    "Implement missing production wiring.",
                    "Record runtime reachability evidence.",
                ],
                integration_role="wiring",
            )
        for assumption in result.assumptions:
            self.taskboard.add_assumption(task_id, assumption)
        return result
