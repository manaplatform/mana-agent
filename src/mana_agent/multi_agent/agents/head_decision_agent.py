from __future__ import annotations

from mana_agent.multi_agent.agents.base_agent import BaseAgent
from mana_agent.multi_agent.communication.decision_room import DecisionRoom
from mana_agent.multi_agent.core.types import MessageType, RouteDecision, TaskStatus


class HeadDecisionAgent(BaseAgent):
    def decide(self, task_id: str, route: RouteDecision, decision_room: DecisionRoom) -> None:
        task = self.taskboard.get_task(task_id)
        task.required_capabilities = list(route.required_capabilities)
        task.risk_level = route.risk_level
        self.taskboard.add_evidence(task_id, f"Route `{route.route_name}` selected for {route.task_size} task.")
        if route.required_subagents:
            self.taskboard.add_evidence(task_id, f"Required subagents: {', '.join(route.required_subagents)}.")
        if route.requires_discussion:
            self.taskboard.update_status(task_id, TaskStatus.DISCUSSING, reason=route.reason_summary)
            discussion = decision_room.open_discussion(
                task_id,
                f"Route decision: {route.route_name}",
                route.required_agents,
                created_by_agent_id=self.agent_id,
            )
            decision_room.post_message(
                discussion.discussion_id,
                task_id=task_id,
                from_agent_id=self.agent_id,
                to_agent_id=None,
                message_type=MessageType.PROPOSAL,
                content=f"Route `{route.route_name}` selected. {route.reason_summary}",
            )
            decision_room.close_with_decision(
                task_id=task_id,
                discussion_id=discussion.discussion_id,
                made_by_agent_id=self.agent_id,
                summary=f"Use {route.route_name} multi-agent route.",
                rationale_summary=route.reason_summary,
                selected_route=route.route_name,
                assigned_agent_ids=route.required_agents,
                required_verification=["VerifierAgent required"] if route.requires_verification else [],
                risks=[route.risk_level.value],
                assumptions=["No single-agent fallback is allowed."],
            )
        else:
            decision_room.close_with_decision(
                task_id=task_id,
                discussion_id=None,
                made_by_agent_id=self.agent_id,
                summary=f"Use {route.route_name} multi-agent route.",
                rationale_summary=route.reason_summary,
                selected_route=route.route_name,
                assigned_agent_ids=route.required_agents,
                required_verification=[],
            )
        self.taskboard.update_status(task_id, TaskStatus.ROUTED, reason=route.reason_summary)
