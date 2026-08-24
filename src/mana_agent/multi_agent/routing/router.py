from __future__ import annotations

from dataclasses import dataclass

from mana_agent.multi_agent.core.types import RiskLevel, RouteDecision
from mana_agent.multi_agent.routing.agent_decision import AgentDecision, AgentDecisionEngine


class RoutingDecisionError(RuntimeError):
    """Raised when the required model routing decision cannot be executed safely."""


@dataclass(frozen=True, slots=True)
class RoutingContext:
    """Transport metadata kept separate from the semantic user request."""

    semantic_text: str
    command_hint: str = ""


class Router:
    def __init__(self, *, llm=None, decision_engine: AgentDecisionEngine | None = None) -> None:  # noqa: ANN001
        self.decision_engine = decision_engine or AgentDecisionEngine(llm=llm)

    def route(self, *, task_id: str, user_request: str, command_hint: str = "") -> RouteDecision:
        context = RoutingContext(semantic_text=user_request, command_hint=command_hint)
        agent_decision = self.decision_engine.decide(
            user_request=context.semantic_text,
            command_hint=context.command_hint,
        )
        if not agent_decision.verifier_passed:
            raise RoutingDecisionError(
                "Model decision failed: agent_route. No route was executed. "
                f"Reason: {agent_decision.verifier_summary}"
            )
        kind = self._route_kind(agent_decision)
        if kind == "coding":
            subagents = list(agent_decision.required_subagents)
            return RouteDecision(task_id, "coding", "large" if subagents else "medium", ["main", "head_decision", "planner", "coding", "tool", "verifier", "reviewer", "summarizer"], subagents, ["planning", "coding", "tool_execution", "verification", "review", "summarization"], True, True, RiskLevel.MEDIUM, agent_decision.reasoning_summary or "Code mutation or repository edit request.", agent_decision.runtime_capability_change)
        if kind == "analyze":
            return RouteDecision(task_id, "analyze", "medium", ["main", "head_decision", "research", "planner", "reviewer", "summarizer"], ["repo_inventory"], ["repo_search", "repo_read", "planning", "review", "summarization"], True, False, RiskLevel.LOW, agent_decision.reasoning_summary or "Repository analysis request.", agent_decision.runtime_capability_change)
        if kind == "planning":
            return RouteDecision(task_id, "planning", "medium", ["main", "head_decision", "planner", "reviewer", "summarizer"], [], ["planning", "review", "summarization"], True, False, RiskLevel.LOW, agent_decision.reasoning_summary or "Planning request.", agent_decision.runtime_capability_change)
        if kind == "high_risk_tool":
            return RouteDecision(task_id, "high_risk_tool", "medium", ["main", "head_decision", "tool", "verifier", "reviewer", "summarizer"], [], ["decision", "tool_execution"], True, True, RiskLevel.HIGH, agent_decision.reasoning_summary or "High-risk shell or git operation requires approval.", agent_decision.runtime_capability_change)
        if kind == "tool":
            return RouteDecision(task_id, "tool", "medium", ["main", "head_decision", "tool", "verifier", "summarizer"], [], ["tool_execution", "verification", "summarization"], agent_decision.repo_context_needed, True, RiskLevel.MEDIUM, agent_decision.reasoning_summary or "Tool-heavy request.", agent_decision.runtime_capability_change)
        if agent_decision.intent == "web_research":
            external_tools = [
                tool for tool in agent_decision.selected_tools if tool in {"web_search", "github_search"}
            ]
            return RouteDecision(task_id, "research", "medium", ["main", "head_decision", "research", "summarizer"], [], [*external_tools, "summarization"], False, False, RiskLevel.LOW, agent_decision.reasoning_summary or "External research request.", agent_decision.runtime_capability_change)
        if agent_decision.intent == "repo_search":
            return RouteDecision(task_id, "repo_search", "medium", ["main", "head_decision", "research", "summarizer"], ["repo_inventory"], ["repo_search", "repo_read", "summarization"], True, False, RiskLevel.LOW, agent_decision.reasoning_summary or "Repository search request.", agent_decision.runtime_capability_change)
        return RouteDecision(task_id, "simple", "simple", ["main", "head_decision", "summarizer"], [], ["conversation", "summarization"], agent_decision.repo_context_needed, False, RiskLevel.LOW, agent_decision.reasoning_summary or "Simple explanation or Q&A request.", agent_decision.runtime_capability_change)

    @staticmethod
    def _route_kind(decision: AgentDecision) -> str:
        if decision.intent == "edit" or decision.code_editing_needed:
            return "coding"
        if decision.intent == "analyze":
            return "analyze"
        if decision.intent == "plan":
            return "planning"
        if decision.intent == "high_risk_tool":
            return "high_risk_tool"
        if decision.intent in {"tool", "verify"}:
            return "tool"
        return "simple"
