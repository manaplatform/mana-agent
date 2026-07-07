from __future__ import annotations

from mana_agent.multi_agent.core.types import RiskLevel, RouteDecision
from mana_agent.multi_agent.routing.agent_decision import AgentDecision, AgentDecisionEngine


class Router:
    def __init__(self, *, llm=None, decision_engine: AgentDecisionEngine | None = None) -> None:  # noqa: ANN001
        self.decision_engine = decision_engine or AgentDecisionEngine(llm=llm)

    def route(self, *, task_id: str, user_request: str) -> RouteDecision:
        agent_decision = self.decision_engine.decide(user_request=user_request)
        kind = self._route_kind(agent_decision)
        if kind == "coding":
            subagents = ["repo_inventory", "docs"] if self._is_docs_inventory_request(user_request) else []
            return RouteDecision(task_id, "coding", "large" if subagents else "medium", ["main", "head_decision", "planner", "coding", "tool", "verifier", "reviewer", "summarizer"], subagents, ["planning", "coding", "tool_execution", "verification", "review", "summarization"], True, True, RiskLevel.MEDIUM, agent_decision.reasoning_summary or "Code mutation or repository edit request.")
        if kind == "analyze":
            return RouteDecision(task_id, "analyze", "medium", ["main", "head_decision", "research", "planner", "reviewer", "summarizer"], ["repo_inventory"], ["repo_search", "repo_read", "planning", "review", "summarization"], True, False, RiskLevel.LOW, agent_decision.reasoning_summary or "Repository analysis request.")
        if kind == "planning":
            return RouteDecision(task_id, "planning", "medium", ["main", "head_decision", "planner", "reviewer", "summarizer"], [], ["planning", "review", "summarization"], True, False, RiskLevel.LOW, agent_decision.reasoning_summary or "Planning request.")
        if kind == "high_risk_tool":
            return RouteDecision(task_id, "high_risk_tool", "medium", ["main", "head_decision", "tool", "verifier", "reviewer", "summarizer"], [], ["decision", "tool_execution"], True, True, RiskLevel.HIGH, agent_decision.reasoning_summary or "High-risk shell or git operation requires approval.")
        if kind == "tool":
            return RouteDecision(task_id, "tool", "medium", ["main", "head_decision", "tool", "verifier", "summarizer"], [], ["tool_execution", "verification", "summarization"], agent_decision.repo_context_needed, True, RiskLevel.MEDIUM, agent_decision.reasoning_summary or "Tool-heavy request.")
        if agent_decision.intent == "web_research":
            external_tools = [
                tool for tool in agent_decision.selected_tools if tool in {"web_search", "github_search"}
            ] or ["web_search"]
            return RouteDecision(task_id, "research", "medium", ["main", "head_decision", "research", "summarizer"], [], [*external_tools, "summarization"], False, False, RiskLevel.LOW, agent_decision.reasoning_summary or "External research request.")
        if agent_decision.intent == "repo_search":
            return RouteDecision(task_id, "repo_search", "medium", ["main", "head_decision", "research", "summarizer"], ["repo_inventory"], ["repo_search", "repo_read", "summarization"], True, False, RiskLevel.LOW, agent_decision.reasoning_summary or "Repository search request.")
        return RouteDecision(task_id, "simple", "simple", ["main", "head_decision", "summarizer"], [], ["conversation", "summarization"], agent_decision.repo_context_needed, False, RiskLevel.LOW, agent_decision.reasoning_summary or "Simple explanation or Q&A request.")

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

    def _is_docs_inventory_request(self, user_request: str) -> bool:
        text = str(user_request or "").lower()
        return "readme" in text or ("architecture" in text and ("docs" in text or "documentation" in text))
