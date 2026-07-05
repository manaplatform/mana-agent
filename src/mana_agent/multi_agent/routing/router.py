from __future__ import annotations

from mana_agent.multi_agent.core.types import RiskLevel, RouteDecision
from mana_agent.multi_agent.routing.policies import classify_request


class Router:
    def route(self, *, task_id: str, user_request: str) -> RouteDecision:
        kind = classify_request(user_request)
        if kind == "coding":
            subagents = ["repo_inventory", "docs"] if self._is_docs_inventory_request(user_request) else []
            return RouteDecision(task_id, "coding", "large" if subagents else "medium", ["main", "head_decision", "planner", "coding", "tool", "verifier", "reviewer", "summarizer"], subagents, ["planning", "coding", "tool_execution", "verification", "review", "summarization"], True, True, RiskLevel.MEDIUM, "Code mutation or repository edit request.")
        if kind == "analyze":
            return RouteDecision(task_id, "analyze", "medium", ["main", "head_decision", "research", "planner", "reviewer", "summarizer"], ["repo_inventory"], ["repo_search", "repo_read", "planning", "review", "summarization"], True, False, RiskLevel.LOW, "Repository analysis request.")
        if kind == "planning":
            return RouteDecision(task_id, "planning", "medium", ["main", "head_decision", "planner", "reviewer", "summarizer"], [], ["planning", "review", "summarization"], True, False, RiskLevel.LOW, "Planning request.")
        if kind == "high_risk_tool":
            return RouteDecision(task_id, "high_risk_tool", "medium", ["main", "head_decision", "tool", "verifier", "reviewer", "summarizer"], [], ["decision", "tool_execution"], True, True, RiskLevel.HIGH, "High-risk shell or git operation requires approval.")
        if kind == "tool":
            return RouteDecision(task_id, "tool", "medium", ["main", "head_decision", "tool", "verifier", "summarizer"], [], ["tool_execution", "verification", "summarization"], False, True, RiskLevel.MEDIUM, "Tool-heavy request.")
        return RouteDecision(task_id, "simple", "simple", ["main", "head_decision", "summarizer"], [], ["conversation", "summarization"], False, False, RiskLevel.LOW, "Simple explanation or Q&A request.")

    def _is_docs_inventory_request(self, user_request: str) -> bool:
        text = str(user_request or "").lower()
        return "readme" in text or ("architecture" in text and ("docs" in text or "documentation" in text))
