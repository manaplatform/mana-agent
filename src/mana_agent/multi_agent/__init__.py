"""Mandatory hierarchical multi-agent execution system."""

from __future__ import annotations

from typing import Any

__all__ = ["ApprovalAgent", "MainAgent", "MainAgentResult"]


def __getattr__(name: str) -> Any:
    if name in {"MainAgent", "MainAgentResult"}:
        from mana_agent.multi_agent.agents.main_agent import MainAgent, MainAgentResult

        return {"MainAgent": MainAgent, "MainAgentResult": MainAgentResult}[name]
    if name == "ApprovalAgent":
        from mana_agent.multi_agent.agents.approval_agent import ApprovalAgent

        return ApprovalAgent
    raise AttributeError(name)
