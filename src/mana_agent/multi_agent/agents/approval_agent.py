"""Approval agent for model-driven human approval and decision gating."""

from __future__ import annotations

from typing import Any

from mana_agent.multi_agent.agents.base_agent import BaseAgent
from mana_agent.multi_agent.core.types import AgentRole

APPROVAL_ALLOWED_TOOLS = [
    "request_user_approval",
    "wait_for_approval",
    "git_status",
    "git_diff",
    "run_command",
]


class ApprovalAgent(BaseAgent):
    """Specialist agent responsible for human approvals, decision gating, and verification."""

    def __init__(
        self,
        agent_id: str = "agent_approval_0001",
        *,
        role: AgentRole = AgentRole.APPROVAL,
        parent_agent_id: str | None = None,
        capabilities: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        mailbox: Any = None,
        taskboard: Any = None,
        message_bus: Any = None,
        inbox_service: Any = None,
        execution_supervisor: Any = None,
        **kwargs,
    ) -> None:
        from mana_agent.multi_agent.registry.capability_registry import DEFAULT_CAPABILITIES

        caps = (
            list(capabilities)
            if capabilities is not None
            else list(DEFAULT_CAPABILITIES.get(role, []))
        )
        tools_list = list(
            allowed_tools if allowed_tools is not None else APPROVAL_ALLOWED_TOOLS
        )
        super().__init__(
            agent_id=agent_id,
            role=role,
            parent_agent_id=parent_agent_id,
            capabilities=caps,
            allowed_tools=tools_list,
            mailbox=mailbox,
            taskboard=taskboard,
            message_bus=message_bus,
            **kwargs,
        )
        self.inbox_service = inbox_service
        self.execution_supervisor = execution_supervisor

    def tools(self) -> list[str]:
        """Return the allowed tools for the approval agent."""
        return list(self.allowed_tools)

    def may_continue(self, result: Any) -> bool:
        """Return True only when the approval verdict is explicitly approved."""
        if result == "approved":
            return True
        if isinstance(result, dict):
            return result.get("verdict") == "approved" and bool(result.get("may_continue", True))
        verdict = getattr(result, "verdict", None)
        if verdict is not None:
            may_cont = getattr(result, "may_continue", True)
            return verdict == "approved" and bool(may_cont)
        return False


__all__ = ["APPROVAL_ALLOWED_TOOLS", "ApprovalAgent"]
