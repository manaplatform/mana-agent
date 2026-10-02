"""Tests for ApprovalAgent, AgentRole.APPROVAL, and approval agent capabilities."""

from __future__ import annotations

from mana_agent.human_inbox.approval_tools import ApprovalWaitResult
from mana_agent.multi_agent.agents.approval_agent import (
    APPROVAL_ALLOWED_TOOLS,
    ApprovalAgent,
)
from mana_agent.multi_agent.core.types import AgentRole
from mana_agent.multi_agent.registry.capability_registry import DEFAULT_CAPABILITIES
from mana_agent.multi_agent.runtime.model_levels import (
    MODEL_LEVEL_3_HIGH_REASONING,
    model_level_for_role,
)


def test_approval_role_and_capabilities() -> None:
    assert AgentRole.APPROVAL.value == "approval"
    caps = DEFAULT_CAPABILITIES[AgentRole.APPROVAL]
    assert caps == [
        "transactional_actions",
        "execution",
        "human_inbox",
        "approval_request",
        "approval_wait",
    ]


def test_approval_model_level() -> None:
    assignment = model_level_for_role(AgentRole.APPROVAL)
    assert assignment.role == AgentRole.APPROVAL
    assert assignment.env_var == "MANA_MODEL_APPROVAL"
    assert assignment.model_level == MODEL_LEVEL_3_HIGH_REASONING


def test_approval_agent_tools() -> None:
    agent = ApprovalAgent(agent_id="test_approval_agent")
    assert agent.role == AgentRole.APPROVAL
    assert agent.tools() == APPROVAL_ALLOWED_TOOLS
    assert agent.tools() == [
        "approval_request",
        "request_user_approval",
        "wait_for_approval",
        "git_status",
        "git_diff",
        "run_command",
    ]


def test_approval_agent_may_continue() -> None:
    agent = ApprovalAgent(agent_id="test_approval_agent")

    # Approved verdicts
    approved_res = ApprovalWaitResult(
        verdict="approved",
        may_continue=True,
        inbox_item_id="inbox_1",
    )
    assert agent.may_continue(approved_res) is True
    assert agent.may_continue({"verdict": "approved", "may_continue": True}) is True
    assert agent.may_continue("approved") is True

    # Non-approved verdicts
    denied_res = ApprovalWaitResult(
        verdict="denied",
        may_continue=False,
        inbox_item_id="inbox_2",
    )
    assert agent.may_continue(denied_res) is False
    assert agent.may_continue({"verdict": "denied", "may_continue": False}) is False

    expired_res = ApprovalWaitResult(
        verdict="expired",
        may_continue=False,
        inbox_item_id="inbox_3",
    )
    assert agent.may_continue(expired_res) is False

    cancelled_res = ApprovalWaitResult(
        verdict="cancelled",
        may_continue=False,
        inbox_item_id="inbox_4",
    )
    assert agent.may_continue(cancelled_res) is False

    pending_res = ApprovalWaitResult(
        verdict="still_pending",
        may_continue=False,
        inbox_item_id="inbox_5",
    )
    assert agent.may_continue(pending_res) is False

    assert agent.may_continue(None) is False
    assert agent.may_continue({}) is False
