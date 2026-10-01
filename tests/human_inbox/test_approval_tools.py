"""Tests for model-driven human approval tools and decision wait contracts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mana_agent.human_inbox.approval_tools import (
    ApprovalWaitResult,
    build_approval_tools,
    request_user_approval,
    wait_for_approval,
)
from mana_agent.human_inbox.identity import ReviewerIdentity, StaticIdentityDirectory
from mana_agent.human_inbox.models import (
    InboxStatus,
    ResponseOperation,
    ResponseSubmission,
)
from mana_agent.human_inbox.repository import LocalInboxRepository
from mana_agent.human_inbox.service import HumanInboxService
from mana_agent.human_inbox.tokens import ResponseTokenSigner
from mana_agent.multi_agent.agents.approval_agent import ApprovalAgent


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


def create_test_inbox_service(tmp_path: Path, clock: Clock) -> HumanInboxService:
    root = tmp_path / "inbox"
    return HumanInboxService(
        repository=LocalInboxRepository(root),
        identities=StaticIdentityDirectory(
            [ReviewerIdentity(identity_id="reviewer-1", roles={"security"})]
        ),
        token_signer=ResponseTokenSigner(root / "signing.key", clock=clock),
        notification_adapters=[],
        clock=clock,
    )


def submit_decision(
    inbox: HumanInboxService,
    inbox_item_id: str,
    operation: ResponseOperation,
) -> None:
    item = inbox.repository.get(inbox_item_id)
    inbox.respond(
        ResponseSubmission(
            inbox_item_id=item.inbox_item_id,
            operation=operation,
            actor_id="reviewer-1",
            channel="test",
            idempotency_key=f"resp_{item.inbox_item_id}_{operation.value}",
            expected_version=item.version,
            current_action_digest=item.action_digest,
        )
    )


def test_pending_then_approved_returns_may_continue(tmp_path: Path) -> None:
    clock = Clock()
    inbox = create_test_inbox_service(tmp_path, clock)

    task_id = "task_approve_001"
    agent_id = "agent_approval_001"
    decision_id = "dec_mutation_patch"

    # Step 1: Create approval request
    inbox_item_id = request_user_approval(
        source_decision_id=decision_id,
        task_id=task_id,
        agent_id=agent_id,
        title="Approve repository patch",
        summary="Apply mutation to core files.",
        reviewer_id="reviewer-1",
        inbox_service=inbox,
    )
    assert isinstance(inbox_item_id, str)
    assert inbox_item_id.startswith("inbox_")

    initial_item = inbox.repository.get(inbox_item_id)
    assert initial_item.status is InboxStatus.PENDING

    # Step 2: Human reviewer approves
    submit_decision(inbox, inbox_item_id, ResponseOperation.APPROVE)

    # Step 3: Wait tool polls and observes approved verdict
    result = wait_for_approval(
        inbox_item_id=inbox_item_id,
        source_decision_id=decision_id,
        task_id=task_id,
        agent_id=agent_id,
        timeout_seconds=2.0,
        poll_interval_seconds=0.01,
        inbox_service=inbox,
    )

    assert result.verdict == "approved"
    assert result.may_continue is True
    assert result["verdict"] == "approved"
    assert result["may_continue"] is True

    agent = ApprovalAgent(agent_id=agent_id)
    assert agent.may_continue(result) is True
    assert agent.may_continue("approved") is True


def test_denied_and_expired_never_continue(tmp_path: Path) -> None:
    clock = Clock()
    inbox = create_test_inbox_service(tmp_path, clock)
    agent = ApprovalAgent(agent_id="agent_approval_001")

    # Case A: Denied
    denied_item_id = request_user_approval(
        source_decision_id="dec_deny_001",
        task_id="task_denied_001",
        agent_id="agent_approval_001",
        title="Delete production resources",
        reviewer_id="reviewer-1",
        inbox_service=inbox,
    )
    submit_decision(inbox, denied_item_id, ResponseOperation.DENY)

    denied_result = wait_for_approval(
        inbox_item_id=denied_item_id,
        source_decision_id="dec_deny_001",
        task_id="task_denied_001",
        agent_id="agent_approval_001",
        timeout_seconds=1.0,
        poll_interval_seconds=0.01,
        inbox_service=inbox,
    )
    assert denied_result.verdict == "denied"
    assert denied_result.may_continue is False
    assert agent.may_continue(denied_result) is False

    # Case B: Expired
    clock.advance(10)
    expire_time = clock() + timedelta(seconds=30)
    expired_item_id = request_user_approval(
        source_decision_id="dec_expire_001",
        task_id="task_expired_001",
        agent_id="agent_approval_001",
        title="Expiring request",
        reviewer_id="reviewer-1",
        expires_at=expire_time,
        inbox_service=inbox,
    )
    clock.advance(60)
    inbox.expire_due()

    expired_result = wait_for_approval(
        inbox_item_id=expired_item_id,
        source_decision_id="dec_expire_001",
        task_id="task_expired_001",
        agent_id="agent_approval_001",
        timeout_seconds=1.0,
        poll_interval_seconds=0.01,
        inbox_service=inbox,
    )
    assert expired_result.verdict == "expired"
    assert expired_result.may_continue is False
    assert agent.may_continue(expired_result) is False


def test_timeout_returns_still_pending_and_calls_suspend_hook(tmp_path: Path) -> None:
    clock = Clock()
    inbox = create_test_inbox_service(tmp_path, clock)
    agent = ApprovalAgent(agent_id="agent_approval_001")

    item_id = request_user_approval(
        source_decision_id="dec_timeout_001",
        task_id="task_timeout_001",
        agent_id="agent_approval_001",
        title="Unattended approval",
        reviewer_id="reviewer-1",
        inbox_service=inbox,
    )

    mock_suspend_hook = MagicMock()

    result = wait_for_approval(
        inbox_item_id=item_id,
        source_decision_id="dec_timeout_001",
        task_id="task_timeout_001",
        agent_id="agent_approval_001",
        timeout_seconds=0.05,
        poll_interval_seconds=0.01,
        on_timeout_suspend=mock_suspend_hook,
        inbox_service=inbox,
    )

    assert result.verdict == "still_pending"
    assert result.may_continue is False
    assert mock_suspend_hook.called is True
    assert agent.may_continue(result) is False


def test_wrong_agent_or_task_raises_permission_error(tmp_path: Path) -> None:
    clock = Clock()
    inbox = create_test_inbox_service(tmp_path, clock)

    item_id = request_user_approval(
        source_decision_id="dec_perm_001",
        task_id="task_perm_001",
        agent_id="agent_authorized",
        reviewer_id="reviewer-1",
        inbox_service=inbox,
    )

    # Wrong agent cannot observe or wait for this approval
    with pytest.raises(PermissionError, match="not authorized"):
        wait_for_approval(
            inbox_item_id=item_id,
            source_decision_id="dec_perm_001",
            task_id="task_perm_001",
            agent_id="agent_imposter",
            timeout_seconds=0.5,
            inbox_service=inbox,
        )

    # Wrong task ID cannot observe or wait for this approval
    with pytest.raises(PermissionError, match="not authorized"):
        wait_for_approval(
            inbox_item_id=item_id,
            source_decision_id="dec_perm_001",
            task_id="wrong_task_id",
            agent_id="agent_authorized",
            timeout_seconds=0.5,
            inbox_service=inbox,
        )


def test_invalid_or_missing_decision_stops_safely_with_no_fallback(tmp_path: Path) -> None:
    clock = Clock()
    inbox = create_test_inbox_service(tmp_path, clock)

    # Missing or empty source_decision_id in request_user_approval
    with pytest.raises(ValueError, match="valid source_decision_id"):
        request_user_approval(
            source_decision_id="",
            task_id="task-1",
            agent_id="agent-1",
            inbox_service=inbox,
        )

    with pytest.raises(ValueError, match="valid source_decision_id"):
        request_user_approval(
            source_decision_id="   ",
            task_id="task-1",
            agent_id="agent-1",
            inbox_service=inbox,
        )

    # Missing task_id or agent_id
    with pytest.raises(ValueError, match="task_id is required"):
        request_user_approval(
            source_decision_id="dec-1",
            task_id="",
            agent_id="agent-1",
            inbox_service=inbox,
        )

    with pytest.raises(ValueError, match="agent_id is required"):
        request_user_approval(
            source_decision_id="dec-1",
            task_id="task-1",
            agent_id="",
            inbox_service=inbox,
        )

    # Missing source_decision_id in wait_for_approval
    with pytest.raises(ValueError, match="valid source_decision_id"):
        wait_for_approval(
            inbox_item_id="inbox_123",
            source_decision_id="",
            task_id="task-1",
            agent_id="agent-1",
            inbox_service=inbox,
        )

    # Missing inbox_item_id in wait_for_approval
    with pytest.raises(ValueError, match="inbox_item_id is required"):
        wait_for_approval(
            inbox_item_id="",
            source_decision_id="dec-1",
            task_id="task-1",
            agent_id="agent-1",
            inbox_service=inbox,
        )


def test_build_approval_tools(tmp_path: Path) -> None:
    clock = Clock()
    inbox = create_test_inbox_service(tmp_path, clock)

    tools = build_approval_tools(inbox_service=inbox)
    assert len(tools) == 2
    tool_names = {t.name for t in tools}
    assert tool_names == {"request_user_approval", "wait_for_approval"}
    for tool in tools:
        assert tool.metadata.get("read_only") is True
        assert tool.metadata.get("inbox_only") is True
