"""Tests for real-time tool call approval display and modal interaction in TUI."""

from __future__ import annotations

import asyncio
import getpass
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mana_agent.chat.events import CodingActivityEvent, ToolCallEvent, ToolResultEvent
from mana_agent.chat.history import ChatHistory
from mana_agent.coding.programmatic import ToolExecutionDispatcher
from mana_agent.coding.models import WorkspaceContext
from mana_agent.human_inbox import default_human_inbox_service
from mana_agent.human_inbox.models import InboxStatus, ResponseOperation, ResponseSubmission
from mana_agent.transactional_actions.runtime import default_action_gateway
from mana_agent.tui.app import ManaChatApp
from mana_agent.tui.computer_permission import ComputerPermissionChoice, ComputerPermissionRequested, ComputerPermissionScreen
from mana_agent.tui.widgets.activity_boxes import CodingActivityBox, MainActivityBox
from mana_agent.tui.widgets.execution_panel import ExecutionPanel
from mana_agent.tui.widgets.tool_card import ToolCard


def test_main_activity_box_renders_waiting_approval_realtime() -> None:
    box = MainActivityBox(turn_id="turn-1")
    box.update_event({
        "event_id": "ev-1",
        "event_type": "tool.call.started",
        "tool_name": "repo_search",
        "status": "success",
        "summary": "found files",
    })
    assert "✓" in box.header.renderable
    assert "completed" in box.header.renderable

    # Post an approval required event
    box.update_event({
        "event_id": "ev-2",
        "event_type": "action.approval.required",
        "tool_name": "approval_request",
        "status": "waiting_approval",
        "summary": "Run git commit",
    })
    header_text = str(box.header.renderable)
    assert "⚠️" in header_text
    assert "WAITING APPROVAL" in header_text
    assert box._active_status == "waiting_approval"

    stream_text = box.activity_log.text
    assert "⚠️ approval_request" in stream_text
    assert "[WAITING FOR APPROVAL]" in stream_text


def test_coding_activity_box_renders_waiting_approval_realtime() -> None:
    box = CodingActivityBox(turn_id="turn-2")
    box.backend = "internal"
    box.phase = "coding"

    box.update_event({
        "event_id": "ev-c1",
        "event_type": "action.approval.required",
        "tool_name": "approval_request",
        "title": "Approval required: pytest",
        "status": "waiting_approval",
        "output_preview": "pytest tests/",
    })

    header_text = str(box.header.renderable)
    assert "⚠️" in header_text
    assert "WAITING APPROVAL" in header_text

    log_text = box.activity_log.text
    assert "⚠️" in log_text
    assert "[WAITING FOR APPROVAL]" in log_text


def test_tool_card_approval_header_badge() -> None:
    call_ev = ToolCallEvent(
        call_id="call-appr-1",
        tool_name="approval_request",
        args={"command": "pytest", "reason": "run tests"},
        summary="run pytest",
    )
    card = ToolCard(call_ev)
    header = card._build_call_header()
    assert "⚠️ [APPROVAL REQUIRED]" in header
    assert "approval_request" in header


@pytest.mark.parametrize("tool_name", ["approval_request", "request_user_approval", "wait_for_approval"])
def test_approval_tools_do_not_create_separate_cards(tool_name: str) -> None:
    from mana_agent.tui.widgets.chat_log import ChatLog

    log = ChatLog(history=ChatHistory())
    panel = MagicMock()
    log._get_or_create_execution_panel = MagicMock(return_value=panel)
    event = ToolCallEvent(call_id="call-1", tool_name=tool_name, args={}, turn_id="turn-1")
    assert log._add_tool_call(event) is panel
    panel.update_event.assert_called_once()
    panel.main_box.add_tool_call.assert_not_called()


def test_transactional_permission_screen_offers_matching_batch() -> None:
    import anyio
    from textual.app import App

    class TestApp(App):
        choice = None

        def on_mount(self) -> None:
            self.push_screen(ComputerPermissionScreen(
                request_id="request-1", scope="transactional_action.once",
                preview="Preview", transactional=True, pending_count=1,
            ), self.record_choice)

        def record_choice(self, choice) -> None:
            self.choice = choice

    async def exercise() -> None:
        app = TestApp()
        async with app.run_test() as pilot:
            assert app.screen.query_one("#permission-all").label.plain == "Allow all matching queued"
            await pilot.click("#permission-all")
            await pilot.pause()
            assert app.choice.approve_all is True
            assert app.choice.request_id == "request-1"

    anyio.run(exercise)


def test_execution_panel_renders_waiting_approval() -> None:
    panel = ExecutionPanel(turn_id="turn-3")
    panel.update_event({
        "event_id": "ev-p1",
        "event_type": "action.approval.required",
        "tool_name": "approval_request",
        "title": "Approval required: git push",
        "status": "waiting_approval",
        "detail": "git push origin main",
    })

    header_text = str(panel.header.renderable)
    assert "⚠️" in header_text
    assert "waiting for approval" in header_text

    steps_text = str(panel.steps_view.renderable)
    assert "⚠️" in steps_text
    assert "waiting for approval" in steps_text


def test_tool_execution_dispatcher_dispatches_approval_request(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    ws = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="readOnly",
    )
    dispatcher = ToolExecutionDispatcher(ws)

    # Calling with timeout_seconds=0.01 to test direct execution path without hanging
    result = dispatcher.dispatch(
        "approval_request",
        {
            "command": "git status",
            "reason": "Test status check",
            "timeout_seconds": 0.01,
        },
    )
    assert isinstance(result, dict)
    assert "ok" in result
    assert "status" in result
    assert result.get("error_code") != "UNKNOWN_TOOL"


def test_app_complete_computer_permission_fallback(tmp_path: Path, monkeypatch) -> None:
    import anyio

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    history = ChatHistory()
    app = ManaChatApp(history=history, repo_root=tmp_path)
    app.chat_log = MagicMock()

    inbox = default_human_inbox_service()
    from mana_agent.human_inbox.models import (
        InboxRequest,
        InboxRequestType,
        ResponseOperation,
        ReviewerAssignment,
        ReviewerType,
        RiskLevel,
    )
    item = inbox.create(InboxRequest(
        request_type=InboxRequestType.APPROVAL,
        task_id="task-1",
        branch_id="task-1",
        requested_by_agent_id="agent-1",
        reviewer=ReviewerAssignment(reviewer_type=ReviewerType.PERSON, reviewer_id=getpass.getuser()),
        title="Approve deploy",
        summary="Deploy to prod",
        risk_level=RiskLevel.HIGH,
        allowed_responses=[ResponseOperation.APPROVE, ResponseOperation.DENY],
    ))

    choice = ComputerPermissionChoice(
        request_id=item.inbox_item_id,
        decision=MagicMock(),  # truthy decision = approve
        transactional=True,
    )

    anyio.run(app._complete_computer_permission, choice)

    updated_item = inbox.repository.get(item.inbox_item_id)
    assert updated_item.status == InboxStatus.APPROVED

    app.chat_log._scroll_to_latest.assert_called_once()

    # Verify event was posted to history
    events = history.get_events()
    granted_events = [e for e in events if isinstance(e, CodingActivityEvent) and e.activity.get("event_type") == "action.approval.granted"]
    assert len(granted_events) == 1
    assert granted_events[0].activity["status"] == "completed"


def test_app_enqueues_transactional_modal_from_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    history = ChatHistory()
    app = ManaChatApp(history=history, repo_root=tmp_path)
    messages_posted = []
    app.post_message = lambda msg: messages_posted.append(msg)

    # Post CodingActivityEvent for approval
    act_event = CodingActivityEvent(activity={
        "event_type": "action.approval.required",
        "title": "Approval required for command",
        "status": "waiting_approval",
        "metadata": {
            "inbox_item_id": "inbox-custom-123",
            "permission_request_id": "inbox-custom-123",
            "transactional_action_approval": True,
            "preview": {"command": "nmap -p 80 target.local", "reason": "security scan"},
        },
    })

    app._handle_computer_permission_event(act_event)

    assert len(messages_posted) == 1
    req = messages_posted[0]
    assert isinstance(req, ComputerPermissionRequested)
    assert req.request_id == "inbox-custom-123"
    assert req.transactional is True
    assert "nmap" in req.preview


def test_app_handles_hub_session_event_approval(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    history = ChatHistory()
    app = ManaChatApp(history=history, repo_root=tmp_path)
    messages_posted = []
    app.post_message = lambda msg: messages_posted.append(msg)

    hub_event = {
        "event_id": "hub-ev-1",
        "event_type": "action.approval.required",
        "title": "Transactional action approval required",
        "status": "waiting_approval",
        "metadata": {
            "inbox_item_id": "inbox-hub-456",
            "permission_request_id": "inbox-hub-456",
            "transactional_action_approval": True,
            "preview": {"command": "rm -rf cache", "reason": "clean cache"},
        },
    }

    app._handle_hub_session_event(hub_event)

    assert len(messages_posted) == 1
    req = messages_posted[0]
    assert isinstance(req, ComputerPermissionRequested)
    assert req.request_id == "inbox-hub-456"
    assert "rm -rf" in req.preview


def test_git_branch_approval_request_flow_displays_modal_and_unblocks(tmp_path: Path, monkeypatch) -> None:
    """Verify that when git_create_branch requires approval and approval_request is invoked,
    the TUI modal is enqueued in real-time and user approval issues the valid grant."""
    import anyio
    from mana_agent.chat.history import get_history, reset_global_history
    from mana_agent.human_inbox.approval_tools import approval_request
    from mana_agent.multi_agent.tools.git_tools import create_branch
    from mana_agent.transactional_actions.runtime import default_action_gateway

    mana_home_dir = tmp_path / "mana_home"
    monkeypatch.setenv("MANA_HOME", str(mana_home_dir))

    # Initialize a mock git repo in axis_dir
    axis_dir = tmp_path / "axis"
    axis_dir.mkdir()
    import subprocess
    subprocess.run(["git", "init", "-b", "main"], cwd=axis_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=axis_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=axis_dir, check=True, capture_output=True)
    (axis_dir / "README.md").write_text("Hello", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=axis_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=axis_dir, check=True, capture_output=True)

    # Set up global history and TUI app
    reset_global_history()
    global_hist = get_history()
    app = ManaChatApp(history=global_hist, repo_root=axis_dir)
    messages_posted = []
    app.post_message = lambda msg: messages_posted.append(msg)
    app.on_mount()

    # Step 1: git_create_branch is executed and requires approval
    branch_res = create_branch(branch_name="feat/implementation", repo_path=axis_dir, switch_to=True)
    assert branch_res["ok"] is False
    assert branch_res["error_code"] == "approval_required"
    action_id = branch_res["action_id"]
    assert action_id.startswith("act_")

    # Step 2: The model calls approval_request with the action_id
    # Run approval_request in a background thread with short timeout
    import threading
    req_result = {}

    def run_approval():
        req_result["result"] = approval_request(
            command="git switch -c feat/implementation",
            title="Create and switch to feature branch",
            reason="Create feature branch in /Users/ah/Documents/axis",
            inbox_item_id=action_id,
            action_id=action_id,
            risk_level="LOCAL_SAFE_WRITE",
            timeout_seconds=2.0,
            workspace_root=axis_dir,
        )

    t = threading.Thread(target=run_approval)
    t.start()

    # Wait briefly for the event to be published and processed by the TUI
    import time
    time.sleep(0.1)

    # The modal request MUST have been posted to the TUI message pump!
    assert len(messages_posted) >= 1
    modal_req = messages_posted[-1]
    assert isinstance(modal_req, ComputerPermissionRequested)
    assert modal_req.transactional is True

    # Step 3: Simulate user approving the modal in the TUI
    choice = ComputerPermissionChoice(
        request_id=modal_req.request_id,
        decision=MagicMock(),  # truthy decision = approve
        transactional=True,
    )
    anyio.run(app._complete_computer_permission, choice)

    # The approval_request thread should now terminate with approved=True
    t.join(timeout=3.0)
    assert not t.is_alive()
    res = req_result.get("result")
    assert res is not None
    assert res.get("approved") is True
    assert res.get("approval_id")

    # Verify that the action is now granted in the gateway
    gw = default_action_gateway(axis_dir)
    action = gw.store.get_action(action_id)
    assert action is not None
    assert gw.approvals.find_valid(action) is not None


def test_app_complete_computer_permission_falls_back_when_gateway_inbox_misses(tmp_path: Path, monkeypatch) -> None:
    """Verify that when app.gateway is present but misses the inbox item,
    app._complete_computer_permission falls back to default_human_inbox_service."""
    import anyio
    from mana_agent.human_inbox.models import (
        InboxRequest,
        InboxRequestType,
        InboxStatus,
        ResponseOperation,
        ReviewerAssignment,
        ReviewerType,
        RiskLevel,
    )

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    history = ChatHistory()
    app = ManaChatApp(history=history, repo_root=tmp_path)

    # Mock gateway with an inbox service that does not contain the item
    mock_gw = MagicMock()
    mock_gw.transactional_action_approval_command.side_effect = LookupError("Not in gateway inbox")
    mock_inbox = MagicMock()
    mock_inbox.repository.get.side_effect = LookupError("Not in gateway inbox")
    mock_inbox.repository.find_for_action.return_value = []
    mock_gw.human_inbox_service = mock_inbox
    app.gateway = mock_gw

    # Item is in default_human_inbox_service()
    inbox = default_human_inbox_service()
    item = inbox.create(InboxRequest(
        request_type=InboxRequestType.APPROVAL,
        task_id="task-isolated-1",
        branch_id="task-isolated-1",
        requested_by_agent_id="agent-1",
        reviewer=ReviewerAssignment(reviewer_type=ReviewerType.PERSON, reviewer_id=getpass.getuser()),
        title="Approve isolated task action",
        summary="Action created outside gateway inbox",
        risk_level=RiskLevel.HIGH,
        allowed_responses=[ResponseOperation.APPROVE, ResponseOperation.DENY],
    ))

    choice = ComputerPermissionChoice(
        request_id=item.inbox_item_id,
        decision=MagicMock(),  # truthy = approve
        transactional=True,
    )

    anyio.run(app._complete_computer_permission, choice)

    updated_item = inbox.repository.get(item.inbox_item_id)
    assert updated_item.status == InboxStatus.APPROVED


def test_app_complete_computer_permission_idempotent_when_action_already_committed(
    tmp_path: Path, monkeypatch
) -> None:
    """Verify that if an action was already committed/executed, repeat approval succeeds idempotently
    without 'only an action awaiting approval may receive a grant' error message."""
    import anyio
    from mana_agent.chat.events import AssistantMessageEvent
    from mana_agent.human_inbox.models import (
        InboxRequest,
        InboxRequestType,
        InboxStatus,
        ResponseOperation,
        ReviewerAssignment,
        ReviewerType,
        RiskLevel,
    )
    from mana_agent.transactional_actions.adapters import ShellActionAdapter
    from mana_agent.transactional_actions.models import ActionState

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    history = ChatHistory()
    app = ManaChatApp(history=history, repo_root=tmp_path)

    gw = default_action_gateway(tmp_path)
    adapter = ShellActionAdapter(
        argv=["echo", "done"],
        cwd=tmp_path,
        environment={},
        parent_task_id="task-idem-1",
        actor="user",
        originating_agent="test",
        policy_context={"tool": "shell"},
        idempotency_key="key-idem-1",
        expected_outputs=[],
    )
    action = gw.propose(adapter)
    if action.state is ActionState.PROPOSED:
        action.transition(ActionState.PREVIEWING)
        action.transition(ActionState.AWAITING_POLICY)
        action.transition(ActionState.AWAITING_APPROVAL)
        action.transition(ActionState.APPROVED)
    elif action.state is ActionState.AWAITING_APPROVAL:
        action.transition(ActionState.APPROVED)
    if action.state is ActionState.APPROVED:
        action.transition(ActionState.EXECUTING)
        action.transition(ActionState.VERIFYING)
        action.transition(ActionState.COMMITTED)
    gw.store.save_action(action)

    inbox = default_human_inbox_service()
    item = inbox.create(InboxRequest(
        request_type=InboxRequestType.APPROVAL,
        task_id="task-idem-1",
        branch_id="task-idem-1",
        requested_by_agent_id="agent-1",
        action_intent_id=action.action_id,
        action_digest=action.approval_digest(),
        reviewer=ReviewerAssignment(reviewer_type=ReviewerType.PERSON, reviewer_id=getpass.getuser()),
        title="Execute echo done",
        summary="Mutating action already executed",
        risk_level=RiskLevel.MEDIUM,
        allowed_responses=[ResponseOperation.APPROVE, ResponseOperation.DENY],
    ))

    choice = ComputerPermissionChoice(
        request_id=item.inbox_item_id,
        decision=MagicMock(),
        transactional=True,
    )

    anyio.run(app._complete_computer_permission, choice)

    # When approved, no redundant assistant message box is added to chat history (only a toast is shown)
    messages = [e for e in history.get_events() if isinstance(e, AssistantMessageEvent)]
    assert len(messages) == 0

    # Verify that the approval grant event was recorded in CodingActivityEvent without failure
    granted_events = [
        e for e in history.get_events()
        if isinstance(e, CodingActivityEvent) and e.activity.get("event_type") == "action.approval.granted"
    ]
    assert len(granted_events) == 1
    assert granted_events[0].activity["status"] == "completed"


def test_app_complete_computer_permission_multi_request_approve_all(
    tmp_path: Path, monkeypatch
) -> None:
    """An unrelated queued request is not approved by approve-all."""
    import anyio
    from mana_agent.human_inbox.models import (
        InboxRequest,
        InboxRequestType,
        InboxStatus,
        ResponseOperation,
        ReviewerAssignment,
        ReviewerType,
        RiskLevel,
    )

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    history = ChatHistory()
    app = ManaChatApp(history=history, repo_root=tmp_path)
    inbox = default_human_inbox_service()

    item1 = inbox.create(InboxRequest(
        request_type=InboxRequestType.APPROVAL,
        task_id="task-multi-1",
        branch_id="task-multi-1",
        requested_by_agent_id="agent-1",
        reviewer=ReviewerAssignment(reviewer_type=ReviewerType.PERSON, reviewer_id=getpass.getuser()),
        title="Request 1",
        summary="Action 1",
        risk_level=RiskLevel.HIGH,
        allowed_responses=[ResponseOperation.APPROVE, ResponseOperation.DENY],
    ))
    item2 = inbox.create(InboxRequest(
        request_type=InboxRequestType.APPROVAL,
        task_id="task-multi-2",
        branch_id="task-multi-2",
        requested_by_agent_id="agent-1",
        reviewer=ReviewerAssignment(reviewer_type=ReviewerType.PERSON, reviewer_id=getpass.getuser()),
        title="Request 2",
        summary="Action 2",
        risk_level=RiskLevel.HIGH,
        allowed_responses=[ResponseOperation.APPROVE, ResponseOperation.DENY],
    ))

    app._transactional_modal_queue.append(item2.inbox_item_id)

    choice = ComputerPermissionChoice(
        request_id=item1.inbox_item_id,
        decision=MagicMock(),
        transactional=True,
        approve_all=True,
    )

    anyio.run(app._complete_computer_permission, choice)

    up1 = inbox.repository.get(item1.inbox_item_id)
    up2 = inbox.repository.get(item2.inbox_item_id)
    assert up1.status == InboxStatus.APPROVED
    assert up2.status in {InboxStatus.PENDING, InboxStatus.DELIVERED}
    assert app._transactional_modal_queue == [item2.inbox_item_id]


def test_approve_chat_command_does_not_approve_unrelated_requests(tmp_path: Path, monkeypatch) -> None:
    """A plan approval command must not silently approve inbox requests."""
    from mana_agent.chat_commands.builtins import _approve_command
    from mana_agent.chat_commands.models import CommandContext
    from mana_agent.human_inbox.models import (
        InboxRequest,
        InboxRequestType,
        InboxStatus,
        ResponseOperation,
        ReviewerAssignment,
        ReviewerType,
        RiskLevel,
    )

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    inbox = default_human_inbox_service()

    item1 = inbox.create(InboxRequest(
        request_type=InboxRequestType.APPROVAL,
        task_id="task-cmd-1",
        branch_id="task-cmd-1",
        requested_by_agent_id="agent-1",
        reviewer=ReviewerAssignment(reviewer_type=ReviewerType.PERSON, reviewer_id=getpass.getuser()),
        title="Command Request 1",
        summary="Action 1",
        risk_level=RiskLevel.MEDIUM,
        allowed_responses=[ResponseOperation.APPROVE, ResponseOperation.DENY],
    ))
    item2 = inbox.create(InboxRequest(
        request_type=InboxRequestType.APPROVAL,
        task_id="task-cmd-2",
        branch_id="task-cmd-2",
        requested_by_agent_id="agent-1",
        reviewer=ReviewerAssignment(reviewer_type=ReviewerType.PERSON, reviewer_id=getpass.getuser()),
        title="Command Request 2",
        summary="Action 2",
        risk_level=RiskLevel.MEDIUM,
        allowed_responses=[ResponseOperation.APPROVE, ResponseOperation.DENY],
    ))

    gateway = MagicMock()
    gateway.get_session_active_plan.return_value = None
    gateway.approve_session_plan.return_value = False
    gateway.human_inbox_service = inbox

    ctx = CommandContext(
        session_id="sess-cmd-multi",
        gateway=gateway,
        frontend="tui",
    )

    res = _approve_command(ctx, [])
    assert res.status == "error"

    up1 = inbox.repository.get(item1.inbox_item_id)
    up2 = inbox.repository.get(item2.inbox_item_id)
    assert up1.status in {InboxStatus.PENDING, InboxStatus.DELIVERED}
    assert up2.status in {InboxStatus.PENDING, InboxStatus.DELIVERED}

