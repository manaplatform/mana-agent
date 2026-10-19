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
        reviewer=ReviewerAssignment(reviewer_type=ReviewerType.PERSON, reviewer_id="user"),
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
