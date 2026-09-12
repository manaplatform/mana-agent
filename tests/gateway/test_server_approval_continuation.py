"""Tests for server action approval listening and model continuation."""

from __future__ import annotations

import getpass
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mana_agent.chat.events import AssistantMessageEvent, CodingActivityEvent
from mana_agent.chat.history import ChatHistory
from mana_agent.gateway import AgentChatGateway
from mana_agent.human_inbox.identity import ReviewerIdentity, StaticIdentityDirectory
from mana_agent.human_inbox.models import (
    InboxRequest,
    InboxRequestType,
    ResponseOperation,
    ReviewerAssignment,
    ReviewerType,
)
from mana_agent.human_inbox.repository import LocalInboxRepository
from mana_agent.human_inbox.service import HumanInboxService
from mana_agent.human_inbox.tokens import ResponseTokenSigner
from mana_agent.server.executor import ServerApprovalRequired
from mana_agent.server.models import (
    RemoteCommandResult,
    ServerActionDecision,
    ServerActionKind,
)
from mana_agent.services.execution_event_hub import (
    get_execution_event_hub,
    reset_execution_event_hub_for_tests,
)
from mana_agent.tui.app import ManaChatApp
from mana_agent.workspaces.paths import repository_id_for_path


def _human_inbox(tmp_path: Path) -> HumanInboxService:
    root = tmp_path / "inbox"
    reviewer = getpass.getuser()
    return HumanInboxService(
        repository=LocalInboxRepository(root),
        identities=StaticIdentityDirectory(
            [
                ReviewerIdentity(identity_id=reviewer),
            ]
        ),
        token_signer=ResponseTokenSigner(root / "signing.key"),
    )


def _persist_server_approval(
    gateway: AgentChatGateway,
    tmp_path: Path,
    *,
    request_id: str,
    pending: dict[str, Any],
) -> None:
    gateway.human_inbox_service = _human_inbox(tmp_path)
    decision = dict(pending.get("decision") or {})
    reviewer = getpass.getuser()
    gateway.human_inbox_service.create(
        InboxRequest(
            request_type=InboxRequestType.APPROVAL,
            task_id=str(
                pending.get("lane_task_id") or f"server:{pending['session_id']}"
            ),
            branch_id=str(
                pending.get("lane_task_id") or f"server:{pending['session_id']}"
            ),
            policy_decision_id=str(
                decision.get("decision_id") or "server-test-decision"
            ),
            permission_request_id=request_id,
            action_intent_id=(
                f"server:{decision.get('decision_id') or 'server-test-decision'}"
            ),
            action_digest=str(pending["exact_action_key"]),
            requested_by_agent_id="chat_gateway",
            reviewer=ReviewerAssignment(
                reviewer_type=ReviewerType.PERSON,
                reviewer_id=reviewer,
            ),
            title="Approve exact server action",
            summary="Review one exact server operation.",
            allowed_responses=[ResponseOperation.APPROVE, ResponseOperation.DENY],
            minimal_context={"action_count": 1, "resource_count": 1},
            protected_context={"server_action": pending},
            disclosed_fields=["action_count", "resource_count"],
            expires_at=gateway.human_inbox_service.clock() + timedelta(minutes=15),
            idempotency_key=f"server-test:{request_id}",
            deduplication_key=f"server-test:{request_id}",
        )
    )


def _make_server_decision(action: ServerActionKind = ServerActionKind.SHELL) -> ServerActionDecision:
    return ServerActionDecision.model_validate({
        "decision_id": "decision-srv-1",
        "server_id": "prod-srv-1",
        "action": action.value if hasattr(action, "value") else str(action),
        "tool_name": "server_command_execute",
        "arguments": {"argv": ["uptime"]},
        "required_capability": "shell.execute",
        "read_only": False,
        "consequential": True,
        "affected_resources": ["server:prod-srv-1"],
        "safe_to_continue": True,
        "reason": "check server uptime",
    })


def test_server_approval_yield_emits_hub_and_preserves_task_intent(tmp_path: Path) -> None:
    reset_execution_event_hub_for_tests()
    hub = get_execution_event_hub()
    emitted_events: list[dict[str, Any]] = []
    hub.subscribe("session-srv-1", lambda evt: emitted_events.append(evt))

    gateway = object.__new__(AgentChatGateway)
    gateway.root = tmp_path
    gateway._pending_server_approvals = {}

    sink_events: list[tuple[str, str, dict[str, Any]]] = []

    def sink(event_type: str, title: str, **kwargs: Any) -> None:
        sink_events.append((event_type, title, kwargs))

    gateway._event_sink = sink

    # Simulate pending state with task intent
    pending = {
        "session_id": "session-srv-1",
        "decision": _make_server_decision().model_dump(mode="json"),
        "argv": ["uptime"],
        "exact_action_key": "exact-uptime-key",
        "task_intent": "Check server uptime and tell me how long it has been running",
        "execution_id": "turn-123",
        "turn_id": "turn-123",
    }
    gateway._pending_server_approvals["srv_req_1"] = pending
    _persist_server_approval(
        gateway,
        tmp_path,
        request_id="srv_req_1",
        pending=pending,
    )

    item, loaded_pending = gateway._server_inbox_pending("srv_req_1")
    assert loaded_pending["task_intent"] == "Check server uptime and tell me how long it has been running"
    assert loaded_pending["argv"] == ["uptime"]


def test_server_approval_command_resumes_model_continuation(tmp_path: Path) -> None:
    reset_execution_event_hub_for_tests()
    hub = get_execution_event_hub()
    hub_events: list[dict[str, Any]] = []
    hub.subscribe("session-srv-continuation", lambda evt: hub_events.append(evt))

    transitions: list[tuple[str, str, str]] = []
    finishes: list[tuple[str, str]] = []
    executed: list[dict[str, Any]] = []
    sink_events: list[tuple[str, str, dict[str, Any]]] = []

    gateway = object.__new__(AgentChatGateway)
    gateway.root = tmp_path
    gateway._index_dir = str(tmp_path)
    gateway._resolved_k = 6
    gateway._agent_timeout_seconds = 60
    gateway.config = SimpleNamespace(agent_max_steps=6)
    gateway._fenced_sessions = set()
    gateway._session_store = {}
    gateway._history_store = SimpleNamespace(
        append=lambda *args, **kwargs: SimpleNamespace(
            to_dict=lambda: {"message_id": "msg-resumed"}
        )
    )
    gateway._workspaces = SimpleNamespace(touch_session=lambda sid: None)
    gateway._session = lambda sid: gateway._session_store.setdefault(sid, {})
    gateway._event_sink = lambda event_type, title, **kwargs: sink_events.append(
        (event_type, title, kwargs)
    )

    gateway._lane_coordinator = SimpleNamespace(
        transition=lambda task_id, state, *, reason: transitions.append(
            (task_id, state.value, reason)
        ),
        execution_supervisor=None,
        inspect_task=None,
    )
    gateway._finish_lane = lambda task_id, *, state, verification_state=None, error="": (
        finishes.append((task_id, state.value)),
        SimpleNamespace(state=state, error=error),
    )[1]

    # Server outcome
    mock_outcome = RemoteCommandResult(
        server_id="prod-srv-1",
        command_id="cmd-1",
        command="uptime",
        exit_code=0,
        stdout=" 13:42:01 up 42 days, 3 users, load average: 0.08, 0.05, 0.01",
        stderr="",
        started_at=datetime.now(timezone.utc),
    )

    async def _execute(*args, **kwargs):
        executed.append(kwargs)
        return mock_outcome

    gateway.server_management_service = SimpleNamespace(execute=_execute)

    # Ask agent mock to verify continuation prompt
    prompts_received: list[str] = []

    class MockAskAgent:
        def run(self, *, question: str, **kwargs: Any) -> Any:
            prompts_received.append(question)
            return SimpleNamespace(
                answer="The server prod-srv-1 has been up for 42 days with a low load average of 0.08."
            )

    gateway._stack = SimpleNamespace(
        ask_service=SimpleNamespace(ask_agent=MockAskAgent())
    )

    pending = {
        "session_id": "session-srv-continuation",
        "decision": _make_server_decision().model_dump(mode="json"),
        "argv": ["uptime"],
        "exact_action_key": "exact-uptime-key",
        "lane_task_id": "lane-srv-task",
        "timeout_seconds": 30,
        "pty": False,
        "environment": {},
        "task_intent": "Check server uptime and summarize it",
        "execution_id": "turn-exec-1",
    }
    gateway._pending_server_approvals = {"server_approval_cont_1": pending}
    _persist_server_approval(
        gateway,
        tmp_path,
        request_id="server_approval_cont_1",
        pending=pending,
    )

    result = gateway.server_approval_command(
        "server_approval_cont_1",
        session_id="session-srv-continuation",
        client_type="tui",
    )

    assert result["status"] == "succeeded"
    assert result["approved"] is True
    assert result["executed"] is True
    assert result["resume"] == "completed"
    assert result["answer"] == "The server prod-srv-1 has been up for 42 days with a low load average of 0.08."
    assert "Remote command output:" in result["message"]
    assert "The server prod-srv-1 has been up for 42 days" in result["message"]

    # Verify continuation prompt was passed to model
    assert len(prompts_received) == 1
    assert "Check server uptime and summarize it" in prompts_received[0]
    assert "uptime" in prompts_received[0]
    assert "42 days" in prompts_received[0]

    # Verify lane transitions
    assert transitions == [("lane-srv-task", "running", "server action approved by the user")]
    assert finishes == [("lane-srv-task", "completed")]

    # Verify Hub events
    hub_types = [e.get("type") for e in hub_events]
    assert "turn.resume_requested" in hub_types
    assert "server.approval_decided" in hub_types
    assert "turn.finished" in hub_types

    finished_evt = next(e for e in hub_events if e.get("type") == "turn.finished")
    assert finished_evt["status"] == "success"
    assert finished_evt["metadata"]["server_approval"] is True
    assert finished_evt["metadata"]["resumed"] is True


def test_server_approval_command_falls_back_to_summary_when_ask_agent_missing(tmp_path: Path) -> None:
    gateway = object.__new__(AgentChatGateway)
    gateway.root = tmp_path
    gateway._lane_coordinator = SimpleNamespace(
        transition=lambda *args, **kwargs: None,
        execution_supervisor=None,
        inspect_task=None,
    )
    gateway._finish_lane = lambda task_id, *, state, **kwargs: SimpleNamespace(state=state, error="")
    gateway._fenced_sessions = set()
    gateway._session_store = {}
    gateway._session = lambda sid: gateway._session_store.setdefault(sid, {})
    gateway._history_store = SimpleNamespace(append=lambda *args, **kwargs: SimpleNamespace(to_dict=lambda: {}))
    gateway._workspaces = SimpleNamespace(touch_session=lambda sid: None)
    gateway._event_sink = None
    gateway._stack = None

    mock_outcome = RemoteCommandResult(
        server_id="prod-srv-1",
        command_id="cmd-2",
        command="uname -r",
        exit_code=0,
        stdout="kernel 6.8.0",
        stderr="",
        started_at=datetime.now(timezone.utc),
    )
    async def _execute(*args, **kwargs):
        return mock_outcome

    gateway.server_management_service = SimpleNamespace(execute=_execute)

    pending = {
        "session_id": "session-no-agent",
        "decision": _make_server_decision().model_dump(mode="json"),
        "argv": ["uname", "-r"],
        "exact_action_key": "exact-uname-key",
        "lane_task_id": "lane-task-uname",
        "timeout_seconds": 30,
        "pty": False,
        "environment": {},
        "task_intent": "what kernel is running?",
    }
    gateway._pending_server_approvals = {"server_approval_no_agent": pending}
    _persist_server_approval(
        gateway,
        tmp_path,
        request_id="server_approval_no_agent",
        pending=pending,
    )

    result = gateway.server_approval_command(
        "server_approval_no_agent",
        session_id="session-no-agent",
        client_type="dashboard",
    )

    assert result["status"] == "succeeded"
    assert "Remote command output:\nkernel 6.8.0" in result["message"]
    assert result["answer"] == result["message"]


def test_server_approval_denial_emits_hub_events_and_stops_safely(tmp_path: Path) -> None:
    reset_execution_event_hub_for_tests()
    hub = get_execution_event_hub()
    hub_events: list[dict[str, Any]] = []
    hub.subscribe("session-deny-test", lambda evt: hub_events.append(evt))

    cancellations: list[tuple[str, str]] = []
    gateway = object.__new__(AgentChatGateway)
    gateway.root = tmp_path
    gateway._fenced_sessions = set()
    gateway._session_store = {}
    gateway._session = lambda sid: gateway._session_store.setdefault(sid, {})
    gateway._history_store = SimpleNamespace(append=lambda *args, **kwargs: SimpleNamespace(to_dict=lambda: {}))
    gateway._workspaces = SimpleNamespace(touch_session=lambda sid: None)
    gateway._event_sink = None
    gateway._lane_coordinator = SimpleNamespace(
        cancel_task=lambda task_id, *, reason: cancellations.append((task_id, reason))
    )

    pending = {
        "session_id": "session-deny-test",
        "decision": _make_server_decision().model_dump(mode="json"),
        "argv": ["reboot"],
        "exact_action_key": "exact-reboot-key",
        "lane_task_id": "lane-reboot-task",
        "task_intent": "reboot the server",
    }
    gateway._pending_server_approvals = {"server_approval_deny_1": pending}
    _persist_server_approval(
        gateway,
        tmp_path,
        request_id="server_approval_deny_1",
        pending=pending,
    )

    result = gateway.deny_server_approval_command(
        "server_approval_deny_1",
        session_id="session-deny-test",
    )

    assert result["status"] == "denied"
    assert result["approved"] is False
    assert result["executed"] is False
    assert "denied" in result["message"]
    assert cancellations == [("lane-reboot-task", "Server action denied by the user.")]

    hub_types = [e.get("type") for e in hub_events]
    assert "server.approval_decided" in hub_types
    assert "turn.finished" in hub_types
    decided_evt = next(e for e in hub_events if e.get("type") == "server.approval_decided")
    assert decided_evt["status"] == "cancelled"
    assert decided_evt["metadata"]["decision"] == "deny"


def test_tui_handles_server_waiting_approval_from_hub(tmp_path: Path) -> None:
    history = ChatHistory()
    app = ManaChatApp(history=history, repo_root=tmp_path, model="gpt-test")
    app._gateway_session_id = "session-tui-test"

    event = {
        "type": "server.waiting_approval",
        "event_id": "evt-srv-waiting-1",
        "conversation_id": "session-tui-test",
        "execution_id": "exec-1",
        "title": "Server action approval required",
        "status": "running",
        "metadata": {
            "permission_request_id": "srv_req_tui_1",
            "permission_scope": "server.action.execute",
            "preview": "uptime",
            "server_approval": True,
            "server_id": "prod-srv-1",
        },
    }

    app._handle_hub_session_event(event)

    # Activity should be posted into history
    activities = [
        item for item in history.get_events()
        if isinstance(item, CodingActivityEvent)
    ]
    assert len(activities) == 1
    assert activities[0].activity["event_type"] == "server.waiting_approval"
    assert activities[0].activity["metadata"]["permission_request_id"] == "srv_req_tui_1"
    assert activities[0].activity["metadata"]["server_approval"] is True


def test_tui_record_server_approval_completion(tmp_path: Path) -> None:
    history = ChatHistory()
    app = ManaChatApp(history=history, repo_root=tmp_path, model="gpt-test")

    app._record_server_approval_completion(
        "srv_req_tui_1",
        {
            "status": "succeeded",
            "answer": "Uptime is 42 days.",
            "message": "Approved server action completed.\n\nRemote command output:\n42 days\n\nUptime is 42 days.",
        },
    )

    assistant_events = [
        item for item in history.get_events()
        if isinstance(item, AssistantMessageEvent)
    ]
    assert len(assistant_events) == 1
    assert assistant_events[0].content == "Uptime is 42 days."
    assert assistant_events[0].turn_id == "srv_req_tui_1"
    assert app.status_text == "Server action completed"
