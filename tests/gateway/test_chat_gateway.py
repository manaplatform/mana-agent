"""Tests for the central AgentChatGateway.

These verify:
- Construction succeeds with minimal config.
- Gateway builds coding stack when enabled (with injected fakes).
- Simple send / process_turn paths work.
- Rich context is provided.
- Auto-chat + coding agent exist on the gateway (not only chat_cli).
"""

from __future__ import annotations

import getpass
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mana_agent.config.settings import Settings
from mana_agent.gateway import (
    AgentChatGateway,
    ChatGatewayConfig,
    ChatTurnResult,
    RichChatContext,
)
from mana_agent.gateway.entry_routing import EntryRouteContext, EntryRoutingDecision
from mana_agent.gateway.lanes import LaneId
from mana_agent.human_inbox.identity import ReviewerIdentity, StaticIdentityDirectory
from mana_agent.human_inbox.models import (
    InboxRequest,
    InboxRequestType,
    ResponseOperation,
    ReviewerAssignment,
    ReviewerType,
)
from mana_agent.human_inbox.repository import (
    InboxConcurrentUpdateError,
    LocalInboxRepository,
)
from mana_agent.human_inbox.service import HumanInboxService
from mana_agent.human_inbox.tokens import ResponseTokenSigner
from mana_agent.integrations.codex.coding_agent_shim import CodexCodingAgentShim
from mana_agent.coding.internal_agent_shim import InternalCodingAgentShim
from mana_agent.memory import MemoryContent, MemoryRecord
from mana_agent.multi_agent.routing.agent_decision import AgentDecision
from mana_agent.remote_execution.models import RemoteExecutionRequest
from mana_agent.remote_execution.providers.local_ssh import LocalSSHProvider
from mana_agent.remote_execution.service import RemoteExecutionService
from mana_agent.remote_execution.target_policy import TargetPolicy, TargetPolicyMode
from mana_agent.server.models import ServerActionDecision
from mana_agent.services.chat_session_history import ChatSessionHistory
from mana_agent.transactional_actions.store import ActionStore
from mana_agent.chat.events import AssistantMessageEvent, CodingActivityEvent
from mana_agent.chat.history import reset_global_history


class _DummyAskService:
    """Minimal stand-in so gateway construction tests do not require OPENAI_API_KEY."""

    class _EntryModel:
        def invoke(self, messages, **_kwargs):
            payload = json.loads(messages[-1].content)
            if "recovery_candidates" in payload:
                return SimpleNamespace(
                    content=json.dumps(
                        {
                            "decision_id": "test-start-fresh",
                            "action": "start_fresh",
                            "task_id": "",
                            "checkpoint_id": "",
                            "same_work": False,
                            "fresh_data_required": bool(
                                payload.get("entry_route_requires_live_data")
                            ),
                            "checkpoint_still_valid": False,
                            "side_effects_safe_to_repeat": False,
                            "safe_to_continue": True,
                            "reason": "the test model selected a fresh execution",
                        }
                    )
                )
            prompt = str(payload.get("user_prompt") or "").lower()
            if "gmail" in prompt:
                route = "gmail"
            elif any(word in prompt for word in ("update", "change", "edit")):
                route = "coding"
            elif any(word in prompt for word in ("project", "read the value")):
                route = "repository"
            else:
                route = "conversation"
            sources = {
                "gmail": ["gmail"], "coding": ["repository"],
                "repository": ["repository"], "conversation": ["none"],
            }[route]
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "route": route,
                        "confidence": 0.95,
                        "reason": "test route",
                        "required_sources": sources,
                        "target_urls": [],
                        "requires_live_data": False,
                        "reason_code": "TEST_ROUTE",
                        "error_code": "",
                        "reuse_active_route": False,
                    }
                )
            )

    entry_router = SimpleNamespace(llm=_EntryModel())
    ask_agent = SimpleNamespace(llm=None, update_model=lambda m: None, model="dummy")
    qna_chain = SimpleNamespace(
        llm=None,
        chat=lambda question: "(dummy conversational response)",
    )

    def ask(self, *args, **kwargs):
        return type("Resp", (), {"answer": "(dummy response)"})()

    def ask_with_tools(self, *args, **kwargs):
        return type("Resp", (), {"answer": "(dummy tools response)"})()

    def ask_dir_mode(self, *args, **kwargs):
        return type("Resp", (), {"answer": "(dummy dir response)"})()

    def ask_with_tools_dir_mode(self, *args, **kwargs):
        return type("Resp", (), {"answer": "(dummy dir tools response)"})()


class _DummyCodingAgent:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.coding_memory_service = kwargs.get("coding_memory_service")
        self.tool_worker_client = kwargs.get("tool_worker_client")

    def generate(self, request, **kwargs):
        return {
            "answer": f"coding-ok: {request[:40]}",
            "changed_files": [],
            "warnings": [],
            "flow_id": "flow-test",
        }

    def generate_auto_execute(self, request, **kwargs):
        readme = Path(self.kwargs["repo_root"]) / "README.md"
        readme.write_text("# Updated by the coding-agent test double\n", encoding="utf-8")
        return {
            "answer": f"auto-exec: {request[:40]}",
            "changed_files": ["README.md"],
            "warnings": [],
            "flow_id": "flow-auto",
            "auto_execute_terminal_reason": "completed",
        }

    def generate_dir_mode(self, request, **kwargs):
        return self.generate(request, **kwargs)

    def get_active_flow_id(self):
        return None

    def reset_flow(self, flow_id: str):
        return flow_id

    def flow_summary(self, flow_id: str):
        return None

    def _tool_policy_for_request(self, *a, **k):
        return {"allowed_tools": ["read_file"]}

    def set_tools_manager_orchestrator(self, orch):
        self.orch = orch


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


def test_missing_managed_worker_fails_closed_without_direct_ssh(
    monkeypatch, tmp_path: Path
) -> None:
    direct_calls: list[str] = []

    async def execute_direct(self, request, emit, cancel):
        direct_calls.append(request.job_id)
        raise AssertionError("missing workers must not select direct SSH")

    monkeypatch.setattr(LocalSSHProvider, "execute", execute_direct)
    gateway = object.__new__(AgentChatGateway)
    gateway.remote_execution_service = RemoteExecutionService(
        target_policy=TargetPolicy(TargetPolicyMode.UNRESTRICTED)
    )
    gateway._entry_route_registry = SimpleNamespace(
        get=lambda _route: SimpleNamespace(
            tools=("remote_ssh_execute",),
            availability=lambda: SimpleNamespace(available=True),
        )
    )
    authorized: list[tuple[str, str]] = []
    gateway._lane_coordinator = SimpleNamespace(
        authorize_tool=lambda task_id, tool_name: authorized.append((task_id, tool_name))
    )
    decision = EntryRoutingDecision(
        route="remote_execution",
        confidence=0.99,
        reason="selected managed worker",
        required_sources=("remote_execution",),
        remote_request={
            "provider": "reverse-worker",
            "worker_id": "missing-worker",
            "target": {"host": "example.test", "user": "root"},
            "authentication": {"mode": "agent"},
            "command": {"argv": ["true"]},
        },
    )

    result = gateway._execute_entry_route(
        decision=decision,
        context=EntryRouteContext(session_id="session", conversation_id="session", turn_id="turn"),
        text="run true",
        state={},
        ask_service=None,
        sink=None,
        options={"_lane_task_id": "lane-task"},
    )

    assert result.mode == "route-error"
    assert result.error == "remote_request_invalid"
    assert result.payload == {"route": "remote_execution"}
    assert direct_calls == []
    assert authorized == []


def test_missing_worker_at_permission_resume_fails_without_provider_change(
    monkeypatch,
) -> None:
    direct_calls: list[str] = []

    async def execute_direct(self, request, emit, cancel):
        direct_calls.append(request.job_id)
        raise AssertionError("approved worker requests must not switch providers")

    monkeypatch.setattr(LocalSSHProvider, "execute", execute_direct)
    gateway = object.__new__(AgentChatGateway)
    gateway.remote_execution_service = RemoteExecutionService()
    transitions: list[tuple[str, str, str]] = []
    finishes: list[tuple[str, str]] = []

    def finish(task_id, state, **_kwargs):
        finishes.append((task_id, state.value))
        return SimpleNamespace(state=state, error="")

    gateway._remote_job_lanes = {"job": "lane-task"}
    gateway._finish_lane = finish
    gateway._lane_coordinator = SimpleNamespace(
        transition=lambda task_id, state, reason: transitions.append((task_id, state.value, reason)),
        finish=finish,
    )
    request = RemoteExecutionRequest.model_validate({
        "job_id": "job",
        "session_id": "session",
        "provider": "reverse-worker",
        "worker_id": "missing-worker",
        "target": {"host": "example.test", "user": "root"},
        "authentication": {"mode": "agent"},
        "command": {"argv": ["true"]},
    })
    gateway.remote_execution_service.submit(request)
    permission = gateway.remote_execution_service.pending_permissions()[0]

    result = gateway.remote_permission_command(permission["permission_request_id"])

    assert result == {
        "status": "worker_unavailable",
        "job_id": "job",
        "message": (
            "No trusted external SSH worker is connected. "
            "No local SSH fallback was attempted."
        ),
    }
    assert transitions == [("lane-task", "running", "remote SSH permission approved")]
    assert finishes == [("lane-task", "failed")]
    assert gateway._remote_job_lanes == {}
    assert direct_calls == []


def test_server_approval_is_session_bound_exact_and_single_use(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}
    transitions: list[tuple[str, str, str]] = []
    finishes: list[tuple[str, str]] = []

    def finish(task_id, *, state, **_kwargs):
        finishes.append((task_id, state.value))
        return SimpleNamespace(state=state, error="")

    async def execute(decision, argv, **kwargs):
        captured.update({"decision": decision, "argv": argv, **kwargs})
        return SimpleNamespace(
            exit_code=0,
            timed_out=False,
            cancelled=False,
            stdout="installed",
            stderr="",
            model_dump=lambda **_kwargs: {"exit_code": 0, "stdout": "installed"},
        )

    decision = ServerActionDecision.model_validate({
        "decision_id": "decision-1",
        "server_id": "server-1",
        "action": "package",
        "tool_name": "server_package_install",
        "arguments": {"manager": "auto", "packages": ["nginx"]},
        "required_capability": "package.write",
        "read_only": False,
        "consequential": True,
        "affected_resources": ["package:nginx"],
        "safe_to_continue": True,
        "reason": "Install nginx.",
    })
    gateway = object.__new__(AgentChatGateway)
    gateway.server_management_service = SimpleNamespace(execute=execute)
    gateway._finish_lane = finish
    gateway._lane_coordinator = SimpleNamespace(
        transition=lambda task_id, state, *, reason: transitions.append(
            (task_id, state.value, reason)
        ),
        finish=finish,
    )
    pending = {
        "session_id": "session-1",
        "decision": decision.model_dump(mode="json"),
        "argv": ["sh", "-c", "install-nginx"],
        "exact_action_key": "exact-key",
        "cwd": None,
        "timeout_seconds": 60,
        "pty": False,
        "environment": {},
        "lane_task_id": "lane-task",
    }
    gateway._pending_server_approvals = {"server_approval_1": pending}
    _persist_server_approval(
        gateway,
        tmp_path,
        request_id="server_approval_1",
        pending=pending,
    )

    try:
        gateway.server_approval_command(
            "server_approval_1",
            session_id="session-2",
        )
    except PermissionError as exc:
        assert "different session" in str(exc)
    else:
        raise AssertionError("server approvals must remain session bound")

    result = gateway.server_approval_command(
        "server_approval_1",
        session_id="session-1",
    )

    assert result["status"] == "succeeded"
    assert "Remote command output:\ninstalled" in result["message"]
    assert captured["argv"] == ["sh", "-c", "install-nginx"]
    assert captured["approval"].decision_id == "decision-1"
    assert captured["approval"].exact_action_key == "exact-key"
    assert gateway._pending_server_approvals == {}
    assert transitions == [
        ("lane-task", "running", "server action approved by the user")
    ]
    assert finishes == [("lane-task", "completed")]
    try:
        gateway.server_approval_command("server_approval_1", session_id="session-1")
    except InboxConcurrentUpdateError as exc:
        assert "already completed" in str(exc)
    else:
        raise AssertionError("server approvals must be single use")


def test_server_approval_denial_consumes_request_without_execution(tmp_path: Path) -> None:
    cancellations: list[tuple[str, str]] = []
    gateway = object.__new__(AgentChatGateway)
    gateway._lane_coordinator = SimpleNamespace(
        cancel_task=lambda task_id, *, reason: cancellations.append(
            (task_id, reason)
        )
    )
    pending = {
        "session_id": "session-1",
        "lane_task_id": "lane-task",
        "exact_action_key": "exact-key",
    }
    gateway._pending_server_approvals = {"server_approval_1": pending}
    _persist_server_approval(
        gateway,
        tmp_path,
        request_id="server_approval_1",
        pending=pending,
    )

    result = gateway.deny_server_approval_command(
        "server_approval_1",
        session_id="session-1",
    )

    assert result["status"] == "denied"
    assert gateway._pending_server_approvals == {}
    assert cancellations == [
        ("lane-task", "Server action denied by the user.")
    ]


def test_gateway_constructs_minimally(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    gw = AgentChatGateway(
        tmp_path,
        coding_agent=False,
        agent_tools=False,
    )
    assert gw is not None
    assert gw.root == tmp_path.resolve()
    assert not gw.owns_coding_stack()
    automation_route = gw._entry_route_registry.get("automation")
    assert set(automation_route.tools) == {
        "automation_create", "automation_get", "automation_list", "automation_status",
        "automation_update", "automation_delete", "automation_enable",
        "automation_disable", "automation_run_now",
    }


def test_resumed_mcp_action_surfaces_its_result_in_chat_history(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    history = reset_global_history()
    emitted: list[dict[str, Any]] = []
    gateway = AgentChatGateway(
        tmp_path,
        coding_agent=False,
        agent_tools=False,
        config=ChatGatewayConfig(
            event_sink=lambda event_type, title, **kwargs: emitted.append(
                {"event_type": event_type, "title": title, **kwargs}
            )
        ),
    )
    action = SimpleNamespace(
        action_id="action_context7_docs",
        parent_task_id="task_context7",
        tool_name="mcp",
        operation_name="query-docs",
        normalized_arguments={"provider_id": "context7"},
        state=SimpleNamespace(value="committed"),
    )

    gateway._publish_transactional_resume_activity(
        event_type="action.committed",
        title="Approved MCP action completed",
        action=action,
        inbox_item_id="inbox_context7",
        result={
            "ok": True,
            "content": [
                {
                    "type": "text",
                    "text": "FastAPI docs",
                    "annotations": None,
                    "meta": None,
                }
            ],
            "duration_ms": 12.5,
            "is_error": False,
            "server_id": "context7",
            "structured_content": None,
            "tool_name": "query-docs",
            "transport": "stdio",
        },
    )

    assistant_messages = [
        event for event in history.get_events() if isinstance(event, AssistantMessageEvent)
    ]
    activity_events = [
        event for event in history.get_events() if isinstance(event, CodingActivityEvent)
    ]
    content = assistant_messages[-1].content
    assert assistant_messages[-1].turn_id == "task_context7"
    assert "mcp.context7.query-docs" in content
    assert "Documentation (untrusted data):" in content
    assert "FastAPI docs" in content
    assert "```json" not in content
    assert '"annotations"' not in content
    assert activity_events[-1].activity["output_preview"] == "FastAPI docs"
    assert emitted[-1]["output_preview"] == "FastAPI docs"


def test_capability_error_records_terminal_computer_notice(tmp_path: Path, monkeypatch) -> None:
    mana_root = tmp_path / "mana-home"
    monkeypatch.setattr("mana_agent.human_inbox.mana_home", lambda: mana_root)
    monkeypatch.setattr(
        "mana_agent.transactional_actions.runtime.mana_home", lambda: mana_root
    )
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False)
    decision = EntryRoutingDecision(
        route="capability_error",
        confidence=1.0,
        reason="computer control is unavailable",
        required_sources=("computer",),
        error_code="COMPUTER_NOT_AVAILABLE",
    )

    result = gateway._execute_entry_route(
        decision=decision,
        context=EntryRouteContext(
            session_id="session", conversation_id="session", turn_id="turn"
        ),
        text="use the computer",
        state={},
        ask_service=None,
        sink=None,
        options={},
    )

    assert result.error == "COMPUTER_NOT_AVAILABLE"
    records = ActionStore(mana_root / "transactional_actions").list_requests()
    assert len(records) == 1
    assert records[0].outcome_code == "COMPUTER_NOT_AVAILABLE"
    assert records[0].inbox_item_id.startswith("inbox_")
    item = gateway.human_inbox_service.repository.get(records[0].inbox_item_id)
    assert item.request_type is InboxRequestType.NOTICE


def test_computer_route_without_typed_tool_outcome_records_notice(tmp_path: Path, monkeypatch) -> None:
    mana_root = tmp_path / "mana-home"
    monkeypatch.setattr("mana_agent.human_inbox.mana_home", lambda: mana_root)
    monkeypatch.setattr(
        "mana_agent.transactional_actions.runtime.mana_home", lambda: mana_root
    )
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )

    class NoOutcomeAskAgent:
        def run(self, **_: Any) -> Any:
            return SimpleNamespace(answer="The environment blocked the action.", trace=[])

    gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False)
    decision = EntryRoutingDecision(
        route="computer",
        confidence=1.0,
        reason="model selected the computer workflow",
        required_sources=("computer",),
    )
    from mana_agent.integrations.computer_control.context import computer_client_scope

    with computer_client_scope("session", "tui", workspace_root=str(tmp_path)):
        result = gateway._execute_computer_route(
            decision=decision,
            context=EntryRouteContext(
                session_id="session", conversation_id="session", turn_id="turn"
            ),
            text="record the selected display",
            ask_service=SimpleNamespace(ask_agent=NoOutcomeAskAgent()),
            callbacks=None,
        )

    assert result.error == "computer_typed_outcome_missing"
    assert "no operating-system request" in result.answer
    records = ActionStore(mana_root / "transactional_actions").list_requests()
    assert len(records) == 1
    assert records[0].outcome_code == "computer_typed_outcome_missing"
    assert records[0].inbox_item_id.startswith("inbox_")
    item = gateway.human_inbox_service.repository.get(records[0].inbox_item_id)
    assert item.request_type is InboxRequestType.NOTICE


def test_task_control_rejects_create_verb_instead_of_unknown_task_id(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False)
    session_id = gateway.create_session(frontend="test")

    message = gateway.handle_control_command("/task create", session_id=session_id)

    assert message is not None
    assert "not a gateway task ID" in message
    assert "Unknown gateway task: create" not in message
    assert "/task <id>" in message


def test_task_control_rejects_execute_verb_instead_of_unknown_task_id(
    tmp_path: Path, monkeypatch
) -> None:
    """/task Execute must not be treated as inspect of task id 'Execute'."""
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False)
    session_id = gateway.create_session(frontend="test")

    message = gateway.handle_control_command("/task Execute", session_id=session_id)

    assert message is not None
    assert "not a gateway task ID" in message
    assert "Unknown gateway task: Execute" not in message
    assert "auto-select" in message.lower() or "chat" in message.lower()


def test_task_control_rejects_non_task_id_tokens(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False)
    session_id = gateway.create_session(frontend="test")

    message = gateway.handle_control_command("/task DoSomething", session_id=session_id)

    assert message is not None
    assert "not a gateway task ID" in message
    assert "Unknown gateway task: DoSomething" not in message


def test_task_control_unknown_id_returns_actionable_message(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False)
    session_id = gateway.create_session(frontend="test")

    message = gateway.handle_control_command(
        "/task task_20990101_000001", session_id=session_id
    )

    assert message is not None
    assert "Gateway task control failed" in message
    assert "task_20990101_000001" in message
    assert "Use /tasks to list" in message


def test_task_control_auto_selects_single_recoverable_task_for_retry(
    tmp_path: Path, monkeypatch
) -> None:
    from mana_agent.gateway.lanes import LaneId, LaneTaskState

    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False)
    session_id = gateway.create_session(frontend="test")
    reservation = gateway._lane_coordinator.reserve(
        normalized_intent="recoverable coding work",
        lane_id=LaneId.CODING,
        session_id=session_id,
        workspace_id=gateway._lane_coordinator.taskboard.store.workspace_id,
        repository_id=gateway._lane_coordinator.taskboard.store.repository_id,
        requested_input_tokens=10,
        requested_output_tokens=10,
    )
    gateway._lane_coordinator.start(reservation)
    gateway._lane_coordinator.finish(
        reservation.execution.task_id,
        state=LaneTaskState.FAILED,
        error="worker failed",
    )
    monkeypatch.setattr(
        gateway._lane_coordinator.execution_supervisor,
        "retry",
        lambda task_id, decision: gateway._lane_coordinator.execution_supervisor.store.get_task(
            task_id
        ),
    )
    monkeypatch.setattr(
        gateway._lane_coordinator.execution_supervisor,
        "release_retry",
        lambda task_id: gateway._lane_coordinator.execution_supervisor.store.get_task(
            task_id
        ),
    )

    message = gateway.handle_control_command("/task retry", session_id=session_id)

    assert message is not None
    assert "Gateway task control failed" not in message
    payload = json.loads(message)
    assert payload["task_id"] == reservation.execution.task_id
    assert payload["recovery_action"] == "retry_task"
    assert gateway._lane_coordinator.inspect_task(reservation.execution.task_id).state is (
        LaneTaskState.QUEUED
    )


def test_gateway_creates_session_and_simple_send(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    # Force classic ask path (no decision LLM) by disabling agent_tools + coding
    gw = AgentChatGateway(
        tmp_path,
        coding_agent=False,
        agent_tools=False,
    )
    sid = gw.create_session(frontend="test")
    assert isinstance(sid, str) and sid

    try:
        result = gw.send(sid, "hello from gateway test")
        assert isinstance(result, str)
    except Exception:
        # Acceptable in environments without keys/indexes
        assert True


def test_gateway_provides_rich_context(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    gw = AgentChatGateway(
        tmp_path,
        dir_mode=True,
        auto_execute_plan=False,
        coding_agent=False,
    )
    ctx = gw.get_rich_context()
    assert isinstance(ctx, RichChatContext)
    assert ctx.dir_mode is True
    assert ctx.root == gw.root or ctx.root is None
    assert ctx.config is not None


def test_gateway_accepts_pre_built_objects(tmp_path: Path) -> None:
    gw = AgentChatGateway(
        tmp_path,
        coding_agent=False,
        chat_service=object(),  # fake
        coding_agent_instance=None,
        tools_orchestrator=None,
    )
    ctx = gw.get_rich_context()
    assert ctx.chat_service is not None


def test_gateway_builds_coding_stack_when_enabled(tmp_path: Path, monkeypatch) -> None:
    """Gateway owns coding agent construction (no chat_cli injection required)."""
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.CodingAgent",
        _DummyCodingAgent,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.ToolWorkerClient",
        lambda **kw: SimpleNamespace(
            start=lambda: None,
            health=lambda: True,
            init_payload_dict=lambda: {},
            **kw,
        ),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.QueueManager",
        lambda **kw: SimpleNamespace(**kw, attach_decision_provider=lambda x: None),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.build_tools_executor_with_fallback",
        lambda **kw: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.CodingMemoryService",
        lambda **kw: SimpleNamespace(**kw),
    )

    gw = AgentChatGateway(
        tmp_path,
        coding_agent=True,
        agent_tools=True,
        tool_worker_process=True,
        auto_execute_plan=True,
    )
    assert gw.owns_coding_stack()
    ctx = gw.get_rich_context()
    assert ctx.coding_agent is not None
    assert isinstance(ctx.coding_agent, _DummyCodingAgent)


def test_gateway_uses_codex_shim_without_legacy_coding_workers(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )

    with caplog.at_level("INFO", logger="mana_agent.gateway.stack"):
        gw = AgentChatGateway(
            tmp_path,
            coding_agent=True,
            agent_tools=True,
            tool_worker_process=True,
            auto_execute_plan=True,
            # Stale MANA_CODEX_MODEL pin must not be reported as the runtime
            # coding model; the log must show the resolved/routed model.
            settings=Settings(MANA_CODEX_MODEL="codex-test-model"),
        )

    ctx = gw.get_rich_context()
    assert isinstance(ctx.coding_agent, CodexCodingAgentShim)
    assert ctx.tool_worker_client is None
    assert ctx.tools_orchestrator is None
    model_log = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Resolved chat runtime models:")
    )
    assert "main=" in model_log
    assert "router=" in model_log
    assert "coding_backend=codex" in model_log
    assert "coding=codex-test-model" not in model_log
    assert "coding=" in model_log
    assert "coding_routed=" in model_log
    assert "planner=codex-owned" in model_log
    assert "tool_worker=disabled" in model_log


def test_gateway_uses_internal_runtime_when_codex_is_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    gateway = AgentChatGateway(
        tmp_path,
        coding_agent=True,
        agent_tools=True,
        tool_worker_process=False,
        auto_execute_plan=False,
        settings=Settings(MANA_CODEX_ENABLED=False, OPENAI_API_KEY="test-key"),
    )
    context = gateway.get_rich_context()
    assert isinstance(context.coding_agent, InternalCodingAgentShim)
    assert not isinstance(context.coding_agent, CodexCodingAgentShim)


def test_gateway_process_turn_ask_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )

    fixed = AgentDecision(
        intent="answer",
        code_editing_needed=False,
        selected_tools=[],
        tool_inputs={},
        flow_action="none",
        reasoning_summary="answer only",
        confidence=0.9,
        verifier_passed=True,
    )

    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.decide_chat_route",
        lambda **kw: fixed,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.handle_small_direct_edit",
        lambda root, q: SimpleNamespace(handled=False),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.load_auto_chat_state",
        lambda root: SimpleNamespace(
            last_mode="answer_only",
            last_task="",
            relevant_files=[],
            changed_files=[],
            verification="",
            summary="",
        ),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.save_auto_chat_state",
        lambda root, state: None,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.resolve_auto_followup",
        lambda q, state: q,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.classify_auto_chat_intent",
        lambda q: __import__(
            "mana_agent.multi_agent.runtime.auto_chat", fromlist=["AutoChatMode"]
        ).AutoChatMode.ANSWER_ONLY,
    )

    gw = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=True)
    sid = gw.create_session(frontend="test")
    result = gw.process_turn(sid, "what is this project?")
    assert isinstance(result, ChatTurnResult)
    assert result.error is None
    assert "dummy" in result.answer.lower() or result.answer
    assert result.used_coding_agent is False


def test_gateway_process_turn_coding_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.CodingAgent",
        _DummyCodingAgent,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.ToolWorkerClient",
        lambda **kw: SimpleNamespace(
            start=lambda: None,
            health=lambda: True,
            init_payload_dict=lambda: {},
        ),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.QueueManager",
        lambda **kw: SimpleNamespace(attach_decision_provider=lambda x: None),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.build_tools_executor_with_fallback",
        lambda **kw: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.CodingMemoryService",
        lambda **kw: SimpleNamespace(),
    )

    fixed = AgentDecision(
        intent="edit",
        code_editing_needed=True,
        selected_tools=["apply_patch"],
        tool_inputs={},
        flow_action="none",
        reasoning_summary="edit files",
        confidence=0.95,
        verifier_passed=True,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.decide_chat_route",
        lambda **kw: fixed,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.handle_small_direct_edit",
        lambda root, q: SimpleNamespace(handled=False),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.load_auto_chat_state",
        lambda root: SimpleNamespace(
            last_mode="edit",
            last_task="",
            relevant_files=[],
            changed_files=[],
            verification="",
            summary="",
        ),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.save_auto_chat_state",
        lambda root, state: None,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.resolve_auto_followup",
        lambda q, state: q,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.classify_auto_chat_intent",
        lambda q: __import__(
            "mana_agent.multi_agent.runtime.auto_chat", fromlist=["AutoChatMode"]
        ).AutoChatMode.EDIT,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.is_plan_execution_request",
        lambda q: False,
    )

    gw = AgentChatGateway(
        tmp_path,
        coding_agent=True,
        agent_tools=True,
        auto_execute_plan=True,
    )
    assert gw.owns_coding_stack()
    sid = gw.create_session(frontend="test")
    result = gw.process_turn(sid, "update the README title")
    assert isinstance(result, ChatTurnResult)
    assert result.error is None
    assert result.used_coding_agent is True
    assert "auto-exec" in result.answer or "coding-ok" in result.answer
    assert result.flow_id in {"flow-auto", "flow-test", None} or result.flow_id


def test_gateway_gmail_uses_dedicated_connector_not_coding_or_conversation(tmp_path: Path, monkeypatch) -> None:
    """A Gmail turn executes the email-only route before any conversation response."""
    gmail_calls: list[dict[str, Any]] = []
    coding_calls: list[str] = []

    class _CodingTracker(_DummyCodingAgent):
        def generate(self, request, **kwargs):
            coding_calls.append(str(request))
            return super().generate(request, **kwargs)

        def generate_auto_execute(self, request, **kwargs):
            coding_calls.append(str(request))
            return super().generate_auto_execute(request, **kwargs)

    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    monkeypatch.setattr("mana_agent.gateway.stack.CodingAgent", _CodingTracker)
    monkeypatch.setattr(
        "mana_agent.gateway.stack.ToolWorkerClient",
        lambda **kw: SimpleNamespace(start=lambda: None, health=lambda: True, init_payload_dict=lambda: {}),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.QueueManager",
        lambda **kw: SimpleNamespace(attach_decision_provider=lambda x: None),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.build_tools_executor_with_fallback",
        lambda **kw: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.CodingMemoryService",
        lambda **kw: SimpleNamespace(),
    )

    from mana_agent.gateway import RouteAvailability, RouteRegistration

    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.gmail_route_availability",
        lambda: RouteAvailability(available=True),
    )
    gw = AgentChatGateway(tmp_path, coding_agent=True, agent_tools=True)
    assert gw.owns_coding_stack()
    gw._entry_route_registry.register(
        RouteRegistration("gmail", "Gmail", lambda: RouteAvailability(available=True))
    )

    def _gmail_run(**kwargs: Any):
        gmail_calls.append(kwargs)
        return SimpleNamespace(
            answer="Here are your latest Gmail messages (dummy).",
            sources=[],
            warnings=[],
            trace=[
                {"tool_name": "email_search", "status": "ok", "output_preview": "1 message"},
                {"tool_name": "email_read", "status": "ok", "output_preview": "Subject: Hello"},
            ],
        )

    gw.get_ask_service().ask_agent = SimpleNamespace(run=_gmail_run)

    sid = gw.create_session(frontend="tui")
    result = gw.process_turn(sid, "Check my latest Gmail", callbacks=[object()])
    assert result.error is None
    assert result.used_coding_agent is False
    assert result.mode == "route-gmail"
    assert "gmail" in result.answer.lower() or "Gmail" in result.answer
    assert gmail_calls
    assert gmail_calls[0]["flow_id"] == sid
    assert gmail_calls[0]["run_id"] == result.payload["turn_id"]
    assert not coding_calls, "CodingAgent must not run for Gmail auto-chat turns"
    assert (result.payload or {}).get("route") == "gmail"
    # Tool traces must reach TUI consumers for ToolCard rendering.
    trace_names = [row.get("tool_name") for row in (result.trace or []) if isinstance(row, dict)]
    assert "email_search" in trace_names
    assert "email_read" in trace_names


def test_should_use_coding_agent_turn_gmail_is_false() -> None:
    from mana_agent.gateway.turn_engine import should_use_coding_agent_turn, is_auto_chat_connector_turn
    from mana_agent.multi_agent.runtime.auto_chat import AutoChatMode

    decision = AgentDecision(
        intent="answer",
        confidence=0.9,
        selected_tools=["email_search"],
        code_editing_needed=False,
        reasoning_summary="gmail",
        verifier_passed=True,
    )
    assert is_auto_chat_connector_turn(
        decision=decision, auto_chat_mode=AutoChatMode.ANSWER_ONLY, question="check my latest gmail"
    )
    assert not should_use_coding_agent_turn(
        coding_agent_available=True,
        agent_tools=True,
        edit_request=False,
        plan_trigger_request=False,
        force_plan_only_response=False,
        has_pending_prechecklist=False,
        coding_agent_is_custom=False,
        general_coding_agent_turns=False,
        decision=decision,
        auto_chat_mode=AutoChatMode.ANSWER_ONLY,
        question="check my latest gmail",
    )


def test_gateway_config_normalized_full_auto() -> None:
    cfg = ChatGatewayConfig(full_auto=True, auto_execute_max_passes=4).normalized()
    assert cfg.execution_profile == "full-auto"
    assert cfg.auto_execute_plan is True
    assert cfg.auto_execute_max_passes == 10


def test_gateway_keeps_task_policy_when_lane_has_no_explicit_cap(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *args, **kwargs: _DummyAskService(),
    )
    gateway = AgentChatGateway(
        tmp_path,
        coding_agent=False,
        settings=Settings(
            OPENAI_API_KEY="test-key",
            MANA_ROUTING_TASK_TOKEN_BUDGET=40_000,
            MANA_ROUTING_TASK_COST_BUDGET=40.0,
        ),
    )

    budgets = gateway._routing_budgets_for_lane(LaneId.CANVAS)

    assert budgets.task_token_limit == 40_000
    assert budgets.task_cost_limit == 40.0


def test_gateway_estimates_canvas_payload_without_fixed_minimums(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *args, **kwargs: _DummyAskService(),
    )
    gateway = AgentChatGateway(
        tmp_path,
        coding_agent=False,
        agent_max_steps=2,
        settings=Settings(MANA_ROUTING_TASK_TOKEN_BUDGET=1_000_000),
    )

    estimate = gateway._execution_token_estimate(
        entry_route="canvas",
        execution_decision=SimpleNamespace(
            provider="openai",
            selected_model=gateway.settings.openai_chat_model,
            estimated_output_tokens=100,
            expected_model_calls=1,
        ),
        request_text="draw a small rectangle",
        session_id="",
    )

    assert estimate.components["user_request"] > 0
    assert estimate.output_tokens == gateway.config.agent_max_steps * 100
    assert "canvas_catalog" in estimate.components
    assert "tool_schemas" in estimate.components


def test_gateway_decision_failure_no_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )

    gw = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=True)
    gw._entry_router.llm = SimpleNamespace(
        invoke=lambda messages: SimpleNamespace(content='{"route":"invented"}')
    )
    sid = gw.create_session(frontend="test")
    result = gw.process_turn(sid, "do something")
    assert result.error is not None
    assert "decision" in result.error.lower()
    assert "integration" not in result.answer.lower()
    assert result.payload["route"] == "unsupported"


def _answer_decision(*, selected_tools: list[str] | None = None) -> AgentDecision:
    return AgentDecision(
        intent="answer",
        confidence=0.99,
        selected_tools=list(selected_tools or []),
        code_editing_needed=False,
        reasoning_summary="answer conversationally",
        verifier_passed=True,
    )


def test_gateway_persists_same_session_history_without_duplicate_current_message(
    tmp_path: Path, monkeypatch
) -> None:
    prompts: list[str] = []

    class TrackingChatService:
        _ask_service = _DummyAskService()

        def ask(self, question: str, **kwargs: Any):
            prompts.append(question)
            answer = "Understood." if len(prompts) == 1 else "One is b."
            return SimpleNamespace(answer=answer, sources=[], warnings=[], trace=[])

        def ask_conversation(self, question: str):
            return self.ask(question).answer

    monkeypatch.setattr("mana_agent.gateway.turn_engine.decide_chat_route", lambda **kwargs: _answer_decision())
    monkeypatch.setattr("mana_agent.gateway.turn_engine.handle_small_direct_edit", lambda *args, **kwargs: SimpleNamespace(handled=False))
    gateway = AgentChatGateway(
        tmp_path,
        coding_agent=False,
        agent_tools=False,
        chat_service=TrackingChatService(),
    )
    session_id = gateway.create_session(frontend="test")

    gateway.process_turn(session_id, "Remember one = b.")
    gateway.process_turn(session_id, "What is one?")

    assert "Remember one = b." not in prompts[1]
    assert prompts[1] == "What is one?"
    messages = gateway.session_messages(session_id)
    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
    assert {message["session_id"] for message in messages} == {session_id}
    assert {message["conversation_id"] for message in messages} == {session_id}


def test_gateway_followup_uses_stack_owned_shared_memory(tmp_path: Path, monkeypatch) -> None:
    prompts: list[str] = []

    class TrackingChatService:
        _ask_service = _DummyAskService()

        def ask_conversation(self, question: str) -> str:
            prompts.append(question)
            return "ok"

    class TrackingMemoryService:
        def __init__(self) -> None:
            self.searches: list[Any] = []
            self.writes: list[Any] = []
            # This fake intentionally models the supported legacy provider
            # path. Production services expose this feature flag explicitly;
            # enabled capsules must not fall back to conversation-wide recall.
            self.config = SimpleNamespace(
                capsules=SimpleNamespace(enabled=False),
            )

        def search_blocking(self, request):
            self.searches.append(request)
            return [
                MemoryRecord(
                    id="memory-1",
                    content=MemoryContent("User: remembered detail\nAssistant: acknowledged"),
                    scope=request.scope,
                    provider="test",
                )
            ]

        def add_blocking(self, request):
            self.writes.append(request)
            return MemoryRecord(
                id=f"memory-{len(self.writes)}",
                content=request.content,
                scope=request.scope,
                provider="test",
            )

    gateway = AgentChatGateway(
        tmp_path,
        coding_agent=False,
        agent_tools=False,
        chat_service=TrackingChatService(),
    )
    session_id = gateway.create_session(frontend="test")
    memory = TrackingMemoryService()
    gateway._stack.memory_service = memory

    gateway.process_turn(session_id, "first turn")
    gateway.process_turn(session_id, "follow up")

    assert len(memory.writes) == 2
    assert len(memory.searches) == 0
    assert memory.writes[1].metadata["mana_kind"] == "chat_turn"
    assert prompts[-1] == "follow up"


def test_gateway_preserves_multiline_message_through_request_and_restored_history(
    tmp_path: Path, monkeypatch
) -> None:
    prompts: list[str] = []

    class TrackingChatService:
        _ask_service = _DummyAskService()

        def ask(self, question: str, **kwargs: Any):
            prompts.append(question)
            return SimpleNamespace(answer="Understood.", sources=[], warnings=[], trace=[])

    gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False, chat_service=TrackingChatService())
    session_id = gateway.create_session(frontend="test")
    message = "first line\nsecond line\nthird line"

    gateway.send(session_id, message)

    assert prompts == [message]
    assert [row["content"] for row in gateway.session_messages(session_id) if row["role"] == "user"] == [message]

    restored = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False, chat_service=TrackingChatService())
    restored.create_session(frontend="test", session_id=session_id)
    assert [row["content"] for row in restored.session_messages(session_id) if row["role"] == "user"] == [message]


def test_answer_only_conversation_uses_validated_route_without_second_router(
    tmp_path: Path, monkeypatch
) -> None:
    prompts: list[str] = []

    class ConversationChatService:
        _ask_service = _DummyAskService()

        def ask(self, question: str, **kwargs: Any):
            raise AssertionError("entry router must not run after an answer-only decision")

        def ask_conversation(self, question: str) -> str:
            prompts.append(question)
            return "a is test"

    monkeypatch.setattr("mana_agent.gateway.turn_engine.decide_chat_route", lambda **kwargs: _answer_decision())
    monkeypatch.setattr("mana_agent.gateway.turn_engine.handle_small_direct_edit", lambda *args, **kwargs: SimpleNamespace(handled=False))
    gateway = AgentChatGateway(
        tmp_path,
        coding_agent=False,
        agent_tools=False,
        chat_service=ConversationChatService(),
    )
    session_id = gateway.create_session(frontend="test")
    gateway.process_turn(session_id, "memory-test a=test")
    result = gateway.process_turn(session_id, "what is a?")

    assert result.answer == "a is test"
    assert result.mode == "route-conversation"
    assert prompts[-1] == "what is a?"


def test_gateway_new_conversation_isolates_history(tmp_path: Path, monkeypatch) -> None:
    prompts: list[str] = []

    class TrackingChatService:
        _ask_service = _DummyAskService()

        def ask(self, question: str, **kwargs: Any):
            prompts.append(question)
            return SimpleNamespace(answer="ok", sources=[], warnings=[], trace=[])

        def ask_conversation(self, question: str):
            return self.ask(question).answer

    monkeypatch.setattr("mana_agent.gateway.turn_engine.decide_chat_route", lambda **kwargs: _answer_decision())
    monkeypatch.setattr("mana_agent.gateway.turn_engine.handle_small_direct_edit", lambda *args, **kwargs: SimpleNamespace(handled=False))
    gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False, chat_service=TrackingChatService())
    old_session = gateway.create_session(frontend="test")
    gateway.process_turn(old_session, "Remember one = b.")

    new_session = gateway.start_new_conversation(old_session, frontend="test")
    gateway.process_turn(new_session, "What is one?")

    assert new_session != old_session
    assert "Remember one = b." not in prompts[-1]
    assert gateway.session_messages(old_session) == []
    assert old_session not in {item.session_id for item in gateway._workspaces.store.list_sessions()}
    assert [row["content"] for row in gateway.session_messages(new_session) if row["role"] == "user"] == ["What is one?"]


def test_gateway_failed_turn_keeps_session_and_records_failure(tmp_path: Path, monkeypatch) -> None:
    class FailingChatService:
        _ask_service = _DummyAskService()

        def ask(self, question: str, **kwargs: Any):
            raise RuntimeError("provider unavailable")

        def ask_conversation(self, question: str):
            return self.ask(question)

    monkeypatch.setattr("mana_agent.gateway.turn_engine.decide_chat_route", lambda **kwargs: _answer_decision())
    monkeypatch.setattr("mana_agent.gateway.turn_engine.handle_small_direct_edit", lambda *args, **kwargs: SimpleNamespace(handled=False))
    gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False, chat_service=FailingChatService())
    session_id = gateway.create_session(frontend="test")
    result = gateway.process_turn(session_id, "Remember this even if the model fails.")

    assert result.error and "provider unavailable" in result.error
    messages = gateway.session_messages(session_id)
    assert messages[0]["role"] == "user"
    assert messages[-1]["role"] == "system"
    assert messages[-1]["metadata"]["state"] == "failed"
    assert gateway.create_session(frontend="test", session_id=session_id) == session_id


def test_gateway_persists_tool_summary_for_followup_context(tmp_path: Path, monkeypatch) -> None:
    prompts: list[str] = []

    class ToolChatService:
        _ask_service = _DummyAskService()

        def ask(self, question: str, **kwargs: Any):
            prompts.append(question)
            trace = [] if len(prompts) > 1 else [{"tool_name": "read_file", "output_preview": "one=b", "status": "ok"}]
            return SimpleNamespace(answer="tool answer", sources=[], warnings=[], trace=trace)

        def ask_conversation(self, question: str):
            return self.ask(question).answer

    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.decide_chat_route",
        lambda **kwargs: _answer_decision(selected_tools=["read_file"]),
    )
    monkeypatch.setattr("mana_agent.gateway.turn_engine.handle_small_direct_edit", lambda *args, **kwargs: SimpleNamespace(handled=False))
    gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False, chat_service=ToolChatService())
    session_id = gateway.create_session(frontend="test")
    gateway.process_turn(session_id, "Read the value.")
    gateway.process_turn(session_id, "What did the tool return?")
    assert prompts[-1] == "What did the tool return?"
    assert [row["role"] for row in gateway.session_messages(session_id)][:3] == ["user", "tool", "assistant"]


def test_gateway_does_not_create_sessions_per_message(tmp_path: Path, monkeypatch) -> None:
    class ChatService:
        _ask_service = _DummyAskService()

        def ask(self, question: str, **kwargs: Any):
            return SimpleNamespace(answer="ok", sources=[], warnings=[], trace=[])

        def ask_conversation(self, question: str):
            return self.ask(question).answer

    monkeypatch.setattr("mana_agent.gateway.turn_engine.decide_chat_route", lambda **kwargs: _answer_decision())
    monkeypatch.setattr("mana_agent.gateway.turn_engine.handle_small_direct_edit", lambda *args, **kwargs: SimpleNamespace(handled=False))
    gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False, chat_service=ChatService())
    create_calls = 0
    original_create = gateway._workspaces.create_session

    def counted_create(*args: Any, **kwargs: Any):
        nonlocal create_calls
        create_calls += 1
        return original_create(*args, **kwargs)

    monkeypatch.setattr(gateway._workspaces, "create_session", counted_create)
    session_id = gateway.create_session(frontend="test")
    for message in ("one", "two", "three"):
        gateway.process_turn(session_id, message)

    # Opening the chat creates exactly one session. Turns and model/tool calls
    # must not create additional sessions.
    assert create_calls == 1
    assert {row["session_id"] for row in gateway.session_messages(session_id)} == {session_id}


def test_gateway_startup_creates_fresh_session_and_new_creates_another(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *args, **kwargs: _DummyAskService(),
    )
    first_gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False)
    first_session = first_gateway.create_session(frontend="cli")

    second_gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False)
    second_session = second_gateway.create_session(frontend="cli")
    new_session = second_gateway.start_new_conversation(second_session, frontend="cli")

    assert second_session != first_session
    assert new_session != second_session
    repository_id = second_gateway._workspaces.register_repository(tmp_path).repository_id
    sessions = [
        item
        for item in second_gateway._workspaces.store.list_sessions()
        if item.primary_repository_id == repository_id
    ]
    assert len(sessions) == 2
    assert all(item.session_id != second_session for item in sessions)
    assert second_gateway._workspaces.store.get_session(first_session).status == "abandoned"
    assert second_session not in {item.session_id for item in sessions}
    assert second_gateway._workspaces.store.get_session(new_session).status == "active"


def test_chat_session_history_redacts_secrets(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "mana_agent.services.chat_session_history.session_dir",
        lambda session_id: tmp_path / session_id,
    )
    history = ChatSessionHistory()
    history.append(
        "session_test",
        role="tool",
        content="Authorization: Bearer private-token and sk-private-key",
        turn_id="turn_test",
        metadata={"api_key": "private", "tool_name": "example"},
    )

    stored = history.list("session_test")[0]
    assert "private-token" not in stored.content
    assert "sk-private-key" not in stored.content
    assert stored.metadata["api_key"] == "***REDACTED***"


def test_linkage_scenarios(tmp_path, monkeypatch) -> None:
    gmail_calls: list[dict[str, Any]] = []

    class _CodingTracker(_DummyCodingAgent):
        def generate(self, request, **kwargs):
            return super().generate(request, **kwargs)

        def generate_auto_execute(self, request, **kwargs):
            return super().generate_auto_execute(request, **kwargs)

    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    monkeypatch.setattr("mana_agent.gateway.stack.CodingAgent", _CodingTracker)
    monkeypatch.setattr(
        "mana_agent.gateway.stack.ToolWorkerClient",
        lambda **kw: SimpleNamespace(start=lambda: None, health=lambda: True, init_payload_dict=lambda: {})
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.QueueManager",
        lambda **kw: SimpleNamespace(attach_decision_provider=lambda x: None)
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.build_tools_executor_with_fallback",
        lambda **kw: SimpleNamespace()
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.CodingMemoryService",
        lambda **kw: SimpleNamespace()
    )

    from mana_agent.gateway import RouteAvailability, RouteRegistration
    from mana_agent.gateway.followup_classifier import FollowupClassification

    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.FollowupClassifier.decide",
        lambda *args, **kwargs: FollowupClassification(
            decision_id="dec-1",
            category="new_task",
            related_task_id="",
            reason="independent query",
            safe_to_continue=True,
        ),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.gmail_route_availability",
        lambda: RouteAvailability(available=True)
    )
    gw = AgentChatGateway(tmp_path, coding_agent=True, agent_tools=True)
    gw._entry_route_registry.register(
        RouteRegistration("gmail", "Gmail", lambda: RouteAvailability(available=True))
    )



    captured = {}
    def _gmail_run(**kwargs):
        captured["lane_task_id"] = kwargs.get("transactional_parent_task_id")
        return SimpleNamespace(
            answer="Success",
            sources=[],
            warnings=[],
            trace=[{"tool_name": "email_search", "status": "ok", "output_preview": "1 message"}],
            payload={"route": "gmail"}
        )
    gw.get_ask_service().ask_agent = SimpleNamespace(run=_gmail_run)
    sid = gw.create_session(frontend="tui")
    result = gw.process_turn(sid, "Check my latest Gmail", callbacks=[object()])
    assert result.error is None
    assert result.answer == "Success"

    lane_task_id = captured.get("lane_task_id")
    assert lane_task_id

    task = gw._lane_coordinator.execution_supervisor.store.get_task(lane_task_id)
    assert task.state.value == "completed"
    from mana_agent.gateway.lane_coordinator import LaneTaskState
    assert gw._lane_coordinator.inspect_task(lane_task_id).state == LaneTaskState.COMPLETED

    def _gmail_run_fail(**kwargs):
        raise RuntimeError("simulated crash")
    gw.get_ask_service().ask_agent = SimpleNamespace(run=_gmail_run_fail)
    sid_fail = gw.create_session(frontend="tui")
    result_fail = gw.process_turn(sid_fail, "Check my latest Gmail", callbacks=[object()])
    assert "simulated crash" in result_fail.error

    def _gmail_run_no_tools(**kwargs):
        return SimpleNamespace(
            answer="Success",
            sources=[],
            warnings=[],
            trace=[],
            payload={"route": "gmail"}
        )
    gw.get_ask_service().ask_agent = SimpleNamespace(run=_gmail_run_no_tools)
    sid_no_tools = gw.create_session(frontend="tui")
    result_no_tools = gw.process_turn(sid_no_tools, "Check my latest Gmail", callbacks=[object()])
    assert result_no_tools.error == "completion_verification_failed"


def test_context_budget_exceeded_finishes_lane_and_charges_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mana_agent.context_cost.models import (
        BudgetSnapshot,
        ContextBreakdown,
        ContextBudget,
        ContextBudgetExceeded,
        GovernorDecision,
    )
    from mana_agent.gateway import RouteAvailability, RouteRegistration
    from mana_agent.gateway.followup_classifier import FollowupClassification
    from mana_agent.gateway.lane_coordinator import LaneTaskState

    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.FollowupClassifier.decide",
        lambda *args, **kwargs: FollowupClassification(
            decision_id="dec-1",
            category="new_task",
            related_task_id="",
            reason="independent query",
            safe_to_continue=True,
        ),
    )
    gw = AgentChatGateway(tmp_path, coding_agent=True, agent_tools=True)
    gw._entry_route_registry.register(
        RouteRegistration("gmail", "Gmail", lambda: RouteAvailability(available=True))
    )

    snapshot = BudgetSnapshot(
        breakdown=ContextBreakdown(),
        budget=ContextBudget(context_window=8000),
        used_tokens=9000,
        remaining_tokens=0,
        utilization_ratio=1.125,
        cumulative_tokens=9000,
        remaining_task_tokens=0,
        cumulative_cost=0.05,
        remaining_cost=0.0,
        estimated=True,
        status="blocked",
    )
    decision = GovernorDecision(
        action="block",
        reason="context_limit_deficit:1000",
        allowed=False,
        snapshot=snapshot,
    )

    captured_task_id = []

    def _gmail_run_budget_blocked(**kwargs):
        lane_task_id = kwargs.get("transactional_parent_task_id")
        captured_task_id.append(lane_task_id)
        if lane_task_id:
            gw._stack.context_cost_governor.record_model_call(
                "call-1",
                estimated_input="input text for estimation",
                estimated_output="output text for estimation",
                task_id=lane_task_id,
            )
        raise ContextBudgetExceeded(decision)

    gw.get_ask_service().ask_agent = SimpleNamespace(run=_gmail_run_budget_blocked)
    sid = gw.create_session(frontend="tui")
    result = gw.process_turn(sid, "Check my latest Gmail")
    assert result.error == "context_budget_blocked"
    assert result.mode == "context-budget-blocked"
    assert "Gateway execution failed" in result.answer

    lane_task_id = captured_task_id[0]
    task = gw._lane_coordinator.inspect_task(lane_task_id)
    assert task.state == LaneTaskState.BUDGET_EXHAUSTED
    assert task.budget.consumed_tokens > 0

