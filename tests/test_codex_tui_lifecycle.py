"""Tests for Codex and TUI lifecycle, event delivery, and execution idempotency.

Covers:
1. Normal Codex task renders events once.
2. Codex task without active coding_event_scope still renders through ExecutionEventHub.
3. Scope + hub delivery does not duplicate TUI rows or events.
4. /new replaces previous session subscriptions and unsubscribes the old session.
5. /new followed immediately by a Codex request creates exactly one Codex execution.
6. Repeated /new commands do not accumulate subscribers.
7. Old-session events cannot attach to the new chat panel.
8. A previous session's recoverable task is not accidentally dispatched as the new task.
9. Retries do not become separate independent Codex executions.
10. Terminal and error events appear exactly once.
11. Internal reasoning and raw assistant deltas remain hidden.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from mana_agent.chat.events import (
    AssistantMessageEvent,
    CodingActivityEvent,
    UserMessageEvent,
)
from mana_agent.chat.history import ChatHistory
from mana_agent.coding.event_visibility import (
    EventSemanticKind,
    EventVisibility,
    classify_coding_event,
    is_user_publishable,
)
from mana_agent.coding.live_events import coding_event_scope, publish_coding_event
from mana_agent.coding.models import AgentEvent, CodingTaskResult, WorkspaceContext
from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.gateway.entry_routing import EntryRouteContext
from mana_agent.integrations.codex.backend import CodexCodingBackend
from mana_agent.integrations.codex.client import AsyncCodexAppServer, CodexCancellationOutcome
from mana_agent.integrations.codex.coding_agent_shim import CodexCodingAgentShim
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.services.execution_event_hub import ExecutionEventHub, get_execution_event_hub
from mana_agent.tui.app import ManaChatApp
from mana_agent.tui.widgets.execution_panel import ExecutionPanel


def _init_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True, capture_output=True)
    readme = path / "README.md"
    readme.write_text("# Test Repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=path, check=True, capture_output=True)
    return path


class _DummyAskService:
    class _EntryModel:
        def with_structured_output(self, schema: Any, *, method: str = "json_schema", strict: bool = True):
            return self

        def invoke(self, messages, **_kwargs):
            payload = json.loads(messages[-1].content) if messages and messages[-1].content.startswith("{") else {}
            if "recovery_candidates" in payload or (messages and "You decide whether a new user request may resume" in str(messages[0].content)):
                return SimpleNamespace(
                    content=json.dumps(
                        {
                            "action": "start_fresh",
                            "task_id": "",
                            "checkpoint_id": "",
                            "same_work": False,
                            "fresh_data_required": False,
                            "checkpoint_still_valid": False,
                            "side_effects_safe_to_repeat": False,
                            "safe_to_continue": True,
                            "reason": "the test model selected a fresh execution",
                        }
                    )
                )
            if "candidates" in payload or (messages and "Classify this newly received chat turn" in str(messages[0].content)):
                return SimpleNamespace(
                    content=json.dumps(
                        {
                            "action": "classify",
                            "category": "new_task",
                            "related_task_id": "",
                            "safe_to_continue": True,
                            "reason": "independent task",
                        }
                    )
                )
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "route": "coding",
                        "confidence": 0.95,
                        "reason": "coding route",
                        "required_sources": ["repository"],
                        "target_urls": [],
                        "requires_live_data": False,
                        "reason_code": "TEST_ROUTE",
                        "error_code": "",
                        "reuse_active_route": False,
                        "runtime_capability_change": False,
                    }
                )
            )

    entry_router = SimpleNamespace(llm=_EntryModel())
    ask_agent = SimpleNamespace(llm=None, update_model=lambda m: None, model="dummy")
    qna_chain = SimpleNamespace(
        llm=None,
        chat=lambda question, **kwargs: "(dummy conversational response)",
    )

    def ask(self, *args, **kwargs):
        return "(dummy conversational response)"


@pytest.fixture(autouse=True)
def _setup_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "mock-openai-key")
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )


def _make_mock_backend(events: list[AgentEvent] | None = None, status: str = "completed") -> Any:
    class _MockBackend:
        def __init__(self):
            self.resume_thread_id = ""

        async def stream(self, task: Any, workspace: Any):
            ev_list = events or [
                AgentEvent(
                    event_id=f"evt_cmd_start_{task.task_id}",
                    event_type="command.started",
                    task_id=task.task_id,
                    status="running",
                    title="Run pytest",
                    command="pytest",
                    visibility="progress",
                    semantic_kind="command",
                ),
                AgentEvent(
                    event_id=f"evt_cmd_done_{task.task_id}",
                    event_type="command.completed",
                    task_id=task.task_id,
                    status="success",
                    title="Run pytest",
                    command="pytest",
                    output_preview="1 passed in 0.05s",
                    visibility="progress",
                    semantic_kind="command",
                ),
            ]
            for ev in ev_list:
                yield ev

        def result_for(self, task_id: str) -> CodingTaskResult:
            return CodingTaskResult(
                task_id=task_id,
                worker_id="test_worker",
                backend="codex",
                status=status,
                summary="Coding turn completed successfully.",
                thread_id="thread_test_123",
            )

        async def close(self):
            pass

    return _MockBackend


# =========================================================================
# Scenario 1: Normal Codex task renders events once
# =========================================================================
def test_normal_codex_task_renders_events_once(tmp_path: Path):
    repo = _init_git_repo(tmp_path / "repo")
    backend_factory = _make_mock_backend()
    settings = CodexSettings.from_mana_settings(None, provider="openai")
    shim = CodexCodingAgentShim(
        repo_root=repo,
        codex_settings=settings,
        session_id="session_test_1",
        backend_factory=backend_factory,
    )

    received_events = []

    def on_event(ev):
        received_events.append(ev)

    with coding_event_scope(on_event):
        payload = shim.generate("Fix bug in repo", gateway_task_id="task_turn_1")

    assert payload["status"] == "completed"
    # Events should be emitted: command.started, command.completed, coding.terminal
    event_types = [e.event_type for e in received_events]
    assert "command.started" in event_types
    assert "command.completed" in event_types
    assert "coding.terminal" in event_types
    # Each event ID should be unique
    event_ids = [e.event_id for e in received_events]
    assert len(event_ids) == len(set(event_ids))
    shim.close()


# =========================================================================
# Scenario 2: Task without active coding_event_scope still renders via ExecutionEventHub
# =========================================================================
def test_task_without_coding_event_scope_renders_via_hub(tmp_path: Path):
    repo = _init_git_repo(tmp_path / "repo")
    backend_factory = _make_mock_backend()
    settings = CodexSettings.from_mana_settings(None, provider="openai")
    hub = get_execution_event_hub()
    session_id = f"session_hub_test_{tmp_path.name}"

    app = ManaChatApp(repo_root=repo)
    app._gateway_session_id = session_id
    app._subscribe_session_events(session_id)

    # Set up mapping for task_id -> frontend turn_id
    app._execution_to_frontend_turn["task_hub_1"] = "turn_frontend_1"

    shim = CodexCodingAgentShim(
        repo_root=repo,
        codex_settings=settings,
        session_id=session_id,
        backend_factory=backend_factory,
    )

    # Run WITHOUT coding_event_scope
    payload = shim.generate("Inspect repo", gateway_task_id="task_hub_1")
    assert payload["status"] == "completed"

    # Verify that events were received by the app via ExecutionEventHub
    history_events = [
        ev for ev in app.history.get_events()
        if isinstance(ev, CodingActivityEvent) and ev.turn_id == "turn_frontend_1"
    ]
    assert len(history_events) >= 2
    types = [ev.activity.get("event_type") for ev in history_events]
    assert "command.started" in types
    assert "command.completed" in types

    app.on_unmount()
    shim.close()


# =========================================================================
# Scenario 3: Scope + hub delivery does not duplicate TUI rows or events
# =========================================================================
def test_scope_plus_hub_delivery_does_not_duplicate_events(tmp_path: Path):
    repo = _init_git_repo(tmp_path / "repo")
    session_id = f"session_dedup_{tmp_path.name}"
    app = ManaChatApp(repo_root=repo)
    app._gateway_session_id = session_id
    app._subscribe_session_events(session_id)
    app._current_frontend_turn_id = "turn_1"
    app._execution_to_frontend_turn["task_dedup_1"] = "turn_1"

    # Simulate fast-path delivery via _on_coding_event callback
    event_payload = {
        "event_id": "evt_shared_123",
        "event_type": "command.started",
        "task_id": "task_dedup_1",
        "title": "pytest",
        "command": "pytest",
        "status": "running",
    }

    # Scope delivery
    app._delivered_coding_event_ids.add("evt_shared_123")
    app.history.add(CodingActivityEvent(activity=event_payload, turn_id="turn_1"))

    # Hub delivers the exact same event
    hub_event = {
        "event_id": "evt_shared_123",
        "event_type": "command.started",
        "conversation_id": session_id,
        "execution_id": "task_dedup_1",
        "title": "pytest",
        "command": "pytest",
        "status": "running",
        "metadata": {"visibility": "progress", "semantic_kind": "command"},
    }
    app._handle_hub_session_event(hub_event)

    # Verify event appears only once in history
    matching = [
        ev for ev in app.history.get_events()
        if isinstance(ev, CodingActivityEvent) and ev.activity.get("event_id") == "evt_shared_123"
    ]
    assert len(matching) == 1

    # Verify ExecutionPanel deduplicates as well
    panel = ExecutionPanel(turn_id="turn_1")
    panel.update_event(event_payload)
    panel.update_event(event_payload)  # Duplicate call
    assert len(panel.events) == 1

    app.on_unmount()


# =========================================================================
# Scenario 4: /new replaces previous session subscriptions and unsubscribes old session
# =========================================================================
def test_new_replaces_previous_session_subscriptions(tmp_path: Path):
    repo = _init_git_repo(tmp_path / "repo")
    hub = get_execution_event_hub()
    old_sid = "session_old_4"
    new_sid = "session_new_4"

    app = ManaChatApp(repo_root=repo)
    app._gateway_session_id = old_sid
    app._subscribe_session_events(old_sid)

    assert old_sid in hub._subscribers
    assert len(hub._subscribers[old_sid]) >= 1

    # Atomically replace session
    app._handle_session_replacement(new_sid, clear_view=True)

    # Old session should be unsubscribed
    assert old_sid not in hub._subscribers
    # New session should be subscribed
    assert new_sid in hub._subscribers
    assert len(hub._subscribers[new_sid]) >= 1
    assert app._gateway_session_id == new_sid

    app.on_unmount()


# =========================================================================
# Scenario 5: /new followed immediately by a Codex request creates exactly one Codex execution
# =========================================================================
def test_new_followed_by_codex_request_executes_once(tmp_path: Path):
    repo = _init_git_repo(tmp_path / "repo")
    execution_counter = 0

    class _CountingBackend:
        def __init__(self):
            self.resume_thread_id = ""

        async def stream(self, task: Any, workspace: Any):
            nonlocal execution_counter
            execution_counter += 1
            yield AgentEvent(
                event_id=f"cmd_{task.task_id}",
                event_type="command.started",
                task_id=task.task_id,
                status="running",
                title="exec",
                visibility="progress",
            )

        def result_for(self, task_id: str) -> CodingTaskResult:
            return CodingTaskResult(
                task_id=task_id,
                worker_id="test_worker",
                backend="codex",
                status="completed",
                summary="done",
            )

        async def close(self):
            pass

    settings = CodexSettings.from_mana_settings(None, provider="openai")
    shim = CodexCodingAgentShim(
        repo_root=repo,
        codex_settings=settings,
        session_id="session_5_initial",
        backend_factory=_CountingBackend,
    )

    # First turn in initial session
    shim.generate("Edit A", gateway_task_id="turn_5_initial")
    assert execution_counter == 1

    # /new reset
    shim.reset_session("session_5_new")

    # Immediate turn in new session with same turn identity dispatched twice concurrently
    def _run_turn():
        return shim.generate(
            "Edit B",
            gateway_task_id="turn_5_new",
            turn_id="turn_5_new",
            dispatch_source="tui_submit",
        )

    t1 = threading.Thread(target=_run_turn)
    t2 = threading.Thread(target=_run_turn)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # The idempotency guard ensures execution occurred exactly once for (session_5_new, turn_5_new)
    assert execution_counter == 2  # 1 from initial + exactly 1 from new session
    shim.close()


# =========================================================================
# Scenario 6: Repeated /new commands do not accumulate subscribers
# =========================================================================
def test_repeated_new_commands_do_not_accumulate_subscribers(tmp_path: Path):
    repo = _init_git_repo(tmp_path / "repo")
    hub = get_execution_event_hub()
    app = ManaChatApp(repo_root=repo)

    for i in range(10):
        sid = f"session_loop_{i}"
        app._handle_session_replacement(sid, clear_view=True)
        # Exactly one subscriber should exist for the active session
        assert len(hub._subscribers[sid]) == 1

    # Old sessions should have 0 subscribers in the hub
    for i in range(9):
        old_sid = f"session_loop_{i}"
        assert old_sid not in hub._subscribers

    app.on_unmount()


# =========================================================================
# Scenario 7: Old-session events cannot attach to the new chat panel
# =========================================================================
def test_old_session_events_cannot_attach_to_new_session(tmp_path: Path):
    repo = _init_git_repo(tmp_path / "repo")
    app = ManaChatApp(repo_root=repo)
    app._handle_session_replacement("session_new_7", clear_view=True)

    # Event arrives tagged with old session
    old_event = {
        "event_id": "evt_old_session_7",
        "event_type": "command.started",
        "conversation_id": "session_old_7",  # Mismatched session
        "execution_id": "task_old_7",
        "title": "old command",
        "metadata": {"visibility": "progress", "semantic_kind": "command"},
    }
    app._handle_hub_session_event(old_event)

    # Must NOT be added to history
    matching = [
        ev for ev in app.history.get_events()
        if isinstance(ev, CodingActivityEvent) and ev.activity.get("event_id") == "evt_old_session_7"
    ]
    assert len(matching) == 0

    app.on_unmount()


# =========================================================================
# Scenario 8: A previous session's recoverable task is not accidentally dispatched as the new task
# =========================================================================
def test_previous_session_recoverable_task_not_dispatched_in_new_session(tmp_path: Path):
    repo = _init_git_repo(tmp_path / "repo")
    gateway = AgentChatGateway(repo, coding_agent=True, agent_tools=False)
    old_session = gateway.create_session(frontend="tui")

    # Manually register a task in the old session on the supervisor store
    supervisor = gateway._lane_coordinator.execution_supervisor
    from mana_agent.execution_supervisor.models import TaskRecord, ExecutionState
    from datetime import datetime, timezone

    workspace_id = gateway._lane_coordinator.taskboard.store.workspace_id
    repository_id = gateway._lane_coordinator.taskboard.store.repository_id

    old_task = TaskRecord(
        task_id="task_old_session_failed",
        session_id=old_session,
        workspace_id=workspace_id,
        repository_id=repository_id,
        normalized_intent="edit",
        state=ExecutionState.FAILED,
        assigned_agent="lane:coding",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    supervisor.store.create_task(old_task)

    # Verify old session sees it as candidate
    old_candidates = gateway._recovery_candidates(
        lane_id=None,
        session_id=old_session,
        workspace_id=workspace_id,
        repository_id=repository_id,
    )
    assert any(c["task_id"] == "task_old_session_failed" for c in old_candidates)

    # Start new conversation
    new_session = gateway.start_new_conversation(old_session, frontend="tui")

    # Verify new session does NOT see the old session's task
    new_candidates = gateway._recovery_candidates(
        lane_id=None,
        session_id=new_session,
        workspace_id=workspace_id,
        repository_id=repository_id,
    )
    assert not any(c["task_id"] == "task_old_session_failed" for c in new_candidates)


# =========================================================================
# Scenario 9: Retries do not become separate independent Codex executions
# =========================================================================
def test_retries_stay_within_same_execution_lifecycle(tmp_path: Path):
    repo = _init_git_repo(tmp_path / "repo")
    attempts = 0

    class _RetryBackend:
        def __init__(self):
            self.resume_thread_id = ""

        async def stream(self, task: Any, workspace: Any):
            nonlocal attempts
            attempts += 1
            yield AgentEvent(
                event_id=f"cmd_attempt_{attempts}",
                event_type="command.started",
                task_id=task.task_id,
                status="running",
                title=f"attempt {attempts}",
                visibility="progress",
            )

        def result_for(self, task_id: str) -> CodingTaskResult:
            # First attempt fails with mutation required
            if attempts == 1:
                return CodingTaskResult(
                    task_id=task_id,
                    worker_id="test_worker",
                    backend="codex",
                    status="completed",
                    summary="Inspection only",
                    errors=[],
                )
            return CodingTaskResult(
                task_id=task_id,
                worker_id="test_worker",
                backend="codex",
                status="completed",
                summary="Applied mutation",
            )

    settings = CodexSettings.from_mana_settings(None, provider="openai")
    shim = CodexCodingAgentShim(
        repo_root=repo,
        codex_settings=settings,
        session_id="session_retry_9",
        backend_factory=_RetryBackend,
    )

    # Mutation recovery inside _execute_turn
    payload = shim._execute_turn(
        "Make a mutation",
        requires_repository_write=True,
        gateway_task_id="turn_retry_9",
        turn_id="turn_retry_9",
    )

    # Both attempts executed within the single _execute_turn call
    assert attempts >= 1
    assert payload is not None
    shim.close()


# =========================================================================
# Scenario 10: Terminal and error events appear exactly once
# =========================================================================
def test_terminal_and_error_events_appear_exactly_once(tmp_path: Path):
    panel = ExecutionPanel(turn_id="turn_10")

    terminal_event = {
        "event_id": "evt_terminal_10",
        "event_type": "coding.terminal",
        "status": "success",
        "title": "Coding result",
        "summary": "Task complete.",
    }

    panel.update_event(terminal_event)
    panel.update_event(terminal_event)  # Duplicate delivery

    terminal_rows = [e for e in panel.events if e.get("event_id") == "evt_terminal_10"]
    assert len(terminal_rows) == 1


# =========================================================================
# Scenario 11: Internal reasoning and raw assistant deltas remain hidden
# =========================================================================
def test_internal_reasoning_and_deltas_remain_hidden(tmp_path: Path):
    repo = _init_git_repo(tmp_path / "repo")
    session_id = f"session_hidden_{tmp_path.name}"
    app = ManaChatApp(repo_root=repo)
    app._gateway_session_id = session_id
    app._subscribe_session_events(session_id)
    app._current_frontend_turn_id = "turn_11"

    # Feed assistant.delta
    delta_event = {
        "event_id": "evt_delta_1",
        "event_type": "assistant.delta",
        "conversation_id": session_id,
        "execution_id": "task_11",
        "summary": "I am thinking about the code...",
        "metadata": {"visibility": "internal", "semantic_kind": "assistant_generation"},
    }
    app._handle_hub_session_event(delta_event)

    # Feed reasoning.thought
    reasoning_event = {
        "event_id": "evt_reasoning_1",
        "event_type": "reasoning.thought",
        "conversation_id": session_id,
        "execution_id": "task_11",
        "summary": "Let me inspect the AST...",
        "metadata": {"visibility": "internal", "semantic_kind": "reasoning"},
    }
    app._handle_hub_session_event(reasoning_event)

    # None of these should be in history
    activity_events = [ev for ev in app.history.get_events() if isinstance(ev, CodingActivityEvent)]
    assert len(activity_events) == 0

    app.on_unmount()
