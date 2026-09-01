"""Tests for gateway session lifecycle, /new hard conversation boundaries, and session-generation fencing."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from mana_agent.coding.models import AgentEvent, CodingTaskResult, WorkspaceContext
from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.integrations.codex.backend import CodexCodingBackend
from mana_agent.integrations.codex.client import AsyncCodexAppServer, CodexCancellationOutcome
from mana_agent.integrations.codex.coding_agent_shim import CodexCodingAgentShim
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.exceptions import CodexTimeoutError
from mana_agent.multi_agent.routing.agent_decision import AgentDecision


class _DummyAskService:
    """Minimal stand-in so gateway construction tests do not require real LLM credentials."""

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


@pytest.fixture(autouse=True)
def _setup_test_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "mock-openai-key")
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    _init_git_repo(tmp_path)


def _settings(**updates: object) -> CodexSettings:
    values = {
        "enabled": True,
        "provider": "openai",
        "provider_display_name": "OpenAI",
        "api_key": "mana-secret-key",
        "base_url": "https://api.example.test/v1/responses/",
        "model": "gpt-5-codex",
        "supports_responses_api": True,
    }
    values.update(updates)
    return CodexSettings(**values)


def _edit_decision() -> AgentDecision:
    return AgentDecision(
        intent="edit",
        code_editing_needed=True,
        selected_tools=["apply_patch"],
        tool_inputs={},
        flow_action="none",
        reasoning_summary="edit is required",
        confidence=0.99,
        verifier_passed=True,
    )


def test_new_conversation_during_hanging_codex_interrupt_bounds_latency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Latency regression test: /new must complete within a small bounded deadline (<3.5s) instead of 30s."""
    closed = False

    class HangingInterruptClient:
        running = True

        async def start(self) -> None:
            return None

        async def request(self, method: str, params: dict[str, Any], *, timeout_seconds: float | None = None) -> dict[str, Any]:
            if method == "turn/interrupt":
                # Simulate stuck Codex app-server that never responds to turn/interrupt
                await asyncio.sleep(60.0)
            return {}

        async def notifications(self, thread_id: str):
            # Hang forever streaming
            await asyncio.sleep(60.0)
            yield {}

        async def close(self, *, wait_timeout: float = 1.0) -> None:
            nonlocal closed
            closed = True
            self.running = False

    monkeypatch.setattr("mana_agent.gateway.turn_engine.decide_chat_route", lambda **kwargs: _edit_decision())
    monkeypatch.setattr("mana_agent.gateway.turn_engine.handle_small_direct_edit", lambda *args, **kwargs: SimpleNamespace(handled=False))

    gateway = AgentChatGateway(tmp_path, coding_agent=True, auto_execute_plan=True, agent_tools=False)
    session_id = gateway.create_session(frontend="cli")

    # Wire coding agent shim with hanging backend
    shim = CodexCodingAgentShim(
        repo_root=tmp_path,
        codex_settings=_settings(),
        backend_factory=lambda: CodexCodingBackend(_settings(), client_factory=lambda cmd: HangingInterruptClient()),
    )
    gateway._coding_agent = shim

    # Start a turn in background or mark as running
    gateway._active.add(session_id)
    gateway._coding_agent._active_backend = ("task-hang", shim._backend_factory())
    gateway._coding_agent._active_backend[1]._client = HangingInterruptClient()
    gateway._coding_agent._active_backend[1]._active["task-hang"] = ("th-1", "tu-1")

    start_time = time.monotonic()
    new_session_id = gateway.start_new_conversation(session_id, frontend="cli")
    elapsed = time.monotonic() - start_time

    assert new_session_id != session_id
    assert elapsed < 3.5, f"/new took {elapsed:.2f}s, exceeding bounded cancellation threshold"
    assert session_id in gateway._fenced_sessions


def test_new_conversation_after_completed_codex_work_uses_thread_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    threads_started: list[str] = []
    threads_resumed: list[str] = []

    class MockAppServerClient:
        def __init__(self, command: tuple[str, ...]) -> None:
            self.running = True

        async def start(self) -> None:
            return None

        async def request(self, method: str, params: dict[str, Any], *, timeout_seconds: float | None = None) -> dict[str, Any]:
            if method == "thread/start":
                th_id = f"thread-{len(threads_started) + 1}"
                threads_started.append(th_id)
                return {"thread": {"id": th_id}}
            if method == "thread/resume":
                th_id = params.get("threadId", "unknown")
                threads_resumed.append(th_id)
                return {"thread": {"id": th_id}}
            if method == "turn/start":
                return {"turn": {"id": "turn-1"}}
            return {}

        async def notifications(self, thread_id: str):
            yield {
                "method": "turn/completed",
                "params": {"threadId": thread_id, "turn": {"id": "turn-1"}},
            }

        async def close(self, *, wait_timeout: float = 1.0) -> None:
            self.running = False

    monkeypatch.setattr("mana_agent.gateway.turn_engine.decide_chat_route", lambda **kwargs: _edit_decision())
    monkeypatch.setattr("mana_agent.gateway.turn_engine.handle_small_direct_edit", lambda *args, **kwargs: SimpleNamespace(handled=False))

    gateway = AgentChatGateway(tmp_path, coding_agent=True, auto_execute_plan=True, agent_tools=False)
    session_1 = gateway.create_session(frontend="cli")

    settings = _settings(worktree_isolation=False)
    shim = CodexCodingAgentShim(
        repo_root=tmp_path,
        codex_settings=settings,
        backend_factory=lambda: CodexCodingBackend(settings, client_factory=lambda cmd: MockAppServerClient(cmd)),
    )
    gateway._coding_agent = shim
    gateway._stack.coding_agent = shim

    from mana_agent.gateway.entry_routing import EntryRoutingDecision

    monkeypatch.setattr(
        gateway._entry_router,
        "route",
        lambda *args, **kwargs: EntryRoutingDecision(
            route="coding",
            confidence=0.99,
            reason="coding task",
            required_sources=(),
            reuse_active_route=bool(gateway._coding_agent.get_active_flow_id()),
        ),
    )

    # Turn 1 in session 1: starts thread-1
    gateway.process_turn(session_1, "First task")
    assert threads_started == ["thread-1"]

    # Turn 2 in session 1: reuses runtime and calls thread/resume (no new thread/start)
    gateway.process_turn(session_1, "Second task in same session")
    assert threads_started == ["thread-1"]
    assert "thread-1" in threads_resumed

    # Start new conversation via /new
    session_2 = gateway.start_new_conversation(session_1, frontend="cli")
    assert session_2 != session_1
    assert shim.resume_thread_id == ""

    # Turn 1 in session 2: MUST start fresh thread-2 via thread/start
    gateway.process_turn(session_2, "First task in new session")
    assert threads_started == ["thread-1", "thread-2"]


def test_session_generation_fence_rejects_late_events(tmp_path: Path) -> None:
    gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False)
    session_1 = gateway.create_session(frontend="cli")
    gateway.process_turn(session_1, "Hello from session 1")

    # Start new conversation
    session_2 = gateway.start_new_conversation(session_1, frontend="cli")

    # Attempt to append message to old fenced session
    msg = gateway._append_session_message(
        session_1,
        role="assistant",
        content="Late message",
        turn_id="turn-late",
    )
    assert msg is None
    assert gateway.session_messages(session_1) == []


def test_repeated_new_conversations_under_rapid_invocations(tmp_path: Path) -> None:
    gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False)
    current = gateway.create_session(frontend="cli")
    created_sessions = [current]

    for _ in range(5):
        current = gateway.start_new_conversation(current, frontend="cli")
        created_sessions.append(current)

    # All session IDs must be distinct and non-empty
    assert len(set(created_sessions)) == 6
    for old_sid in created_sessions[:-1]:
        assert old_sid in gateway._fenced_sessions
    assert created_sessions[-1] not in gateway._fenced_sessions
