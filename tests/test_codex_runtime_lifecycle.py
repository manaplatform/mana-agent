"""Tests for Codex runtime lifecycle, session-scoped backend reuse, bounded cancellation, and process cleanup."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from mana_agent.coding.models import AgentEvent, CodingTask, CodingTaskResult, WorkspaceContext
from mana_agent.integrations.codex.backend import CodexCodingBackend
from mana_agent.integrations.codex.client import AsyncCodexAppServer, CodexCancellationOutcome
from mana_agent.integrations.codex.coding_agent_shim import CodexCodingAgentShim, _CodexRuntimeRunner
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.exceptions import CodexTimeoutError, CodexUnavailableError


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


def _workspace(tmp_path: Path) -> WorkspaceContext:
    repo = _init_git_repo(tmp_path / "repo")
    return WorkspaceContext(
        repository_path=repo,
        worktree_path=repo,
        working_directory=repo,
        sandbox="readOnly",
        approval_policy="never",
    )


def test_runtime_runner_executes_on_single_stable_loop() -> None:
    runner = _CodexRuntimeRunner()
    try:
        async def _get_loop_id() -> int:
            await asyncio.sleep(0.01)
            return id(asyncio.get_running_loop())

        loop_id_1 = runner.run(_get_loop_id())
        loop_id_2 = runner.run(_get_loop_id())
        assert loop_id_1 == loop_id_2
    finally:
        runner.close()


def test_async_codex_app_server_interrupt_returns_acknowledged_outcome() -> None:
    class MockClient(AsyncCodexAppServer):
        def __init__(self) -> None:
            super().__init__(("codex", "app-server"))
            self._process = MagicMock(returncode=None)

        async def request(
            self,
            method: str,
            params: dict[str, Any],
            *,
            timeout_seconds: float | None = None,
        ) -> dict[str, Any]:
            assert method == "turn/interrupt"
            assert params == {"threadId": "th-1", "turnId": "tu-1"}
            return {"status": "ok"}

    client = MockClient()
    outcome = asyncio.run(client.interrupt(thread_id="th-1", turn_id="tu-1", timeout_seconds=1.0))
    assert isinstance(outcome, CodexCancellationOutcome)
    assert outcome.acknowledged is True
    assert outcome.status == "acknowledged"
    assert outcome.thread_id == "th-1"
    assert outcome.turn_id == "tu-1"


def test_async_codex_app_server_interrupt_handles_timeout_as_typed_outcome() -> None:
    class HangingInterruptClient(AsyncCodexAppServer):
        def __init__(self) -> None:
            super().__init__(("codex", "app-server"))
            self._process = MagicMock(returncode=None)

        async def request(
            self,
            method: str,
            params: dict[str, Any],
            *,
            timeout_seconds: float | None = None,
        ) -> dict[str, Any]:
            if method == "turn/interrupt":
                raise CodexTimeoutError("interrupt timed out", method=method, timeout_seconds=1)
            return {}

    client = HangingInterruptClient()
    outcome = asyncio.run(client.interrupt(thread_id="th-1", turn_id="tu-1", timeout_seconds=0.1))
    assert isinstance(outcome, CodexCancellationOutcome)
    assert outcome.acknowledged is False
    assert outcome.status == "timed_out"
    assert outcome.error is not None
    assert "interrupt timed out" in outcome.error


def test_codex_coding_backend_cancel_forces_close_on_unacknowledged_interrupt() -> None:
    closed = False

    class StuckClient:
        running = True

        async def interrupt(self, *, thread_id: str, turn_id: str, timeout_seconds: float = 2.0) -> CodexCancellationOutcome:
            return CodexCancellationOutcome(
                acknowledged=False,
                status="timed_out",
                thread_id=thread_id,
                turn_id=turn_id,
                error="Stuck backend",
            )

        async def close(self, *, wait_timeout: float = 1.0) -> None:
            nonlocal closed
            closed = True
            self.running = False

    backend = CodexCodingBackend(_settings(), client_factory=lambda cmd: StuckClient())
    backend._client = StuckClient()
    backend._active["task-1"] = ("th-1", "tu-1")

    outcome = asyncio.run(backend.cancel("task-1", timeout_seconds=0.1))
    assert outcome.acknowledged is False
    assert closed is True
    assert backend._client is None


def test_codex_coding_agent_shim_session_scoped_backend_reuse(tmp_path: Path) -> None:
    clients_created = 0

    class TrackingClient:
        def __init__(self, command: tuple[str, ...]) -> None:
            nonlocal clients_created
            clients_created += 1
            self.running = True
            self.closed = False
            self.requests: list[str] = []

        async def start(self) -> None:
            return None

        async def request(self, method: str, params: dict[str, Any], *, timeout_seconds: float | None = None) -> dict[str, Any]:
            self.requests.append(method)
            if method == "thread/start":
                return {"thread": {"id": "thread-session-1"}}
            if method == "thread/resume":
                return {"thread": {"id": params.get("threadId", "thread-session-1")}}
            if method == "turn/start":
                return {"turn": {"id": "turn-1"}}
            return {}

        async def notifications(self, thread_id: str):
            yield {
                "method": "turn/completed",
                "params": {
                    "threadId": thread_id,
                    "turn": {"id": "turn-1"},
                    "usage": {"inputTokens": 10},
                },
            }

        async def close(self, *, wait_timeout: float = 1.0) -> None:
            self.closed = True
            self.running = False

    ws = _workspace(tmp_path)
    shim = CodexCodingAgentShim(
        repo_root=ws.repository_path,
        codex_settings=_settings(),
        backend_factory=lambda: CodexCodingBackend(_settings(), client_factory=lambda cmd: TrackingClient(cmd)),
    )

    try:
        # Turn 1: Should start fresh client and call thread/start
        res1 = shim.generate("First turn")
        assert clients_created == 1
        assert shim.resume_thread_id == "thread-session-1"

        # Turn 2: Should REUSE client inside same session and call thread/resume
        res2 = shim.generate("Second turn in same session")
        assert clients_created == 1  # No new client created!
        assert shim.resume_thread_id == "thread-session-1"

        # Turn 3 after reset_session: Should close old client and create fresh one with thread/start
        shim.reset_session("new-session-id")
        assert shim.resume_thread_id == ""

        res3 = shim.generate("First turn in new session")
        assert clients_created == 2  # New client created!
    finally:
        shim.close()


def test_codex_coding_agent_shim_restarts_on_model_change(tmp_path: Path) -> None:
    clients_created = 0

    class DummyClient:
        def __init__(self, command: tuple[str, ...]) -> None:
            nonlocal clients_created
            clients_created += 1
            self.running = True

        async def start(self) -> None:
            return None

        async def request(self, method: str, params: dict[str, Any], *, timeout_seconds: float | None = None) -> dict[str, Any]:
            if method == "thread/start":
                return {"thread": {"id": f"th-{clients_created}"}}
            if method == "turn/start":
                return {"turn": {"id": "turn-1"}}
            return {}

        async def notifications(self, thread_id: str):
            yield {"method": "turn/completed", "params": {"threadId": thread_id, "turn": {"id": "turn-1"}}}

        async def close(self, *, wait_timeout: float = 1.0) -> None:
            self.running = False

    ws = _workspace(tmp_path)
    shim = CodexCodingAgentShim(
        repo_root=ws.repository_path,
        codex_settings=_settings(),
        backend_factory=lambda: CodexCodingBackend(_settings(), client_factory=lambda cmd: DummyClient(cmd)),
    )

    try:
        shim.generate("Turn 1")
        assert clients_created == 1

        # Changing model invalidates session backend
        shim.update_model("gpt-5-turbo")
        shim.generate("Turn 2 with new model")
        assert clients_created == 2
    finally:
        shim.close()
