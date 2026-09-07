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
from mana_agent.integrations.codex.exceptions import (
    CodexThreadStateMissingError,
    CodexTimeoutError,
    CodexUnavailableError,
)
from mana_agent.integrations.codex.runtime_environment import get_codex_session_home
from mana_agent.integrations.codex.session_store import (
    clear_codex_session_thread,
    load_codex_session_thread,
    save_codex_session_thread,
)


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


def test_codex_coding_agent_shim_session_scoped_backend_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mana_agent.integrations.codex.backend._git_changed_files",
        lambda *args, **kwargs: ["README.md"],
    )
    clients_created = 0
    created_clients: list[TrackingClient] = []

    class TrackingClient:
        def __init__(self, command: tuple[str, ...]) -> None:
            nonlocal clients_created
            clients_created += 1
            created_clients.append(self)
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
        # Turn 1: Should start fresh client and call thread/start + turn/start
        res1 = shim.generate("First turn")
        assert clients_created == 1
        assert shim.resume_thread_id == "thread-session-1"
        assert created_clients[0].requests == ["thread/start", "turn/start"]

        # Turn 2: Should REUSE live client in same session and call turn/start ONLY (never thread/resume)
        res2 = shim.generate("Second turn in same session")
        assert clients_created == 1  # No new client created!
        assert shim.resume_thread_id == "thread-session-1"
        assert created_clients[0].requests == ["thread/start", "turn/start", "turn/start"]

        # Turn 3: Also REUSES live client and calls turn/start ONLY
        res3 = shim.generate("Third turn in same session")
        assert clients_created == 1
        assert created_clients[0].requests == ["thread/start", "turn/start", "turn/start", "turn/start"]

        # Recreated client: Reconstruct session on fresh backend/client with persisted thread ID
        fresh_backend = CodexCodingBackend(
            _settings(),
            client_factory=lambda cmd: TrackingClient(cmd),
            resume_thread_id=shim.resume_thread_id,
        )
        task = CodingTask(task_id="recreated-task", goal="Turn on recreated client", requires_repository_write=False)
        result = asyncio.run(fresh_backend.execute(task, ws))
        assert result.status == "completed"
        assert clients_created == 2
        assert created_clients[1].requests == ["thread/resume", "turn/start"]

        # Turn after reset_session (/new): Should close old client and create fresh one with thread/start + turn/start
        shim.reset_session("new-session-id")
        assert shim.resume_thread_id == ""

        res4 = shim.generate("First turn in new session")
        assert clients_created == 3  # New client created!
        assert created_clients[2].requests == ["thread/start", "turn/start"]
    finally:
        shim.close()


def test_codex_resident_session_bypasses_hanging_thread_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test: A normal second message on a live session must never invoke thread/resume.

    If thread/resume is hanging or times out, the second turn must still succeed without
    CODING_PROVIDER_TIMEOUT because resident threads call turn/start directly.
    """
    monkeypatch.setattr(
        "mana_agent.integrations.codex.backend._git_changed_files",
        lambda *args, **kwargs: ["README.md"],
    )
    clients_created = 0
    created_clients: list[Any] = []

    class HangingResumeClient:
        def __init__(self, command: tuple[str, ...]) -> None:
            nonlocal clients_created
            clients_created += 1
            created_clients.append(self)
            self.running = True
            self.closed = False
            self.requests: list[str] = []

        async def start(self) -> None:
            return None

        async def request(self, method: str, params: dict[str, Any], *, timeout_seconds: float | None = None) -> dict[str, Any]:
            self.requests.append(method)
            if method == "thread/start":
                return {"thread": {"id": "thread-hanging-resume-test"}}
            if method == "thread/resume":
                # Deliberately hang/timeout to simulate the bug scenario
                raise CodexTimeoutError(
                    "Codex request timed out: thread/resume",
                    method="thread/resume",
                    timeout_seconds=1,
                )
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
        backend_factory=lambda: CodexCodingBackend(
            _settings(),
            client_factory=lambda cmd: HangingResumeClient(cmd),
        ),
    )

    try:
        # Turn 1: Fresh session, uses thread/start + turn/start
        res1 = shim.generate("Turn 1 goal")
        assert res1.get("status") != "failed"
        assert shim.resume_thread_id == "thread-hanging-resume-test"
        assert created_clients[0].requests == ["thread/start", "turn/start"]

        # Turn 2: Second message in live session. MUST NEVER invoke thread/resume!
        # Should complete successfully without CODING_PROVIDER_TIMEOUT.
        res2 = shim.generate("Turn 2 follow-up in same live session")
        assert res2.get("status") != "failed"
        assert created_clients[0].requests == ["thread/start", "turn/start", "turn/start"]
        assert "thread/resume" not in created_clients[0].requests
    finally:
        shim.close()


def test_codex_thread_resume_timeout_explicit_recovery(tmp_path: Path) -> None:
    """A real resume timeout on an uncertain connection closes and recreates the app-server,
    initializes it, and resumes once on the fresh client.
    """
    attempt = 0
    clients: list[Any] = []

    class RecoveringClient:
        def __init__(self, command: tuple[str, ...]) -> None:
            clients.append(self)
            self.running = True
            self.closed = False
            self.requests: list[str] = []

        async def start(self) -> None:
            return None

        async def request(self, method: str, params: dict[str, Any], *, timeout_seconds: float | None = None) -> dict[str, Any]:
            nonlocal attempt
            self.requests.append(method)
            if method == "thread/resume":
                attempt += 1
                if attempt == 1:
                    # First client times out on resume (unhealthy connection)
                    raise CodexTimeoutError(
                        "Codex request timed out: thread/resume",
                        method="thread/resume",
                        timeout_seconds=1,
                    )
                # Second client succeeds on resume
                return {"thread": {"id": "thread-recovered"}}
            if method == "turn/start":
                return {"turn": {"id": "turn-recovered"}}
            return {}

        async def notifications(self, thread_id: str):
            yield {
                "method": "turn/completed",
                "params": {
                    "threadId": thread_id,
                    "turn": {"id": "turn-recovered"},
                },
            }

        async def close(self, *, wait_timeout: float = 1.0) -> None:
            self.closed = True
            self.running = False

    ws = _workspace(tmp_path)
    backend = CodexCodingBackend(
        _settings(),
        client_factory=lambda cmd: RecoveringClient(cmd),
        resume_thread_id="thread-persisted-1",
    )
    task = CodingTask(task_id="recovery-task", goal="Recovery after resume timeout", requires_repository_write=False)
    result = asyncio.run(backend.execute(task, ws))
    assert result.status == "completed"
    assert len(clients) == 2  # First client closed, second client created
    assert clients[0].closed is True
    assert clients[0].requests == ["thread/resume"]
    assert clients[1].requests == ["thread/resume", "turn/start"]


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


def test_codex_recreated_client_uses_durable_home_and_resumes_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test 19: Message 1 completes, backend process is deliberately destroyed,
    message 2 recreates app-server using the same durable Codex home, thread/resume succeeds,
    then turn/start succeeds.
    """
    monkeypatch.setattr(
        "mana_agent.integrations.codex.backend._git_changed_files",
        lambda *args, **kwargs: ["README.md"],
    )
    clients: list[Any] = []

    class LifecycleClient:
        def __init__(self, command: tuple[str, ...]) -> None:
            clients.append(self)
            self.running = True
            self.closed = False
            self.requests: list[str] = []

        async def start(self) -> None:
            return None

        async def request(self, method: str, params: dict[str, Any], *, timeout_seconds: float | None = None) -> dict[str, Any]:
            self.requests.append(method)
            if method == "thread/start":
                return {"thread": {"id": "thread-durable-123"}}
            if method == "thread/resume":
                return {"thread": {"id": params.get("threadId", "thread-durable-123")}}
            if method == "turn/start":
                return {"turn": {"id": "turn-1"}}
            return {}

        async def notifications(self, thread_id: str):
            yield {
                "method": "turn/completed",
                "params": {"threadId": thread_id, "turn": {"id": "turn-1"}},
            }

        async def close(self, *, wait_timeout: float = 1.0) -> None:
            self.closed = True
            self.running = False

    ws = _workspace(tmp_path)
    shim = CodexCodingAgentShim(
        repo_root=ws.repository_path,
        codex_settings=_settings(),
        session_id="session-durable-test",
        repository_id="repo-durable-test",
        backend_factory=lambda: CodexCodingBackend(
            _settings(),
            session_id=shim.session_id,
            repository_id=shim.repository_id or "",
            resume_thread_id=shim.resume_thread_id,
            client_factory=lambda cmd: LifecycleClient(cmd),
        ),
    )

    try:
        # Message 1 completes
        res1 = shim.generate("First turn")
        assert res1.get("status") == "completed"
        assert shim.resume_thread_id == "thread-durable-123"
        assert len(clients) == 1
        assert clients[0].requests == ["thread/start", "turn/start"]
        # Verify thread was persisted in durable store
        assert load_codex_session_thread("repo-durable-test", "session-durable-test") == "thread-durable-123"

        # Deliberately destroy the backend process
        shim._runner.run(shim._session_backend.close())
        assert clients[0].closed is True

        # Message 2 recreates app-server using the same durable Codex home
        res2 = shim.generate("Second turn after process destruction")
        assert res2.get("status") == "completed"
        assert len(clients) == 2
        assert clients[1].requests == ["thread/resume", "turn/start"]
        assert shim.resume_thread_id == "thread-durable-123"
    finally:
        shim.close()


def test_codex_different_worktree_paths_in_same_session_reuses_backend_and_rollout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test 20: Message 1 and message 2 use different task/worktree paths but the same Mana session;
    this must not lose the Codex rollout or recreate the backend.
    """
    monkeypatch.setattr(
        "mana_agent.integrations.codex.backend._git_changed_files",
        lambda *args, **kwargs: ["README.md"],
    )
    clients: list[Any] = []

    class ReusedClient:
        def __init__(self, command: tuple[str, ...]) -> None:
            clients.append(self)
            self.running = True
            self.closed = False
            self.requests: list[str] = []
            self.turn_cwds: list[str] = []

        async def start(self) -> None:
            return None

        async def request(self, method: str, params: dict[str, Any], *, timeout_seconds: float | None = None) -> dict[str, Any]:
            self.requests.append(method)
            if method == "thread/start":
                return {"thread": {"id": "thread-reused-worktree"}}
            if method == "turn/start":
                self.turn_cwds.append(str(params.get("cwd") or ""))
                return {"turn": {"id": "turn-1"}}
            return {}

        async def notifications(self, thread_id: str):
            yield {
                "method": "turn/completed",
                "params": {"threadId": thread_id, "turn": {"id": "turn-1"}},
            }

        async def close(self, *, wait_timeout: float = 1.0) -> None:
            self.closed = True
            self.running = False

    ws = _workspace(tmp_path)
    shim = CodexCodingAgentShim(
        repo_root=ws.repository_path,
        codex_settings=_settings(),
        session_id="session-worktree-reuse",
        repository_id="repo-worktree-reuse",
        backend_factory=lambda: CodexCodingBackend(
            _settings(),
            session_id=shim.session_id,
            repository_id=shim.repository_id or "",
            resume_thread_id=shim.resume_thread_id,
            client_factory=lambda cmd: ReusedClient(cmd),
        ),
    )

    try:
        # Message 1
        res1 = shim.generate("Turn 1 with default worktree")
        assert res1.get("status") == "completed"
        assert len(clients) == 1
        assert clients[0].requests == ["thread/start", "turn/start"]

        # Message 2 in same session with custom working directory
        sub_dir = ws.repository_path / "subdir"
        sub_dir.mkdir(exist_ok=True)
        shim.working_directory = sub_dir
        res2 = shim.generate("Turn 2 with different directory")
        assert res2.get("status") == "completed"
        # Backend and client MUST be reused!
        assert len(clients) == 1
        assert clients[0].requests == ["thread/start", "turn/start", "turn/start"]
        assert "thread/resume" not in clients[0].requests
    finally:
        shim.close()


def test_codex_missing_rollout_recovers_with_fresh_thread_and_completes_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test 23: Legacy-state test where stored thread ID has no rollout (-32600):
    exactly one controlled fresh-thread recovery occurs, new thread created and saved,
    and the current coding request proceeds without an infinite resume loop.
    """
    monkeypatch.setattr(
        "mana_agent.integrations.codex.backend._git_changed_files",
        lambda *args, **kwargs: ["README.md"],
    )
    clients: list[Any] = []

    class MissingRolloutClient:
        def __init__(self, command: tuple[str, ...]) -> None:
            clients.append(self)
            self.running = True
            self.closed = False
            self.requests: list[str] = []

        async def start(self) -> None:
            return None

        async def request(self, method: str, params: dict[str, Any], *, timeout_seconds: float | None = None) -> dict[str, Any]:
            self.requests.append(method)
            if method == "thread/resume":
                # Simulate Codex -32600: no rollout found for thread id
                raise CodexThreadStateMissingError(
                    "Codex JSON-RPC error: {'code': -32600, 'message': 'no rollout found for thread id stale-thread-old'}",
                    original_error="{'code': -32600, 'message': 'no rollout found for thread id stale-thread-old'}",
                )
            if method == "thread/start":
                return {"thread": {"id": "thread-recovered-fresh"}}
            if method == "turn/start":
                return {"turn": {"id": "turn-recovered-1"}}
            return {}

        async def notifications(self, thread_id: str):
            yield {
                "method": "turn/completed",
                "params": {"threadId": thread_id, "turn": {"id": "turn-recovered-1"}},
            }

        async def close(self, *, wait_timeout: float = 1.0) -> None:
            self.closed = True
            self.running = False

    ws = _workspace(tmp_path)
    backend = CodexCodingBackend(
        _settings(),
        session_id="legacy-session-id",
        repository_id="legacy-repo-id",
        resume_thread_id="stale-thread-old",
        client_factory=lambda cmd: MissingRolloutClient(cmd),
    )
    task = CodingTask(task_id="missing-rollout-recovery-task", goal="Recover missing rollout", requires_repository_write=False)
    result = asyncio.run(backend.execute(task, ws))

    # Turn completes successfully
    assert result.status == "completed"
    assert result.thread_id == "thread-recovered-fresh"
    assert backend.resume_thread_id == "thread-recovered-fresh"
    # Exactly one client, with resume -> start -> turn/start (NO infinite retry!)
    assert len(clients) == 1
    assert clients[0].requests == ["thread/resume", "thread/start", "turn/start"]
    # Durable store now has the recovered thread ID
    assert load_codex_session_thread("legacy-repo-id", "legacy-session-id") == "thread-recovered-fresh"


def test_codex_generation_tracking_prevents_stale_residency(tmp_path: Path) -> None:
    """Regression test 10: Generation tracking ensures a thread marked loaded in process A
    is not considered resident in restarted process B.
    """
    client = AsyncCodexAppServer(("codex", "app-server"))
    assert client.client_generation == 0

    # Start generation 1
    client.generation = 1
    client.mark_thread_loaded("th-1")
    assert client.is_thread_loaded("th-1") is False  # not running

    # Mock running process
    client._process = MagicMock(returncode=None)
    assert client.is_thread_loaded("th-1") is True
    assert client.is_thread_loaded("th-1", generation=1) is True
    assert client.is_thread_loaded("th-1", generation=2) is False

    # Simulate process restart (generation increments to 2)
    client.generation = 2
    client.clear_loaded_threads()
    assert client.is_thread_loaded("th-1") is False
    assert client.is_thread_loaded("th-1", generation=1) is False


def test_codex_durable_session_home_never_contains_credentials(tmp_path: Path) -> None:
    """Regression test 3 & 25: API/provider credentials must never be written to durable files."""
    from mana_agent.integrations.codex.runtime_config import CodexRuntimeConfigBuilder
    from mana_agent.integrations.codex.runtime_environment import CodexRuntimeEnvironment

    settings = _settings(api_key="super-secret-mana-token-12345")
    runtime_config = CodexRuntimeConfigBuilder.build(settings)

    session_home = get_codex_session_home("test-repo-cred", "test-session-cred")
    context = CodexRuntimeEnvironment.create(runtime_config, home=session_home, durable=True)
    try:
        config_path = session_home / "config.toml"
        assert config_path.is_file()
        config_text = config_path.read_text(encoding="utf-8")
        assert "super-secret-mana-token-12345" not in config_text

        # Verify thread_state.json also does not have credentials
        save_codex_session_thread("test-repo-cred", "test-session-cred", "th-secret-check")
        state_file = session_home / "thread_state.json"
        assert state_file.is_file()
        state_text = state_file.read_text(encoding="utf-8")
        assert "super-secret-mana-token-12345" not in state_text
        assert "th-secret-check" in state_text
    finally:
        context.close()
        # Home must survive close because durable=True
        assert session_home.is_dir()
