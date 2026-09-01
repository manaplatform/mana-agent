"""Regression suite: Codex console output live streaming & timeout recovery.

Verifies:
1. Normal execution: command output chunks (item/commandExecution/outputDelta)
   are published live with progress visibility and output_preview to console/UI subscribers.
2. Timeout before first output: clean interruption, error emission, and taskboard cleanup.
3. Timeout after partial output: already-received output chunks are preserved in trace.
4. Retry after timeout: clean state reattachment, no duplicate output, fresh stream.
5. Session /new lifecycle: clean reader/process/queue termination.
6. Successful command output after a previous timeout.
7. Exit code 0 separation: command success does not equal task completion if turn fails.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import pytest

from mana_agent.coding.event_visibility import (
    EventSemanticKind,
    EventVisibility,
    classify_coding_event,
    is_user_publishable,
    progress_event_payload,
)
from mana_agent.coding.live_events import (
    coding_event_scope,
    subscribe_coding_events,
)
from mana_agent.coding.models import (
    AgentEvent,
    CodingTask,
    CodingTaskResult,
    WorkspaceContext,
)
from mana_agent.integrations.codex.backend import (
    CodexCodingBackend,
)
from mana_agent.integrations.codex.client import AsyncCodexAppServer
from mana_agent.integrations.codex.coding_agent_shim import CodexCodingAgentShim
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.event_adapter import adapt_codex_event
from mana_agent.integrations.codex.result_parser import parse_codex_result


def _git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not (path / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=path, check=True)
        (path / "app.py").write_text("print('hello')\n", encoding="utf-8")
        subprocess.run(["git", "add", "app.py"], cwd=path, check=True)
        subprocess.run(["git", "commit", "-qm", "initial"], cwd=path, check=True)


def _workspace(tmp_path: Path) -> WorkspaceContext:
    repo = tmp_path / "repo"
    _git_repo(repo)
    return WorkspaceContext(
        repository_path=repo,
        worktree_path=repo,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )


class _MockCodexClient:
    def __init__(self) -> None:
        self.running = True
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.interrupted: list[tuple[str, str]] = []
        self._notification_queues: dict[str, list[dict[str, Any]]] = {}
        self.delay_before_notifications: float = 0.0
        self.closed = False

    async def start(self) -> None:
        return None

    async def request(self, method: str, params: dict[str, Any], *, timeout_seconds: float | None = None) -> dict[str, Any]:
        self.requests.append((method, params))
        if method == "thread/start":
            return {"thread": {"id": params.get("threadId") or "thread-mock-1"}}
        if method == "turn/start":
            return {"turn": {"id": "turn-mock-1"}}
        if method == "turn/interrupt":
            self.interrupted.append((params.get("threadId", ""), params.get("turnId", "")))
            return {"status": "ok"}
        return {}

    def queue_notifications(self, thread_id: str, items: list[dict[str, Any]]) -> None:
        self._notification_queues[thread_id] = list(items)

    async def notifications(self, thread_id: str):
        if self.delay_before_notifications > 0:
            await asyncio.sleep(self.delay_before_notifications)
        for item in self._notification_queues.get(thread_id, []):
            params = item.get("params") if isinstance(item.get("params"), dict) else {}
            delay = item.pop("_delay", 0) or params.pop("_delay", 0)
            if delay:
                await asyncio.sleep(delay)
            yield item

    async def interrupt(self, *, thread_id: str, turn_id: str, timeout_seconds: float = 2.0) -> None:
        self.interrupted.append((thread_id, turn_id))

    async def deny_server_request(self, request: dict[str, Any]) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# 1. Normal execution: command output chunks streamed live to subscribers
# ---------------------------------------------------------------------------


def test_codex_command_output_streamed_to_subscribers_and_ui(tmp_path: Path) -> None:
    """Verify item/commandExecution/outputDelta emits command.output with progress visibility."""
    script = [
        {"method": "turn/started", "params": {"threadId": "th-1", "turnId": "tu-1"}},
        {
            "method": "item/started",
            "params": {
                "threadId": "th-1",
                "item": {"id": "cmd-1", "type": "commandExecution", "command": "pytest -q"},
            },
        },
        {
            "method": "item/commandExecution/outputDelta",
            "params": {
                "threadId": "th-1",
                "itemId": "cmd-1",
                "delta": "collected 5 items\n",
            },
        },
        {
            "method": "item/commandExecution/outputDelta",
            "params": {
                "threadId": "th-1",
                "itemId": "cmd-1",
                "delta": "..... [100%]\n5 passed in 0.12s\n",
            },
        },
        {
            "method": "item/completed",
            "params": {
                "threadId": "th-1",
                "item": {
                    "id": "cmd-1",
                    "type": "commandExecution",
                    "command": "pytest -q",
                    "output": "collected 5 items\n..... [100%]\n5 passed in 0.12s\n",
                    "exitCode": 0,
                    "status": "completed",
                },
            },
        },
        {"method": "turn/completed", "params": {"threadId": "th-1", "turn": {"status": "completed"}}},
    ]

    events = [adapt_codex_event("task-1", n, requires_repository_write=False) for n in script]

    # Verify event adapter classification
    output_events = [e for e in events if e.event_type == "command.output"]
    assert len(output_events) == 2
    for oe in output_events:
        assert oe.visibility == EventVisibility.PROGRESS.value
        assert oe.semantic_kind == EventSemanticKind.COMMAND.value
        assert is_user_publishable(oe.visibility)
        assert oe.output_preview

    # Verify shim emission delivers to live coding subscribers
    received: list[AgentEvent] = []
    unsubscribe = subscribe_coding_events(received.append)

    client = _MockCodexClient()
    client.queue_notifications("thread-mock-1", script)
    backend = CodexCodingBackend(CodexSettings(enabled=True), client_factory=lambda *a: client)

    shim = CodexCodingAgentShim(
        repo_root=tmp_path,
        codex_settings=CodexSettings(enabled=True),
        backend_factory=lambda: backend,
    )

    try:
        async def _run():
            async for ev in backend.stream(
                CodingTask(task_id="task-1", goal="run tests", requires_repository_write=False),
                _workspace(tmp_path),
            ):
                shim._emit_event(ev, requires_repository_write=False)

        asyncio.run(_run())
    finally:
        unsubscribe()

    emitted_outputs = [e for e in received if e.event_type == "command.output"]
    assert len(emitted_outputs) == 2
    assert "collected 5 items" in emitted_outputs[0].output_preview
    assert "5 passed" in emitted_outputs[1].output_preview


# ---------------------------------------------------------------------------
# 2. Timeout before first output
# ---------------------------------------------------------------------------


def test_codex_timeout_before_first_output(tmp_path: Path) -> None:
    """Verify timeout before any output triggers interruption and clean error event."""
    client = _MockCodexClient()
    # Hang before yielding any notification
    client.delay_before_notifications = 1.5

    backend = CodexCodingBackend(
        CodexSettings(enabled=True, task_timeout_seconds=1),
        client_factory=lambda *a: client,
    )

    received: list[AgentEvent] = []
    task = CodingTask(task_id="task-timeout-1", goal="hang task", requires_repository_write=False)

    async def _run():
        async for ev in backend.stream(task, _workspace(tmp_path)):
            received.append(ev)

    asyncio.run(_run())

    # Verify error event was emitted
    errors = [e for e in received if e.event_type == "error"]
    assert len(errors) == 1
    assert "timed out" in errors[0].title.lower() or "timed out" in errors[0].error.lower()
    assert errors[0].status == "failed"

    # Verify active task was cleanly removed
    assert task.task_id not in backend._active

    # Verify interrupt was called
    assert len(client.interrupted) == 1


# ---------------------------------------------------------------------------
# 3. Timeout after partial output
# ---------------------------------------------------------------------------


def test_codex_timeout_after_partial_output(tmp_path: Path) -> None:
    """Verify partial output is preserved and visible when a subsequent timeout occurs."""
    client = _MockCodexClient()
    script = [
        {"method": "turn/started", "params": {"threadId": "th-1", "turnId": "tu-1"}},
        {
            "method": "item/started",
            "params": {
                "threadId": "th-1",
                "item": {"id": "cmd-1", "type": "commandExecution", "command": "cargo test"},
            },
        },
        {
            "method": "item/commandExecution/outputDelta",
            "params": {
                "threadId": "th-1",
                "itemId": "cmd-1",
                "delta": "   Compiling mana v0.1.0\n",
            },
        },
        {
            "method": "item/commandExecution/outputDelta",
            "params": {
                "threadId": "th-1",
                "itemId": "cmd-1",
                "delta": "    Running unittests\n",
                "_delay": 1.5,  # will trigger task_timeout_seconds (1s)
            },
        },
    ]
    client.queue_notifications("thread-mock-1", script)

    backend = CodexCodingBackend(
        CodexSettings(enabled=True, task_timeout_seconds=1),
        client_factory=lambda *a: client,
    )

    received: list[AgentEvent] = []
    task = CodingTask(task_id="task-partial-1", goal="partial run", requires_repository_write=False)

    async def _run():
        async for ev in backend.stream(task, _workspace(tmp_path)):
            received.append(ev)

    asyncio.run(_run())

    # Partial output must be present in received stream
    partial_outputs = [e for e in received if e.event_type == "command.output"]
    assert len(partial_outputs) >= 1
    assert "Compiling mana" in partial_outputs[0].output_preview

    # Error event was emitted at the end
    assert received[-1].event_type == "error"
    assert "timed out" in received[-1].error.lower() or "timed out" in received[-1].title.lower()


# ---------------------------------------------------------------------------
# 4. Retry after timeout: clean state & fresh stream
# ---------------------------------------------------------------------------


def test_codex_retry_after_timeout_cleans_state_and_reattaches(tmp_path: Path) -> None:
    """Verify retrying after a timeout establishes a fresh stream without duplicate output."""
    client = _MockCodexClient()
    # Turn 1: hangs and times out
    client.queue_notifications("thread-mock-1", [{"method": "turn/started", "_delay": 1.5}])

    backend = CodexCodingBackend(
        CodexSettings(enabled=True, task_timeout_seconds=1),
        client_factory=lambda *a: client,
    )

    task1 = CodingTask(task_id="task-retry-1", goal="attempt 1", requires_repository_write=False)
    events_turn1: list[AgentEvent] = []

    async def _run_turn1():
        async for ev in backend.stream(task1, _workspace(tmp_path)):
            events_turn1.append(ev)

    asyncio.run(_run_turn1())
    assert events_turn1[-1].event_type == "error"
    assert task1.task_id not in backend._active

    # Turn 2: succeeds immediately
    turn2_script = [
        {"method": "turn/started", "params": {"threadId": "thread-mock-1", "turnId": "tu-2"}},
        {
            "method": "item/commandExecution/outputDelta",
            "params": {"threadId": "thread-mock-1", "delta": "All tests passed!\n"},
        },
        {"method": "turn/completed", "params": {"threadId": "thread-mock-1", "turn": {"status": "completed"}}},
    ]
    client.queue_notifications("thread-mock-1", turn2_script)

    task2 = CodingTask(task_id="task-retry-2", goal="attempt 2", requires_repository_write=False)
    events_turn2: list[AgentEvent] = []

    async def _run_turn2():
        async for ev in backend.stream(task2, _workspace(tmp_path)):
            events_turn2.append(ev)

    asyncio.run(_run_turn2())

    # Turn 2 has fresh output without leftover Turn 1 error
    output_evs = [e for e in events_turn2 if e.event_type == "command.output"]
    assert len(output_evs) == 1
    assert "All tests passed!" in output_evs[0].output_preview
    assert events_turn2[-1].event_type == "turn.finalizing"


# ---------------------------------------------------------------------------
# 5. Session /new lifecycle: clean reader/process/queue termination
# ---------------------------------------------------------------------------


def test_codex_new_session_cleans_up_readers_and_queues() -> None:
    """Verify AsyncCodexAppServer.clear_notifications and close empty queues properly."""
    server = AsyncCodexAppServer(command=("codex", "app-server"))
    server._notifications["th-test"].put_nowait({"method": "item/test", "params": {}})
    server._notifications["th-test"].put_nowait({"method": "item/test2", "params": {}})

    assert not server._notifications["th-test"].empty()
    server.clear_notifications("th-test")
    assert server._notifications["th-test"].empty()

    server._notifications["th-test"].put_nowait({"method": "item/test3", "params": {}})
    asyncio.run(server.close())
    assert len(server._notifications) == 0


# ---------------------------------------------------------------------------
# 6. Exit code 0 separation: command success does not equal task completion
# ---------------------------------------------------------------------------


def test_command_exit_code_zero_does_not_complete_task_if_turn_fails(tmp_path: Path) -> None:
    """Verify command exit code 0 does not mark task completed if turn itself failed/cancelled."""
    notifications = [
        {"method": "turn/started", "params": {"threadId": "th-1"}},
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "commandExecution",
                    "command": "git status",
                    "exitCode": 0,
                    "status": "completed",
                }
            },
        },
        {
            "method": "turn/failed",
            "params": {
                "message": "Timed out waiting for upstream LLM",
                "error_code": "CODING_PROVIDER_TIMEOUT",
                "reason": "timeout",
            },
        },
    ]

    result = parse_codex_result(
        task=CodingTask(task_id="task-exit-sep", goal="test separation", requires_repository_write=False),
        workspace=_workspace(tmp_path),
        worker_id="w1",
        thread_id="th-1",
        turn_id="tu-1",
        notifications=notifications,
        changed_files=[],
    )

    assert result.status == "failed"
    assert any("CODING_PROVIDER_TIMEOUT" in e for e in result.errors)
    assert result.status != "completed"
