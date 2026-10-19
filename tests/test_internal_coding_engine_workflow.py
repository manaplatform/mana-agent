from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mana_agent.coding.facade import UnifiedCodingAgentFacade
from mana_agent.coding.internal_engine import InternalCodingEngine
from mana_agent.coding.models import AgentEvent, CodingTask, WorkspaceContext
from mana_agent.coding.live_events import coding_event_scope


class _FakeMultiToolLLM:
    """Simulates an LLM returning multiple parallel tool calls on turn 1, then completing on turn 2."""

    def __init__(self, tool_calls_plan: list[list[dict[str, Any]]], final_response: str = "Work verified and complete.") -> None:
        self.tool_calls_plan = tool_calls_plan
        self.final_response = final_response
        self.turn = 0

    def bind_tools(self, tools: list[dict[str, Any]]) -> "_FakeMultiToolLLM":
        return self

    async def ainvoke(self, messages: list[Any]) -> Any:
        if self.turn < len(self.tool_calls_plan):
            tcs = self.tool_calls_plan[self.turn]
            self.turn += 1
            return SimpleNamespace(
                tool_calls=tcs,
                content="",
            )
        return SimpleNamespace(
            tool_calls=[],
            content=self.final_response,
        )


class _MockSupervisor:
    def __init__(self) -> None:
        self.recorded_actions: list[dict[str, Any]] = []

    def record_action(self, task_id: str, action_name: str, changed_files: list[str]) -> None:
        self.recorded_actions.append({"task_id": task_id, "action": action_name, "files": changed_files})


class _MockTaskBoard:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update_task(self, task_id: str, progress_summary: str, changed_files: list[str]) -> None:
        self.updates.append({"task_id": task_id, "summary": progress_summary, "files": changed_files})


class _MockGlobalMemory:
    def __init__(self) -> None:
        self.failures: list[dict[str, Any]] = []

    def retrieve_relevant(self, action_or_query: str, limit: int = 5) -> list[Any]:
        return []

    def format_retrieved_rules(self, memories: list[Any]) -> str:
        return ""

    def record_failure(self, category: str, attempted_action: str, error: str, metadata: dict[str, Any] | None = None) -> None:
        self.failures.append({"category": category, "action": attempted_action, "error": error, "meta": metadata})


def test_internal_coding_engine_parallel_workflow(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")

    # Turn 0: Parallel reads of a.py and b.py
    # Turn 1: Batch write modifying both files
    llm = _FakeMultiToolLLM(
        tool_calls_plan=[
            [
                {"id": "c1", "name": "read_file", "args": {"path": "a.py"}},
                {"id": "c2", "name": "read_file", "args": {"path": "b.py"}},
            ],
            [
                {
                    "id": "c3",
                    "name": "batch_write_files",
                    "args": {
                        "writes": [
                            {"path": "a.py", "content": "x = 100\n"},
                            {"path": "b.py", "content": "y = 200\n"},
                        ]
                    },
                }
            ],
        ]
    )

    supervisor = _MockSupervisor()
    taskboard = _MockTaskBoard()

    engine = InternalCodingEngine(
        model="fake-model",
        llm_client=llm,
        execution_supervisor=supervisor,
        taskboard=taskboard,
        max_concurrency=4,
    )

    task = CodingTask(
        task_id="task_parallel_test",
        goal="Read both files and update values",
        allowed_files=["a.py", "b.py"],
    )
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )

    events_emitted: list[AgentEvent] = []

    def _collector(ev: AgentEvent) -> None:
        events_emitted.append(ev)

    with coding_event_scope(_collector):
        result = asyncio.run(engine.execute(task, workspace))

    assert result.status == "completed"
    assert "a.py" in result.changed_files
    assert "b.py" in result.changed_files
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 100\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "y = 200\n"

    # Verify event stream contained parallel groups and phase tracking
    event_types = [e.event_type for e in events_emitted]
    assert "parallel.group.started" in event_types
    assert "parallel.group.completed" in event_types
    assert "turn.completed" in event_types

    # Verify supervisor & taskboard integrations
    assert len(supervisor.recorded_actions) > 0
    assert len(taskboard.updates) > 0


def test_internal_coding_engine_global_memory_recording_on_error(tmp_path: Path) -> None:
    # Tool call targeting nonexistent path
    llm = _FakeMultiToolLLM(
        tool_calls_plan=[
            [
                {"id": "c_bad", "name": "batch_delete_files", "args": {"paths": ["../../etc/forbidden"]}},
                {"id": "c_good", "name": "list_files", "args": {}},
            ]
        ]
    )
    memory = _MockGlobalMemory()

    engine = InternalCodingEngine(
        model="fake-model",
        llm_client=llm,
        global_memory=memory,
        max_concurrency=4,
    )

    task = CodingTask(
        task_id="task_mem_error_test",
        goal="Attempt invalid operation",
    )
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )

    asyncio.run(engine.execute(task, workspace))

    # GlobalMemory should record failure signature
    assert len(memory.failures) > 0
    assert any(f["category"] == "tool" for f in memory.failures)


def test_unified_facade_external_codex_remains_functional(tmp_path: Path) -> None:
    """Verify that external Codex mode remains fully backward compatible and unaffected."""
    facade = UnifiedCodingAgentFacade(
        repo_root=tmp_path,
        working_directory=tmp_path,
        default_mode="external",
    )
    assert facade.active_engine_mode == "external"

    facade.set_engine_mode("internal")
    assert facade.active_engine_mode == "internal"

    facade.set_engine_mode("external")
    assert facade.active_engine_mode == "external"
