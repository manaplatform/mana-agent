from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from mana_agent.coding.internal_engine import InternalCodingEngine
from mana_agent.coding.models import CodingTask, WorkspaceContext


class _ProgressiveModel:
    """Simulates an LLM that modifies files and continues for several tool rounds."""

    def __init__(self, total_tool_calls: int = 8) -> None:
        self.total_tool_calls = total_tool_calls
        self.call_count = 0

    def bind_tools(self, _tools: object) -> "_ProgressiveModel":
        return self

    async def ainvoke(self, messages: list[object]) -> SimpleNamespace:
        # Check if this is an evaluation prompt from _evaluate_progress_and_renew_budget
        for msg in messages:
            content = str(getattr(msg, "content", "") or "")
            if "progress evaluation judge" in content:
                return SimpleNamespace(
                    content=json.dumps({
                        "improvement_accepted": True,
                        "reason": "Files have been modified and progress is demonstrable.",
                        "confidence": 1.0,
                    }),
                    tool_calls=[],
                )

        if self.call_count < self.total_tool_calls:
            self.call_count += 1
            # Perform a file write on turn 1 to show concrete progress
            if self.call_count == 1:
                return SimpleNamespace(
                    content="",
                    tool_calls=[{
                        "name": "write_file",
                        "args": {"path": "output.txt", "content": "hello world\n"},
                        "id": f"call-{self.call_count}",
                    }],
                )
            return SimpleNamespace(
                content="",
                tool_calls=[{
                    "name": "noop",
                    "args": {},
                    "id": f"call-{self.call_count}",
                }],
            )

        return SimpleNamespace(content="Task successfully completed with budget renewal.", tool_calls=[])


class _NoProgressModel:
    """Simulates an LLM that calls noop without touching files or achieving progress."""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, _tools: object) -> "_NoProgressModel":
        return self

    async def ainvoke(self, messages: list[object]) -> SimpleNamespace:
        for msg in messages:
            content = str(getattr(msg, "content", "") or "")
            if "progress evaluation judge" in content:
                return SimpleNamespace(
                    content=json.dumps({
                        "improvement_accepted": False,
                        "reason": "No meaningful progress detected; identical repeating calls.",
                        "confidence": 1.0,
                    }),
                    tool_calls=[],
                )
        self.calls += 1
        return SimpleNamespace(
            content="",
            tool_calls=[{"name": "noop", "args": {}, "id": f"noop-{self.calls}"}],
        )


class _DummyMemory:
    def retrieve_relevant(self, **_kwargs: object) -> list[object]:
        return []


def test_internal_engine_renews_budget_when_progress_accepted(tmp_path: Path) -> None:
    # Set initial max_tool_calls to 3; model wants 8 tool calls.
    # Because a file is written and model accepts progress, it should auto-renew and complete!
    model = _ProgressiveModel(total_tool_calls=7)
    engine = InternalCodingEngine(
        llm_client=model,
        supports_responses_api=False,
        programmatic_tool_calling_enabled=False,
        max_tool_calls=3,
        extra_tools={"noop": lambda: "ok"},
        global_memory=_DummyMemory(),
        batch_manager=object(),
        parallel_scheduling_enabled=False,
    )
    engine.renew_tool_calls = 5
    engine.renew_model_turns = 10
    engine.max_auto_renewals = 3

    task = CodingTask(task_id="task_auto_renew", goal="Implement feature with budget extension")
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        working_directory=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )

    result = asyncio.run(engine.execute(task, workspace))

    assert result.status == "completed"
    assert "budget renewal" in result.summary.lower()
    assert (tmp_path / "output.txt").is_file()
    assert engine.renewals_count >= 1


def test_internal_engine_rejects_renewal_without_progress(tmp_path: Path) -> None:
    # Model makes no progress (no files changed, no commands run).
    # Engine should not renew and fail cleanly on budget exhaustion.
    model = _NoProgressModel()
    engine = InternalCodingEngine(
        llm_client=model,
        supports_responses_api=False,
        programmatic_tool_calling_enabled=False,
        max_tool_calls=3,
        extra_tools={"noop": lambda: "ok"},
        global_memory=_DummyMemory(),
        batch_manager=object(),
        parallel_scheduling_enabled=False,
    )
    engine.max_auto_renewals = 2

    task = CodingTask(task_id="task_stalled", goal="Stalled task without progress")
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        working_directory=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )

    result = asyncio.run(engine.execute(task, workspace))

    assert result.status == "failed"
    assert "budget" in result.summary.lower() or any("budget" in e.lower() for e in result.errors)
    assert engine.renewals_count == 0
