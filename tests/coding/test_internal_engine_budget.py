from __future__ import annotations

import asyncio
from types import SimpleNamespace

from mana_agent.coding.internal_engine import InternalCodingEngine
from mana_agent.coding.models import CodingTask, WorkspaceContext


def _registered(name, callback):
    from mana_agent.tools.contracts import ToolContract, coding_tool_contracts
    contract = next((c for c in coding_tool_contracts() if c.name.replace(".", "_") == name), None)
    callback.tool_contract = contract or ToolContract(name=name, description="Test callback", input_schema={"type": "object"}, output_schema={}, error_format={}, required_capability="test_callback", effect="read_only", repository_operation=False)
    return callback


class _Model:
    """Return more than 30 tool rounds before reporting task completion."""

    def __init__(self, tool_rounds: int) -> None:
        self.tool_rounds = tool_rounds
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        if self.calls < self.tool_rounds:
            self.calls += 1
            return SimpleNamespace(
                content="",
                tool_calls=[{"name": "noop", "args": {}, "id": f"call-{self.calls}"}],
            )
        return SimpleNamespace(content="Task completed", tool_calls=[])


class _Memory:
    def retrieve_relevant(self, **_kwargs):
        return []


def test_internal_coding_lifecycle_completes_after_more_than_30_model_turns(tmp_path) -> None:
    model = _Model(tool_rounds=31)
    engine = InternalCodingEngine(
        llm_client=model,
        supports_responses_api=False,
        programmatic_tool_calling_enabled=False,
        max_tool_calls=50,
        extra_tools={"noop": _registered("noop", lambda: "ok")},
        global_memory=_Memory(),
        batch_manager=object(),
        parallel_scheduling_enabled=False,
    )
    task = CodingTask(task_id="long-loop", goal="Exercise the internal model loop")
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        working_directory=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )

    result = asyncio.run(engine.execute(task, workspace))

    assert model.calls == 31
    assert result.status == "completed"
    assert result.summary == "Task completed"
    assert result.errors == []
