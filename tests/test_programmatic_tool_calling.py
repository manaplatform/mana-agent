from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from mana_agent.coding.internal_engine import InternalCodingEngine
from mana_agent.coding.models import CodingTask, WorkspaceContext
from mana_agent.coding.programmatic import (
    ProgrammaticExecutionSession,
    ToolExecutionDispatcher,
    build_programmatic_tool_definitions,
)


def test_programmatic_tool_definitions_contains_hosted_tool_and_allowed_callers() -> None:
    """Check tool schemas for allowed_callers and output_schema."""
    tools = build_programmatic_tool_definitions()

    # Must contain the hosted tool
    assert {"type": "programmatic_tool_calling"} in tools

    # Check function tool definitions
    fn_tools = [t for t in tools if t.get("type") == "function"]
    assert len(fn_tools) > 0

    tool_names = {t["name"] for t in fn_tools}
    assert {"read_file", "write_file", "edit_file", "apply_patch", "list_files", "repo_search"} <= tool_names

    for t in fn_tools:
        assert "allowed_callers" in t
        assert "programmatic" in t["allowed_callers"]
        assert "parameters" in t
        assert t["parameters"]["type"] == "object"


def test_tool_execution_dispatcher_file_ops_and_sandboxing(tmp_path: Path) -> None:
    """Tool dispatcher executes read, write, edit, and obeys sandbox."""
    workspace_write = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )
    dispatcher = ToolExecutionDispatcher(workspace_write)

    # 1. Write file
    res_w = dispatcher.dispatch("write_file", {"path": "module.py", "content": "def func():\n    return 42\n"})
    assert res_w["ok"] is True
    assert (tmp_path / "module.py").exists()

    # 2. Read file
    res_r = dispatcher.dispatch("read_file", {"path": "module.py"})
    assert res_r["ok"] is True
    assert "return 42" in res_r["content"]

    # 3. Edit file
    res_e = dispatcher.dispatch("edit_file", {"path": "module.py", "old_string": "42", "new_string": "100"})
    assert res_e["ok"] is True
    assert (tmp_path / "module.py").read_text(encoding="utf-8") == "def func():\n    return 100\n"

    # 4. Sandbox readOnly blocks write tools
    workspace_readonly = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="readOnly",
        allow_in_place_write=True,
    )
    dispatcher_ro = ToolExecutionDispatcher(workspace_readonly)
    res_blocked = dispatcher_ro.dispatch("write_file", {"path": "blocked.py", "content": "x = 1"})
    assert res_blocked["ok"] is False
    assert res_blocked["error_code"] == "SANDBOX_VIOLATION"


def test_tool_execution_dispatcher_custom_and_mcp_tools(tmp_path: Path) -> None:
    """Custom/MCP tools can be registered and invoked."""
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )

    def custom_echo(text: str) -> dict[str, Any]:
        return {"ok": True, "echo": text.upper()}

    dispatcher = ToolExecutionDispatcher(workspace, extra_tools={"custom_echo": custom_echo})
    res = dispatcher.dispatch("custom_echo", {"text": "mana"})
    assert res["ok"] is True
    assert res["echo"] == "MANA"


def test_programmatic_session_single_and_parallel_tool_calls(tmp_path: Path) -> None:
    """Programmatic execution session handles program, function calls, and preserves call_id."""
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )
    dispatcher = ToolExecutionDispatcher(workspace)
    session = ProgrammaticExecutionSession(
        task_id="prog_task_1",
        dispatcher=dispatcher,
        max_tool_calls=10,
    )

    # 1. Incoming program item
    session.handle_program_item({
        "call_id": "prog_123",
        "code": "const f1 = await tools.write_file({path: 'a.txt', content: 'A'});\nconst f2 = await tools.write_file({path: 'b.txt', content: 'B'});",
    })
    assert session.state.program_id == "prog_123"

    # 2. Parallel / multiple independent tool calls
    call1 = {
        "call_id": "call_1",
        "name": "write_file",
        "arguments": json.dumps({"path": "a.txt", "content": "Hello A"}),
        "caller": {"type": "program", "call_id": "prog_123"},
    }
    call2 = {
        "call_id": "call_2",
        "name": "write_file",
        "arguments": json.dumps({"path": "b.txt", "content": "Hello B"}),
        "caller": {"type": "program", "call_id": "prog_123"},
    }

    out1 = session.handle_function_call(call1)
    out2 = session.handle_function_call(call2)

    assert out1["type"] == "function_call_output"
    assert out1["call_id"] == "call_1"
    assert json.loads(out1["output"])["ok"] is True

    assert out2["type"] == "function_call_output"
    assert out2["call_id"] == "call_2"
    assert json.loads(out2["output"])["ok"] is True

    assert "a.txt" in session.state.changed_files
    assert "b.txt" in session.state.changed_files

    # 3. Incoming program_output item
    session.handle_program_output({
        "call_id": "prog_123",
        "output": "Files created successfully",
    })


def test_programmatic_session_budget_limit(tmp_path: Path) -> None:
    """Session enforces max tool calls budget."""
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )
    dispatcher = ToolExecutionDispatcher(workspace)
    session = ProgrammaticExecutionSession(
        task_id="budget_task",
        dispatcher=dispatcher,
        max_tool_calls=2,
    )

    # Call 1 & 2 pass
    session.handle_function_call({"call_id": "c1", "name": "list_files", "arguments": {}})
    session.handle_function_call({"call_id": "c2", "name": "list_files", "arguments": {}})

    # Call 3 exceeds budget
    out3 = session.handle_function_call({"call_id": "c3", "name": "list_files", "arguments": {}})
    data3 = json.loads(out3["output"])
    assert data3["ok"] is False
    assert data3["error_code"] == "BUDGET_EXCEEDED"


def test_direct_tool_calling_fallback_for_unsupported_models(tmp_path: Path) -> None:
    """Internal engine falls back to direct tool calling for unsupported models/providers."""
    f = tmp_path / "readme.txt"
    f.write_text("Test direct fallback", encoding="utf-8")

    # Provider = openrouter, supports_responses_api = False
    engine = InternalCodingEngine(
        model="deepseek-coder",
        provider="openrouter",
        supports_responses_api=False,
    )
    assert engine._should_use_programmatic() is False

    task = CodingTask(
        task_id="direct_fallback_task",
        goal="Direct tool execution",
        allowed_files=["readme.txt"],
    )
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )

    res = asyncio.run(engine.execute(task, workspace))
    assert res.status == "completed"
    assert res.backend == "internal"
    assert "direct tool" in res.summary.lower()
