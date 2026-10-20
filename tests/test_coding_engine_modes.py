from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mana_agent.coding.engine import CodingEngine
from mana_agent.coding.external_engine import CodexProviderAdapter, ExternalCodingEngine
from mana_agent.coding.facade import UnifiedCodingAgentFacade
from mana_agent.coding.internal_engine import InternalCodingEngine
from mana_agent.coding.models import (
    AgentEvent,
    CodingBackendDecision,
    CodingEngineSettings,
    CodingStatus,
    CodingTask,
    CodingTaskResult,
    WorkspaceContext,
)
from mana_agent.coding.registry import CodingBackendDecisionError, CodingBackendRegistry
from mana_agent.coding.selection import (
    CodingBackendConfigurationError,
    resolve_coding_backend,
)
from mana_agent.integrations.codex.coding_agent_shim import CodexCodingAgentShim
from mana_agent.integrations.codex.config import CodexSettings


class _FakeCodexBackend:
    name = "codex"

    def __init__(self) -> None:
        self.started = False
        self.executed_tasks: list[CodingTask] = []
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def execute(self, task: CodingTask, workspace: WorkspaceContext) -> CodingTaskResult:
        self.executed_tasks.append(task)
        return CodingTaskResult(
            task_id=task.task_id,
            worker_id="fake-codex-worker",
            backend="codex",
            status="completed",
            summary=f"Codex executed: {task.goal}",
            changed_files=["file_from_codex.py"],
            commands_run=["pytest"],
        )

    async def stream(self, task: CodingTask, workspace: WorkspaceContext):
        yield AgentEvent(
            event_type="turn.started",
            task_id=task.task_id,
            backend="codex",
            title="Codex turn started",
        )
        yield AgentEvent(
            event_type="turn.completed",
            task_id=task.task_id,
            backend="codex",
            title="Codex turn completed",
            status="success",
        )

    async def cancel(self, task_id: str, timeout_seconds: float = 2.0) -> None:
        pass

    async def close(self) -> None:
        self.closed = True


def test_internal_engine_initialization_and_coding_execution(tmp_path: Path) -> None:
    """1. Internal engine initialization and coding execution."""
    src_file = tmp_path / "hello.py"
    src_file.write_text("print('hello')", encoding="utf-8")

    engine = InternalCodingEngine(
        model="gpt-4o",
        supports_responses_api=True,
        programmatic_tool_calling_enabled=True,
        llm_client=lambda task, workspace, tools, session: "Observed hello.py structure",
    )
    assert engine.mode == "internal"
    assert engine.name == "internal"

    task = CodingTask(
        task_id="task_int_1",
        goal="Read hello.py and verify structure",
        allowed_files=["hello.py"],
    )
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )

    result = asyncio.run(engine.execute(task, workspace))
    assert result.status == "completed"
    assert result.backend == "internal"
    assert result.task_id == "task_int_1"

    status = asyncio.run(engine.get_status("task_int_1"))
    assert status.status == "completed"
    assert status.backend == "internal"


def test_external_codex_execution_compatibility(tmp_path: Path) -> None:
    """2. External Codex execution compatibility."""
    fake_backend = _FakeCodexBackend()
    adapter = CodexProviderAdapter(lambda: fake_backend)
    engine = ExternalCodingEngine(provider_adapter=adapter)

    assert engine.mode == "external"
    assert engine.name == "codex"

    task = CodingTask(
        task_id="task_ext_1",
        goal="Implement external change",
    )
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )

    asyncio.run(engine.start())
    assert fake_backend.started is True

    result = asyncio.run(engine.execute(task, workspace))
    assert result.status == "completed"
    assert result.backend == "codex"
    assert result.changed_files == ["file_from_codex.py"]

    status = asyncio.run(engine.get_status("task_ext_1"))
    assert status.status == "completed"
    assert status.backend == "codex"


def test_switching_engine_modes_between_tasks(tmp_path: Path) -> None:
    """5. Switching engine modes between tasks."""
    facade = UnifiedCodingAgentFacade(
        repo_root=tmp_path,
        working_directory=tmp_path,
        default_mode="external",
        codex_settings=CodexSettings(),
    )
    assert facade.active_engine_mode == "external"

    facade.set_engine_mode("internal")
    assert facade.active_engine_mode == "internal"

    facade.set_engine_mode("external")
    assert facade.active_engine_mode == "external"

    # Switching to codex sets external
    facade.set_engine_mode("codex")
    assert facade.active_engine_mode == "external"

    with pytest.raises(ValueError, match="Invalid coding engine mode"):
        facade.set_engine_mode("unknown_engine")


def test_concurrent_workspace_modification_protection(tmp_path: Path) -> None:
    """10. Concurrent workspace modification protection."""
    facade = UnifiedCodingAgentFacade(
        repo_root=tmp_path,
        working_directory=tmp_path,
        default_mode="internal",
        codex_settings=CodexSettings(),
    )

    # Simulate an active task
    facade._task_in_progress = True
    facade._active_task_id = "running_task_1"

    # Switching engine while running is blocked
    with pytest.raises(RuntimeError, match="Cannot switch coding engine"):
        facade.set_engine_mode("external")

    # Starting another task while running is blocked
    with pytest.raises(RuntimeError, match="already executing in this workspace"):
        facade.generate("Another request")

    facade._task_in_progress = False
    facade._active_task_id = None


def test_per_task_and_persisted_configuration() -> None:
    """6. Per-task and persisted configuration."""
    cfg = CodingEngineSettings(
        mode="internal",
    )
    assert cfg.mode == "internal"
    assert cfg.internal.enabled is True
    assert cfg.internal.programmatic_tool_calling is True
    assert cfg.external.provider == "codex"

    # Selection resolution tests
    sel_int = resolve_coding_backend(SimpleNamespace(
        mana_coding_backend="internal",
        mana_internal_coding_enabled=True,
    ))
    assert sel_int.backend == "internal"

    sel_ext = resolve_coding_backend(SimpleNamespace(
        mana_coding_backend="external",
        mana_codex_enabled=True,
    ))
    assert sel_ext.backend == "external"

    sel_codex = resolve_coding_backend(SimpleNamespace(
        mana_coding_backend="codex",
        mana_codex_enabled=True,
    ))
    assert sel_codex.backend == "codex"

    # User config dictionary validation
    from mana_agent.config.user_config import validate_config_values
    cleaned_internal = validate_config_values({
        "coding_engine": {
            "mode": "internal",
            "internal": {
                "enabled": True,
                "programmatic_tool_calling": True,
                "use_registered_tools": True,
            },
            "external": {
                "provider": "codex",
                "enabled": True,
            },
        }
    })
    assert cleaned_internal["MANA_CODING_BACKEND"] == "internal"
    assert cleaned_internal["coding_engine"]["mode"] == "internal"

    cleaned_codex = validate_config_values({
        "coding_engine": {
            "mode": "external",
            "external": {
                "provider": "codex",
                "enabled": True,
            },
        }
    })
    assert cleaned_codex["MANA_CODING_BACKEND"] == "external"
    assert cleaned_codex["coding_engine"]["mode"] == "external"


def test_engine_cancellation_and_failure_recovery(tmp_path: Path) -> None:
    """9. Engine cancellation and failure recovery."""
    engine = InternalCodingEngine()
    task = CodingTask(task_id="task_cancel_1", goal="Task to cancel")
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )

    # Cancel before running
    asyncio.run(engine.cancel("task_cancel_1"))
    status = asyncio.run(engine.get_status("task_cancel_1"))
    assert status.status == "cancelled"


def test_registry_supports_internal_and_external() -> None:
    """Registry resolves internal and external backends."""
    registry = CodingBackendRegistry()
    fake_codex = _FakeCodexBackend()
    fake_internal = InternalCodingEngine()

    registry.register(fake_codex)
    registry.register(fake_internal)

    dec_ext = CodingBackendDecision(
        decision_id="dec_ext",
        coding_required=True,
        selected_backend="codex",
        estimated_complexity="medium",
        requires_repository_write=True,
        reasons=["Codex selected"],
        safe_to_continue=True,
    )
    assert registry.resolve(dec_ext) is fake_codex

    dec_int = CodingBackendDecision(
        decision_id="dec_int",
        coding_required=True,
        selected_backend="internal",
        estimated_complexity="low",
        requires_repository_write=True,
        reasons=["Internal engine selected"],
        safe_to_continue=True,
    )
    assert registry.resolve(dec_int) is fake_internal


def test_internal_engine_model_tool_calling_loop_and_events(tmp_path: Path) -> None:
    """Internal coding engine executes model tool calls, sets system instruction, and emits events."""
    from langchain_core.messages import AIMessage
    from mana_agent.coding.live_events import coding_event_scope
    from mana_agent.coding.prompts import INTERNAL_CODING_SYSTEM_INSTRUCTION

    events: list[AgentEvent] = []

    class _MockChatModel:
        def __init__(self) -> None:
            self.turn = 0
            self.received_messages: list[Any] = []

        def bind_tools(self, tools: list[dict[str, Any]]) -> _MockChatModel:
            return self

        def invoke(self, messages: list[Any]) -> AIMessage:
            self.received_messages = list(messages)
            self.turn += 1
            if self.turn == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "write_file",
                        "args": {"path": "output.py", "content": "x = 42\n"},
                        "id": "call_write_1",
                    }],
                )
            return AIMessage(
                content="### Result\nSuccessfully created output.py\n\n### Changes\noutput.py\n\n### Verification\nNone\n\n### Notes\nDone."
            )

    mock_llm = _MockChatModel()
    engine = InternalCodingEngine(
        model="gpt-4o",
        llm_client=mock_llm,
    )

    assert "# Mana-Agent — Internal Coding Engine" in engine.system_instruction
    assert "## System Instruction — Autonomous Software Engineering Agent" in engine.system_instruction

    task = CodingTask(
        task_id="task_tool_loop_1",
        goal="Create output.py with x = 42",
    )
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )

    with coding_event_scope(lambda ev: events.append(ev)):
        result = asyncio.run(engine.execute(task, workspace))

    assert result.status == "completed"
    assert result.backend == "internal"
    assert "output.py" in result.changed_files
    assert (tmp_path / "output.py").read_text(encoding="utf-8") == "x = 42\n"

    # Verify tool events were published
    event_types = [e.event_type for e in events]
    assert "tool.call.started" in event_types
    assert "tool.call.completed" in event_types
    assert "turn.completed" in event_types

    # Verify first message was the exact system instruction
    assert mock_llm.received_messages[0].content == INTERNAL_CODING_SYSTEM_INSTRUCTION


def test_tool_definitions_conform_to_openai_name_pattern(tmp_path: Path) -> None:
    """Verify tool names strictly conform to OpenAI's ^[a-zA-Z0-9_-]+$ pattern and git tools execute."""
    import re
    from mana_agent.coding.internal_engine import InternalCodingEngine
    from mana_agent.coding.programmatic import (
        ToolExecutionDispatcher,
        build_programmatic_tool_definitions,
    )

    pattern = re.compile(r"^[a-zA-Z0-9_-]+$")

    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )
    dispatcher = ToolExecutionDispatcher(workspace)
    engine = InternalCodingEngine(model="gpt-4o")

    # 1. Check internal engine tool definitions
    internal_tool_defs = engine._build_tool_definitions(dispatcher)
    assert len(internal_tool_defs) > 0
    for td in internal_tool_defs:
        name = td["function"]["name"]
        assert pattern.match(name), f"Internal tool name {name!r} violates OpenAI pattern ^[a-zA-Z0-9_-]+$"

    # 2. Check programmatic tool definitions
    prog_tool_defs = build_programmatic_tool_definitions()
    assert len(prog_tool_defs) > 1
    for td in prog_tool_defs:
        if td.get("type") == "programmatic_tool_calling":
            continue
        name = td.get("name")
        assert name is not None
        assert pattern.match(name), f"Programmatic tool name {name!r} violates OpenAI pattern ^[a-zA-Z0-9_-]+$"

    # 3. Verify git_status and git.status dispatch
    res_git_status = dispatcher.dispatch("git_status", {})
    assert "error" not in res_git_status or res_git_status.get("error_code") != "UNKNOWN_TOOL"

    # 4. Verify read_skill dispatch
    res_skill = dispatcher.dispatch("read_skill", {"skill_name": "nonexistent"})
    assert res_skill.get("error_code") != "UNKNOWN_TOOL"

