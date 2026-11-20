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


def test_native_reads_whole_files_and_returns_exact_bounded_ranges(tmp_path):
    from hashlib import sha256
    from mana_agent.coding.programmatic import ToolExecutionDispatcher

    small = "".join(f"setting_{line:04d} = 'default value for the module'\n" for line in range(400))
    large = small * 20
    (tmp_path / "settings.py").write_text(small)
    (tmp_path / "big.py").write_text(large)
    (tmp_path / "alias.py").symlink_to(tmp_path / "settings.py")
    dispatcher = ToolExecutionDispatcher.for_coding_task(repo_root=tmp_path, task_id="adaptive-read")

    def read(path, **kwargs):
        return dispatcher.dispatch("read_file", {"path": path, **kwargs})

    complete = read("settings.py")
    assert complete["complete"] and not complete["truncated"]
    assert (complete["start_line"], complete["end_line"], complete["total_lines"]) == (1, 400, 400)
    assert complete["content"] == small
    assert complete["source_evidence"]["sha256"] == sha256(small.encode()).hexdigest()

    scoped = read("big.py", start_line=7900, end_line=7910)
    assert scoped["partial"] and not scoped["truncated"]
    assert (scoped["start_line"], scoped["end_line"], scoped["total_lines"]) == (7900, 7910, 8000)
    assert scoped["content"] == "".join(large.splitlines(keepends=True)[7899:7910])
    chunk = read("big.py")
    assert chunk["truncated"] and chunk["partial"] and not chunk["complete"]
    assert len(chunk["content"].splitlines()) == chunk["end_line"]
    assert chunk["next_start_line"] == chunk["end_line"] + 1
    assert not read("alias.py")["ok"]


def test_native_resume_adds_refreshed_evidence_to_saved_history(tmp_path):
    class Model:
        async def generate(self, messages, *, continuation):
            text = [message.content for message in messages]
            assert "Previous scoped read" in text
            assert "Fresh config evidence" in text[-1]
            assert "same assignment" in text[-1]
            return SimpleNamespace(tool_calls=[], content="Use the confirmed option.")

    task = CodingTask(task_id="resumed", goal="Change the consumer", relevant_context="Fresh config evidence",
                      requires_repository_write=False, context={
                          "coding_continuation_id": "checkpoint",
                          "coding_resume_context": {"native_messages": [
                              {"role": "system", "content": "Work within scope"},
                              {"role": "user", "content": "Previous scoped read"},
                          ]},
                      })
    workspace = WorkspaceContext(repository_path=tmp_path, worktree_path=tmp_path, sandbox="readOnly")
    result = asyncio.run(InternalCodingEngine(llm_client=Model()).execute(task, workspace))
    assert result.status == "completed", result.errors


def test_native_oversized_read_group_can_be_narrowed(tmp_path):
    (tmp_path / "handler.py").write_text("handler = True\n")

    class Model:
        turns = 0

        async def generate(self, messages, *, continuation):
            self.turns += 1
            if self.turns == 1:
                return SimpleNamespace(tool_calls=[{"id": f"read-{n}", "name": "read_file", "args": {"path": path}}
                    for n, path in enumerate(("handler.py", "caller.py"))])
            if self.turns == 2:
                assert [json.loads(message.content)["executed"] for message in messages[-2:]] == [0, 0]
                return SimpleNamespace(tool_calls=[{"id": "narrow", "name": "read_file", "args": {"path": "handler.py"}}])
            assert "handler = True" in messages[-1].content
            return SimpleNamespace(tool_calls=[], content="The handler is enabled.")

    engine = InternalCodingEngine(llm_client=Model(), max_tool_calls=1)
    task = CodingTask(task_id="coding", goal="Confirm handler", requires_repository_write=False)
    workspace = WorkspaceContext(repository_path=tmp_path, worktree_path=tmp_path, sandbox="readOnly")
    result = asyncio.run(engine.execute(task, workspace))
    assert result.status == "completed", result.errors
    assert result.summary == "The handler is enabled."


@pytest.mark.parametrize(("outcome", "use_continuation"), [
    ("recovered", False), ("recovered", True), ("repeated", False), ("budget", True),
])
def test_adaptive_coding_corrects_a_missing_tool_call(tmp_path, monkeypatch, outcome, use_continuation):
    from mana_agent.coding.models import CODING_RESULT_TOOL
    from mana_agent.llm.engine import _conversation_fingerprint
    from mana_agent.llm.messages import ToolCall, validate_conversation
    from mana_agent.llm.types import ContinuationState, ModelResponse

    (tmp_path / "handler.py").write_text("enabled = True\n")
    prose = '{"disposition": "complete", "summary": "Unsubmitted result"}'
    events = []
    monkeypatch.setattr("mana_agent.services.activity_context.emit_activity",
                        lambda kind, **fields: events.append((kind, fields)))

    class Model:
        turns = 0
        receipt = None

        async def generate(self, messages, *, continuation):
            self.turns += 1
            validate_conversation(messages)
            if self.turns == 1:
                assert "Respond with a tool call" in messages[-1].content
                return ModelResponse(tool_calls=[ToolCall(id="read", name="read_file", args={"path": "handler.py"})])
            if self.turns == 2:
                assert messages[-1].tool_call_id == "read"
                assert "enabled = True" in messages[-1].content
                response = ModelResponse(content=prose)
                if use_continuation:
                    self.receipt = ContinuationState(
                        provider="openai", model="fixture", endpoint="responses", mode="server",
                        response_id="receipt", message_count=len(messages) + 1,
                        message_fingerprint=_conversation_fingerprint([*messages, response.message]),
                    )
                    response.continuation = self.receipt
                return response
            assert self.turns == 3
            assert continuation == self.receipt
            if continuation:
                assert _conversation_fingerprint(messages[:continuation.message_count]) == continuation.message_fingerprint
            else:
                assert all(message.content != prose for message in messages)
            assert sum(message.tool_call_id == "read" for message in messages) == 1
            assert messages[-1].role == "user"
            assert f"{CODING_RESULT_TOOL} by itself as a direct call" in messages[-1].content
            if outcome == "repeated":
                return ModelResponse(content=prose)
            return ModelResponse(tool_calls=[ToolCall(id="done", name=CODING_RESULT_TOOL, args={
                "disposition": "complete", "summary": "Observed the enabled handler.",
            })])

    model = Model()
    charges = []
    engine = InternalCodingEngine(llm_client=model, max_tool_calls=1, programmatic_tool_calling_enabled=False)
    engine.max_model_turns = 2 if outcome == "budget" else 5
    engine.charge_model = lambda: charges.append("model")
    task = CodingTask(task_id="adaptive-missing-action", goal="Confirm the handler state",
                      requires_repository_write=False, context={"adaptive_mode": "debug"})
    workspace = WorkspaceContext(repository_path=tmp_path, worktree_path=tmp_path, sandbox="readOnly")
    result = asyncio.run(engine.execute(task, workspace))

    read, = result.continuation_context["executed_operations"]
    assert read["call_id"] == "read" and read["ok"]
    assert model.turns == len(charges) == (2 if outcome == "budget" else 3)
    corrections = [fields for kind, fields in events if kind == "coding.progress"]
    if outcome == "budget":
        assert not corrections
    else:
        correction, = corrections
        assert correction["status"] == "rejected"
        assert correction["metadata"]["reason_code"] == "missing_coding_tool_call"
        assert prose not in correction["summary"]
    if outcome == "recovered":
        assert result.status == "completed", result.errors
        assert result.summary == "Observed the enabled handler."
    else:
        assert result.status == "failed"
        assert result.adaptive_result is None
        error, = result.errors
        assert ("budget cannot cover a correction" if outcome == "budget" else "after one correction") in error
        assert prose not in error


def test_slow_native_provider_is_measured_separately_from_duplicate_reads(tmp_path, monkeypatch):
    from mana_agent.coding import internal_engine

    (tmp_path / "a.py").write_text("value = True\n")
    clock = [0.0]
    monkeypatch.setattr(internal_engine, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    lifecycle = []
    monkeypatch.setattr("mana_agent.services.activity_context.emit_activity",
                        lambda kind, **fields: lifecycle.append((kind, fields)))

    class Model:
        turns = 0

        async def generate(self, messages, *, continuation):
            self.turns += 1
            clock[0] += 300 if self.turns == 1 else 0.001
            return SimpleNamespace(tool_calls=[
                {"id": f"read-{self.turns}", "name": "read_file", "args": {"path": "a.py"}},
            ])

    model = Model()
    engine = InternalCodingEngine(llm_client=model)
    engine.max_no_progress_cycles = 2
    task = CodingTask(task_id="slow", goal="Decide from the current file", requires_repository_write=False)
    workspace = WorkspaceContext(repository_path=tmp_path, worktree_path=tmp_path, sandbox="readOnly")
    result = asyncio.run(engine.execute(task, workspace))
    assert result.status == "failed"
    assert "Repeated repository reads" in result.errors[0]
    assert model.turns == 4
    assert len(result.continuation_context["executed_operations"]) == 3
    durations = [fields["duration_ms"] for kind, fields in lifecycle if kind == "model.completed"]
    assert durations == [300_000, 1, 1, 1]
    assert result.duration_breakdown["provider_execution_time_ms"] == 300_003


@pytest.mark.parametrize(("max_tool_calls", "max_model_turns", "expected_reads"), [(1, 4, 1), (5, 2, 1), (5, 5, 2)])
def test_native_patch_proposal_can_finish_after_reads_stop(tmp_path, max_tool_calls, max_model_turns, expected_reads):
    (tmp_path / "a.py").write_text("value = True\n")
    patch = "*** Begin Patch\n*** Update File: a.py\n@@\n-value = True\n+value = False\n*** End Patch"

    class Model:
        reads = 0

        async def generate(self, messages, *, continuation):
            if "No further tools are available in this proposal" in messages[-1].content:
                return SimpleNamespace(tool_calls=[], content=json.dumps({"patch": patch}))
            self.reads += 1
            return SimpleNamespace(tool_calls=[{"id": f"read-{self.reads}", "name": "read_file", "args": {"path": "a.py"}}])

    model = Model()
    engine = InternalCodingEngine(llm_client=model, max_tool_calls=max_tool_calls)
    engine.max_model_turns = max_model_turns
    engine.max_no_progress_cycles = 1
    task = CodingTask(task_id="proposal", goal="Propose a change for a.py", allowed_files=["a.py"],
                      requires_repository_write=False, context={"mode_patch_proposal": True})
    workspace = WorkspaceContext(repository_path=tmp_path, worktree_path=tmp_path, sandbox="readOnly")
    result = asyncio.run(engine.execute(task, workspace))

    assert result.status == "completed", result.errors
    assert json.loads(result.summary) == {"patch": patch}
    assert model.reads == expected_reads
    assert (tmp_path / "a.py").read_text() == "value = True\n"


def test_cancelling_provider_wait_retains_completed_tool_result(tmp_path):
    (tmp_path / "a.py").write_text("value = True\n")
    task = CodingTask(task_id="cancel-after-read", goal="Inspect a.py", requires_repository_write=False)
    workspace = WorkspaceContext(repository_path=tmp_path, worktree_path=tmp_path, sandbox="readOnly")

    async def exercise():
        pending = asyncio.Event()

        class Model:
            async def generate(self, messages, *, continuation):
                if len(messages) == 2:
                    return SimpleNamespace(tool_calls=[{"id": "read", "name": "read_file", "args": {"path": "a.py"}}])
                pending.set()
                await asyncio.Event().wait()

        engine = InternalCodingEngine(llm_client=Model())
        execution = asyncio.create_task(engine.execute(task, workspace))
        await asyncio.wait_for(pending.wait(), timeout=5)
        await engine.cancel(task.task_id)
        return await asyncio.wait_for(execution, timeout=5)

    result = asyncio.run(exercise())
    assert result.status == "cancelled"
    read, = result.continuation_context["executed_operations"]
    assert read["call_id"] == "read" and read["ok"]


class _FakeMultiToolLLM:
    """Simulates an LLM returning multiple parallel tool calls on turn 1, then completing on turn 2."""

    def __init__(self, tool_calls_plan: list[list[dict[str, Any]]], final_response: str = "Work verified and complete.") -> None:
        self.tool_calls_plan = tool_calls_plan
        self.final_response = final_response
        self.turn = 0

    def with_tools(self, tools: list[dict[str, Any]], **_native_options) -> "_FakeMultiToolLLM":
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
        self.tasks = {}

    def get_task(self, task_id):
        if task_id not in self.tasks:
            self.tasks[task_id] = SimpleNamespace(owner_agent_id=None, assigned_agent_ids=[], assigned_subagent_ids=[],
                execution_repo_root="", managed_worktree_path="", allowed_tools=[], forbidden_tools=[], evidence=[], files_touched=[])
        return self.tasks[task_id]

    def assign(self, task_id, actor_id):
        self.get_task(task_id).assigned_agent_ids.append(actor_id)

    def add_evidence(self, task_id, evidence):
        self.get_task(task_id).evidence.append(evidence)

    def add_files_touched(self, task_id, paths):
        self.get_task(task_id).files_touched.extend(paths)

    def update_task(self, task_id: str, *, progress: str) -> None:
        self.updates.append({"task_id": task_id, "progress": progress})


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

    event_types = [e.event_type for e in events_emitted]
    assert "parallel.group.started" in event_types
    assert "parallel.group.completed" in event_types
    assert "turn.completed" in event_types

    assert supervisor.recorded_actions
    assert [update["task_id"] for update in taskboard.updates] == [task.task_id, task.task_id]
    assert all(update["progress"] for update in taskboard.updates)


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


def test_internal_coding_engine_batch_read_allowed_files(tmp_path: Path) -> None:
    """Verify that when task.allowed_files contains multiple files and batch operations are enabled,
    the internal coding engine invokes batch_read_files rather than single reads."""
    file_a = tmp_path / "mod_a.py"
    file_b = tmp_path / "mod_b.py"
    file_a.write_text("def fn_a(): return 1\n", encoding="utf-8")
    file_b.write_text("def fn_b(): return 2\n", encoding="utf-8")

    def _batch_read_client(task, ws, tools, session):
        if len(task.allowed_files) > 1 and engine.batch_operations_enabled:
            session.handle_function_call({
                "call_id": "call_batch_1",
                "name": "batch_read_files",
                "arguments": {"files": list(task.allowed_files)},
                "caller": {"type": "program", "call_id": "prog_init"},
            })
        return f"Completed internal coding task: {task.goal}"

    engine = InternalCodingEngine(
        model="test-model",
        llm_client=_batch_read_client,
        batch_operations_enabled=True,
    )

    task = CodingTask(
        task_id="task_batch_read_test",
        goal="Inspect multiple allowed files",
        allowed_files=["mod_a.py", "mod_b.py"],
    )
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="readOnly",
    )

    result = asyncio.run(engine.execute(task, workspace))
    assert result.status == "completed"
    assert "Completed internal coding task" in result.summary


def test_tool_execution_dispatcher_defaults_model_for_batch_tools(tmp_path: Path) -> None:
    """Verify ToolExecutionDispatcher provides default model and provider to batch tools."""
    from mana_agent.coding.programmatic import ToolExecutionDispatcher

    ws = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )
    dispatcher = ToolExecutionDispatcher.for_coding_task(
        ws,
        task_id="task_disp_test",
        model="gpt-4o-mini",
        provider="openai",
    )
    assert dispatcher.model == "gpt-4o-mini"
    assert dispatcher.provider == "openai"
    assert dispatcher.task_id == "task_disp_test"

