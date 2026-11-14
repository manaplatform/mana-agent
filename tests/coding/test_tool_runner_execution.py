"""Explicitly user-run regression coverage for the deterministic execution gate."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest
from mana_agent.coding.models import WorkspaceContext
from mana_agent.coding.programmatic import ToolExecutionDispatcher
from mana_agent.multi_agent.agents.tool_runner_agent import (
    ExecutionAuthorityError, ToolRunnerAgent, bind_tool_authority,
)
from mana_agent.multi_agent.core.types import AgentRole
from mana_agent.multi_agent.registry.agent_registry import AgentRegistry
from mana_agent.multi_agent.tools.operation_policy import OperationPolicy
from mana_agent.utils.process_stream import run_streaming_process


def make_runner(root, *, effect="read_only", sandbox="workspaceWrite", cancel=lambda: False):
    registry = AgentRegistry()
    owner = registry.find_by_role(AgentRole.CODING)
    authority = bind_tool_authority(registry=registry, actor_id=owner.agent_id, task_id="task-test",
        workspace=WorkspaceContext(repository_path=root, worktree_path=root, allow_in_place_write=True, sandbox=sandbox),
        policies={"selected": OperationPolicy("repository_read" if effect == "read_only" else "repository_write", effect)},
        schemas={"selected": {"type": "object"}}, cancel=cancel)
    return ToolRunnerAgent(authority)


def invoke(runner, arguments, callback, identity="call-one"):
    request = runner.request("selected", arguments, call_id=identity, decision_id="decision-one")
    return runner.execute(request, callback)


def test_unbound_dispatcher_cannot_execute(tmp_path):
    dispatcher = ToolExecutionDispatcher(repo_root=tmp_path)
    result = dispatcher.dispatch("write_file", {"path": "file.py", "content": "x"})
    assert result["error_code"] == "EXECUTION_AUTHORITY_REQUIRED"
    assert not (tmp_path / "file.py").exists()


@pytest.mark.parametrize("role", [AgentRole.MAIN, AgentRole.HEAD_DECISION, AgentRole.VERIFIER])
def test_wrong_level_cannot_bind_repository_tools(tmp_path, role):
    registry = AgentRegistry()
    with pytest.raises(ExecutionAuthorityError):
        bind_tool_authority(registry=registry, actor_id=registry.find_by_role(role).agent_id,
            task_id="task", workspace=WorkspaceContext(repository_path=tmp_path, worktree_path=tmp_path, allow_in_place_write=True),
            policies={"read": OperationPolicy("repository_read", "read_only")}, schemas={})


@pytest.mark.parametrize("field,value", [
    ("task_id", "foreign-task"), ("agent_id", "agent_tool_worker_forged"),
    ("requested_by_agent_id", "main"), ("authorization_scope", "approved"),
    ("attempt_id", "retired"), ("attempt_generation", 99), ("workspace_root", "/outside"),
    ("root_task_id", "foreign-root"), ("execution_id", "foreign-execution"), ("cancellation_id", "foreign-cancel"),
])
def test_forged_requests_fail_before_calling_implementation(tmp_path, field, value):
    runner = make_runner(tmp_path)
    request = runner.request("selected", {}, call_id="one", decision_id="decision")
    called = []
    with pytest.raises(ExecutionAuthorityError):
        runner.execute(request.model_copy(update={field: value}), lambda args: called.append(args))
    assert not called


def test_duplicate_delivery_is_single_execution_even_when_concurrent(tmp_path):
    runner = make_runner(tmp_path)
    calls = []
    def execute(args):
        calls.append(args)
        return {"ok": True, "content": "observed"}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: invoke(runner, {}, execute), range(4)))
    assert len(calls) == 1
    assert all(result.result["content"] == "observed" for result in results)
    with pytest.raises(ExecutionAuthorityError):
        invoke(runner, {"different": True}, execute)


def test_schema_and_readonly_scope_are_enforced(tmp_path):
    runner = make_runner(tmp_path, effect="mutation", sandbox="readOnly")
    with pytest.raises(ExecutionAuthorityError):
        invoke(runner, {}, lambda args: {"ok": True})
    dispatcher = ToolExecutionDispatcher.for_coding_task(repo_root=tmp_path, task_id="typed")
    result = dispatcher.dispatch("read_file", {"path": 12})
    assert result["error_code"] == "EXECUTION_AUTHORITY_REJECTED"


def test_mutation_diff_uses_immediate_dirty_baseline(tmp_path):
    target = tmp_path / "dirty.py"
    target.write_text("preexisting user work\nold\n")
    runner = make_runner(tmp_path, effect="mutation")
    def edit(args):
        target.write_text("preexisting user work\nnew\n")
        return {"ok": True}
    result = invoke(runner, {"path": "dirty.py"}, edit)
    mutation = result.mutations[0]
    assert "-old" in mutation.diff and "+new" in mutation.diff
    assert "+preexisting user work" not in mutation.diff
    assert mutation.added_lines == mutation.removed_lines == 1


def test_partial_failure_preserves_real_changes_and_rename_evidence(tmp_path):
    (tmp_path / "old.txt").write_text("content\n")
    runner = make_runner(tmp_path, effect="mutation")
    def partial(args):
        (tmp_path / "old.txt").rename(tmp_path / "new.txt")
        return {"ok": False, "error": "second operation failed"}
    result = invoke(runner, {"moves": [{"source": "old.txt", "destination": "new.txt"}]}, partial)
    assert result.status == "failed"
    assert len(result.mutations) == 1
    assert result.mutations[0].operation == "move"
    assert result.mutations[0].old_path == "old.txt"
    assert result.mutations[0].status == "partial"


def test_failed_noop_and_rollback_do_not_fabricate_changes(tmp_path):
    path = tmp_path / "same.txt"
    path.write_text("original")
    runner = make_runner(tmp_path, effect="mutation")
    def rolled_back(args):
        path.write_text("temporary")
        path.write_text("original")
        return {"ok": False, "rollback_performed": True}
    result = invoke(runner, {"path": "same.txt"}, rolled_back)
    assert result.status == "failed" and not result.mutations


def test_command_output_streams_before_exit_with_separate_channels(tmp_path, monkeypatch):
    import mana_agent.coding.live_events as live
    events = []
    finished = threading.Event()
    monkeypatch.setattr(live, "publish_coding_event", lambda event: events.append((event, finished.is_set())))
    script = tmp_path / "streams.py"
    script.write_text("import sys,time\nprint('out',flush=True)\nprint('err',file=sys.stderr,flush=True)\ntime.sleep(.1)\n")
    runner = make_runner(tmp_path, effect="command")
    def execute(args):
        completed = run_streaming_process([sys.executable, str(script)], cwd=tmp_path, timeout=5)
        finished.set()
        return {"ok": completed.returncode == 0, "returncode": completed.returncode,
                "stdout": completed.stdout, "stderr": completed.stderr}
    result = invoke(runner, {"argv": [sys.executable, str(script)]}, execute)
    assert result.exit_code == 0
    assert any(event.event_type == "tool.stdout" and not done for event, done in events)
    assert any(event.event_type == "tool.stderr" and not done for event, done in events)
    commands = [event for event, _ in events if event.event_type == "command.started"]
    assert commands[0].payload["display_argv"] == [sys.executable, str(script)]
    assert commands[0].payload["cwd"] == str(tmp_path)


def test_split_secrets_are_redacted_before_trace_publication(tmp_path, monkeypatch):
    import mana_agent.coding.live_events as live
    events = []
    monkeypatch.setenv("EXAMPLE_API_KEY", "private-split-credential")
    monkeypatch.setattr(live, "publish_coding_event", lambda event: events.append(event.model_dump(mode="json")))
    script = tmp_path / "secret.py"
    script.write_text("import sys,time\nsys.stdout.write('private-split-');sys.stdout.flush();time.sleep(.02)\nprint('credential')\n")
    runner = make_runner(tmp_path, effect="command")
    invoke(runner, {}, lambda args: {"ok": True, "stdout": run_streaming_process([sys.executable, str(script)], cwd=tmp_path).stdout})
    serialized = json.dumps(events, default=str)
    assert "private-split-credential" not in serialized
    assert "private-split-" not in serialized


def test_timeout_and_cancellation_keep_partial_output(tmp_path):
    script = tmp_path / "long.py"
    script.write_text("import time\nprint('partial',flush=True)\ntime.sleep(10)\n")
    runner = make_runner(tmp_path, effect="command")
    result = invoke(runner, {}, lambda args: {"ok": True, "stdout": run_streaming_process(
        [sys.executable, str(script)], cwd=tmp_path, timeout=.1).stdout})
    assert result.status == "timed_out"
    assert "partial" in result.result["stdout"]
    cancelled = make_runner(tmp_path, cancel=lambda: True)
    with pytest.raises(InterruptedError):
        invoke(cancelled, {}, lambda args: {"ok": True})


def test_unknown_custom_tools_have_no_implicit_capability(tmp_path):
    calls = []
    dispatcher = ToolExecutionDispatcher.for_coding_task(repo_root=tmp_path, task_id="custom",
        extra_tools={"unknown": lambda **args: calls.append(args)})
    assert dispatcher.dispatch("unknown", {})["error_code"] == "UNKNOWN_TOOL"
    assert not calls


def test_nonzero_exit_without_ok_is_not_success(tmp_path):
    result = invoke(make_runner(tmp_path), {}, lambda args: {"returncode": 9, "stdout": "failed"})
    assert result.status == "failed"
    assert result.exit_code == 9


def test_unknown_binary_and_overflow_have_honest_evidence(tmp_path):
    path = tmp_path / "binary.dat"
    path.write_bytes(b"\xff\xfeold")
    runner = make_runner(tmp_path, effect="mutation")
    runner.capture_limit = 32
    result = invoke(runner, {"path": "binary.dat"}, lambda args: (path.write_bytes(b"\xff\xfenew") and {"ok": True, "stdout": "x" * 100}))
    assert result.mutations[0].binary is True
    assert result.mutations[0].diff == ""
    assert result.omitted_bytes > 0
    assert result.trace_complete is False


def test_registered_actor_revocation_and_path_confinement(tmp_path):
    from mana_agent.multi_agent.core.types import AgentState
    runner = make_runner(tmp_path)
    with pytest.raises(ExecutionAuthorityError):
        invoke(runner, {"repo_path": "/outside"}, lambda args: {"ok": True})
    with pytest.raises(ExecutionAuthorityError):
        invoke(runner, {"path": ".env"}, lambda args: {"ok": True})
    runner.authority.registry.agents[runner.authority.actor_id].state = AgentState.DONE
    with pytest.raises(ExecutionAuthorityError):
        invoke(runner, {}, lambda args: {"ok": True})


def test_schema_reference_validation_is_not_skipped():
    from mana_agent.multi_agent.tools.operation_policy import validate_arguments
    schema = {"type": "object", "properties": {"item": {"$ref": "#/$defs/Item"}},
        "$defs": {"Item": {"type": "object", "required": ["count"], "properties": {"count": {"type": "integer", "minimum": 1}}}}}
    validate_arguments({"item": {"count": 2}}, schema)
    with pytest.raises(ValueError):
        validate_arguments({"item": {"count": "two"}}, schema)


def test_worker_single_operation_has_no_model_calls_and_deduplicates(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from mana_agent.multi_agent.runtime import tool_worker_process as worker
    calls = []
    class RegisteredTool:
        name = "selected"
        input_model = None
        metadata = {"execution_capability": "repository_read", "execution_effect": "read_only", "repository_operation": True}
        def call(self, arguments, **kwargs):
            calls.append(arguments)
            return {"ok": True, "content": "observed"}
    server = worker._ToolWorkerServer()
    server._repo_root = str(tmp_path)
    server._ask_agent = SimpleNamespace(_build_tools=lambda **kwargs: ([RegisteredTool()], [], [], []))
    replies = []
    monkeypatch.setattr(server, "_emit", replies.append)
    runner = make_runner(tmp_path)
    request = runner.request("selected", {}, call_id="one", decision_id="decision")
    for _ in range(2):
        server._handle_execute_operation(worker.WorkerEnvelope(type="execute_operation", request_id="ipc", payload=request.model_dump(mode="json")))
    assert calls == [{}]
    assert len(replies) == 2 and all(reply.type == "ok" for reply in replies)


def test_legacy_redis_payload_does_not_recreate_authority(monkeypatch, tmp_path):
    from mana_agent.multi_agent.runtime import redis_tool_tasks as redis
    from mana_agent.multi_agent.runtime.tool_worker_process import ToolWorkerClient, ToolRunRequest
    monkeypatch.setattr(redis, "run_tool_request_once", lambda **kwargs: pytest.fail("Unbound serialized requests cannot start planning"))
    client = ToolWorkerClient(api_key="test", model="fake", session_id="session", repo_root=tmp_path, project_root=tmp_path)
    result = redis.run_tool_request_task(client.init_payload_dict(), ToolRunRequest(question="read", run_id="old-task").model_dump())
    assert result["ok"] is False
    assert "current registered execution authority" in result["error_message"]


def test_document_dispatch_uses_the_existing_registered_adapter(tmp_path):
    (tmp_path / "sample.csv").write_text("key,value\nexample,1\n")
    dispatcher = ToolExecutionDispatcher.for_coding_task(repo_root=tmp_path, task_id="documents")
    result = dispatcher.dispatch("document_detect", {"path": "sample.csv"})
    assert result["ok"] is True
    assert result["execution"]["tool_name"] == "document_detect"
