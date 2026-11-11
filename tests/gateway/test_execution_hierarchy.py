"""Hierarchy, shared projection, and recovery contracts; execution is user-owned."""
from __future__ import annotations

import json
import pytest
from mana_agent.chat.activity import operation_key, project_activity
from mana_agent.chat.operations import merge_operation
from mana_agent.coding.models import WorkspaceContext
from mana_agent.coding.programmatic import ToolExecutionDispatcher
from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
from mana_agent.execution_supervisor.models import SideEffectClassification
from mana_agent.multi_agent.core.types import AgentRole
from mana_agent.multi_agent.registry.agent_registry import AgentRegistry
from mana_agent.services.activity_context import ActivityContext, activity_scope


def test_registered_coding_role_has_main_supervision():
    registry = AgentRegistry()
    coding = registry.find_by_role(AgentRole.CODING)
    assert registry.agents[coding.parent_agent_id].role == AgentRole.MAIN
    assert AgentRole.TOOL_RUNNER.value == "tool_worker"


def test_operation_projection_preserves_safe_execution_fields_and_drops_raw_arguments():
    event = project_activity({"type": "tool.started", "task_id": "task", "operation_id": "op",
        "operation_attempt_id": "attempt-op", "status": "running", "metadata": {
            "cwd": "/workspace", "display_argv": ["python", "example.py"], "exit_code": 0,
            "arguments": {"password": "private"}, "environment": {"API_KEY": "secret"},
            "diff": "-old\n+new", "provenance": "native_tool_runner"}})
    assert event["metadata"]["cwd"] == "/workspace"
    assert event["metadata"]["exit_code"] == 0
    assert "private" not in json.dumps(event)
    assert "environment" not in event["metadata"]


def test_stream_revisions_preserve_inputs_and_repeated_chunks():
    initial = {"type": "tool.started", "operation_id": "same", "status": "running",
               "metadata": {"command": "python streams.py", "cwd": "/workspace"}}
    first = merge_operation(initial, {"type": "tool.stdout", "output_preview": "same\n", "metadata": {"stream_channel": "stdout", "stream_offset": 0, "content_length": 5}})
    second = merge_operation(first, {"type": "tool.stdout", "output_preview": "same\n", "metadata": {"stream_channel": "stdout", "stream_offset": 5, "content_length": 5}})
    duplicate = merge_operation(second, {"type": "tool.stdout", "output_preview": "same\n", "metadata": {"stream_channel": "stdout", "stream_offset": 5, "content_length": 5}})
    assert duplicate["metadata"]["streams"]["stdout"] == "same\nsame\n"
    assert duplicate["metadata"]["command"] == "python streams.py"
    assert duplicate["metadata"]["cwd"] == "/workspace"
    assert operation_key({"operation_id": "same", "operation_attempt_id": "one"}) != operation_key({"operation_id": "same", "operation_attempt_id": "two"})


def test_stale_attempt_cannot_execute_after_cancellation(tmp_path):
    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig(root=tmp_path / "execution", startup_recovery=False))
    task = supervisor.create_task(routing_decision_id="decision", workspace_path=tmp_path, side_effect_classification=SideEffectClassification.READ_ONLY)
    supervisor.queue(task.task_id)
    leased, token = supervisor.acquire_lease(task.task_id, owner="coding")
    supervisor.start(task.task_id, attempt_id=leased.attempt_id, lease_token=token)
    context = ActivityContext(conversation_id="session", execution_id="execution", task_id=task.task_id,
        root_task_id=task.task_id, attempt_id=leased.attempt_id, attempt_generation=leased.attempt_generation,
        supervisor=supervisor, lease_token=token)
    with activity_scope(context):
        dispatcher = ToolExecutionDispatcher.for_coding_task(repo_root=tmp_path, task_id=task.task_id)
    supervisor.cancel(task.task_id, reason="user cancelled")
    with pytest.raises(Exception):
        dispatcher.dispatch("read_file", {"path": "missing.py"})


def test_domain_execution_retains_specialist_lane_and_exact_result(tmp_path):
    from types import SimpleNamespace
    from mana_agent.gateway.chat_gateway import AgentChatGateway
    gateway = AgentChatGateway.__new__(AgentChatGateway)
    gateway.root = tmp_path
    registry = AgentRegistry()
    calls = []
    context = ActivityContext(conversation_id="", execution_id="turn", task_id="task",
        root_task_id="task", execution_registry=registry, authorize_tool=lambda name: calls.append(name))
    expected = SimpleNamespace(status="submitted")
    with activity_scope(context):
        result = gateway._execute_domain_operation(tool_name="get_media_generation_status", arguments={},
            read_only=True, invoke=lambda: expected)
    assert result is expected
    assert calls == ["get_media_generation_status"]
    worker = next(node for node in registry.agents.values() if node.role == AgentRole.TOOL_WORKER)
    assert registry.agents[worker.parent_agent_id].role == AgentRole.TOOL
    with pytest.raises(PermissionError):
        gateway._execute_domain_operation(tool_name="get_media_generation_status", arguments={}, invoke=lambda: expected)


def test_receipt_preparation_deduplicates_before_start(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from mana_agent.execution_supervisor.errors import RetrySafetyError
    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig(root=tmp_path / "execution", startup_recovery=False))
    task = supervisor.create_task(routing_decision_id="decision", workspace_path=tmp_path,
        side_effect_classification=SideEffectClassification.NON_IDEMPOTENT)
    supervisor.queue(task.task_id)
    leased, token = supervisor.acquire_lease(task.task_id, owner="coding")
    supervisor.start(task.task_id, attempt_id=leased.attempt_id, lease_token=token)
    def prepare(_):
        try:
            return supervisor.prepare_action(task.task_id, attempt_id=leased.attempt_id, lease_token=token,
                tool_name="selected", action_fingerprint="same", classification=SideEffectClassification.NON_IDEMPOTENT)
        except RetrySafetyError:
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(prepare, range(4)))
    assert sum(result is not None for result in results) == 1
    assert len(supervisor.store.actions_for_task(task.task_id)) == 1
