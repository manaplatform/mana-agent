"""Runtime-boundary regressions for the public activity stream (no network)."""
from __future__ import annotations

import asyncio
import json
import multiprocessing
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mana_agent.chat.events import ActivityEvent, UserMessageEvent
from mana_agent.chat.history import ChatHistory
from mana_agent.coding.internal_engine import InternalCodingEngine
from mana_agent.coding.live_events import publish_coding_event
from mana_agent.coding.models import AgentEvent, CodingTask
from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
from mana_agent.services.activity_context import ActivityContext, activity_scope
from mana_agent.services.execution_event_hub import ExecutionEventHub, reset_execution_event_hub_for_tests


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    hub = reset_execution_event_hub_for_tests()
    yield hub
    hub.close()


def running(supervisor):
    task = supervisor.create_task(task_id="task_activity", session_id="session_activity", requested_operation="Inspect activity")
    supervisor.queue(task.task_id)
    task, token = supervisor.acquire_lease(task.task_id, owner="activity-worker")
    task = supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)
    return task, token


def test_original_lease_fences_publication_after_cancellation(tmp_path, isolated_state):
    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig(root=tmp_path / "execution", startup_recovery=False))
    task, token = running(supervisor)
    context = ActivityContext(conversation_id=task.session_id, execution_id="turn_activity", repository_id="repo_activity",
        task_id=task.task_id, root_task_id=task.root_task_id, attempt_id=task.attempt_id,
        attempt_generation=task.attempt_generation, supervisor=supervisor, lease_token=token)
    with activity_scope(context):
        first = isolated_state.publish({"type": "tool.started", "event_id": "first", "title": "Inspecting"})
        assert first["attempt_id"] == task.attempt_id
        supervisor.cancel(task.task_id, reason="cancelled")
        stale = isolated_state.publish({"type": "tool.finished", "event_id": "stale", "status": "success"})
    assert stale["visibility"] == "internal"
    public = json.dumps(isolated_state.history(conversation_id=task.session_id))
    assert token not in public
    assert '"event_id": "stale"' not in public


def test_internal_stream_delivers_before_execution_finishes(monkeypatch):
    async def scenario():
        release = asyncio.Event()
        finished = asyncio.Event()
        engine = InternalCodingEngine(model="fixture", provider="openai")
        async def execute(task, workspace):
            publish_coding_event(AgentEvent(event_type="tool.call.started", task_id=task.task_id,
                title="Reading a file", visibility="progress", semantic_kind="tool_execution"))
            await release.wait()
            finished.set()
        monkeypatch.setattr(engine, "execute", execute)
        stream = engine.stream(CodingTask(task_id="task_stream", goal="Inspect a fixture"), MagicMock())
        first = await asyncio.wait_for(anext(stream), timeout=2)
        assert first.event_type == "tool.call.started"
        assert not finished.is_set()
        release.set()
        assert [item async for item in stream] == []
        assert finished.is_set()
    asyncio.run(scenario())


def test_tui_general_hub_event_and_lifecycle_revision_are_visible(tmp_path, isolated_state):
    from mana_agent.tui.app import ManaChatApp
    history = ChatHistory()
    app = ManaChatApp(repo_root=tmp_path, history=history)
    app._gateway_session_id = "session_activity"
    common = {"conversation_id": "session_activity", "execution_id": "turn_activity",
              "event_id": "same-operation", "operation_id": "operation-a"}
    app._handle_hub_session_event({**common, "type": "agent.planning", "status": "running", "title": "Investigating requirements"})
    app._handle_hub_session_event({**common, "type": "agent.planning", "status": "success", "title": "Requirements identified"})
    app._handle_hub_session_event({**common, "type": "agent.planning", "status": "success", "title": "Requirements identified"})
    activities = [item for item in history if isinstance(item, ActivityEvent)]
    assert len(activities) == 1
    assert activities[0].activity["status"] == "success"
    assert activities[0].turn_id == "turn_activity"


def test_steering_message_is_visible_before_admission_returns(tmp_path):
    from mana_agent.tui.app import ManaChatApp
    async def scenario():
        import threading
        entered = threading.Event()
        release = threading.Event()
        gateway = MagicMock()
        def submit(**kwargs):
            entered.set()
            release.wait(timeout=2)
            return {"status": "pending", "sequence": 1}
        gateway.submit_steering.side_effect = submit
        history = ChatHistory()
        app = ManaChatApp(repo_root=tmp_path, history=history, gateway=gateway)
        app._gateway_session_id = "session_activity"
        app._current_frontend_turn_id = "turn_activity"
        pending = asyncio.create_task(app._submit_live_steering("Inspect the public contract first"))
        try:
            await asyncio.to_thread(entered.wait, 1)
            steering = [item for item in history if isinstance(item, UserMessageEvent) and item.is_steering]
            assert len(steering) == 1
            assert steering[0].turn_id == "turn_activity"
            assert steering[0].steering_status == "pending"
            assert not pending.done()
        finally:
            release.set()
            await pending
    asyncio.run(scenario())


def test_dashboard_steering_uses_existing_gateway_without_starting_turn(tmp_path, monkeypatch):
    from mana_agent.ui.streamlit_helpers import submit_dashboard_steering
    gateway = MagicMock()
    gateway.submit_steering.return_value = {"request_id": "steer-one", "status": "pending"}
    monkeypatch.setattr("mana_agent.ui.streamlit_helpers._dashboard_gateway_cache",
        lambda: {(str(tmp_path.resolve()), "session_activity"): gateway})
    result = submit_dashboard_steering("session_activity", root=tmp_path,
        execution_id="turn_activity", request_id="steer-one", text="Use the existing event hub")
    assert result["status"] == "pending"
    gateway.process_turn.assert_not_called()
    gateway.submit_steering.assert_called_once()


def _publish_in_process(state_home: str, prefix: str):
    import os
    os.environ["MANA_HOME"] = state_home
    hub = ExecutionEventHub()
    for index in range(8):
        hub.publish({"type": "task.updated", "event_id": f"{prefix}-{index}"},
                    conversation_id="session_process", repository_id="repo_process", execution_id="turn_process")
    hub.close()


def test_process_publishers_share_one_durable_sequence(tmp_path):
    home = str(tmp_path / "shared-state")
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_publish_in_process, args=(home, prefix)) for prefix in ("a", "b")]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    import os
    from unittest.mock import patch
    with patch.dict(os.environ, {"MANA_HOME": home}):
        reader = ExecutionEventHub()
        rows = reader.history(conversation_id="session_process", repository_id="repo_process", limit=100)
        reader.close()
    assert [row["sequence"] for row in rows] == list(range(1, 17))
    assert len({row["event_id"] for row in rows}) == 16


def test_failed_steering_acknowledgement_never_publishes_applied(tmp_path, isolated_state, monkeypatch):
    from mana_agent.execution.steering import SteeringCoordinator, SteeringSubmission
    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig(root=tmp_path / "execution", startup_recovery=False))
    task, token = running(supervisor)
    coordinator = SteeringCoordinator(supervisor)
    coordinator.submit(SteeringSubmission(execution_id=task.task_id, session_id=task.session_id,
        attempt_id=task.attempt_id, generation=task.attempt_generation, request_id="steer-one",
        kind="guidance", text="Inspect the existing implementation"), authenticated_session_id=task.session_id)
    original = supervisor.store.update_task
    def fail_ack(*args, **kwargs):
        raise OSError("fixture acknowledgement failure")
    monkeypatch.setattr(supervisor.store, "update_task", fail_ack)
    with pytest.raises(OSError):
        coordinator.apply_steering(task.task_id, attempt_id=task.attempt_id, generation=task.attempt_generation,
                                   lease_token=token, request_ids=["steer-one"])
    assert not any(row["type"] == "steering.applied" for row in isolated_state.history(conversation_id=task.session_id))
    monkeypatch.setattr(supervisor.store, "update_task", original)
    coordinator.apply_steering(task.task_id, attempt_id=task.attempt_id, generation=task.attempt_generation,
                               lease_token=token, request_ids=["steer-one"])
    records = supervisor.get_task(task.task_id).steering_inputs
    assert records[0]["status"] == "applied"
    applied = [row for row in isolated_state.history(conversation_id=task.session_id) if row["type"] == "steering.applied"]
    assert len(applied) == 1


def test_parallel_tool_callbacks_keep_original_operation_identity(monkeypatch):
    from mana_agent.commands.ui_helpers import RichToolCallbackHandler
    received = []
    monkeypatch.setattr("mana_agent.commands.ui_helpers.emit_tool_event", lambda *args, **kwargs: received.append((args, kwargs)))
    callback = RichToolCallbackHandler()
    callback.on_tool_start({"name": "read_file"}, '{"path":"one.py"}', run_id="one")
    callback.on_tool_start({"name": "read_file"}, '{"path":"two.py"}', run_id="two")
    callback.on_tool_end("done", run_id="two")
    callback.on_tool_end("done", run_id="one")
    assert received[0][1]["event_id"] == received[3][1]["event_id"]
    assert received[1][1]["event_id"] == received[2][1]["event_id"]
    assert received[0][1]["event_id"] != received[1][1]["event_id"]
    assert "one.py" not in json.dumps(received)
