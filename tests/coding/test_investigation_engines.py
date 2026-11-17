"""Selected engine handoff, source validation, and no-fallback behavior."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from mana_agent.persistence.workspace_db import WorkspaceDatabase
from mana_agent.repository_evidence.investigation import execute_investigation
from mana_agent.repository_evidence.models import EvidenceCandidate, EvidenceProvenance, EvidenceQuery, EvidenceRequirement, EvidenceSource
from mana_agent.repository_evidence.observations import read_source
from mana_agent.repository_evidence.service import RepositoryEvidenceBoard
from mana_agent.multi_agent.core.types import RiskLevel, TaskBoardItem, TaskStatus


@pytest.mark.parametrize("backend", ["internal", "external"])
@pytest.mark.parametrize("classification", ["inferred", "observed"])
@pytest.mark.parametrize("outcome_kind", ["valid", "failed", "malformed"])
def test_investigation_uses_selected_engine_and_keeps_findings_as_inferences(tmp_path, monkeypatch, backend, classification, outcome_kind):
    (tmp_path / "app.py").write_text("value = 1\n")
    db = WorkspaceDatabase("engine-test", db_path=tmp_path / "state.db")
    knowledge = RepositoryEvidenceBoard(db=db, repository_id="repo", root=tmp_path)
    source = EvidenceSource.model_validate(read_source(tmp_path, "app.py")["source_evidence"])
    task = TaskBoardItem(task_id="task", parent_task_id=None, root_task_id="task", title="Understand value",
        user_request="Determine the value", normalized_goal="Determine the value", status=TaskStatus.NEW,
        priority=100, risk_level=RiskLevel.LOW, primary_repository_id="repo", task_kind="investigation",
        investigation_scope=["app.py"], evidence_requirements=[EvidenceRequirement(requirement_id="value", objective="Determine the value",
            acceptable_classifications=(classification,), acceptance_criteria=("Cite current source",)).model_dump(mode="json")])

    class Board:
        store = SimpleNamespace(root=tmp_path)
        def get_task(self, _):
            return task
        def evidence_board(self, **_):
            return knowledge
        def update_task(self, _, **values):
            values.pop("expected_version", None)
            for key, value in values.items():
                setattr(task, key, value)
            task.row_version += 1
            return task

    monkeypatch.setattr("mana_agent.services.activity_context.emit_activity", lambda *args, **kwargs: None)
    calls = []
    class Worker:
        def generate(self, prompt, **arguments):
            calls.append(arguments)
            if outcome_kind == "failed":
                return {"status": "failed", "error": "Selected backend unavailable"}
            if outcome_kind == "malformed":
                return {"status": "completed", "investigation_result": {"unsupported": True}}
            return {"status": "completed", "investigation_result": {
                "findings": [EvidenceCandidate(category="code_behavior", statement="The value is one", classification="inferred",
                    sources=(source,), confidence=0.9).model_dump(mode="json")], "requirement_bindings": {"value": [0]}}}
        def close(self):
            calls.append("closed")
    engine = SimpleNamespace(active_engine_mode=backend, for_assignment=lambda **_: Worker())
    monkeypatch.setattr("mana_agent.services.activity_context.current_activity_context", lambda: SimpleNamespace(
        task_id="task", attempt_id="attempt", attempt_generation=1, protected_text=(), publication=nullcontext))
    def validate(citation):
        result = read_source(tmp_path, citation.path)
        record = knowledge.record_observation(provenance=EvidenceProvenance(task_id="task", attempt_id="attempt", attempt_generation=1,
            decision_id="decision", invocation_id="source-read", agent_id="runner", backend="native_tool_runner", result_digest="receipt"),
            category="code_behavior", statement="Observed source", observation={}, sources=(source,))
        result["repository_evidence_refs"] = [record.ref.model_dump(mode="json")]
        return result
    try:
        if outcome_kind != "valid":
            with pytest.raises((RuntimeError, ValueError)):
                execute_investigation(taskboard=Board(), task_id="task", coding_agent=engine,
                    decision_id="decision", validate_source=validate)
            assert len(calls) == 2 and calls[-1] == "closed"
            assert knowledge.query(EvidenceQuery())["records"] == []
            return
        result = execute_investigation(taskboard=Board(), task_id="task", coding_agent=engine,
            decision_id="decision", validate_source=validate)
        assert result["goal_satisfied"]
        assert calls[0]["repository_investigation"]["backend"] == backend
        assert calls[0]["execution_mode"] == "read_only"
        assert calls[-1] == "closed"
        assert task.task_kind == "investigation"
        assert (tmp_path / "app.py").read_text() == "value = 1\n"
        assert knowledge.query(EvidenceQuery())["records"][0]["classification"] == "inferred"
    finally:
        db.close()


def test_missing_evidence_handoff_is_typed_and_preserves_task_identity():
    from mana_agent.repository_evidence.handoff import missing_evidence_from_text
    import json
    request = {"task_id": "coding-task", "requirement": {"requirement_id": "caller", "objective": "Identify the caller",
        "paths": ["src"], "acceptance_criteria": ["Cite a call site"]}, "reason": "The caller is unresolved",
        "continuation_reference": "saved-turn"}
    assert missing_evidence_from_text(json.dumps({"missing_evidence": [request]}))[0]["task_id"] == "coding-task"
    assert missing_evidence_from_text("```json\n" + json.dumps({"missing_evidence": [request]}) + "\n```")[0]["task_id"] == "coding-task"
    with pytest.raises(ValueError):
        missing_evidence_from_text(json.dumps({"missing_evidence": [{"reason": "missing identity"}]}))


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "```\r\n{}\r\n```"])
def test_investigation_result_accepts_one_json_document_with_optional_fence(wrapper):
    from mana_agent.repository_evidence.handoff import parse_investigation_result

    report = parse_investigation_result(wrapper.format('{"limitations": ["No matching file"]}'))
    assert report.limitations == ("No matching file",)


@pytest.mark.parametrize("response", [
    'Results: ```json\n{"untrusted_sample":"PRIVATE_FIXTURE"}\n```',
    '```json\n{"untrusted_sample":"PRIVATE_FIXTURE"}\n```',
    '```json\n{}\n```\n```json\n{"untrusted_sample":"PRIVATE_FIXTURE"}\n```',
])
def test_invalid_investigation_result_does_not_expose_provider_payload(response):
    from mana_agent.repository_evidence.handoff import parse_investigation_result

    with pytest.raises(ValueError, match="Invalid InvestigationResult") as error:
        parse_investigation_result(response)
    assert "PRIVATE_FIXTURE" not in str(error.value)
    assert "untrusted_sample" not in str(error.value)


def test_cancelling_coding_task_interrupts_active_model_wait(monkeypatch):
    import asyncio

    from mana_agent.coding.internal_engine import InternalCodingEngine

    lifecycle = []
    monkeypatch.setattr("mana_agent.services.activity_context.emit_activity",
                        lambda event, **fields: lifecycle.append((event, fields)))

    async def exercise():
        engine = InternalCodingEngine()
        called = asyncio.Event()
        interrupted = asyncio.Event()

        class Model:
            async def generate(self, messages, *, continuation):
                called.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    interrupted.set()

        pending = asyncio.create_task(engine._await_model_turn("assigned", 0, Model(), [], None, asyncio.Event()))
        await asyncio.wait_for(called.wait(), timeout=1)
        await asyncio.to_thread(lambda: asyncio.run(engine.cancel("assigned")))
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, timeout=1)
        assert interrupted.is_set()
        assert "assigned" not in engine._pending_model_calls

    asyncio.run(exercise())
    assert [(event, fields["status"]) for event, fields in lifecycle] == [
        ("model.started", "running"), ("model.failed", "cancelled"),
    ]


@pytest.mark.parametrize("backend", ["internal", "external"])
def test_task_local_facades_keep_engine_scope_and_propagate_parent_cancellation(tmp_path, monkeypatch, backend):
    from mana_agent.coding.facade import UnifiedCodingAgentFacade
    from mana_agent.integrations.codex.config import CodexSettings
    from mana_agent.services.activity_context import ActivityContext, activity_scope
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    parent = UnifiedCodingAgentFacade(repo_root=tmp_path, default_mode=backend, session_id="chat",
        codex_settings=CodexSettings(worktree_isolation=True))
    with activity_scope(ActivityContext(conversation_id="chat", execution_id="turn", task_id="child",
                                      root_task_id="parent", parent_task_id="parent")):
        worker = parent.for_assignment(root=tmp_path, task_id="child")
    called = []
    monkeypatch.setattr(worker, "cancel", lambda task_id=None: called.append(task_id) or True)
    try:
        assert worker.active_engine_mode == parent.active_engine_mode == backend
        assert worker.session_id == "chat-child"
        assert worker.repo_root == tmp_path
        assert not worker.codex_settings.worktree_isolation
        assert parent.codex_settings.worktree_isolation
        assert parent.cancel("parent")
        assert called == [None]
    finally:
        parent.close()


def test_registered_read_ingests_host_bound_evidence_and_operation_receipt(tmp_path, monkeypatch):
    from mana_agent.coding.programmatic import ToolExecutionDispatcher
    from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
    from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
    from mana_agent.multi_agent.core.types import AgentRole
    from mana_agent.multi_agent.registry.agent_registry import AgentRegistry
    from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
    from mana_agent.services.activity_context import ActivityContext, activity_scope
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text("value = 1\n")
    board, registry = TaskBoard(root), AgentRegistry()
    owner = registry.find_by_role(AgentRole.CODING)
    task = board.create_task(title="Inspect", user_request="Inspect value", owner_agent_id=owner.agent_id)
    board.update_task(task.task_id, task_kind="investigation", investigation_scope=["app.py"])
    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig(root=tmp_path / "supervisor", startup_recovery=False))
    supervisor.create_task(task_id=task.task_id, routing_decision_id="selected", workspace_path=root)
    supervisor.queue(task.task_id)
    leased, token = supervisor.acquire_lease(task.task_id, owner=owner.agent_id)
    supervisor.start(task.task_id, attempt_id=leased.attempt_id, lease_token=token)
    context = ActivityContext(conversation_id="chat", execution_id="turn", task_id=task.task_id,
        attempt_id=leased.attempt_id, attempt_generation=leased.attempt_generation, lease_token=token,
        supervisor=supervisor, execution_registry=registry, taskboard=board)
    with activity_scope(context):
        dispatcher = ToolExecutionDispatcher.for_coding_task(repo_root=root, task_id=task.task_id)
        result = dispatcher.dispatch("read_file", {"path": "app.py"}, call_id="read-value")
    assert result["ok"]
    assert result["repository_evidence_refs"]
    rows = board.evidence_board().query(EvidenceQuery(task_id=task.task_id))["records"]
    assert rows[0]["classification"] == "observed"
    assert rows[0]["sources"][0]["path"] == "app.py"
    with board.store.db.connect() as conn:
        operation = conn.execute("SELECT task_id,status FROM tool_queue_jobs WHERE kind='operation'").fetchone()
    assert operation["task_id"] == task.task_id and operation["status"] == "success"


@pytest.mark.parametrize("method", ["_run_programmatic_workflow", "_run_direct_tool_workflow"])
def test_unavailable_internal_model_cannot_select_reads_or_claim_completion(tmp_path, monkeypatch, method):
    import asyncio
    from mana_agent.coding.internal_engine import InternalCodingEngine
    from mana_agent.coding.models import CodingTask, WorkspaceContext
    engine = InternalCodingEngine()
    monkeypatch.setattr(engine, "_get_or_create_llm", lambda: None)
    monkeypatch.setattr(engine, "_build_tool_definitions", lambda *_: [])
    calls = []
    session = SimpleNamespace(dispatcher=None, handle_function_call=lambda call: calls.append(call))
    task = CodingTask(task_id="assigned", goal="Implement change", allowed_files=["app.py"])
    workspace = WorkspaceContext(repository_path=tmp_path, worktree_path=tmp_path)
    with pytest.raises(RuntimeError, match="model is unavailable"):
        asyncio.run(getattr(engine, method)(task, workspace, session, asyncio.Event()))
    assert calls == []
