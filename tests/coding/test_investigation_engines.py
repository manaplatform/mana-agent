"""Selected engine handoff, source validation, and no-fallback behavior."""

from contextlib import nullcontext
from hashlib import sha256
from types import SimpleNamespace

import pytest

from mana_agent.persistence.workspace_db import WorkspaceDatabase
from mana_agent.repository_evidence.investigation import execute_investigation
from mana_agent.repository_evidence.models import EvidenceProvenance, EvidenceQuery, EvidenceRequirement, EvidenceSource
from mana_agent.repository_evidence.observations import read_directory, read_source
from mana_agent.repository_evidence.service import RepositoryEvidenceBoard
from mana_agent.multi_agent.core.types import RiskLevel, TaskBoardItem, TaskStatus


@pytest.mark.parametrize("backend", ["internal", "external"])
@pytest.mark.parametrize("classification,outcome_kind", [
    ("inferred", "valid"), ("observed", "valid"), ("observed", "file_location"),
    ("observed", "file_dedented"), ("inferred", "directory_location"), ("inferred", "directory_named"),
    ("inferred", "artifact_hash"), ("observed", "wrong_excerpt"), ("observed", "directory_wrong_excerpt"),
    ("observed", "wrong_range"), ("observed", "reuse"), ("observed", "failed"), ("observed", "malformed"),
])
def test_investigation_uses_selected_engine_and_keeps_findings_as_inferences(tmp_path, monkeypatch, backend, classification, outcome_kind):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/app.py").write_bytes(b"value = 1\r\n")
    db = WorkspaceDatabase("engine-test", db_path=tmp_path / "state.db")
    knowledge = RepositoryEvidenceBoard(db=db, repository_id="repo", root=tmp_path)
    directory = outcome_kind in {"directory_location", "directory_named", "directory_wrong_excerpt", "artifact_hash"}
    result_source = read_directory(tmp_path, "src") if directory else read_source(tmp_path, "src/app.py")
    source = EvidenceSource.model_validate(result_source["source_evidence"])
    citation = source.model_dump(mode="json")
    if outcome_kind in {"file_location", "file_dedented", "directory_location", "directory_named", "directory_wrong_excerpt", "wrong_excerpt", "wrong_range", "reuse"}:
        citation.pop("sha256")
        citation["excerpt"] = "" if directory else "value = 1\n"
    if outcome_kind == "file_dedented":
        citation["excerpt"] = "value = 1"
    if outcome_kind == "directory_named":
        citation["excerpt"] = "app.py"
    if outcome_kind == "directory_wrong_excerpt":
        citation["excerpt"] = "app.py, missing.py"
    if outcome_kind == "artifact_hash":
        citation["sha256"] = sha256(b"serialized search result").hexdigest()
    if outcome_kind == "wrong_excerpt":
        citation["excerpt"] = "value = 2"
    if outcome_kind == "wrong_range":
        citation["end_line"] = 2
    task = TaskBoardItem(task_id="task", parent_task_id=None, root_task_id="task", title="Understand value",
        user_request="Determine the value", normalized_goal="Determine the value", status=TaskStatus.NEW,
        priority=100, risk_level=RiskLevel.LOW, primary_repository_id="repo", task_kind="investigation",
        investigation_scope=["src"], evidence_requirements=[EvidenceRequirement(requirement_id="value", objective="Determine the value",
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
    supporting = []
    class Worker:
        def generate(self, prompt, **arguments):
            calls.append(arguments)
            if outcome_kind == "failed":
                return {"status": "failed", "error": "Selected backend unavailable"}
            if outcome_kind == "malformed":
                return {"status": "completed", "investigation_result": {"unsupported": True}}
            return {"status": "completed", "investigation_result": {
                "findings": [{"category": "code_behavior", "statement": "The repository has an application file",
                    "classification": "inferred", "sources": [citation], "supporting_evidence": supporting,
                    "confidence": 0.9}], "requirement_bindings": {"value": [0]}}}
        def close(self):
            calls.append("closed")
    engine = SimpleNamespace(active_engine_mode=backend, for_assignment=lambda **_: Worker())
    monkeypatch.setattr("mana_agent.services.activity_context.current_activity_context", lambda: SimpleNamespace(
        task_id="task", attempt_id="attempt", attempt_generation=1, protected_text=(), publication=nullcontext))
    reads = []
    def validate(selected):
        reads.append(selected)
        result = read_directory(tmp_path, selected.path) if directory else read_source(
            tmp_path, selected.path, start_line=selected.start_line, end_line=selected.end_line)
        record = knowledge.record_observation(provenance=EvidenceProvenance(task_id="task", attempt_id="attempt", attempt_generation=1,
            decision_id="decision", invocation_id="source-read", agent_id="runner", backend="native_tool_runner", result_digest="receipt"),
            category="code_behavior", statement="Observed source", observation={}, sources=(source,))
        result["repository_evidence_refs"] = [record.ref.model_dump(mode="json")]
        return result
    try:
        if outcome_kind == "reuse":
            supporting.extend(validate(source)["repository_evidence_refs"])
            reads.clear()
        if outcome_kind in {"failed", "malformed"}:
            with pytest.raises((RuntimeError, ValueError)):
                execute_investigation(taskboard=Board(), task_id="task", coding_agent=engine,
                    decision_id="decision", validate_source=validate)
            assert len(calls) == 2 and calls[-1] == "closed"
            assert knowledge.query(EvidenceQuery())["records"] == []
            return
        if outcome_kind in {"artifact_hash", "wrong_excerpt", "wrong_range", "directory_wrong_excerpt"}:
            with pytest.raises(ValueError, match="does not match the observed path, version, or excerpt"):
                execute_investigation(taskboard=Board(), task_id="task", coding_agent=engine,
                    decision_id="decision", validate_source=validate)
            assert len(reads) == 1
            assert [record["classification"] for record in knowledge.query(EvidenceQuery())["records"]] == ["observed"]
            return
        result = execute_investigation(taskboard=Board(), task_id="task", coding_agent=engine,
            decision_id="decision", validate_source=validate)
        assert result["goal_satisfied"]
        assert calls[0]["repository_investigation"]["backend"] == backend
        assert calls[0]["execution_mode"] == "read_only"
        assert calls[-1] == "closed"
        assert task.task_kind == "investigation"
        assert (tmp_path / "src/app.py").read_bytes() == b"value = 1\r\n"
        assert len(reads) == (0 if outcome_kind == "reuse" else 1)
        finding = knowledge.query(EvidenceQuery())["records"][0]
        assert finding["classification"] == "inferred"
        assert finding["sources"][0]["sha256"] == source.sha256
    finally:
        db.close()


def test_investigation_matches_multiline_code_citation_with_dedented_whitespace_and_blank_lines(tmp_path, monkeypatch):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    code = (
        "int cmd_recon(void)\n"
        "{\n"
        "\n"
        "    if (rc == NP_OK)\n"
        "        rc = np_recon_persist_flush(ctx);\n"
        "}\n"
    )
    (src_dir / "cli_recon.c").write_text(code)
    db = WorkspaceDatabase("engine-test-multiline", db_path=tmp_path / "state.db")
    knowledge = RepositoryEvidenceBoard(db=db, repository_id="repo", root=tmp_path)
    task = TaskBoardItem(
        task_id="task", parent_task_id=None, root_task_id="task", title="Understand recon",
        user_request="Understand recon persist", normalized_goal="Understand recon persist",
        status=TaskStatus.NEW, priority=100, risk_level=RiskLevel.LOW,
        primary_repository_id="repo", task_kind="investigation", investigation_scope=["src"],
        evidence_requirements=[EvidenceRequirement(
            requirement_id="persist", objective="Locate persist",
            acceptable_classifications=("inferred",), acceptance_criteria=("Cite flush",)
        ).model_dump(mode="json")],
    )

    class Board:
        store = SimpleNamespace(root=tmp_path)
        def get_task(self, _):
            return task
        def evidence_board(self, **_):
            return knowledge
        def update_task(self, _, **values):
            values.pop("expected_version", None)
            for k, v in values.items():
                setattr(task, k, v)
            task.row_version += 1
            return task

    monkeypatch.setattr("mana_agent.services.activity_context.emit_activity", lambda *args, **kwargs: None)
    monkeypatch.setattr("mana_agent.services.activity_context.current_activity_context", lambda: SimpleNamespace(
        task_id="task", attempt_id="attempt", attempt_generation=1, protected_text=(), publication=nullcontext
    ))

    # Citation covers lines 3 to 5 (line 3 is blank) with dedented 4-space body
    citation = {
        "path": "src/cli_recon.c",
        "source_kind": "file",
        "start_line": 3,
        "end_line": 5,
        "excerpt": "if (rc == NP_OK)\n    rc = np_recon_persist_flush(ctx);",
    }
    class Worker:
        def generate(self, prompt, **arguments):
            return {"status": "completed", "investigation_result": {
                "findings": [{
                    "category": "control_flow", "statement": "Persist flush called when OK",
                    "classification": "inferred", "sources": [citation], "supporting_evidence": [],
                    "confidence": 0.9,
                }],
                "requirement_bindings": {"persist": [0]},
            }}
        def close(self):
            pass

    engine = SimpleNamespace(active_engine_mode="internal", for_assignment=lambda **_: Worker())
    reads = []
    def validate(selected):
        reads.append(selected)
        result = read_source(tmp_path, selected.path, start_line=selected.start_line, end_line=selected.end_line)
        source = EvidenceSource.model_validate(result["source_evidence"])
        record = knowledge.record_observation(
            provenance=EvidenceProvenance(task_id="task", attempt_id="attempt", attempt_generation=1,
                decision_id="decision", invocation_id="source-read", agent_id="runner",
                backend="native_tool_runner", result_digest="receipt"),
            category="code_behavior", statement="Observed source", observation={}, sources=(source,),
        )
        result["repository_evidence_refs"] = [record.ref.model_dump(mode="json")]
        return result

    result = execute_investigation(taskboard=Board(), task_id="task", coding_agent=engine,
        decision_id="decision", validate_source=validate)
    assert result["goal_satisfied"]
    assert len(reads) == 1

    # Broad observation covering lines 1-6 in supporting_evidence reuses without reading
    broad_result = read_source(tmp_path, "src/cli_recon.c", start_line=1, end_line=6)
    broad_source = EvidenceSource.model_validate(broad_result["source_evidence"])
    broad_record = knowledge.record_observation(
        provenance=EvidenceProvenance(task_id="task", attempt_id="attempt", attempt_generation=1,
            decision_id="decision", invocation_id="broad-read", agent_id="runner",
            backend="native_tool_runner", result_digest="receipt"),
        category="code_behavior", statement="Observed broad", observation={}, sources=(broad_source,),
    )
    reads.clear()

    class WorkerReuse:
        def generate(self, prompt, **arguments):
            return {"status": "completed", "investigation_result": {
                "findings": [{
                    "category": "control_flow", "statement": "Persist flush called when OK",
                    "classification": "inferred", "sources": [citation],
                    "supporting_evidence": [broad_record.ref.model_dump(mode="json")],
                    "confidence": 0.9,
                }],
                "requirement_bindings": {"persist": [0]},
            }}
        def close(self):
            pass

    engine_reuse = SimpleNamespace(active_engine_mode="internal", for_assignment=lambda **_: WorkerReuse())
    result_reuse = execute_investigation(taskboard=Board(), task_id="task", coding_agent=engine_reuse,
        decision_id="decision", validate_source=validate)
    assert result_reuse["goal_satisfied"]
    assert len(reads) == 0

    # Wrong excerpt raises ValueError
    wrong_citation = dict(citation, excerpt="rc = np_wrong_call(ctx);")
    class WorkerWrong:
        def generate(self, prompt, **arguments):
            return {"status": "completed", "investigation_result": {
                "findings": [{
                    "category": "control_flow", "statement": "Wrong call",
                    "classification": "inferred", "sources": [wrong_citation], "supporting_evidence": [],
                    "confidence": 0.9,
                }],
                "requirement_bindings": {"persist": [0]},
            }}
        def close(self):
            pass

    engine_wrong = SimpleNamespace(active_engine_mode="internal", for_assignment=lambda **_: WorkerWrong())
    with pytest.raises(ValueError, match="does not match the observed path, version, or excerpt"):
        execute_investigation(taskboard=Board(), task_id="task", coding_agent=engine_wrong,
            decision_id="decision", validate_source=validate)

    db.close()


def test_investigation_wire_schema_keeps_source_versions_host_owned():
    from mana_agent.repository_evidence.models import InvestigationCitation

    properties = InvestigationCitation.model_json_schema()["properties"]
    assert "path" in properties
    assert "sha256" not in properties
    assert "revision" not in properties


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


@pytest.mark.parametrize("tool", ["read_file", "batch_read_files"])
def test_registered_read_ingests_host_bound_evidence_and_operation_receipt(tmp_path, monkeypatch, tool):
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
        arguments = {"path": "app.py"} if tool == "read_file" else {"files": [{"path": "app.py", "start_line": 1, "end_line": 1}]}
        result = dispatcher.dispatch(tool, arguments, call_id="read-value")
    assert result["ok"]
    assert result["repository_evidence_refs"]
    rows = board.evidence_board().query(EvidenceQuery(task_id=task.task_id))["records"]
    assert rows[0]["classification"] == "observed"
    assert rows[0]["sources"][0]["path"] == "app.py"
    if tool == "batch_read_files":
        assert result["files"][0]["repository_evidence_refs"] == [{"evidence_id": rows[0]["evidence_id"], "revision": rows[0]["revision"]}]
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
    workspace = WorkspaceContext(repository_path=tmp_path, worktree_path=tmp_path, allow_in_place_write=True)
    with pytest.raises(RuntimeError, match="model is unavailable"):
        asyncio.run(getattr(engine, method)(task, workspace, session, asyncio.Event()))
    assert calls == []
