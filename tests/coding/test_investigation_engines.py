"""Selected engine handoff, source validation, and no-fallback behavior."""

import json
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
    ("inferred", "multiple_findings"), ("inferred", "legacy_binding"),
    ("inferred", "negative_index"), ("inferred", "past_last_finding"), ("inferred", "unknown_requirement"),
    ("inferred", "unknown_evidence_requirement"), ("inferred", "unknown_legacy_requirement"),
    ("inferred", "observed_label"), ("observed", "observed_label"), ("inferred", "unbacked_observed_label"),
    ("observed", "shallow_root_citation"), ("inferred", "shallow_file_citation"),
    ("inferred", "shallow_nested_citation"), ("inferred", "audit_file_citation"),
    ("inferred", "extra_report_metadata"),
])
def test_investigation_uses_selected_engine_and_keeps_findings_as_inferences(tmp_path, monkeypatch, backend, classification, outcome_kind):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/app.py").write_bytes(b"value = 1\r\n")
    db = WorkspaceDatabase("engine-test", db_path=tmp_path / ".state.db")
    knowledge = RepositoryEvidenceBoard(db=db, repository_id="repo", root=tmp_path)
    layout_only = outcome_kind.startswith("shallow_")
    directory = outcome_kind in {"directory_location", "directory_named", "directory_wrong_excerpt",
                                  "artifact_hash", "shallow_root_citation", "shallow_nested_citation"}
    directory_path = "." if outcome_kind == "shallow_root_citation" else "src"
    result_source = read_directory(tmp_path, directory_path) if directory else read_source(tmp_path, "src/app.py")
    source = EvidenceSource.model_validate(result_source["source_evidence"])
    citation = source.model_dump(mode="json")
    if outcome_kind in {"file_location", "file_dedented", "directory_location", "directory_named", "directory_wrong_excerpt", "wrong_excerpt", "wrong_range", "reuse"}:
        citation.pop("sha256")
        citation["excerpt"] = "" if directory else "value = 1\n"
    if outcome_kind in {"file_location", "wrong_range"}:
        citation.pop("excerpt")
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
    if outcome_kind == "shallow_root_citation":
        citation = {"path": ".", "source_kind": "directory"}
    task = TaskBoardItem(task_id="task", parent_task_id=None, root_task_id="task", title="Understand value",
        user_request="Determine the value", normalized_goal="Determine the value", status=TaskStatus.NEW,
        priority=100, risk_level=RiskLevel.LOW, primary_repository_id="repo", task_kind="investigation",
        investigation_scope=["."] if layout_only or outcome_kind == "audit_file_citation" else ["src"],
        evidence_requirements=[EvidenceRequirement(requirement_id="value", objective="Determine the value",
            acceptable_classifications=(classification,), acceptance_criteria=("Cite current source",)).model_dump(mode="json")])
    if outcome_kind == "audit_file_citation":
        task.mode_workflow["initial_repository_scope"] = "explicit_audit"

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
    architecture_link = {"source": "CLI", "relation": "calls", "target": "app"}
    class Worker:
        def generate(self, prompt, **arguments):
            calls.append(arguments)
            schema = json.loads(prompt.rsplit("\n", 1)[-1])["result_schema"]
            finding_ids = schema["$defs"]["InvestigationFinding"]["properties"]["requirement_ids"]
            assert finding_ids["items"]["enum"] == ["value"]
            assert set(schema["properties"]["evidence_bindings"]["properties"]) == {"value"}
            assert '"entrypoints"' not in prompt
            properties = schema["$defs"]["InvestigationCitation"]["properties"]
            if layout_only:
                assert set(properties) == {"path", "source_kind"}
                assert properties["path"]["const"] == "."
                assert properties["source_kind"]["const"] == "directory"
                assert "Request missing evidence if you need file content" in prompt
            else:
                assert {"start_line", "end_line"} <= properties.keys()
            if outcome_kind == "failed":
                return {"status": "failed", "error": "Selected backend unavailable"}
            if outcome_kind == "malformed":
                return {"status": "completed", "investigation_result": {"unsupported": True}}
            finding: dict[str, object] = {"category": "code_behavior", "statement": "The repository has an application file",
                "classification": "inferred", "sources": [citation], "supporting_evidence": supporting,
                "architecture_relationships": [architecture_link], "confidence": 0.9}
            if outcome_kind in {"observed_label", "unbacked_observed_label"}:
                finding["classification"] = "observed"
            if outcome_kind == "unbacked_observed_label":
                finding["sources"] = []
            findings = [finding]
            report: dict[str, object] = {"findings": findings}
            legacy_indexes = {"legacy_binding": [0], "negative_index": [-1], "past_last_finding": [1]}
            if outcome_kind in legacy_indexes:
                report["requirement_bindings"] = {"value": legacy_indexes[outcome_kind]}
            elif outcome_kind == "unknown_legacy_requirement":
                report["requirement_bindings"] = {"PRIVATE_REQUIREMENT_ID": [0]}
            elif outcome_kind == "unknown_requirement":
                finding["requirement_ids"] = ["value", "PRIVATE_REQUIREMENT_ID"]
            else:
                finding["requirement_ids"] = ["value"]
            if outcome_kind == "unknown_evidence_requirement":
                report["evidence_bindings"] = {"PRIVATE_REQUIREMENT_ID": []}
            if outcome_kind == "extra_report_metadata":
                report["architecture_relationships"] = [{"source": "PRIVATE_FIXTURE", "relation": "calls", "target": "app"}]
                report["PRIVATE_FIELD"] = {"requirement_ids": ["outside-assignment"]}
            if outcome_kind == "multiple_findings":
                findings.insert(0, {"category": "code_behavior", "statement": "There may be another module",
                    "classification": "hypothetical", "confidence": 0.5})
            return {"status": "completed", "investigation_result": report}
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
        rejected_before_publication = {
            "negative_index": "index outside the findings array",
            "past_last_finding": "index outside the findings array",
            "unknown_requirement": r"unknown evidence requirement in findings\[\]\.requirement_ids",
            "unknown_evidence_requirement": "unknown evidence requirement in evidence_bindings",
            "unknown_legacy_requirement": "unknown evidence requirement in requirement_bindings",
            "unbacked_observed_label": "Inferences require actual supporting observations",
            "shallow_file_citation": "Shallow layout investigations can cite only the root directory",
            "shallow_nested_citation": "Shallow layout investigations can cite only the root directory",
        }
        if outcome_kind in rejected_before_publication:
            reason = rejected_before_publication[outcome_kind]
            with pytest.raises((ValueError, PermissionError), match=reason) as error:
                execute_investigation(taskboard=Board(), task_id="task", coding_agent=engine,
                    decision_id="decision", validate_source=validate)
            assert "PRIVATE_REQUIREMENT_ID" not in str(error.value)
            assert reads == []
            assert knowledge.query(EvidenceQuery(include_stale=True))["records"] == []
            assert task.evidence_refs == []
            assert task.evidence_requirements[0]["evidence"] == []
            return
        citation_failures = {
            "artifact_hash": "source digest differs from the read",
            "wrong_excerpt": "file excerpt does not occur in the selected lines",
            "wrong_range": "line range ends after the read",
            "directory_wrong_excerpt": "directory excerpt contains unobserved entries",
        }
        if outcome_kind in citation_failures:
            with pytest.raises(ValueError, match=citation_failures[outcome_kind]) as error:
                execute_investigation(taskboard=Board(), task_id="task", coding_agent=engine,
                    decision_id="decision", validate_source=validate)
            assert "findings[0].sources[0]" in str(error.value)
            assert "value = 2" not in str(error.value)
            assert len(reads) == 1
            assert [record["classification"] for record in knowledge.query(EvidenceQuery())["records"]] == ["observed"]
            return
        result = execute_investigation(taskboard=Board(), task_id="task", coding_agent=engine,
            decision_id="decision", validate_source=validate)
        assert result["goal_satisfied"]
        assert calls[0]["repository_investigation"]["backend"] == backend
        assert calls[0]["repository_investigation"]["layout_only"] is layout_only
        assert calls[0]["execution_mode"] == "read_only"
        assert calls[-1] == "closed"
        assert task.task_kind == "investigation"
        assert (tmp_path / "src/app.py").read_bytes() == b"value = 1\r\n"
        assert len(reads) == (0 if outcome_kind == "reuse" else 1)
        records = knowledge.query(EvidenceQuery())["records"]
        finding = records[0]
        assert finding["classification"] == "inferred"
        assert finding["sources"][0]["sha256"] == source.sha256
        if outcome_kind == "file_location":
            assert finding["sources"][0]["excerpt"] == result_source["content"]
        assert finding["architecture_relationships"] == [architecture_link]
        if outcome_kind == "extra_report_metadata":
            assert result["limitations"] == ["Ignored 2 unsupported top-level report fields."]
        if outcome_kind == "multiple_findings":
            assert len(result["evidence_refs"]) == 2
            assert task.evidence_requirements[0]["evidence"] == [result["evidence_refs"][1]]
        if outcome_kind == "observed_label" and classification == "observed":
            observed = records[1]
            assert observed["classification"] == "observed"
            assert task.evidence_requirements[0]["evidence"] == [
                {"evidence_id": observed["evidence_id"], "revision": observed["revision"]},
            ]
    finally:
        db.close()


@pytest.mark.parametrize("include_excerpt", [False, True])
def test_investigation_matches_multiline_code_citation(tmp_path, monkeypatch, include_excerpt):
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

    citation = {
        "path": "src/cli_recon.c",
        "source_kind": "file",
        "start_line": 3,
        "end_line": 5,
    }
    if include_excerpt:
        citation["excerpt"] = "if (rc == NP_OK)\n    rc = np_recon_persist_flush(ctx);"
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
    if not include_excerpt:
        from mana_agent.repository_evidence.models import EvidenceRef

        reused = knowledge.get(EvidenceRef.model_validate(result_reuse["evidence_refs"][0]))
        assert reused.sources[0].excerpt == "".join(code.splitlines(keepends=True)[2:5])
        assert reused.sources[0].start_line == 3
        assert reused.sources[0].end_line == 5

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
    with pytest.raises(ValueError, match="file excerpt does not occur in the selected lines"):
        execute_investigation(taskboard=Board(), task_id="task", coding_agent=engine_wrong,
            decision_id="decision", validate_source=validate)

    db.close()


def test_investigation_wire_schema_keeps_source_versions_host_owned():
    from mana_agent.repository_evidence.models import InvestigationCitation

    properties = InvestigationCitation.model_json_schema()["properties"]
    assert {"path", "start_line", "end_line"} <= properties.keys()
    assert "sha256" not in properties
    assert "revision" not in properties
    assert "excerpt" not in properties


def test_investigation_file_citation_requires_a_range_or_legacy_proof():
    from mana_agent.repository_evidence.models import InvestigationCitation

    with pytest.raises(ValueError, match="File citations require an exact line range"):
        InvestigationCitation.model_validate({"path": "src/app.py"})


def test_investigation_citation_rejects_unavailable_read_tool():
    from mana_agent.repository_evidence.models import InvestigationCitation
    from mana_agent.repository_evidence.observations import validate_selected_source

    runner = SimpleNamespace(
        authority=SimpleNamespace(schemas={"list_files": {"properties": {}}}),
        request=lambda *args, **kwargs: pytest.fail("Requested a tool outside the bound scope"),
    )
    citation = InvestigationCitation(path="src/app.py", start_line=1, end_line=1)
    with pytest.raises(PermissionError, match="read_file.*unavailable in the assigned scope"):
        validate_selected_source(runner, citation, decision_id="decision")


def test_investigation_wire_schema_binds_requirements_on_each_finding():
    from mana_agent.repository_evidence.models import InvestigationResult

    schema = InvestigationResult.model_json_schema()
    assert "requirement_bindings" not in schema["properties"]
    assert "evidence_bindings" in schema["properties"]
    requirement_ids = schema["$defs"]["InvestigationFinding"]["properties"]["requirement_ids"]
    assert requirement_ids["type"] == "array"
    assert requirement_ids["items"]["type"] == "string"


@pytest.mark.parametrize("ids", [("repository_finding", "caller"), ("other-task-requirement",), ()])
def test_investigation_schema_scopes_requirement_ids_to_assignment(ids):
    from mana_agent.repository_evidence.handoff import investigation_result_schema
    from mana_agent.repository_evidence.models import CodingAssignment

    assignment = CodingAssignment(
        task_id="task", task_version=1, objective="Find entrypoints", task_kind="investigation",
        backend="internal", source_view=".", decision_id="decision",
        requirements=tuple(EvidenceRequirement(requirement_id=requirement_id, objective="Trace a call",
                          acceptance_criteria=("Find a caller",)) for requirement_id in ids),
    )
    schema = investigation_result_schema(assignment)
    finding_ids = schema["$defs"]["InvestigationFinding"]["properties"]["requirement_ids"]
    if ids:
        assert finding_ids["items"]["enum"] == list(ids)
    else:
        assert finding_ids["maxItems"] == 0
    bindings = schema["properties"]["evidence_bindings"]
    assert set(bindings["properties"]) == set(ids)
    assert bindings["additionalProperties"] is False
    assert all(prop["items"] == {"$ref": "#/$defs/EvidenceRef"} for prop in bindings["properties"].values())


def test_missing_evidence_handoff_is_typed_and_preserves_task_identity():
    from mana_agent.repository_evidence.handoff import missing_evidence_from_text
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


def test_investigation_result_reports_incomplete_json_without_exposing_content():
    from mana_agent.repository_evidence.handoff import parse_investigation_result

    with pytest.raises(ValueError, match="expected one complete JSON object") as error:
        parse_investigation_result('{"limitations": ["PRIVATE_FIXTURE"')
    assert "PRIVATE_FIXTURE" not in str(error.value)


@pytest.mark.parametrize(("programmatic", "outcome"), [
    (False, "report"), (True, "report"), (False, "resume"), (False, "text"), (False, "mixed"), (True, "hosted"),
    (True, "pending"),
])
def test_native_investigation_completes_with_one_direct_report(tmp_path, monkeypatch, programmatic, outcome):
    import asyncio

    from mana_agent.coding.internal_engine import InternalCodingEngine
    from mana_agent.coding.models import CodingTask, WorkspaceContext
    from mana_agent.coding.programmatic import ProgrammaticExecutionSession
    from mana_agent.llm.errors import StructuredOutputError
    from mana_agent.llm.messages import ToolCall, UserMessage
    from mana_agent.llm.types import ModelResponse
    from mana_agent.repository_evidence.handoff import INVESTIGATION_REPORT_TOOL, parse_investigation_result
    from mana_agent.repository_evidence.models import CodingAssignment

    assignment = CodingAssignment(
        task_id="task", task_version=1, objective="Read the file", task_kind="investigation",
        backend="internal", source_view=str(tmp_path), decision_id="selected",
        allowed_scope=("app.py",), requirements=(EvidenceRequirement(
            requirement_id="value", objective="Find the value", acceptance_criteria=("Cite a read",),
        ),),
    )
    context = {"repository_investigation": assignment.model_dump(mode="json")}
    if outcome == "resume":
        context["coding_resume_context"] = {
            "native_messages": [UserMessage("Use the saved assignment.").model_dump(mode="json")],
        }
    task = CodingTask(task_id="task", goal=assignment.objective, requires_repository_write=False, context=context)
    workspace = WorkspaceContext(repository_path=tmp_path, worktree_path=tmp_path, sandbox="readOnly")
    read = ToolCall(id="read", name="read_file", args={"path": "app.py"})
    result = {"limitations": ["No matching declaration"]}
    report = ToolCall(id="report", name=INVESTIGATION_REPORT_TOOL, args=result)
    terminal = {
        "text": ModelResponse(content='{"limitations": ["PRIVATE_FIXTURE"'),
        "mixed": ModelResponse(tool_calls=[report, read.model_copy(update={"id": "extra-read"})]),
        "hosted": ModelResponse(tool_calls=[report.model_copy(update={"caller": {"type": "programmatic"}})]),
        "pending": ModelResponse(tool_calls=[report], protocol_pending=True, protocol_events=[{"type": "program"}]),
    }.get(outcome, ModelResponse(tool_calls=[report]))
    responses = iter((ModelResponse(tool_calls=[read]), terminal))
    dispatched = []

    class Model:
        def with_tools(self, tools):
            definitions = [tool.get("function", tool) for tool in tools]
            submitted = next(tool for tool in definitions if tool.get("name") == INVESTIGATION_REPORT_TOOL)
            findings = submitted["parameters"]["$defs"]["InvestigationFinding"]["properties"]
            assert findings["requirement_ids"]["items"]["enum"] == ["value"]
            if programmatic:
                assert submitted["allowed_callers"] == ["direct"]
            return self

        async def generate(self, messages, *, continuation):
            assert any(INVESTIGATION_REPORT_TOOL in message.content for message in messages if message.role == "user")
            if dispatched:
                assert messages[-1].tool_call_id == "read"
            return next(responses)

    def dispatch(tool, arguments, *, call_id):
        dispatched.append((tool, arguments, call_id))
        return {"ok": True, "content": "value = 1"}

    session = ProgrammaticExecutionSession(task_id="task", dispatcher=SimpleNamespace(dispatch=dispatch), max_tool_calls=5)
    engine = InternalCodingEngine(model="fixture", provider="openai")
    monkeypatch.setattr(engine, "_should_use_programmatic", lambda: programmatic)
    monkeypatch.setattr(engine, "_workspace_instruction", lambda workspace: "Inspect the assigned source.")
    monkeypatch.setattr(engine, "_build_tool_definitions", lambda dispatcher: [{
        "type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}},
    }])

    async def run():
        return await engine._run_model_tool_loop(task, workspace, session, asyncio.Event(), llm=Model())

    if outcome in {"report", "resume"}:
        parsed = parse_investigation_result(asyncio.run(run()))
        assert parsed.limitations == ("No matching declaration",)
        assert session.state.native_pending_calls == []
    else:
        with pytest.raises(StructuredOutputError, match="calling report_investigation exactly once") as error:
            asyncio.run(run())
        assert "PRIVATE_FIXTURE" not in str(error.value)
    assert dispatched == [("read_file", {"path": "app.py"}, "read")]
    assert session.state.tool_calls_executed == 1


@pytest.mark.parametrize(("label", "canonical"), [
    ("inferred", "inferred"), (" Inferred ", "inferred"),
    ("hypothetical", "hypothetical"), (" HYPOTHETICAL ", "hypothetical"),
    ("observed", "inferred"), (" Observed ", "inferred"),
])
def test_investigation_classification_normalizes_known_labels(label, canonical):
    from mana_agent.repository_evidence.handoff import parse_investigation_result

    report = parse_investigation_result(json.dumps({"findings": [{
        "category": "layout", "statement": "The application calls its handler.",
        "classification": label, "confidence": 0.8,
    }]}))
    assert report.findings[0].classification == canonical


def test_investigation_classification_schema_advertises_only_model_labels():
    from mana_agent.repository_evidence.models import InvestigationFinding

    classification = InvestigationFinding.model_json_schema()["properties"]["classification"]
    assert classification["enum"] == ["inferred", "hypothetical"]


def test_investigation_classification_rejection_names_allowed_labels_and_redacts_values():
    from mana_agent.repository_evidence.handoff import parse_investigation_result

    findings = [{
        "category": "layout", "statement": "The application calls its handler.",
        "classification": label, "confidence": 0.8,
    } for label in ("PRIVATE_FIXTURE", ["inferred"])]
    with pytest.raises(ValueError, match=r"findings\.0\.classification") as error:
        parse_investigation_result(json.dumps({"findings": findings}))

    message = str(error.value)
    assert "findings.1.classification" in message
    assert "Use 'inferred' or 'hypothetical'" in message
    assert "PRIVATE_FIXTURE" not in message


def test_investigation_architecture_relationships_use_a_closed_wire_record():
    from mana_agent.repository_evidence.models import InvestigationResult

    definitions = InvestigationResult.model_json_schema()["$defs"]
    items = definitions["InvestigationFinding"]["properties"]["architecture_relationships"]["items"]
    assert items == {"$ref": "#/$defs/ArchitectureRelationship"}
    link = definitions["ArchitectureRelationship"]
    assert link["additionalProperties"] is False
    assert set(link["properties"]) == set(link["required"]) == {"source", "relation", "target"}
    assert all(prop["type"] == "string" for prop in link["properties"].values())


def test_investigation_architecture_relationships_round_trip_through_the_parser():
    from mana_agent.repository_evidence.handoff import parse_investigation_result

    links = [
        {"source": "CLI", "relation": "calls", "target": "handler"},
        {"source": "CLI", "relation": "reads from", "target": "config"},
        {"from": "handler", "calls": "store"},  # Valid under the earlier contract.
    ]
    report = parse_investigation_result(json.dumps({"findings": [{
        "category": "architecture", "statement": "The CLI calls the handler.",
        "classification": "inferred", "confidence": 0.9, "architecture_relationships": links,
    }]}))
    assert report.model_dump(mode="json")["findings"][0]["architecture_relationships"] == links


@pytest.mark.parametrize("fields,issue", [
    ({"findings": [], "evidence_bindings": {"value": []}}, "Put architecture_relationships inside the finding"),
    ({"limitations": ["No source"], "supporting_evidence": []}, "supporting_evidence: extra forbidden"),
    ({"findings": [{"category": "layout", "statement": "A file exists.", "confidence": 0.9,
                   "classification": "inferred", "PRIVATE_NESTED_FIELD": "PRIVATE_VALUE"}]}, r"findings\.0\.<key>"),
])
def test_investigation_extra_report_metadata_cannot_replace_or_change_typed_content(fields, issue):
    from mana_agent.repository_evidence.handoff import parse_investigation_result

    payload = {"architecture_relationships": [{"source": "CLI", "relation": "calls", "target": "handler"}],
               "PRIVATE_FIELD": "PRIVATE_VALUE", **fields}
    with pytest.raises(ValueError, match=issue) as error:
        parse_investigation_result(json.dumps(payload))
    assert "PRIVATE_" not in str(error.value)


def test_investigation_architecture_relationships_reject_nested_values_without_leaking_them():
    from mana_agent.repository_evidence.handoff import parse_investigation_result

    findings: list[dict[str, object]] = [{
        "category": "architecture", "statement": "A component may call other components.",
        "classification": "hypothetical", "confidence": 0.5,
    } for _ in range(8)]
    findings[7]["architecture_relationships"] = [
        {"source": "CLI", "relation": "calls", "target": ["PRIVATE_FIXTURE"]},
        {"source": "CLI", "relation": "calls", "target": {"private_key_fixture": "PRIVATE_FIXTURE"}},
    ]
    with pytest.raises(ValueError, match=r"findings\.7\.architecture_relationships\.0") as error:
        parse_investigation_result(json.dumps({"findings": findings}))
    assert "findings.7.architecture_relationships.1" in str(error.value)
    assert "PRIVATE_FIXTURE" not in str(error.value)
    assert "private_key_fixture" not in str(error.value)


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
