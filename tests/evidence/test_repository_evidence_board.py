"""Evidence reliability and persistence, without provider calls or project commands."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from mana_agent.persistence.workspace_db import WorkspaceDatabase
from mana_agent.persistence.workspace_repository import WorkspaceRepository
from mana_agent.repository_evidence.models import (
    EvidenceCandidate, EvidenceDetailQuery, EvidenceProvenance, EvidenceQuery, EvidenceRelation, EvidenceRequirement, EvidenceSource,
)
from mana_agent.repository_evidence.observations import read_directory, read_source
from mana_agent.repository_evidence.service import EvidenceConflict, RepositoryEvidenceBoard
from mana_agent.multi_agent.core.types import RiskLevel, TaskBoardItem, TaskStatus


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text("def handler():\n    return 1\n")
    db = WorkspaceDatabase("test-evidence", db_path=tmp_path / "workspace.db")
    board = RepositoryEvidenceBoard(db=db, repository_id="repo", root=root)
    yield board
    db.close()


def provenance(task="task-1"):
    return EvidenceProvenance(task_id=task, attempt_id="attempt-1", attempt_generation=1,
        decision_id="decision-1", invocation_id="read-1", agent_id="worker", backend="native_tool_runner",
        result_digest="observed-result-digest")


def observe(board, path="app.py", *, task="task-1"):
    source = EvidenceSource.model_validate(read_source(board.root, path)["source_evidence"])
    return board.record_observation(provenance=provenance(task), category="code_behavior",
        statement="The registered read returned the handler source", observation={"tool": "read_file"}, sources=(source,))


def observe_distinct_files(board, count):
    records = []
    for index in range(count):
        path = f"handler_{index:02d}.py"
        (board.root / path).write_text("def handler():\n    return 1\n")
        records.append(observe(board, path))
    return records


def inference(board, observed, **kwargs):
    candidate = EvidenceCandidate(category="code_behavior", statement="The handler returns one", classification="inferred",
        confidence=0.9, sources=observed.sources, supporting_evidence=(observed.ref,), **kwargs)
    return board.submit_findings(candidate, provenance=provenance())


def test_empty_board_has_no_invented_evidence(evidence):
    assert evidence.query(EvidenceQuery())["records"] == []
    requirement = EvidenceRequirement(requirement_id="behavior", objective="Determine handler behavior",
                                     acceptance_criteria=("Cite the current source",))
    assert not evidence.satisfies(requirement)


def test_observation_and_inference_have_distinct_provenance(evidence):
    observed = observe(evidence)
    finding = inference(evidence, observed)
    assert evidence.get(observed.ref).classification == "observed"
    assert evidence.get(finding.ref).classification == "inferred"
    assert finding.supporting_evidence == (observed.ref,)
    assert finding.provenance.invocation_id == "read-1"


def test_models_cannot_submit_verified_observations(evidence):
    with pytest.raises(ValidationError):
        EvidenceCandidate(category="runtime", statement="All checks passed", classification="observed", confidence=1)
    with pytest.raises(ValueError, match="supporting observations"):
        evidence.submit_findings(EvidenceCandidate(category="runtime", statement="Checks passed",
            classification="inferred", confidence=1), provenance=provenance())


def test_hypotheses_never_satisfy_requirements(evidence):
    record = evidence.submit_findings(EvidenceCandidate(category="root_cause", statement="A race may exist",
        classification="hypothetical", confidence=1), provenance=provenance())
    requirement = EvidenceRequirement(requirement_id="cause", objective="Find the cause",
        acceptance_criteria=("Supported causal evidence",), evidence=(record.ref,))
    assert record.validation == "pending"
    assert not evidence.satisfies(requirement)


def test_valid_evidence_is_reusable_by_multiple_tasks(evidence):
    record = observe(evidence)
    evidence.link("other-task", (record.ref,))
    assert evidence.query(EvidenceQuery(task_id="other-task"))["records"][0]["evidence_id"] == record.evidence_id
    reopened = RepositoryEvidenceBoard(db=evidence.db, repository_id="repo", root=evidence.root)
    assert reopened.get(record.ref).freshness == "current"


@pytest.mark.parametrize("change", ["edit", "delete", "rename"])
def test_source_changes_invalidate_reuse_and_derived_findings(evidence, change):
    observed = observe(evidence)
    derived = inference(evidence, observed)
    path = evidence.root / "app.py"
    if change == "edit":
        path.write_text("def handler():\n    return 2\n")
    elif change == "delete":
        path.unlink()
    else:
        path.rename(evidence.root / "renamed.py")
    assert evidence.get(observed.ref).freshness == "stale"
    assert evidence.get(derived.ref).freshness == "stale"
    assert evidence.query(EvidenceQuery())["records"] == []
    assert evidence.query(EvidenceQuery(include_stale=True))["records"]


def test_mutation_publication_keeps_history_and_invalidates_dependents(evidence):
    observed = observe(evidence)
    derived = inference(evidence, observed)
    changed = evidence.invalidate_sources(["app.py"], task_id="task-1")
    assert {ref.evidence_id for ref in changed} == {observed.evidence_id, derived.evidence_id}
    assert evidence.get(observed.ref).freshness == "invalidated"
    assert evidence.get(observed.ref, refresh=False).freshness == "current"


def test_other_worktree_changes_do_not_invalidate_matching_sources(evidence, tmp_path):
    record = observe(evidence)
    other = tmp_path / "other"
    other.mkdir()
    (other / "app.py").write_text((evidence.root / "app.py").read_text())
    destination = RepositoryEvidenceBoard(db=evidence.db, repository_id="repo", root=other)
    assert destination.get(record.ref).freshness == "current"
    destination.invalidate_sources(["app.py"], task_id="task-1")
    assert evidence.get(record.ref).freshness == "current"


def test_contradictions_remain_visible_and_prevent_requirement_satisfaction(evidence):
    observed = observe(evidence)
    first = inference(evidence, observed)
    other = inference(evidence, observed, relationships=(EvidenceRelation(relation="contradicts", target=first.ref),))
    assert evidence.get(first.ref).validation == "contradicted"
    assert evidence.get(other.ref).validation == "contradicted"


def test_repository_and_symbol_queries_are_scoped(evidence):
    observed = observe(evidence)
    evidence.link("task-query", (observed.ref,))
    assert evidence.query(EvidenceQuery(paths=("app.py",), task_id="task-query"))["records"]
    foreign = RepositoryEvidenceBoard(db=evidence.db, repository_id="foreign", root=evidence.root)
    with pytest.raises(KeyError):
        foreign.get(observed.ref)


def test_relationship_and_dependency_queries_are_bounded(evidence):
    observed = observe(evidence)
    derived = inference(evidence, observed, dependencies=("service-contract",))
    rows = evidence.query(EvidenceQuery(related_to=(observed.ref,), dependencies=("service-contract",)))["records"]
    assert [row["evidence_id"] for row in rows] == [derived.evidence_id]


def test_large_board_context_is_bounded_and_paginated(evidence):
    observe_distinct_files(evidence, 30)
    page = evidence.query(EvidenceQuery(limit=3, max_context_tokens=1500))
    assert 1 <= len(page["records"]) <= 3
    assert page["context_bytes"] <= 1500
    assert page["next_offset"] == len(page["records"])
    assert page["has_more"]


def test_equivalent_source_observations_share_one_record(evidence):
    first = observe(evidence)
    repeat = observe(evidence, task="other-task")
    assert repeat.ref == first.ref
    for task in ("", "task-1", "other-task"):
        page = evidence.query(EvidenceQuery(task_id=task))
        assert [row["evidence_id"] for row in page["records"]] == [first.evidence_id]
        assert not page["has_more"]


def test_unknown_custom_category_preserves_structured_sources(evidence):
    observed = observe(evidence)
    custom = evidence.submit_findings(EvidenceCandidate(category="extension.custom_relationship", statement="Related contract",
        classification="inferred", confidence=0.7, sources=observed.sources, supporting_evidence=(observed.ref,)), provenance=provenance())
    assert evidence.query(EvidenceQuery(categories=(custom.category,)))["records"]


def test_stale_attempt_cannot_publish(evidence):
    @contextmanager
    def retired():
        raise PermissionError("retired attempt")
        yield
    evidence.publication = retired
    with pytest.raises(PermissionError, match="retired"):
        observe(evidence)
    assert evidence.query(EvidenceQuery())["records"] == []


@pytest.mark.parametrize("path", ["../outside", "/absolute", ".env", ".ssh/key", "secrets.toml"])
def test_protected_and_escaping_sources_are_rejected(evidence, path):
    with pytest.raises(ValueError):
        read_source(evidence.root, path)


def test_directory_source_detects_new_entries_without_recursive_content_reads(evidence):
    captured = EvidenceSource.model_validate(read_directory(evidence.root)["source_evidence"])
    record = evidence.record_observation(provenance=provenance(), category="repository_structure",
        statement="Observed immediate checkout entries", observation={}, sources=(captured,))
    assert evidence.get(record.ref).freshness == "current"
    (evidence.root / "new.py").write_text("new = True")
    assert evidence.get(record.ref).freshness == "stale"


def test_task_row_compare_and_set_rejects_lost_updates(evidence):
    repository = WorkspaceRepository(evidence.db.workspace_id, db=evidence.db)
    task = TaskBoardItem(task_id="task-cas", parent_task_id=None, root_task_id="task-cas", title="Inspect",
        user_request="Inspect handler", normalized_goal="Inspect handler", status=TaskStatus.NEW,
        priority=100, risk_level=RiskLevel.LOW)
    repository.save_task(task)
    first, second = repository.get_task(task.task_id), repository.get_task(task.task_id)
    first.title = "First writer"
    repository.save_task(first)
    second.title = "Stale writer"
    with pytest.raises(EvidenceConflict):
        repository.save_task(second)
    assert repository.get_task(task.task_id).title == "First writer"


def test_pagination_does_not_skip_records_when_context_is_full(evidence):
    expected = {record.evidence_id for record in observe_distinct_files(evidence, 8)}
    budget = evidence.query(EvidenceQuery(limit=1))["context_bytes"]
    offset, seen = 0, set()
    while True:
        page = evidence.query(EvidenceQuery(limit=3, offset=offset, max_context_tokens=budget))
        record, = page["records"]
        assert record["evidence_id"] not in seen
        seen.add(record["evidence_id"])
        assert page["context_bytes"] <= budget
        assert page["next_offset"] == offset + 1
        if not page["has_more"]:
            break
        offset = page["next_offset"]
    assert seen == expected


def test_detail_pages_preserve_utf8_and_bound_source_context(evidence):
    import json
    (evidence.root / "app.py").write_text("value = 'مشاهده'\n")
    observed = observe(evidence)
    text = ""
    offset = 0
    while True:
        page = evidence.detail(EvidenceDetailQuery(**observed.ref.model_dump(), offset=offset, max_bytes=256))
        assert len(page["content"].encode()) <= 256
        text += page["content"]
        if not page["has_more"]:
            break
        offset = page["next_offset"]
    assert json.loads(text)["provenance"]["invocation_id"] == observed.provenance.invocation_id


def test_requirements_enforce_actual_tools_and_supporting_observations(evidence):
    observed = observe(evidence)
    finding = inference(evidence, observed)
    requirement = EvidenceRequirement(requirement_id="behavior", objective="Inspect handler",
        acceptable_classifications=("inferred",), required_tools=("read_file",), paths=("app.py",),
        acceptance_criteria=("Cite actual file read",), evidence=(finding.ref,))
    assert evidence.satisfies(requirement)
    assert not evidence.satisfies(requirement.model_copy(update={"required_tools": ("run_tests",)}))


def test_equivalent_concurrent_findings_keep_one_identity_and_both_task_links(evidence):
    from concurrent.futures import ThreadPoolExecutor
    observed = observe(evidence)
    candidate = EvidenceCandidate(category="code_behavior", statement="Handler returns one", classification="inferred",
        confidence=0.9, sources=observed.sources, supporting_evidence=(observed.ref,))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(evidence.submit_findings, candidate, provenance=provenance(task)) for task in ("a", "b")]
        records = [future.result() for future in futures]
    assert records[0].ref == records[1].ref
    for task in ("a", "b"):
        assert evidence.query(EvidenceQuery(task_id=task))["records"][0]["evidence_id"] == records[0].evidence_id


def test_symlink_to_protected_source_is_rejected(evidence):
    (evidence.root / ".env").write_text("PRIVATE_VALUE=not-for-models")
    (evidence.root / "public.py").symlink_to(evidence.root / ".env")
    with pytest.raises(PermissionError, match="cannot use symlinks"):
        read_source(evidence.root, "public.py")


def test_v1_workspace_migration_preserves_legacy_tasks(tmp_path):
    import sqlite3
    from mana_agent.persistence.workspace_db import SCHEMA_V1_STATEMENTS
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    for statement in SCHEMA_V1_STATEMENTS:
        conn.execute(statement)
    conn.execute("INSERT INTO schema_migrations VALUES (1, '2026-11-17', 'initial_normalized_workspace_schema', '{}')")
    conn.execute("""INSERT INTO tasks(task_id,root_task_id,title,user_request,normalized_goal,status,workspace_id,created_at,updated_at)
        VALUES ('legacy','legacy','Saved task','Saved objective','Saved objective','new','legacy-workspace',
                '2026-11-17T00:00:00+00:00','2026-11-17T00:00:00+00:00')""")
    conn.commit()
    conn.close()
    db = WorkspaceDatabase("legacy-workspace", db_path=path)
    try:
        task = WorkspaceRepository(db.workspace_id, db=db).get_task("legacy")
        assert task.user_request == "Saved objective"
        assert task.row_version == 0
        assert task.task_kind == "general" and task.evidence_refs == []
        with db.connect() as conn:
            assert conn.execute("SELECT version FROM schema_migrations WHERE version=2").fetchone()
            assert conn.execute("SELECT COUNT(*) FROM repository_evidence").fetchone()[0] == 0
    finally:
        db.close()
