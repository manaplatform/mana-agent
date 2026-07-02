from __future__ import annotations

from pathlib import Path

from mana_agent.llm.tool_worker_process import ToolRunResponse
from mana_agent.llm.goal_profiles import ModelDocsGoalProfile, active_goal_profile
from mana_agent.llm.agent_work_queue import QueueManager
from mana_agent.llm.tools_manager import (
    RunStateStore,
    ToolsPlan,
    ToolsPlanStep,
    _forced_mutation_prompt,
    _missing_required_files,
    _mutation_fallback_tool_allowed,
    _required_file_satisfied,
    _resolve_required_deliverables,
)
from mana_agent.services.coding_memory_service import CodingMemoryService


class _NoopWorker:
    requests: list[object]

    def __init__(self) -> None:
        self.requests = []

    def run_tools(self, _request, on_event=None):  # noqa: ANN001
        _ = on_event
        self.requests.append(_request)
        return ToolRunResponse(answer="ok", sources=[], mode="agent-tools", trace=[], warnings=[])


def test_no_fabricated_content_when_worker_writes_nothing(tmp_path: Path) -> None:
    # There is no template fallback: a worker that authors nothing yields an
    # honest failure rather than fabricated boilerplate. The deliverable is not
    # created and the run is blocked.
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("# Demo\n\nA small CLI project.\n", encoding="utf-8")
    worker = _NoopWorker()
    manager = QueueManager(worker_client=worker, repo_root=tmp_path)

    result = manager.run(
        request="analyze project and create a analyze.md in docs",
        index_dir=str(tmp_path / ".mana" / "index"),
        requires_edit=True,
        target_files=[],
        pass_cap=1,
        max_steps=1,
    )

    assert not (tmp_path / "docs" / "analyze.md").exists()
    assert result.run_status == "blocked"
    decision = result.planner_decisions[0]
    assert decision["verification_passed"] is False
    assert "docs/analyze.md" in decision["missing_required_files"]


def test_discovery_pass_does_not_run_under_mutation_required(tmp_path: Path) -> None:
    # The first (discovery) pass must NOT be strict-mutation: the agent must be
    # free to read/search before it authors anything. Only the edit pass carries
    # mutation_required. With nothing authored, the run fails honestly.
    (tmp_path / "src" / "mana_agent").mkdir(parents=True)
    (tmp_path / "src" / "mana_agent" / "app.py").write_text("def main():\n    return 'ok'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Old\n", encoding="utf-8")

    class _StrictAwareWorker:
        policies: list[dict]

        def __init__(self) -> None:
            self.policies = []

        def run_tools(self, request, on_event=None):  # noqa: ANN001
            _ = on_event
            policy = dict(request.tool_policy or {})
            self.policies.append(policy)
            if (request.tool_name or "") == "repo_search" and policy.get("mutation_required"):
                raise AssertionError("discovery must not run with mutation_required")
            return ToolRunResponse(answer="ok", sources=[], mode="agent-tools", trace=[], warnings=[])

    worker = _StrictAwareWorker()
    result = QueueManager(worker_client=worker, repo_root=tmp_path).run(
        request="analyze whole project,and update readme.md",
        index_dir=str(tmp_path / ".mana" / "index"),
        requires_edit=True,
        target_files=[],
        pass_cap=1,
        max_steps=1,
    )

    # Worker authored nothing, so the stub README is not a satisfied deliverable.
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "# Old\n"
    assert result.run_status == "blocked"
    assert worker.policies[0].get("mutation_required") is None


def test_edit_pass_can_read_and_search_to_ground_content(tmp_path: Path) -> None:
    # The edit/forced pass is agentic: its policy must expose read/search tools so
    # the worker can analyze the project before authoring (not just mutation tools).
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    body = "# Overview\n\n" + ("Grounded overview content for the demo project. " * 6) + "\n"

    class _PolicyCapturingWorker:
        def __init__(self) -> None:
            self.edit_policies: list[dict] = []

        def run_tools(self, request, on_event=None):  # noqa: ANN001
            _ = on_event
            policy = dict(request.tool_policy or {})
            if policy.get("mutation_required"):
                self.edit_policies.append(policy)
            path = str((request.tool_args or {}).get("path") or "")
            if path == "docs/overview.md":
                (tmp_path / path).write_text(body, encoding="utf-8")
                return ToolRunResponse(
                    answer="wrote overview",
                    sources=[],
                    mode="agent-tools",
                    trace=[{"tool_name": "create_file", "status": "ok", "files_changed": [path]}],
                    warnings=[],
                )
            return ToolRunResponse(answer="ok", sources=[], mode="agent-tools", trace=[], warnings=[])

    worker = _PolicyCapturingWorker()
    QueueManager(worker_client=worker, repo_root=tmp_path).run(
        request="create overview.md in docs",
        index_dir=str(tmp_path / ".mana" / "index"),
        requires_edit=True,
        pass_cap=1,
        max_steps=1,
    )

    assert worker.edit_policies, "expected at least one mutation-required edit pass"
    allowed = set(worker.edit_policies[0].get("allowed_tools") or [])
    assert {"read_file", "repo_search"}.issubset(allowed)
    assert {"edit_file", "multi_edit_file", "create_file", "write_file", "apply_patch", "delete_file"}.issubset(allowed)


def test_mutation_fallback_allowlist_blocks_discovery_tools() -> None:
    for tool in ("repo_search", "read_file", "ls", "list_files"):
        assert _mutation_fallback_tool_allowed(tool, target_exists=False, prior_target_evidence=True) is False
    for tool in ("edit_file", "multi_edit_file", "create_file", "write_file", "apply_patch", "delete_file"):
        assert _mutation_fallback_tool_allowed(tool, target_exists=False, prior_target_evidence=True) is True
    assert _mutation_fallback_tool_allowed("read_file", target_exists=True, prior_target_evidence=False) is True


def test_run_state_model_docs_queue_prioritizes_relevant_files(tmp_path: Path) -> None:
    (tmp_path / "src" / "billing").mkdir(parents=True)
    (tmp_path / "src" / "billing" / "models.py").write_text("from django.db import models\n", encoding="utf-8")
    (tmp_path / "src" / "billing" / "invoice_models.py").write_text("class Invoice(BaseModel):\n    pass\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "models.md").write_text("# Models\n", encoding="utf-8")
    (tmp_path / "Front").mkdir()
    (tmp_path / "Front" / "package-lock.json").write_text("{}", encoding="utf-8")
    (tmp_path / "src" / "billing" / "migrations").mkdir()
    (tmp_path / "src" / "billing" / "migrations" / "0001_initial.py").write_text(
        "migrations.CreateModel(name='Invoice')\n",
        encoding="utf-8",
    )

    store = RunStateStore(repo_root=tmp_path, run_id="queue-test")
    store.ensure(goal="find all models and update docs/models.md")
    store.seed_candidate_queue()

    pending = store.read_json("todo.json", {})["pending_file_reads"]
    assert pending[:3] == [
        "src/billing/invoice_models.py",
        "src/billing/models.py",
        "docs/models.md",
    ]
    assert "Front/package-lock.json" not in pending
    assert "src/billing/migrations/0001_initial.py" not in pending


def test_model_docs_goal_profile_matching() -> None:
    assert active_goal_profile("find all models and update docs/models.md").id == "model_docs"  # type: ignore[union-attr]
    assert active_goal_profile("update models documentation").id == "model_docs"  # type: ignore[union-attr]
    assert active_goal_profile("summarize the auth middleware") is None


def test_model_docs_goal_profile_candidate_priority_and_excludes(tmp_path: Path) -> None:
    profile = ModelDocsGoalProfile()
    (tmp_path / "src" / "app").mkdir(parents=True)
    (tmp_path / "src" / "app" / "models.py").write_text("from django.db import models\n", encoding="utf-8")
    (tmp_path / "src" / "app" / "foo_models.py").write_text("class Item(BaseModel):\n    pass\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "models.md").write_text("# Models\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "models.py").write_text("class Bad(BaseModel):\n    pass\n", encoding="utf-8")
    (tmp_path / "build" / "lib" / "src" / "app").mkdir(parents=True)
    (tmp_path / "build" / "lib" / "src" / "app" / "models.py").write_text("class Bad(BaseModel):\n    pass\n", encoding="utf-8")

    assert profile.priority("src/app/models.py", tmp_path) == 1
    assert profile.priority("src/app/foo_models.py", tmp_path) == 1
    assert profile.is_relevant("docs/models.md", tmp_path)
    assert profile.is_relevant("src/app/models.py", tmp_path)
    assert not profile.is_relevant("node_modules/models.py", tmp_path)
    assert not profile.is_relevant("build/lib/src/app/models.py", tmp_path)
    assert not profile.is_relevant("package-lock.json", tmp_path)


def test_run_state_profile_sorting_and_generic_goal_behavior(tmp_path: Path) -> None:
    (tmp_path / "src" / "app").mkdir(parents=True)
    (tmp_path / "src" / "app" / "models.py").write_text("from django.db import models\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "models.md").write_text("# Models\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Readme\n", encoding="utf-8")
    store = RunStateStore(repo_root=tmp_path, run_id="profile-sort")
    store.ensure(goal="update docs/models.md")

    assert store._sort_pending_reads(["README.md", "src/app/models.py", "docs/models.md"], goal="update docs/models.md") == [
        "src/app/models.py",
        "docs/models.md",
    ]
    assert store._sort_pending_reads(["README.md", "src/app/models.py"], goal="summarize files") == [
        "README.md",
        "src/app/models.py",
    ]
    assert not hasattr(store, "_is_model_docs_goal")


def test_todo_edit_cannot_complete_with_empty_modified_files(tmp_path: Path) -> None:
    store = RunStateStore(repo_root=tmp_path, run_id="todo-edit-empty")
    store.ensure(goal="update docs/models.md", flow_id="")
    todo = store.current_todo_for_gate("apply_changes", goal="update docs/models.md")
    worker_done = store.mark_todo_worker_done(
        todo,
        tool_name="apply_patch",
        files_changed=[],
        tool_call_id="tool-1",
    )

    result = store.confirm_or_reject_todo(worker_done, files_changed=[])

    assert result.status == "failed"
    assert result.worker_checked is False
    assert result.agent_confirmed is False
    assert "no target file was modified" in result.reason


def test_todo_edit_confirms_only_when_target_file_modified(tmp_path: Path) -> None:
    store = RunStateStore(repo_root=tmp_path, run_id="todo-edit-target")
    store.ensure(goal="update docs/models.md", flow_id="")
    todo = store.current_todo_for_gate("apply_changes", goal="update docs/models.md")
    worker_done = store.mark_todo_worker_done(
        todo,
        tool_name="write_file",
        files_changed=["docs/models.md"],
        tool_call_id="tool-2",
    )

    result = store.confirm_or_reject_todo(worker_done, files_changed=["docs/models.md"])

    assert result.status == "agent_confirmed"
    assert result.worker_checked is True
    assert result.agent_confirmed is True


def test_todo_worker_cannot_use_tool_outside_allowed_tools(tmp_path: Path) -> None:
    store = RunStateStore(repo_root=tmp_path, run_id="todo-disallowed")
    store.ensure(goal="update docs/models.md", flow_id="")
    todo = store.current_todo_for_gate("apply_changes", goal="update docs/models.md")
    worker_done = store.mark_todo_worker_done(
        todo,
        tool_name="repo_search",
        tool_call_id="tool-3",
    )

    result = store.confirm_or_reject_todo(worker_done, files_changed=["docs/models.md"])

    assert result.status == "failed"
    assert "disallowed tool" in result.reason


def test_todo_board_prints_worker_and_agent_state(tmp_path: Path) -> None:
    store = RunStateStore(repo_root=tmp_path, run_id="todo-board")
    store.ensure(goal="update docs/models.md", flow_id="")
    todo = store.current_todo_for_gate("apply_changes", goal="update docs/models.md")
    failed = store.mark_todo_worker_done(todo, tool_name="apply_patch", tool_call_id="tool-4")
    store.confirm_or_reject_todo(failed, files_changed=[])

    board = store.todo_board()

    assert "Todo Board:" in board
    assert "[!][ ] update_docs_models_md - failed" in board


def test_planner_cannot_confirm_edit_without_worker_modified_proof(tmp_path: Path) -> None:
    store = RunStateStore(repo_root=tmp_path, run_id="todo-contradiction")
    store.ensure(goal="update docs/models.md", flow_id="")
    todo = store.current_todo_for_gate("apply_changes", goal="update docs/models.md")
    store._write_todo_item(
        todo.model_copy(
            update={
                "status": "agent_confirmed",
                "worker_checked": True,
                "agent_confirmed": True,
                "proof": {"tool_name": "apply_patch", "modified_files": []},
            }
        )
    )

    warnings = store.validate_planner_todo_claims(changed_files=[])
    result = store.current_todo_for_gate("apply_changes", goal="update docs/models.md")

    assert warnings == ["planner_contradiction_edit_without_modified_target"]
    assert result.status == "failed"


def test_polluted_candidate_paths_are_rejected_unless_real_repo_files(tmp_path: Path) -> None:
    (tmp_path / "src" / "app").mkdir(parents=True)
    (tmp_path / "src" / "app" / "models.py").write_text("class User(BaseModel):\n    pass\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "models.md").write_text("# Models\n", encoding="utf-8")
    store = RunStateStore(repo_root=tmp_path, run_id="polluted")
    store.ensure(goal="create docs/models.md for all models", flow_id="")

    response = ToolRunResponse(
        answer="apply_patch.py docsmodels.md docs/models.md src/app/models.py /tmp/outside/models.py",
        trace=[{"tool_name": "repo_search", "status": "ok", "output_preview": "apply_patch.py docsmodels.md"}],
    )
    counts = store.record_evidence_from_response(
        gate="locate_candidates",
        source_tool="repo_search",
        response=response,
    )

    pending = store.read_json("todo.json")["pending_file_reads"]
    assert counts["discovered"] == 2
    assert pending == ["src/app/models.py", "docs/models.md"]
    assert "apply_patch.py" not in pending
    assert "docsmodels.md" not in pending


def test_run_state_action_fingerprint_ignores_planner_prose(tmp_path: Path) -> None:
    store = RunStateStore(repo_root=tmp_path, run_id="fp-test")
    first = store.fingerprint(
        gate="locate_candidates",
        tool_name="repo_search",
        args={"question": "Planner pass 1: repo_search query='class models.Model' glob='back/**/*.py'"},
        filters={"k": 8},
    )
    second = store.fingerprint(
        gate="read_candidates",
        tool_name="repo_search",
        args={"question": "Fallback request. repo_search query='class models.Model' glob='back/**/*.py'"},
        filters={"k": 50},
    )
    assert first == second


def test_run_state_located_not_read_then_successful_read_updates_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "src" / "app").mkdir(parents=True)
    (tmp_path / "src" / "app" / "models.py").write_text("class User(BaseModel):\n    pass\n", encoding="utf-8")
    store = RunStateStore(repo_root=tmp_path, run_id="read-reconcile")
    store.ensure(goal="read model docs", flow_id="")
    store.write_json(
        "todo.json",
        {"pending_file_reads": ["src/app/models.py"], "pending_edits": [], "verification_status": "pending"},
    )
    store.append_jsonl(
        "evidence.jsonl",
        {
            "timestamp": "2026-06-22T00:00:00+00:00",
            "file_path": "src/app/models.py",
            "status": "located_not_read",
            "evidence_type": "candidate_file",
        },
    )

    counts = store.record_evidence_from_response(
        gate="read_candidates",
        source_tool="read_file",
        response=ToolRunResponse(
            answer="read src/app/models.py",
            trace=[{"tool_name": "read_file", "path": "src/app/models.py", "status": "ok"}],
        ),
    )
    store.record_tool_call(
        gate="read_candidates",
        tool_name="read_file",
        normalized_args={"path": "src/app/models.py"},
        fingerprint="read-src-app-models",
        status="ok",
        files_read=["src/app/models.py"],
    )
    store.update_state(status="needs_resume", next_action=store.next_action())
    store.write_checkpoint(
        status="needs_resume",
        completed_gates=["locate_candidates", "read_candidates"],
        pending_gates=["classify_evidence"],
        files_changed=[],
        verification_status="pending",
    )

    checkpoint = store.read_json("checkpoint.json")
    ledger = store.read_json("work_ledger.json")
    state = store.read_json("state.json")
    summary = (store.run_dir / "summary.md").read_text(encoding="utf-8")
    resume_prompt = (store.run_dir / "resume_prompt.md").read_text(encoding="utf-8")
    assert counts["read"] == 1
    assert store.read_files() == {"src/app/models.py"}
    assert checkpoint["read_files"] == ["src/app/models.py"]
    assert checkpoint["progress_counters"]["files_read"] == 1
    assert checkpoint["pending_files"] == []
    assert ledger["read_files"] == checkpoint["read_files"]
    assert ledger["pending_work"]["pending_file_reads"] == []
    assert state["next_action"] == checkpoint["next_action"] == ledger["next_action"]
    assert "- files_read: 1" in summary
    assert f"Next action: {checkpoint['next_action']}" in resume_prompt


def test_run_state_skipped_no_progress_then_retry_read_wins(tmp_path: Path) -> None:
    (tmp_path / "src" / "app").mkdir(parents=True)
    (tmp_path / "src" / "app" / "models.py").write_text("class User(BaseModel):\n    pass\n", encoding="utf-8")
    store = RunStateStore(repo_root=tmp_path, run_id="retry-read")
    store.ensure(goal="read retry", flow_id="")
    store.write_json(
        "todo.json",
        {"pending_file_reads": ["src/app/models.py"], "pending_edits": [], "verification_status": "pending"},
    )
    store.mark_read_skipped("src/app/models.py", reason="read_file_no_files_read")

    counts = store.record_evidence_from_response(
        gate="read_candidates",
        source_tool="read_file",
        response=ToolRunResponse(
            answer="retry read src/app/models.py",
            trace=[{"tool_name": "read_file", "path": "src/app/models.py", "status": "ok"}],
        ),
    )
    store.record_tool_call(
        gate="read_candidates",
        tool_name="read_file",
        normalized_args={"path": "src/app/models.py"},
        fingerprint="retry-src-app-models",
        status="ok",
        files_read=["src/app/models.py"],
    )
    store.write_checkpoint(
        status="needs_resume",
        completed_gates=["locate_candidates", "read_candidates"],
        pending_gates=["classify_evidence"],
        files_changed=[],
        verification_status="pending",
    )

    assert counts["read"] == 1
    assert store._latest_file_statuses()["src/app/models.py"] == "read"
    assert store.read_json("todo.json")["pending_file_reads"] == []
    assert store.read_json("visited_files.json")["files"] == ["src/app/models.py"]
    checkpoint = store.read_json("checkpoint.json")
    assert checkpoint["read_files"] == ["src/app/models.py"]
    assert checkpoint["progress_counters"]["files_read"] == 1


def test_run_state_tool_call_files_read_reconciles_evidence_and_visited(tmp_path: Path) -> None:
    store = RunStateStore(repo_root=tmp_path, run_id="tool-call-files-read")
    store.ensure(goal="read via tool call", flow_id="")
    store.write_json(
        "todo.json",
        {"pending_file_reads": ["src/app/models.py"], "pending_edits": [], "verification_status": "pending"},
    )

    store.record_tool_call(
        gate="read_candidates",
        tool_name="read_file",
        normalized_args={"path": "src/app/models.py"},
        fingerprint="read-tool-call-only",
        status="ok",
        files_read=["src/app/models.py"],
    )
    store.write_checkpoint(
        status="needs_resume",
        completed_gates=["locate_candidates", "read_candidates"],
        pending_gates=["classify_evidence"],
        files_changed=[],
        verification_status="pending",
    )

    assert store.read_files() == {"src/app/models.py"}
    assert store.read_json("todo.json")["pending_file_reads"] == []
    assert store.read_json("visited_files.json")["files"] == ["src/app/models.py"]
    checkpoint = store.read_json("checkpoint.json")
    assert checkpoint["read_files"] == ["src/app/models.py"]
    assert checkpoint["progress_counters"]["files_read"] == 1


def test_run_state_store_loop_detector_next_action_prefers_pending_read(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "models.py").write_text("class User(BaseModel):\n    pass\n", encoding="utf-8")
    store = RunStateStore(repo_root=tmp_path, run_id="loop")
    store.ensure(goal="loop", flow_id="")
    store.write_json("todo.json", {"pending_file_reads": ["a/models.py"], "pending_edits": [], "verification_status": "pending"})

    assert store.next_action() == "read_file a/models.py"


def test_run_state_apply_changes_not_completed_without_changed_files(tmp_path: Path) -> None:
    store = RunStateStore(repo_root=tmp_path, run_id="no-apply-complete")
    store.ensure(goal="update docs/models.md", flow_id="")
    plan = ToolsPlan(
        objective="Update docs/models.md",
        steps=[
            ToolsPlanStep(id="s1", title="Read", tool_intent="inspect"),
            ToolsPlanStep(id="s2", title="Patch docs", tool_intent="edit"),
            ToolsPlanStep(id="s3", title="Verify", tool_intent="verify"),
        ],
        current_step_id="s3",
        decision="continue",
    )

    state = store.update_state(
        plan=plan,
        step=plan.steps[2],
        status="needs_resume",
        changed_files=[],
    )

    assert state["current_gate"] == "apply_changes"
    assert "apply_changes" not in state["completed_gates"]
    assert "apply_changes" in state["pending_gates"]
    assert state["blocking_reason"] == "missing_edit_payload"


def test_run_state_plan_patch_not_completed_without_mutation_payload(tmp_path: Path) -> None:
    store = RunStateStore(repo_root=tmp_path, run_id="plan-patch-needs-payload")
    store.ensure(goal="update docs/models.md", flow_id="")
    plan = ToolsPlan(
        objective="Update docs/models.md",
        steps=[
            ToolsPlanStep(id="s1", title="Plan patch", tool_intent="edit"),
            ToolsPlanStep(id="s2", title="Verify", tool_intent="verify"),
        ],
        current_step_id="s2",
        decision="continue",
    )

    state = store.update_state(plan=plan, step=plan.steps[1], status="needs_resume", changed_files=[])

    assert "plan_patch" not in state["completed_gates"]
    assert "plan_patch" in state["pending_gates"]
    assert state["blocking_reason"] == "missing_edit_payload"


def test_run_state_read_file_action_key_includes_nested_tool_args_path(tmp_path: Path) -> None:
    key = RunStateStore.normalized_action_key(
        tool_name="read_file",
        args={
            "question": "Read pending candidate file before mutation: docs/coding-flows.md",
            "tool_args": {"path": "docs/coding-flows.md"},
        },
    )

    assert key == "read_file:docs/coding-flows.md"


def test_run_state_store_sanitizes_dependency_pending_reads(tmp_path: Path) -> None:
    store = RunStateStore(repo_root=tmp_path, run_id="deps")
    store.ensure(goal="loop", flow_id="")
    dependency_path = tmp_path / "back" / "venv" / "lib" / "python3.14" / "site-packages" / "django" / "forms" / "models.py"
    app_path = tmp_path / "back" / "billing" / "models.py"
    dependency_path.parent.mkdir(parents=True)
    dependency_path.write_text("class Dependency:\n    pass\n", encoding="utf-8")
    app_path.parent.mkdir(parents=True)
    app_path.write_text("class Invoice(BaseModel):\n    pass\n", encoding="utf-8")
    store.write_json(
        "todo.json",
        {
            "pending_file_reads": [str(dependency_path), str(app_path), "back/.venv/lib/python/site-packages/x/models.py"],
            "pending_edits": [],
            "verification_status": "pending",
        },
    )

    assert store.next_action() == "read_file back/billing/models.py"
    assert store.read_json("todo.json")["pending_file_reads"] == ["back/billing/models.py"]


def test_run_state_model_docs_queue_prioritizes_model_schema_files(tmp_path: Path) -> None:
    (tmp_path / "src" / "mana_agent").mkdir(parents=True)
    (tmp_path / "src" / "mana_agent" / "models.py").write_text("class User(BaseModel):\n    pass\n")
    (tmp_path / "src" / "mana_agent" / "schema_models.py").write_text("class Item(TypedDict):\n    pass\n")
    (tmp_path / "src" / "mana_agent" / "__init__.py").write_text("")
    (tmp_path / "src" / "mana_agent" / "commands").mkdir()
    (tmp_path / "src" / "mana_agent" / "commands" / "chat_cli.py").write_text("class Cli:\n    pass\n")
    (tmp_path / "src" / "mana_agent" / "tools").mkdir()
    (tmp_path / "src" / "mana_agent" / "tools" / "apply_patch.py").write_text("class Patch:\n    pass\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_models.py").write_text("from mana_agent.models import User\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "models.md").write_text("# Models\n")
    (tmp_path / "README.md").write_text("# Readme\n")

    store = RunStateStore(repo_root=tmp_path, run_id="rank")
    store.ensure(goal="create docs/models.md for all models", flow_id="")
    store.seed_candidate_queue()

    pending = store.read_json("todo.json")["pending_file_reads"]
    assert pending[:2] == ["src/mana_agent/models.py", "src/mana_agent/schema_models.py"]
    assert "src/mana_agent/commands/chat_cli.py" not in pending
    assert "src/mana_agent/tools/apply_patch.py" not in pending
    assert "src/mana_agent/__init__.py" not in pending
    assert pending[-1] == "docs/models.md"


def test_run_state_model_docs_profile_excludes_build_lib_and_broad_tests(tmp_path: Path) -> None:
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "models.py").write_text("class User(BaseModel):\n    pass\n")
    (tmp_path / "build" / "lib" / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "build" / "lib" / "src" / "pkg" / "models.py").write_text("class Built(BaseModel):\n    pass\n")
    (tmp_path / "dist" / "pkg").mkdir(parents=True)
    (tmp_path / "dist" / "pkg" / "models.py").write_text("class Dist(BaseModel):\n    pass\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_models.py").write_text("class Fixture(BaseModel):\n    pass\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "models.md").write_text("# Models\n")

    store = RunStateStore(repo_root=tmp_path, run_id="model-docs-excludes")
    store.ensure(goal="update docs/models.md for all models", flow_id="")
    store.seed_candidate_queue()

    pending = store.read_json("todo.json")["pending_file_reads"]
    assert pending == ["src/pkg/models.py", "docs/models.md"]
    assert not any(path.startswith(("build/", "dist/", "tests/")) for path in pending)


def test_run_state_model_docs_profile_does_not_queue_unrelated_utility_files(tmp_path: Path) -> None:
    (tmp_path / "src" / "pkg" / "commands").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "models.py").write_text("class User(BaseModel):\n    pass\n")
    (tmp_path / "src" / "pkg" / "commands" / "sync.py").write_text("class SyncCommand:\n    pass\n")
    (tmp_path / "src" / "pkg" / "utils.py").write_text("def normalize(value):\n    return str(value).strip()\n")
    (tmp_path / "src" / "pkg" / "typed_payload.py").write_text("class Payload(TypedDict):\n    id: str\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "models.md").write_text("# Models\n")

    store = RunStateStore(repo_root=tmp_path, run_id="model-docs-utilities")
    store.ensure(goal="update docs/models.md for all models", flow_id="")
    store.seed_candidate_queue()

    pending = store.read_json("todo.json")["pending_file_reads"]
    assert "src/pkg/models.py" in pending
    assert "src/pkg/typed_payload.py" in pending
    assert "src/pkg/commands/sync.py" not in pending
    assert "src/pkg/utils.py" not in pending


def test_run_state_model_docs_goal_accepts_create_in_docs_wording(tmp_path: Path) -> None:
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "models.py").write_text("class User(BaseModel):\n    pass\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "coding-flows.md").write_text("# Flows\n")
    (tmp_path / "docs" / "project_structure_analysis.json").write_text("{}\n")
    (tmp_path / "docs" / "models.md").write_text("# Models\n")

    store = RunStateStore(repo_root=tmp_path, run_id="natural-model-docs")
    store.ensure(
        goal="create in docs a models.md and add a document for all models exist in this project.",
        flow_id="",
    )
    store.seed_candidate_queue()

    assert store.read_json("todo.json")["pending_file_reads"] == [
        "src/pkg/models.py",
        "docs/models.md",
    ]


def _edit_plan() -> ToolsPlan:
    return ToolsPlan(
        objective="create docs/analyze.md",
        steps=[
            ToolsPlanStep(id="s1", title="inspect structure", tool_intent="inspect"),
            ToolsPlanStep(id="s2", title="inspect src", tool_intent="inspect"),
            ToolsPlanStep(id="s3", title="read tests", tool_intent="search"),
            ToolsPlanStep(id="s4", title="write analyze.md", tool_intent="edit"),
            ToolsPlanStep(id="s5", title="verify file", tool_intent="verify"),
        ],
        current_step_id="s1",
        decision="continue",
    )


def _inspect_only_plan() -> ToolsPlan:
    return ToolsPlan(
        objective="explain the project",
        steps=[
            ToolsPlanStep(id="s1", title="inspect", tool_intent="inspect"),
            ToolsPlanStep(id="s2", title="answer", tool_intent="answer"),
        ],
        current_step_id="s1",
        decision="continue",
    )


# ---------------------------------------------------------------------------
# Strict multi-file mutation workflow (required deliverables + verification)
# ---------------------------------------------------------------------------


class _OneFileWorker:
    """Worker that writes exactly one of the requested files, then goes idle.

    Simulates partial success: the LLM produces only ``made_path`` and the
    supervisor must create the remaining required file deterministically.
    """

    def __init__(self, repo_root: Path, made_path: str, content: str = "installation guide body\n") -> None:
        self.repo_root = Path(repo_root)
        self.made_path = made_path
        self.content = content
        self.made = False
        self.questions: list[str] = []

    def run_tools(self, request, on_event=None):  # noqa: ANN001
        _ = on_event
        self.questions.append(str(getattr(request, "question", "") or ""))
        if not self.made:
            self.made = True
            target = self.repo_root / self.made_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.content, encoding="utf-8")
            return ToolRunResponse(
                answer="created installation guide",
                sources=[],
                mode="agent-tools",
                trace=[{"tool_name": "create_file", "status": "ok", "files_changed": [self.made_path]}],
                warnings=[],
            )
        return ToolRunResponse(answer="ok", sources=[], mode="agent-tools", trace=[], warnings=[])


def test_resolve_required_deliverables_handles_and_and_ampersand(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    assert _resolve_required_deliverables("create 01-overview.md & 02-installation.md in docs", tmp_path) == [
        "docs/01-overview.md",
        "docs/02-installation.md",
    ]
    assert _resolve_required_deliverables("create 01-overview.md and 02-installation.md in docs", tmp_path) == [
        "docs/01-overview.md",
        "docs/02-installation.md",
    ]


def test_two_files_authored_under_docs_complete_the_run(tmp_path: Path) -> None:
    # Agentic success: a worker that authors substantive, correctly-placed content
    # for every deliverable completes the run. No templates involved.
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")

    overview_body = "# Overview\n\n" + ("This project is a demo CLI for analysis. " * 8) + "\n"
    install_body = "# Installation\n\n" + ("Run pip install -e . to set up the project. " * 6) + "\n"
    bodies = {"docs/01-overview.md": overview_body, "docs/02-installation.md": install_body}

    class _TwoFileAuthoringWorker:
        def __init__(self) -> None:
            self.seen: set[str] = set()

        def run_tools(self, request, on_event=None):  # noqa: ANN001
            _ = on_event
            path = str((request.tool_args or {}).get("path") or "")
            if path in bodies and path not in self.seen:
                self.seen.add(path)
                target = tmp_path / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(bodies[path], encoding="utf-8")
                return ToolRunResponse(
                    answer=f"wrote {path}",
                    sources=[],
                    mode="agent-tools",
                    trace=[{"tool_name": "create_file", "status": "ok", "files_changed": [path]}],
                    warnings=[],
                )
            return ToolRunResponse(answer="ok", sources=[], mode="agent-tools", trace=[], warnings=[])

    result = QueueManager(worker_client=_TwoFileAuthoringWorker(), repo_root=tmp_path).run(
        request="create 01-overview.md & 02-installation.md in docs",
        index_dir=str(tmp_path / ".mana" / "index"),
        requires_edit=True,
        pass_cap=1,
        max_steps=1,
    )

    assert (tmp_path / "docs" / "01-overview.md").read_text(encoding="utf-8") == overview_body
    assert (tmp_path / "docs" / "02-installation.md").read_text(encoding="utf-8") == install_body
    # The wrong (repo-root) paths must NOT be created.
    assert not (tmp_path / "01-overview.md").exists()
    assert not (tmp_path / "02-installation.md").exists()
    assert result.run_status == "completed"
    assert result.changed_files == ["docs/01-overview.md", "docs/02-installation.md"]
    decision = result.planner_decisions[0]
    assert decision["required_files"] == ["docs/01-overview.md", "docs/02-installation.md"]
    assert decision["missing_required_files"] == []
    assert decision["verification_passed"] is True


def test_partial_success_blocks_on_missing_file(tmp_path: Path) -> None:
    # Honest partial failure: the worker authors 02 but never 01. The run is
    # blocked on the missing deliverable; nothing is fabricated for 01.
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    install_body = "# Installation\n\n" + ("Install the demo project with pip. " * 6) + "\n"
    worker = _OneFileWorker(tmp_path, made_path="docs/02-installation.md", content=install_body)

    result = QueueManager(worker_client=worker, repo_root=tmp_path).run(
        request="create 01-overview.md and 02-installation.md in docs",
        index_dir=str(tmp_path / ".mana" / "index"),
        requires_edit=True,
        pass_cap=1,
        max_steps=1,
    )

    # Worker produced only 02; 01 is never fabricated, so the run fails honestly.
    assert (tmp_path / "docs" / "02-installation.md").exists()
    assert not (tmp_path / "docs" / "01-overview.md").exists()
    assert result.run_status == "blocked"
    assert result.planner_decisions[0]["missing_required_files"] == ["docs/01-overview.md"]


def test_missing_required_file_blocks_completion(tmp_path: Path) -> None:
    # "refactor" intent is a mutation but not a create/analysis artifact, so the
    # deterministic generator declines and the missing file cannot be produced.
    worker = _NoopWorker()

    result = QueueManager(worker_client=worker, repo_root=tmp_path).run(
        request="refactor docs/01-overview.md and docs/02-installation.md in docs",
        index_dir=str(tmp_path / ".mana" / "index"),
        requires_edit=True,
        pass_cap=1,
        max_steps=1,
    )

    assert result.run_status == "blocked"
    assert not (tmp_path / "docs" / "01-overview.md").exists()
    decision = result.planner_decisions[0]
    assert decision["verification_passed"] is False
    assert set(decision["missing_required_files"]) == {
        "docs/01-overview.md",
        "docs/02-installation.md",
    }


def test_wrong_path_at_repo_root_does_not_satisfy_docs_requirement(tmp_path: Path) -> None:
    # File created at repo root must NOT satisfy a docs/ deliverable.
    body = "# Overview\n\n" + ("Real substantive overview content for the project. " * 6) + "\n"
    (tmp_path / "01-overview.md").write_text(body, encoding="utf-8")
    assert _required_file_satisfied(tmp_path, "01-overview.md") is True
    assert _required_file_satisfied(tmp_path, "docs/01-overview.md") is False
    assert _missing_required_files(tmp_path, ["docs/01-overview.md"]) == ["docs/01-overview.md"]


def test_empty_file_does_not_satisfy_requirement(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "01-overview.md").write_text("", encoding="utf-8")
    assert _required_file_satisfied(tmp_path, "docs/01-overview.md") is False


def test_forced_mutation_prompt_drives_agentic_authoring() -> None:
    prompt = _forced_mutation_prompt("create docs/01-overview.md", "docs/01-overview.md")
    assert "ANALYZING THIS PROJECT" in prompt
    # It must point the worker at read/search tools, not just mutation tools.
    assert "read_file" in prompt and "repo_search" in prompt
    assert "edit_file" in prompt and "multi_edit_file" in prompt
    assert "create_file" in prompt and "write_file" in prompt and "apply_patch" in prompt and "delete_file" in prompt
    # No placeholders/boilerplate — there is no template fallback behind it.
    assert "Do NOT write placeholders" in prompt
    assert "Target file: docs/01-overview.md" in prompt
    assert "User request: create docs/01-overview.md" in prompt


class _WrongPathStubWorker:
    """Reproduces the live bug: writes stub files at the repo ROOT (not docs/).

    Mirrors the observed failure where the worker, after repeated
    ``tools_only_violation`` errors, finally calls write_file with bare
    filenames and ``TODO:`` placeholder content.
    """

    def __init__(self, repo_root: Path, names: list[str]) -> None:
        self.repo_root = Path(repo_root)
        self.names = names
        self.done = False

    def run_tools(self, request, on_event=None):  # noqa: ANN001
        _ = on_event
        if self.done:
            return ToolRunResponse(answer="ok", sources=[], mode="agent-tools", trace=[], warnings=[])
        self.done = True
        rows = []
        for name in self.names:
            target = self.repo_root / name  # bare name -> repo root (WRONG)
            target.write_text(f"# {name}\n\nTODO: add content\n", encoding="utf-8")
            rows.append({"tool_name": "write_file", "status": "ok", "files_changed": [name]})
        return ToolRunResponse(answer="wrote stubs", sources=[], mode="agent-tools", trace=rows, warnings=[])


def test_wrong_path_stub_is_not_rescued(tmp_path: Path) -> None:
    # A stub written to the wrong path is neither relocated nor regenerated: there
    # is no template fallback, and stub content is not worth relocating. The
    # deliverable stays missing and the run is blocked (honest failure).
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("# Demo\n\nA small CLI.\n", encoding="utf-8")
    worker = _WrongPathStubWorker(tmp_path, names=["01-overview.md", "02-installation.md"])

    result = QueueManager(worker_client=worker, repo_root=tmp_path).run(
        request="create 01-overview.md & 02-installation.md in docs",
        index_dir=str(tmp_path / ".mana" / "index"),
        requires_edit=True,
        pass_cap=1,
        max_steps=1,
    )

    # The docs/ deliverables were never authored with real content.
    assert not (tmp_path / "docs" / "01-overview.md").exists()
    assert not (tmp_path / "docs" / "02-installation.md").exists()
    # The stub stays where the worker put it; it is not promoted into docs/.
    assert (tmp_path / "01-overview.md").exists()
    assert result.run_status == "blocked"
    decision = result.planner_decisions[0]
    assert set(decision["missing_required_files"]) == {
        "docs/01-overview.md",
        "docs/02-installation.md",
    }


def test_substantial_wrong_path_content_is_relocated_not_discarded(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    rich = "# Project Overview\n\n" + ("This project does many useful things. " * 20) + "\n"

    class _RichWrongPathWorker:
        def __init__(self) -> None:
            self.done = False

        def run_tools(self, request, on_event=None):  # noqa: ANN001
            _ = on_event
            if self.done:
                return ToolRunResponse(answer="ok", sources=[], mode="agent-tools", trace=[], warnings=[])
            self.done = True
            (tmp_path / "01-overview.md").write_text(rich, encoding="utf-8")
            return ToolRunResponse(
                answer="wrote overview",
                sources=[],
                mode="agent-tools",
                trace=[{"tool_name": "write_file", "status": "ok", "files_changed": ["01-overview.md"]}],
                warnings=[],
            )

    result = QueueManager(worker_client=_RichWrongPathWorker(), repo_root=tmp_path).run(
        request="create 01-overview.md in docs",
        index_dir=str(tmp_path / ".mana" / "index"),
        requires_edit=True,
        pass_cap=1,
        max_steps=1,
    )

    # The agent's real content is preserved, just moved into docs/.
    assert (tmp_path / "docs" / "01-overview.md").read_text(encoding="utf-8") == rich
    assert not (tmp_path / "01-overview.md").exists()
    assert result.run_status == "completed"
    assert result.changed_files == ["docs/01-overview.md"]




def test_preview_plan_deterministic_materializes_todos(tmp_path: Path) -> None:
    # With no decision provider attached, preview_plan must still produce a
    # memory-backed checklist and connect it to the todo ledger.
    (tmp_path / "docs").mkdir()
    memory = CodingMemoryService(project_root=tmp_path)
    manager = QueueManager(
        worker_client=_NoopWorker(),
        repo_root=tmp_path,
        coding_memory_service=memory,
    )

    payload = manager.preview_plan(request="create overview.md in docs")

    assert payload["prechecklist_source"] == "deterministic"
    assert payload["requires_edit"] is True
    assert payload["flow_id"]
    # The plan was synced into the durable todo ledger for the flow.
    todos = payload["todos"]
    assert any(t["kind"] == "edit" for t in todos)
    assert memory.list_plan_steps(payload["flow_id"]) == todos


def test_preview_plan_delegates_to_decision_provider(tmp_path: Path) -> None:
    memory = CodingMemoryService(project_root=tmp_path)
    manager = QueueManager(
        worker_client=_NoopWorker(),
        repo_root=tmp_path,
        coding_memory_service=memory,
    )

    class _Provider:
        def __init__(self) -> None:
            self.called_with: dict | None = None

        def preview_execution_checklist(self, request, *, flow_id=None, flow_context=None):
            self.called_with = {"request": request, "flow_id": flow_id}
            return {
                "flow_id": "flow-xyz",
                "prechecklist": {
                    "objective": request,
                    "requires_edit": True,
                    "target_files": ["a.py"],
                    "steps": [{"id": "edit", "title": "Edit a.py", "requires_tools": ["apply_patch"]}],
                    "source": "planner",
                },
                "prechecklist_source": "planner",
                "prechecklist_warning": "",
                "requires_edit": True,
                "target_files": ["a.py"],
                "warnings": [],
            }

    provider = _Provider()
    manager.attach_decision_provider(provider)

    payload = manager.preview_plan(request="fix the bug", flow_id="flow-xyz")

    assert provider.called_with == {"request": "fix the bug", "flow_id": "flow-xyz"}
    assert payload["prechecklist_source"] == "planner"
    # Provider output is still synced into the todo ledger.
    assert payload["todos"][0]["kind"] == "edit"
    assert memory.list_plan_steps("flow-xyz")[0]["id"] == "edit"
