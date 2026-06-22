from __future__ import annotations

import json
from pathlib import Path

from mana_analyzer.llm.tool_worker_process import ToolRunResponse
from mana_analyzer.llm.goal_profiles import ModelDocsGoalProfile, active_goal_profile
from mana_analyzer.llm.tools_manager import (
    AutoExecuteResult,
    RunStateStore,
    ToolsManagerBatch,
    ToolsManagerOrchestrator,
    ToolsManagerRequest,
    ToolsPlan,
    ToolsPlanStep,
)
from mana_analyzer.llm.tools_executor import BatchExecutionResult
from mana_analyzer.services.coding_memory_service import CodingMemoryService


class _NoopWorker:
    def run_tools(self, _request, on_event=None):  # noqa: ANN001
        _ = on_event
        return ToolRunResponse(answer="ok", sources=[], mode="agent-tools", trace=[], warnings=[])


def _build_orchestrator(tmp_path: Path) -> ToolsManagerOrchestrator:
    orchestrator = object.__new__(ToolsManagerOrchestrator)
    orchestrator.llm = None
    orchestrator.worker_client = _NoopWorker()
    orchestrator.repo_root = tmp_path
    return orchestrator


def test_tools_manager_constructor_uses_top_level_executor_types(tmp_path: Path) -> None:
    orchestrator = ToolsManagerOrchestrator(
        api_key="test",
        model="fake",
        worker_client=_NoopWorker(),
        repo_root=tmp_path,
    )

    assert orchestrator.repo_root == tmp_path.resolve()
    assert orchestrator.execution_config is not None
    assert orchestrator.executor is not None


def test_tools_manager_planner_schema_parses_strict_json(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    valid_plan = (
        '{"objective":"Ship feature","steps":['
        '{"id":"s1","title":"Inspect files","tool_intent":"inspect","args_hint":"read foo.py",'
        '"success_signal":"inspected","fallback":"search"}'
        '],"stop_conditions":["done"],"finalize_action":"summarize"}'
    )
    monkeypatch.setattr(orchestrator, "_invoke_model", lambda **_kwargs: valid_plan)

    plan, warnings = orchestrator._plan(request="implement plan", flow_context=None)
    assert warnings == []
    assert isinstance(plan, ToolsPlan)
    assert plan is not None
    assert plan.objective == "Ship feature"
    assert len(plan.steps) == 1


def test_tools_manager_planner_parser_rejects_markdown_plan_text(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    markdown_plan = (
        "Execution Plan:\\n"
        "1. Inspect src/mana_analyzer/llm/coding_agent.py\\n"
        "2. Apply targeted patch\\n"
        "3. Verify with tests\\n"
    )
    plan = orchestrator.parse_tools_plan(markdown_plan, request="implement planner", previous_plan=None)
    assert plan is None


def test_tools_manager_planner_parser_does_not_infer_keyword_intents(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    keyword_plan = (
        "Execution Plan:\\n"
        "1. find all models\\n"
        "2. update docs/models.md\\n"
        "3. test the docs report\\n"
    )
    plan = orchestrator.parse_tools_plan(
        keyword_plan,
        request="find all models and update docs/models.md",
        previous_plan=None,
    )
    assert plan is None


def test_tools_manager_markdown_planner_output_uses_repaired_llm_intent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    markdown_plan = (
        "Execution Plan:\\n"
        "1. find all models\\n"
        "2. update docs/models.md\\n"
    )
    repaired_plan = (
        '{"objective":"Document models","steps":['
        '{"id":"s1","title":"Inspect models","tool_intent":"inspect","args_hint":"read model files"},'
        '{"id":"s2","title":"Update docs/models.md","tool_intent":"edit","args_hint":"apply docs patch"},'
        '{"id":"s3","title":"Verify docs","tool_intent":"verify","args_hint":"run focused checks"}'
        '],"current_step_id":"s2","decision":"continue","stop_conditions":["done"],'
        '"finalize_action":"summarize"}'
    )

    def _invoke_model(*, system_prompt: str, human_prompt: str) -> str:
        _ = human_prompt
        if system_prompt == "tools_planner":
            return markdown_plan
        if system_prompt == "tools_planner_repair":
            return repaired_plan
        raise AssertionError(f"unexpected prompt: {system_prompt}")

    monkeypatch.setattr(orchestrator, "_invoke_model", _invoke_model)

    plan, warnings = orchestrator._plan(
        request="find all models and update docs/models.md",
        flow_context=None,
    )

    assert warnings == ["head_tools_planner parse failed; attempting repair"]
    assert plan.current_step_id == "s2"
    assert [step.tool_intent for step in plan.steps] == ["inspect", "edit", "verify"]


def test_tools_manager_model_docs_fallback_uses_repository_local_evidence(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    plan = ToolsPlan(
        objective="Find all models and update docs/models.md",
        steps=[
            ToolsPlanStep(
                id="s1",
                title="Find all models",
                tool_intent="inspect",
                args_hint="enumerate model definitions",
            ),
            ToolsPlanStep(
                id="s2",
                title="Update docs/models.md",
                tool_intent="edit",
                args_hint="document model inventory",
            ),
        ],
        current_step_id="s1",
        decision="continue",
    )

    request = orchestrator._deterministic_fallback_request(
        request="find all models and update docs/models.md",
        flow_context=None,
        plan=plan,
        step=plan.steps[0],
        pass_index=1,
    )

    assert request is not None
    assert "docs/models.md" in request.question
    assert "rg -n" in request.question
    assert "AbstractUser" in request.question
    assert "Do not ask the user to paste files" in request.question


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


def test_tools_manager_successful_tool_status_initializes_trace_flags() -> None:
    empty_response = ToolRunResponse(answer="", sources=[], mode="agent-tools", trace=[], warnings=[])
    traced_response = ToolRunResponse(
        answer="",
        sources=[],
        mode="agent-tools",
        trace=[{"tool_name": "custom_tool", "status": "ok"}],
        warnings=[],
    )

    assert ToolsManagerOrchestrator._successful_tool_status(
        tool_name="verify_project",
        evidence_counts={},
        response_paths=[],
        response=empty_response,
    ) == (False, "verify_result_missing")
    assert ToolsManagerOrchestrator._successful_tool_status(
        tool_name="custom_tool",
        evidence_counts={},
        response_paths=[],
        response=empty_response,
    ) == (False, "tool_result_missing")
    assert ToolsManagerOrchestrator._successful_tool_status(
        tool_name="custom_tool",
        evidence_counts={},
        response_paths=[],
        response=traced_response,
    ) == (True, "")


def test_tools_manager_planner_parser_accepts_wrapped_payload(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    wrapped = (
        "[{'type': 'text', 'text': '{\"objective\": \"Ship feature\", \"steps\": "
        "[{\"id\": \"s1\", \"title\": \"Inspect\", \"tool_intent\": \"inspect\", "
        "\"args_hint\": \"\", \"success_signal\": \"\", \"fallback\": \"\", \"status\": \"in_progress\"}], "
        "\"current_step_id\": \"s1\", \"decision\": \"continue\", \"decision_reason\": \"start\", "
        "\"stop_conditions\": [\"done\"], \"finalize_action\": \"summarize\"}'}]"
    )
    plan = orchestrator.parse_tools_plan(wrapped, request="execute", previous_plan=None)
    assert isinstance(plan, ToolsPlan)
    assert plan is not None
    assert plan.objective == "Ship feature"
    assert plan.current_step_id == "s1"


def test_tools_manager_invalid_batch_triggers_repair_then_terminal_stop(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    plan = ToolsPlan(
        objective="Run",
        steps=[],
        stop_conditions=["done"],
        finalize_action="answer",
    )

    responses = iter(["{ broken", "not-json"])
    monkeypatch.setattr(orchestrator, "_invoke_model", lambda **_kwargs: next(responses))

    batch, issues = orchestrator._build_batch(
        request="execute",
        flow_context=None,
        plan=plan,
        pass_index=1,
        pass_cap=4,
        pass_logs=[],
        warnings=[],
        changed_files=[],
    )
    assert batch is None
    assert any("attempting repair" in issue for issue in issues)
    assert any("repair failed" in issue for issue in issues)


def test_tools_manager_planner_invalid_uses_deterministic_fallback(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    monkeypatch.setattr(orchestrator, "_invoke_model", lambda **_kwargs: "not-json-at-all")
    plan, warnings = orchestrator._plan(
        request="implement planner",
        flow_context=None,
        pass_index=0,
        pass_cap=4,
        previous_plan=None,
        pass_logs=[],
        warnings=[],
        changed_files=[],
        latest_answer="",
    )
    assert isinstance(plan, ToolsPlan)
    assert plan.steps
    assert plan.decision == "continue"
    assert any("deterministic fallback" in warning for warning in warnings)


def test_tools_manager_stalled_twice_stops_loop(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    plan = ToolsPlan(
        objective="Run",
        steps=[],
        stop_conditions=["done"],
        finalize_action="answer",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (plan, [], "planner"))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(requests=[], continue_after=True, expected_progress="waiting"),
            [],
        ),
    )

    result = orchestrator.run(
        request="execute",
        flow_context=None,
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=6,
    )
    assert isinstance(result, AutoExecuteResult)
    assert result.terminal_reason == "stalled_no_actionable_requests"
    assert result.passes == 2
    assert result.answer
    assert "terminal_reason=stalled_no_actionable_requests" in result.answer


def test_tools_manager_empty_batch_uses_deterministic_request_fallback(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    plan = ToolsPlan(
        objective="Run",
        steps=[
            {
                "id": "s1",
                "title": "Inspect key files",
                "tool_intent": "inspect",
                "args_hint": "read TODO.md and cli.py",
                "status": "in_progress",
            }
        ],
        current_step_id="s1",
        stop_conditions=["done"],
        finalize_action="answer",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (plan, [], "planner"))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(requests=[], continue_after=True, expected_progress="waiting"),
            [],
        ),
    )

    result = orchestrator.run(
        request="execute",
        flow_context=None,
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
    )
    assert isinstance(result, AutoExecuteResult)
    assert result.terminal_reason == "pass_cap_reached"
    assert result.toolsmanager_requests_count == 1
    assert result.pass_logs
    assert result.pass_logs[0]["requests_count"] == 1
    assert result.pass_logs[0]["batch_reason"] == "deterministic_empty_batch_fallback"
    assert any("deterministic fallback request" in str(item).lower() for item in result.warnings)


def test_tools_manager_batch_requests_include_flow_id(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    class _Executor:
        def __init__(self) -> None:
            self.seen_flow_ids: list[str | None] = []

        def run_batch(self, *, run_id, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, on_event)
            self.seen_flow_ids.extend([req.request.flow_id for req in requests])
            return [
                BatchExecutionResult(
                    request_index=req.request_index,
                    ok=True,
                    response={
                        "answer": "ok",
                        "sources": [],
                        "mode": "agent-tools",
                        "trace": [{"tool_name": "read_file", "status": "ok"}],
                        "warnings": [],
                    },
                )
                for req in requests
            ]

    executor = _Executor()
    orchestrator.executor = executor  # type: ignore[attr-defined]
    monkeypatch.setattr(
        orchestrator,
        "_plan_with_source",
        lambda **_kwargs: (
            ToolsPlan(
                objective="Run",
                steps=[
                    {
                        "id": "s1",
                        "title": "Inspect README",
                        "tool_intent": "inspect",
                        "args_hint": "read README.md",
                        "status": "in_progress",
                    }
                ],
                current_step_id="s1",
                decision="continue",
                decision_reason="inspect target file",
                stop_conditions=["done"],
                finalize_action="answer",
            ),
            [],
            "planner",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="inspect",
                requests=[
                    {
                        "question": "Inspect README.md",
                        "timeout_seconds": 20,
                    }
                ],
                continue_after=False,
                expected_progress="Inspect target file",
            ),
            [],
        ),
    )
    monkeypatch.setattr(orchestrator, "_git_status_paths", lambda: set())  # type: ignore[method-assign]

    result = orchestrator.run(
        request="Inspect README",
        flow_context=None,
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
        flow_id="flow-tools-1",
    )
    assert result.passes == 1
    assert executor.seen_flow_ids == ["flow-tools-1"]


def test_tools_manager_invalid_request_batch_terminal_reason(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    plan = ToolsPlan(
        objective="Run",
        steps=[],
        stop_conditions=["done"],
        finalize_action="answer",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (plan, [], "planner"))
    monkeypatch.setattr(orchestrator, "_build_batch", lambda **_kwargs: (None, ["bad batch"]))

    result = orchestrator.run(
        request="execute",
        flow_context=None,
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=4,
    )
    assert result.terminal_reason == "invalid_request_batch"
    assert "bad batch" in result.warnings
    assert result.passes == 0
    assert result.answer
    assert "terminal_reason=invalid_request_batch" in result.answer


def test_tools_manager_preview_plan_returns_normalized_prechecklist_without_worker_calls(
    tmp_path: Path, monkeypatch
) -> None:
    class _BoomWorker:
        def run_tools(self, _request, on_event=None):  # noqa: ANN001
            _ = on_event
            raise AssertionError("preview_plan must not execute worker tools")

    orchestrator = _build_orchestrator(tmp_path)
    orchestrator.worker_client = _BoomWorker()
    plan = ToolsPlan(
        objective="Implement planner flow",
        steps=[],
        stop_conditions=["done"],
        finalize_action="answer",
    )
    monkeypatch.setattr(
        orchestrator,
        "_plan_with_source",
        lambda **_kwargs: (plan, [], "planner"),
    )
    preview = orchestrator.preview_plan(request="implement plan.", flow_context=None, pass_cap=4)
    assert isinstance(preview.get("prechecklist"), dict)
    assert preview.get("prechecklist_source") == "planner"
    assert str(preview.get("prechecklist_warning", "")).strip() == ""


def test_tools_manager_preview_plan_surfaces_deterministic_fallback_warning(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    plan = ToolsPlan(
        objective="Fallback objective",
        steps=[],
        stop_conditions=["done"],
        finalize_action="answer",
    )
    monkeypatch.setattr(
        orchestrator,
        "_plan_with_source",
        lambda **_kwargs: (plan, ["planner parse failed"], "deterministic_fallback"),
    )
    preview = orchestrator.preview_plan(request="implement plan.", flow_context=None, pass_cap=4)
    assert preview.get("prechecklist_source") == "deterministic_fallback"
    assert "deterministic fallback checklist" in str(preview.get("prechecklist_warning", "")).lower()


def test_tools_manager_retries_conversational_finalize_without_edits(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    first_plan = ToolsPlan(
        objective="Run",
        steps=[],
        decision="finalize",
        decision_reason="If you want, I can continue with edits.",
        stop_conditions=["done"],
        finalize_action="Reply yes to continue.",
    )
    second_plan = ToolsPlan(
        objective="Run",
        steps=[],
        decision="finalize",
        decision_reason="Completed.",
        stop_conditions=["done"],
        finalize_action="Done.",
    )
    calls = {"count": 0}

    def _plan_with_source(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return first_plan, [], "planner"
        return second_plan, [], "planner"

    monkeypatch.setattr(orchestrator, "_plan_with_source", _plan_with_source)
    monkeypatch.setattr(
        orchestrator,
        "_deterministic_fallback_plan",
        lambda **_kwargs: ToolsPlan(
            objective="Run",
            steps=[
                {
                    "id": "s1",
                    "title": "Inspect files",
                    "tool_intent": "inspect",
                    "args_hint": "read_file",
                    "success_signal": "context gathered",
                    "fallback": "search",
                    "status": "in_progress",
                }
            ],
            current_step_id="s1",
            decision="continue",
            decision_reason="forced retry",
            stop_conditions=["done"],
            finalize_action="done",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="retry",
                requests=[{"question": "Inspect relevant files and continue execution."}],
                continue_after=True,
                expected_progress="retry progress",
            ),
            [],
        ),
    )

    result = orchestrator.run(
        request="execute",
        flow_context=None,
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=2,
    )
    assert result.toolsmanager_requests_count == 1
    assert any("planner_finalize_conversational_without_edits" in str(item) for item in result.warnings)


def test_tools_manager_retries_non_hard_blocker_terminal_without_edits(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    first_plan = ToolsPlan(
        objective="Run",
        steps=[],
        decision="finalize",
        decision_reason="Blocker: I need a scope choice. Please choose option 1 or 2.",
        stop_conditions=["done"],
        finalize_action="Awaiting scope decision.",
    )
    second_plan = ToolsPlan(
        objective="Run",
        steps=[],
        decision="finalize",
        decision_reason="Completed.",
        stop_conditions=["done"],
        finalize_action="Done.",
    )
    calls = {"count": 0}

    def _plan_with_source(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return first_plan, [], "planner"
        return second_plan, [], "planner"

    monkeypatch.setattr(orchestrator, "_plan_with_source", _plan_with_source)
    monkeypatch.setattr(
        orchestrator,
        "_deterministic_fallback_plan",
        lambda **_kwargs: ToolsPlan(
            objective="Run",
            steps=[
                {
                    "id": "s1",
                    "title": "Inspect files",
                    "tool_intent": "inspect",
                    "args_hint": "read_file",
                    "success_signal": "context gathered",
                    "fallback": "search",
                    "status": "in_progress",
                }
            ],
            current_step_id="s1",
            decision="continue",
            decision_reason="forced retry",
            stop_conditions=["done"],
            finalize_action="done",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="retry",
                requests=[{"question": "Inspect relevant files and continue execution."}],
                continue_after=True,
                expected_progress="retry progress",
            ),
            [],
        ),
    )

    result = orchestrator.run(
        request="execute",
        flow_context=None,
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=2,
    )
    assert result.toolsmanager_requests_count == 1
    assert any("planner_terminal_nonhard_blocker_retry" in str(item) for item in result.warnings)


def test_tools_manager_retries_repository_access_soft_blocker_terminal(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    first_plan = ToolsPlan(
        objective="Run",
        steps=[],
        decision="finalize",
        decision_reason=(
            "I'm blocked on making a safe, accurate update because I need to read the current repository files first."
        ),
        stop_conditions=["done"],
        finalize_action="Please share permission to proceed.",
    )
    second_plan = ToolsPlan(
        objective="Run",
        steps=[],
        decision="finalize",
        decision_reason="Completed.",
        stop_conditions=["done"],
        finalize_action="Done.",
    )
    calls = {"count": 0}

    def _plan_with_source(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return first_plan, [], "planner"
        return second_plan, [], "planner"

    monkeypatch.setattr(orchestrator, "_plan_with_source", _plan_with_source)
    monkeypatch.setattr(
        orchestrator,
        "_deterministic_fallback_plan",
        lambda **_kwargs: ToolsPlan(
            objective="Run",
            steps=[
                {
                    "id": "s1",
                    "title": "Inspect files",
                    "tool_intent": "inspect",
                    "args_hint": "read_file",
                    "success_signal": "context gathered",
                    "fallback": "search",
                    "status": "in_progress",
                }
            ],
            current_step_id="s1",
            decision="continue",
            decision_reason="forced retry",
            stop_conditions=["done"],
            finalize_action="done",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="retry",
                requests=[{"question": "Inspect relevant files and continue execution."}],
                continue_after=True,
                expected_progress="retry progress",
            ),
            [],
        ),
    )

    result = orchestrator.run(
        request="execute",
        flow_context=None,
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=2,
    )
    assert result.toolsmanager_requests_count == 1
    assert any("planner_terminal_nonhard_blocker_retry" in str(item) for item in result.warnings)


def test_tools_manager_preserves_stop_for_hard_blocker(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    stop_plan = ToolsPlan(
        objective="Run",
        steps=[],
        decision="stop",
        decision_reason="Permission denied and missing credential for write access.",
        stop_conditions=["done"],
        finalize_action="Blocked.",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (stop_plan, [], "planner"))

    result = orchestrator.run(
        request="execute",
        flow_context=None,
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=3,
    )
    assert result.terminal_reason == "planner_stop"
    assert result.toolsmanager_requests_count == 0
    assert any("planner_terminal_hard_blocker_stop" in str(item) for item in result.warnings)


def test_tools_manager_merges_batch_results_in_input_order(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    continue_plan = ToolsPlan(
        objective="Run",
        steps=[
            {
                "id": "s1",
                "title": "Inspect",
                "tool_intent": "inspect",
                "args_hint": "",
                "success_signal": "",
                "fallback": "",
                "status": "in_progress",
            }
        ],
        current_step_id="s1",
        decision="continue",
        stop_conditions=["done"],
        finalize_action="done",
    )
    finalize_plan = ToolsPlan(
        objective="Run",
        steps=[],
        decision="finalize",
        decision_reason="done",
        stop_conditions=["done"],
        finalize_action="done",
    )

    calls = {"n": 0}

    def _plan_with_source(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return continue_plan, [], "planner"
        return finalize_plan, [], "planner"

    monkeypatch.setattr(orchestrator, "_plan_with_source", _plan_with_source)
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="parallel",
                requests=[{"question": "q0"}, {"question": "q1"}],
                continue_after=True,
                expected_progress="inspect",
            ),
            [],
        ),
    )

    class _OutOfOrderExecutor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, requests, on_event)
            return [
                BatchExecutionResult(
                    request_index=1,
                    ok=True,
                    response={
                        "answer": "second",
                        "sources": [],
                        "mode": "agent-tools",
                        "trace": [{"tool_name": "read_file", "status": "ok", "idx": 1}],
                        "warnings": [],
                    },
                    backend="redis",
                ),
                BatchExecutionResult(
                    request_index=0,
                    ok=True,
                    response={
                        "answer": "first",
                        "sources": [],
                        "mode": "agent-tools",
                        "trace": [{"tool_name": "semantic_search", "status": "ok", "idx": 0}],
                        "warnings": [],
                    },
                    backend="redis",
                ),
            ]

    orchestrator.executor = _OutOfOrderExecutor()

    result = orchestrator.run(
        request="execute",
        flow_context=None,
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=2,
    )
    assert result.toolsmanager_requests_count == 2
    assert result.execution_requests_ok == 2
    assert result.execution_requests_failed == 0
    assert len(result.trace) == 2
    assert result.trace[0].get("idx") == 0
    assert result.trace[1].get("idx") == 1


def test_tools_manager_mixed_batch_failures_are_warnings_not_crash(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)

    continue_plan = ToolsPlan(
        objective="Run",
        steps=[
            {
                "id": "s1",
                "title": "Inspect",
                "tool_intent": "inspect",
                "args_hint": "",
                "success_signal": "",
                "fallback": "",
                "status": "in_progress",
            }
        ],
        current_step_id="s1",
        decision="continue",
        stop_conditions=["done"],
        finalize_action="done",
    )
    finalize_plan = ToolsPlan(
        objective="Run",
        steps=[],
        decision="finalize",
        decision_reason="done",
        stop_conditions=["done"],
        finalize_action="done",
    )

    calls = {"n": 0}

    def _plan_with_source(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return continue_plan, [], "planner"
        return finalize_plan, [], "planner"

    monkeypatch.setattr(orchestrator, "_plan_with_source", _plan_with_source)
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="parallel",
                requests=[{"question": "q0"}, {"question": "q1"}],
                continue_after=True,
                expected_progress="inspect",
            ),
            [],
        ),
    )

    class _MixedExecutor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, requests, on_event)
            return [
                BatchExecutionResult(
                    request_index=0,
                    ok=False,
                    error_code="job_timeout",
                    error_message="timed out",
                    backend="redis",
                ),
                BatchExecutionResult(
                    request_index=1,
                    ok=True,
                    response={
                        "answer": "ok",
                        "sources": [],
                        "mode": "agent-tools",
                        "trace": [{"tool_name": "read_file", "status": "ok"}],
                        "warnings": [],
                    },
                    backend="redis",
                ),
            ]

    orchestrator.executor = _MixedExecutor()

    result = orchestrator.run(
        request="execute",
        flow_context=None,
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=2,
    )
    assert result.toolsmanager_requests_count == 2
    assert result.execution_requests_ok == 1
    assert result.execution_requests_failed >= 1
    assert result.request_retry_attempts == 1
    assert result.request_retry_exhausted == 1
    assert any("job_timeout" in str(item) for item in result.warnings)


def test_deterministic_fallback_edit_directive_mentions_write_retry_and_change_evidence(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    plan = ToolsPlan(
        objective="Update README section",
        steps=[
            {
                "id": "s1",
                "title": "Edit README",
                "tool_intent": "edit",
                "args_hint": "insert diagram section",
                "success_signal": "README updated",
                "fallback": "write_file fallback",
                "status": "in_progress",
            }
        ],
        current_step_id="s1",
        decision="continue",
        stop_conditions=["done"],
        finalize_action="done",
    )

    req = orchestrator._deterministic_fallback_request(
        request="insert project diagram into README.md",
        flow_context=None,
        plan=plan,
        step=plan.steps[0],
        pass_index=1,
    )

    assert req is not None
    text = str(req.question).lower()
    assert "apply_patch" in text
    assert "write_file" in text
    assert "changed_files" in text
    assert "conversational terminal" in text


def test_tools_manager_skips_duplicate_request_fingerprints_across_passes(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    orchestrator.coding_memory_service = CodingMemoryService(project_root=tmp_path)

    continue_plan = ToolsPlan(
        objective="Run",
        steps=[
            {
                "id": "s1",
                "title": "Inspect",
                "tool_intent": "inspect",
                "args_hint": "read files",
                "success_signal": "done",
                "fallback": "",
                "status": "in_progress",
            }
        ],
        current_step_id="s1",
        decision="continue",
        stop_conditions=["done"],
        finalize_action="done",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (continue_plan, [], "planner"))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="inspect",
                requests=[{"question": "Inspect the same files again"}],
                continue_after=True,
                expected_progress="inspect",
            ),
            [],
        ),
    )

    class _Executor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, on_event)
            return [
                BatchExecutionResult(
                    request_index=int(req.request_index),
                    ok=True,
                    response={"answer": "ok", "sources": [], "mode": "agent-tools", "trace": [], "warnings": []},
                    backend="local",
                )
                for req in requests
            ]

    orchestrator.executor = _Executor()
    result = orchestrator.run(
        request="execute",
        flow_context=None,
        flow_id="flow-dup-1",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=2,
    )
    assert result.duplicate_request_skips >= 1
    assert any("duplicate_request_skipped" in str(item) for item in result.warnings)


def test_tools_manager_skips_duplicate_semantic_search_across_turns_with_same_flow(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    orchestrator.coding_memory_service = CodingMemoryService(project_root=tmp_path)

    continue_plan = ToolsPlan(
        objective="Run",
        steps=[
            {
                "id": "s1",
                "title": "Search",
                "tool_intent": "search",
                "args_hint": "semantic search",
                "success_signal": "done",
                "fallback": "",
                "status": "in_progress",
            }
        ],
        current_step_id="s1",
        decision="continue",
        stop_conditions=["done"],
        finalize_action="done",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (continue_plan, [], "planner"))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="search",
                requests=[{"question": "Run semantic_search query=foo k=2"}],
                continue_after=True,
                expected_progress="search",
            ),
            [],
        ),
    )

    class _Executor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, requests, on_event)
            return [
                BatchExecutionResult(
                    request_index=0,
                    ok=True,
                    response={
                        "answer": "ok",
                        "sources": [],
                        "mode": "agent-tools",
                        "trace": [{"tool_name": "semantic_search", "status": "ok", "args_summary": "query=foo k=2"}],
                        "warnings": [],
                    },
                    backend="local",
                )
            ]

    orchestrator.executor = _Executor()
    first = orchestrator.run(
        request="execute",
        flow_context=None,
        flow_id="flow-search-1",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
    )
    assert first.duplicate_semantic_search_skips == 0

    second = orchestrator.run(
        request="execute",
        flow_context=None,
        flow_id="flow-search-1",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
    )
    assert second.duplicate_semantic_search_skips >= 1
    assert any("duplicate_semantic_search_skipped" in str(item) for item in second.warnings)


def test_tools_manager_duplicate_suppression_forces_deterministic_fallback_request(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    service = CodingMemoryService(project_root=tmp_path)
    orchestrator.coding_memory_service = service

    continue_plan = ToolsPlan(
        objective="Run",
        steps=[
            {
                "id": "s1",
                "title": "Inspect",
                "tool_intent": "inspect",
                "args_hint": "",
                "success_signal": "",
                "fallback": "",
                "status": "in_progress",
            }
        ],
        current_step_id="s1",
        decision="continue",
        stop_conditions=["done"],
        finalize_action="done",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (continue_plan, [], "planner"))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="inspect",
                requests=[{"question": "Inspect same file"}],
                continue_after=True,
                expected_progress="inspect",
            ),
            [],
        ),
    )

    fp = orchestrator._fingerprint_request("Inspect same file", {}, 30)
    service.record_tool_fingerprint(flow_id="flow-dup-2", kind="request_fingerprint", fingerprint=fp)

    class _Executor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, on_event)
            assert len(requests) == 1
            return [
                BatchExecutionResult(
                    request_index=int(requests[0].request_index),
                    ok=True,
                    response={"answer": "ok", "sources": [], "mode": "agent-tools", "trace": [], "warnings": []},
                    backend="local",
                )
            ]

    orchestrator.executor = _Executor()
    result = orchestrator.run(
        request="execute",
        flow_context=None,
        flow_id="flow-dup-2",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
    )
    assert result.pass_logs
    assert result.pass_logs[0]["batch_reason"] == "deterministic_duplicate_suppression_fallback"


def test_tools_manager_edit_retry_mode_suppresses_new_search_after_patch_noop(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    orchestrator.coding_memory_service = CodingMemoryService(project_root=tmp_path)

    continue_plan = ToolsPlan(
        objective="Run",
        steps=[
            {
                "id": "s1",
                "title": "Edit",
                "tool_intent": "edit",
                "args_hint": "",
                "success_signal": "",
                "fallback": "",
                "status": "in_progress",
            }
        ],
        current_step_id="s1",
        decision="continue",
        stop_conditions=["done"],
        finalize_action="done",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (continue_plan, [], "planner"))
    calls = {"n": 0}

    def _build_batch(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return (
                ToolsManagerBatch(
                    planner_step_id="s1",
                    batch_reason="edit",
                    requests=[{"question": "apply_patch README"}],
                    continue_after=True,
                    expected_progress="edit",
                ),
                [],
            )
        return (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="search",
                requests=[{"question": "semantic_search README"}],
                continue_after=True,
                expected_progress="search",
            ),
            [],
        )

    monkeypatch.setattr(orchestrator, "_build_batch", _build_batch)

    class _Executor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, on_event)
            question = str(requests[0].request.question)
            if "apply_patch" in question:
                return [
                    BatchExecutionResult(
                        request_index=int(requests[0].request_index),
                        ok=True,
                        response={
                            "answer": "no-op",
                            "sources": [],
                            "mode": "agent-tools",
                            "trace": [
                                {
                                    "tool_name": "apply_patch",
                                    "status": "ok",
                                    "output_preview": '{"ok": false, "error": "no changes"}',
                                }
                            ],
                            "warnings": [],
                        },
                        backend="local",
                    )
                ]
            return [
                BatchExecutionResult(
                    request_index=int(requests[0].request_index),
                    ok=True,
                    response={"answer": "ok", "sources": [], "mode": "agent-tools", "trace": [], "warnings": []},
                    backend="local",
                )
            ]

    orchestrator.executor = _Executor()
    result = orchestrator.run(
        request="execute",
        flow_context=None,
        flow_id="flow-edit-1",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=2,
    )
    assert any("edit_retry_mode_activated" in str(item) for item in result.warnings)
    assert result.pass_logs
    assert len(result.pass_logs) >= 2
    assert bool(result.pass_logs[1].get("edit_retry_mode_active", False)) is True


def test_tools_manager_model_docs_blocks_mutation_until_inventory_read(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "src" / "accounts").mkdir(parents=True)
    (tmp_path / "src" / "accounts" / "models.py").write_text(
        "from django.db import models\nclass Customer(models.Model):\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "models.md").write_text("# Models\n", encoding="utf-8")
    orchestrator = _build_orchestrator(tmp_path)
    monkeypatch.setattr(orchestrator, "_git_status_paths", lambda: set())

    edit_plan = ToolsPlan(
        objective="Find all models and update docs/models.md",
        steps=[
            {
                "id": "s1",
                "title": "Update docs/models.md",
                "tool_intent": "edit",
                "args_hint": "apply docs patch",
                "success_signal": "docs updated",
                "fallback": "",
                "status": "in_progress",
            }
        ],
        current_step_id="s1",
        decision="continue",
        stop_conditions=["done"],
        finalize_action="done",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (edit_plan, [], "planner"))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="premature edit",
                requests=[{"question": "apply_patch docs/models.md", "tool_name": "apply_patch"}],
                continue_after=True,
                expected_progress="edit docs",
            ),
            [],
        ),
    )

    class _Executor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, on_event)
            assert requests[0].request.tool_name == "read_file"
            assert requests[0].request.tool_args["path"] == "src/accounts/models.py"
            return [
                BatchExecutionResult(
                    request_index=int(requests[0].request_index),
                    ok=True,
                    response={
                        "answer": "class Customer(models.Model): pass",
                        "sources": [{"file_path": "src/accounts/models.py"}],
                        "mode": "agent-tools",
                        "trace": [
                            {
                                "tool_name": "read_file",
                                "status": "ok",
                                "output_preview": '{"file_path":"src/accounts/models.py"}',
                            }
                        ],
                        "warnings": [],
                    },
                    backend="local",
                )
            ]

    orchestrator.executor = _Executor()
    result = orchestrator.run(
        request="find all models and update docs/models.md",
        flow_context=None,
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
    )
    assert any("mutation_blocked_until_model_docs_evidence_complete" in str(item) for item in result.warnings)
    assert result.pass_logs[0]["new_files_read"] == 1
    assert result.next_action == "read_file docs/models.md"


def test_tools_manager_failed_request_retries_once_and_records_signature(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    orchestrator.coding_memory_service = CodingMemoryService(project_root=tmp_path)

    continue_plan = ToolsPlan(
        objective="Run",
        steps=[
            {
                "id": "s1",
                "title": "Inspect",
                "tool_intent": "inspect",
                "args_hint": "",
                "success_signal": "",
                "fallback": "",
                "status": "in_progress",
            }
        ],
        current_step_id="s1",
        decision="continue",
        stop_conditions=["done"],
        finalize_action="done",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (continue_plan, [], "planner"))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="inspect",
                requests=[{"question": "inspect once"}],
                continue_after=True,
                expected_progress="inspect",
            ),
            [],
        ),
    )

    class _Executor:
        def __init__(self) -> None:
            self.calls = 0

        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, requests, on_event)
            self.calls += 1
            return [
                BatchExecutionResult(
                    request_index=0,
                    ok=False,
                    error_code="job_failed",
                    error_message=f"boom-{self.calls}",
                    backend="local",
                )
            ]

    orchestrator.executor = _Executor()
    result = orchestrator.run(
        request="execute",
        flow_context=None,
        flow_id="flow-retry-1",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
    )
    assert result.request_retry_attempts == 1
    assert result.request_retry_exhausted == 1
    assert any("toolsmanager_request_retry_once" in str(item) for item in result.warnings)
    assert any("toolsmanager_retry_exhausted_signature=" in str(item) for item in result.warnings)


def test_tools_manager_auto_advances_duplicate_planner_task_to_next_step(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    orchestrator.coding_memory_service = CodingMemoryService(project_root=tmp_path)

    continue_plan = ToolsPlan(
        objective="Run",
        steps=[
            {
                "id": "s1",
                "title": "Inspect files",
                "tool_intent": "inspect",
                "args_hint": "",
                "success_signal": "",
                "fallback": "",
                "status": "in_progress",
            },
            {
                "id": "s2",
                "title": "Edit files",
                "tool_intent": "edit",
                "args_hint": "",
                "success_signal": "",
                "fallback": "",
                "status": "pending",
            },
        ],
        current_step_id="s1",
        decision="continue",
        decision_reason="start with inspection",
        stop_conditions=["done"],
        finalize_action="done",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (continue_plan, [], "planner"))

    def _build_batch(**kwargs):
        plan = kwargs["plan"]
        if str(plan.current_step_id) == "s1":
            return (
                ToolsManagerBatch(
                    planner_step_id="s1",
                    batch_reason="inspect",
                    requests=[{"question": "Inspect files"}],
                    continue_after=True,
                    expected_progress="inspect",
                ),
                [],
            )
        return (
            ToolsManagerBatch(
                planner_step_id="s2",
                batch_reason="edit",
                requests=[{"question": "Edit files"}],
                continue_after=True,
                expected_progress="edit",
            ),
            [],
        )

    monkeypatch.setattr(orchestrator, "_build_batch", _build_batch)

    class _Executor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, on_event)
            question = str(requests[0].request.question)
            return [
                BatchExecutionResult(
                    request_index=0,
                    ok=True,
                    response={
                        "answer": question,
                        "sources": [],
                        "mode": "agent-tools",
                        "trace": [{"tool_name": "read_file" if "Inspect" in question else "apply_patch", "status": "ok"}],
                        "warnings": [],
                    },
                    backend="local",
                )
            ]

    orchestrator.executor = _Executor()
    result = orchestrator.run(
        request="execute",
        flow_context=None,
        flow_id="flow-task-advance",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=2,
    )
    # Edit tasks now reserve enough passes to reach edit + verify, so the cap is
    # raised above the configured value of 2; the first two passes still cover
    # the inspect step then the auto-advanced edit step.
    assert len(result.pass_logs) >= 2
    assert result.pass_logs[0]["planner_step_id"] == "s1"
    assert result.pass_logs[1]["planner_step_id"] == "s2"
    assert any("planner_duplicate_task_advanced" in str(item) for item in result.warnings)


def test_tools_manager_pass_cap_unfinished_edit_does_not_surface_incidental_answer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    monkeypatch.setattr(orchestrator, "_git_status_paths", lambda: set())
    monkeypatch.setattr(orchestrator, "_compute_effective_pass_cap", lambda **_kwargs: 1)

    continue_plan = ToolsPlan(
        objective="Update .gitignore",
        steps=[
            ToolsPlanStep(
                id="s1",
                title="Append .mana to .gitignore",
                tool_intent="edit",
                status="in_progress",
            )
        ],
        current_step_id="s1",
        decision="continue",
        stop_conditions=["done"],
        finalize_action="done",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (continue_plan, [], "planner"))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="read irrelevant missing file",
                requests=[{"question": "Read .mana/analyze.json"}],
                continue_after=True,
                expected_progress="inspect",
            ),
            [],
        ),
    )

    class _Executor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, requests, on_event)
            return [
                BatchExecutionResult(
                    request_index=0,
                    ok=True,
                    response={
                        "answer": "The file .mana/analyze.json is not present in the repository.",
                        "sources": [],
                        "mode": "agent-tools",
                        "trace": [{"tool_name": "read_file", "status": "error"}],
                        "warnings": [],
                    },
                    backend="local",
                )
            ]

    orchestrator.executor = _Executor()
    result = orchestrator.run(
        request="update .gitignore add .mana",
        flow_context=None,
        flow_id="flow-unfinished-edit",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
    )

    assert result.terminal_reason == "pass_cap_reached"
    assert "should continue" in result.answer
    assert "Auto-execute ended without a direct answer" not in result.answer
    assert ".mana/analyze.json is not present" not in result.answer
    assert any("pass_cap_reached_with_pending_work" in str(item) for item in result.warnings)
    assert any("edit_task_pass_cap_without_changed_files" in str(item) for item in result.warnings)


def test_tools_manager_pass_cap_docs_update_emits_resumable_status_and_local_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    monkeypatch.setattr(orchestrator, "_git_status_paths", lambda: set())
    monkeypatch.setattr(orchestrator, "_compute_effective_pass_cap", lambda **_kwargs: 1)

    plan = ToolsPlan(
        objective="Find all models and update docs/models.md",
        steps=[
            ToolsPlanStep(id="s1", title="Find all models", tool_intent="inspect"),
            ToolsPlanStep(id="s2", title="Update docs/models.md", tool_intent="edit"),
            ToolsPlanStep(id="s3", title="Verify docs/models.md", tool_intent="verify"),
        ],
        current_step_id="s1",
        decision="continue",
        stop_conditions=["done"],
        finalize_action="summarize",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (plan, [], "planner"))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="planner emitted no concrete request",
                requests=[],
                continue_after=True,
                expected_progress="inspect models",
            ),
            [],
        ),
    )

    seen_questions: list[str] = []

    class _Executor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, on_event)
            seen_questions.extend(str(item.request.question) for item in requests)
            return [
                BatchExecutionResult(
                    request_index=0,
                    ok=True,
                    response={
                        "answer": "Collected repository evidence; docs update remains pending.",
                        "sources": [],
                        "mode": "agent-tools",
                        "trace": [{"tool_name": "run_command", "status": "ok"}],
                        "warnings": [],
                    },
                    backend="local",
                )
            ]

    orchestrator.executor = _Executor()
    result = orchestrator.run(
        request="find all models and update docs/models.md",
        flow_context=None,
        flow_id="flow-docs-models",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
    )

    assert result.terminal_reason == "pass_cap_reached"
    assert "should continue" in result.answer
    assert "Auto-execute ended without a direct answer" not in result.answer
    assert seen_questions
    assert "docs/models.md" in seen_questions[0]
    assert "rg -n" in seen_questions[0]
    assert "Do not ask the user to paste files" in seen_questions[0]
    assert any("pass_cap_reached_with_pending_work" in str(item) for item in result.warnings)


def test_tools_manager_pass_cap_writes_persistent_checkpoint(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "models.py").write_text("class User(BaseModel):\n    pass\n", encoding="utf-8")
    orchestrator = _build_orchestrator(tmp_path)
    monkeypatch.setattr(orchestrator, "_git_status_paths", lambda: set())
    monkeypatch.setattr(orchestrator, "_compute_effective_pass_cap", lambda **_kwargs: 1)
    plan = ToolsPlan(
        objective="Inventory models",
        steps=[ToolsPlanStep(id="s1", title="Locate candidates", tool_intent="search")],
        current_step_id="s1",
        decision="continue",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (plan, [], "planner"))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                requests=[ToolsManagerRequest(question="repo_search models.py", tool_name="repo_search")],
            ),
            [],
        ),
    )

    class _Executor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, requests, on_event)
            return [
                BatchExecutionResult(
                    request_index=0,
                    ok=True,
                    response=ToolRunResponse(
                        answer="found app/models.py",
                        trace=[{"tool_name": "repo_search", "path": "app/models.py"}],
                    ).model_dump(),
                    backend="local",
                )
            ]

    orchestrator.executor = _Executor()
    result = orchestrator.run(
        request="find model files",
        flow_context=None,
        flow_id="flow",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
    )

    run_dir = tmp_path / ".mana" / "runs" / result.run_id
    assert result.run_status == "needs_resume"
    assert f"mana-analyzer continue --root-dir {tmp_path}" in result.answer
    assert f"--run-id {result.run_id}" in result.answer
    assert result.resume_command == f"mana-analyzer continue --root-dir {tmp_path} --run-id {result.run_id}"
    assert (run_dir / "state.json").exists()
    assert (run_dir / "todo.json").exists()
    assert (run_dir / "evidence.jsonl").exists()
    assert (run_dir / "visited_files.json").exists()
    assert (run_dir / "tool_calls.jsonl").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "resume_prompt.md").exists()
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text())
    assert checkpoint["run_id"] == result.run_id
    assert checkpoint["root_dir"] == str(tmp_path.resolve())
    assert checkpoint["current_phase"] in {
        "DISCOVERY",
        "READING",
        "EXTRACTION",
        "PATCHING",
        "VERIFYING",
        "FINAL",
    }
    assert checkpoint["original_user_task"] == "find model files"
    assert checkpoint["candidate_files"] == ["app/models.py"]
    assert checkpoint["pending_files"] == ["app/models.py"]
    assert checkpoint["next_exact_action"] == "read_file app/models.py"
    assert checkpoint["progress_counters"]["candidate_files"] == 1
    assert "app/models.py" in (run_dir / "evidence.jsonl").read_text()
    assert "repo_search" in (run_dir / "tool_calls.jsonl").read_text()


def test_tools_manager_writes_public_work_ledger_and_trace_metadata(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "models.py").write_text("class User(BaseModel):\n    pass\n", encoding="utf-8")
    orchestrator = _build_orchestrator(tmp_path)
    monkeypatch.setattr(orchestrator, "_git_status_paths", lambda: set())
    monkeypatch.setattr(orchestrator, "_compute_effective_pass_cap", lambda **_kwargs: 1)
    plan = ToolsPlan(
        objective="Inventory models",
        steps=[ToolsPlanStep(id="s1", title="Locate candidates", tool_intent="search")],
        current_step_id="s1",
        decision="continue",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (plan, [], "planner"))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                requests=[ToolsManagerRequest(question="repo_search models.py", tool_name="repo_search")],
            ),
            [],
        ),
    )

    class _Executor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, requests, on_event)
            return [
                BatchExecutionResult(
                    request_index=0,
                    ok=True,
                    response=ToolRunResponse(
                        answer="found app/models.py",
                        trace=[{"tool_name": "repo_search", "path": "app/models.py"}],
                    ).model_dump(),
                    backend="local",
                )
            ]

    orchestrator.executor = _Executor()
    result = orchestrator.run(
        request="find model files",
        flow_context=None,
        flow_id="flow-ledger",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
    )

    run_dir = tmp_path / ".mana" / "runs" / result.run_id
    ledger = json.loads((run_dir / "work_ledger.json").read_text())
    assert ledger["run_id"] == result.run_id
    assert ledger["objective"] == "find model files"
    assert ledger["current_phase"] in {"DISCOVERY", "READING", "EXTRACTION", "PATCHING", "VERIFYING", "FINAL"}
    assert ledger["candidate_files"] == ["app/models.py"]
    assert ledger["next_action"] == "read_file app/models.py"
    assert ledger["tool_call_history"][0]["normalized_key"]
    assert ledger["tool_call_history"][0]["purpose"] == "locate_candidates"
    assert ledger["tool_call_history"][0]["phase"] == "DISCOVERY"
    assert result.trace
    assert result.trace[0]["normalized_key"]
    assert result.trace[0]["purpose"] == "locate_candidates"
    assert result.trace[0]["ledger_checkpoint_path"].endswith("work_ledger.json")


def test_tools_manager_pending_read_queue_forces_read_before_search(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "models.py").write_text("class User(BaseModel):\n    pass\n", encoding="utf-8")
    orchestrator = _build_orchestrator(tmp_path)
    monkeypatch.setattr(orchestrator, "_git_status_paths", lambda: set())
    monkeypatch.setattr(orchestrator, "_compute_effective_pass_cap", lambda **_kwargs: 1)
    store = RunStateStore(repo_root=tmp_path, run_id="queued")
    store.ensure(goal="read queued models", flow_id="flow")
    store.write_json("todo.json", {"pending_file_reads": ["app/models.py"], "pending_edits": [], "verification_status": "pending"})
    plan = ToolsPlan(
        objective="Read candidates",
        steps=[ToolsPlanStep(id="s1", title="Read candidates", tool_intent="search")],
        current_step_id="s1",
        decision="continue",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (plan, [], "planner"))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                requests=[ToolsManagerRequest(question="repo_search models.py", tool_name="repo_search")],
            ),
            [],
        ),
    )
    seen: list[tuple[str, dict[str, object]]] = []

    class _Executor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, on_event)
            req = requests[0].request
            seen.append((req.tool_name, dict(req.tool_args)))
            return [
                BatchExecutionResult(
                    request_index=0,
                    ok=True,
                    response=ToolRunResponse(
                        answer="read app/models.py",
                        trace=[{"tool_name": "read_file", "path": "app/models.py"}],
                    ).model_dump(),
                    backend="local",
                )
            ]

    orchestrator.executor = _Executor()
    result = orchestrator.run(
        request="continue reading candidates",
        flow_context=None,
        flow_id="flow",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
        run_id="queued",
    )

    assert seen == [("read_file", {"path": "app/models.py"})]
    assert "pending_read_queue_forced_progress" in result.warnings
    assert RunStateStore(repo_root=tmp_path, run_id="queued").read_json("todo.json")["pending_file_reads"] == []


def test_tools_manager_resume_uses_existing_run_and_skips_completed_search(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    monkeypatch.setattr(orchestrator, "_git_status_paths", lambda: set())
    monkeypatch.setattr(orchestrator, "_compute_effective_pass_cap", lambda **_kwargs: 1)
    store = RunStateStore(repo_root=tmp_path, run_id="resume1")
    store.ensure(goal="resume search", flow_id="flow")
    fp = store.fingerprint(
        gate="locate_candidates",
        tool_name="repo_search",
        args={"question": "repo_search models.py", "tool_args": {}, "policy": {}, "timeout_seconds": 30, "index_dir": str((tmp_path / ".mana/index").resolve()), "index_dirs": []},
        filters={"k": 8, "max_steps": 6},
    )
    store.record_tool_call(
        gate="locate_candidates",
        tool_name="repo_search",
        normalized_args={"question": "repo_search models.py"},
        fingerprint=fp,
        status="ok",
        result_summary="cached search",
    )
    plan = ToolsPlan(
        objective="resume search",
        steps=[ToolsPlanStep(id="s1", title="Locate candidates", tool_intent="search")],
        current_step_id="s1",
        decision="continue",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (plan, [], "planner"))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                requests=[ToolsManagerRequest(question="repo_search models.py", tool_name="repo_search")],
            ),
            [],
        ),
    )

    class _Executor:
        def run_batch(self, **_kwargs):  # noqa: ANN001
            raise AssertionError("cached search should not execute")

    orchestrator.executor = _Executor()
    result = orchestrator.resume_run(
        run_id="resume1",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
    )

    assert "duplicate_tool_call_skipped" in " ".join(result.warnings)
    assert result.run_id == "resume1"


def test_tools_manager_resume_starts_from_exact_pending_read(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "back" / "billing").mkdir(parents=True)
    (tmp_path / "back" / "billing" / "admin.py").write_text("from .models import Invoice\n", encoding="utf-8")
    (tmp_path / "back" / "billing" / "models.py").write_text("class Invoice(BaseModel):\n    pass\n", encoding="utf-8")
    orchestrator = _build_orchestrator(tmp_path)
    monkeypatch.setattr(orchestrator, "_git_status_paths", lambda: set())
    monkeypatch.setattr(orchestrator, "_compute_effective_pass_cap", lambda **_kwargs: 1)
    store = RunStateStore(repo_root=tmp_path, run_id="exact-read")
    store.ensure(goal="resume exact read", flow_id="flow")
    store.write_json(
        "todo.json",
        {
            "pending_file_reads": ["back/billing/admin.py", "back/billing/models.py"],
            "pending_edits": [],
            "verification_status": "pending",
        },
    )
    store.update_state(status="needs_resume", next_action="read_file back/billing/admin.py")
    store.write_checkpoint(
        status="needs_resume",
        completed_gates=["locate_candidates"],
        pending_gates=["read_candidates"],
        files_changed=[],
        verification_status="pending",
    )
    plan = ToolsPlan(
        objective="resume exact read",
        steps=[ToolsPlanStep(id="s1", title="Locate candidates", tool_intent="search")],
        current_step_id="s1",
        decision="continue",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (plan, [], "planner"))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                requests=[ToolsManagerRequest(question="repo_search billing", tool_name="repo_search")],
            ),
            [],
        ),
    )
    seen: list[tuple[str, dict[str, object]]] = []

    class _Executor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, on_event)
            req = requests[0].request
            seen.append((req.tool_name, dict(req.tool_args)))
            return [
                BatchExecutionResult(
                    request_index=0,
                    ok=True,
                    response=ToolRunResponse(
                        answer="read admin",
                        trace=[{"tool_name": "read_file", "path": "back/billing/admin.py"}],
                    ).model_dump(),
                    backend="local",
                )
            ]

    orchestrator.executor = _Executor()
    result = orchestrator.resume_run(
        run_id="exact-read",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
    )

    assert seen == [("read_file", {"path": "back/billing/admin.py"})]
    assert result.next_action == "read_file back/billing/models.py"
    checkpoint = RunStateStore(repo_root=tmp_path, run_id="exact-read").read_json("checkpoint.json")
    assert checkpoint["read_files"] == ["back/billing/admin.py"]
    assert checkpoint["pending_files"] == ["back/billing/models.py"]


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


def test_tools_manager_resume_without_decision_provider_uses_deterministic_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(ToolsManagerOrchestrator, "_git_status_paths", lambda _self: set())
    orchestrator = ToolsManagerOrchestrator(
        api_key="test",
        model="fake",
        worker_client=_NoopWorker(),
        repo_root=tmp_path,
    )
    monkeypatch.setattr(orchestrator, "_compute_effective_pass_cap", lambda **_kwargs: 1)
    store = RunStateStore(repo_root=tmp_path, run_id="resume-no-provider")
    store.ensure(goal="resume without planner", flow_id="flow")

    class _Executor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, requests, on_event)
            return [
                BatchExecutionResult(
                    request_index=0,
                    ok=True,
                    response=ToolRunResponse(
                        answer="deterministic fallback ran",
                        trace=[{"tool_name": "read_file", "path": "app/models.py"}],
                    ).model_dump(),
                    backend="local",
                )
            ]

    orchestrator.executor = _Executor()
    result = orchestrator.resume_run(
        run_id="resume-no-provider",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
    )

    assert result.run_id == "resume-no-provider"
    assert any("planner unavailable" in warning for warning in result.warnings)
    assert any("toolsmanager unavailable" in warning for warning in result.warnings)


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


def test_tools_manager_resumed_apply_changes_routes_to_mutation_not_pending_read(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    git_calls = {"n": 0}

    def _git_status_paths() -> set[str]:
        git_calls["n"] += 1
        return set() if git_calls["n"] == 1 else {"docs/models.md"}

    monkeypatch.setattr(orchestrator, "_git_status_paths", _git_status_paths)
    store = RunStateStore(repo_root=tmp_path, run_id="resume-apply-mutation")
    store.ensure(goal="find all models and update docs/models.md", flow_id="flow")
    store.write_json(
        "todo.json",
        {
            "pending_file_reads": ["src/app/models.py"],
            "pending_edits": [],
            "verification_status": "pending",
        },
    )
    state = store.read_json("state.json")
    state["current_phase"] = "DISCOVERY"
    state["current_gate"] = "locate_candidates"
    state["completed_gates"] = ["locate_candidates", "read_candidates", "classify_evidence", "plan_patch"]
    state["pending_gates"] = ["apply_changes", "verify_changes", "final_report"]
    store.write_json("state.json", state)

    plan = ToolsPlan(
        objective="Find all models and update docs/models.md",
        steps=[ToolsPlanStep(id="s1", title="Accidental discovery step", tool_intent="search")],
        current_step_id="s1",
        decision="continue",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (plan, [], "planner"))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="resume apply",
                requests=[
                    ToolsManagerRequest(
                        question="apply_patch docs/models.md",
                        tool_name="apply_patch",
                        tool_args={"path": "docs/models.md", "patch": "*** Begin Patch\n*** End Patch\n"},
                    )
                ],
            ),
            [],
        ),
    )
    seen_tools: list[str] = []

    class _Executor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, on_event)
            seen_tools.extend(str(req.request.tool_name) for req in requests)
            assert requests[0].request.tool_name == "apply_patch"
            return [
                BatchExecutionResult(
                    request_index=0,
                    ok=True,
                    response=ToolRunResponse(
                        answer="patched docs/models.md",
                        trace=[{"tool_name": "apply_patch", "status": "ok"}],
                    ).model_dump(),
                    backend="local",
                )
            ]

    orchestrator.executor = _Executor()
    result = orchestrator.run(
        request="find all models and update docs/models.md",
        flow_context="resume",
        flow_id="flow",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
        run_id="resume-apply-mutation",
    )

    assert seen_tools == ["apply_patch"]
    assert "pending_read_queue_forced_progress" not in result.warnings
    assert result.changed_files == ["docs/models.md"]


def test_tools_manager_resumed_apply_changes_missing_payload_fails_without_search_loop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    monkeypatch.setattr(orchestrator, "_git_status_paths", lambda: set())
    store = RunStateStore(repo_root=tmp_path, run_id="resume-apply-missing-payload")
    store.ensure(goal="find all models and update docs/models.md", flow_id="flow")
    store.write_json(
        "todo.json",
        {
            "pending_file_reads": ["src/app/models.py"],
            "pending_edits": [],
            "verification_status": "pending",
        },
    )
    state = store.read_json("state.json")
    state["current_phase"] = "DISCOVERY"
    state["current_gate"] = "locate_candidates"
    state["completed_gates"] = ["locate_candidates", "read_candidates", "classify_evidence", "plan_patch"]
    state["pending_gates"] = ["apply_changes", "verify_changes", "final_report"]
    store.write_json("state.json", state)

    plan = ToolsPlan(
        objective="Find all models and update docs/models.md",
        steps=[ToolsPlanStep(id="s1", title="Accidental discovery step", tool_intent="search")],
        current_step_id="s1",
        decision="continue",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (plan, [], "planner"))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="bad resumed discovery",
                requests=[ToolsManagerRequest(question="repo_search models.py", tool_name="repo_search")],
            ),
            [],
        ),
    )

    class _Executor:
        def run_batch(self, **_kwargs):  # noqa: ANN001
            raise AssertionError("missing edit payload should fail before search/read execution")

    orchestrator.executor = _Executor()
    result = orchestrator.run(
        request="find all models and update docs/models.md",
        flow_context="resume",
        flow_id="flow",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
        run_id="resume-apply-missing-payload",
    )

    state_after = RunStateStore(repo_root=tmp_path, run_id="resume-apply-missing-payload").read_json("state.json")
    assert result.run_status == "failed_no_progress"
    assert result.terminal_reason == "missing_edit_payload"
    assert "missing_edit_payload" in result.warnings
    assert state_after["failure_gate"] == "apply_changes"
    assert state_after["required_next_tool"] == "apply_patch|write_file|create_file"
    assert "no mutation tool" in state_after["why_not_executed"]


def test_tools_only_violation_retries_same_edit_todo_with_required_mutation_tool(
    tmp_path: Path,
    monkeypatch,
) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    git_calls = {"n": 0}

    def _git_status_paths() -> set[str]:
        git_calls["n"] += 1
        return set() if git_calls["n"] == 1 else {"docs/models.md"}

    monkeypatch.setattr(orchestrator, "_git_status_paths", _git_status_paths)
    store = RunStateStore(repo_root=tmp_path, run_id="tools-only-edit-retry")
    store.ensure(goal="find all models and update docs/models.md", flow_id="flow")
    state = store.read_json("state.json")
    state["completed_gates"] = ["locate_candidates", "read_candidates", "classify_evidence", "plan_patch"]
    state["pending_gates"] = ["apply_changes", "verify_changes", "final_report"]
    store.write_json("state.json", state)

    plan = ToolsPlan(
        objective="Find all models and update docs/models.md",
        steps=[ToolsPlanStep(id="s1", title="Update docs/models.md", tool_intent="edit")],
        current_step_id="s1",
        decision="continue",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (plan, [], "planner"))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                batch_reason="edit",
                requests=[
                    ToolsManagerRequest(
                        question="worker must mutate docs/models.md",
                        tool_name="write_file",
                        tool_args={"path": "docs/models.md", "content": "# Models\n"},
                    )
                ],
            ),
            [],
        ),
    )
    seen_batches: list[list[str]] = []

    class _Executor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, on_event)
            seen_batches.append([str(req.request.tool_name) for req in requests])
            if len(seen_batches) == 1:
                return [
                    BatchExecutionResult(
                        request_index=0,
                        ok=False,
                        error_code="tools_only_violation",
                        error_message="tools-only mode requires at least one successful tool call",
                        backend="local",
                    )
                ]
            assert requests[0].request.tool_name == "apply_patch"
            return [
                BatchExecutionResult(
                    request_index=0,
                    ok=True,
                    response=ToolRunResponse(
                        answer="patched docs/models.md",
                        trace=[{"tool_name": "apply_patch", "status": "ok"}],
                    ).model_dump(),
                    backend="local",
                )
            ]

    orchestrator.executor = _Executor()
    result = orchestrator.run(
        request="find all models and update docs/models.md",
        flow_context="resume",
        flow_id="flow",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
        run_id="tools-only-edit-retry",
    )
    todo_after = RunStateStore(repo_root=tmp_path, run_id="tools-only-edit-retry").current_todo_for_gate(
        "apply_changes",
        goal="find all models and update docs/models.md",
    )

    assert seen_batches == [["write_file"], ["apply_patch"]]
    assert "tools_only_violation_current_todo_failed" in result.warnings
    assert result.changed_files == ["docs/models.md"]
    assert todo_after.status == "agent_confirmed"


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
    (tmp_path / "src" / "mana_analyzer").mkdir(parents=True)
    (tmp_path / "src" / "mana_analyzer" / "models.py").write_text("class User(BaseModel):\n    pass\n")
    (tmp_path / "src" / "mana_analyzer" / "schema_models.py").write_text("class Item(TypedDict):\n    pass\n")
    (tmp_path / "src" / "mana_analyzer" / "__init__.py").write_text("")
    (tmp_path / "src" / "mana_analyzer" / "commands").mkdir()
    (tmp_path / "src" / "mana_analyzer" / "commands" / "chat_cli.py").write_text("class Cli:\n    pass\n")
    (tmp_path / "src" / "mana_analyzer" / "tools").mkdir()
    (tmp_path / "src" / "mana_analyzer" / "tools" / "apply_patch.py").write_text("class Patch:\n    pass\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_models.py").write_text("from mana_analyzer.models import User\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "models.md").write_text("# Models\n")
    (tmp_path / "README.md").write_text("# Readme\n")

    store = RunStateStore(repo_root=tmp_path, run_id="rank")
    store.ensure(goal="create docs/models.md for all models", flow_id="")
    store.seed_candidate_queue()

    pending = store.read_json("todo.json")["pending_file_reads"]
    assert pending[:2] == ["src/mana_analyzer/models.py", "src/mana_analyzer/schema_models.py"]
    assert "src/mana_analyzer/commands/chat_cli.py" not in pending
    assert "src/mana_analyzer/tools/apply_patch.py" not in pending
    assert "src/mana_analyzer/__init__.py" not in pending
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


def test_tools_manager_repairs_forced_read_policy_and_rejects_noop_success(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "models.py").write_text("class User(BaseModel):\n    pass\n", encoding="utf-8")
    orchestrator = _build_orchestrator(tmp_path)
    monkeypatch.setattr(orchestrator, "_git_status_paths", lambda: set())
    monkeypatch.setattr(orchestrator, "_compute_effective_pass_cap", lambda **_kwargs: 1)
    store = RunStateStore(repo_root=tmp_path, run_id="noop-read")
    store.ensure(goal="read queued models", flow_id="flow")
    store.write_json(
        "todo.json",
        {
            "pending_file_reads": ["app/models.py"],
            "pending_edits": [],
            "verification_status": "pending",
        },
    )
    plan = ToolsPlan(
        objective="Read candidates",
        steps=[ToolsPlanStep(id="s1", title="Read candidates", tool_intent="search")],
        current_step_id="s1",
        decision="continue",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (plan, [], "planner"))
    monkeypatch.setattr(
        orchestrator,
        "_build_batch",
        lambda **_kwargs: (
            ToolsManagerBatch(
                planner_step_id="s1",
                requests=[ToolsManagerRequest(question="repo_search models.py", tool_name="repo_search")],
            ),
            [],
        ),
    )
    seen_policies: list[dict[str, object]] = []
    seen_retry_attempts: list[int] = []

    class _Executor:
        def run_batch(self, *, run_id: str, requests, on_event=None):  # noqa: ANN001
            _ = (run_id, on_event)
            req = requests[0].request
            seen_policies.append(dict(req.tool_policy or {}))
            seen_retry_attempts.append(int(req.retry_attempt or 0))
            return [
                BatchExecutionResult(
                    request_index=0,
                    ok=True,
                    response=ToolRunResponse(
                        answer="No file read",
                        trace=[{"tool_name": "read_file", "status": "ok"}],
                    ).model_dump(),
                    backend="local",
                )
            ]

    orchestrator.executor = _Executor()
    result = orchestrator.run(
        request="continue reading candidates",
        flow_context=None,
        flow_id="flow",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={"allowed_tools": ["repo_search"]},
        pass_cap=1,
        run_id="noop-read",
        max_no_progress_passes=1,
    )

    assert "read_file" in seen_policies[0]["allowed_tools"]
    assert seen_retry_attempts == [0, 1]
    assert result.run_status == "failed_no_progress"
    assert result.changed_files == []
    assert result.next_action != "read_file app/models.py"
    ledger = RunStateStore(repo_root=tmp_path, run_id="noop-read").read_json("work_ledger.json")
    assert ledger["last_successful_action"]["tool_name"] == ""
    assert "app/models.py" not in ledger["pending_work"]["pending_file_reads"]
    tool_history = ledger["tool_call_history"]
    assert all(row["status"] != "ok" for row in tool_history)


def test_tools_manager_final_report_contains_partial_progress(tmp_path: Path, monkeypatch) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    monkeypatch.setattr(orchestrator, "_git_status_paths", lambda: set())
    monkeypatch.setattr(orchestrator, "_compute_effective_pass_cap", lambda **_kwargs: 1)
    plan = ToolsPlan(
        objective="Partial progress",
        steps=[ToolsPlanStep(id="s1", title="Locate candidates", tool_intent="search")],
        current_step_id="s1",
        decision="continue",
    )
    monkeypatch.setattr(orchestrator, "_plan_with_source", lambda **_kwargs: (plan, [], "planner"))
    monkeypatch.setattr(orchestrator, "_build_batch", lambda **_kwargs: (ToolsManagerBatch(planner_step_id="s1", requests=[]), []))

    result = orchestrator.run(
        request="partial",
        flow_context=None,
        flow_id="flow",
        index_dir=tmp_path / ".mana/index",
        index_dirs=None,
        k=8,
        max_steps=6,
        timeout_seconds=30,
        tool_policy={},
        pass_cap=1,
    )

    summary = (tmp_path / ".mana" / "runs" / result.run_id / "summary.md").read_text()
    assert "completed_gates" in summary
    assert "pending_gates" in summary
    assert "next_action" in summary


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


def test_is_edit_task_detects_edit_step(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    assert orchestrator._is_edit_task(_edit_plan(), "do something") is True
    assert orchestrator._is_edit_task(_inspect_only_plan(), "explain the project") is False


def test_is_edit_task_detects_create_request_text(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    # No edit step, but the request reads as a create task.
    assert orchestrator._is_edit_task(_inspect_only_plan(), "create a analyze.md") is True


def test_pass_cap_raised_for_edit_task(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    # Configured cap of 4 is too low: inspect(1) + edit(1) + verify(1) + 2 = 5.
    effective = orchestrator._compute_effective_pass_cap(
        configured_pass_cap=4,
        plan=_edit_plan(),
        request="analyze fully project and create a analyze.md",
    )
    assert effective >= 5
    assert effective <= 12


def test_pass_cap_unchanged_for_non_edit_task(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    effective = orchestrator._compute_effective_pass_cap(
        configured_pass_cap=4,
        plan=_inspect_only_plan(),
        request="explain the project",
    )
    assert effective == 4


def test_pass_cap_never_exceeds_max_allowed(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    effective = orchestrator._compute_effective_pass_cap(
        configured_pass_cap=4,
        plan=_edit_plan(),
        request="create files",
        max_allowed_passes=12,
    )
    assert effective <= 12
