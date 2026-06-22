from __future__ import annotations

from pathlib import Path

from mana_analyzer.llm.tool_worker_process import ToolRunResponse
from mana_analyzer.llm.tools_manager import (
    AutoExecuteResult,
    ToolsManagerBatch,
    ToolsManagerOrchestrator,
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
    assert "Auto-execute ended without a direct answer" in result.answer
    assert ".mana/analyze.json is not present" not in result.answer
    assert any("edit_task_pass_cap_without_changed_files" in str(item) for item in result.warnings)


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
