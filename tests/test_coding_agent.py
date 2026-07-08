from __future__ import annotations

import json
from pathlib import Path

from mana_agent.multi_agent.runtime.coding_agent import CodingAgent, FlowChecklist, FlowStep
from mana_agent.multi_agent.runtime.agent_work_queue import WorkItem
from mana_agent.multi_agent.runtime.tool_worker_process import ToolRunResponse, ToolWorkerProcessError
from mana_agent.multi_agent.runtime.tools_manager import AutoExecuteResult
from mana_agent.services.coding_memory_service import CodingMemoryService


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeAskAgent:
    def __init__(self, response_payload: dict) -> None:
        self.tools: list[object] = []
        self.model = "fake"
        self.calls: list[dict] = []
        self.response_payload = response_payload

    def run(
        self,
        question: str,
        index_dir: str | Path,
        k: int,
        max_steps: int,
        timeout_seconds: int,
        callbacks: list[object] | None = None,
        system_prompt: str | None = None,
        tool_policy: dict | None = None,
        flow_id: str | None = None,
    ) -> str:
        _ = (index_dir, k, max_steps, timeout_seconds, callbacks, system_prompt)
        self.calls.append({"question": question, "tool_policy": tool_policy or {}, "flow_id": flow_id})
        return json.dumps(self.response_payload)

    def run_multi(
        self,
        question: str,
        index_dirs,
        k: int,
        max_steps: int,
        timeout_seconds: int,
        callbacks: list[object] | None = None,
        system_prompt: str | None = None,
        tool_policy: dict | None = None,
        flow_id: str | None = None,
    ) -> str:
        _ = (index_dirs, k, max_steps, timeout_seconds, callbacks, system_prompt)
        self.calls.append({"question": question, "tool_policy": tool_policy or {}, "flow_id": flow_id})
        return json.dumps(self.response_payload)


class _SequenceAskAgent(_FakeAskAgent):
    def __init__(self, response_payloads: list[dict]) -> None:
        super().__init__(response_payload=response_payloads[0] if response_payloads else {})
        self._payloads = list(response_payloads)
        self._cursor = 0

    def _next_payload(self) -> dict:
        if not self._payloads:
            return {}
        if self._cursor >= len(self._payloads):
            return self._payloads[-1]
        payload = self._payloads[self._cursor]
        self._cursor += 1
        return payload

    def run(
        self,
        question: str,
        index_dir: str | Path,
        k: int,
        max_steps: int,
        timeout_seconds: int,
        callbacks: list[object] | None = None,
        system_prompt: str | None = None,
        tool_policy: dict | None = None,
        flow_id: str | None = None,
    ) -> str:
        _ = (index_dir, k, max_steps, timeout_seconds, callbacks, system_prompt)
        self.calls.append({"question": question, "tool_policy": tool_policy or {}, "flow_id": flow_id})
        return json.dumps(self._next_payload())

    def run_multi(
        self,
        question: str,
        index_dirs,
        k: int,
        max_steps: int,
        timeout_seconds: int,
        callbacks: list[object] | None = None,
        system_prompt: str | None = None,
        tool_policy: dict | None = None,
        flow_id: str | None = None,
    ) -> str:
        _ = (index_dirs, k, max_steps, timeout_seconds, callbacks, system_prompt)
        self.calls.append({"question": question, "tool_policy": tool_policy or {}, "flow_id": flow_id})
        return json.dumps(self._next_payload())


class _FakeOrchestrator:
    def __init__(self, response_payload: dict) -> None:
        self.calls: list[dict] = []
        self.response_payload = response_payload

    def run(self, **kwargs):
        self.calls.append(dict(kwargs))
        return AutoExecuteResult(
            answer=json.dumps(self.response_payload),
            trace=list(self.response_payload.get("trace", [])),
            warnings=list(self.response_payload.get("warnings", [])),
            sources=list(self.response_payload.get("sources", [])),
            changed_files=list(self.response_payload.get("changed_files", [])),
            passes=1,
            terminal_reason="manager_stop",
            toolsmanager_requests_count=1,
        )


class _SequenceOrchestrator(_FakeOrchestrator):
    def __init__(self, response_payloads: list[dict]) -> None:
        super().__init__(response_payload=response_payloads[0] if response_payloads else {})
        self._payloads = list(response_payloads)
        self._cursor = 0

    def _next_payload(self) -> dict:
        if not self._payloads:
            return {}
        if self._cursor >= len(self._payloads):
            return self._payloads[-1]
        payload = self._payloads[self._cursor]
        self._cursor += 1
        return payload

    def run(self, **kwargs):
        self.calls.append(dict(kwargs))
        payload = self._next_payload()
        return AutoExecuteResult(
            answer=json.dumps(payload),
            trace=list(payload.get("trace", [])),
            warnings=list(payload.get("warnings", [])),
            sources=list(payload.get("sources", [])),
            changed_files=list(payload.get("changed_files", [])),
            passes=1,
            terminal_reason="manager_stop",
            toolsmanager_requests_count=1,
        )


def _fixed_checklist() -> FlowChecklist:
    return FlowChecklist(
        objective="Implement request",
        requires_edit=True,
        constraints=["scope src/ tests/ only"],
        acceptance=["tests pass"],
        steps=[
            FlowStep(id="s1", title="Discover relevant files", reason="find exact code", status="in_progress"),
            FlowStep(id="s2", title="Inspect file contents", reason="validate behavior", status="pending"),
            FlowStep(id="s3", title="Apply edit", reason="implement request", status="pending"),
        ],
        next_action="Inspect target files.",
    )


def _git_checklist(*, tools: list[str] | None = None) -> FlowChecklist:
    return FlowChecklist(
        objective="Run git operation",
        requires_edit=False,
        steps=[
            FlowStep(
                id="s1",
                title="Inspect git context",
                reason="Git operation requires repository state first",
                status="in_progress",
                requires_tools=tools or ["git_status"],
            )
        ],
        next_action="Inspect git context.",
    )


class _RecordingToolWorker:
    def __init__(self) -> None:
        self.requests = []

    def run_tools(self, request, on_event=None):  # noqa: ANN001
        _ = on_event
        self.requests.append(request)
        return ToolRunResponse(
            answer="ok",
            trace=[{"tool_name": request.tool_name or "", "status": "ok"}],
        )


def test_checklist_requires_edit_recognizes_mutation_tools() -> None:
    # Edit intent is recognized from the planner's planned tools, not the text.
    edit_plan = FlowChecklist(
        objective="add docs",
        steps=[
            FlowStep(id="s1", title="Discover", reason="r", requires_tools=["semantic_search", "read_file"]),
            FlowStep(id="s2", title="Apply", reason="r", requires_tools=["create_file", "write_file"]),
        ],
    )
    readonly_plan = FlowChecklist(
        objective="explain config",
        steps=[
            FlowStep(id="s1", title="Discover", reason="r", requires_tools=["semantic_search"]),
            FlowStep(id="s2", title="Read", reason="r", requires_tools=["read_file"]),
        ],
    )
    assert CodingAgent._checklist_requires_edit(edit_plan) is True
    assert CodingAgent._checklist_requires_edit(readonly_plan) is False
    # No checklist (planner unavailable): err toward acting.
    assert CodingAgent._checklist_requires_edit(None) is True


def test_checklist_requires_edit_uses_structured_planner_flag_without_tool_list() -> None:
    edit_plan = FlowChecklist(
        objective="add docs",
        requires_edit=True,
        steps=[
            FlowStep(id="s1", title="Inspect docs", reason="r"),
            FlowStep(id="s2", title="Write analyze.md", reason="add project description"),
        ],
    )

    assert CodingAgent._checklist_requires_edit(edit_plan) is True


def test_checklist_requires_edit_does_not_infer_from_step_text() -> None:
    readonly_plan = FlowChecklist(
        objective="explain update flow",
        requires_edit=False,
        steps=[
            FlowStep(id="s1", title="Explain write path", reason="describe update behavior"),
        ],
    )

    assert CodingAgent._checklist_requires_edit(readonly_plan) is False


def _build_agent(
    tmp_path: Path,
    monkeypatch,
    *,
    payload: dict,
    memory: bool = True,
    full_auto_mode: bool = False,
) -> CodingAgent:
    monkeypatch.setattr("mana_agent.multi_agent.runtime.coding_agent.build_write_file_tool", lambda **_kwargs: _Tool("write_file"))
    monkeypatch.setattr("mana_agent.multi_agent.runtime.coding_agent.build_create_file_tool", lambda **_kwargs: _Tool("create_file"))
    monkeypatch.setattr("mana_agent.multi_agent.runtime.coding_agent.build_delete_file_tool", lambda **_kwargs: _Tool("delete_file"))
    monkeypatch.setattr("mana_agent.multi_agent.runtime.coding_agent.build_apply_patch_tool", lambda **_kwargs: _Tool("apply_patch"))
    ask_agent = _FakeAskAgent(payload)
    svc = CodingMemoryService(project_root=tmp_path, max_turns=5, max_tasks=20) if memory else None
    agent = CodingAgent(
        api_key="test-key",
        repo_root=tmp_path,
        ask_agent=ask_agent,
        allowed_prefixes=None,
        coding_memory_service=svc,
        coding_memory_enabled=memory,
        plan_max_steps=8,
        search_budget=4,
        read_budget=6,
        require_read_files=2,
        full_auto_mode=full_auto_mode,
    )
    monkeypatch.setattr(agent, "_plan_checklist", lambda request, flow_context=None: (_fixed_checklist(), []))
    monkeypatch.setattr(agent, "_git_status_paths", lambda: set())  # type: ignore[method-assign]
    monkeypatch.setattr(agent, "_git_diff", lambda _paths: "")  # type: ignore[method-assign]
    monkeypatch.setattr(agent, "_run_static_analysis", lambda _paths: [])  # type: ignore[method-assign]
    agent.set_tools_manager_orchestrator(_FakeOrchestrator(payload))
    return agent


def _build_agent_with_ask(tmp_path: Path, monkeypatch, ask_agent, *, full_auto_mode: bool = False) -> CodingAgent:
    monkeypatch.setattr("mana_agent.multi_agent.runtime.coding_agent.build_write_file_tool", lambda **_kwargs: _Tool("write_file"))
    monkeypatch.setattr("mana_agent.multi_agent.runtime.coding_agent.build_create_file_tool", lambda **_kwargs: _Tool("create_file"))
    monkeypatch.setattr("mana_agent.multi_agent.runtime.coding_agent.build_delete_file_tool", lambda **_kwargs: _Tool("delete_file"))
    monkeypatch.setattr("mana_agent.multi_agent.runtime.coding_agent.build_apply_patch_tool", lambda **_kwargs: _Tool("apply_patch"))
    svc = CodingMemoryService(project_root=tmp_path, max_turns=5, max_tasks=20)
    agent = CodingAgent(
        api_key="test-key",
        repo_root=tmp_path,
        ask_agent=ask_agent,
        allowed_prefixes=None,
        coding_memory_service=svc,
        coding_memory_enabled=True,
        plan_max_steps=8,
        search_budget=4,
        read_budget=6,
        require_read_files=1,
        full_auto_mode=full_auto_mode,
    )
    monkeypatch.setattr(agent, "_plan_checklist", lambda request, flow_context=None: (_fixed_checklist(), []))
    monkeypatch.setattr(agent, "_git_diff", lambda _paths: "")  # type: ignore[method-assign]
    monkeypatch.setattr(agent, "_run_static_analysis", lambda _paths: [])  # type: ignore[method-assign]
    payloads = list(getattr(ask_agent, "_payloads", []) or [getattr(ask_agent, "response_payload", {})])
    agent.set_tools_manager_orchestrator(_SequenceOrchestrator(payloads))
    return agent


def _orchestrator_calls(agent: CodingAgent) -> list[dict]:
    orchestrator = agent.tools_manager_orchestrator
    assert isinstance(orchestrator, _FakeOrchestrator)
    return orchestrator.calls


def test_work_queue_seed_git_commit_does_not_start_with_repo_search(tmp_path: Path, monkeypatch) -> None:
    agent = _build_agent(tmp_path, monkeypatch, payload={"answer": "ok", "trace": [], "warnings": []})
    monkeypatch.setattr(agent, "_plan_checklist", lambda request, flow_context=None: (_git_checklist(), []))

    seeds = agent._seed_items_for_request("git commit")

    assert seeds
    assert seeds[0].tool_name == "git_status"
    assert all(seed.tool_name != "repo_search" for seed in seeds)
    assert all("Locate files relevant to" not in seed.question for seed in seeds)


def test_work_queue_seed_git_status_uses_git_context_or_tool_decision(tmp_path: Path, monkeypatch) -> None:
    agent = _build_agent(tmp_path, monkeypatch, payload={"answer": "ok", "trace": [], "warnings": []})
    monkeypatch.setattr(agent, "_plan_checklist", lambda request, flow_context=None: (_git_checklist(), []))

    seeds = agent._seed_items_for_request("git status")

    assert seeds
    assert seeds[0].tool_name in {"git_status", ""}
    assert all(seed.tool_name != "repo_search" for seed in seeds)


def test_work_queue_seed_git_add_commit_push_starts_with_git_context(tmp_path: Path, monkeypatch) -> None:
    agent = _build_agent(tmp_path, monkeypatch, payload={"answer": "ok", "trace": [], "warnings": []})
    monkeypatch.setattr(agent, "_plan_checklist", lambda request, flow_context=None: (_git_checklist(), []))

    seeds = agent._seed_items_for_request("git add and git commit and push to feature/model-router-entry-no-fallback")

    assert seeds
    assert seeds[0].gate == "git_context"
    assert seeds[0].tool_name == "git_status"
    assert all(seed.tool_name != "repo_search" for seed in seeds)


def test_work_queue_seed_broad_code_request_can_use_repo_search(tmp_path: Path, monkeypatch) -> None:
    agent = _build_agent(tmp_path, monkeypatch, payload={"answer": "ok", "trace": [], "warnings": []})

    seeds = agent._seed_items_for_request("fix bug in router")

    assert seeds
    assert seeds[0].tool_name == "repo_search"
    assert "Locate files relevant to" in seeds[0].question


def test_work_queue_seed_exact_target_file_reads_without_broad_repo_search(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    agent = _build_agent(tmp_path, monkeypatch, payload={"answer": "ok", "trace": [], "warnings": []})

    seeds = agent._seed_items_for_request("update README.md")

    assert [seed.tool_name for seed in seeds] == ["read_file"]
    assert seeds[0].tool_args == {"path": "README.md"}
    assert all(seed.tool_name != "repo_search" for seed in seeds)
    assert all("Locate files relevant to" not in seed.question for seed in seeds)


def test_run_via_work_queue_preserves_explicit_seeds(tmp_path: Path, monkeypatch) -> None:
    agent = _build_agent(tmp_path, monkeypatch, payload={"answer": "ok", "trace": [], "warnings": []})
    worker = _RecordingToolWorker()
    agent.tool_worker_client = worker  # type: ignore[assignment]

    def _fail_auto_seed(*_args, **_kwargs):
        raise AssertionError("automatic seed decision should not run when explicit seeds are provided")

    monkeypatch.setattr(agent, "_seed_items_for_request", _fail_auto_seed)
    explicit = WorkItem(
        kind="inspect",
        tool_name="git_status",
        tool_args={},
        question="custom explicit seed",
        gate="explicit",
        priority=3,
    )

    result = agent.run_via_work_queue("git status", seeds=[explicit], max_steps=1)

    assert result["ok"] is True
    assert worker.requests
    assert worker.requests[0].tool_name == "git_status"
    assert worker.requests[0].question == "custom explicit seed"


def test_coding_agent_builds_structured_checklist_before_tools(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    result = agent.generate("Implement planner", index_dir=tmp_path / ".mana/index", k=4)
    assert isinstance(result.get("plan"), dict)
    assert result["plan"]["objective"] == "Implement request"
    assert result["checklist"]["total"] >= 1


def test_coding_agent_blocks_answer_until_required_file_reads_met(tmp_path: Path, monkeypatch) -> None:
    payload = {
        "answer": "ok",
        "trace": [
            {"tool_name": "semantic_search", "status": "ok", "duration_ms": 2.0, "args_summary": "q=planner"},
            {"tool_name": "read_file", "status": "ok", "duration_ms": 3.0, "args_summary": "one"},
        ],
        "warnings": [],
    }
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    result = agent.generate("Implement planner", index_dir=tmp_path / ".mana/index", k=4)
    assert result["progress"]["phase"] == "blocked"
    assert "Need at least 2 unique read_file inspections" in result["progress"]["why"]


def test_coding_agent_prevents_duplicate_semantic_search_loops(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    agent.generate("Implement planner", index_dir=tmp_path / ".mana/index", k=4)
    calls = _orchestrator_calls(agent)
    assert calls
    policy = calls[0]["tool_policy"]
    assert policy["search_repeat_limit"] == 1


def test_coding_agent_auto_chat_answer_mode_blocks_mutation_tools(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    agent.generate(
        "Where is approval handled?",
        index_dir=tmp_path / ".mana/index",
        k=4,
        auto_chat_mode="answer_only",
    )
    policy = _orchestrator_calls(agent)[0]["tool_policy"]
    assert "apply_patch" not in policy["allowed_tools"]
    assert "write_file" not in policy["allowed_tools"]
    assert policy["auto_chat_mode"] == "answer_only"
    assert policy["read_budget"] <= 6


def test_coding_agent_auto_chat_edit_mode_allows_mutation_tools(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    agent.generate(
        "Fix duplicate approval",
        index_dir=tmp_path / ".mana/index",
        k=4,
        auto_chat_mode="edit",
    )
    policy = _orchestrator_calls(agent)[0]["tool_policy"]
    assert "apply_patch" in policy["allowed_tools"]
    assert policy["auto_chat_mode"] == "edit"


def test_coding_agent_effective_prompt_includes_language_tooling_guide(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    prompt = agent._effective_system_prompt_for("Fix failing pytest and npm test flows")
    lowered = prompt.lower()
    assert ".venv" in lowered
    assert "node_modules" in lowered
    assert "npm install" in lowered
    assert "pytest -q" in lowered
    assert "mode rules" in lowered
    assert "compact skills index" in lowered
    assert "current task context" in lowered


def test_coding_agent_enforces_search_budget_and_transitions_phase(tmp_path: Path, monkeypatch) -> None:
    payload = {
        "answer": "ok",
        "trace": [
            {"tool_name": "semantic_search", "status": "ok", "duration_ms": 1.0, "args_summary": "a"},
            {"tool_name": "semantic_search", "status": "ok", "duration_ms": 1.0, "args_summary": "b"},
            {"tool_name": "read_file", "status": "ok", "duration_ms": 1.0, "args_summary": "file1"},
            {"tool_name": "read_file", "status": "ok", "duration_ms": 1.0, "args_summary": "file2"},
        ],
        "warnings": [],
    }
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    result = agent.generate("Implement planner", index_dir=tmp_path / ".mana/index", k=4)
    policy = _orchestrator_calls(agent)[0]["tool_policy"]
    assert policy["search_budget"] == 4
    assert result["progress"]["budgets"]["search_used"] == 2


def test_coding_agent_full_auto_dynamic_read_policy_clamps_to_caps(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload, full_auto_mode=True)

    class _StructuredPlanner:
        class _Runner:
            def invoke(self, _messages):
                return {"read_budget": 999, "read_line_window": 9999, "reason": "broad discovery"}

        def with_structured_output(self, _schema):
            return self._Runner()

    agent.planner_llm = _StructuredPlanner()
    policy = agent._tool_policy_for_request("update docs and tests")
    assert policy["read_budget"] == 6
    assert policy["read_budget_cap"] == 6
    assert policy["read_line_window"] == 2000
    assert policy["dynamic_read_budget_used"] is True
    assert policy["dynamic_read_budget_fallback_used"] is False


def test_coding_agent_model_docs_read_budget_counts_model_files_and_docs(tmp_path: Path, monkeypatch) -> None:
    for idx in range(9):
        path = tmp_path / "back" / f"app{idx}" / "models.py"
        path.parent.mkdir(parents=True)
        path.write_text("from django.db import models\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "models.md").write_text("# Models\n", encoding="utf-8")
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload, full_auto_mode=True)

    policy = agent._tool_policy_for_request("find all models and update docs/models.md")

    assert policy["read_budget"] == 10
    assert policy["read_budget_cap"] == 10
    assert policy["dynamic_read_budget_used"] is True
    assert policy["dynamic_read_budget_reason"] == "model_docs_inventory"


def test_coding_agent_non_full_auto_keeps_static_read_budget_without_dynamic_invoke(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload, full_auto_mode=False)

    class _NoInvokePlanner:
        def with_structured_output(self, _schema):  # pragma: no cover - should not be called
            raise AssertionError("dynamic planner must not be invoked in non-full-auto")

        def invoke(self, _messages):  # pragma: no cover - should not be called
            raise AssertionError("dynamic planner must not be invoked in non-full-auto")

    agent.planner_llm = _NoInvokePlanner()
    policy = agent._tool_policy_for_request("update docs and tests")
    assert policy["read_budget"] == 6
    assert policy["read_line_window"] == 400
    assert policy["dynamic_read_budget_used"] is False
    assert policy["dynamic_read_budget_fallback_used"] is False


def test_coding_agent_full_auto_dynamic_read_policy_fallback_on_parse_failure(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload, full_auto_mode=True)

    class _BrokenPlanner:
        class _Runner:
            def invoke(self, _messages):
                raise RuntimeError("structured output unavailable")

        def with_structured_output(self, _schema):
            return self._Runner()

        def invoke(self, _messages):
            raise RuntimeError("llm parse failure")

    agent.planner_llm = _BrokenPlanner()
    policy = agent._tool_policy_for_request("update docs and tests")
    assert policy["read_budget"] == 6
    assert policy["read_line_window"] == 400
    assert policy["dynamic_read_budget_used"] is False
    assert policy["dynamic_read_budget_fallback_used"] is True


def test_coding_agent_progress_budgets_include_dynamic_read_metadata(tmp_path: Path, monkeypatch) -> None:
    payload = {
        "answer": "ok",
        "trace": [
            {"tool_name": "read_file", "status": "ok", "duration_ms": 1.0, "output_preview": '{"file_path":"src/a.py"}'},
            {"tool_name": "read_file", "status": "ok", "duration_ms": 1.0, "output_preview": '{"file_path":"src/b.py"}'},
        ],
        "warnings": [],
    }
    agent = _build_agent(tmp_path, monkeypatch, payload=payload, full_auto_mode=True)
    monkeypatch.setattr(
        agent,
        "_dynamic_read_policy_for_request",
        lambda request, flow_context=None: {
            "read_budget": 5,
            "read_line_window": 900,
            "dynamic_read_budget_used": True,
            "dynamic_read_budget_fallback_used": False,
            "dynamic_read_budget_reason": "targeted module sweep",
        },
    )

    result = agent.generate("Implement planner", index_dir=tmp_path / ".mana/index", k=4)
    budgets = result["progress"]["budgets"]
    assert budgets["read_budget"] == 5
    assert budgets["read_budget_cap"] == 6
    assert budgets["read_line_window"] == 900
    assert budgets["dynamic_read_budget_used"] is True
    assert budgets["dynamic_read_budget_fallback_used"] is False


def test_coding_agent_tool_policy_includes_full_read_preferences(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload, full_auto_mode=True)
    monkeypatch.setattr(
        agent,
        "_dynamic_read_policy_for_request",
        lambda request, flow_context=None: {
            "read_budget": 4,
            "read_line_window": 700,
            "dynamic_read_budget_used": True,
            "dynamic_read_budget_fallback_used": False,
            "dynamic_read_budget_reason": "focused file inspection",
        },
    )

    policy = agent._tool_policy_for_request("update README.md and prompts")
    assert policy["read_mode_preference"] == "full_preferred"
    assert policy["read_full_file_max_lines"] == 5000
    assert policy["read_full_file_max_chars"] == 250000
    assert policy["read_cache_scope"] == "flow"
    assert "create_file" in policy["allowed_tools"]


def test_coding_agent_tool_policy_treats_dotgitignore_as_single_file_edit(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload, full_auto_mode=True)
    monkeypatch.setattr(
        agent,
        "_dynamic_read_policy_for_request",
        lambda request, flow_context=None: {
            "read_budget": 4,
            "read_line_window": 700,
            "dynamic_read_budget_used": True,
            "dynamic_read_budget_fallback_used": False,
            "dynamic_read_budget_reason": "focused file inspection",
        },
    )

    policy = agent._tool_policy_for_request("update .gitignore add .mana")

    assert policy["require_read_files"] == 1


def test_coding_agent_progress_budgets_use_cache_aware_read_metrics(tmp_path: Path, monkeypatch) -> None:
    payload = {
        "answer": "ok",
        "trace": [
            {
                "tool_name": "read_file",
                "status": "ok",
                "duration_ms": 1.0,
                "output_preview": '{"file_path":"README.md","mode":"full","cache_hit":false,"cache_source":"disk"}',
            },
            {
                "tool_name": "read_file",
                "status": "ok",
                "duration_ms": 1.0,
                "output_preview": '{"file_path":"README.md","mode":"line","cache_hit":true,"cache_source":"flow_full"}',
            },
        ],
        "warnings": [],
    }
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    result = agent.generate("update README.md", index_dir=tmp_path / ".mana/index", k=4)
    budgets = result["progress"]["budgets"]
    assert budgets["read_used"] == 1
    assert budgets["read_cache_hits"] == 1
    assert budgets["read_cache_misses"] == 1
    assert budgets["read_full_mode_used"] == 1
    assert budgets["read_cache_scope"] == "flow"


def test_flow_checklist_persists_and_resumes_across_turns(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    first = agent.generate("Implement A", index_dir=tmp_path / ".mana/index", k=4)
    second = agent.generate("Continue A", index_dir=tmp_path / ".mana/index", k=4, flow_id=first["flow_id"])
    assert isinstance(first.get("flow_id"), str)
    assert second["flow_id"] == first["flow_id"]
    summary = agent.flow_summary(first["flow_id"])
    assert isinstance(summary, dict)
    assert isinstance(summary.get("checklist"), dict)


def test_coding_agent_propagates_flow_id_to_ask_agent_run(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    result = agent.generate("Implement A", index_dir=tmp_path / ".mana/index", k=4)
    calls = _orchestrator_calls(agent)
    assert calls
    assert calls[0]["flow_id"] == result["flow_id"]


def test_dir_mode_coding_agent_flow_context_included(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    first = agent.generate_dir_mode(
        "Implement dir-mode flow",
        index_dirs=[tmp_path / "proj/.mana/index"],
        k=4,
    )
    assert isinstance(first.get("plan"), dict)


def test_dir_mode_rewrites_ambiguous_followup_with_flow_context(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    first = agent.generate_dir_mode(
        "Refactor parser and update prompts",
        index_dirs=[tmp_path / "proj/.mana/index"],
        k=4,
    )
    flow_id = str(first.get("flow_id") or "")
    second = agent.generate_dir_mode(
        "yes.",
        index_dirs=[tmp_path / "proj/.mana/index"],
        k=4,
        flow_id=flow_id,
    )
    _ = second

    calls = _orchestrator_calls(agent)
    assert len(calls) >= 2
    followup_question = str(calls[1]["request"])
    assert followup_question != "yes."
    assert "Current objective:" in followup_question
    warnings = second.get("warnings") or []
    assert any("followup_request_rewritten_from_flow_context" in str(item) for item in warnings)


def test_plan_trigger_followup_rewrites_to_execute_checklist(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    first = agent.generate(
        "Update docs and type hints in TODO.md",
        index_dir=tmp_path / ".mana/index",
        k=4,
    )
    flow_id = str(first.get("flow_id") or "")
    second = agent.generate(
        "implement plan.",
        index_dir=tmp_path / ".mana/index",
        k=4,
        flow_id=flow_id,
    )

    calls = _orchestrator_calls(agent)
    assert len(calls) >= 2
    followup_question = str(calls[1]["request"])
    assert "Execute the active flow checklist now." in followup_question
    assert "Current objective:" in followup_question
    warnings = second.get("warnings") or []
    assert any("followup_request_rewritten_from_flow_context" in str(item) for item in warnings)


def test_plan_trigger_followup_not_classified_as_conflict(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    first = agent.generate(
        "Refactor coding agent flow prompts and docs",
        index_dir=tmp_path / ".mana/index",
        k=4,
    )
    flow_id = str(first.get("flow_id") or "")
    assert agent.is_conflicting_request("implement plan.", flow_id) is False


def test_coding_agent_without_orchestrator_does_not_call_ask_agent_run(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    agent.tools_manager_orchestrator = None
    result = agent.generate("Implement planner", index_dir=tmp_path / ".mana/index", k=4)
    assert "orchestrator is unavailable" in str(result.get("answer", "")).lower()
    warnings = result.get("warnings") or []
    assert any("auto_execute_orchestrator_unavailable" in str(item) for item in warnings)
    fake = agent.ask_agent
    assert isinstance(fake, _FakeAskAgent)
    assert fake.calls == []


def test_coding_agent_does_not_retry_tools_only_violation_through_orchestrator(tmp_path: Path, monkeypatch) -> None:
    class _RetryOrchestrator:
        def __init__(self) -> None:
            self.calls = 0
            self.kwargs = []

        def run(self, **kwargs):
            self.calls += 1
            self.kwargs.append(dict(kwargs))
            raise ToolWorkerProcessError(
                code="tools_only_violation",
                message="no successful tool calls",
                retriable=False,
            )

    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    orchestrator = _RetryOrchestrator()
    agent.set_tools_manager_orchestrator(orchestrator)
    result = agent.generate("Implement planner", index_dir=tmp_path / ".mana/index", k=4)
    assert orchestrator.calls == 1
    assert result.get("render_mode") == "answer_only"
    assert result.get("fallback_reason") == "tools_only_violation"
    assert result.get("fallback_retry_attempted") is False
    fake = agent.ask_agent
    assert isinstance(fake, _FakeAskAgent)
    assert fake.calls == []


def test_coding_agent_provider_error_does_not_fallback_to_direct_ask_agent(tmp_path: Path, monkeypatch) -> None:
    class _BadProviderOrchestrator:
        def run(self, **_kwargs):
            raise ToolWorkerProcessError(
                code="run_failed",
                message="Error code: 400 - {'error': {'message': 'openai_error', 'type': 'bad_response_status_code'}}",
                retriable=False,
            )

    payload = {"answer": "direct fallback must not run", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    agent.set_tools_manager_orchestrator(_BadProviderOrchestrator())

    result = agent.generate("Implement planner", index_dir=tmp_path / ".mana/index", k=4)

    assert result.get("fallback_reason") == "run_failed"
    assert result.get("fallback_retry_attempted") is False

    fake = agent.ask_agent
    assert isinstance(fake, _FakeAskAgent)
    assert fake.calls == []


def test_coding_agent_does_not_retry_after_patch_failure(tmp_path: Path, monkeypatch) -> None:
    ask_agent = _SequenceAskAgent(
        [
            {
                "answer": "first",
                "trace": [
                    {"tool_name": "read_file", "status": "ok", "output_preview": '{"file_path":"README.md"}'},
                    {"tool_name": "apply_patch", "status": "error"},
                    {"tool_name": "apply_patch", "status": "error"},
                ],
                "warnings": [],
            }
        ]
    )
    agent = _build_agent_with_ask(tmp_path, monkeypatch, ask_agent)
    monkeypatch.setattr(agent, "_git_status_paths", lambda: set())  # type: ignore[method-assign]

    result = agent.generate("update README section", index_dir=tmp_path / ".mana/index", k=4)
    _ = result
    calls = _orchestrator_calls(agent)
    assert len(calls) == 1
    warnings = result.get("warnings") or []
    assert any("mutation_failed_no_changes" in str(item) for item in warnings)
    assert result["progress"]["phase"] == "blocked"


def test_coding_agent_noop_patch_does_not_attempt_write_retry(tmp_path: Path, monkeypatch) -> None:
    ask_agent = _SequenceAskAgent(
        [
            {
                "answer": "first",
                "trace": [
                    {"tool_name": "read_file", "status": "ok", "output_preview": '{"file_path":"README.md"}'},
                    {"tool_name": "apply_patch", "status": "error"},
                    {"tool_name": "apply_patch", "status": "error"},
                ],
                "warnings": [],
            },
        ]
    )
    agent = _build_agent_with_ask(tmp_path, monkeypatch, ask_agent)
    monkeypatch.setattr(agent, "_git_status_paths", lambda: set())  # type: ignore[method-assign]

    result = agent.generate("update README section", index_dir=tmp_path / ".mana/index", k=4)
    calls = _orchestrator_calls(agent)
    assert len(calls) == 1
    assert result["progress"]["phase"] == "blocked"
    assert result["changed_files"] == []


def test_coding_agent_true_blocker_after_mutation_noop(tmp_path: Path, monkeypatch) -> None:
    ask_agent = _SequenceAskAgent(
        [
            {
                "answer": "first",
                "trace": [
                    {"tool_name": "read_file", "status": "ok", "output_preview": '{"file_path":"README.md"}'},
                    {"tool_name": "apply_patch", "status": "error"},
                    {"tool_name": "apply_patch", "status": "error"},
                ],
                "warnings": [],
            },
        ]
    )
    agent = _build_agent_with_ask(tmp_path, monkeypatch, ask_agent)
    monkeypatch.setattr(agent, "_git_status_paths", lambda: set())  # type: ignore[method-assign]

    result = agent.generate("update README section", index_dir=tmp_path / ".mana/index", k=4)
    assert result["progress"]["phase"] == "blocked"
    assert "mutation_failed_no_changes" in str(result["progress"]["why"])
    warnings = result.get("warnings") or []
    assert any("mutation_failed_no_changes" in str(item) for item in warnings)


def test_coding_agent_conversational_noop_retries_before_terminal(tmp_path: Path, monkeypatch) -> None:
    ask_agent = _SequenceAskAgent(
        [
            {
                "answer": "If you want, I can proceed with edits.",
                "trace": [
                    {"tool_name": "read_file", "status": "ok", "output_preview": '{"file_path":"README.md"}'},
                ],
                "warnings": [],
            },
            {
                "answer": "completed edit",
                "trace": [
                    {"tool_name": "write_file", "status": "ok"},
                ],
                "warnings": [],
            },
        ]
    )
    agent = _build_agent_with_ask(tmp_path, monkeypatch, ask_agent)
    states = iter([set(), set(), {"README.md"}])
    monkeypatch.setattr(agent, "_git_status_paths", lambda: next(states, {"README.md"}))  # type: ignore[method-assign]

    result = agent.generate("please update readme.md", index_dir=tmp_path / ".mana/index", k=4)
    assert len(_orchestrator_calls(agent)) >= 2
    assert result["changed_files"] == ["README.md"]
    warnings = result.get("warnings") or []
    assert any("edit_intent_conversational_noop_detected" in str(item) for item in warnings)


def test_plan_checklist_falls_back_when_planner_json_is_invalid(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    agent._plan_checklist = CodingAgent._plan_checklist.__get__(agent, CodingAgent)  # type: ignore[method-assign]

    class _BadPlanner:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            # first call: invalid schema text, second call: invalid repair text
            if self.calls == 1:
                return type("_Msg", (), {"content": "not json"})()
            return type("_Msg", (), {"content": "```json\n{ broken }\n```"})()

    agent.planner_llm = _BadPlanner()
    checklist, warnings = agent._plan_checklist("update README.md with new version")
    assert checklist is not None
    assert checklist.steps
    assert any("planner parse failed" in str(item) for item in warnings)


def test_generate_auto_execute_delegates_to_tools_manager_orchestrator(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)

    class _Orchestrator:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, **_kwargs):
            self.calls += 1
            return AutoExecuteResult(
                answer="auto-complete",
                trace=[{"tool_name": "read_file", "status": "ok", "duration_ms": 1.0, "args_summary": "x"}],
                warnings=[],
                changed_files=[],
                plan={"objective": "Implement request", "steps": []},
                passes=1,
                terminal_reason="manager_stop",
                toolsmanager_requests_count=1,
                pass_logs=[{"pass_index": 1, "requests_count": 1}],
            )

    orchestrator = _Orchestrator()
    agent.set_tools_manager_orchestrator(orchestrator)
    result = agent.generate_auto_execute(
        "Implement planner",
        index_dir=tmp_path / ".mana/index",
        k=4,
        pass_cap=3,
    )
    assert orchestrator.calls == 1
    assert result["answer"] == "auto-complete"
    assert result["auto_execute_passes"] == 1
    assert result["auto_execute_terminal_reason"] == "manager_stop"


def test_generate_auto_execute_propagates_flow_id_to_tools_manager(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)

    class _Orchestrator:
        def __init__(self) -> None:
            self.seen_flow_id = None

        def run(self, **kwargs):
            self.seen_flow_id = kwargs.get("flow_id")
            return AutoExecuteResult(
                answer="auto-complete",
                trace=[],
                warnings=[],
                changed_files=[],
                plan={"objective": "Implement request", "steps": []},
                passes=1,
                terminal_reason="manager_stop",
                toolsmanager_requests_count=1,
                pass_logs=[{"pass_index": 1, "requests_count": 1}],
            )

    orchestrator = _Orchestrator()
    agent.set_tools_manager_orchestrator(orchestrator)
    result = agent.generate_auto_execute(
        "Implement planner",
        index_dir=tmp_path / ".mana/index",
        k=4,
        pass_cap=3,
        flow_id="flow-xyz",
    )
    assert orchestrator.seen_flow_id == "flow-xyz"
    assert result["flow_id"] == "flow-xyz"


def test_generate_auto_execute_returns_optional_dedup_retry_telemetry(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)

    class _Orchestrator:
        def run(self, **_kwargs):
            return AutoExecuteResult(
                answer="auto-complete",
                trace=[],
                warnings=[],
                changed_files=[],
                plan={"objective": "Implement request", "steps": []},
                passes=1,
                terminal_reason="manager_stop",
                toolsmanager_requests_count=1,
                pass_logs=[],
                duplicate_request_skips=2,
                duplicate_semantic_search_skips=3,
                request_retry_attempts=1,
                request_retry_exhausted=1,
                edit_retry_mode_activations=1,
                persisted_fingerprint_counts={"request_fingerprint": 4},
            )

    agent.set_tools_manager_orchestrator(_Orchestrator())
    result = agent.generate_auto_execute("Implement planner", index_dir=tmp_path / ".mana/index", k=4, pass_cap=3)
    assert result["duplicate_request_skips"] == 2
    assert result["duplicate_semantic_search_skips"] == 3
    assert result["request_retry_attempts"] == 1
    assert result["request_retry_exhausted"] == 1
    assert result["edit_retry_mode_activations"] == 1
    assert result["persisted_fingerprint_counts"] == {"request_fingerprint": 4}


def test_generate_auto_execute_preserves_flow_id_continuity(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)

    class _Orchestrator:
        def run(self, **_kwargs):
            return AutoExecuteResult(
                answer="auto",
                trace=[],
                warnings=[],
                changed_files=[],
                plan={"objective": "Implement request", "steps": []},
                passes=1,
                terminal_reason="manager_stop",
                toolsmanager_requests_count=1,
                pass_logs=[{"pass_index": 1, "requests_count": 1}],
            )

    agent.set_tools_manager_orchestrator(_Orchestrator())
    first = agent.generate_auto_execute("Implement A", index_dir=tmp_path / ".mana/index", k=4)
    second = agent.generate_auto_execute(
        "Continue A",
        index_dir=tmp_path / ".mana/index",
        k=4,
        flow_id=first["flow_id"],
    )
    assert isinstance(first.get("flow_id"), str)
    assert second.get("flow_id") == first.get("flow_id")


def test_plan_checklist_parses_python_literal_payload_without_repair(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    agent._plan_checklist = CodingAgent._plan_checklist.__get__(agent, CodingAgent)  # type: ignore[method-assign]

    class _LiteralPlanner:
        def invoke(self, _messages):
            return type(
                "_Msg",
                (),
                {
                    "content": (
                        "{'objective': 'Update README', 'constraints': [], 'acceptance': ['done'], "
                        "'steps': [{'id': 's1', 'title': 'Inspect README.md', 'reason': 'Gather context', "
                        "'status': 'in_progress', 'requires_tools': ['read_file']}], "
                        "'next_action': 'Inspect README.md'}"
                    )
                },
            )()

    agent.planner_llm = _LiteralPlanner()
    checklist, warnings = agent._plan_checklist("update README.md with new version")
    assert checklist is not None
    assert checklist.objective == "Update README"
    assert checklist.steps
    assert warnings == []


def test_plan_checklist_parses_list_blocks_with_plan_text(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    agent._plan_checklist = CodingAgent._plan_checklist.__get__(agent, CodingAgent)  # type: ignore[method-assign]

    class _ListPlanner:
        def invoke(self, _messages):
            return type(
                "_Msg",
                (),
                {
                    "content": (
                        "[{'id': 'rs_1', 'summary': [], 'type': 'reasoning'}, "
                        "{'type': 'text', 'text': 'Plan:\\n1. Inspect README.md\\n2. Update docs\\n3. Verify formatting'}]"
                    )
                },
            )()

    agent.planner_llm = _ListPlanner()
    checklist, warnings = agent._plan_checklist("update README.md with new version")
    assert checklist is not None
    assert len(checklist.steps) >= 3
    assert checklist.steps[0].title.startswith("Inspect README.md")
    assert warnings == []


def test_plan_checklist_parses_wrapped_answer_markdown_payload_without_repair(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    agent._plan_checklist = CodingAgent._plan_checklist.__get__(agent, CodingAgent)  # type: ignore[method-assign]

    class _WrappedAnswerPlanner:
        def invoke(self, _messages):
            wrapped = {
                "answer": (
                    "[{'id': 'rs_1', 'summary': [], 'type': 'reasoning'}, "
                    "{'type': 'text', 'text': '**Execution Plan**\\n"
                    "1. Inspect README.md\\n2. Update docs\\n3. Verify formatting'}]"
                )
            }
            return type("_Msg", (), {"content": json.dumps(wrapped)})()

    agent.planner_llm = _WrappedAnswerPlanner()
    checklist, warnings = agent._plan_checklist("update README.md with new version")
    assert checklist is not None
    assert len(checklist.steps) >= 3
    assert checklist.steps[0].title.startswith("Inspect README.md")
    assert warnings == []


def test_plan_checklist_parses_wrapped_content_markdown_fence_without_repair(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    agent._plan_checklist = CodingAgent._plan_checklist.__get__(agent, CodingAgent)  # type: ignore[method-assign]

    class _WrappedContentPlanner:
        def invoke(self, _messages):
            wrapped = {
                "content": (
                    "```markdown\n"
                    "**Execution Plan**\n"
                    "1. Inspect README.md\n"
                    "2. Apply targeted patch\n"
                    "3. Verify with tests\n"
                    "```"
                )
            }
            return type("_Msg", (), {"content": json.dumps(wrapped)})()

    agent.planner_llm = _WrappedContentPlanner()
    checklist, warnings = agent._plan_checklist("update README.md with new version")
    assert checklist is not None
    assert len(checklist.steps) >= 3
    assert checklist.steps[0].title.startswith("Inspect README.md")
    assert warnings == []


def test_parse_flow_checklist_json_uses_nested_answer_field() -> None:
    wrapped = {
        "answer": (
            "[{'id': 'rs_1', 'summary': [], 'type': 'reasoning'}, "
            "{'type': 'text', 'text': 'Execution Plan:\\n"
            "1. Inspect src/example.py\\n2. Apply change\\n3. Verify output'}]"
        )
    }
    checklist = CodingAgent._parse_flow_checklist_json(
        json.dumps(wrapped),
        request="update src/example.py",
    )
    assert checklist is not None
    assert len(checklist.steps) >= 3
    assert checklist.steps[0].title.startswith("Inspect src/example.py")


def test_preview_execution_checklist_uses_planner_and_persists_to_flow_memory(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    monkeypatch.setattr(
        agent,
        "_plan_checklist_with_source",
        lambda request, flow_context=None: (_fixed_checklist(), [], "planner"),
    )

    preview = agent.preview_execution_checklist("implement plan.")
    assert preview["prechecklist_source"] == "planner"
    assert isinstance(preview.get("prechecklist"), dict)
    assert preview["prechecklist"]["objective"] == "Implement request"
    flow_id = preview.get("flow_id")
    assert isinstance(flow_id, str) and flow_id
    summary = agent.flow_summary(flow_id)
    assert isinstance(summary, dict)
    checklist = summary.get("checklist")
    assert isinstance(checklist, dict)
    assert checklist.get("objective") == "Implement request"


def test_preview_execution_checklist_reports_repair_source(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    monkeypatch.setattr(
        agent,
        "_plan_checklist_with_source",
        lambda request, flow_context=None: (_fixed_checklist(), ["planner parse failed; attempting repair"], "planner_repair"),
    )
    preview = agent.preview_execution_checklist("implement plan.")
    assert preview["prechecklist_source"] == "planner_repair"
    assert str(preview.get("prechecklist_warning", "")).strip() == ""


def test_preview_execution_checklist_surfaces_deterministic_fallback_warning(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)
    monkeypatch.setattr(
        agent,
        "_plan_checklist_with_source",
        lambda request, flow_context=None: (_fixed_checklist(), ["planner parse failed"], "deterministic_fallback"),
    )
    preview = agent.preview_execution_checklist("implement plan.")
    assert preview["prechecklist_source"] == "deterministic_fallback"
    assert "deterministic fallback checklist" in str(preview.get("prechecklist_warning", "")).lower()


def test_explicit_file_heading_task_skips_planner_questions(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "README.md").write_text("# Demo\n\n## Project Layout\n\nOld\n", encoding="utf-8")
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)

    class _FailingPlanner:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            raise AssertionError("planner should not be called for clear target tasks")

    planner = _FailingPlanner()
    agent.planner_llm = planner

    checklist, warnings, source = agent._plan_checklist_with_source("task:update readme.md ## Project Layout")

    assert source == "deterministic_clear_task"
    assert warnings == []
    assert checklist is not None
    assert checklist.requires_edit is True
    assert checklist.target_files == ["README.md"]
    assert planner.calls == 0


def test_planner_failure_circuit_breaker_uses_fallback_once(tmp_path: Path, monkeypatch) -> None:
    payload = {"answer": "ok", "trace": [], "warnings": []}
    agent = _build_agent(tmp_path, monkeypatch, payload=payload)

    class _FailingPlanner:
        calls = 0

        def invoke(self, _messages):
            self.calls += 1
            raise RuntimeError("invalid api key")

    planner = _FailingPlanner()
    agent.planner_llm = planner

    first, first_warnings, first_source = agent._plan_checklist_with_source("explain the project structure")
    second, second_warnings, second_source = agent._plan_checklist_with_source("explain another area")

    assert first is not None
    assert second is not None
    assert first_source == "deterministic_fallback"
    assert second_source == "deterministic_fallback"
    assert planner.calls == 1
    assert any("planner unavailable" in warning for warning in first_warnings)
    assert second_warnings == ["planner unavailable; using deterministic checklist"]
