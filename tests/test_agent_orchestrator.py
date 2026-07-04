from __future__ import annotations

import logging
from pathlib import Path

from mana_agent.agent.orchestrator import AgentOrchestrator
from mana_agent.agent.task_classifier import classify_task
from mana_agent.agent.verification_planner import plan_verification
from mana_agent.llm.agent_work_queue import AgentWorkQueue, WorkItem, WorkQueueRunner, WorkResult
from mana_agent.llm.tool_worker_process import ToolRunRequest, ToolRunResponse
from mana_agent.llm.tools_executor import (
    ToolsExecutionConfig,
    _FALLBACK_WARNINGS_EMITTED,
    build_tools_executor_with_fallback,
)


def test_readme_project_layout_task_classifies_as_single_file_section(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n\n## Project Layout\n\nOld\n", encoding="utf-8")

    decision = classify_task(
        "task:update readme.md ## Project Layout",
        repo_root=tmp_path,
    )

    assert decision.task_type == "mutation_required"
    assert decision.target_files == ("README.md",)
    assert decision.target_sections == ("Project Layout",)
    assert decision.needs_repo_search is False
    assert decision.needs_file_read is True
    assert decision.needs_mutation is True
    assert decision.needs_verification is True
    assert decision.scope == "single_file_section"
    assert decision.confidence >= 0.9


def test_enough_evidence_gate_skips_unrelated_docs_reads(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n\n## Project Layout\n\nOld\n", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "01-overview.md").write_text("# Overview\n", encoding="utf-8")

    orchestrator = AgentOrchestrator.start(
        "task:update readme.md ## Project Layout",
        repo_root=tmp_path,
        requires_edit=True,
    )
    queue = AgentWorkQueue()
    queue.submit(WorkItem(kind="read", tool_name="read_file", tool_args={"path": "README.md"}, priority=10))
    queue.submit(WorkItem(kind="read", tool_name="read_file", tool_args={"path": "docs/01-overview.md"}, priority=20))
    calls: list[str] = []

    def execute(item: WorkItem) -> WorkResult:
        path = str(item.tool_args["path"])
        calls.append(path)
        return WorkResult(ok=True, files_read=[path], answer=(tmp_path / path).read_text(encoding="utf-8"))

    report = WorkQueueRunner(queue=queue, execute=execute, orchestrator=orchestrator, max_steps=5).run()

    assert calls == ["README.md"]
    assert report.skipped == 1
    skipped = [item for item in report.items if item.status == "skipped"]
    assert skipped[0].tool_args["path"] == "docs/01-overview.md"
    assert skipped[0].error == "evaluation_gate_evidence_sufficient"
    assert any(row.get("decision") == "start_mutation" for row in orchestrator.trace)


def test_docs_only_verification_selects_task_profile() -> None:
    decision = plan_verification(changed_files=["README.md"])

    assert decision.verification_profile == "task_verification"
    assert decision.commands == ("git status --short", "git diff -- README.md")
    assert decision.skip_full_pytest_reason == "README-only documentation change"


def test_core_agent_change_selects_project_verification() -> None:
    decision = plan_verification(changed_files=["src/mana_agent/agent/orchestrator.py"], core_agent_change=True)

    assert decision.verification_profile == "project_verification"
    assert decision.commands == ("pytest -q",)


def test_fake_worker_implements_lifecycle_protocol(caplog) -> None:
    class _FakeWorkerClient:
        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def run_tools(self, request: ToolRunRequest, on_event=None) -> ToolRunResponse:  # noqa: ANN001
            _ = (request, on_event)
            return ToolRunResponse(answer="ok", trace=[{"tool_name": "read_file", "status": "ok"}])

    worker = _FakeWorkerClient()
    with caplog.at_level(logging.ERROR):
        worker.start()
        response = worker.run_tools(ToolRunRequest(question="read"))
        worker.stop()

    assert response.answer == "ok"
    assert "_FakeWorkerClient" not in caplog.text
    assert "AttributeError" not in caplog.text


def test_redis_fallback_warning_is_deduped(caplog) -> None:
    class _Local:
        def __init__(self, *, worker_client) -> None:  # noqa: ANN001
            self.worker_client = worker_client

    class _Redis:
        def __init__(self, **_kwargs) -> None:  # noqa: ANN001
            raise RuntimeError("redis unavailable")

    warnings: list[str] = []
    _FALLBACK_WARNINGS_EMITTED.discard("test-run")
    with caplog.at_level(logging.WARNING):
        first = build_tools_executor_with_fallback(
            worker_client=object(),
            config=ToolsExecutionConfig(backend="redis"),
            worker_init_payload={},
            warnings=warnings,
            warning_key="test-run",
            local_executor_cls=_Local,
            redis_executor_cls=_Redis,
        )
        second = build_tools_executor_with_fallback(
            worker_client=object(),
            config=ToolsExecutionConfig(backend="redis"),
            worker_init_payload={},
            warnings=warnings,
            warning_key="test-run",
            local_executor_cls=_Local,
            redis_executor_cls=_Redis,
        )

    assert first.__class__.__name__ == "_Local"
    assert second.__class__.__name__ == "_Local"
    assert len(warnings) == 1
    assert caplog.text.count("redis executor unavailable") == 1
