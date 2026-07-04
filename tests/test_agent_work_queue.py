from __future__ import annotations

import json
import subprocess
from pathlib import Path

from mana_agent.llm.agent_work_queue import (
    AgentWorkQueue,
    EventBus,
    TaskBoard,
    WorkItem,
    WorkQueueRunner,
    WorkResult,
    compute_fingerprint,
)
from mana_agent.llm.agent_work_queue_adapters import (
    CodingAgentSniffer,
    classify_result,
    make_worker_executor,
)
from mana_agent.llm.tool_worker_process import ToolRunResponse


# Edit/forced passes run with mutation tools only; discovery/read jobs gather
# evidence before the edit item is claimed.
_AGENTIC_EDIT_TOOLS = [
    "edit_file",
    "multi_edit_file",
    "apply_patch",
    "write_file",
    "create_file",
    "delete_file",
]


def _substantive(title: str) -> str:
    """Body long enough to clear the deliverable stub check (>=120 chars)."""
    return f"# {title}\n\n" + ("Real, substantive content describing the project. " * 6) + "\n"


def _write_real(repo_root: Path, rel: str) -> None:
    """Author a deliverable on disk the way a real mutation tool would."""
    target = Path(repo_root) / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_substantive(rel), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Fingerprint / dedup
# --------------------------------------------------------------------------- #
def test_same_read_path_collapses_to_one_fingerprint(tmp_path: Path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "x.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    a = compute_fingerprint(kind="read", tool_name="read_file", tool_args={"path": "src/x.py"})
    b = compute_fingerprint(kind="read", tool_name="read_file", tool_args={"path": "./src/x.py "})
    c = compute_fingerprint(kind="read", tool_name="read_file", tool_args={"path": str(tmp_path / "src" / "x.py")})
    assert a == b
    assert b == c


def test_queue_rejects_duplicate_idempotent_jobs():
    q = AgentWorkQueue()
    assert q.submit(WorkItem(kind="read", tool_name="read_file", tool_args={"path": "src/x.py"}))
    # Identical read should be suppressed.
    assert not q.submit(WorkItem(kind="read", tool_name="read_file", tool_args={"path": "src/x.py"}))
    assert len(q.items()) == 1


def test_board_not_complete_with_failed_edit_step() -> None:
    q = AgentWorkQueue()
    edit = WorkItem(kind="edit", tool_name="write_file", question="update docs/08-architecture.md")
    verify = WorkItem(kind="verify", tool_name="verify", dependencies=[edit.id])
    q.submit(edit)
    q.submit(verify)
    q.complete(edit.id, status="failed", result=WorkResult(ok=False, error="mutation did not run"))

    snap = q.snapshot()
    assert snap["complete"] is False
    assert snap["remaining"] == 2
    assert snap["failed"] == 1
    assert snap["blocked"] == 1


def test_duplicate_mutation_work_items_same_target_collapse() -> None:
    q = AgentWorkQueue()
    first = WorkItem(
        kind="edit",
        tool_name="write_file",
        tool_args={"path": "docs/08-architecture.md"},
        question="update architecture docs",
    )
    second = WorkItem(
        kind="edit",
        tool_name="write_file",
        tool_args={"path": "docs/08-architecture.md"},
        question="update architecture docs",
    )

    assert q.submit(first) is True
    assert q.submit(second) is False
    assert len(q.items()) == 1


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
def test_dependencies_gate_readiness_and_block_on_failure():
    q = AgentWorkQueue()
    parent = WorkItem(kind="search", tool_name="repo_search", tool_args={"query": "x"})
    child = WorkItem(kind="edit", tool_name="apply_patch", tool_args={"path": "a.py"}, dependencies=[parent.id])
    q.submit(parent)
    q.submit(child)

    claimed = q.claim()
    assert claimed.id == parent.id  # child not runnable yet
    assert q.claim() is None  # nothing else ready while parent runs

    q.complete(parent.id, status="failed", result=WorkResult(ok=False, error="boom"))
    assert q.get(child.id).status == "blocked"
    assert q.is_drained()


def test_child_runs_after_parent_done():
    q = AgentWorkQueue()
    parent = WorkItem(kind="search", tool_name="repo_search", tool_args={"query": "x"}, priority=10)
    child = WorkItem(kind="read", tool_name="read_file", tool_args={"path": "a.py"}, dependencies=[parent.id])
    q.submit(parent)
    q.submit(child)
    q.claim()
    q.complete(parent.id, status="done", result=WorkResult(ok=True))
    nxt = q.claim()
    assert nxt.id == child.id


# --------------------------------------------------------------------------- #
# Read-success fix: a read with content is NOT no_progress
# --------------------------------------------------------------------------- #
def test_read_success_does_not_require_path_in_prose():
    item = WorkItem(kind="read", tool_name="read_file", tool_args={"path": "src/pkg/mod.py"})
    # Worker returned the file body but never echoed the path string.
    response = ToolRunResponse(answer="def foo():\n    return 1\n", trace=[{"tool": "read_file", "status": "ok"}])
    result = classify_result(item, response, repo_root=Path("/nonexistent"))
    assert result.ok is True
    assert "src/pkg/mod.py" in result.files_read  # bookkeeping still records target


def test_read_failure_when_worker_errors():
    item = WorkItem(kind="read", tool_name="read_file", tool_args={"path": "src/pkg/mod.py"})
    response = ToolRunResponse(answer="", trace=[{"tool": "read_file", "status": "error", "error": "missing"}])
    result = classify_result(item, response, repo_root=Path("/nonexistent"))
    assert result.ok is False


def test_mutation_result_requires_changed_files():
    item = WorkItem(kind="edit", tool_name="apply_patch", tool_args={"path": "docs/overview.md"})
    response = ToolRunResponse(answer="patched", trace=[{"tool_name": "apply_patch", "status": "ok", "changed_files": []}])

    result = classify_result(item, response, repo_root=Path("/nonexistent"))

    assert result.ok is False
    assert result.error == "mutation_no_modified_files"


def test_delete_result_reports_deleted_file_as_changed():
    item = WorkItem(kind="edit", tool_name="delete_file", tool_args={"path": "src/old.py"})
    response = ToolRunResponse(
        answer="deleted",
        trace=[{"tool_name": "delete_file", "status": "ok", "files_changed": ["src/old.py"]}],
    )

    result = classify_result(item, response, repo_root=Path("/nonexistent"))

    assert result.ok is True
    assert result.files_changed == ["src/old.py"]


# --------------------------------------------------------------------------- #
# Live loop: executor runs each fingerprint exactly once
# --------------------------------------------------------------------------- #
def test_runner_executes_each_read_once_not_twice():
    q = AgentWorkQueue()
    for path in ("a.py", "b.py", "c.py"):
        q.submit(WorkItem(kind="read", tool_name="read_file", tool_args={"path": path}))

    calls: list[str] = []

    def execute(item: WorkItem) -> WorkResult:
        calls.append(item.tool_args["path"])
        return WorkResult(ok=True, summary="ok")

    runner = WorkQueueRunner(queue=q, execute=execute, max_steps=20)
    report = runner.run()

    assert report.done == 3
    assert sorted(calls) == ["a.py", "b.py", "c.py"]  # each read exactly once
    assert len(calls) == 3  # no double reads


def test_runner_retries_transient_failure_then_succeeds():
    q = AgentWorkQueue()
    q.submit(WorkItem(kind="read", tool_name="read_file", tool_args={"path": "a.py"}, max_attempts=2))
    attempts = {"n": 0}

    def execute(item: WorkItem) -> WorkResult:
        attempts["n"] += 1
        return WorkResult(ok=attempts["n"] >= 2)

    report = WorkQueueRunner(queue=q, execute=execute, max_steps=10).run()
    assert report.done == 1
    assert attempts["n"] == 2


# --------------------------------------------------------------------------- #
# Sniffer: discovery emits reads
# --------------------------------------------------------------------------- #
def test_sniffer_emits_reads_from_discovery(tmp_path: Path):
    (tmp_path / "found.py").write_text("x = 1\n")
    q = AgentWorkQueue()
    board = TaskBoard(queue=q)
    sniffer = CodingAgentSniffer(repo_root=tmp_path)

    search = WorkItem(kind="search", tool_name="repo_search", tool_args={"query": "x"})
    result = WorkResult(ok=True, files_discovered=["found.py"])
    new_items = sniffer.on_result(search, result, board=board)
    assert len(new_items) == 1
    assert new_items[0].kind == "read"
    assert new_items[0].tool_args["path"] == "found.py"
    assert new_items[0].created_by == "coding_agent_sniffer"


def test_end_to_end_search_then_sniffed_reads(tmp_path: Path):
    (tmp_path / "mod_a.py").write_text("import os\n")
    (tmp_path / "mod_b.py").write_text("import sys\n")
    q = AgentWorkQueue()
    board = TaskBoard(queue=q)
    q.submit(WorkItem(kind="search", tool_name="repo_search", tool_args={"query": "mod"}, priority=10))

    def execute(item: WorkItem) -> WorkResult:
        if item.kind == "search":
            return WorkResult(ok=True, files_discovered=["mod_a.py", "mod_b.py"])
        return WorkResult(ok=True, summary=f"read {item.tool_args['path']}")

    sniffer = CodingAgentSniffer(repo_root=tmp_path)
    report = WorkQueueRunner(queue=q, execute=execute, sniffer=sniffer, board=board, max_steps=20).run()

    # 1 search + 2 sniffed reads, all done, nothing duplicated.
    assert report.done == 3
    assert report.emitted_by_sniffer == 2
    assert report.terminal_reason == "drained"


def test_sniffer_emits_edit_and_verify_after_discovery(tmp_path: Path):
    (tmp_path / "found.py").write_text("x = 1\n")
    q = AgentWorkQueue()
    board = TaskBoard(queue=q)
    sniffer = CodingAgentSniffer(
        repo_root=tmp_path, request="create docs/analyze.md and link it", emit_edit=True
    )

    search = WorkItem(kind="search", tool_name="repo_search", tool_args={"query": "x"})
    result = WorkResult(ok=True, files_discovered=["found.py"])
    new_items = sniffer.on_result(search, result, board=board)

    kinds = [it.kind for it in new_items]
    assert kinds == ["read", "edit", "verify"]
    edit = next(it for it in new_items if it.kind == "edit")
    verify = next(it for it in new_items if it.kind == "verify")
    # Edit/verify run after reads (higher priority number) and verify waits on edit.
    read = next(it for it in new_items if it.kind == "read")
    assert edit.priority > read.priority
    assert verify.dependencies == [edit.id]

    # Finalization is emitted exactly once even across multiple discoveries.
    again = sniffer.on_result(search, result, board=board)
    assert all(it.kind != "edit" for it in again)


def test_sniffer_without_edit_signal_does_not_finalize(tmp_path: Path):
    # No emit_edit signal from the planner: never invent an edit from request text.
    sniffer = CodingAgentSniffer(repo_root=tmp_path, request="add a docs folder and describe the project")
    board = TaskBoard(queue=AgentWorkQueue())
    search = WorkItem(kind="search", tool_name="repo_search", tool_args={"query": "x"})
    new_items = sniffer.on_result(search, WorkResult(ok=True, files_discovered=[]), board=board)
    assert all(it.kind not in {"edit", "verify"} for it in new_items)


def test_queue_manager_runs_edit_and_verify_for_mutating_request(tmp_path: Path):
    """End-to-end through the LIVE path (QueueManager.run), with a fake worker."""
    from mana_agent.llm.tool_worker_process import ToolRunResponse
    from mana_agent.llm.agent_work_queue import QueueManager

    (tmp_path / "found.py").write_text("x = 1\n")

    class _FakeWorker:
        def __init__(self) -> None:
            self.questions: list[str] = []

        def run_tools(self, request, on_event=None):  # noqa: ANN001
            self.questions.append(request.question)
            if (request.tool_name or "") == "repo_search":
                # Surface a real candidate file so the sniffer emits a read.
                return ToolRunResponse(
                    answer="candidate: found.py",
                    sources=[],
                    mode="agent-tools",
                    trace=[{"tool_name": "repo_search", "status": "ok"}],
                    warnings=[],
                )
            if "MutationCommand" in str(request.question):
                return ToolRunResponse(
                    answer=json.dumps(
                        {
                            "tool_name": "write_file",
                            "tool_args": {"path": "docs/overview.md", "content": _substantive("Overview")},
                        }
                    ),
                    sources=[],
                    mode="agent-tools",
                    trace=[],
                    warnings=[],
                )
            return ToolRunResponse(
                answer="ok",
                sources=[],
                mode="agent-tools",
                trace=[
                    {
                        "tool_name": request.tool_name or "tool",
                        "status": "ok",
                        "changed_files": [],
                    }
                ],
                warnings=[],
            )

    worker = _FakeWorker()
    mgr = QueueManager(worker_client=worker, repo_root=tmp_path)
    # requires_edit is the planner-recognized signal threaded down from CodingAgent.
    result = mgr.run(
        request="add a docs folder and describe the project",
        index_dir=str(tmp_path),
        requires_edit=True,
        target_files=["docs/overview.md"],
    )

    joined = "\n".join(worker.questions)
    assert "MutationCommand" in joined  # command synthesis ran
    assert "Target file: docs/overview.md" in joined
    assert "Verify the changes" in joined       # the verify job ran (after edit)
    assert result.execution_backend == "work_queue"
    assert result.run_status == "completed"
    assert result.changed_files == ["docs/overview.md"]


def test_queue_manager_targets_default_skill_registry_without_framework_search_loops(tmp_path: Path):
    from mana_agent.llm.agent_work_queue import QueueManager

    manager = tmp_path / "src" / "mana_agent" / "skills" / "manager.py"
    skills_dir = tmp_path / "src" / "mana_agent" / "default_skills"
    manager.parent.mkdir(parents=True)
    skills_dir.mkdir(parents=True)
    manager.write_text("DEFAULT_SKILL_NAMES = ()\n_KEYWORDS = {}\n", encoding="utf-8")
    (tmp_path / "src" / "mana_agent" / "dependencies").mkdir(parents=True)
    (tmp_path / "src" / "mana_agent" / "dependencies" / "dependency_service.py").write_text(
        "react fastapi nextjs nestjs\n",
        encoding="utf-8",
    )

    class _FakeWorker:
        def __init__(self) -> None:
            self.requests: list[object] = []

        def run_tools(self, request, on_event=None):  # noqa: ANN001
            self.requests.append(request)
            name = request.tool_name or ""
            if name == "repo_search":
                assert request.tool_args == {"query": "DEFAULT_SKILL_NAMES"}
                return ToolRunResponse(
                    answer="src/mana_agent/skills/manager.py",
                    mode="agent-tools",
                    trace=[{"tool_name": "repo_search", "status": "ok", "result": "src/mana_agent/skills/manager.py"}],
                )
            if name == "list_files":
                assert request.tool_args == {"glob": "src/mana_agent/default_skills/*.md"}
                return ToolRunResponse(
                    answer="src/mana_agent/default_skills/vue.md",
                    mode="agent-tools",
                    trace=[{"tool_name": "list_files", "status": "ok"}],
                )
            if "MutationCommand" in str(request.question):
                target_file = request.question.split("Target file:", 1)[1].split(". User goal", 1)[0].strip()
                return ToolRunResponse(
                    answer=json.dumps(
                        {
                            "tool_name": "write_file",
                            "tool_args": {"path": target_file, "content": _substantive(target_file)},
                        }
                    ),
                    mode="agent-tools",
                    trace=[],
                )
            return ToolRunResponse(
                answer="ok",
                mode="agent-tools",
                trace=[{"tool_name": name or "read_file", "status": "ok"}],
            )

    request_text = "add in default skills:\n\n* nestjs\n* nextjs\n* reactjs\n* fastapi"
    worker = _FakeWorker()
    result = QueueManager(worker_client=worker, repo_root=tmp_path).run(
        request=request_text,
        index_dir=str(tmp_path),
    )

    tool_names = [str(item.tool_name or "") for item in worker.requests]
    questions = "\n".join(str(item.question) for item in worker.requests)
    read_paths = [
        str((item.tool_args or {}).get("path"))
        for item in worker.requests
        if str(item.tool_name or "") == "read_file"
    ]

    assert tool_names.count("repo_search") == 1
    assert tool_names.count("list_files") == 1
    assert tool_names.count("write_file") == 0
    assert all("dependency_service.py" not in path for path in read_paths)
    assert "DEFAULT_SKILL_NAMES" in questions
    assert "src/mana_agent/default_skills/*.md" in questions
    assert result.run_status == "completed"


def test_queue_manager_blocks_edit_when_no_mutation_tool_attempted(tmp_path: Path):
    from mana_agent.llm.agent_work_queue import QueueManager

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "overview.md").write_text(_substantive("Overview"), encoding="utf-8")

    class _FakeWorker:
        def __init__(self) -> None:
            self.policies: list[dict] = []

        def run_tools(self, request, on_event=None):  # noqa: ANN001
            self.policies.append(dict(request.tool_policy or {}))
            return ToolRunResponse(
                answer="only prose",
                sources=[],
                mode="agent-tools",
                trace=[],
                warnings=[],
            )

    worker = _FakeWorker()
    result = QueueManager(worker_client=worker, repo_root=tmp_path).run(
        request="update docs/overview.md",
        index_dir=str(tmp_path),
        requires_edit=True,
        target_files=["docs/overview.md"],
    )

    assert result.run_status == "blocked"
    assert result.terminal_reason == "mutation_command_missing"
    decision = result.planner_decisions[0]
    assert decision["forced_mutation_retry_ran"] is True
    assert decision["forced_retry_mutation_attempted"] is False
    assert decision["mutation_tool_attempted"] is False
    assert decision["verification_passed"] is False
    assert decision["verify_requires_mutation"] is True
    assert "no executable MutationCommand" in result.answer
    assert worker.policies[-1]["allowed_tools"] == _AGENTIC_EDIT_TOOLS


def test_bare_docs_filename_resolves_existing_docs_file(tmp_path: Path):
    from mana_agent.llm.agent_work_queue import QueueManager

    (tmp_path / "docs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "docs" / "08-architecture.md").write_text("# Architecture\n", encoding="utf-8")

    payload = QueueManager(worker_client=object(), repo_root=tmp_path).preview_plan(
        request="analyze src architecture and update 08-architecture.md",
        requires_edit=True,
        target_files=["src/08-architecture.md"],
    )

    assert "docs/08-architecture.md" in payload["target_files"]
    assert "src/08-architecture.md" not in payload["target_files"]


def test_typo_target_resolution_clears_missing_required_files(tmp_path: Path):
    from mana_agent.llm.agent_work_queue import QueueManager

    (tmp_path / "docs").mkdir()
    body = "# Architecture\n\n" + ("Current architecture evidence. " * 8) + "\n"
    target = tmp_path / "docs" / "08-architecture.md"
    target.write_text(body, encoding="utf-8")

    class _TypoResolutionWorker:
        def __init__(self) -> None:
            self.requests: list[object] = []

        def run_tools(self, request, on_event=None):  # noqa: ANN001
            self.requests.append(request)
            tool = request.tool_name or ""
            if tool == "repo_search":
                return ToolRunResponse(
                    answer="docs/08-architecture.md",
                    mode="agent-tools",
                    trace=[{"tool_name": "repo_search", "status": "ok", "path": "docs/08-architecture.md"}],
                )
            if tool == "read_file":
                return ToolRunResponse(
                    answer=target.read_text(encoding="utf-8"),
                    mode="agent-tools",
                    trace=[{"tool_name": "read_file", "status": "ok", "path": request.tool_args.get("path")}],
                )
            if "MutationCommand" in str(request.question):
                return ToolRunResponse(
                    answer=json.dumps(
                        {
                            "tool_name": "write_file",
                            "tool_args": {
                                "path": "docs/08-architecture.md",
                                "content": body + "\n" + ("Updated with new architecture details. " * 8) + "\n",
                            },
                        }
                    ),
                    mode="agent-tools",
                    trace=[],
                )
            return ToolRunResponse(answer="ok", mode="agent-tools", trace=[])

    worker = _TypoResolutionWorker()
    result = QueueManager(worker_client=worker, repo_root=tmp_path).run(
        request="update the architectue.md with new architecture.",
        index_dir=str(tmp_path),
        requires_edit=True,
    )

    decision = result.planner_decisions[0]
    assert decision["raw_target_files"] == ["architectue.md"]
    assert decision["resolved_target_files"] == ["docs/08-architecture.md"]
    assert decision["required_files"] == ["docs/08-architecture.md"]
    assert decision["missing_required_files"] == []
    assert "architectue.md" not in decision["missing_required_files"]
    read_paths = [
        str(req.tool_args.get("path"))
        for req in worker.requests
        if (req.tool_name or "") == "read_file"
    ]
    assert "docs/08-architecture.md" in read_paths
    edit_questions = [str(req.question) for req in worker.requests if "MutationCommand" in str(req.question)]
    assert any(
        "MutationPlan" in str(req.question) and "docs/08-architecture.md" in str(req.question)
        for req in worker.requests
        if "MutationCommand" in str(req.question)
    )
    assert all("Using the file evidence already gathered in this run, update architectue.md" not in question for question in edit_questions)


def test_edit_intent_forces_mutation_tool_attempt(tmp_path: Path):
    from mana_agent.llm.agent_work_queue import QueueManager

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "08-architecture.md").write_text("# Old\n", encoding="utf-8")

    class _FakeWorker:
        def __init__(self) -> None:
            self.mutation_tools_called: list[str] = []

        def run_tools(self, request, on_event=None):  # noqa: ANN001
            tool = request.tool_name or ""
            if tool in {"write_file", "edit_file", "multi_edit_file", "apply_patch", "create_file", "delete_file"}:
                raise AssertionError("worker must not execute mutation tools")
            if "MutationCommand" in str(request.question):
                return ToolRunResponse(
                    answer=json.dumps(
                        {
                            "tool_name": "write_file",
                            "tool_args": {"path": "docs/08-architecture.md", "content": _substantive("Architecture")},
                        }
                    ),
                    mode="agent-tools",
                    trace=[],
                )
            return ToolRunResponse(
                answer="found docs/08-architecture.md",
                mode="agent-tools",
                trace=[{"tool_name": tool or "repo_search", "status": "ok"}],
            )

    worker = _FakeWorker()
    result = QueueManager(worker_client=worker, repo_root=tmp_path).run(
        request="update docs/08-architecture.md",
        index_dir=str(tmp_path),
        requires_edit=None,
    )

    decision = result.planner_decisions[0]
    assert decision["mutation_required"] is True
    assert decision["mutation_tool_attempted"] is True
    assert decision["mutation_tools_called"]
    assert worker.mutation_tools_called == []


def test_docs_edit_without_approved_fallback_blocks_when_worker_writes_nothing(tmp_path: Path):
    from mana_agent.llm.agent_work_queue import QueueManager

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "docs").mkdir()
    body = "# Architecture\n\n" + ("Existing architecture details for the project. " * 8) + "\n"
    (tmp_path / "docs" / "08-architecture.md").write_text(body, encoding="utf-8")
    subprocess.run(
        ["git", "add", "docs/08-architecture.md"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    class _ReadOnlyWorker:
        def __init__(self) -> None:
            self.requests: list[object] = []

        def run_tools(self, request, on_event=None):  # noqa: ANN001
            self.requests.append(request)
            tool = request.tool_name or ""
            if tool == "repo_search":
                return ToolRunResponse(
                    answer="docs/08-architecture.md",
                    mode="agent-tools",
                    trace=[{"tool_name": "repo_search", "status": "ok"}],
                )
            if tool == "read_file":
                return ToolRunResponse(
                    answer=body,
                    mode="agent-tools",
                    trace=[{"tool_name": "read_file", "status": "ok", "path": "docs/08-architecture.md"}],
                )
            return ToolRunResponse(
                answer="prose only",
                mode="agent-tools",
                trace=[],
            )

    worker = _ReadOnlyWorker()
    result = QueueManager(worker_client=worker, repo_root=tmp_path).run(
        request="update 08-architecture.md in docs",
        index_dir=str(tmp_path),
        requires_edit=None,
    )

    decision = result.planner_decisions[0]
    assert result.run_status == "blocked"
    assert decision["mutation_required"] is True
    assert decision["forced_mutation_retry_ran"] is True
    assert decision["mutation_tool_successful"] is False
    assert decision["mutation_plan_approved"] is True
    assert decision["mutation_plan_executed"] is False
    assert "docs/08-architecture.md" not in result.changed_files


def test_explicit_fallback_decision_can_mutate_existing_update_notes_section(tmp_path: Path):
    from mana_agent.llm.agent_work_queue import QueueManager

    (tmp_path / "docs").mkdir()
    body = (
        "# Architecture\n\n"
        + ("Existing architecture details for the project. " * 8)
        + "\n\n## Update Notes\n\nRequested change: update 08-architecture.md in docs.\n"
    )
    target = tmp_path / "docs" / "08-architecture.md"
    target.write_text(body, encoding="utf-8")

    class _ProseOnlyWorker:
        def __init__(self) -> None:
            self.requests: list[object] = []

        def run_tools(self, request, on_event=None):  # noqa: ANN001
            self.requests.append(request)
            tool = request.tool_name or ""
            if tool == "repo_search":
                return ToolRunResponse(
                    answer="docs/08-architecture.md",
                    mode="agent-tools",
                    trace=[{"tool_name": "repo_search", "status": "ok", "path": "docs/08-architecture.md"}],
                )
            if tool == "read_file":
                return ToolRunResponse(
                    answer=target.read_text(encoding="utf-8"),
                    mode="agent-tools",
                    trace=[{"tool_name": "read_file", "status": "ok", "path": request.tool_args.get("path")}],
                )
            return ToolRunResponse(answer="only prose", mode="agent-tools", trace=[])

    result = QueueManager(worker_client=_ProseOnlyWorker(), repo_root=tmp_path).run(
        request="update the architectue.md with new architecture.",
        index_dir=str(tmp_path),
        requires_edit=True,
        tool_policy={"fallback_decision": True},
    )

    decision = result.planner_decisions[0]
    assert result.run_status == "completed"
    assert decision["raw_target_files"] == ["architectue.md"]
    assert decision["resolved_target_files"] == ["docs/08-architecture.md"]
    assert decision["mutation_tools_called"] == ["write_file"]
    assert decision["missing_required_files"] == []
    assert "docs/08-architecture.md" in result.changed_files
    assert "Requested change: update the architectue.md with new architecture." in target.read_text(encoding="utf-8")


def test_source_architecture_update_reads_src_evidence_before_mutation(tmp_path: Path):
    from mana_agent.llm.agent_work_queue import QueueManager
    from mana_agent.llm.mutation_plan import ARCHITECTURE_SOURCE_DIRS

    (tmp_path / "docs").mkdir()
    target = tmp_path / "docs" / "08-architecture.md"
    target.write_text("# Architecture\n\nExisting architecture.\n", encoding="utf-8")
    src_files: list[str] = []
    for dirname in ARCHITECTURE_SOURCE_DIRS:
        root = tmp_path / dirname
        root.mkdir(parents=True, exist_ok=True)
        rel = f"{dirname}module.py"
        (tmp_path / rel).write_text("class Component:\n    pass\n", encoding="utf-8")
        src_files.append(rel)

    class _ArchitectureWorker:
        def __init__(self) -> None:
            self.requests: list[object] = []

        def run_tools(self, request, on_event=None):  # noqa: ANN001
            self.requests.append(request)
            tool = request.tool_name or ""
            if tool == "repo_search":
                return ToolRunResponse(
                    answer="tests/test_architecture.py\nCHANGELOG.md",
                    mode="agent-tools",
                    trace=[{"tool_name": "repo_search", "status": "ok"}],
                )
            if tool == "read_file":
                path = str(request.tool_args.get("path"))
                return ToolRunResponse(
                    answer=(tmp_path / path).read_text(encoding="utf-8"),
                    mode="agent-tools",
                    trace=[{"tool_name": "read_file", "status": "ok", "path": path}],
                )
            if "MutationCommand" in str(request.question):
                return ToolRunResponse(
                    answer=json.dumps(
                        {
                            "tool_name": "write_file",
                            "tool_args": {
                                "path": "docs/08-architecture.md",
                                "content": (
                                    "# Architecture\n\n"
                                    "Updated from src/mana_agent/llm/module.py evidence.\n\n"
                                    + ("Source-backed architecture section. " * 12)
                                    + "\n"
                                ),
                            },
                        }
                    ),
                    mode="agent-tools",
                    trace=[],
                )
            return ToolRunResponse(answer="ok", mode="agent-tools", trace=[])

    worker = _ArchitectureWorker()
    result = QueueManager(worker_client=worker, repo_root=tmp_path).run(
        request="08-architecture.md update this files with new architecture exist in src",
        index_dir=str(tmp_path),
        requires_edit=True,
    )

    read_paths = [
        str(req.tool_args.get("path"))
        for req in worker.requests
        if (req.tool_name or "") == "read_file"
    ]
    assert result.run_status == "completed"
    assert "docs/08-architecture.md" in read_paths
    assert set(src_files).issubset(set(read_paths))
    decision = result.planner_decisions[0]
    assert decision["mutation_plan_approved"] is True
    assert decision["mutation_plan_executed"] is True


def test_source_architecture_update_rejects_tests_changelog_only_evidence(tmp_path: Path):
    from mana_agent.llm.mutation_plan import build_mutation_plan

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "08-architecture.md").write_text("# Architecture\n", encoding="utf-8")
    (tmp_path / "src/mana_agent/llm").mkdir(parents=True)
    (tmp_path / "src/mana_agent/llm/module.py").write_text("class Queue: pass\n", encoding="utf-8")

    plan = build_mutation_plan(
        repo_root=tmp_path,
        user_goal="08-architecture.md update this files with new architecture exist in src",
        target_files=["docs/08-architecture.md"],
        evidence_files_read=["docs/08-architecture.md", "tests/test_architecture.py", "CHANGELOG.md"],
    )

    assert plan.allowed_to_mutate is False
    assert "required evidence files not read" in str(plan.blocked_reason)


def test_approved_mutation_plan_compiles_to_registered_tool(tmp_path: Path):
    from mana_agent.llm.mutation_plan import build_mutation_plan, compile_mutation_command, validate_mutation_command

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "08-architecture.md").write_text("# Old\n", encoding="utf-8")
    plan = build_mutation_plan(
        repo_root=tmp_path,
        user_goal="update docs/08-architecture.md",
        target_files=["docs/08-architecture.md"],
        evidence_files_read=["docs/08-architecture.md"],
    )

    command = compile_mutation_command(
        repo_root=tmp_path,
        plan=plan,
        current_files={"docs/08-architecture.md": "# Old\n"},
        synthesized_content="# New\n\nUpdated architecture.\n",
    )

    assert command.tool_name in {"write_file", "apply_patch"}
    assert command.tool_args.get("path") or command.tool_args.get("patch")
    assert not validate_mutation_command(command)


def test_work_item_tool_name_is_not_ignored(tmp_path: Path):
    from mana_agent.llm.mutation_plan import build_mutation_plan

    (tmp_path / "docs").mkdir()
    target = tmp_path / "docs" / "note.md"
    target.write_text("old\n", encoding="utf-8")
    plan = build_mutation_plan(
        repo_root=tmp_path,
        user_goal="update docs/note.md",
        target_files=["docs/note.md"],
        evidence_files_read=["docs/note.md"],
    )

    class _ShouldNotRunWorker:
        def run_tools(self, request, on_event=None):  # noqa: ANN001
            raise AssertionError("worker should not select mutation tools")

    executor = make_worker_executor(worker_client=_ShouldNotRunWorker(), repo_root=tmp_path)
    result = executor(
        WorkItem(
            kind="edit",
            tool_name="write_file",
            tool_args={
                "path": "docs/note.md",
                "content": "new\n",
                "mutation_plan": plan.model_dump(),
                "mutation_plan_id": plan.plan_id,
            },
            question="update docs/note.md",
        )
    )

    assert result.ok is True
    assert target.read_text(encoding="utf-8") == "new\n"
    assert result.trace[0]["mutation_plan_id"] == plan.plan_id
    assert result.trace[0]["created_by"] == "mutation_command_executor"


def test_write_file_without_content_is_incomplete(tmp_path: Path):
    from mana_agent.llm.mutation_plan import build_mutation_plan

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "note.md").write_text("old\n", encoding="utf-8")
    plan = build_mutation_plan(
        repo_root=tmp_path,
        user_goal="update docs/note.md",
        target_files=["docs/note.md"],
        evidence_files_read=["docs/note.md"],
    )

    class _ShouldNotRunWorker:
        def run_tools(self, request, on_event=None):  # noqa: ANN001
            raise AssertionError("worker should not receive incomplete mutation command")

    executor = make_worker_executor(worker_client=_ShouldNotRunWorker(), repo_root=tmp_path)
    result = executor(
        WorkItem(
            kind="edit",
            tool_name="write_file",
            tool_args={
                "path": "docs/note.md",
                "mutation_plan": plan.model_dump(),
                "mutation_plan_id": plan.plan_id,
            },
            question="update docs/note.md",
        )
    )

    assert result.ok is False
    assert "mutation_command_incomplete" in result.error
    assert result.trace[0]["error"] == "mutation_command_incomplete"


def test_architecture_update_does_not_depend_on_model_tool_selection(tmp_path: Path):
    from mana_agent.llm.agent_work_queue import QueueManager

    (tmp_path / "docs").mkdir()
    target = tmp_path / "docs" / "08-architecture.md"
    target.write_text("# Old\n", encoding="utf-8")

    class _StructuredCommandWorker:
        def __init__(self) -> None:
            self.requests: list[object] = []

        def run_tools(self, request, on_event=None):  # noqa: ANN001
            self.requests.append(request)
            tool = request.tool_name or ""
            if tool == "repo_search":
                return ToolRunResponse(answer="docs/08-architecture.md", mode="agent-tools", trace=[{"tool_name": "repo_search", "status": "ok"}])
            if tool == "read_file":
                return ToolRunResponse(answer=target.read_text(encoding="utf-8"), mode="agent-tools", trace=[{"tool_name": "read_file", "status": "ok", "path": "docs/08-architecture.md"}])
            if "MutationCommand" in str(request.question):
                return ToolRunResponse(
                    answer=json.dumps(
                        {
                            "tool_name": "write_file",
                            "tool_args": {
                                "path": "docs/08-architecture.md",
                                "content": "# New\n\n" + ("Updated by structured command. " * 8) + "\n",
                            },
                        }
                    ),
                    mode="agent-tools",
                    trace=[],
                )
            if tool in {"write_file", "create_file", "apply_patch", "delete_file"}:
                raise AssertionError("worker must not execute mutation tools")
            return ToolRunResponse(answer="verified", mode="agent-tools", trace=[{"tool_name": tool or "verify", "status": "ok"}])

    worker = _StructuredCommandWorker()
    result = QueueManager(worker_client=worker, repo_root=tmp_path).run(
        request="update docs/08-architecture.md",
        index_dir=str(tmp_path),
        requires_edit=True,
        target_files=["docs/08-architecture.md"],
    )

    assert result.run_status == "completed"
    assert result.changed_files == ["docs/08-architecture.md"]
    assert target.read_text(encoding="utf-8").startswith("# New")
    assert all((req.tool_name or "") != "write_file" for req in worker.requests)
    assert any(row.get("created_by") == "mutation_command_executor" for row in result.trace)


def test_mutation_only_worker_prose_is_rejected(tmp_path: Path):
    from mana_agent.llm.agent_work_queue import QueueManager

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "note.md").write_text("old\n", encoding="utf-8")

    class _ProseWorker:
        def __init__(self) -> None:
            self.mutation_calls = 0

        def run_tools(self, request, on_event=None):  # noqa: ANN001
            tool = request.tool_name or ""
            if tool == "repo_search":
                return ToolRunResponse(answer="docs/note.md", mode="agent-tools", trace=[{"tool_name": "repo_search", "status": "ok"}])
            if tool == "read_file":
                return ToolRunResponse(answer="old\n", mode="agent-tools", trace=[{"tool_name": "read_file", "status": "ok", "path": "docs/note.md"}])
            if tool in {"write_file", "create_file", "apply_patch", "delete_file"}:
                self.mutation_calls += 1
            return ToolRunResponse(answer="I updated it", mode="agent-tools", trace=[])

    worker = _ProseWorker()
    result = QueueManager(worker_client=worker, repo_root=tmp_path).run(
        request="update docs/note.md",
        index_dir=str(tmp_path),
        requires_edit=True,
        target_files=["docs/note.md"],
    )

    assert result.run_status == "blocked"
    assert result.terminal_reason == "mutation_command_missing"
    assert worker.mutation_calls == 0
    assert "Mutation plan was approved" in result.answer
    assert "no executable MutationCommand" in result.answer


def test_simple_docs_edit_does_not_read_all_docs(tmp_path: Path):
    from mana_agent.llm.agent_work_queue import QueueManager

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "docs").mkdir()
    body = "# Architecture\n\n" + ("Existing architecture details for the project. " * 8) + "\n"
    (tmp_path / "docs" / "08-architecture.md").write_text(body, encoding="utf-8")
    for idx in range(1, 8):
        (tmp_path / "docs" / f"{idx:02d}-other.md").write_text("# Other\n\n" + ("Other docs. " * 20), encoding="utf-8")
    subprocess.run(
        ["git", "add", "docs"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    class _ReadOnlyWorker:
        def __init__(self) -> None:
            self.requests: list[object] = []

        def run_tools(self, request, on_event=None):  # noqa: ANN001
            self.requests.append(request)
            tool = request.tool_name or ""
            if tool == "repo_search":
                return ToolRunResponse(
                    answer="docs/08-architecture.md",
                    mode="agent-tools",
                    trace=[{"tool_name": "repo_search", "status": "ok"}],
                )
            if tool == "read_file":
                return ToolRunResponse(
                    answer=body,
                    mode="agent-tools",
                    trace=[{"tool_name": "read_file", "status": "ok", "path": request.tool_args.get("path")}],
                )
            return ToolRunResponse(answer="prose only", mode="agent-tools", trace=[])

    worker = _ReadOnlyWorker()
    result = QueueManager(worker_client=worker, repo_root=tmp_path).run(
        request="update 08-architecture.md in docs",
        index_dir=str(tmp_path),
        requires_edit=None,
    )

    read_files = [
        str(req.tool_args.get("path"))
        for req in worker.requests
        if (req.tool_name or "") == "read_file" and str(req.tool_args.get("path", "")).startswith("docs/")
    ]
    assert result.run_status == "blocked"
    assert "docs/08-architecture.md" in read_files
    assert len(read_files) <= 3


def test_deterministic_preview_uses_project_level_edit_checklist(tmp_path: Path):
    from mana_agent.llm.agent_work_queue import QueueManager

    mgr = QueueManager(worker_client=object(), repo_root=tmp_path)
    payload = mgr.preview_plan(
        request="create src/mana_agent/commands/new_command.py",
        requires_edit=True,
        target_files=["src/mana_agent/commands/new_command.py"],
    )

    steps = payload["prechecklist"]["steps"]
    edit = next(step for step in steps if step["id"] == "edit")
    assert "imports, exports" in edit["title"]
    assert "registries" in edit["title"]
    assert "call sites" in edit["title"]
    assert edit["requires_tools"] == ["edit_file", "multi_edit_file", "apply_patch", "apply_patch_batch", "write_file", "create_file", "delete_file"]
    assert edit["checks"] == [
        "target file changed/created/deleted",
        "related imports/usages updated",
        "integration path updated",
        "stale references removed",
        "verification selected and executed when possible",
    ]


def test_queue_manager_blocks_edit_when_mutation_has_no_changed_files(tmp_path: Path):
    from mana_agent.llm.agent_work_queue import QueueManager

    class _FakeWorker:
        def run_tools(self, request, on_event=None):  # noqa: ANN001
            return ToolRunResponse(
                answer="mutation attempted",
                sources=[],
                mode="agent-tools",
                trace=[{"tool_name": "apply_patch", "status": "ok", "changed_files": []}],
                warnings=[],
            )

    result = QueueManager(worker_client=_FakeWorker(), repo_root=tmp_path).run(
        request="update docs/overview.md",
        index_dir=str(tmp_path),
        tool_policy={"mutation_required": True},
        target_files=["docs/overview.md"],
    )

    assert result.run_status == "blocked"
    assert result.terminal_reason == "mutation_required_but_no_changed_files"
    assert "forced_mutation_retry_no_changed_files" in result.warnings


def test_queue_manager_uses_latest_useful_answer_only_for_edit_success(tmp_path: Path):
    from mana_agent.llm.agent_work_queue import QueueManager

    class _FakeWorker:
        def run_tools(self, request, on_event=None):  # noqa: ANN001
            if (request.tool_name or "") == "repo_search":
                return ToolRunResponse(
                    answer="intermediate search answer",
                    sources=[],
                    mode="agent-tools",
                    trace=[{"tool_name": "repo_search", "status": "ok"}],
                    warnings=[],
                )
            _write_real(tmp_path, "docs/overview.md")
            return ToolRunResponse(
                answer="final mutation answer",
                sources=[],
                mode="agent-tools",
                trace=[{"tool_name": "write_file", "status": "ok", "changed_files": ["docs/overview.md"]}],
                warnings=[],
            )

    result = QueueManager(worker_client=_FakeWorker(), repo_root=tmp_path).run(
        request="update docs/overview.md",
        index_dir=str(tmp_path),
        requires_edit=True,
        target_files=["docs/overview.md"],
    )

    assert result.run_status == "completed"
    # The final answer is rebuilt from authoritative state: it reports the changed
    # file, keeps the consistent worker answer, and drops the stale intermediate.
    assert result.changed_files == ["docs/overview.md"]
    assert "docs/overview.md" in result.answer
    assert "final mutation answer" in result.answer
    assert "intermediate search answer" not in result.answer


def test_sniffer_uses_planner_target_file_for_edit_job(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    sniffer = CodingAgentSniffer(
        repo_root=tmp_path,
        request="in docs add analyze and describe about this project.",
        emit_edit=True,
        target_files=["docs/analyze.md"],
    )
    board = TaskBoard(queue=AgentWorkQueue())
    search = WorkItem(kind="discover", tool_name="repo_search", tool_args={"query": "docs"})

    new_items = sniffer.on_result(search, WorkResult(ok=True, files_discovered=[]), board=board)
    edit = next(item for item in new_items if item.kind == "edit")

    assert edit.tool_name == "write_file"
    assert edit.tool_args["path"] == "docs/analyze.md"
    assert edit.tool_args["mutation_plan_id"]
    assert "Target file: docs/analyze.md" in edit.question
    assert "related importers, exports, registries" in edit.question
    assert "stale docs/config references" in edit.question


def test_edit_with_evidence_uses_agentic_policy_without_duplicate_reads(tmp_path: Path):
    from mana_agent.llm.evidence_memory import EvidenceMemory
    from mana_agent.llm.mutation_plan import build_mutation_plan

    (tmp_path / "src").mkdir()
    target = tmp_path / "src" / "app.py"
    target.write_text("old\n", encoding="utf-8")
    memory = EvidenceMemory(repo_root=tmp_path, run_id="edit-memory")
    memory.store(
        original_path="src/app.py",
        resolved=target.resolve(),
        mode="full",
        start_line=1,
        end_line=1,
        line_count=1,
        content="old\n",
        summary="full file read, 1 lines",
    )

    class _FakeWorker:
        def run_tools(self, request, on_event=None):  # noqa: ANN001
            raise AssertionError("complete mutation command should execute directly")

    executor = make_worker_executor(
        worker_client=_FakeWorker(),
        repo_root=tmp_path,
        index_dir=str(tmp_path / ".mana/index"),
        run_id="edit-memory",
        tool_policy={"allowed_tools": ["read_file", "write_file"], "require_read_files": 2},
    )
    plan = build_mutation_plan(
        repo_root=tmp_path,
        user_goal="Update src/app.py using existing evidence.",
        target_files=["src/app.py"],
        evidence_files_read=["src/app.py"],
    )
    result = executor(
        WorkItem(
            kind="edit",
            tool_name="write_file",
            tool_args={
                "path": "src/app.py",
                "content": "new\n",
                "mutation_plan": plan.model_dump(),
                "mutation_plan_id": plan.plan_id,
            },
            question="Update src/app.py using existing evidence.",
        )
    )

    assert result.ok is True
    assert target.read_text(encoding="utf-8") == "new\n"
    assert result.trace[0]["created_by"] == "mutation_command_executor"


def test_sniffer_skips_edit_for_non_mutating_request(tmp_path: Path):
    (tmp_path / "found.py").write_text("x = 1\n")
    sniffer = CodingAgentSniffer(repo_root=tmp_path, request="how does x work?", emit_edit=False)
    board = TaskBoard(queue=AgentWorkQueue())
    search = WorkItem(kind="search", tool_name="repo_search", tool_args={"query": "x"})
    new_items = sniffer.on_result(search, WorkResult(ok=True, files_discovered=["found.py"]), board=board)
    assert all(it.kind not in {"edit", "verify"} for it in new_items)


# --------------------------------------------------------------------------- #
# EventBus / TaskBoard
# --------------------------------------------------------------------------- #
def test_eventbus_broadcasts_transitions():
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(lambda e: seen.append(e.type))
    q = AgentWorkQueue(bus=bus)
    item = WorkItem(kind="read", tool_name="read_file", tool_args={"path": "a.py"})
    q.submit(item)
    q.claim()
    q.complete(item.id, status="done", result=WorkResult(ok=True))
    assert "job_submitted" in seen
    assert "job_running" in seen
    assert "job_done" in seen


# --------------------------------------------------------------------------- #
# Final-answer aggregation from authoritative execution state
# --------------------------------------------------------------------------- #
def test_apply_patch_run_never_claims_no_edit_tool(tmp_path: Path):
    """A worker that wrongly says 'no edit tool was available' must not win when
    the trace proves apply_patch executed and changed a file."""
    from mana_agent.llm.agent_work_queue import QueueManager

    class _FakeWorker:
        def run_tools(self, request, on_event=None):  # noqa: ANN001
            if (request.tool_name or "") == "repo_search":
                return ToolRunResponse(
                    answer="searching",
                    mode="agent-tools",
                    trace=[{"tool_name": "repo_search", "status": "ok"}],
                )
            # The worker prose is stale/wrong; the trace is authoritative.
            return ToolRunResponse(
                answer="Sorry, no edit tool was available so no changes were made.",
                mode="agent-tools",
                trace=[
                    {
                        "tool_name": "apply_patch",
                        "status": "ok",
                        "changed_files": ["src/app.py"],
                    }
                ],
            )

    result = QueueManager(worker_client=_FakeWorker(), repo_root=tmp_path).run(
        request="edit src/app.py",
        index_dir=str(tmp_path),
        requires_edit=True,
        target_files=["src/app.py"],
    )

    assert result.changed_files == ["src/app.py"]
    assert "no edit tool" not in result.answer.lower()
    assert "no changes were made" not in result.answer.lower()
    assert "src/app.py" in result.answer


def test_non_empty_changed_files_never_claims_no_changes(tmp_path: Path):
    from mana_agent.llm.agent_work_queue import QueueManager

    class _FakeWorker:
        def run_tools(self, request, on_event=None):  # noqa: ANN001
            if (request.tool_name or "") == "repo_search":
                return ToolRunResponse(answer="ok", mode="agent-tools",
                                       trace=[{"tool_name": "repo_search", "status": "ok"}])
            return ToolRunResponse(
                answer="I did not make any changes.",
                mode="agent-tools",
                trace=[{"tool_name": "write_file", "status": "ok", "changed_files": ["docs/x.md"]}],
            )

    result = QueueManager(worker_client=_FakeWorker(), repo_root=tmp_path).run(
        request="update docs/x.md",
        index_dir=str(tmp_path),
        requires_edit=True,
        target_files=["docs/x.md"],
    )

    assert result.changed_files == ["docs/x.md"]
    assert "no change" not in result.answer.lower()
    assert "did not make any changes" not in result.answer.lower()


def test_failed_verify_project_is_surfaced_in_final_answer(tmp_path: Path):
    from mana_agent.llm.agent_work_queue import QueueManager

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("old\n", encoding="utf-8")

    class _FakeWorker:
        def run_tools(self, request, on_event=None):  # noqa: ANN001
            name = request.tool_name or ""
            if name == "repo_search":
                return ToolRunResponse(answer="src/app.py", mode="agent-tools",
                                       trace=[{"tool_name": "repo_search", "status": "ok"}])
            if name == "read_file":
                return ToolRunResponse(answer="old\n", mode="agent-tools",
                                       trace=[{"tool_name": "read_file", "status": "ok", "path": "src/app.py"}])
            if "Verify" in (request.question or "") or name in {"verify", "verify_project"}:
                return ToolRunResponse(
                    answer="verification done",
                    mode="agent-tools",
                    trace=[
                        {
                            "tool_name": "verify_project",
                            "status": "failed",
                            "checks": [
                                {"name": "pytest", "status": "failed",
                                 "reason": "tests/test_app.py::test_x assertion error"},
                                {"name": "ruff", "status": "passed"},
                            ],
                        }
                    ],
                )
            if "MutationCommand" in str(request.question):
                return ToolRunResponse(
                    answer=json.dumps(
                        {
                            "tool_name": "write_file",
                            "tool_args": {"path": "src/app.py", "content": "print('new')\n" * 20},
                        }
                    ),
                    mode="agent-tools",
                    trace=[],
                )
            return ToolRunResponse(
                answer="ok",
                mode="agent-tools",
                trace=[],
            )

    result = QueueManager(worker_client=_FakeWorker(), repo_root=tmp_path).run(
        request="edit src/app.py",
        index_dir=str(tmp_path),
        requires_edit=True,
        target_files=["src/app.py"],
    )

    assert "Verification: FAILED" in result.answer
    assert "pytest" in result.answer
    assert "tests/test_app.py::test_x" in result.answer
    decisions = result.planner_decisions[0]
    assert decisions["verification_failed"] is True


def test_passed_verify_reports_changed_files_and_checks_passed(tmp_path: Path):
    from mana_agent.llm.agent_work_queue import QueueManager

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("old\n", encoding="utf-8")

    class _FakeWorker:
        def run_tools(self, request, on_event=None):  # noqa: ANN001
            name = request.tool_name or ""
            if name == "repo_search":
                return ToolRunResponse(answer="src/app.py", mode="agent-tools",
                                       trace=[{"tool_name": "repo_search", "status": "ok"}])
            if name == "read_file":
                return ToolRunResponse(answer="old\n", mode="agent-tools",
                                       trace=[{"tool_name": "read_file", "status": "ok", "path": "src/app.py"}])
            if "Verify" in (request.question or "") or name in {"verify", "verify_project"}:
                return ToolRunResponse(
                    answer="verification done",
                    mode="agent-tools",
                    trace=[{"tool_name": "verify_project", "status": "ok",
                            "checks": [{"name": "pytest", "status": "passed"}]}],
                )
            if "MutationCommand" in str(request.question):
                return ToolRunResponse(
                    answer=json.dumps(
                        {
                            "tool_name": "write_file",
                            "tool_args": {"path": "src/app.py", "content": "print('new')\n" * 20},
                        }
                    ),
                    mode="agent-tools",
                    trace=[],
                )
            return ToolRunResponse(
                answer="ok",
                mode="agent-tools",
                trace=[],
            )

    result = QueueManager(worker_client=_FakeWorker(), repo_root=tmp_path).run(
        request="edit src/app.py",
        index_dir=str(tmp_path),
        requires_edit=True,
        target_files=["src/app.py"],
    )

    assert "src/app.py" in result.answer
    assert "Verification: passed" in result.answer
    assert result.planner_decisions[0]["verification_passed"] is True


def test_edit_request_cannot_finalize_after_only_read_search(tmp_path: Path):
    """An edit request where the worker only ever reads/searches must end blocked,
    and the final answer must not claim success."""
    from mana_agent.llm.agent_work_queue import QueueManager

    class _FakeWorker:
        def run_tools(self, request, on_event=None):  # noqa: ANN001
            # Never runs a mutation tool; only returns prose + read/search traces.
            return ToolRunResponse(
                answer="Here is what the file looks like.",
                mode="agent-tools",
                trace=[{"tool_name": request.tool_name or "read_file", "status": "ok"}],
            )

    result = QueueManager(worker_client=_FakeWorker(), repo_root=tmp_path).run(
        request="edit src/app.py to add a function",
        index_dir=str(tmp_path),
        requires_edit=True,
        target_files=["src/app.py"],
    )

    assert result.run_status == "blocked"
    assert result.terminal_reason in {
        "mutation_required_but_no_mutation_tool_attempted",
        "mutation_required_but_no_changed_files",
        "mutation_command_missing",
    }
    assert result.changed_files == []
    assert "could not be completed" in result.answer.lower()
    assert "No edit tool was executed" not in result.answer
    decision = result.planner_decisions[0]
    assert decision["forced_mutation_retry_ran"] is True
    assert decision["verify_requires_mutation"] is True
    assert decision["verification_passed"] is False
    # Must not claim a successful edit when nothing actually changed.
    assert "applied changes" not in result.answer.lower()
