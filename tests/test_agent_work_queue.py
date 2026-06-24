from __future__ import annotations

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
    from mana_agent.llm.tools_manager import QueueManager

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
            return ToolRunResponse(
                answer="ok",
                sources=[],
                mode="agent-tools",
                trace=[
                    {
                        "tool_name": request.tool_name or "tool",
                        "status": "ok",
                        "changed_files": ["docs/overview.md"] if (request.tool_name or "") == "write_file" else [],
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
    assert "Apply concrete changes" in joined  # the edit job ran
    assert "Target file: docs/overview.md" in joined
    assert "Verify the changes" in joined       # the verify job ran (after edit)
    assert result.execution_backend == "work_queue"
    assert result.run_status == "completed"
    assert result.changed_files == ["docs/overview.md"]


def test_queue_manager_blocks_edit_when_no_mutation_tool_attempted(tmp_path: Path):
    from mana_agent.llm.tools_manager import QueueManager

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
    assert result.terminal_reason == "mutation_required_but_no_mutation_tool_attempted"
    assert "forced_mutation_retry_no_mutation_tool_attempted" in result.warnings
    assert worker.policies[-1]["allowed_tools"] == ["apply_patch", "write_file", "create_file"]


def test_queue_manager_blocks_edit_when_mutation_has_no_changed_files(tmp_path: Path):
    from mana_agent.llm.tools_manager import QueueManager

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
    from mana_agent.llm.tools_manager import QueueManager

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
    assert edit.tool_args == {"path": "docs/analyze.md"}
    assert "Target file: docs/analyze.md" in edit.question


def test_edit_with_evidence_uses_mutation_only_policy_without_duplicate_reads(tmp_path: Path):
    from mana_agent.llm.evidence_memory import EvidenceMemory

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

    seen: list[object] = []

    class _FakeWorker:
        def run_tools(self, request, on_event=None):  # noqa: ANN001
            seen.append(request)
            return ToolRunResponse(
                answer="changed",
                mode="agent-tools",
                trace=[{"tool_name": "write_file", "status": "ok", "changed_files": ["src/app.py"]}],
            )

    executor = make_worker_executor(
        worker_client=_FakeWorker(),
        repo_root=tmp_path,
        index_dir=str(tmp_path / ".mana/index"),
        run_id="edit-memory",
        tool_policy={"allowed_tools": ["read_file", "write_file"], "require_read_files": 2},
    )
    result = executor(
        WorkItem(
            kind="edit",
            tool_name="write_file",
            tool_args={"path": "src/app.py"},
            question="Update src/app.py using existing evidence.",
        )
    )

    assert result.ok is True
    assert len(seen) == 1
    request = seen[0]
    assert request.run_id == "edit-memory"
    assert request.tool_policy["allowed_tools"] == ["apply_patch", "write_file", "create_file", "git_diff", "git_status"]
    assert request.tool_policy["require_read_files"] == 0
    assert request.tool_name == "write_file"


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
    from mana_agent.llm.tools_manager import QueueManager

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
    from mana_agent.llm.tools_manager import QueueManager

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
    from mana_agent.llm.tools_manager import QueueManager

    class _FakeWorker:
        def run_tools(self, request, on_event=None):  # noqa: ANN001
            name = request.tool_name or ""
            if name == "repo_search":
                return ToolRunResponse(answer="ok", mode="agent-tools",
                                       trace=[{"tool_name": "repo_search", "status": "ok"}])
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
            return ToolRunResponse(
                answer="edited",
                mode="agent-tools",
                trace=[{"tool_name": "apply_patch", "status": "ok", "changed_files": ["src/app.py"]}],
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
    from mana_agent.llm.tools_manager import QueueManager

    class _FakeWorker:
        def run_tools(self, request, on_event=None):  # noqa: ANN001
            name = request.tool_name or ""
            if name == "repo_search":
                return ToolRunResponse(answer="ok", mode="agent-tools",
                                       trace=[{"tool_name": "repo_search", "status": "ok"}])
            if "Verify" in (request.question or "") or name in {"verify", "verify_project"}:
                return ToolRunResponse(
                    answer="verification done",
                    mode="agent-tools",
                    trace=[{"tool_name": "verify_project", "status": "ok",
                            "checks": [{"name": "pytest", "status": "passed"}]}],
                )
            return ToolRunResponse(
                answer="edited",
                mode="agent-tools",
                trace=[{"tool_name": "write_file", "status": "ok", "changed_files": ["src/app.py"]}],
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
    from mana_agent.llm.tools_manager import QueueManager

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
    }
    assert result.changed_files == []
    assert "could not be completed" in result.answer.lower()
    # Must not claim a successful edit when nothing actually changed.
    assert "applied changes" not in result.answer.lower()
