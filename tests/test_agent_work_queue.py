from __future__ import annotations

from pathlib import Path

from mana_analyzer.llm.agent_work_queue import (
    AgentWorkQueue,
    EventBus,
    TaskBoard,
    WorkItem,
    WorkQueueRunner,
    WorkResult,
    compute_fingerprint,
)
from mana_analyzer.llm.agent_work_queue_adapters import (
    CodingAgentSniffer,
    classify_result,
    make_worker_executor,
)
from mana_analyzer.llm.tool_worker_process import ToolRunResponse


# --------------------------------------------------------------------------- #
# Fingerprint / dedup
# --------------------------------------------------------------------------- #
def test_same_read_path_collapses_to_one_fingerprint():
    a = compute_fingerprint(kind="read", tool_name="read_file", tool_args={"path": "src/x.py"})
    b = compute_fingerprint(kind="read", tool_name="read_file", tool_args={"path": "./src/x.py "})
    assert a == b


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
