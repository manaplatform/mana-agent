from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from mana_agent.coding.scheduler import (
    DependencyGraph,
    OperationKind,
    OperationReadCache,
    ParallelToolExecutor,
    classify_tool_operation,
    extract_target_paths,
)


class MockDispatcher:
    def __init__(self, execution_delays: dict[str, float] | None = None) -> None:
        self.delays = execution_delays or {}
        self.invocations: list[tuple[str, dict[str, Any]]] = []
        self.fail_tools: set[str] = set()

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.invocations.append((tool_name, arguments))
        delay = self.delays.get(tool_name, 0.01)
        time.sleep(delay)
        if tool_name in self.fail_tools:
            return {"ok": False, "error": f"Simulated failure in {tool_name}"}
        return {"ok": True, "result": f"Executed {tool_name}", "files_changed": [arguments.get("path")] if "path" in arguments else []}


def test_dependency_graph_independent_reads_have_no_dependencies() -> None:
    tool_calls = [
        {"id": "call_1", "name": "read_file", "arguments": {"path": "src/a.py"}},
        {"id": "call_2", "name": "read_file", "arguments": {"path": "src/b.py"}},
        {"id": "call_3", "name": "repo_search", "arguments": {"query": "auth"}},
    ]
    graph = DependencyGraph.build(tool_calls)

    assert len(graph.nodes) == 3
    assert len(graph.nodes["call_1"].dependencies) == 0
    assert len(graph.nodes["call_2"].dependencies) == 0
    assert len(graph.nodes["call_3"].dependencies) == 0


def test_dependency_graph_conflicting_writes_serialized() -> None:
    tool_calls = [
        {"id": "call_w1", "name": "write_file", "arguments": {"path": "src/main.py", "content": "v1"}},
        {"id": "call_w2", "name": "edit_file", "arguments": {"path": "src/main.py", "old_string": "v1", "new_string": "v2"}},
    ]
    graph = DependencyGraph.build(tool_calls)

    assert len(graph.nodes["call_w1"].dependencies) == 0
    assert "call_w1" in graph.nodes["call_w2"].dependencies
    assert "call_w2" in graph.nodes["call_w1"].dependents


def test_dependency_graph_disjoint_writes_parallel() -> None:
    tool_calls = [
        {"id": "call_w1", "name": "write_file", "arguments": {"path": "src/module1.py", "content": "1"}},
        {"id": "call_w2", "name": "write_file", "arguments": {"path": "src/module2.py", "content": "2"}},
    ]
    graph = DependencyGraph.build(tool_calls)

    # Disjoint paths => independent writes can run in parallel
    assert len(graph.nodes["call_w1"].dependencies) == 0
    assert len(graph.nodes["call_w2"].dependencies) == 0


def test_dependency_graph_execute_after_write() -> None:
    tool_calls = [
        {"id": "call_w", "name": "write_file", "arguments": {"path": "src/module.py", "content": "new"}},
        {"id": "call_t", "name": "verify_project", "arguments": {}},
    ]
    graph = DependencyGraph.build(tool_calls)

    assert "call_w" in graph.nodes["call_t"].dependencies


def test_parallel_executor_concurrent_execution() -> None:
    # 4 independent reads with 0.05s delay each
    dispatcher = MockDispatcher(execution_delays={"read_file": 0.05})
    executor = ParallelToolExecutor(max_concurrency=4)

    tool_calls = [
        {"id": f"call_{i}", "name": "read_file", "arguments": {"path": f"file_{i}.txt"}}
        for i in range(4)
    ]
    graph = DependencyGraph.build(tool_calls)

    observed_concurrency: list[int] = []

    def _event_handler(event_name: str, payload: dict[str, Any]) -> None:
        if event_name == "tool.started" and "active_concurrency" in payload:
            observed_concurrency.append(payload["active_concurrency"])

    t0 = time.perf_counter()
    results = asyncio.run(executor.execute_turn(graph, dispatcher, on_event=_event_handler))
    elapsed = time.perf_counter() - t0

    assert len(results) == 4
    for res in results.values():
        assert res["status"] == "completed"

    # Concurrency verification: verify multiple tasks were running concurrently
    assert any(c > 1 for c in observed_concurrency)

    # In parallel, total elapsed should not exceed serial time (with CI timing tolerance)
    assert elapsed < 0.35


def test_parallel_executor_read_caching_and_write_invalidation() -> None:
    dispatcher = MockDispatcher()
    executor = ParallelToolExecutor(max_concurrency=4, cache_enabled=True)

    # Read 1
    graph1 = DependencyGraph.build([{"id": "c1", "name": "read_file", "arguments": {"path": "cached.py"}}])
    asyncio.run(executor.execute_turn(graph1, dispatcher))
    assert len(dispatcher.invocations) == 1

    # Read 2: same path => should hit cache, not invoke dispatcher
    graph2 = DependencyGraph.build([{"id": "c2", "name": "read_file", "arguments": {"path": "cached.py"}}])
    asyncio.run(executor.execute_turn(graph2, dispatcher))
    assert len(dispatcher.invocations) == 1  # No additional dispatcher invocation

    # Write: should invalidate cached.py
    graph3 = DependencyGraph.build([{"id": "c3", "name": "write_file", "arguments": {"path": "cached.py", "content": "updated"}}])
    asyncio.run(executor.execute_turn(graph3, dispatcher))
    assert len(dispatcher.invocations) == 2

    # Read 3: after write invalidation => invokes dispatcher again
    graph4 = DependencyGraph.build([{"id": "c4", "name": "read_file", "arguments": {"path": "cached.py"}}])
    asyncio.run(executor.execute_turn(graph4, dispatcher))
    assert len(dispatcher.invocations) == 3


def test_parallel_executor_partial_failure_isolation() -> None:
    dispatcher = MockDispatcher()
    dispatcher.fail_tools.add("fail_tool")

    tool_calls = [
        {"id": "c_fail", "name": "fail_tool", "arguments": {"path": "bad.txt"}},
        {"id": "c_ok", "name": "read_file", "arguments": {"path": "good.txt"}},
        {"id": "c_child", "name": "edit_file", "arguments": {"path": "bad.txt", "old_string": "a", "new_string": "b"}},
    ]
    graph = DependencyGraph.build(tool_calls)

    results = asyncio.run(ParallelToolExecutor().execute_turn(graph, dispatcher))

    # Independent sibling c_ok must succeed
    assert results["c_ok"]["status"] == "completed"

    # c_fail must fail
    assert results["c_fail"]["status"] == "failed"

    # Downstream dependent c_child must be skipped due to upstream failure
    assert results["c_child"]["status"] == "skipped"


def test_dependency_graph_approval_request_gates_subsequent_mutations() -> None:
    tool_calls = [
        {"id": "call_appr", "name": "approval_request", "arguments": {"command": "git commit -m fix"}},
        {"id": "call_exec", "name": "run_command", "arguments": {"command": "git commit -m fix"}},
        {"id": "call_read", "name": "read_file", "arguments": {"path": "README.md"}},
    ]
    graph = DependencyGraph.build(tool_calls)

    # run_command must depend on approval_request
    assert "call_appr" in graph.nodes["call_exec"].dependencies
    assert "call_exec" in graph.nodes["call_appr"].dependents

    # read_file is read-only and does not depend on approval_request
    assert "call_appr" not in graph.nodes["call_read"].dependencies


def test_parallel_executor_approval_denied_skips_downstream_action() -> None:
    class ApprovalDispatcher:
        def __init__(self, approved: bool) -> None:
            self.approved = approved
            self.executed_tools: list[str] = []

        def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            self.executed_tools.append(tool_name)
            if tool_name == "approval_request":
                return {
                    "ok": True,
                    "approved": self.approved,
                    "status": "approved" if self.approved else "denied",
                    "message": "Approved" if self.approved else "Denied by user",
                }
            return {"ok": True, "output": "success"}

    # Case 1: Denied approval skips downstream execution
    disp_denied = ApprovalDispatcher(approved=False)
    graph1 = DependencyGraph.build([
        {"id": "appr1", "name": "approval_request", "arguments": {"command": "git commit"}},
        {"id": "exec1", "name": "run_command", "arguments": {"command": "git commit"}},
    ])
    results1 = asyncio.run(ParallelToolExecutor().execute_turn(graph1, disp_denied))
    assert results1["appr1"]["status"] == "failed"
    assert results1["exec1"]["status"] == "skipped"
    assert "run_command" not in disp_denied.executed_tools

    # Case 2: Granted approval allows downstream execution
    disp_granted = ApprovalDispatcher(approved=True)
    graph2 = DependencyGraph.build([
        {"id": "appr2", "name": "approval_request", "arguments": {"command": "git commit"}},
        {"id": "exec2", "name": "run_command", "arguments": {"command": "git commit"}},
    ])
    results2 = asyncio.run(ParallelToolExecutor().execute_turn(graph2, disp_granted))
    assert results2["appr2"]["status"] == "completed"
    assert results2["exec2"]["status"] == "completed"
    assert "run_command" in disp_granted.executed_tools
