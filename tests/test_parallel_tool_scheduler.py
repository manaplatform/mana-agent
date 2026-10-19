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

    t0 = time.perf_counter()
    results = asyncio.run(executor.execute_turn(graph, dispatcher))
    elapsed = time.perf_counter() - t0

    assert len(results) == 4
    for res in results.values():
        assert res["status"] == "completed"

    # In parallel, total elapsed should be substantially less than 4 * 0.05s (0.20s)
    assert elapsed < 0.15


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
