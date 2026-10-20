from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from mana_agent.coding.scheduler import DependencyGraph, ParallelToolExecutor
from mana_agent.tools.batch_file_ops import batch_read_files, batch_write_files
from mana_agent.tools.write_file import safe_write_file


class _BenchmarkingDispatcher:
    def __init__(self, io_delay: float = 0.02) -> None:
        self.io_delay = io_delay

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        time.sleep(self.io_delay)
        return {"ok": True, "tool": tool_name}


def test_benchmark_sequential_vs_parallel_tool_calls() -> None:
    """Benchmark comparing sequential execution against parallel execution of independent tool calls."""
    op_count = 8
    io_delay = 0.02  # 20ms simulated I/O per tool
    dispatcher = _BenchmarkingDispatcher(io_delay=io_delay)

    tool_calls = [
        {"id": f"call_{i}", "name": "read_file", "arguments": {"path": f"src/file_{i}.py"}}
        for i in range(op_count)
    ]

    # 1. Sequential execution
    t0 = time.perf_counter()
    for tc in tool_calls:
        dispatcher.dispatch(tc["name"], tc["arguments"])
    sequential_time = time.perf_counter() - t0

    # 2. Parallel execution
    executor = ParallelToolExecutor(max_concurrency=op_count)
    graph = DependencyGraph.build(tool_calls)

    t1 = time.perf_counter()
    results = asyncio.run(executor.execute_turn(graph, dispatcher))
    parallel_time = time.perf_counter() - t1

    assert len(results) == op_count
    speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0

    print(
        f"\n[BENCHMARK] Tool Execution ({op_count} operations):\n"
        f"  Sequential: {sequential_time * 1000:.1f}ms\n"
        f"  Parallel:   {parallel_time * 1000:.1f}ms\n"
        f"  Speedup:    {speedup:.2f}x"
    )

    # Parallel must be substantially faster than sequential
    assert parallel_time < sequential_time
    assert speedup >= 1.5


def test_benchmark_individual_vs_batch_file_operations(tmp_path: Path) -> None:
    """Benchmark comparing individual file writes vs batch file operations."""
    file_count = 20
    test_files = [f"file_{i}.txt" for i in range(file_count)]
    content = "Hello World from Mana-Agent\n" * 50

    # 1. Individual safe_write_file calls
    t0 = time.perf_counter()
    for f in test_files:
        safe_write_file(repo_root=tmp_path, path=f, content=content, force=True)
    individual_time = time.perf_counter() - t0

    # 2. Batch write_files call
    batch_writes = [{"path": f"batch_{f}", "content": content} for f in test_files]
    t1 = time.perf_counter()
    res = batch_write_files(tmp_path, writes=batch_writes)
    batch_time = time.perf_counter() - t1

    assert res["ok"] is True
    assert len(res["changed_files"]) == file_count

    # 3. Read comparison: individual vs batch_read_files
    t2 = time.perf_counter()
    for f in test_files:
        (tmp_path / f).read_text(encoding="utf-8")
    individual_read_time = time.perf_counter() - t2

    t3 = time.perf_counter()
    read_res = batch_read_files(tmp_path, files=test_files, concurrency=8)
    batch_read_time = time.perf_counter() - t3

    assert read_res["ok"] is True
    assert read_res["total_files"] == file_count

    print(
        f"\n[BENCHMARK] File Operations ({file_count} files):\n"
        f"  Individual Writes: {individual_time * 1000:.1f}ms | Batch Writes: {batch_time * 1000:.1f}ms\n"
        f"  Individual Reads:  {individual_read_time * 1000:.1f}ms | Batch Reads:  {batch_read_time * 1000:.1f}ms"
    )
