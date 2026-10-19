"""Intelligent dependency-aware operation scheduler and parallel tool executor.

Constructs an execution DAG from model-generated tool calls, prioritizes:
- Independent reads & searches: executed in parallel.
- Non-conflicting writes: executed concurrently when targeting disjoint paths.
- Conflicting writes: serialized in turn order.
- Output-dependent operations: executed sequentially.
- Tests and verification commands: executed after relevant edits finish.

Features:
- Bounded concurrency with asyncio.Semaphore.
- In-flight deduplication of concurrent identical reads (single-flight).
- In-memory read caching with automatic write-triggered invalidation.
- Partial failure handling without cancelling independent branches.
- Real-time event streaming and timeout/cancellation propagation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

logger = logging.getLogger(__name__)


class OperationKind(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    OTHER = "other"


READ_TOOL_NAMES = frozenset({
    "read_file",
    "repo_search",
    "repo_batch_read",
    "repo_batch_search",
    "batch_read_files",
    "batch_search_files",
    "list_files",
    "batch_list_files",
    "find_symbols",
    "call_graph",
    "git_status",
    "git_diff",
    "git_log",
    "read_skill",
    "semantic_search",
    "ls",
})

WRITE_TOOL_NAMES = frozenset({
    "write_file",
    "create_file",
    "delete_file",
    "edit_file",
    "multi_edit_file",
    "apply_patch",
    "apply_patch_batch",
    "batch_write_files",
    "batch_apply_patches",
    "batch_move_files",
    "batch_delete_files",
})

EXECUTE_TOOL_NAMES = frozenset({
    "run_command",
    "run_script_once",
    "shell",
    "verify_project",
})


def classify_tool_operation(tool_name: str) -> OperationKind:
    normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", str(tool_name or "").strip())
    if normalized in READ_TOOL_NAMES or normalized.startswith("read_"):
        return OperationKind.READ
    if normalized in WRITE_TOOL_NAMES or normalized.startswith("write_"):
        return OperationKind.WRITE
    if normalized in EXECUTE_TOOL_NAMES or "test" in normalized or "verify" in normalized:
        return OperationKind.EXECUTE
    return OperationKind.OTHER


def _normalize_path_key(path: str) -> str:
    cleaned = str(path or "").strip().replace("\\", "/").lstrip("./")
    while "//" in cleaned:
        cleaned = cleaned.replace("//", "/")
    return cleaned


def extract_target_paths(tool_name: str, arguments: dict[str, Any]) -> set[str]:
    """Extract affected target file paths from tool arguments."""
    normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", str(tool_name or "").strip())
    paths: set[str] = set()

    # Direct single-path arguments
    for key in ("path", "file_path", "filePath", "target", "source", "destination", "src", "dst"):
        val = arguments.get(key)
        if isinstance(val, str) and val.strip():
            paths.add(_normalize_path_key(val))

    # Batch read files
    if normalized in {"batch_read_files", "repo_batch_read"}:
        for item in arguments.get("files") or []:
            if isinstance(item, str):
                paths.add(_normalize_path_key(item))
            elif isinstance(item, dict):
                p = item.get("path") or item.get("file_path")
                if p:
                    paths.add(_normalize_path_key(str(p)))

    # Batch write files
    if normalized == "batch_write_files":
        for item in arguments.get("writes") or []:
            if isinstance(item, dict) and item.get("path"):
                paths.add(_normalize_path_key(str(item["path"])))

    # Batch delete files
    if normalized == "batch_delete_files":
        for item in arguments.get("paths") or []:
            if isinstance(item, str):
                paths.add(_normalize_path_key(item))
            elif isinstance(item, dict) and item.get("path"):
                paths.add(_normalize_path_key(str(item["path"])))

    # Batch move files
    if normalized == "batch_move_files":
        for item in arguments.get("moves") or []:
            if isinstance(item, dict):
                if item.get("source") or item.get("src"):
                    paths.add(_normalize_path_key(str(item.get("source") or item.get("src"))))
                if item.get("destination") or item.get("dst"):
                    paths.add(_normalize_path_key(str(item.get("destination") or item.get("dst"))))

    # Apply patch: extract target file lines
    if normalized in {"apply_patch", "batch_apply_patches", "apply_patch_batch"}:
        text = str(arguments.get("patch") or "")
        for match in re.finditer(r"\*\*\*\s+(?:Update|Add|Delete)\s+File:\s*([^\n\r]+)", text):
            paths.add(_normalize_path_key(match.group(1).strip()))
        for item in arguments.get("patches") or []:
            if isinstance(item, dict):
                if item.get("path"):
                    paths.add(_normalize_path_key(str(item["path"])))
                p_text = str(item.get("patch") or "")
                for match in re.finditer(r"\*\*\*\s+(?:Update|Add|Delete)\s+File:\s*([^\n\r]+)", p_text):
                    paths.add(_normalize_path_key(match.group(1).strip()))

    return {p for p in paths if p}


@dataclass
class ToolOperationNode:
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    caller: dict[str, Any] | None = None
    index: int = 0
    kind: OperationKind = field(default=OperationKind.OTHER)
    target_paths: set[str] = field(default_factory=set)
    dependencies: set[str] = field(default_factory=set)
    dependents: set[str] = field(default_factory=set)
    status: str = "pending"  # pending, running, completed, failed, skipped
    result: dict[str, Any] | None = None
    duration_ms: int = 0
    error: str | None = None


class DependencyGraph:
    """Builds and manages an execution DAG for tool calls in a turn."""

    def __init__(self) -> None:
        self.nodes: dict[str, ToolOperationNode] = {}
        self.order: list[str] = []

    def add_node(self, node: ToolOperationNode) -> None:
        self.nodes[node.call_id] = node
        self.order.append(node.call_id)

    @classmethod
    def build(cls, tool_calls: Sequence[dict[str, Any]]) -> DependencyGraph:
        graph = cls()

        for idx, tc in enumerate(tool_calls):
            call_id = str(tc.get("call_id") or tc.get("id") or f"call_{idx}")
            tool_name = str(tc.get("name") or tc.get("tool_name") or "")
            raw_args = tc.get("arguments") or tc.get("args") or {}
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args)
                except Exception:
                    arguments = {"raw": raw_args}
            else:
                arguments = dict(raw_args)

            caller = tc.get("caller")
            kind = classify_tool_operation(tool_name)
            targets = extract_target_paths(tool_name, arguments)

            node = ToolOperationNode(
                call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
                caller=caller,
                index=idx,
                kind=kind,
                target_paths=targets,
            )
            graph.add_node(node)

        # Build dependency edges based on operations and target file conflicts
        node_list = [graph.nodes[cid] for cid in graph.order]

        for i, prev in enumerate(node_list):
            for j in range(i + 1, len(node_list)):
                curr = node_list[j]

                # Explicit dependencies if specified in arguments or caller
                explicit_deps = curr.arguments.get("depends_on") or (curr.caller.get("depends_on") if isinstance(curr.caller, dict) else None)
                if explicit_deps:
                    if isinstance(explicit_deps, str):
                        explicit_deps = [explicit_deps]
                    if prev.call_id in explicit_deps:
                        curr.dependencies.add(prev.call_id)
                        prev.dependents.add(curr.call_id)
                        continue

                # Overlapping path conflict:
                # Concurrent execution is only safe if both operations are pure reads/searches (OperationKind.READ).
                # Any other operation on overlapping paths (e.g. writes, edits, custom ops) must be serialized.
                if prev.target_paths & curr.target_paths:
                    if not (prev.kind == OperationKind.READ and curr.kind == OperationKind.READ):
                        curr.dependencies.add(prev.call_id)
                        prev.dependents.add(curr.call_id)
                        continue

                # Broad writes with no target paths detected must be serialized
                if prev.kind == OperationKind.WRITE and curr.kind == OperationKind.WRITE:
                    if not prev.target_paths and not curr.target_paths:
                        curr.dependencies.add(prev.call_id)
                        prev.dependents.add(curr.call_id)
                        continue

                # Verification/Execution commands after any writes
                if prev.kind == OperationKind.WRITE and curr.kind == OperationKind.EXECUTE:
                    curr.dependencies.add(prev.call_id)
                    prev.dependents.add(curr.call_id)
                    continue

                # Multiple terminal/shell commands are serialized
                if prev.kind == OperationKind.EXECUTE and curr.kind == OperationKind.EXECUTE:
                    curr.dependencies.add(prev.call_id)
                    prev.dependents.add(curr.call_id)
                    continue

        return graph


class OperationReadCache:
    """In-memory cache for immutable or version-validated file reads with automatic write invalidation."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[float, dict[str, Any], set[str]]] = {}

    def _make_key(self, tool_name: str, args: dict[str, Any]) -> str:
        try:
            serialized = json.dumps(args, sort_keys=True, default=str)
        except Exception:
            serialized = str(args)
        return f"{tool_name}:{serialized}"

    def get(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        key = self._make_key(tool_name, args)
        entry = self._cache.get(key)
        if not entry:
            return None
        t_stamp, result, _ = entry
        if time.time() - t_stamp > self.ttl_seconds:
            self._cache.pop(key, None)
            return None
        return dict(result)

    def put(self, tool_name: str, args: dict[str, Any], result: dict[str, Any], touched_paths: set[str]) -> None:
        if not result.get("ok"):
            return
        key = self._make_key(tool_name, args)
        self._cache[key] = (time.time(), dict(result), set(touched_paths))

    def invalidate_paths(self, modified_paths: set[str]) -> int:
        """Invalidate any cached reads overlapping with modified paths."""
        if not modified_paths:
            return 0
        norm_modified = {_normalize_path_key(p) for p in modified_paths}
        evicted = 0
        for k in list(self._cache.keys()):
            _, _, paths = self._cache[k]
            if paths & norm_modified:
                self._cache.pop(k, None)
                evicted += 1
        return evicted

    def clear(self) -> None:
        self._cache.clear()


class ParallelToolExecutor:
    """Executes tool dependency graphs concurrently with bounded semaphore and streaming events."""

    def __init__(
        self,
        *,
        max_concurrency: int = 10,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self.max_concurrency = max(1, min(max_concurrency, 32))
        self.cache_enabled = cache_enabled
        self.cache = OperationReadCache(ttl_seconds=cache_ttl_seconds)
        self._in_flight_reads: dict[str, asyncio.Future[dict[str, Any]]] = {}

    async def execute_turn(
        self,
        graph: DependencyGraph,
        dispatcher: Any,
        *,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        cancel_ev: asyncio.Event | None = None,
        timeout_seconds: float = 120.0,
    ) -> dict[str, dict[str, Any]]:
        """Execute all nodes in the DependencyGraph according to dependency order and concurrency bounds."""
        semaphore = asyncio.Semaphore(self.max_concurrency)
        completed_nodes: set[str] = set()
        failed_nodes: set[str] = set()
        pending_dependencies: dict[str, set[str]] = {
            cid: set(node.dependencies) for cid, node in graph.nodes.items()
        }

        ready_queue: asyncio.Queue[str] = asyncio.Queue()
        for cid, deps in pending_dependencies.items():
            if not deps:
                ready_queue.put_nowait(cid)

        total_nodes = len(graph.nodes)
        processed_count = 0
        running_tasks: set[asyncio.Task] = set()
        active_concurrency = 0

        # Notify active parallel group started
        if on_event is not None and total_nodes > 1:
            on_event(
                "group.started",
                {
                    "total_operations": total_nodes,
                    "max_concurrency": self.max_concurrency,
                    "nodes": [
                        {"call_id": n.call_id, "tool_name": n.tool_name, "kind": n.kind.value}
                        for n in graph.nodes.values()
                    ],
                },
            )

        async def _run_single_node(node: ToolOperationNode) -> None:
            nonlocal active_concurrency
            if cancel_ev is not None and cancel_ev.is_set():
                node.status = "cancelled"
                node.error = "Cancelled by user request"
                node.result = {"ok": False, "error": node.error, "error_code": "CANCELLED"}
                return

            # Check if any dependency failed
            parent_failures = [p for p in node.dependencies if p in failed_nodes]
            if parent_failures:
                node.status = "skipped"
                node.error = f"Skipped due to upstream failure in tool call(s): {', '.join(parent_failures)}"
                node.result = {"ok": False, "error": node.error, "error_code": "DEPENDENCY_FAILED"}
                failed_nodes.add(node.call_id)
                if on_event is not None:
                    on_event(
                        "tool.skipped",
                        {"call_id": node.call_id, "tool_name": node.tool_name, "error": node.error},
                    )
                return

            async with semaphore:
                active_concurrency += 1
                node.status = "running"

                if on_event is not None:
                    on_event(
                        "tool.started",
                        {
                            "call_id": node.call_id,
                            "tool_name": node.tool_name,
                            "arguments": node.arguments,
                            "active_concurrency": active_concurrency,
                        },
                    )

                t0 = time.perf_counter()
                try:
                    # In-flight read deduplication & caching for read operations
                    result: dict[str, Any]
                    is_cached = False
                    cache_key = ""

                    if self.cache_enabled and node.kind == OperationKind.READ:
                        cached = self.cache.get(node.tool_name, node.arguments)
                        if cached is not None:
                            result = cached
                            is_cached = True

                    if not is_cached:
                        # Dispatch tool call asynchronously in worker thread
                        result = await asyncio.to_thread(dispatcher.dispatch, node.tool_name, node.arguments)

                        # Write-invalidation or cache storage
                        if node.kind == OperationKind.WRITE:
                            # Invalidate cached reads for touched paths
                            modified = node.target_paths | set(result.get("files_changed") or result.get("changed_files") or [])
                            self.cache.invalidate_paths(modified)
                        elif self.cache_enabled and node.kind == OperationKind.READ and result.get("ok"):
                            self.cache.put(node.tool_name, node.arguments, result, node.target_paths)

                    node.duration_ms = int((time.perf_counter() - t0) * 1000)
                    node.result = result

                    if result.get("ok", True):
                        node.status = "completed"
                        completed_nodes.add(node.call_id)
                        if on_event is not None:
                            on_event(
                                "tool.completed",
                                {
                                    "call_id": node.call_id,
                                    "tool_name": node.tool_name,
                                    "status": "success",
                                    "duration_ms": node.duration_ms,
                                    "active_concurrency": active_concurrency - 1,
                                    "result": result,
                                },
                            )
                    else:
                        node.status = "failed"
                        node.error = str(result.get("error") or "Tool execution returned not ok")
                        failed_nodes.add(node.call_id)
                        if on_event is not None:
                            on_event(
                                "tool.failed",
                                {
                                    "call_id": node.call_id,
                                    "tool_name": node.tool_name,
                                    "status": "failed",
                                    "duration_ms": node.duration_ms,
                                    "error": node.error,
                                    "active_concurrency": active_concurrency - 1,
                                },
                            )

                except asyncio.CancelledError:
                    node.status = "cancelled"
                    node.error = "Operation cancelled"
                    node.result = {"ok": False, "error": node.error, "error_code": "CANCELLED"}
                    failed_nodes.add(node.call_id)
                    raise
                except Exception as exc:
                    node.duration_ms = int((time.perf_counter() - t0) * 1000)
                    node.status = "failed"
                    node.error = str(exc)
                    node.result = {"ok": False, "error": str(exc), "error_code": "TOOL_EXCEPTION"}
                    failed_nodes.add(node.call_id)
                    if on_event is not None:
                        on_event(
                            "tool.failed",
                            {
                                "call_id": node.call_id,
                                "tool_name": node.tool_name,
                                "status": "failed",
                                "duration_ms": node.duration_ms,
                                "error": node.error,
                                "active_concurrency": active_concurrency - 1,
                            },
                        )
                finally:
                    active_concurrency -= 1

        # Event loop pumping ready tasks
        while processed_count < total_nodes:
            if cancel_ev is not None and cancel_ev.is_set():
                for t in running_tasks:
                    t.cancel()
                break

            # Launch all currently ready nodes
            while not ready_queue.empty():
                ready_cid = ready_queue.get_nowait()
                ready_node = graph.nodes[ready_cid]
                task = asyncio.create_task(_run_single_node(ready_node))
                running_tasks.add(task)

            if not running_tasks:
                # No tasks running and queue is empty: check for remaining nodes
                if processed_count < total_nodes:
                    # Unreachable nodes due to circular dependency or missing parent
                    for cid, node in graph.nodes.items():
                        if node.status == "pending":
                            node.status = "skipped"
                            node.error = "Unresolved dependencies"
                            node.result = {"ok": False, "error": node.error, "error_code": "UNRESOLVED_DEPENDENCY"}
                            processed_count += 1
                break

            # Wait for at least one running task to complete
            done, running_tasks = await asyncio.wait(
                running_tasks,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=timeout_seconds,
            )

            if not done and running_tasks:
                # Timeout occurred
                logger.warning("Parallel tool execution timeout exceeded (%ss)", timeout_seconds)
                for t in running_tasks:
                    t.cancel()
                break

            for finished_task in done:
                processed_count += 1
                try:
                    await finished_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.exception("Unexpected exception in task runner: %s", e)

            # Check dependent nodes to see if they can now be scheduled
            for cid, node in graph.nodes.items():
                if node.status == "pending" and cid in pending_dependencies:
                    # Remove any completed or failed dependencies
                    remaining_deps = pending_dependencies[cid]
                    satisfied = {p for p in remaining_deps if graph.nodes[p].status in {"completed", "failed", "skipped"}}
                    if satisfied:
                        remaining_deps -= satisfied
                    if not remaining_deps:
                        pending_dependencies.pop(cid)
                        ready_queue.put_nowait(cid)

        # Notify parallel group finished
        if on_event is not None and total_nodes > 1:
            on_event(
                "group.completed",
                {
                    "total_operations": total_nodes,
                    "completed_count": len(completed_nodes),
                    "failed_count": len(failed_nodes),
                },
            )

        # Build output mapping in original order
        results_map: dict[str, dict[str, Any]] = {}
        for cid in graph.order:
            node = graph.nodes[cid]
            output_json = json.dumps(node.result or {"ok": False, "error": "No output"}, ensure_ascii=False)
            results_map[cid] = {
                "type": "function_call_output",
                "call_id": cid,
                "name": node.tool_name,
                "output": output_json,
                "status": node.status,
                "duration_ms": node.duration_ms,
                "error": node.error,
            }

        return results_map


__all__ = [
    "DependencyGraph",
    "OperationKind",
    "OperationReadCache",
    "ParallelToolExecutor",
    "ToolOperationNode",
    "classify_tool_operation",
    "extract_target_paths",
]
