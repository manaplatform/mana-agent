"""Adapters wiring the Live Agent Work Queue to the real tool stack.

Two pieces live here so the core ``agent_work_queue`` module stays
dependency-light and unit-testable without a worker subprocess:

* ``make_worker_executor`` -- turns a :class:`ToolWorkerClient` into the
  ``execute`` callable the runner needs, and applies the **read-success fix**:
  a ``read_file`` job succeeds when the worker actually returned the file
  (no error + non-empty result), *not* when a path can be regex-scraped out of
  the answer prose. That false-negative was making every file read twice.

* ``CodingAgentSniffer`` -- the default coding-agent steering hook. After each
  job it inspects the result and emits the next jobs: searches that surface
  candidate files emit (deduplicated) read jobs; reads emit follow-up reads for
  referenced local modules; once enough evidence is gathered it can emit the
  edit + verify jobs. This is the coding agent sitting on top of the hierarchy
  and feeding work back into the queue.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

from mana_analyzer.llm.agent_work_queue import TaskBoard, WorkItem, WorkResult
from mana_analyzer.llm.tool_worker_process import ToolRunRequest, ToolRunResponse

logger = logging.getLogger(__name__)

_PATH_RE = re.compile(r"[\w./-]*?[\w-]+\.(?:py|md|txt|toml|yaml|yml|json|cfg|ini)\b")
_LOCAL_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([\w.]+)", re.MULTILINE)


def _extract_paths(response: ToolRunResponse, *, repo_root: Path) -> set[str]:
    """Best-effort repo-relative paths mentioned anywhere in a response.

    Used only for *bookkeeping* (what got read / discovered) -- never to decide
    whether a read succeeded.
    """
    blobs: list[str] = [str(response.answer or "")]
    for src in response.sources:
        if isinstance(src, dict):
            blobs.append(" ".join(str(v) for v in src.values()))
    for row in response.trace:
        if isinstance(row, dict):
            blobs.append(" ".join(str(v) for v in row.values()))
    found: set[str] = set()
    for blob in blobs:
        for match in _PATH_RE.findall(blob):
            rel = match.lstrip("./")
            try:
                if (repo_root / rel).is_file():
                    found.add(rel)
            except OSError:
                continue
    return found


def _response_has_error(response: ToolRunResponse) -> str:
    """Return an error string if any trace row reports a hard failure."""
    for row in response.trace:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "") or "").strip().lower()
        if status in {"error", "timeout", "failed"}:
            return str(row.get("error") or row.get("output_preview") or status)
        preview = str(row.get("output_preview", "") or "").lower()
        if '"ok": false' in preview or "'ok': false" in preview:
            return "tool reported ok=false"
    return ""


def classify_result(item: WorkItem, response: ToolRunResponse, *, repo_root: Path) -> WorkResult:
    """Turn a worker response into a :class:`WorkResult`.

    The read-success rule is the fix for the double-read bug: a read is
    successful if the worker came back without an error and produced *any*
    content. We still record the concrete paths for the sniffer, but they do
    not gate success.
    """
    paths = _extract_paths(response, repo_root=repo_root)
    error = _response_has_error(response)
    has_content = bool(str(response.answer or "").strip() or response.trace or response.sources)
    tool = (item.tool_name or "").strip().lower()

    if tool == "read_file":
        target = str((item.tool_args or {}).get("path") or "").lstrip("./")
        ok = not error and has_content
        files_read = sorted(paths | ({target} if (ok and target) else set()))
        return WorkResult(
            ok=ok,
            summary=f"read {target or 'file'} ({len(str(response.answer or ''))} chars)" if ok else "read produced no content",
            error=error or ("" if ok else "read_file_empty_response"),
            files_read=files_read,
            answer=str(response.answer or ""),
            sources=list(response.sources),
            trace=list(response.trace),
        )

    if tool in {"apply_patch", "write_file", "create_file"}:
        ok = not error and has_content
        return WorkResult(
            ok=ok,
            summary="mutation applied" if ok else "mutation produced no change",
            error=error or ("" if ok else "mutation_no_modified_files"),
            files_changed=sorted(paths),
            answer=str(response.answer or ""),
            trace=list(response.trace),
        )

    if tool in {"repo_search", "semantic_search", "list_files"}:
        ok = not error and (bool(paths) or has_content)
        return WorkResult(
            ok=ok,
            summary=f"discovered {len(paths)} file(s)" if paths else "search returned context",
            error=error or ("" if ok else "search_no_candidates"),
            files_discovered=sorted(paths),
            answer=str(response.answer or ""),
            sources=list(response.sources),
            trace=list(response.trace),
        )

    # run_command / verify / generic
    ok = not error and has_content
    return WorkResult(
        ok=ok,
        summary=(str(response.answer or "")[:160]) if ok else "no result",
        error=error or ("" if ok else "tool_result_missing"),
        files_discovered=sorted(paths),
        answer=str(response.answer or ""),
        trace=list(response.trace),
    )


def make_worker_executor(
    *,
    worker_client: Any,
    repo_root: Path,
    on_event: Callable[[Any], None] | None = None,
    default_timeout: int = 60,
    default_k: int = 8,
    default_max_steps: int = 6,
    tool_policy: dict[str, Any] | None = None,
    index_dir: str | None = None,
    flow_id: str | None = None,
) -> Callable[[WorkItem], WorkResult]:
    """Build the ``execute`` callable that runs a :class:`WorkItem` on the worker."""
    repo_root = Path(repo_root).resolve()

    def _execute(item: WorkItem) -> WorkResult:
        question = item.question or (f"run tool {item.tool_name}" if item.tool_name else item.title)
        request = ToolRunRequest(
            question=question,
            index_dir=index_dir,
            flow_id=flow_id,
            k=int(default_k),
            max_steps=int(default_max_steps),
            timeout_seconds=int(default_timeout),
            tool_policy=tool_policy,
            tool_name=item.tool_name or "",
            tool_args=dict(item.tool_args or {}),
        )
        t0 = time.perf_counter()
        try:
            response = worker_client.run_tools(request, on_event=on_event)
        except TypeError:
            response = worker_client.run_tools(request)
        except Exception as exc:
            return WorkResult(ok=False, error=f"worker_error: {exc}")
        result = classify_result(item, response, repo_root=repo_root)
        result.duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return result

    return _execute


class CodingAgentSniffer:
    """Default live-steering hook: the coding agent emitting follow-up jobs.

    Heuristic, deterministic, and dedup-safe (the queue rejects duplicate
    fingerprints, so over-emitting is harmless):

    * a **search/discover** job that surfaced candidate files -> emit a ``read``
      job per file (capped) that the eventual edit depends on;
    * a **read** job whose content references sibling local modules -> emit
      reads for those modules (one hop, capped), so the agent follows the code;
    * never re-emits a path already read or queued.
    """

    def __init__(
        self,
        *,
        repo_root: Path,
        max_reads: int = 40,
        max_follow_per_read: int = 4,
        relevant: Callable[[str], bool] | None = None,
    ) -> None:
        self._repo_root = Path(repo_root).resolve()
        self._max_reads = int(max_reads)
        self._max_follow_per_read = int(max_follow_per_read)
        self._relevant = relevant or (lambda _path: True)
        self._reads_emitted = 0

    def on_result(self, item: WorkItem, result: WorkResult, *, board: TaskBoard) -> list[WorkItem]:
        if not result.ok:
            return []
        kind = item.kind
        if kind in {"search", "discover"}:
            return self._reads_from_discovery(result.files_discovered, parent=item)
        if kind == "read":
            return self._follow_local_imports(result, parent=item)
        return []

    def _reads_from_discovery(self, paths: list[str], *, parent: WorkItem) -> list[WorkItem]:
        out: list[WorkItem] = []
        for path in paths:
            if self._reads_emitted >= self._max_reads:
                break
            if not self._relevant(path):
                continue
            out.append(
                WorkItem(
                    kind="read",
                    tool_name="read_file",
                    tool_args={"path": path},
                    question=f"Read candidate file {path}",
                    gate="read_candidates",
                    priority=30,
                    created_by="coding_agent_sniffer",
                    dependencies=[],
                )
            )
            self._reads_emitted += 1
        return out

    def _follow_local_imports(self, result: WorkResult, *, parent: WorkItem) -> list[WorkItem]:
        modules = _LOCAL_IMPORT_RE.findall(result.answer or "")
        out: list[WorkItem] = []
        follows = 0
        for module in modules:
            if follows >= self._max_follow_per_read or self._reads_emitted >= self._max_reads:
                break
            candidate = self._module_to_path(module)
            if candidate is None or not self._relevant(candidate):
                continue
            out.append(
                WorkItem(
                    kind="read",
                    tool_name="read_file",
                    tool_args={"path": candidate},
                    question=f"Read referenced module {candidate}",
                    gate="read_candidates",
                    priority=35,
                    created_by="coding_agent_sniffer",
                )
            )
            follows += 1
            self._reads_emitted += 1
        return out

    def _module_to_path(self, module: str) -> str | None:
        rel = module.replace(".", "/")
        for suffix in (f"{rel}.py", f"{rel}/__init__.py"):
            try:
                if (self._repo_root / suffix).is_file():
                    return suffix
            except OSError:
                continue
        return None


__all__ = [
    "CodingAgentSniffer",
    "classify_result",
    "make_worker_executor",
]
