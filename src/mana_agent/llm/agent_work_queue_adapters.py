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

from mana_agent.llm.agent_session import AgentSession
from mana_agent.llm.agent_work_queue import TaskBoard, WorkItem, WorkResult
from mana_agent.llm.mutation_plan import (
    is_architecture_docs_update,
    mutation_trace_has_plan,
    representative_architecture_sources,
)
from mana_agent.llm.tool_worker_process import ToolRunRequest, ToolRunResponse
from mana_agent.llm.tools_executor import BatchToolRequest, ToolsExecutor

logger = logging.getLogger(__name__)

MUTATION_ONLY_TOOLS = [
    "edit_file",
    "multi_edit_file",
    "apply_patch",
    "apply_patch_batch",
    "write_file",
    "create_file",
    "delete_file",
]

_PATH_RE = re.compile(r"[\w./-]*?[\w-]+\.(?:py|md|txt|toml|yaml|yml|json|cfg|ini)\b")
_LOCAL_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([\w.]+)", re.MULTILINE)
_KEYWORD_RE = re.compile(r"[a-z0-9_]+")
# Filler words that carry no targeting signal; dropped before scoring candidates
# so reads are ranked by the request's real subject (e.g. "mana_logs"), not "make".
_REQUEST_STOPWORDS = frozenset(
    {
        "all", "the", "now", "need", "make", "and", "for", "with", "under",
        "that", "this", "from", "into", "new", "create", "add", "update",
        "change", "fix", "separately", "seperately", "please", "want",
    }
)

_AGENTIC_EDIT_TOOLS = [
    "edit_file",
    "multi_edit_file",
    "apply_patch",
    "write_file",
    "create_file",
    "delete_file",
]


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
    non_progress_statuses = {
        "blocked",
        "skipped",
        "duplicate_blocked",
        "not_allowed",
        "verify_project_blocked_until_mutation",
        "no_progress",
        "skipped_no_progress",
    }
    for row in response.trace:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "") or "").strip().lower()
        if status in {"error", "timeout", "failed", *non_progress_statuses}:
            return str(row.get("error") or row.get("output_preview") or status)
        result = str(row.get("result", "") or "").strip().lower()
        if result in non_progress_statuses:
            return str(row.get("error") or row.get("output_preview") or result)
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

    if tool in {"edit_file", "multi_edit_file", "apply_patch", "apply_patch_batch", "write_file", "create_file", "delete_file", "move_file"}:
        changed = sorted(paths)
        for row in response.trace:
            if isinstance(row, dict):
                for key in ("files_changed", "changed_files", "modified_files"):
                    value = row.get(key)
                    if isinstance(value, list):
                        changed.extend(str(path).strip().lstrip("./") for path in value if str(path).strip())
                proof = row.get("proof")
                if isinstance(proof, dict) and isinstance(proof.get("modified_files"), list):
                    changed.extend(str(path).strip().lstrip("./") for path in proof["modified_files"] if str(path).strip())
        changed = sorted(dict.fromkeys(path for path in changed if path))
        ok = not error and bool(changed)
        return WorkResult(
            ok=ok,
            summary="mutation applied" if ok else "mutation produced no change",
            error=error or ("" if ok else "mutation_no_modified_files"),
            files_changed=changed,
            answer=str(response.answer or ""),
            trace=list(response.trace),
        )

    if tool in {"repo_search", "repo_batch_search", "semantic_search", "list_files"}:
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
    run_id: str | None = None,
) -> Callable[[WorkItem], WorkResult]:
    """Build the ``execute`` callable that runs a :class:`WorkItem` on the worker."""
    repo_root = Path(repo_root).resolve()

    def _execute(item: WorkItem) -> WorkResult:
        if item.kind == "edit" and (item.tool_name or "").strip().lower() in MUTATION_ONLY_TOOLS:
            tool_args = dict(item.tool_args or {})
            plan_id = str(tool_args.get("mutation_plan_id") or "").strip()
            if plan_id:
                from mana_agent.llm.agent_work_queue import execute_registered_mutation_command
                from mana_agent.llm.mutation_plan import MutationCommand

                payload_args = {
                    key: value
                    for key, value in tool_args.items()
                    if key not in {"mutation_plan", "mutation_plan_id"}
                }
                plan_payload = tool_args.get("mutation_plan") if isinstance(tool_args.get("mutation_plan"), dict) else {}
                target_files = list(plan_payload.get("target_files") or [])
                if not target_files and payload_args.get("path"):
                    target_files = [str(payload_args.get("path"))]
                command = MutationCommand(
                    plan_id=plan_id,
                    tool_name=(item.tool_name or "").strip().lower(),  # type: ignore[arg-type]
                    tool_args=payload_args,
                    target_files=target_files,
                    reason="executed from queued registered mutation work item",
                )
                t0 = time.perf_counter()
                result = execute_registered_mutation_command(repo_root=repo_root, command=command)
                result.duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                return result
        request = build_tool_run_request(
            item,
            repo_root=repo_root,
            index_dir=index_dir,
            flow_id=flow_id,
            run_id=run_id,
            k=default_k,
            max_steps=default_max_steps,
            timeout_seconds=default_timeout,
            tool_policy=tool_policy,
        )
        t0 = time.perf_counter()
        try:
            response = worker_client.run_tools(request, on_event=on_event)
        except TypeError:
            response = worker_client.run_tools(request)
        except Exception as exc:
            return WorkResult(ok=False, error=f"worker_error: {exc}")
        if item.kind == "edit":
            tool_args = dict(item.tool_args or {})
            plan_id = str(tool_args.get("mutation_plan_id") or "").strip()
            if plan_id:
                for row in response.trace:
                    if not isinstance(row, dict):
                        continue
                    tool = str(row.get("tool_name") or row.get("tool") or row.get("name") or "").strip().lower()
                    if tool in set(MUTATION_ONLY_TOOLS):
                        row.setdefault("mutation_plan_id", plan_id)
                if not mutation_trace_has_plan(response.trace, plan_id):
                    return WorkResult(
                        ok=False,
                        summary="mutation did not execute with approved plan",
                        error="mutation_tool_missing_plan_id",
                        trace=list(response.trace),
                    )
        result = classify_result(item, response, repo_root=repo_root)
        result.duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return result

    return _execute


def _normalized_repo_path(path: str, *, repo_root: Path) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    try:
        candidate = Path(text)
        resolved = candidate if candidate.is_absolute() else (repo_root / candidate)
        return resolved.resolve().relative_to(repo_root).as_posix()
    except Exception:
        return text.replace("\\", "/").lstrip("./")


def _policy_for_item(item: WorkItem, tool_policy: dict[str, Any] | None) -> dict[str, Any]:
    item_policy = dict(tool_policy or {})
    if item.kind == "edit":
        # An edit item is an agentic analyze-then-write pass: the worker may
        # inspect the repo before authoring, but must finish with a mutation.
        item_policy["allowed_tools"] = list(_AGENTIC_EDIT_TOOLS)
        item_policy["require_read_files"] = 0
        item_policy["mutation_required"] = True
        item_policy["verify_requires_mutation"] = True
    else:
        item_policy.pop("mutation_required", None)
        item_policy.pop("mutation_strict", None)
        item_policy.pop("verify_requires_mutation", None)
    return item_policy


def build_tool_run_request(
    item: WorkItem,
    *,
    repo_root: Path,
    index_dir: str | None = None,
    index_dirs: list[str] | None = None,
    flow_id: str | None = None,
    run_id: str | None = None,
    k: int = 8,
    max_steps: int = 6,
    timeout_seconds: int = 60,
    tool_policy: dict[str, Any] | None = None,
) -> ToolRunRequest:
    question = item.question or (f"run tool {item.tool_name}" if item.tool_name else item.title)
    tool_args = dict(item.tool_args or {})
    if (item.tool_name or "").strip().lower() == "read_file" and tool_args.get("path"):
        tool_args["path"] = _normalized_repo_path(str(tool_args.get("path")), repo_root=repo_root)
    return ToolRunRequest(
        question=question,
        index_dir=index_dir,
        index_dirs=list(index_dirs or []) or None,
        flow_id=flow_id,
        run_id=run_id,
        k=int(k),
        max_steps=int(max_steps),
        timeout_seconds=int(timeout_seconds),
        tool_policy=_policy_for_item(item, tool_policy),
        tool_name=item.tool_name or "",
        tool_args=tool_args,
    )


def make_batch_executor(
    *,
    executor: ToolsExecutor,
    session: AgentSession,
    on_event: Callable[[Any], None] | None = None,
    default_timeout: int = 60,
    default_k: int = 8,
    default_max_steps: int = 6,
) -> Callable[[WorkItem], WorkResult]:
    """Build an execute callable that routes WorkItems through ToolsExecutor.run_batch."""
    repo_root = Path(session.repo_root).resolve()

    def _execute(item: WorkItem) -> WorkResult:
        request = build_tool_run_request(
            item,
            repo_root=repo_root,
            index_dir=session.index_dir,
            index_dirs=session.index_dirs,
            flow_id=session.flow_id,
            run_id=session.run_id,
            k=default_k,
            max_steps=default_max_steps,
            timeout_seconds=default_timeout,
            tool_policy=session.tool_policy,
        )
        results = executor.run_batch(
            run_id=session.run_id,
            requests=[BatchToolRequest(request_index=0, request=request)],
            on_event=on_event,
        )
        if not results:
            return WorkResult(ok=False, error="executor_returned_no_results")
        result = results[0]
        if not result.ok:
            trace = [
                {
                    "tool_name": item.tool_name or "",
                    "status": "failed",
                    "error_code": result.error_code,
                    "error": result.error_message,
                    "backend": result.backend,
                }
            ]
            return WorkResult(
                ok=False,
                summary=result.error_message or result.error_code or "tool execution failed",
                error=result.error_message or result.error_code or "tool execution failed",
                trace=trace,
                duration_ms=float(result.duration_ms or 0.0),
            )
        if not isinstance(result.response, dict):
            return WorkResult(ok=False, error="executor_result_missing_response")
        try:
            response = ToolRunResponse.model_validate(result.response)
        except Exception as exc:
            return WorkResult(ok=False, error=f"executor_result_decode_failed: {exc}")
        if item.kind == "edit":
            tool_args = dict(item.tool_args or {})
            plan_id = str(tool_args.get("mutation_plan_id") or "").strip()
            if plan_id:
                for row in response.trace:
                    if not isinstance(row, dict):
                        continue
                    tool = str(row.get("tool_name") or row.get("tool") or row.get("name") or "").strip().lower()
                    if tool in set(MUTATION_ONLY_TOOLS):
                        row.setdefault("mutation_plan_id", plan_id)
                if not mutation_trace_has_plan(response.trace, plan_id):
                    return WorkResult(
                        ok=False,
                        summary="mutation did not execute with approved plan",
                        error="mutation_tool_missing_plan_id",
                        trace=list(response.trace),
                    )
        work_result = classify_result(item, response, repo_root=repo_root)
        work_result.duration_ms = float(result.duration_ms or 0.0)
        return work_result

    return _execute


class CodingAgentSniffer:
    """Default live-steering hook: the coding agent emitting follow-up jobs.

    Heuristic, deterministic, and dedup-safe (the queue rejects duplicate
    fingerprints, so over-emitting is harmless):

    * a **search/discover** job that surfaced candidate files -> emit a ``read``
      job per file (capped) that the eventual edit depends on;
    * a **read** job whose content references sibling local modules -> emit
      reads for those modules (one hop, capped), so the agent follows the code;
    * once discovery has run, for a **mutating** request, emit the ``edit`` +
      ``verify`` jobs that actually fulfil it. They are queued at a high priority
      number so every read (priority ~30) is claimed first: the edit only runs
      once the evidence-gathering reads have drained. This is the transition
      from "read forever" to "act", and it keeps all next-step control in the
      coding-agent layer (the sniffer) rather than the worker.
    * never re-emits a path already read or queued, and emits finalization once.
    """

    def __init__(
        self,
        *,
        repo_root: Path,
        request: str = "",
        emit_edit: bool | None = None,
        target_files: list[str] | None = None,
        max_reads: int = 8,
        max_follow_per_read: int = 4,
        relevant: Callable[[str], bool] | None = None,
        orchestrator: Any | None = None,
    ) -> None:
        self._repo_root = Path(repo_root).resolve()
        self._request = str(request or "").strip()
        # Whether this run should end in an edit + verify is *recognized* by the
        # coding agent's planner (the LLM checklist), not guessed from keywords
        # in the request. The caller passes that decision down as ``emit_edit``;
        # when it is unknown (no planner signal) we do not force a mutation.
        self._emit_edit = bool(emit_edit)
        self._target_files = []
        self._max_reads = int(max_reads)
        self._max_follow_per_read = int(max_follow_per_read)
        self._relevant = relevant or (lambda _path: True)
        self._reads_emitted = 0
        self._finalization_emitted = False
        self._read_files: set[str] = set()
        self._orchestrator = orchestrator
        self._target_files = [
            self._normalize_repo_path(str(item))
            for item in (target_files or [])
            if str(item).strip()
        ]
        if is_architecture_docs_update(self._request, self._target_files):
            self._max_reads = max(self._max_reads, len(representative_architecture_sources(self._repo_root)) + len(self._target_files))

    def _normalize_repo_path(self, path: str) -> str:
        text = str(path or "").strip()
        if not text:
            return ""
        try:
            candidate = Path(text)
            resolved = candidate if candidate.is_absolute() else (self._repo_root / candidate)
            return resolved.resolve().relative_to(self._repo_root).as_posix()
        except Exception:
            return text.replace("\\", "/").lstrip("./")

    def on_result(self, item: WorkItem, result: WorkResult, *, board: TaskBoard) -> list[WorkItem]:
        if not result.ok:
            return []
        kind = item.kind
        if kind in {"search", "discover"}:
            out = self._reads_from_discovery(result.files_discovered, parent=item)
            # Discovery has produced candidates (or none): schedule the edit +
            # verify now so they sit behind the reads and run once evidence is in.
            out.extend(self._finalization_jobs())
            return out
        if kind == "read":
            self._read_files.update(result.files_read)
            out: list[WorkItem] = []
            if self._orchestrator is not None and self._orchestrator.state.evidence_sufficient:
                out.extend(self._finalization_jobs())
                return out
            out.extend(self._follow_local_imports(result, parent=item))
            return out
        return []

    def _finalization_jobs(self) -> list[WorkItem]:
        """Emit the edit + verify jobs for a mutating request, exactly once.

        They depend on nothing but carry a high priority *number*, so the runner
        (which claims lowest priority first) drains every read/discovery job
        before claiming the edit, and the verify only runs after the edit
        succeeds.
        """
        if self._finalization_emitted or not self._emit_edit or not self._request:
            return []
        self._finalization_emitted = True
        target_file = self._target_files[0] if self._target_files else ""
        target_instruction = (
            f" Target file: {target_file}. Create it if it does not exist."
            if target_file
            else ""
        )
        edit = WorkItem(
            kind="edit",
            tool_name="",
            tool_args={},
            question=(
                "Using the file evidence already gathered in this run, carry out "
                f"the user's request: {self._request}. "
                "Apply concrete changes with "
                "edit_file/multi_edit_file/apply_patch/create_file/write_file/delete_file and report the changed files. "
                "Before mutating, use bounded exact path/name/symbol evidence to account for "
                "related importers, exports, registries, routers, commands, call sites, tests, "
                "and stale docs/config references; update or remove each one required for the "
                "project to remain working."
                f"{target_instruction}"
            ),
            gate="apply_edit",
            priority=80,
            created_by="coding_agent_sniffer",
        )
        verify = WorkItem(
            kind="verify",
            tool_name="verify",
            question=(
                f"Verify the changes made to {target_file or 'the resolved target files'}. "
                f"User request context: {self._request}. Confirm the new or edited files exist and are "
                "well-formed, and run any available checks."
            ),
            gate="verify_changes",
            priority=90,
            created_by="coding_agent_sniffer",
            dependencies=[edit.id],
        )
        return [edit, verify]

    def _request_keywords(self) -> set[str]:
        """Targeting tokens from the request, with filler words removed."""
        tokens = {tok for tok in _KEYWORD_RE.findall(self._request.lower()) if len(tok) >= 3}
        return tokens - _REQUEST_STOPWORDS

    def _candidate_score(self, path: str, keywords: set[str]) -> int:
        low = path.lower()
        return sum(1 for kw in keywords if kw in low)

    def _rank_candidates(self, paths: list[str]) -> list[str]:
        """Relevant candidates ordered by request-keyword overlap (desc).

        Falls back to deterministic alphabetical order when the request carries
        no usable keywords or no path matches, so the (now bounded) read fan-out
        still spends its budget on the files most likely to matter rather than
        an arbitrary slice of every search hit.
        """
        relevant = list(dict.fromkeys(p for p in (self._normalize_repo_path(path) for path in paths) if p and self._relevant(p)))
        keywords = self._request_keywords()
        # Drop keywords that match every candidate (e.g. the repo name): they add
        # only noise to the score and let an arbitrary file float to the top.
        if relevant:
            keywords = {
                kw for kw in keywords
                if not all(kw in path.lower() for path in relevant)
            }
        if not keywords:
            return sorted(relevant)
        return sorted(relevant, key=lambda p: (-self._candidate_score(p, keywords), p))

    def _reads_from_discovery(self, paths: list[str], *, parent: WorkItem) -> list[WorkItem]:
        out: list[WorkItem] = []
        create_intent = bool(re.search(r"\b(create|add|generate)\b", self._request, re.IGNORECASE))
        ranked = [
            path for path in self._target_files
            if (self._repo_root / path).is_file() or not create_intent
        ]
        if is_architecture_docs_update(self._request, self._target_files):
            ranked.extend(path for path in representative_architecture_sources(self._repo_root) if path not in ranked)
        ranked.extend(path for path in self._rank_candidates(paths) if path not in ranked)
        for path in ranked:
            if self._reads_emitted >= self._max_reads:
                break
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
    "build_tool_run_request",
    "classify_result",
    "make_batch_executor",
    "make_worker_executor",
]
