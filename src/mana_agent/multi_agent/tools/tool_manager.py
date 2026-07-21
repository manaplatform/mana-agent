from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from mana_agent.multi_agent.core.ids import new_message_id
from mana_agent.multi_agent.core.types import QueueJob, QueueJobType, ToolResult
from mana_agent.memory import MultiAgentMemoryService
from mana_agent.documents.service import DocumentService
from mana_agent.multi_agent.tools import git_tools
from mana_agent.multi_agent.tools.permissions import assert_shell_allowed
from mana_agent.tools.apply_patch import safe_apply_patch
from mana_agent.tools.repository import repo_batch_read, repo_search
from mana_agent.mcp import McpClient, load_mcp_servers
from mana_agent.execution import build_execution_manager
from mana_agent.execution.manager import ExecutionManager
from mana_agent.execution.models import (
    ExecutionRequest,
    NetworkPolicy,
    ResourceLimits,
    RoutingRequest,
    SandboxSpec,
)
from mana_agent.config.settings import Settings


class ToolsManager:
    def __init__(
        self,
        root: str | Path = ".",
        *,
        memory_service: MultiAgentMemoryService | None = None,
        execution_manager: ExecutionManager | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.memory_service = memory_service
        self.execution_manager = execution_manager or build_execution_manager(Settings())
        self._result_cache: dict[str, dict[str, Any]] = {}

    def execution_root_for_job(self, job: QueueJob) -> Path:
        """Resolve the repository root for one job without mutating process cwd.

        Preference order:
        1. job.execution_repo_root (managed worktree)
        2. payload execution_repo_root / repo_root / worktree_path
        3. manager default root (primary checkout)
        """

        candidates = [
            getattr(job, "execution_repo_root", None),
            (job.payload or {}).get("execution_repo_root"),
            (job.payload or {}).get("repo_root"),
            (job.payload or {}).get("worktree_path"),
            (job.payload or {}).get("workspace_root"),
        ]
        for raw in candidates:
            text = str(raw or "").strip()
            if not text:
                continue
            path = Path(text).expanduser().resolve()
            if path.is_dir():
                return path
        return self.root

    def execute_job(self, job: QueueJob) -> ToolResult:
        try:
            root = self.execution_root_for_job(job)
            cache_key = self._cache_key(job, root=root)
            if cache_key and self.memory_service is not None:
                cached_tool = self.memory_service.get_reusable_tool_result(
                    tool_name=job.job_type.value,
                    args=job.payload,
                )
                if cached_tool is not None:
                    result = copy.deepcopy(cached_tool.result)
                    result["cache_hit"] = True
                    result["source"] = "memory"
                    result["cache_source"] = "memory"
                    result["execution_repo_root"] = str(root)
                    return ToolResult(new_message_id(), job.task_id, True, result)
            if cache_key:
                cached = self._result_cache.get(cache_key)
                if cached is not None:
                    result = copy.deepcopy(cached)
                    result["cache_hit"] = True
                    result["cache_source"] = "tool_result_cache"
                    result["execution_repo_root"] = str(root)
                    return ToolResult(new_message_id(), job.task_id, True, result)

            if job.job_type == QueueJobType.GIT_STATUS:
                result = git_tools.status(repo_path=root)
                result = {**result, "execution_repo_root": str(root)}
                return self._cache_result(
                    job,
                    ToolResult(
                        new_message_id(),
                        job.task_id,
                        bool(result.get("ok")),
                        result,
                        None if result.get("ok") else str(result.get("stderr") or result.get("error") or "git status failed"),
                    ),
                    cache_key,
                )
            if job.job_type == QueueJobType.GIT_DIFF:
                result = git_tools.diff(
                    repo_path=root,
                    path=str(job.payload.get("path") or ""),
                    staged=bool(job.payload.get("staged", False)),
                )
                result = {**result, "execution_repo_root": str(root)}
                return self._cache_result(
                    job,
                    ToolResult(
                        new_message_id(),
                        job.task_id,
                        bool(result.get("ok")),
                        result,
                        None if result.get("ok") else str(result.get("stderr") or result.get("error") or "git diff failed"),
                    ),
                    cache_key,
                )
            if job.job_type == QueueJobType.GIT:
                tool_name = str(job.payload.get("tool") or job.payload.get("tool_name") or "git.generic")
                tool_args = job.payload.get("args") if isinstance(job.payload.get("args"), dict) else dict(job.payload)
                result = git_tools.execute_tool(tool_name, tool_args, repo_path=root)
                if isinstance(result, dict):
                    result = {**result, "execution_repo_root": str(root)}
                return ToolResult(
                    new_message_id(),
                    job.task_id,
                    bool(result.get("ok")),
                    result,
                    None if result.get("ok") else str(result.get("stderr") or result.get("error") or "git tool failed"),
                )
            if job.job_type == QueueJobType.DOCUMENT:
                result = self._execute_document_tool(job.payload, root=root)
                if isinstance(result, dict):
                    result = {**result, "execution_repo_root": str(root)}
                return ToolResult(
                    new_message_id(),
                    job.task_id,
                    bool(result.get("ok")),
                    result,
                    None if result.get("ok") else str(result.get("error") or "document tool failed"),
                )
            if job.job_type == QueueJobType.BROWSER:
                from mana_agent.connectors.browser.runtime_tools import build_browser_langchain_tools
                tool_name = str(job.payload.get("tool") or job.payload.get("tool_name") or "")
                args = dict(job.payload.get("args") or {})
                tools = {tool.name: tool for tool in build_browser_langchain_tools()}
                selected = tools.get(tool_name)
                if selected is None:
                    return ToolResult(new_message_id(), job.task_id, False, error=f"unsupported browser tool: {tool_name}")
                raw = selected.invoke(args)
                result = json.loads(raw) if isinstance(raw, str) else raw
                ok = bool(isinstance(result, dict) and result.get("ok"))
                return ToolResult(new_message_id(), job.task_id, ok, result, None if ok else str(result.get("message") or result.get("error_code") or "browser tool failed"))
            if job.job_type == QueueJobType.MCP_TOOL:
                client = McpClient(load_mcp_servers(overrides=list(job.payload.get("server_overrides") or [])))
                result = client.call_tool(
                    str(job.payload.get("tool") or job.payload.get("tool_name") or ""),
                    dict(job.payload.get("args") or {}),
                )
                return ToolResult(
                    new_message_id(), job.task_id, bool(result.get("ok")), result,
                    None if result.get("ok") else f"MCP tool failed: {result.get('tool_name') or 'unknown'}",
                )
            if job.job_type == QueueJobType.MCP_RESOURCE_READ:
                client = McpClient(load_mcp_servers(overrides=list(job.payload.get("server_overrides") or [])))
                result = client.read_resource(str(job.payload.get("server_id") or ""), str(job.payload.get("uri") or ""))
                return ToolResult(new_message_id(), job.task_id, bool(result.get("ok")), result, None if result.get("ok") else "MCP resource read failed")
            if job.job_type in {QueueJobType.SHELL, QueueJobType.RUN_TESTS, QueueJobType.RUN_LINT}:
                return self._shell(job, str(job.payload.get("command", "")), root=root)
            if job.job_type == QueueJobType.REPO_READ:
                path = self._resolve_path(str(job.payload.get("path", "")), root=root)
                return self._cache_result(
                    job,
                    ToolResult(
                        new_message_id(),
                        job.task_id,
                        True,
                        {"content": path.read_text(encoding="utf-8"), "path": str(path), "execution_repo_root": str(root)},
                    ),
                    cache_key,
                )
            if job.job_type == QueueJobType.REPO_BATCH_READ:
                paths = job.payload.get("files") or job.payload.get("paths") or []
                if self.memory_service is not None:
                    files = []
                    ok = True
                    for raw in paths:
                        try:
                            content, record, cache_hit = self.memory_service.read_file_with_memory(
                                file_path=str(raw),
                                task_id=job.task_id,
                                agent_id=job.requested_by_agent_id,
                            )
                            files.append(
                                {
                                    "path": record.file_path,
                                    "content": content,
                                    "ok": True,
                                    "source": "memory" if cache_hit else "tool",
                                    "cache_hit": cache_hit,
                                }
                            )
                        except Exception as exc:
                            ok = False
                            files.append({"path": str(raw), "ok": False, "error": str(exc)})
                    return self._cache_result(
                        job,
                        ToolResult(
                            new_message_id(),
                            job.task_id,
                            ok,
                            {"ok": ok, "files": files, "execution_repo_root": str(root)},
                            None if ok else "one or more batch reads failed",
                        ),
                        cache_key,
                    )
                result = repo_batch_read(root, files=[str(item) for item in paths])
                ok = bool(result.get("ok"))
                if isinstance(result, dict):
                    result = {**result, "execution_repo_root": str(root)}
                return self._cache_result(
                    job,
                    ToolResult(new_message_id(), job.task_id, ok, result, None if ok else "one or more batch reads failed"),
                    cache_key,
                )
            if job.job_type == QueueJobType.APPLY_PATCH:
                patch = str(job.payload.get("patch", ""))
                result = safe_apply_patch(repo_root=root, patch=patch, check_only=False)
                ok = bool(result.get("ok"))
                error = None if ok else str(result.get("error") or result.get("message") or "patch failed")
                if not ok and result.get("error_code") == "patch_context_not_found":
                    # Recovery already ran inside apply_patch. Surface diagnostic fields and
                    # force a fresh read of every touched file before any further edit attempt.
                    touched = list(result.get("touched_files") or [])
                    reread_files: list[dict[str, Any]] = []
                    for rel in touched:
                        target = root / str(rel)
                        try:
                            content = target.read_text(encoding="utf-8")
                            reread_files.append(
                                {
                                    "path": str(rel),
                                    "ok": True,
                                    "content": content,
                                    "bytes_read": len(content.encode("utf-8")),
                                }
                            )
                        except OSError as exc:
                            reread_files.append({"path": str(rel), "ok": False, "error": str(exc)})
                    result = {
                        **result,
                        "reread_files": reread_files,
                        "recovery_required": False,
                        "note": (
                            "patch recovery exhausted; fresh file contents attached in reread_files. "
                            "Do not resubmit the original stale patch unchanged."
                        ),
                    }
                    error = (
                        "patch_context_not_found; reread target file before rebuilding patch. "
                        f"strategy={result.get('strategy') or 'none'} "
                        f"candidates={result.get('candidate_count') or 0}"
                    )
                if isinstance(result, dict):
                    result = {**result, "execution_repo_root": str(root)}
                return ToolResult(new_message_id(), job.task_id, ok, result, error)
            if job.job_type == QueueJobType.REPO_SEARCH:
                query = str(job.payload.get("query", ""))
                result = repo_search(
                    root,
                    query=query,
                    glob=str(job.payload.get("glob") or "**/*"),
                    regex=bool(job.payload.get("regex", False)),
                    limit=int(job.payload.get("limit") or 100),
                )
                stdout = "\n".join(
                    f"{item.get('file')}:{item.get('line')}:{item.get('text')}"
                    for item in result.get("matches", [])
                    if isinstance(item, dict)
                )
                return self._cache_result(
                    job,
                    ToolResult(
                        new_message_id(),
                        job.task_id,
                        bool(result.get("ok")),
                        {
                            **result,
                            "stdout": stdout,
                            "stderr": str(result.get("error") or ""),
                            "returncode": 0 if bool(result.get("ok")) else 1,
                            "execution_repo_root": str(root),
                        },
                    ),
                    cache_key,
                )
            return ToolResult(new_message_id(), job.task_id, False, error=f"unsupported tool job: {job.job_type.value}")
        except Exception as exc:
            return ToolResult(new_message_id(), job.task_id, False, error=str(exc))

    def _shell(self, job: QueueJob, command: str, *, root: Path | None = None) -> ToolResult:
        assert_shell_allowed(command)
        cwd = Path(root or self.execution_root_for_job(job)).resolve()
        routing_payload = (job.payload or {}).get("sandbox_routing")
        if isinstance(routing_payload, dict):
            routing = RoutingRequest.model_validate(routing_payload)
        else:
            # Queue jobs are created only after a validated agent decision. This
            # explicit compatibility decision preserves trusted local execution;
            # it is persisted by the fabric and never chosen from command text.
            routing = RoutingRequest(
                decision_id=f"queue:{job.job_id}",
                explicit_provider=self.execution_manager.config.default_provider,
                trust_level="trusted",
                risk_level="low",
                resources=ResourceLimits(),
                network=NetworkPolicy(),
            )
        shell_argv = ["cmd.exe", "/d", "/s", "/c", command] if os.name == "nt" else ["/bin/sh", "-lc", command]
        spec = SandboxSpec(
            provider_override=routing.explicit_provider,
            repository_source=cwd,
            task_id=job.task_id,
            session_id=str(getattr(job, "session_id", "") or ""),
            workspace_id=str(getattr(job, "workspace_id", "") or ""),
            execution_timeout_seconds=int((job.payload or {}).get("timeout_seconds") or 120),
        )
        result = self.execution_manager.execute_once_sync(
            spec,
            routing,
            ExecutionRequest(
                argv=shell_argv,
                timeout_seconds=spec.execution_timeout_seconds,
            ),
        )
        return ToolResult(
            new_message_id(),
            job.task_id,
            result.exit_code == 0,
            {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.exit_code,
                "execution_repo_root": str(cwd),
                "sandbox_id": result.sandbox_id,
                "execution_provider": result.provider,
            },
            None if result.exit_code == 0 else result.stderr,
        )

    def _resolve_path(self, path: str, *, root: Path | None = None) -> Path:
        base = Path(root or self.root).resolve()
        # Reject absolute paths that escape; relative paths resolve under base.
        candidate = Path(path)
        resolved = candidate.resolve() if candidate.is_absolute() else (base / path).resolve()
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise ValueError("path escapes repository root") from exc
        return resolved

    def _cache_key(self, job: QueueJob, *, root: Path | None = None) -> str:
        if job.job_type not in {
            QueueJobType.GIT,
            QueueJobType.GIT_STATUS,
            QueueJobType.GIT_DIFF,
            QueueJobType.REPO_READ,
            QueueJobType.REPO_BATCH_READ,
            QueueJobType.REPO_SEARCH,
            QueueJobType.DOCUMENT,
            QueueJobType.MCP_TOOL,
            QueueJobType.MCP_RESOURCE_READ,
        }:
            return ""
        payload = json.dumps(job.payload, ensure_ascii=False, sort_keys=True, default=str)
        root_key = str(root or self.execution_root_for_job(job))
        return f"{job.job_type.value}:{root_key}:{payload}"

    def _cache_result(self, job: QueueJob, result: ToolResult, cache_key: str) -> ToolResult:
        if cache_key and result.ok:
            payload = copy.deepcopy(result.result)
            payload.setdefault("cache_hit", False)
            payload.setdefault("source", "tool")
            payload.setdefault("cache_source", payload["source"])
            self._result_cache[cache_key] = copy.deepcopy(payload)
            if self.memory_service is not None:
                self.memory_service.record_tool_execution(
                    tool_name=job.job_type.value,
                    args=job.payload,
                    task_id=job.task_id,
                    agent_id=job.requested_by_agent_id,
                    status="ok",
                    result_summary="ok",
                    result=copy.deepcopy(payload),
                )
            result.result = payload
        return result

    def _execute_document_tool(self, payload: dict[str, Any], *, root: Path | None = None) -> dict[str, Any]:
        service = DocumentService(Path(root or self.root).resolve())
        tool_name = str(payload.get("tool") or payload.get("tool_name") or "")
        args = payload.get("args") if isinstance(payload.get("args"), dict) else dict(payload)
        if tool_name == "document_detect":
            return service.detect(str(args.get("path") or ""), mime_type=args.get("mime_type"))
        if tool_name == "document_read":
            return service.read(str(args.get("path") or ""), use_cache=bool(args.get("use_cache", True)), max_chunks=int(args.get("max_chunks") or 400))
        if tool_name == "document_analyze":
            return service.analyze(str(args.get("path") or ""))
        if tool_name == "document_query":
            return service.query(
                str(args.get("query") or ""),
                paths=args.get("paths"),
                file_types=args.get("file_types"),
                path_filter=str(args.get("path_filter") or ""),
                sheet=str(args.get("sheet") or ""),
                page=args.get("page"),
                section=str(args.get("section") or ""),
                limit=int(args.get("limit") or 10),
            )
        if tool_name == "document_create":
            return service.create(
                str(args.get("path") or ""),
                content=args.get("content") or {},
                file_type=args.get("file_type"),
                overwrite=bool(args.get("overwrite", False)),
            )
        if tool_name == "document_update":
            return service.update(
                str(args.get("path") or ""),
                operation=str(args.get("operation") or ""),
                payload=dict(args.get("payload") or {}),
                backup=bool(args.get("backup", True)),
            )
        if tool_name == "document_delete":
            return service.delete(
                str(args.get("path") or ""),
                explicit=bool(args.get("explicit", False)),
                backup=bool(args.get("backup", True)),
            )
        return {"ok": False, "error": "unsupported_document_tool", "tool": tool_name}
