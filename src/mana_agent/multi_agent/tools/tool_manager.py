from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from mana_agent.multi_agent.core.ids import new_message_id
from mana_agent.multi_agent.core.types import QueueJob, QueueJobType, ToolResult
from mana_agent.services.memory_service import MultiAgentMemoryService
from mana_agent.documents.service import DocumentService
from mana_agent.multi_agent.tools import git_tools
from mana_agent.multi_agent.tools.permissions import assert_shell_allowed
from mana_agent.tools.apply_patch import safe_apply_patch
from mana_agent.tools.repository import repo_batch_read, repo_search
from mana_agent.mcp import McpClient, load_mcp_servers


class ToolsManager:
    def __init__(self, root: str | Path = ".", *, memory_service: MultiAgentMemoryService | None = None) -> None:
        self.root = Path(root).resolve()
        self.memory_service = memory_service
        self._result_cache: dict[str, dict[str, Any]] = {}

    def execute_job(self, job: QueueJob) -> ToolResult:
        try:
            cache_key = self._cache_key(job)
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
                    return ToolResult(new_message_id(), job.task_id, True, result)
            if cache_key:
                cached = self._result_cache.get(cache_key)
                if cached is not None:
                    result = copy.deepcopy(cached)
                    result["cache_hit"] = True
                    result["cache_source"] = "tool_result_cache"
                    return ToolResult(new_message_id(), job.task_id, True, result)

            if job.job_type == QueueJobType.GIT_STATUS:
                result = git_tools.status(repo_path=self.root)
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
                    repo_path=self.root,
                    path=str(job.payload.get("path") or ""),
                    staged=bool(job.payload.get("staged", False)),
                )
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
                result = git_tools.execute_tool(tool_name, tool_args, repo_path=self.root)
                return ToolResult(
                    new_message_id(),
                    job.task_id,
                    bool(result.get("ok")),
                    result,
                    None if result.get("ok") else str(result.get("stderr") or result.get("error") or "git tool failed"),
                )
            if job.job_type == QueueJobType.DOCUMENT:
                result = self._execute_document_tool(job.payload)
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
                return self._shell(job, str(job.payload.get("command", "")))
            if job.job_type == QueueJobType.REPO_READ:
                path = self._resolve_path(str(job.payload.get("path", "")))
                return self._cache_result(
                    job,
                    ToolResult(new_message_id(), job.task_id, True, {"content": path.read_text(encoding="utf-8"), "path": str(path)}),
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
                        ToolResult(new_message_id(), job.task_id, ok, {"ok": ok, "files": files}, None if ok else "one or more batch reads failed"),
                        cache_key,
                    )
                result = repo_batch_read(self.root, files=[str(item) for item in paths])
                ok = bool(result.get("ok"))
                return self._cache_result(
                    job,
                    ToolResult(new_message_id(), job.task_id, ok, result, None if ok else "one or more batch reads failed"),
                    cache_key,
                )
            if job.job_type == QueueJobType.APPLY_PATCH:
                patch = str(job.payload.get("patch", ""))
                result = safe_apply_patch(repo_root=self.root, patch=patch, check_only=False)
                ok = bool(result.get("ok"))
                error = None if ok else str(result.get("error") or result.get("message") or "patch failed")
                if not ok and result.get("error_code") == "patch_context_not_found":
                    error = "patch_context_not_found; reread target file before rebuilding patch"
                return ToolResult(new_message_id(), job.task_id, ok, result, error)
            if job.job_type == QueueJobType.REPO_SEARCH:
                query = str(job.payload.get("query", ""))
                result = repo_search(
                    self.root,
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
                        },
                    ),
                    cache_key,
                )
            return ToolResult(new_message_id(), job.task_id, False, error=f"unsupported tool job: {job.job_type.value}")
        except Exception as exc:
            return ToolResult(new_message_id(), job.task_id, False, error=str(exc))

    def _shell(self, job: QueueJob, command: str) -> ToolResult:
        assert_shell_allowed(command)
        result = subprocess.run(command, cwd=self.root, text=True, capture_output=True, shell=True, timeout=120)
        return ToolResult(new_message_id(), job.task_id, result.returncode == 0, {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}, None if result.returncode == 0 else result.stderr)

    def _resolve_path(self, path: str) -> Path:
        resolved = (self.root / path).resolve()
        if self.root not in resolved.parents and resolved != self.root:
            raise ValueError("path escapes repository root")
        return resolved

    def _cache_key(self, job: QueueJob) -> str:
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
        return f"{job.job_type.value}:{payload}"

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

    def _execute_document_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        service = DocumentService(self.root)
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
