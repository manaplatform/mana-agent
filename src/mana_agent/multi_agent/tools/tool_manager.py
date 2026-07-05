from __future__ import annotations

import subprocess
from pathlib import Path

from mana_agent.multi_agent.core.ids import new_message_id
from mana_agent.multi_agent.core.types import QueueJob, QueueJobType, ToolResult
from mana_agent.multi_agent.memory.service import MultiAgentMemoryService
from mana_agent.multi_agent.tools.permissions import assert_shell_allowed
from mana_agent.tools import repo_batch_read, safe_apply_patch


class ToolsManager:
    def __init__(self, root: str | Path = ".", *, memory_service: MultiAgentMemoryService | None = None) -> None:
        self.root = Path(root).resolve()
        self.memory_service = memory_service

    def execute_job(self, job: QueueJob) -> ToolResult:
        try:
            cached = self._cached_tool_result(job)
            if cached is not None:
                return cached
            if job.job_type == QueueJobType.GIT_STATUS:
                return self._record(job, self._shell(job, "git status --short"))
            if job.job_type == QueueJobType.GIT_DIFF:
                return self._record(job, self._shell(job, "git diff"))
            if job.job_type in {QueueJobType.SHELL, QueueJobType.RUN_TESTS, QueueJobType.RUN_LINT}:
                return self._record(job, self._shell(job, str(job.payload.get("command", ""))))
            if job.job_type == QueueJobType.REPO_READ:
                raw_path = str(job.payload.get("path", ""))
                if self.memory_service is not None:
                    content, record, cache_hit = self.memory_service.read_file_with_memory(
                        file_path=raw_path,
                        task_id=job.task_id,
                        agent_id=job.requested_by_agent_id,
                    )
                    result = ToolResult(
                        new_message_id(),
                        job.task_id,
                        True,
                        {
                            "content": content,
                            "path": record.file_path,
                            "source": "memory" if cache_hit else "tool",
                            "cache_hit": cache_hit,
                            "content_hash": record.content_hash,
                        },
                    )
                    return self._record(job, result)
                path = self._resolve_path(raw_path)
                return self._record(job, ToolResult(new_message_id(), job.task_id, True, {"content": path.read_text(encoding="utf-8"), "path": str(path)}))
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
                    return self._record(job, ToolResult(new_message_id(), job.task_id, ok, {"ok": ok, "files": files}, None if ok else "one or more batch reads failed"))
                result = repo_batch_read(self.root, files=[str(item) for item in paths])
                ok = bool(result.get("ok"))
                return self._record(job, ToolResult(new_message_id(), job.task_id, ok, result, None if ok else "one or more batch reads failed"))
            if job.job_type == QueueJobType.APPLY_PATCH:
                patch = str(job.payload.get("patch", ""))
                result = safe_apply_patch(repo_root=self.root, patch=patch, check_only=False)
                ok = bool(result.get("ok"))
                error = None if ok else str(result.get("error") or result.get("message") or "patch failed")
                if not ok and result.get("error_code") == "patch_context_not_found":
                    error = "patch_context_not_found; reread target file before rebuilding patch"
                return self._record(job, ToolResult(new_message_id(), job.task_id, ok, result, error))
            if job.job_type == QueueJobType.REPO_SEARCH:
                query = str(job.payload.get("query", ""))
                result = subprocess.run(["rg", "-n", query, str(self.root)], cwd=self.root, text=True, capture_output=True, timeout=30)
                return self._record(job, ToolResult(new_message_id(), job.task_id, result.returncode in {0, 1}, {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}))
            return self._record(job, ToolResult(new_message_id(), job.task_id, False, error=f"unsupported tool job: {job.job_type.value}"))
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

    def _cached_tool_result(self, job: QueueJob) -> ToolResult | None:
        if self.memory_service is None:
            return None
        record = self.memory_service.get_reusable_tool_result(tool_name=job.job_type.value, args=job.payload)
        if record is None:
            return None
        return ToolResult(
            new_message_id(),
            job.task_id,
            record.status == "ok",
            {**record.result, "source": "memory", "cache_hit": True},
            None if record.status == "ok" else record.result_summary,
        )

    def _record(self, job: QueueJob, result: ToolResult) -> ToolResult:
        if self.memory_service is not None:
            self.memory_service.record_tool_execution(
                tool_name=job.job_type.value,
                args=job.payload,
                task_id=job.task_id,
                agent_id=job.requested_by_agent_id,
                status="ok" if result.ok else "error",
                result_summary="ok" if result.ok else str(result.error or "failed"),
                result=result.result,
            )
        return result
