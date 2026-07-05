from __future__ import annotations

import subprocess
from pathlib import Path

from mana_agent.multi_agent.core.ids import new_message_id
from mana_agent.multi_agent.core.types import QueueJob, QueueJobType, ToolResult
from mana_agent.multi_agent.tools.permissions import assert_shell_allowed
from mana_agent.tools import repo_batch_read, safe_apply_patch


class ToolsManager:
    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()

    def execute_job(self, job: QueueJob) -> ToolResult:
        try:
            if job.job_type == QueueJobType.GIT_STATUS:
                return self._shell(job, "git status --short")
            if job.job_type == QueueJobType.GIT_DIFF:
                return self._shell(job, "git diff")
            if job.job_type in {QueueJobType.SHELL, QueueJobType.RUN_TESTS, QueueJobType.RUN_LINT}:
                return self._shell(job, str(job.payload.get("command", "")))
            if job.job_type == QueueJobType.REPO_READ:
                path = self._resolve_path(str(job.payload.get("path", "")))
                return ToolResult(new_message_id(), job.task_id, True, {"content": path.read_text(encoding="utf-8"), "path": str(path)})
            if job.job_type == QueueJobType.REPO_BATCH_READ:
                paths = job.payload.get("files") or job.payload.get("paths") or []
                result = repo_batch_read(self.root, files=[str(item) for item in paths])
                ok = bool(result.get("ok"))
                return ToolResult(new_message_id(), job.task_id, ok, result, None if ok else "one or more batch reads failed")
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
                result = subprocess.run(["rg", "-n", query, str(self.root)], cwd=self.root, text=True, capture_output=True, timeout=30)
                return ToolResult(new_message_id(), job.task_id, result.returncode in {0, 1}, {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode})
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
