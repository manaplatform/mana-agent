from __future__ import annotations

from pathlib import Path
from typing import Any

from mana_agent.multi_agent.core.ids import new_queue_job_id
from mana_agent.multi_agent.core.types import QueueJob, QueueJobStatus, QueueJobType, utc_now
from mana_agent.multi_agent.memory.service import MultiAgentMemoryService, normalize_file_path, stable_hash
from mana_agent.multi_agent.queue.locks import LockTable
from mana_agent.multi_agent.queue.scheduler import next_job
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.multi_agent.tools.tool_manager import ToolsManager


class QueueManager:
    def __init__(
        self,
        root: str | Path = ".",
        *,
        taskboard: TaskBoard | None = None,
        tools_manager: ToolsManager | None = None,
        memory_service: MultiAgentMemoryService | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.memory_service = memory_service
        self.taskboard = taskboard or TaskBoard(root, memory_service=memory_service)
        self.tools_manager = tools_manager or ToolsManager(root, memory_service=memory_service)
        self.jobs: dict[str, QueueJob] = {}
        self.locks = LockTable()

    def enqueue(self, job: QueueJob | None = None, **kwargs: Any) -> QueueJob:
        if job is None:
            if not kwargs.get("approved_by_agent_id"):
                kwargs["approved_by_agent_id"] = kwargs.get("requested_by_agent_id")
            if not kwargs.get("purpose"):
                kwargs["purpose"] = f"Run {kwargs.get('job_type', 'tool')} for task {kwargs.get('task_id')}"
            kwargs.setdefault("fingerprint", self._fingerprint_from_kwargs(kwargs))
            kwargs.setdefault("related_files", self._related_files_from_kwargs(kwargs))
            if self.memory_service is not None and not kwargs.get("memory_bundle_id"):
                bundle = self.memory_service.build_bundle(
                    agent_id=str(kwargs.get("requested_by_agent_id") or "agent_queue"),
                    agent_role="queue",
                    task_id=str(kwargs.get("task_id") or ""),
                    target_files=list(kwargs.get("related_files") or []),
                )
                kwargs["memory_bundle_id"] = bundle.bundle_id
            job = QueueJob(job_id=new_queue_job_id(), **kwargs)
        elif not job.fingerprint:
            job.fingerprint = self._fingerprint_for_job(job)
        if self.memory_service is not None:
            accepted, existing_id = self.memory_service.register_queue_item(
                queue_item_id=job.job_id,
                fingerprint=job.fingerprint,
            )
            if not accepted:
                job.status = QueueJobStatus.CANCELLED
                job.duplicate_of = existing_id
                job.result_summary = f"skipped duplicate_of:{existing_id}"
                self.jobs[job.job_id] = job
                self.taskboard.add_queue_job(job.task_id, job.job_id)
                self.taskboard.add_evidence(job.task_id, f"Queue job {job.job_id} skipped as duplicate_of:{existing_id}")
                self._bump_memory_status(job.task_id, duplicate_of=existing_id)
                return job
        self.jobs[job.job_id] = job
        self.taskboard.add_queue_job(job.task_id, job.job_id)
        self.taskboard.add_evidence(job.task_id, f"Queue job {job.job_id} queued for {job.job_type.value}: {job.purpose}")
        return job

    def claim_next(self, agent_id: str) -> QueueJob | None:
        job = next_job(list(self.jobs.values()))
        if job is None:
            return None
        job.status = QueueJobStatus.CLAIMED
        job.updated_at = utc_now()
        job.payload = {**job.payload, "claimed_by_agent_id": agent_id}
        return job

    def run_next(self) -> QueueJob | None:
        job = self.claim_next("agent_tool")
        if job is None:
            return None
        lock_key = job.lock_key or ("repo" if job.requires_write_lock else f"job:{job.job_id}")
        lock = self.locks.lock_for(lock_key)
        with lock:
            job.status = QueueJobStatus.RUNNING
            job.updated_at = utc_now()
            result = self.tools_manager.execute_job(job)
            job.result = result.result
            job.error = result.error
            job.result_summary = "ok" if result.ok else str(result.error or "failed")
            job.status = QueueJobStatus.DONE if result.ok else QueueJobStatus.FAILED
            job.updated_at = utc_now()
            if result.result.get("cache_hit"):
                self._bump_memory_status(job.task_id, cache_hit=True, file_read=job.job_type in {QueueJobType.REPO_READ, QueueJobType.REPO_BATCH_READ})
        if job.job_type in {QueueJobType.APPLY_PATCH} and job.result:
            changed = job.result.get("changed_files") or job.result.get("files_changed") or []
            if isinstance(changed, list):
                self.taskboard.add_files_touched(job.task_id, [str(item) for item in changed])
        if self.memory_service is not None:
            self.memory_service.update_task(job.task_id, status=job.status.value, result_summary=job.result_summary or "")
        return job

    def run_until_idle(self, max_jobs: int | None = None) -> list[QueueJob]:
        ran: list[QueueJob] = []
        while max_jobs is None or len(ran) < max_jobs:
            job = self.run_next()
            if job is None:
                break
            ran.append(job)
        return ran

    def get_job(self, job_id: str) -> QueueJob:
        return self.jobs[job_id]

    def cancel_job(self, job_id: str) -> None:
        job = self.jobs[job_id]
        job.status = QueueJobStatus.CANCELLED
        job.updated_at = utc_now()

    def jobs_for_task(self, task_id: str) -> list[QueueJob]:
        return [job for job in self.jobs.values() if job.task_id == task_id]

    def _fingerprint_from_kwargs(self, kwargs: dict[str, Any]) -> str:
        job_type = kwargs.get("job_type")
        tool = job_type.value if hasattr(job_type, "value") else str(job_type or "")
        return stable_hash(
            {
                "task_id": kwargs.get("task_id"),
                "tool": tool,
                "payload": kwargs.get("payload") or {},
                "related_files": kwargs.get("related_files") or self._related_files_from_kwargs(kwargs),
            }
        )

    def _fingerprint_for_job(self, job: QueueJob) -> str:
        return stable_hash(
            {
                "task_id": job.task_id,
                "tool": job.job_type.value,
                "payload": job.payload,
                "related_files": job.related_files or self._related_files_for_payload(job.job_type, job.payload),
            }
        )

    def _related_files_from_kwargs(self, kwargs: dict[str, Any]) -> list[str]:
        job_type = kwargs.get("job_type")
        payload = kwargs.get("payload") or {}
        return self._related_files_for_payload(job_type, payload)

    def _related_files_for_payload(self, job_type: Any, payload: dict[str, Any]) -> list[str]:
        kind = job_type.value if hasattr(job_type, "value") else str(job_type or "")
        if kind == QueueJobType.REPO_READ.value and payload.get("path"):
            return [normalize_file_path(payload.get("path"), root=self.root)]
        if kind == QueueJobType.REPO_BATCH_READ.value:
            return [normalize_file_path(item, root=self.root) for item in payload.get("files") or payload.get("paths") or []]
        if kind == QueueJobType.APPLY_PATCH.value:
            return [normalize_file_path(payload.get("path"), root=self.root)] if payload.get("path") else []
        return []

    def _bump_memory_status(
        self,
        task_id: str,
        *,
        cache_hit: bool = False,
        file_read: bool = False,
        duplicate_of: str | None = None,
    ) -> None:
        try:
            task = self.taskboard.get_task(task_id)
        except KeyError:
            return
        status = dict(task.memory_status or {})
        if cache_hit:
            status["cache_hits"] = int(status.get("cache_hits") or 0) + 1
        if file_read:
            status["file_reads_reused"] = int(status.get("file_reads_reused") or 0) + 1
        if duplicate_of:
            status["duplicate_of"] = duplicate_of
        status["last_memory_check_at"] = utc_now()
        task.memory_status = status
        task.updated_at = utc_now()
        self.taskboard.save()
