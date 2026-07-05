from __future__ import annotations

from pathlib import Path
from typing import Any

from mana_agent.multi_agent.core.ids import new_queue_job_id
from mana_agent.multi_agent.core.types import QueueJob, QueueJobStatus, QueueJobType, utc_now
from mana_agent.multi_agent.queue.locks import LockTable
from mana_agent.multi_agent.queue.scheduler import next_job
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.multi_agent.tools.tool_manager import ToolsManager


class QueueManager:
    def __init__(self, root: str | Path = ".", *, taskboard: TaskBoard | None = None, tools_manager: ToolsManager | None = None) -> None:
        self.root = Path(root).resolve()
        self.taskboard = taskboard or TaskBoard(root)
        self.tools_manager = tools_manager or ToolsManager(root)
        self.jobs: dict[str, QueueJob] = {}
        self.locks = LockTable()

    def enqueue(self, job: QueueJob | None = None, **kwargs: Any) -> QueueJob:
        if job is None:
            if not kwargs.get("approved_by_agent_id"):
                kwargs["approved_by_agent_id"] = kwargs.get("requested_by_agent_id")
            if not kwargs.get("purpose"):
                kwargs["purpose"] = f"Run {kwargs.get('job_type', 'tool')} for task {kwargs.get('task_id')}"
            job = QueueJob(job_id=new_queue_job_id(), **kwargs)
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
        if job.job_type in {QueueJobType.APPLY_PATCH} and job.result:
            changed = job.result.get("changed_files") or job.result.get("files_changed") or []
            if isinstance(changed, list):
                self.taskboard.add_files_touched(job.task_id, [str(item) for item in changed])
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
