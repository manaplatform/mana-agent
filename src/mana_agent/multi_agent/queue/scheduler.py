from __future__ import annotations

from mana_agent.multi_agent.core.types import QueueJob, QueueJobStatus


def next_job(jobs: list[QueueJob], worker_id: str | None = None) -> QueueJob | None:
    by_id = {job.job_id: job for job in jobs}
    pending = [job for job in jobs if job.status in {QueueJobStatus.PENDING, QueueJobStatus.QUEUED}
               and all(dep in by_id and by_id[dep].status == QueueJobStatus.DONE for dep in job.depends_on)]
    if worker_id is not None:
        pending = [job for job in pending if job.assigned_worker_agent_id == worker_id]
    if not pending:
        return None
    return sorted(pending, key=lambda item: (item.priority, item.created_at.isoformat()))[0]
