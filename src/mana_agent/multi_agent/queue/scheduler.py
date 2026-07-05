from __future__ import annotations

from mana_agent.multi_agent.core.types import QueueJob, QueueJobStatus


def next_job(jobs: list[QueueJob]) -> QueueJob | None:
    pending = [job for job in jobs if job.status in {QueueJobStatus.PENDING, QueueJobStatus.QUEUED}]
    if not pending:
        return None
    return sorted(pending, key=lambda item: (item.priority, item.created_at.isoformat()))[0]
