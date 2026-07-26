"""Ordered, bounded Fleet event contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from .models import StrictModel, utc_now

EVENT_TYPES = frozenset({
    "fleet.run.created", "fleet.selection.requested", "fleet.selection.decided",
    "fleet.worker.selected", "fleet.worker.rejected", "fleet.worker.capabilities_changed",
    "fleet.job.queued", "fleet.job.assigned", "fleet.workspace.provisioning",
    "fleet.workspace.ready", "fleet.repository.synced", "fleet.command.started",
    "fleet.command.output", "fleet.command.completed", "fleet.artifact.collected",
    "fleet.job.failed", "fleet.job.completed", "fleet.job.cancelled",
    "fleet.worker.disconnected", "fleet.comparison.completed",
    "fleet.cleanup.completed", "fleet.cleanup.failed",
})


class FleetEvent(StrictModel):
    schema_version: int = 1
    sequence: int = Field(ge=1)
    kind: str
    fleet_run_id: str = ""
    job_id: str = ""
    task_id: str = ""
    session_id: str = ""
    workspace_id: str = ""
    repository_id: str = ""
    worker_id: str = ""
    execution_provider: str = ""
    timestamp: datetime = Field(default_factory=utc_now)
    data: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: object) -> None:
        _ = __context
        if self.kind not in EVENT_TYPES:
            raise ValueError(f"unknown fleet event type: {self.kind}")
