"""Exact Fleet permission scopes and action binding."""

from __future__ import annotations

import hashlib
import json

from pydantic import Field

from .models import StrictModel

FLEET_PERMISSION_SCOPES = frozenset({
    "fleet.workers.read",
    "fleet.workers.manage",
    "fleet.verify.read",
    "fleet.verify.execute",
    "fleet.verify.cancel",
    "fleet.artifacts.read",
    "fleet.workspace.retain",
    "fleet.worker.revoke",
    "fleet.remote.write",
})


class FleetPermissionRequest(StrictModel):
    permission_request_id: str = Field(min_length=1)
    scope: str
    fleet_run_id: str = ""
    job_id: str = ""
    repository_id: str
    repository_commit: str
    worker_ids: tuple[str, ...]
    commands: tuple[tuple[str, ...], ...]

    def model_post_init(self, __context: object) -> None:
        _ = __context
        if self.scope not in FLEET_PERMISSION_SCOPES:
            raise ValueError(f"unknown Fleet permission scope: {self.scope}")

    @property
    def exact_action_key(self) -> str:
        payload = self.model_dump(mode="json", exclude={"permission_request_id"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class FleetPermissionGrant(StrictModel):
    permission_request_id: str
    exact_action_key: str
    scope: str

    def authorizes(self, request: FleetPermissionRequest) -> bool:
        return (
            self.permission_request_id == request.permission_request_id
            and self.scope == request.scope
            and self.exact_action_key == request.exact_action_key
        )
