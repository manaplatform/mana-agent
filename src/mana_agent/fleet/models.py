"""Strict persisted contracts for distributed verification."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def fleet_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_json_value(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {str(key): _canonical_json_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        items = [_canonical_json_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, datetime):
        encoded = value.isoformat()
        return encoded[:-6] + "Z" if encoded.endswith("+00:00") else encoded
    if isinstance(value, Enum):
        return _canonical_json_value(value.value)
    return value


def canonical_model_payload(model: BaseModel, *, exclude: set[str] | None = None) -> bytes:
    """Serialize a model deterministically, including all unordered containers."""
    payload = model.model_dump(mode="python", exclude=exclude)
    return json.dumps(
        _canonical_json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


class WorkerStatus(str, Enum):
    CONNECTED = "connected"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    BUSY = "busy"
    DRAINING = "draining"
    REVOKED = "revoked"
    UNKNOWN = "unknown"


class FleetJobState(str, Enum):
    QUEUED = "queued"
    ASSIGNED = "assigned"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    COLLECTING = "collecting"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    WORKER_DISCONNECTED = "worker_disconnected"
    REVALIDATION_REQUIRED = "revalidation_required"


class WorkspaceState(str, Enum):
    REQUESTED = "requested"
    PROVISIONING = "provisioning"
    READY = "ready"
    SYNCING = "syncing"
    RUNNING = "running"
    COLLECTING = "collecting"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    TERMINATING = "terminating"
    CLEANING = "cleaning"
    CLEANED = "cleaned"
    RETAINED = "retained"


class FailureClassification(str, Enum):
    TEST_FAILURE = "test_failure"
    SETUP_FAILURE = "setup_failure"
    PROVIDER_FAILURE = "provider_failure"
    WORKER_DISCONNECT = "worker_disconnect"
    PERMISSION_DENIAL = "permission_denial"
    CAPABILITY_MISMATCH = "capability_mismatch"
    TIMEOUT = "timeout"
    REPOSITORY_TRANSFER_FAILURE = "repository_transfer_failure"
    ARTIFACT_COLLECTION_FAILURE = "artifact_collection_failure"
    CLEANUP_FAILURE = "cleanup_failure"
    MODEL_ROUTING_FAILURE = "model_or_routing_failure"


class FleetOutcome(str, Enum):
    FULLY_VERIFIED = "fully_verified"
    PARTIALLY_VERIFIED = "partially_verified"
    FAILED_VERIFICATION = "failed_verification"
    INFRASTRUCTURE_INCOMPLETE = "infrastructure_incomplete"
    CANCELLED = "cancelled"


class WorkerIdentity(StrictModel):
    worker_id: str = Field(min_length=1, max_length=200)
    identity_fingerprint: str = Field(min_length=1, max_length=256)
    authenticated: bool = True
    credential_status: Literal["valid", "invalid", "revoked", "unknown"] = "unknown"


class WorkerLabels(StrictModel):
    values: frozenset[str] = Field(default_factory=frozenset, max_length=64)

    @field_validator("values")
    @classmethod
    def bounded_labels(cls, value: frozenset[str]) -> frozenset[str]:
        for item in value:
            if not item or len(item) > 64:
                raise ValueError("worker labels must contain 1-64 characters")
        return value


class GPUCapability(StrictModel):
    count: int = Field(ge=0, le=64)
    vendor: str = Field(default="", max_length=80)
    model: str = Field(default="", max_length=160)


class WorkerCapabilities(StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    worker_id: str = Field(min_length=1, max_length=200)
    platform: Literal["linux", "windows", "macos", "unknown"]
    platform_release: str = Field(default="", max_length=160)
    architecture: str = Field(min_length=1, max_length=80)
    python_versions: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    node_versions: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    available_tools: frozenset[str] = Field(default_factory=frozenset, max_length=128)
    docker: bool | None = None
    gpu: GPUCapability | None = None
    max_concurrency: int = Field(default=1, ge=1, le=64)
    workspace_capacity_bytes: int | None = Field(default=None, ge=0)
    labels: WorkerLabels = Field(default_factory=WorkerLabels)
    workspace_backends: frozenset[str] = Field(default_factory=frozenset, max_length=32)
    execution_providers: frozenset[str] = Field(default_factory=frozenset, max_length=32)
    last_probe_at: datetime

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_model_payload(self, exclude={"last_probe_at"})
        ).hexdigest()


class WorkerHealth(StrictModel):
    status: WorkerStatus = WorkerStatus.UNKNOWN
    last_heartbeat: datetime | None = None
    last_capability_update: datetime | None = None
    active_job_count: int = Field(default=0, ge=0)
    concurrency_limit: int = Field(default=1, ge=1, le=64)
    recent_failures: tuple[str, ...] = Field(default_factory=tuple, max_length=20)
    transport_status: str = Field(default="unknown", max_length=80)
    identity_status: str = Field(default="unknown", max_length=80)
    last_successful_job: datetime | None = None


class FleetWorker(StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    identity: WorkerIdentity
    display_name: str = Field(default="", max_length=160)
    capabilities: WorkerCapabilities
    capability_fingerprint: str
    health: WorkerHealth = Field(default_factory=WorkerHealth)
    registered_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def identities_match(self) -> "FleetWorker":
        if self.identity.worker_id != self.capabilities.worker_id:
            raise ValueError("worker and capability identities do not match")
        if self.capability_fingerprint != self.capabilities.fingerprint:
            raise ValueError("capability fingerprint does not match inventory")
        return self

    @property
    def worker_id(self) -> str:
        return self.identity.worker_id


class RuntimeRequirements(StrictModel):
    python: frozenset[str] = Field(default_factory=frozenset)
    node: frozenset[str] = Field(default_factory=frozenset)


class FleetSelectionRequest(StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    decision_id: str = Field(min_length=1)
    task_id: str = ""
    session_id: str = ""
    workspace_id: str = ""
    repository_id: str = ""
    required_platforms: frozenset[Literal["linux", "windows", "macos"]] = Field(default_factory=frozenset)
    allowed_platforms: frozenset[Literal["linux", "windows", "macos"]] = Field(default_factory=frozenset)
    required_architectures: frozenset[str] = Field(default_factory=frozenset)
    runtime: RuntimeRequirements = Field(default_factory=RuntimeRequirements)
    required_tools: frozenset[str] = Field(default_factory=frozenset)
    required_provider_capabilities: frozenset[str] = Field(default_factory=frozenset)
    network_requirements: frozenset[str] = Field(default_factory=frozenset)
    isolation_requirements: frozenset[str] = Field(default_factory=frozenset)
    artifact_requirements: tuple[str, ...] = Field(default_factory=tuple)
    minimum_workers: int = Field(default=1, ge=1, le=64)
    maximum_workers: int = Field(default=4, ge=1, le=64)
    parallelism: int = Field(default=1, ge=1, le=64)
    timeout_seconds: int = Field(default=1800, ge=1, le=86_400)
    risk_level: Literal["low", "medium", "high"] = "low"
    intent: Literal["read-only", "mutation"] = "read-only"
    budget: float | None = Field(default=None, ge=0)
    preferred_labels: frozenset[str] = Field(default_factory=frozenset)
    forbidden_labels: frozenset[str] = Field(default_factory=frozenset)

    @model_validator(mode="after")
    def coherent_limits(self) -> "FleetSelectionRequest":
        if self.minimum_workers > self.maximum_workers:
            raise ValueError("minimum_workers cannot exceed maximum_workers")
        if self.allowed_platforms and not self.required_platforms.issubset(self.allowed_platforms):
            raise ValueError("required platforms must be allowed")
        if self.preferred_labels & self.forbidden_labels:
            raise ValueError("preferred and forbidden labels must not overlap")
        return self


class CapabilityMismatch(StrictModel):
    worker_id: str
    reasons: tuple[str, ...]


class SelectedWorker(StrictModel):
    worker_id: str
    execution_provider: str
    score: float
    reasons: tuple[str, ...] = Field(default_factory=tuple)


class FleetSelectionDecision(StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    decision_id: str
    selected_workers: tuple[SelectedWorker, ...]
    rejected_workers: tuple[CapabilityMismatch, ...] = Field(default_factory=tuple)
    platform_coverage: frozenset[str] = Field(default_factory=frozenset)
    runtime_coverage: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    estimated_concurrency: int = Field(ge=0)
    estimated_cost: float | None = None
    estimated_latency_seconds: float | None = None
    verification_policy: str
    decided_at: datetime = Field(default_factory=utc_now)


class VerificationCell(StrictModel):
    platform: Literal["linux", "windows", "macos"]
    architecture: str = ""
    python_version: str = ""
    node_version: str = ""
    commands: tuple[tuple[str, ...], ...]
    artifact_paths: tuple[str, ...] = Field(default_factory=tuple)


class FleetVerificationPlan(StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    fleet_run_id: str = Field(default_factory=lambda: fleet_id("fleet_run"))
    decision: FleetSelectionDecision
    task_id: str = ""
    session_id: str = ""
    workspace_id: str = ""
    repository_id: str = ""
    repository_path: str
    repository_commit: str
    transfer_mode: Literal["git-clone", "git-bundle", "archive-upload", "existing-authorized-checkout"]
    cells: tuple[VerificationCell, ...]
    timeout_seconds: int = Field(default=1800, ge=1, le=86_400)
    retain_workspaces: bool = False
    mutation_intent: bool = False


class FleetJob(StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    fleet_run_id: str
    job_id: str = Field(default_factory=lambda: fleet_id("fleet_job"))
    task_id: str = ""
    session_id: str = ""
    workspace_id: str = ""
    repository_id: str = ""
    worker_id: str
    execution_provider: str
    cell: VerificationCell
    state: FleetJobState = FleetJobState.QUEUED
    workspace_state: WorkspaceState = WorkspaceState.REQUESTED
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    revalidated: bool = False


class ArtifactReference(StrictModel):
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    reference: str


class FleetJobResult(StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    fleet_run_id: str
    job_id: str
    task_id: str = ""
    session_id: str = ""
    workspace_id: str = ""
    repository_id: str = ""
    worker_id: str
    execution_provider: str
    state: FleetJobState
    exit_code: int | None = None
    duration_seconds: float = Field(default=0, ge=0)
    stdout: str = ""
    stderr: str = ""
    artifacts: tuple[ArtifactReference, ...] = Field(default_factory=tuple)
    tests_total: int = Field(default=0, ge=0)
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    failure_classification: FailureClassification | None = None
    cleanup_failure: str = ""
    completed_at: datetime = Field(default_factory=utc_now)


class PlatformResult(StrictModel):
    platform: str
    jobs: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    infrastructure_failures: int = Field(ge=0)


class FleetRunSummary(StrictModel):
    fleet_run_id: str
    outcome: FleetOutcome
    required_platforms: frozenset[str]
    tested_platforms: frozenset[str]
    platform_results: tuple[PlatformResult, ...]
    failures_by_classification: dict[str, int] = Field(default_factory=dict)
    completed_jobs: int = Field(ge=0)
    total_jobs: int = Field(ge=0)


class FleetRun(StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    fleet_run_id: str
    plan: FleetVerificationPlan
    jobs: tuple[FleetJob, ...]
    results: tuple[FleetJobResult, ...] = Field(default_factory=tuple)
    summary: FleetRunSummary | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
