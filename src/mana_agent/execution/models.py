"""Provider-neutral, persisted execution fabric contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SandboxState(str, Enum):
    REQUESTED = "requested"
    PROVISIONING = "provisioning"
    READY = "ready"
    RUNNING = "running"
    SUSPENDING = "suspending"
    SUSPENDED = "suspended"
    RESUMING = "resuming"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    CLEANING = "cleaning"
    CLEANED = "cleaned"
    FAILED = "failed"


class EnforcementStrength(str, Enum):
    NONE = "none"
    BEST_EFFORT = "best-effort"
    ENFORCED = "enforced"

    def satisfies(self, required: "EnforcementStrength") -> bool:
        order = {self.NONE: 0, self.BEST_EFFORT: 1, self.ENFORCED: 2}
        return order[self] >= order[required]


class NetworkMode(str, Enum):
    DENY_ALL = "deny-all"
    RESTRICTED_EGRESS = "restricted-egress"
    ALLOWLIST = "allowlist"
    UNRESTRICTED = "unrestricted"


class ResourceLimits(StrictModel):
    cpu_cores: float | None = Field(default=None, gt=0)
    memory_bytes: int | None = Field(default=None, gt=0)
    disk_bytes: int | None = Field(default=None, gt=0)
    pid_limit: int | None = Field(default=None, gt=0)
    gpu_count: int = Field(default=0, ge=0)
    gpu_type: str | None = None


class NetworkPolicy(StrictModel):
    mode: NetworkMode = NetworkMode.UNRESTRICTED
    domains: list[str] = Field(default_factory=list)
    ip_ranges: list[str] = Field(default_factory=list)
    ports: list[int] = Field(default_factory=list)
    protocols: list[Literal["tcp", "udp"]] = Field(default_factory=list)
    dns_enabled: bool = True
    inbound_allowed: bool = False
    proxy_required: bool = False
    required_enforcement: EnforcementStrength = EnforcementStrength.NONE


class SecretInjection(StrictModel):
    reference: str = Field(min_length=1)
    target: str = Field(min_length=1)
    mode: Literal["environment", "file", "provider-native"] = "environment"
    required: bool = True


class LocalProcessOptions(StrictModel):
    kind: Literal["local-process"] = "local-process"


class DockerSandboxOptions(StrictModel):
    kind: Literal["local-docker"] = "local-docker"
    managed_network: str | None = None


class SSHSandboxOptions(StrictModel):
    kind: Literal["remote-ssh"] = "remote-ssh"
    host_pool: str = "default"


class KubernetesSandboxOptions(StrictModel):
    kind: Literal["kubernetes"] = "kubernetes"
    namespace: str | None = None
    service_account: str | None = None
    volume_class: str | None = None


class ModalSandboxOptions(StrictModel):
    kind: Literal["modal"] = "modal"
    app_name: str | None = None
    volume_name: str | None = None


class HTTPSandboxOptions(StrictModel):
    kind: Literal["custom-http-runtime"] = "custom-http-runtime"
    runtime_profile: str | None = None


ProviderSandboxOptions = (
    LocalProcessOptions | DockerSandboxOptions | SSHSandboxOptions |
    KubernetesSandboxOptions | ModalSandboxOptions | HTTPSandboxOptions
)


class SandboxCapabilities(StrictModel):
    snapshots: bool = False
    native_suspend_resume: bool = False
    emulated_suspend_resume: bool = False
    resource_isolation: EnforcementStrength = EnforcementStrength.NONE
    gpu_execution: bool = False
    network_isolation: EnforcementStrength = EnforcementStrength.NONE
    secret_files: bool = False
    secret_environment_variables: bool = False
    persistent_volumes: bool = False
    artifact_streaming: bool = False
    parallel_execution: bool = False


class SandboxSpec(StrictModel):
    provider_override: str | None = None
    repository_source: Path
    workspace_mount_path: str = "/workspace"
    base_image: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    secrets: list[SecretInjection] = Field(default_factory=list)
    resources: ResourceLimits = Field(default_factory=ResourceLimits)
    execution_timeout_seconds: int = Field(default=120, gt=0)
    idle_timeout_seconds: int = Field(default=900, gt=0)
    max_lifetime_seconds: int = Field(default=7200, gt=0)
    network: NetworkPolicy = Field(default_factory=NetworkPolicy)
    artifact_paths: list[str] = Field(default_factory=list)
    snapshot_source: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    task_id: str = ""
    session_id: str = ""
    workspace_id: str = ""
    cleanup_policy: Literal["always", "on-success", "retain"] = "always"
    read_only_root: bool = False
    provider_options: ProviderSandboxOptions | None = Field(default=None, discriminator="kind")

    @field_validator("repository_source")
    @classmethod
    def source_exists(cls, value: Path) -> Path:
        resolved = value.expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError("repository_source must be an existing directory")
        return resolved


class SandboxHandle(StrictModel):
    sandbox_id: str
    provider: str
    external_id: str = ""
    state: SandboxState = SandboxState.REQUESTED
    task_id: str = ""
    session_id: str = ""
    workspace_id: str = ""
    workspace_path: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_heartbeat: datetime = Field(default_factory=utc_now)
    lease_expires_at: datetime | None = None
    snapshot_refs: list[str] = Field(default_factory=list)
    cleanup_complete: bool = False
    failure: str = ""
    cleanup_failure: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionRequest(StrictModel):
    argv: list[str] = Field(min_length=1)
    cwd: str = "."
    environment: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=120, gt=0)
    stdin: str | None = None
    capture_limit_bytes: int = Field(default=10_485_760, gt=0)


class ExecutionResult(StrictModel):
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    started_at: datetime
    completed_at: datetime
    timed_out: bool = False
    cancelled: bool = False
    provider: str
    sandbox_id: str
    resource_usage: dict[str, float] = Field(default_factory=dict)


class ArtifactRequest(StrictModel):
    paths: list[str]
    destination: Path | None = None
    compress: bool = False
    max_file_bytes: int = Field(default=100 * 1024 * 1024, gt=0)
    max_total_bytes: int = Field(default=500 * 1024 * 1024, gt=0)
    missing_ok: bool = False


class ArtifactResult(StrictModel):
    reference: str
    source_path: str
    local_path: Path | None = None
    size_bytes: int
    sha256: str
    mime_type: str = "application/octet-stream"
    compressed: bool = False


class SnapshotRequest(StrictModel):
    name: str = ""
    include_paths: list[str] = Field(default_factory=lambda: ["."])


class SnapshotRef(StrictModel):
    snapshot_id: str
    provider: str
    location: str
    checksum: str
    schema_version: int = 1
    created_at: datetime = Field(default_factory=utc_now)
    image_identity: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderHealth(StrictModel):
    provider: str
    available: bool
    checked_at: datetime = Field(default_factory=utc_now)
    message: str = ""


class RoutingRequest(StrictModel):
    decision_id: str = Field(min_length=1)
    explicit_provider: str | None = None
    trust_level: Literal["trusted", "untrusted"]
    risk_level: Literal["low", "medium", "high"]
    resources: ResourceLimits = Field(default_factory=ResourceLimits)
    network: NetworkPolicy = Field(default_factory=NetworkPolicy)
    required_capabilities: set[str] = Field(default_factory=set)
    required_resource_enforcement: EnforcementStrength = EnforcementStrength.NONE
    expected_duration_seconds: int = Field(default=120, gt=0)
    parallelism: int = Field(default=1, gt=0)
    secret_sensitivity: Literal["none", "standard", "high"] = "none"
    organization_provider: str | None = None


class RoutingDecision(StrictModel):
    decision_id: str
    selected_provider: str
    requirements_considered: list[str]
    policy_rule: str
    rejected_providers: dict[str, str] = Field(default_factory=dict)
    explicit: bool = False
    decided_at: datetime = Field(default_factory=utc_now)


class SandboxExecutionContext(StrictModel):
    handle: SandboxHandle
    spec: SandboxSpec
    routing: RoutingDecision
