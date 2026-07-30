"""Typed contracts for enrolled server management."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


AuthMethod = Literal["ssh_key", "ssh_agent", "password", "token"]
SudoMode = Literal["none", "passwordless", "credential"]
ServerMode = Literal["inspect_only", "managed_admin", "trusted_admin"]


class ServerDefinition(StrictModel):
    server_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    name: str = Field(min_length=1)
    host: str = Field(min_length=1)
    port: int = Field(default=22, ge=1, le=65535)
    username: str = Field(min_length=1)
    auth_method: AuthMethod
    credential_ref: str | None = None
    shell: str = "zsh"
    sudo_mode: SudoMode = "none"
    operating_system: str | None = None
    architecture: str | None = None
    tags: list[str] = Field(default_factory=list)
    provider: str = "ssh"
    provider_server_id: str | None = None
    allowed_capabilities: set[str] = Field(default_factory=lambda: {"inspect"})
    mode: ServerMode = "inspect_only"
    host_key_fingerprint: str = Field(min_length=1)
    known_hosts_file: str
    jump_host: str | None = None
    agent_forwarding: bool = False
    connect_timeout_seconds: int = Field(default=15, gt=0, le=600)
    keepalive_seconds: int = Field(default=30, ge=0, le=600)
    max_concurrent_operations: int = Field(default=1, ge=1, le=16)
    created_at: datetime = Field(default_factory=utc_now)
    last_connected_at: datetime | None = None

    @model_validator(mode="after")
    def validate_authentication(self) -> "ServerDefinition":
        if self.auth_method in {"ssh_key", "password", "token"} and not self.credential_ref:
            raise ValueError(f"{self.auth_method} authentication requires credential_ref")
        if self.agent_forwarding and self.auth_method != "ssh_agent":
            raise ValueError("agent forwarding requires ssh_agent authentication")
        if not self.known_hosts_file:
            raise ValueError("known_hosts_file is required; host-key verification cannot be disabled")
        return self


class RemoteCommandResult(StrictModel):
    server_id: str
    command_id: str
    command: str
    cwd: str | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    started_at: datetime
    completed_at: datetime | None = None
    timed_out: bool = False
    cancelled: bool = False
    changed_system: bool = False


class ServerActionKind(str, Enum):
    INSPECT = "inspect"
    SHELL = "shell"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    PACKAGE = "package"
    SERVICE = "service"
    PROCESS = "process"
    NETWORK = "network"
    FIREWALL = "firewall"
    USER = "user"
    DATABASE = "database"
    CONTAINER = "container"
    DEPLOYMENT = "deployment"
    BACKUP = "backup"
    RESTORE = "restore"
    REBOOT = "reboot"
    SHUTDOWN = "shutdown"
    PROVISION = "provision"
    DELETE_SERVER = "delete_server"


READ_ONLY_ACTIONS = frozenset({
    ServerActionKind.INSPECT,
    ServerActionKind.FILE_READ,
})

DESTRUCTIVE_ACTIONS = frozenset({
    ServerActionKind.RESTORE,
    ServerActionKind.REBOOT,
    ServerActionKind.SHUTDOWN,
    ServerActionKind.DELETE_SERVER,
})


class ServerActionDecision(StrictModel):
    """Model-produced decision validated before any server-side action."""

    decision_id: str = Field(min_length=1)
    server_id: str = Field(min_length=1)
    action: ServerActionKind
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    required_capability: str = Field(min_length=1)
    read_only: bool
    consequential: bool
    destructive: bool = False
    affected_resources: list[str] = Field(default_factory=list)
    recovery_plan: str | None = None
    verification_commands: list[list[str]] = Field(default_factory=list)
    safe_to_continue: bool
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_risk_claims(self) -> "ServerActionDecision":
        if self.action in READ_ONLY_ACTIONS and not self.read_only:
            raise ValueError("read-only actions must be marked read_only")
        if self.action in DESTRUCTIVE_ACTIONS and not self.destructive:
            raise ValueError("destructive action is not classified as destructive")
        if self.destructive and not self.consequential:
            raise ValueError("destructive actions are consequential")
        if not self.read_only and not self.affected_resources:
            raise ValueError("mutating decisions require affected_resources")
        if self.destructive and not self.recovery_plan:
            raise ValueError("destructive decisions require a recovery_plan")
        return self


class ServerApproval(StrictModel):
    approval_id: str
    decision_id: str
    server_id: str
    exact_action_key: str
    approved_at: datetime = Field(default_factory=utc_now)
    approved_by: str = "user"


class ServerHealthReport(StrictModel):
    server_id: str
    collected_at: datetime = Field(default_factory=utc_now)
    operating_system: str = ""
    architecture: str = ""
    load_average: str = ""
    memory: str = ""
    disks: str = ""
    failed_services: str = ""
    listening_ports: str = ""
    reboot_required: bool | None = None
    evidence: list[RemoteCommandResult] = Field(default_factory=list)


class ServerPlanStep(StrictModel):
    step_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    affected_resources: list[str] = Field(default_factory=list)
    verification_commands: list[list[str]] = Field(default_factory=list)
    rollback: dict[str, Any] | None = None


class ServerPlan(StrictModel):
    plan_id: str
    name: str
    server_ids: list[str] = Field(min_length=1)
    steps: list[ServerPlanStep] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    approved_action_keys: set[str] = Field(default_factory=set)


class Region(StrictModel):
    id: str
    name: str


class ServerSize(StrictModel):
    id: str
    name: str
    vcpus: int | None = None
    memory_mb: int | None = None
    hourly_cost: float | None = None
    currency: str | None = None


class ServerImage(StrictModel):
    id: str
    name: str
    operating_system: str = ""


class CreateServerRequest(StrictModel):
    name: str
    region: str
    size: str
    image: str
    ssh_key_ref: str
    user_data: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    private_network_ids: list[str] = Field(default_factory=list)
    firewall_ids: list[str] = Field(default_factory=list)
    volume_ids: list[str] = Field(default_factory=list)
    enable_ipv4: bool = True
    enable_ipv6: bool = True
    cost_approval_id: str | None = None

    @field_validator("ssh_key_ref")
    @classmethod
    def require_secret_reference(cls, value: str) -> str:
        if not value.startswith("secret://"):
            raise ValueError("ssh_key_ref must be a secret reference")
        return value


class ProvisionedServer(StrictModel):
    provider_server_id: str
    name: str
    status: str
    ipv4: str | None = None
    ipv6: str | None = None
    estimated_hourly_cost: float | None = None


class ProviderServer(ProvisionedServer):
    region: str = ""
    size: str = ""
    image: str = ""
