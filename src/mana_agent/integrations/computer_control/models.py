"""Typed contracts for permission-aware desktop automation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from mana_agent.compat import StrEnum


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SupportedPlatform(StrEnum):
    MACOS = "macos"
    WINDOWS = "windows"
    LINUX = "linux"


class ExecutionRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExecutionState(StrEnum):
    PENDING = "pending"
    WAITING_PERMISSION = "waiting_permission"
    WAITING_CONFIRMATION = "waiting_confirmation"
    RUNNING = "running"
    COMPLETED = "completed"
    DENIED = "denied"
    CANCELLED = "cancelled"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class PermissionDecision(StrEnum):
    DENIED = "denied"
    ASK_EVERY_TIME = "ask"
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    ALWAYS_ALLOW = "always"
    MANAGED_BY_OS = "managed_by_os"
    UNAVAILABLE = "unavailable"


class PermissionStatus(BaseModel):
    scope: str
    decision: PermissionDecision
    granted: bool = False
    reason: str = ""
    os_managed: bool = False


class PermissionResult(PermissionStatus):
    requested: bool = True


class ComputerTarget(BaseModel):
    application_id: str | None = None
    resource_id: str | None = None
    path: str | None = None
    url: str | None = None
    display_id: str | None = None

    @field_validator("application_id", "resource_id", "display_id")
    @classmethod
    def validate_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or len(value) > 512 or any(ord(char) < 32 for char in value):
            raise ValueError("identifier contains invalid characters")
        return value


class ComputerAction(BaseModel):
    """A model-produced action decision validated before provider execution."""

    execution_id: str = Field(default_factory=lambda: f"computer-{uuid4().hex}")
    capability: str = Field(min_length=3, max_length=128)
    operation: str = Field(min_length=2, max_length=128, pattern=r"^[a-z][a-z0-9_.-]*$")
    permission_scope: str = Field(min_length=3, max_length=128)
    risk: ExecutionRisk
    target: ComputerTarget = Field(default_factory=ComputerTarget)
    arguments: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=30, ge=0.1, le=300)
    confirmation_token: str | None = None
    source_decision_id: str = Field(min_length=1, max_length=256)


class ComputerActionResult(BaseModel):
    execution_id: str
    state: ExecutionState
    capability: str
    operation: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    sensitive_content_accessed: bool = False
    error_code: str | None = None


class ApplicationDescriptor(BaseModel):
    application_id: str
    name: str
    installed: bool = True
    version: str | None = None
    executable: str | None = None
    capabilities: set[str] = Field(default_factory=set)


class ApplicationStatus(BaseModel):
    application_id: str
    running: bool
    active: bool = False


class CapabilityAvailability(BaseModel):
    name: str
    available: bool
    provider: str
    reason: str = ""
    permission_scopes: set[str] = Field(default_factory=set)
    operations: set[str] = Field(default_factory=set)


class CapabilityReport(BaseModel):
    platform: SupportedPlatform
    provider: str
    capabilities: list[CapabilityAvailability] = Field(default_factory=list)
    applications: list[ApplicationDescriptor] = Field(default_factory=list)
    headless: bool = False

    def supports(self, capability: str, operation: str | None = None) -> bool:
        return any(
            item.name == capability
            and item.available
            and (operation is None or operation in item.operations)
            for item in self.capabilities
        )


class BrowserContext(BaseModel):
    browser_id: str
    tab_id: str | None = None
    title: str = ""
    url: str = ""
    accessible_text: str | None = None


class CalendarEvent(BaseModel):
    event_id: str | None = None
    calendar_id: str | None = None
    title: str
    starts_at: datetime
    ends_at: datetime
    location: str | None = None
    notes: str | None = None


class NoteDocument(BaseModel):
    note_id: str | None = None
    title: str
    content: str | None = None
    updated_at: datetime | None = None


class MediaPlaybackState(BaseModel):
    application_id: str | None = None
    state: str
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    position_seconds: float | None = None
    volume: float | None = Field(default=None, ge=0, le=1)


class SystemState(BaseModel):
    volume: float | None = Field(default=None, ge=0, le=1)
    muted: bool | None = None
    battery_percent: float | None = Field(default=None, ge=0, le=100)
    displays: list[dict[str, Any]] = Field(default_factory=list)


class ComputerControlEvent(BaseModel):
    event_type: str
    execution_id: str
    state: ExecutionState
    message: str
    timestamp: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditRecord(BaseModel):
    execution_id: str
    session_id: str
    client_type: str
    capability: str
    operation: str
    target_application: str | None = None
    risk: ExecutionRisk
    permission_decision: PermissionDecision
    confirmation_result: str
    started_at: datetime
    finished_at: datetime
    final_state: ExecutionState
    sensitive_content_accessed: bool = False
    error_code: str | None = None
