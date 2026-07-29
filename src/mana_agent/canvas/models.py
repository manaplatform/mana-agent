"""Immutable typed contracts for A2UI surfaces and renderer actions."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mana_agent.canvas.config import MANA_CATALOG_ID, WIRE_VERSION


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CanvasSource(str, Enum):
    AGENT = "agent"
    WORKFLOW = "workflow"
    NODE = "node"
    A2A = "a2a"
    SYSTEM = "system"
    RENDERER = "renderer"


class CanvasEventType(str, Enum):
    CREATE = "createSurface"
    COMPONENTS = "updateComponents"
    DATA = "updateDataModel"
    DELETE = "deleteSurface"
    ACTION = "action"
    ERROR = "validationError"
    COMPLETE = "streamComplete"


class OwnerRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    agent_id: str | None = Field(default=None, max_length=128)
    automation_id: str | None = Field(default=None, max_length=128)
    task_id: str | None = Field(default=None, max_length=128)
    workflow_id: str | None = Field(default=None, max_length=128)
    node_id: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def require_owner(self) -> "OwnerRef":
        if not any(
            (
                self.agent_id,
                self.task_id,
                self.workflow_id,
                self.node_id,
                self.automation_id,
            )
        ):
            raise ValueError("At least one runtime owner identifier is required.")
        return self


class ActionDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_.:-]*$"
    )
    context: dict[str, Any] = Field(default_factory=dict)
    side_effect: bool = False
    permission_scope: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def require_permission_for_side_effect(self) -> "ActionDeclaration":
        if self.side_effect and not self.permission_scope:
            raise ValueError(
                "Side-effecting canvas actions require a permission scope."
            )
        if not self.side_effect and self.permission_scope:
            raise ValueError(
                "Read-only canvas actions cannot declare a permission scope."
            )
        return self


class Component(BaseModel):
    """Mana catalog component in the A2UI v0.9 adjacency-list shape."""

    model_config = ConfigDict(extra="allow", frozen=True)
    id: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"
    )
    component: str = Field(min_length=1, max_length=64)
    actions: list[ActionDeclaration] = Field(default_factory=list, max_length=16)

    @field_validator("actions")
    @classmethod
    def unique_actions(cls, value: list[ActionDeclaration]) -> list[ActionDeclaration]:
        names = [item.name for item in value]
        if len(names) != len(set(names)):
            raise ValueError("Component action names must be unique.")
        return value


class RendererCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol_versions: tuple[str, ...] = (WIRE_VERSION,)
    catalog_ids: tuple[str, ...] = (MANA_CATALOG_ID,)
    inline_catalogs: bool = False
    max_components: int = Field(default=250, gt=0)
    supports_actions: bool = True
    supports_data_model: bool = True


class AgentCapabilities(RendererCapabilities):
    validation_retry_limit: int = Field(default=1, ge=0, le=3)


class CanvasEventEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    event_id: str = Field(default_factory=lambda: f"canvas_evt_{uuid4().hex}")
    session_id: str = Field(min_length=1, max_length=160)
    conversation_id: str = Field(min_length=1, max_length=160)
    surface_id: str = Field(min_length=1, max_length=160)
    workflow_id: str | None = Field(default=None, max_length=128)
    node_id: str | None = Field(default=None, max_length=128)
    task_id: str | None = Field(default=None, max_length=128)
    agent_id: str | None = Field(default=None, max_length=128)
    automation_id: str | None = Field(default=None, max_length=128)
    correlation_id: str = Field(min_length=1, max_length=160)
    sequence: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=utc_now)
    protocol_version: Literal["v0.9"] = WIRE_VERSION
    source: CanvasSource
    event_type: CanvasEventType
    parent_event_id: str | None = Field(default=None, max_length=160)
    retain_on_complete: bool | None = None
    payload: dict[str, Any]


class SurfaceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    session_id: str
    conversation_id: str
    surface_id: str
    catalog_id: str
    protocol_version: Literal["v0.9"] = WIRE_VERSION
    owner: OwnerRef
    version: int = Field(ge=1)
    last_sequence: int = Field(ge=1)
    components: tuple[Component, ...] = ()
    data_model: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    deleted: bool = False
    completed: bool = False
    retain_on_complete: bool = True


class CanvasSession(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    session_id: str
    conversation_id: str
    surface_ids: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class RendererAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action_id: str = Field(default_factory=lambda: f"canvas_action_{uuid4().hex}")
    version: Literal["v0.9"] = WIRE_VERSION
    session_id: str = Field(min_length=1, max_length=160)
    conversation_id: str = Field(min_length=1, max_length=160)
    surface_id: str = Field(min_length=1, max_length=160)
    source_component_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    correlation_id: str = Field(min_length=1, max_length=160)
    timestamp: datetime = Field(default_factory=utc_now)
    context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Canvas action timestamps must include a timezone.")
        return value


class ValidationError(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: str = "VALIDATION_FAILED"
    surface_id: str
    path: str
    message: str
    retryable: bool = False


class CanvasActionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action_id: str
    status: Literal["accepted", "permission_required", "delivered", "rejected"]
    routed_to: OwnerRef
    permission_request_id: str | None = None


AgentToRendererMessage = CanvasEventEnvelope
RendererToAgentAction = RendererAction
ComponentTree = tuple[Component, ...]
DataModel = dict[str, Any]
SurfaceVersion = int
