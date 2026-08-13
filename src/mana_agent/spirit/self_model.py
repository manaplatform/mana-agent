"""Runtime Self: Spirit plus the current agent role and runtime model.

Self composes existing execution/model state. It does not own provider routing
or duplicate authoritative model selection.
"""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mana_agent.spirit.registry import resolve_configured_spirit, resolve_spirit
from mana_agent.spirit.schema import Spirit, SpiritRef


class RuntimeAgentIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    role: str = Field(min_length=1)

    @field_validator("name", "role")
    @classmethod
    def normalize_agent_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("runtime agent name and role must be non-empty")
        return text


class RuntimeModelIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = ""
    model: str = ""

    @field_validator("provider", "model")
    @classmethod
    def normalize_runtime_text(cls, value: str) -> str:
        return str(value or "").strip()


class RuntimePurpose(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task: str = ""

    @field_validator("task")
    @classmethod
    def normalize_task(cls, value: str) -> str:
        return str(value or "").strip()


class BaseSelf(BaseModel):
    """Pre-routing identity. Spirit is resolved; no model is bound yet."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spirit: Spirit
    agent: RuntimeAgentIdentity
    purpose: RuntimePurpose = Field(default_factory=RuntimePurpose)

    def ref(self) -> SpiritRef:
        return self.spirit.ref()

    def durable_ref(self) -> dict[str, int | str]:
        return {"id": self.spirit.id, "version": self.spirit.version}


class RuntimeSelf(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    spirit: SpiritRef
    agent: RuntimeAgentIdentity
    runtime: RuntimeModelIdentity
    purpose: RuntimePurpose = Field(default_factory=RuntimePurpose)

    def durable_ref(self) -> dict[str, int | str]:
        """Persist only the versioned Spirit identifier."""

        return {"id": self.spirit.id, "version": self.spirit.version}


def _optional_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _resolve_spirit_object(
    *,
    spirit: Spirit | SpiritRef | None,
    execution_context: Any | None,
    settings: Any | None,
) -> Spirit:
    if isinstance(spirit, Spirit):
        return spirit
    if isinstance(spirit, SpiritRef):
        return resolve_spirit(spirit_id=spirit.id, spirit_version=spirit.version, settings=settings)
    context_id = getattr(execution_context, "spirit_id", None) if execution_context is not None else None
    context_version = getattr(execution_context, "spirit_version", None) if execution_context is not None else None
    if context_id or settings is not None:
        return resolve_spirit(
            spirit_id=str(context_id or "") or None,
            spirit_version=int(context_version) if context_version else None,
            settings=settings,
        )
    return resolve_configured_spirit()


def compose_base_self(
    *,
    spirit: Spirit | SpiritRef | None = None,
    execution_context: Any | None = None,
    agent_name: str | None = None,
    agent_role: str | None = None,
    purpose: str | None = None,
    settings: Any | None = None,
) -> BaseSelf:
    """Resolve stable identity only. Do not compile a model-specific prompt."""

    resolved = _resolve_spirit_object(
        spirit=spirit,
        execution_context=execution_context,
        settings=settings,
    )
    role = _optional_text(agent_role, getattr(execution_context, "agent_role", None), "main")
    name = _optional_text(
        agent_name,
        getattr(execution_context, "agent_id", None),
        f"{role}-agent",
    )
    return BaseSelf(
        spirit=resolved,
        agent=RuntimeAgentIdentity(name=name, role=role),
        purpose=RuntimePurpose(task=_optional_text(purpose)),
    )


def compose_runtime_self(
    *,
    spirit: Spirit | SpiritRef | None = None,
    base_self: BaseSelf | None = None,
    execution_context: Any | None = None,
    agent_name: str | None = None,
    agent_role: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    purpose: str | None = None,
    model_profile: Mapping[str, Any] | None = None,
    settings: Any | None = None,
) -> RuntimeSelf:
    """Compose Self from a Spirit ref and existing runtime/role state."""

    if base_self is not None:
        spirit = spirit or base_self.spirit
        agent_name = agent_name or base_self.agent.name
        agent_role = agent_role or base_self.agent.role
        purpose = purpose or base_self.purpose.task

    ctx = execution_context
    profile = dict(model_profile or {})
    resolved = _resolve_spirit_object(spirit=spirit, execution_context=ctx, settings=settings)
    ref = resolved.ref()

    role = _optional_text(agent_role, getattr(ctx, "agent_role", None), "main")
    name = _optional_text(
        agent_name,
        getattr(ctx, "agent_id", None),
        f"{role}-agent",
    )
    runtime_model = _optional_text(
        model,
        profile.get("model"),
        getattr(ctx, "resolved_model", None),
    )
    runtime_provider = _optional_text(provider, profile.get("provider"))
    task = _optional_text(purpose)
    return RuntimeSelf(
        spirit=ref,
        agent=RuntimeAgentIdentity(name=name, role=role),
        runtime=RuntimeModelIdentity(provider=runtime_provider, model=runtime_model),
        purpose=RuntimePurpose(task=task),
    )
