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


def compose_runtime_self(
    *,
    spirit: Spirit | SpiritRef | None = None,
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

    ctx = execution_context
    profile = dict(model_profile or {})
    if isinstance(spirit, Spirit):
        ref = spirit.ref()
    elif isinstance(spirit, SpiritRef):
        ref = spirit
    else:
        context_id = getattr(ctx, "spirit_id", None) if ctx is not None else None
        context_version = getattr(ctx, "spirit_version", None) if ctx is not None else None
        if context_id or settings is not None:
            resolved = resolve_spirit(
                spirit_id=str(context_id or "") or None,
                spirit_version=int(context_version) if context_version else None,
                settings=settings,
            )
        else:
            resolved = resolve_configured_spirit()
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
