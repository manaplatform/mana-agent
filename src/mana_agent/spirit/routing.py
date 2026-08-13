"""Coordinate Spirit with the existing model router.

Routing chooses how Mana thinks. Routing does not choose who Mana is.
This module does not score models and does not create a second router.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from mana_agent.model_routing.models import RoutingDecision, RoutingRequest
from mana_agent.spirit.compiler import compile_spirit_instruction, estimate_spirit_tokens
from mana_agent.spirit.schema import SpiritRef
from mana_agent.spirit.self_model import (
    BaseSelf,
    RuntimeSelf,
    compose_base_self,
    compose_runtime_self,
)


PRE_ROUTE_MODEL_PLACEHOLDER = "the current model"


@dataclass(frozen=True, slots=True)
class RoutedSpiritBinding:
    """Post-routing Self: the same Spirit bound to one selected model."""

    base_self: BaseSelf
    decision: RoutingDecision
    runtime_self: RuntimeSelf
    compiled: str
    spirit_tokens: int

    def diagnostics(self) -> dict[str, Any]:
        return {
            "spirit_id": self.runtime_self.spirit.id,
            "spirit_version": self.runtime_self.spirit.version,
            "agent_role": self.runtime_self.agent.role,
            "selected_provider": self.runtime_self.runtime.provider,
            "selected_model": self.runtime_self.runtime.model,
        }


def resolve_base_self(
    *,
    agent_id: str | None = None,
    role: str | None = None,
    purpose: str | None = None,
    spirit: Any | None = None,
    execution_context: Any | None = None,
    settings: Any | None = None,
) -> BaseSelf:
    """Pre-routing hook: resolve identity metadata only."""

    return compose_base_self(
        spirit=spirit,
        execution_context=execution_context,
        agent_name=agent_id,
        agent_role=role,
        purpose=purpose,
        settings=settings,
    )


def estimate_pre_route_spirit_tokens(base_self: BaseSelf) -> int:
    """Deterministic small reservation. No large safety margin."""

    placeholder = compose_runtime_self(
        base_self=base_self,
        model=PRE_ROUTE_MODEL_PLACEHOLDER,
    )
    return estimate_spirit_tokens(runtime_self=placeholder)


def attach_spirit_accounting(request: RoutingRequest, base_self: BaseSelf) -> RoutingRequest:
    """Add the compiled-Spirit token count without injecting Spirit prose."""

    tokens = estimate_pre_route_spirit_tokens(base_self)
    return replace(request, expected_prompt_tokens=int(request.expected_prompt_tokens) + tokens)


def bind_runtime_self(
    base_self: BaseSelf,
    *,
    provider: str,
    model: str,
    role: str | None = None,
) -> RuntimeSelf:
    """Post-routing hook: attach the selected provider/model to Base Self."""

    return compose_runtime_self(
        base_self=base_self,
        provider=provider,
        model=model,
        agent_role=role or base_self.agent.role,
    )


def compile_bound_spirit(runtime_self: RuntimeSelf) -> str:
    """Compile the minimal Spirit representation for the bound model."""

    return compile_spirit_instruction(runtime_self)


def bind_after_route(base_self: BaseSelf, decision: RoutingDecision) -> RoutedSpiritBinding:
    runtime_self = bind_runtime_self(
        base_self,
        provider=decision.provider,
        model=decision.selected_model,
        role=decision.selected_role,
    )
    compiled = compile_bound_spirit(runtime_self)
    return RoutedSpiritBinding(
        base_self=base_self,
        decision=decision,
        runtime_self=runtime_self,
        compiled=compiled,
        spirit_tokens=estimate_spirit_tokens(compiled),
    )


def bind_parallel_candidates(
    base_self: BaseSelf,
    decision: RoutingDecision,
) -> tuple[RuntimeSelf, ...]:
    """Every candidate shares the resolved Spirit; only the runtime model differs."""

    keys = decision.competition_candidates or (f"{decision.provider}/{decision.selected_model}",)
    bound: list[RuntimeSelf] = []
    for key in keys:
        provider, separator, model = str(key).partition("/")
        if not separator:
            provider = decision.provider
            model = str(key)
        bound.append(
            bind_runtime_self(
                base_self,
                provider=provider,
                model=model or decision.selected_model,
                role=decision.selected_role,
            )
        )
    return tuple(bound)


def resume_runtime_self(
    *,
    spirit_id: str | None,
    spirit_version: int | None,
    provider: str,
    model: str,
    role: str | None = None,
    agent_name: str | None = None,
    purpose: str | None = None,
    settings: Any | None = None,
) -> RuntimeSelf:
    """Rebuild Self from a durable Spirit ref after routing or resume."""

    ref = None
    if str(spirit_id or "").strip() and spirit_version:
        ref = SpiritRef(id=str(spirit_id).strip(), version=int(spirit_version))
    base_self = resolve_base_self(
        agent_id=agent_name,
        role=role,
        purpose=purpose,
        spirit=ref,
        settings=settings,
    )
    return bind_runtime_self(base_self, provider=provider, model=model, role=role)


def route_with_spirit(
    router: Any,
    request: RoutingRequest,
    *,
    base_self: BaseSelf | None = None,
    settings: Any | None = None,
) -> RoutedSpiritBinding:
    """Resolve Spirit, let the existing router choose the model, then bind."""

    resolved = base_self or resolve_base_self(
        role=request.role,
        purpose=request.task_description,
        settings=settings,
    )
    prepared = attach_spirit_accounting(request, resolved)
    decision = router.route(prepared)
    return bind_after_route(resolved, decision)
