"""Prompt adapter that injects compiled Spirit exactly once.

Provider adapters may change placement or formatting, but must call this
compiler so Spirit meaning stays stable across models.
"""

from __future__ import annotations

from typing import Any, Mapping

from mana_agent.spirit.compiler import (
    compile_spirit_instruction,
    contains_spirit_instruction,
    strip_spirit_instruction,
)
from mana_agent.spirit.self_model import RuntimeSelf, compose_runtime_self


def apply_spirit_instruction(
    system_prompt: str,
    runtime_self: RuntimeSelf | None = None,
    *,
    execution_context: Any | None = None,
    agent_name: str | None = None,
    agent_role: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    purpose: str | None = None,
    model_profile: Mapping[str, Any] | None = None,
    settings: Any | None = None,
) -> str:
    """Prepend the compiled Spirit instruction when it is not already present."""

    current = runtime_self or compose_runtime_self(
        execution_context=execution_context,
        agent_name=agent_name,
        agent_role=agent_role,
        provider=provider,
        model=model,
        purpose=purpose,
        model_profile=model_profile,
        settings=settings,
    )
    remainder = str(system_prompt or "").strip()
    if contains_spirit_instruction(remainder, current.spirit):
        return remainder
    compiled = compile_spirit_instruction(current)
    if not remainder:
        return compiled
    remainder = strip_spirit_instruction(remainder)
    if not remainder:
        return compiled
    return f"{compiled}\n\n{remainder}"
