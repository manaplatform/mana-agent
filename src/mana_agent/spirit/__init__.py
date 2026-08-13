"""Mana Spirit: persistent root identity and temperament."""

from mana_agent.spirit.adapter import apply_spirit_instruction
from mana_agent.spirit.compiler import (
    SPIRIT_MAX_COMPILED_TOKENS,
    compile_spirit_instruction,
    compile_spirit_semantics,
    contains_spirit_instruction,
    estimate_spirit_tokens,
    spirit_prompt_marker,
    strip_spirit_instruction,
)
from mana_agent.spirit.defaults import DEFAULT_MANA_SPIRIT
from mana_agent.spirit.errors import SpiritResolutionError
from mana_agent.spirit.registry import (
    default_mana_spirit,
    default_spirit_ref,
    resolve_configured_spirit,
    resolve_spirit,
)
from mana_agent.spirit.schema import Spirit, SpiritRef, SpiritSettings
from mana_agent.spirit.routing import (
    RoutedSpiritBinding,
    attach_spirit_accounting,
    bind_after_route,
    bind_parallel_candidates,
    resolve_base_self,
    route_with_spirit,
)
from mana_agent.spirit.self_model import BaseSelf, RuntimeSelf, compose_base_self, compose_runtime_self

__all__ = [
    "DEFAULT_MANA_SPIRIT",
    "SPIRIT_MAX_COMPILED_TOKENS",
    "BaseSelf",
    "RoutedSpiritBinding",
    "RuntimeSelf",
    "Spirit",
    "SpiritRef",
    "SpiritResolutionError",
    "SpiritSettings",
    "apply_spirit_instruction",
    "attach_spirit_accounting",
    "bind_after_route",
    "bind_parallel_candidates",
    "compile_spirit_instruction",
    "compile_spirit_semantics",
    "compose_base_self",
    "compose_runtime_self",
    "contains_spirit_instruction",
    "default_mana_spirit",
    "default_spirit_ref",
    "estimate_spirit_tokens",
    "resolve_base_self",
    "resolve_configured_spirit",
    "resolve_spirit",
    "route_with_spirit",
    "spirit_prompt_marker",
    "strip_spirit_instruction",
]
