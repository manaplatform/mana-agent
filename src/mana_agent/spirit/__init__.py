"""Mana Spirit: persistent root identity and temperament."""

from mana_agent.spirit.adapter import apply_spirit_instruction
from mana_agent.spirit.compiler import (
    SPIRIT_MAX_COMPILED_TOKENS,
    compile_spirit_instruction,
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
from mana_agent.spirit.self_model import RuntimeSelf, compose_runtime_self

__all__ = [
    "DEFAULT_MANA_SPIRIT",
    "SPIRIT_MAX_COMPILED_TOKENS",
    "RuntimeSelf",
    "Spirit",
    "SpiritRef",
    "SpiritResolutionError",
    "SpiritSettings",
    "apply_spirit_instruction",
    "compile_spirit_instruction",
    "compose_runtime_self",
    "contains_spirit_instruction",
    "default_mana_spirit",
    "default_spirit_ref",
    "estimate_spirit_tokens",
    "resolve_configured_spirit",
    "resolve_spirit",
    "spirit_prompt_marker",
    "strip_spirit_instruction",
]
