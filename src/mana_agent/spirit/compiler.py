"""Compile Spirit + Self into a small identity instruction."""

from __future__ import annotations

import re

from mana_agent.context_cost.estimator import estimate_value_tokens
from mana_agent.spirit.registry import default_spirit_ref
from mana_agent.spirit.schema import SpiritRef
from mana_agent.spirit.self_model import RuntimeModelIdentity, RuntimeSelf, compose_runtime_self


SPIRIT_MAX_COMPILED_TOKENS = 120
_DISPLAY_MODEL_MAX_CHARS = 48
_SPIRIT_MARKER_RE = re.compile(r"Mana's Spirit \(([^/)]+)/(\d+)\)")
_SPIRIT_BLOCK_RE = re.compile(
    r"(?:You are Mana-Agent, currently instantiated through [^\n]+\n\n|"
    r"You are Mana-Agent, instantiated through [^\n]+\n\n|"
    r"I am Mana-Agent\. I use [^\n]+\n\n)?"
    r"Mana's Spirit \([^)]+\) is curious, bold, and calm:\n"
    r"[^\n]+\n*"
    r"(?:The runtime model is part of (?:this|your current) implementation, not a separate persona[^\n]*\n)?"
    r"(?:When asked who I am, I say I am Mana-Agent using this model\.\n)?"
    r"(?:Show temperament through behavior, not self-description\.\n*)?",
)


def spirit_prompt_marker(ref: SpiritRef) -> str:
    return f"Mana's Spirit ({ref.id}/{ref.version})"


def display_runtime_model(model: str) -> str:
    text = str(model or "").strip() or "the current model"
    if len(text) > _DISPLAY_MODEL_MAX_CHARS:
        return text[: _DISPLAY_MODEL_MAX_CHARS - 3].rstrip() + "..."
    return text


def display_runtime_instantiation(runtime: RuntimeModelIdentity) -> str:
    """Name the selected inference backend as ordinary session metadata."""

    provider = str(runtime.provider or "").strip()
    raw_model = str(runtime.model or "").strip()
    if provider and raw_model:
        prefix = f"{provider}/"
        if raw_model.lower().startswith(prefix.lower()):
            return display_runtime_model(raw_model)
        return display_runtime_model(f"{provider}/{raw_model}")
    if raw_model:
        return display_runtime_model(raw_model)
    if provider:
        return display_runtime_model(provider)
    return "the current model"


def compile_spirit_semantics(ref: SpiritRef | None = None) -> str:
    """Render model-free Spirit meaning. Adapter binds the inference model later."""

    current = ref or default_spirit_ref()
    return (
        f"{spirit_prompt_marker(current)} is curious, bold, and calm:\n"
        "understand before unsupported assumptions, act decisively when justified, "
        "and remain deliberate under uncertainty or failure."
    )


def compile_spirit_instruction(runtime_self: RuntimeSelf | None = None) -> str:
    """Bind unchanged Spirit semantics to the selected inference model.

    Call this only after routing has selected a provider/model. The session
    line names the product and that model as ordinary application metadata.
    Purpose, role, policy, memory, and coding rules are intentionally omitted.
    """

    current = runtime_self or compose_runtime_self()
    instantiation = display_runtime_instantiation(current.runtime)
    return (
        f"You are Mana-Agent, currently instantiated through {instantiation}.\n\n"
        f"{compile_spirit_semantics(current.spirit)}\n\n"
        "The runtime model is part of your current implementation, not a separate persona you must imitate."
    )


def estimate_spirit_tokens(text: str | None = None, *, runtime_self: RuntimeSelf | None = None) -> int:
    payload = text if text is not None else compile_spirit_instruction(runtime_self)
    return estimate_value_tokens(payload)


def contains_spirit_instruction(text: str, ref: SpiritRef | None = None) -> bool:
    payload = str(text or "")
    if ref is not None:
        return spirit_prompt_marker(ref) in payload
    return bool(_SPIRIT_MARKER_RE.search(payload))


def strip_spirit_instruction(text: str) -> str:
    cleaned = _SPIRIT_BLOCK_RE.sub("", str(text or ""), count=1)
    return cleaned.strip()


def extract_spirit_ref_from_prompt(text: str) -> SpiritRef | None:
    match = _SPIRIT_MARKER_RE.search(str(text or ""))
    if match is None:
        return None
    return SpiritRef(id=match.group(1), version=int(match.group(2)))
