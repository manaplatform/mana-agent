"""Compile Spirit + Self into a small identity instruction."""

from __future__ import annotations

import re

from mana_agent.context_cost.estimator import estimate_value_tokens
from mana_agent.spirit.schema import SpiritRef
from mana_agent.spirit.self_model import RuntimeModelIdentity, RuntimeSelf, compose_runtime_self


SPIRIT_MAX_COMPILED_TOKENS = 120
_DISPLAY_MODEL_MAX_CHARS = 48
_SPIRIT_MARKER_RE = re.compile(r"Mana's Spirit \(([^/)]+)/(\d+)\)")
_SPIRIT_BLOCK_RE = re.compile(
    r"(?:I am Mana-Agent\. I use [^\n]+\n\n|"
    r"You are Mana-Agent, instantiated through [^\n]+\n\n)?"
    r"Mana's Spirit \([^)]+\) is curious, bold, and calm:\n"
    r"[^\n]+\n\n"
    r"The runtime model is part of this implementation, not a separate persona\.\n"
    r"(?:When asked who I am, I say I am Mana-Agent using this model\.\n)?"
    r"Show temperament through behavior, not self-description\.\n*",
)


def spirit_prompt_marker(ref: SpiritRef) -> str:
    return f"Mana's Spirit ({ref.id}/{ref.version})"


def display_runtime_model(model: str) -> str:
    text = str(model or "").strip() or "the current model"
    if len(text) > _DISPLAY_MODEL_MAX_CHARS:
        return text[: _DISPLAY_MODEL_MAX_CHARS - 3].rstrip() + "..."
    return text


def display_runtime_backend(runtime: RuntimeModelIdentity) -> str:
    """Render 'the <provider> model <name>' from existing runtime Self fields."""

    provider = str(runtime.provider or "").strip()
    raw_model = str(runtime.model or "").strip()
    if provider and raw_model:
        prefix = f"{provider}/"
        shown = raw_model[len(prefix) :] if raw_model.lower().startswith(prefix.lower()) else raw_model
        return f"the {provider} model {display_runtime_model(shown)}"
    if provider:
        return f"the {provider} model"
    if raw_model:
        return f"the {display_runtime_model(raw_model)} model"
    return "the current model"


def compile_spirit_instruction(runtime_self: RuntimeSelf | None = None) -> str:
    """Render the compact identity/temperament instruction.

    Purpose, role, policy, memory, and coding rules are intentionally omitted.
    Identity is first-person so replies name Mana-Agent and the runtime model.
    """

    current = runtime_self or compose_runtime_self()
    backend = display_runtime_backend(current.runtime)
    return (
        f"I am Mana-Agent. I use {backend}.\n\n"
        f"{spirit_prompt_marker(current.spirit)} is curious, bold, and calm:\n"
        "understand before assuming, act when justified, stay deliberate under failure.\n\n"
        "The runtime model is part of this implementation, not a separate persona.\n"
        "When asked who I am, I say I am Mana-Agent using this model.\n"
        "Show temperament through behavior, not self-description."
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
