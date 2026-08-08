"""NVIDIA Build / NIM model-specific Chat Completions request shaping.

NVIDIA's public integrate API is OpenAI-compatible for transport, but some
hosted models (especially DeepSeek V4) require provider/model body fields
under ``chat_template_kwargs`` rather than OpenAI Responses-style
``reasoning`` / top-level ``reasoning_effort`` alone.
"""

from __future__ import annotations

from typing import Any


def is_nvidia_deepseek_model(*, provider: str | None, model: str | None) -> bool:
    provider_id = str(provider or "").strip().lower()
    model_id = str(model or "").strip().lower()
    return provider_id == "nvidia" and "deepseek" in model_id


def normalize_nvidia_deepseek_effort(effort: str | None) -> str:
    """Map Codex/OpenAI effort names onto DeepSeek V4 NIM values.

    Supported NIM values: ``none``, ``high``, ``max``.
    """
    raw = str(effort or "").strip().lower()
    if not raw:
        return "high"
    mapping = {
        "none": "none",
        "off": "none",
        "minimal": "none",
        "low": "none",
        "medium": "high",
        "high": "high",
        "xhigh": "max",
        "max": "max",
    }
    return mapping.get(raw, "high")


def deepseek_chat_template_kwargs(effort: str | None) -> dict[str, Any]:
    """Return NIM ``chat_template_kwargs`` for DeepSeek V4 Flash/Pro."""
    mapped = normalize_nvidia_deepseek_effort(effort)
    if mapped == "none":
        return {"thinking": False, "reasoning_effort": "none"}
    return {"thinking": True, "reasoning_effort": mapped}


def apply_nvidia_chat_completion_shaping(
    payload: dict[str, Any],
    *,
    provider: str | None,
    model: str | None,
    default_effort: str | None = None,
) -> dict[str, Any]:
    """Mutate a Chat Completions body for NVIDIA model quirks.

    For DeepSeek V4:
    * inject ``chat_template_kwargs.thinking`` + ``reasoning_effort``
    * remove bare top-level ``reasoning_effort`` / ``reasoning`` so NIM does
      not receive ambiguous dual controls
    * never invent secrets or rewrite model ids
    """
    if not is_nvidia_deepseek_model(provider=provider, model=model):
        return payload

    effort: str | None = None
    if isinstance(payload.get("reasoning"), dict):
        effort = payload.get("reasoning", {}).get("effort")
    if payload.get("reasoning_effort") is not None:
        effort = payload.get("reasoning_effort")
    if effort is None:
        effort = default_effort

    template = deepseek_chat_template_kwargs(None if effort is None else str(effort))
    existing = payload.get("chat_template_kwargs")
    if isinstance(existing, dict):
        merged = dict(existing)
        # Explicit caller kwargs win over defaults.
        for key, value in template.items():
            merged.setdefault(key, value)
        payload["chat_template_kwargs"] = merged
    else:
        payload["chat_template_kwargs"] = template

    payload.pop("reasoning_effort", None)
    payload.pop("reasoning", None)
    return payload


__all__ = [
    "apply_nvidia_chat_completion_shaping",
    "deepseek_chat_template_kwargs",
    "is_nvidia_deepseek_model",
    "normalize_nvidia_deepseek_effort",
]
