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


def _existing_chat_template_kwargs(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return caller-supplied template kwargs from top-level or ``extra_body``."""
    top = payload.get("chat_template_kwargs")
    if isinstance(top, dict):
        return dict(top)
    extra = payload.get("extra_body")
    if isinstance(extra, dict):
        nested = extra.get("chat_template_kwargs")
        if isinstance(nested, dict):
            return dict(nested)
    return None


def apply_nvidia_chat_completion_shaping(
    payload: dict[str, Any],
    *,
    provider: str | None,
    model: str | None,
    default_effort: str | None = None,
    nest_under_extra_body: bool = False,
) -> dict[str, Any]:
    """Mutate a Chat Completions body for NVIDIA model quirks.

    For DeepSeek V4:
    * inject ``chat_template_kwargs.thinking`` + ``reasoning_effort``
    * remove bare top-level ``reasoning_effort`` / ``reasoning`` so NIM does
      not receive ambiguous dual controls
    * never invent secrets or rewrite model ids

    Placement of ``chat_template_kwargs``:

    * ``nest_under_extra_body=False`` (default): top-level keys for raw HTTP
      clients (Codex Responses bridge / httpx ``json=payload``).
    * ``nest_under_extra_body=True``: nest under ``extra_body`` for the OpenAI
      Python SDK / LangChain path. ``Completions.create()`` rejects unknown
      kwargs such as top-level ``chat_template_kwargs``; the SDK merges
      ``extra_body`` into the HTTP JSON body.
    """
    if not is_nvidia_deepseek_model(provider=provider, model=model):
        return payload

    effort: str | None = None
    explicit_effort = False
    if isinstance(payload.get("reasoning"), dict) and payload.get("reasoning", {}).get("effort") is not None:
        effort = payload.get("reasoning", {}).get("effort")
        explicit_effort = True
    if payload.get("reasoning_effort") is not None:
        effort = payload.get("reasoning_effort")
        explicit_effort = True
    existing = _existing_chat_template_kwargs(payload)
    if effort is None and isinstance(existing, dict) and existing.get("reasoning_effort") is not None:
        effort = existing.get("reasoning_effort")
    if effort is None:
        effort = default_effort

    template = deepseek_chat_template_kwargs(None if effort is None else str(effort))
    if isinstance(existing, dict):
        merged = dict(existing)
        if explicit_effort:
            # Top-level reasoning_effort / reasoning.effort (including force-none
            # after a tools compatibility retry) overrides init defaults that may
            # already be nested under extra_body.
            merged.update(template)
        else:
            # Nested caller kwargs win; fill only missing keys from defaults.
            for key, value in template.items():
                merged.setdefault(key, value)
    else:
        merged = template

    if nest_under_extra_body:
        extra_body = dict(payload.get("extra_body") or {})
        extra_body["chat_template_kwargs"] = merged
        payload["extra_body"] = extra_body
        payload.pop("chat_template_kwargs", None)
    else:
        payload["chat_template_kwargs"] = merged
        # If callers nested the field under extra_body, promote it for raw HTTP
        # and drop the nested copy so the body is not duplicated.
        extra_body = payload.get("extra_body")
        if isinstance(extra_body, dict) and "chat_template_kwargs" in extra_body:
            extra_body = dict(extra_body)
            extra_body.pop("chat_template_kwargs", None)
            if extra_body:
                payload["extra_body"] = extra_body
            else:
                payload.pop("extra_body", None)

    payload.pop("reasoning_effort", None)
    payload.pop("reasoning", None)
    return payload


__all__ = [
    "apply_nvidia_chat_completion_shaping",
    "deepseek_chat_template_kwargs",
    "is_nvidia_deepseek_model",
    "normalize_nvidia_deepseek_effort",
]
