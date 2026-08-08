"""NVIDIA Build / NIM model-specific Chat Completions request shaping.

NVIDIA's public integrate API is OpenAI-compatible for transport, but some
hosted models (especially DeepSeek V4) require provider/model body fields
under ``chat_template_kwargs`` rather than OpenAI Responses-style
``reasoning`` / top-level ``reasoning_effort`` alone.
"""

from __future__ import annotations

from typing import Any

# Soft ceiling for DeepSeek V4 max_tokens on NVIDIA NIM. Values above this are
# almost always configuration mistakes that produce HTTP 400 rather than useful
# completions. Clamp instead of sending an invalid payload.
NVIDIA_DEEPSEEK_MAX_TOKENS_CEILING = 65_536

# DeepSeek V4 Pro / Flash on NIM: only these reasoning_effort values are valid.
NVIDIA_DEEPSEEK_REASONING_VALUES = frozenset({"none", "high", "max"})


def is_nvidia_deepseek_model(*, provider: str | None, model: str | None) -> bool:
    provider_id = str(provider or "").strip().lower()
    model_id = str(model or "").strip().lower()
    return provider_id == "nvidia" and "deepseek" in model_id


def normalize_nvidia_deepseek_effort(effort: str | None) -> str:
    """Map Codex/OpenAI effort names onto DeepSeek V4 NIM values.

    Supported NIM values: ``none``, ``high``, ``max``.
    Unsupported Codex values such as ``xhigh``, ``minimal``, and ``medium`` are
    intentionally normalized rather than forwarded.
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
    mapped = mapping.get(raw, "high")
    if mapped not in NVIDIA_DEEPSEEK_REASONING_VALUES:
        return "high"
    return mapped


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

    # Clamp max_tokens / max_completion_tokens to provider ceiling when present.
    for key in ("max_tokens", "max_completion_tokens"):
        if key in payload and payload[key] is not None:
            try:
                value = int(payload[key])
            except (TypeError, ValueError):
                continue
            if value > NVIDIA_DEEPSEEK_MAX_TOKENS_CEILING:
                payload[key] = NVIDIA_DEEPSEEK_MAX_TOKENS_CEILING
            elif value < 1:
                payload[key] = 1

    # Normalize message sequence for NIM DeepSeek chat templates.
    messages = payload.get("messages")
    if isinstance(messages, list):
        payload["messages"] = normalize_nvidia_chat_messages(messages)

    return payload


def normalize_nvidia_chat_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Ensure a valid DeepSeek/NVIDIA message sequence.

    Rules applied:
    * system messages are moved to the front (merged when multiple)
    * tool messages retain tool_call_id
    * empty orphan tool messages without tool_call_id are dropped
    * last conversational input remains a user/tool message when possible
    """
    system_parts: list[str] = []
    ordered: list[dict[str, Any]] = []
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        message = dict(raw)
        role = str(message.get("role") or "").strip().lower()
        if role == "developer":
            role = "system"
            message["role"] = "system"
        if role == "system":
            content = message.get("content")
            if content not in (None, ""):
                system_parts.append(str(content))
            continue
        if role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "").strip()
            if not tool_call_id:
                # Invalid tool result without id — skip rather than send 400.
                continue
            message["tool_call_id"] = tool_call_id
            if message.get("content") is None:
                message["content"] = ""
            ordered.append(message)
            continue
        if role in {"user", "assistant"}:
            if message.get("content") is None and not message.get("tool_calls"):
                message["content"] = ""
            # Assistant tool_calls must keep string arguments.
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                cleaned_calls: list[dict[str, Any]] = []
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    call_copy = dict(call)
                    function = call_copy.get("function")
                    if isinstance(function, dict):
                        function = dict(function)
                        arguments = function.get("arguments")
                        if arguments is not None and not isinstance(arguments, str):
                            import json

                            function["arguments"] = json.dumps(arguments, ensure_ascii=False)
                        call_copy["function"] = function
                    if not call_copy.get("id"):
                        call_copy["id"] = f"call_{len(cleaned_calls)+1}"
                    cleaned_calls.append(call_copy)
                message["tool_calls"] = cleaned_calls
            ordered.append(message)
            continue
        # Unknown role: keep as user text when content exists.
        content = message.get("content")
        if content not in (None, ""):
            ordered.append({"role": "user", "content": str(content)})

    result: list[dict[str, Any]] = []
    if system_parts:
        result.append({"role": "system", "content": "\n\n".join(system_parts)})
    result.extend(ordered)
    return result


__all__ = [
    "NVIDIA_DEEPSEEK_MAX_TOKENS_CEILING",
    "NVIDIA_DEEPSEEK_REASONING_VALUES",
    "apply_nvidia_chat_completion_shaping",
    "deepseek_chat_template_kwargs",
    "is_nvidia_deepseek_model",
    "normalize_nvidia_chat_messages",
    "normalize_nvidia_deepseek_effort",
]
