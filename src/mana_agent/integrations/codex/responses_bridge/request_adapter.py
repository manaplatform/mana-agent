"""Convert OpenAI Responses requests into Chat Completions requests."""

from __future__ import annotations

import json
from typing import Any

from mana_agent.config.nvidia_model_requests import (
    apply_nvidia_chat_completion_shaping,
    is_nvidia_deepseek_model,
    normalize_nvidia_chat_messages,
    normalize_nvidia_deepseek_effort,
)
from mana_agent.integrations.codex.responses_bridge.models import BridgeUpstreamConfig
from mana_agent.integrations.codex.text_cleanup import sanitize_assistant_visible_text
from mana_agent.integrations.codex.tool_conversion import convert_responses_tools
from mana_agent.model_routing.models import provider_request_overrides_from_configuration

# Non-HTTP metadata key attached to the chat payload for the local bridge only.
# The server must strip this before sending the body upstream.
MANA_BRIDGE_META_KEY = "_mana_bridge"


def normalize_reasoning_effort(
    *,
    provider: str,
    model: str,
    effort: str | None,
) -> str | None:
    """Map Codex/OpenAI reasoning effort onto provider-supported values."""
    if effort is None or not str(effort).strip():
        return None
    raw = str(effort).strip().lower()
    if is_nvidia_deepseek_model(provider=provider, model=model):
        return normalize_nvidia_deepseek_effort(raw)
    return raw


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type in {"input_text", "output_text", "text"}:
                parts.append(str(item.get("text") or item.get("content") or ""))
            elif item_type == "input_image":
                parts.append("[image]")
            else:
                text = item.get("text") or item.get("content")
                if text is not None:
                    parts.append(str(text))
        return "".join(parts)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or json.dumps(content))
    return str(content)


def _reasoning_item_to_text(item: dict[str, Any]) -> str:
    """Extract chain-of-thought text from a Responses reasoning item."""
    summary = item.get("summary")
    if isinstance(summary, list):
        parts: list[str] = []
        for entry in summary:
            if isinstance(entry, dict):
                text = entry.get("text") or entry.get("content")
                if text not in (None, ""):
                    parts.append(str(text))
            elif isinstance(entry, str) and entry:
                parts.append(entry)
        if parts:
            return "".join(parts)
    content = item.get("content")
    if content not in (None, ""):
        return _content_to_text(content)
    text = item.get("text")
    if text not in (None, ""):
        return str(text)
    return ""


def _convert_tools(
    tools: Any,
    *,
    provider: str = "",
    model: str = "",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Convert Responses tools to Chat Completions; fail on unsupported shapes.

    Silent dropping of Codex tools is forbidden. See ``tool_conversion``.
    Returns the converted tools and their local call metadata.
    """
    report = convert_responses_tools(
        tools,
        provider=provider,
        model=model,
        transport="responses_bridge",
        fail_on_unsupported=True,
    )
    return (
        list(report.converted_tools),
        {
            "tool_origins": dict(report.tool_origins),
            "response_tool_names": dict(report.response_tool_names),
            "tool_namespaces": dict(report.tool_namespaces),
        },
    )


def _convert_tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        # Responses: {"type":"function","name":"shell"}
        if tool_choice.get("type") == "function" and "function" not in tool_choice:
            name = tool_choice.get("name")
            if name:
                return {"type": "function", "function": {"name": name}}
        return tool_choice
    return tool_choice


def _append_message(messages: list[dict[str, Any]], role: str, content: Any, **extra: Any) -> None:
    message: dict[str, Any] = {"role": role, "content": content if content is not None else ""}
    message.update(extra)
    messages.append(message)


def convert_responses_request_to_chat(
    body: dict[str, Any],
    *,
    upstream: BridgeUpstreamConfig,
) -> dict[str, Any]:
    """Map a Responses API request body to Chat Completions."""
    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if instructions not in (None, ""):
        _append_message(messages, "system", _content_to_text(instructions))

    raw_input = body.get("input")
    if isinstance(raw_input, str):
        _append_message(messages, "user", raw_input)
    elif isinstance(raw_input, list):
        pending_tool_calls: list[dict[str, Any]] = []
        # DeepSeek requires reasoning_content on the assistant message that
        # performed tool calls (and on later turns of the same tool loop).
        pending_reasoning: str | None = None

        def take_reasoning_kwargs() -> dict[str, Any]:
            nonlocal pending_reasoning
            if pending_reasoning in (None, ""):
                return {}
            value = pending_reasoning
            pending_reasoning = None
            return {"reasoning_content": value}

        def flush_tool_calls() -> None:
            nonlocal pending_tool_calls
            if not pending_tool_calls:
                return
            _append_message(
                messages,
                "assistant",
                "",
                tool_calls=pending_tool_calls,
                **take_reasoning_kwargs(),
            )
            pending_tool_calls = []

        for item in raw_input:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type == "reasoning":
                # Attach CoT to the next assistant message; never as user text.
                text = _reasoning_item_to_text(item)
                if text:
                    pending_reasoning = (
                        f"{pending_reasoning}\n{text}" if pending_reasoning else text
                    )
                continue
            if item_type in {"", "message"} or item.get("role") in {"user", "assistant", "system", "developer"}:
                flush_tool_calls()
                role = str(item.get("role") or "user")
                if role == "developer":
                    role = "system"
                content = _content_to_text(item.get("content"))
                if role == "assistant":
                    # History rebuild: never re-feed think/DSML soup to DeepSeek.
                    content = sanitize_assistant_visible_text(content)
                # Assistant messages may already carry completed tool_calls.
                tool_calls = item.get("tool_calls")
                extra = take_reasoning_kwargs() if role == "assistant" else {}
                # Prefer explicit reasoning_content on the item when present.
                if role == "assistant" and item.get("reasoning_content") not in (None, ""):
                    extra["reasoning_content"] = str(item.get("reasoning_content"))
                if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
                    _append_message(messages, "assistant", content, tool_calls=tool_calls, **extra)
                else:
                    _append_message(messages, role, content, **extra)
                continue
            if item_type in {"function_call", "custom_tool_call"}:
                call_id = str(item.get("call_id") or item.get("id") or "")
                name = str(item.get("name") or "")
                # Freeform/custom tools use ``input``; function tools use ``arguments``.
                arguments = item.get("arguments")
                if arguments is None and item.get("input") is not None:
                    freeform = item.get("input")
                    if isinstance(freeform, str):
                        arguments = json.dumps({"input": freeform}, ensure_ascii=False)
                    else:
                        arguments = json.dumps({"input": freeform}, ensure_ascii=False)
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments or {}, ensure_ascii=False)
                pending_tool_calls.append(
                    {
                        "id": call_id or f"call_{len(pending_tool_calls)+1}",
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                        # Local-only marker removed before the Chat request.
                        "_mana_namespace": str(item.get("namespace") or ""),
                    }
                )
                continue
            if item_type in {"function_call_output", "custom_tool_call_output"}:
                flush_tool_calls()
                call_id = str(item.get("call_id") or item.get("id") or "")
                output = item.get("output")
                if not isinstance(output, str):
                    output = json.dumps(output, ensure_ascii=False) if output is not None else ""
                _append_message(messages, "tool", output, tool_call_id=call_id)
                continue
            flush_tool_calls()
            # Unknown item: keep textual content when present.
            text = _content_to_text(item.get("content") or item.get("text") or "")
            if text:
                _append_message(messages, "user", text)
        flush_tool_calls()
        # Leftover reasoning without a following assistant turn is dropped —
        # it must not become a user message (that confuses tool loops).
        pending_reasoning = None
    elif raw_input is None and body.get("messages"):
        # Some clients already send chat-shaped messages under alternate keys.
        for message in body.get("messages") or []:
            if isinstance(message, dict):
                messages.append(dict(message))

    model = str(body.get("model") or upstream.model or "").strip() or upstream.model
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": bool(body.get("stream")),
    }
    tools, tool_metadata = _convert_tools(
        body.get("tools"),
        provider=upstream.provider,
        model=model,
    )
    if tools:
        payload["tools"] = tools
    _remap_namespaced_tool_calls(
        messages,
        response_tool_names=tool_metadata["response_tool_names"],
        tool_namespaces=tool_metadata["tool_namespaces"],
    )
    if tool_metadata["tool_origins"]:
        # Local-only metadata for stream/response adapters; stripped before HTTP.
        meta = dict(payload.get(MANA_BRIDGE_META_KEY) or {})
        meta.update(tool_metadata)
        payload[MANA_BRIDGE_META_KEY] = meta
    tool_choice = _convert_tool_choice(body.get("tool_choice"))
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    for source_key, target_key in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("max_output_tokens", "max_tokens"),
        ("max_completion_tokens", "max_completion_tokens"),
        ("presence_penalty", "presence_penalty"),
        ("frequency_penalty", "frequency_penalty"),
        ("user", "user"),
        ("n", "n"),
        ("stop", "stop"),
        ("response_format", "response_format"),
    ):
        if body.get(source_key) is not None:
            payload[target_key] = body[source_key]

    effort = None
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
    if body.get("reasoning_effort") is not None:
        effort = body.get("reasoning_effort")
    mapped = normalize_reasoning_effort(
        provider=upstream.provider, model=model, effort=None if effort is None else str(effort)
    )
    # Non-DeepSeek hosts may still accept top-level reasoning_effort.
    if mapped is not None and not is_nvidia_deepseek_model(
        provider=upstream.provider, model=model
    ):
        payload["reasoning_effort"] = mapped
    elif mapped is not None:
        # Stash for NVIDIA DeepSeek shaping below.
        payload["reasoning_effort"] = mapped

    # Optional provider/model overrides (never secrets / never routing or
    # catalog metadata). Defense in depth: profile.configuration may still
    # contain /v1/models identity fields (id, object, created, owned_by) or
    # routing bookkeeping (source_levels); strip before the HTTP body.
    safe_overrides = provider_request_overrides_from_configuration(
        dict(upstream.request_overrides or {}),
        for_http_body=True,
    )
    for key, value in safe_overrides.items():
        if key in {"api_key", "authorization", "Authorization"}:
            continue
        if key == "extra_body" and isinstance(value, dict):
            existing = dict(payload.get("extra_body") or {})
            existing.update(value)
            payload["extra_body"] = existing
            continue
        if key == "chat_template_kwargs" and isinstance(value, dict):
            extra = dict(payload.get("extra_body") or {})
            nested = dict(extra.get("chat_template_kwargs") or {})
            nested.update(value)
            extra["chat_template_kwargs"] = nested
            payload["extra_body"] = extra
            continue
        payload[key] = value

    # Flatten extra_body into the top-level request for raw HTTP chat/completions.
    # Re-filter after flatten so nested catalog/routing junk never lands as body
    # keys either (NVIDIA: Unsupported parameter(s): created, id, object, owned_by).
    extra_body = payload.pop("extra_body", None)
    if isinstance(extra_body, dict):
        filtered_extra = provider_request_overrides_from_configuration(
            extra_body,
            for_http_body=True,
        )
        for key, value in filtered_extra.items():
            if key == "chat_template_kwargs" and isinstance(value, dict):
                nested = dict(payload.get("chat_template_kwargs") or {})
                nested.update(value)
                payload["chat_template_kwargs"] = nested
            else:
                payload.setdefault(key, value)

    # DeepSeek V4 on NVIDIA requires chat_template_kwargs; bare reasoning_effort
    # alone can hang, 4xx, or produce empty streams (seen as Codex systemError/410).
    #
    # When tools are attached, force thinking off. DeepSeek V4 with
    # thinking=True + tools routinely emits free-form pseudo-tool text
    # (DSML / <invoke name="exec_command">) instead of structured tool_calls,
    # which Codex treats as a completed agentMessage with zero file changes
    # (SWE-bench empty_patch / status=ok). Mirrors multi-agent compatibility:
    # tools + chat reasoning is not supported for NVIDIA.
    effective_effort = mapped or "high"
    if tools and is_nvidia_deepseek_model(provider=upstream.provider, model=model):
        effective_effort = "none"
        # Mark explicit so shaping overrides nested request_overrides defaults.
        payload["reasoning_effort"] = "none"
    apply_nvidia_chat_completion_shaping(
        payload,
        provider=upstream.provider,
        model=model,
        default_effort=effective_effort,
    )

    # For non-DeepSeek NVIDIA / other bridge hosts still ensure a stable sequence:
    # system first, tool results retain tool_call_id.
    if isinstance(payload.get("messages"), list):
        if is_nvidia_deepseek_model(provider=upstream.provider, model=model):
            # Already normalized inside apply_nvidia_chat_completion_shaping.
            pass
        elif str(upstream.provider or "").strip().lower() == "nvidia":
            payload["messages"] = normalize_nvidia_chat_messages(payload["messages"])
        else:
            payload["messages"] = _ensure_system_first(payload["messages"])
    return payload


def _ensure_system_first(messages: list[Any]) -> list[dict[str, Any]]:
    """Move leading system messages to the front without other rewrites."""
    systems: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        message = dict(raw)
        if str(message.get("role") or "").strip().lower() == "system":
            systems.append(message)
        else:
            rest.append(message)
    return systems + rest


def _remap_namespaced_tool_calls(
    messages: list[dict[str, Any]],
    *,
    response_tool_names: dict[str, str],
    tool_namespaces: dict[str, str],
) -> None:
    """Replace Responses ``(namespace, name)`` pairs with Chat function names."""
    for message in messages:
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            namespace = str(tool_call.pop("_mana_namespace", "") or "")
            if not namespace:
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            response_name = str(function.get("name") or "")
            matches = [
                chat_name
                for chat_name, name in response_tool_names.items()
                if name == response_name and tool_namespaces.get(chat_name) == namespace
            ]
            if len(matches) == 1:
                function["name"] = matches[0]


__all__ = [
    "MANA_BRIDGE_META_KEY",
    "convert_responses_request_to_chat",
    "normalize_reasoning_effort",
]
