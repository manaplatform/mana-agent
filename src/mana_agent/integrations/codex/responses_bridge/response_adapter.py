"""Convert Chat Completions responses into Responses API objects."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from mana_agent.integrations.codex.text_cleanup import sanitize_assistant_visible_text
from mana_agent.integrations.codex.tool_conversion import (
    freeform_input_from_function_arguments,
    responses_item_type_for_tool_call,
    responses_tool_call_identity,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _usage_from_chat(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    total_tokens = usage.get("total_tokens")
    result: dict[str, Any] = {}
    if input_tokens is not None:
        result["input_tokens"] = int(input_tokens)
    if output_tokens is not None:
        result["output_tokens"] = int(output_tokens)
    if total_tokens is not None:
        result["total_tokens"] = int(total_tokens)
    elif input_tokens is not None and output_tokens is not None:
        result["total_tokens"] = int(input_tokens) + int(output_tokens)
    # Preserve nested details when present without inventing pricing.
    details = usage.get("completion_tokens_details") or usage.get("output_tokens_details")
    if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
        result["output_tokens_details"] = {
            "reasoning_tokens": int(details["reasoning_tokens"]),
        }
    return result or None


def convert_chat_completion_to_response(
    chat: dict[str, Any],
    *,
    model: str,
    response_id: str | None = None,
    tool_origins: dict[str, str] | None = None,
    response_tool_names: dict[str, str] | None = None,
    tool_namespaces: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a non-streaming Responses object from a Chat Completions payload."""
    response_id = response_id or _new_id("resp")
    origins = dict(tool_origins or {})
    response_names = dict(response_tool_names or {})
    namespaces = dict(tool_namespaces or {})
    choices = chat.get("choices") if isinstance(chat.get("choices"), list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    output: list[dict[str, Any]] = []

    # DeepSeek thinking mode returns chain-of-thought on reasoning_content.
    # Keep it as a Responses reasoning item (never merge into assistant text)
    # so multi-turn tool loops can round-trip it as reasoning_content.
    reasoning_text = message.get("reasoning_content")
    if reasoning_text not in (None, ""):
        output.append(
            {
                "type": "reasoning",
                "id": _new_id("rs"),
                "status": "completed",
                "summary": [{"type": "summary_text", "text": str(reasoning_text)}],
            }
        )

    # Ordinary assistant text only — never fold reasoning_content into content.
    # Strip leaked think/DSML free-form tool noise so Codex does not treat
    # protocol soup as a successful agentMessage (empty-patch failure mode).
    text = message.get("content")
    if text not in (None, ""):
        cleaned = sanitize_assistant_visible_text(str(text))
        if cleaned:
            message_id = _new_id("msg")
            output.append(
                {
                    "type": "message",
                    "id": message_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": cleaned,
                            "annotations": [],
                        }
                    ],
                }
            )

    tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        call_id = str(tool_call.get("id") or _new_id("call"))
        name = str(function.get("name") or "")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False)
        item_type = responses_item_type_for_tool_call(name, tool_origins=origins)
        response_name, namespace = responses_tool_call_identity(
            name,
            response_tool_names=response_names,
            tool_namespaces=namespaces,
        )
        if item_type == "custom_tool_call":
            item: dict[str, Any] = {
                "type": "custom_tool_call",
                "id": _new_id("ctc"),
                "call_id": call_id,
                "name": response_name,
                "input": freeform_input_from_function_arguments(arguments),
                "status": "completed",
            }
        else:
            item = {
                "type": "function_call",
                "id": _new_id("fc"),
                "call_id": call_id,
                "name": response_name,
                "arguments": arguments,
                "status": "completed",
            }
        if namespace:
            item["namespace"] = namespace
        output.append(item)

    finish = str(choice.get("finish_reason") or "stop")
    status = "completed"
    if finish == "length":
        status = "incomplete"
    result = {
        "id": response_id,
        "object": "response",
        "created_at": int(chat.get("created") or time.time()),
        "status": status,
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "model": str(chat.get("model") or model),
        "output": output,
        "usage": _usage_from_chat(chat.get("usage")),
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    return result


def responses_error_body(
    *,
    message: str,
    code: str,
    status_code: int,
    response_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "type": "bridge_error" if status_code >= 500 else "invalid_request_error",
    }
    if extra:
        # Attach structured classification for Codex/Mana diagnostics.
        # Values must already be secret-redacted by the caller.
        for key, value in extra.items():
            if value is not None and key not in error:
                error[key] = value
    return {
        "id": response_id or _new_id("resp"),
        "object": "response",
        "status": "failed",
        "error": error,
        "output": [],
    }


__all__ = [
    "convert_chat_completion_to_response",
    "responses_error_body",
]
