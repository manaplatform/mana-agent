"""Lossless validation of raw bridge conversation records."""
from __future__ import annotations
from typing import Any

def validate_chat_messages(messages: list[Any]) -> list[dict[str, Any]]:
    pending = set()
    seen = set()
    result = []
    for raw in messages:
        if not isinstance(raw, dict) or raw.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError("Invalid chat message role; no coercion or discard was performed")
        message = dict(raw)
        if message["role"] == "tool":
            call_id = message.get("tool_call_id")
            if not call_id or call_id not in pending:
                raise ValueError("Orphan or duplicate tool result; reconcile its original call")
            pending.remove(call_id)
        elif message.get("tool_calls"):
            if pending:
                raise ValueError("Previous tool calls have unresolved results")
            for call in message["tool_calls"]:
                if not call.get("id") or call["id"] in seen or not call.get("function", {}).get("name"):
                    raise ValueError("Missing or duplicate tool identity")
                pending.add(call["id"])
                seen.add(call["id"])
        elif pending and message["role"] not in {"system", "developer"}:
            raise ValueError("Tool results must precede the next conversational input")
        result.append(message)
    if pending:
        raise ValueError("Cannot infer with unresolved tool calls")
    return result
