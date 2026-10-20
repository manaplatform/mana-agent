"""Bounded, role-preserving context shared by routing and task execution."""

from __future__ import annotations

from typing import Any

from mana_agent.context_cost.estimator import estimate_value_tokens


def recent_conversation_context(
    messages: list[dict[str, Any]], *, current_turn_id: str = "", max_tokens: int = 2000,
) -> dict[str, Any]:
    visible = [
        {key: message.get(key, "") for key in ("message_id", "turn_id", "role", "content")}
        for message in messages
        if message.get("role") in {"user", "assistant"}
        and message.get("turn_id") != current_turn_id
        and str(message.get("content") or "").strip()
    ]
    turn_ids = list(dict.fromkeys(str(m["turn_id"]) for m in visible))
    selected: list[dict[str, Any]] = []
    # Keep complete exchanges: dropping a user question independently of its
    # assistant answer changes the meaning of short confirmations.
    for turn_id in reversed(turn_ids):
        exchange = [m for m in visible if str(m["turn_id"]) == turn_id]
        if estimate_value_tokens(exchange + selected) > max(0, max_tokens):
            break
        selected = exchange + selected
        if len({m["turn_id"] for m in selected}) >= 3:
            break
    return {
        "messages": selected,
        "tokens": estimate_value_tokens(selected),
        "truncated": len(selected) < len(visible),
        "latest_turn_id": turn_ids[-1] if turn_ids else "",
        "latest_message_ids": [m["message_id"] for m in visible if turn_ids and m["turn_id"] == turn_ids[-1]],
        "retrieval_required": bool(visible and not selected),
    }
