"""Normalization helpers for structured tool result payloads."""
from __future__ import annotations

import json
from typing import Any


def structured_tool_error_detail(output: Any) -> str | None:
    """Return a failure description when a tool completed with an error payload.

    A tool callback's normal completion only means its function returned. Tools
    commonly encode domain failures in JSON, so callers must inspect that
    payload before presenting the invocation as successful.
    """
    payload = output if isinstance(output, dict) else _parse_payload(output)
    if not isinstance(payload, dict):
        return None
    failed = payload.get("ok") is False or payload.get("is_error") is True
    status = str(payload.get("status", "")).strip().lower()
    failed = failed or status in {"error", "failed", "failure", "blocked"}
    error = payload.get("error")
    if not failed and error and payload.get("ok") is not True:
        failed = True
    if not failed:
        return None
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or "Tool failed.")
    return str(error or payload.get("message") or payload.get("error_code") or "Tool failed.")


def _parse_payload(output: Any) -> dict[str, Any] | None:
    if not isinstance(output, str):
        return None
    text = output.strip()
    if not text:
        return None
    candidates = (text, text.split("\n", 1)[-1].strip())
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict):
            return payload
    return None
