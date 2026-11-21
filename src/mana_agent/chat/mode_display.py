"""Presentation metadata for repository workflow replies."""

from __future__ import annotations

from typing import Any

_MODE_PROMPT_KINDS = ("mode.approval", "mode.clarification")


def mode_reply_metadata(data: dict[str, Any], next_prompt: dict[str, Any] | None) -> dict[str, Any]:
    kind = (next_prompt or {}).get("kind")
    if kind in _MODE_PROMPT_KINDS:
        return {"mode_prompt_kind": kind}
    return {key: data[key] for key in ("mode_response_operation", "mode_status_message") if key in data}


def mode_transcript_text(message: str, metadata: dict[str, Any]) -> str:
    """Keep modal content and inbox receipts out of the Textual transcript."""
    if metadata.get("mode_prompt_kind") in _MODE_PROMPT_KINDS or metadata.get("mode_response_operation"):
        return ""
    return str(metadata.get("mode_status_message") or message)
