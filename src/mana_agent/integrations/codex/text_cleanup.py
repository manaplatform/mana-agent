"""Deterministic cleanup of leaked model control markers.

DeepSeek / NVIDIA tool multi-turn loops sometimes emit free-form think/DSML
markup into ordinary assistant ``content`` instead of structured tool_calls.
Those markers poison multi-turn history and user-facing Codex summaries.
"""

from __future__ import annotations

import re

# Leaked thinking/control markers that must not stay in ordinary assistant text.
# Real chain-of-thought belongs on ``reasoning_content`` / Responses ``reasoning``
# items, not in user-visible content.
_LEAKED_THINK_MARKERS_RE = re.compile(
    r"<think>[\s\S]*?</think>"
    r"|</?think>"
    r"|</?nowarn>"
    r"|<\|DSML\|[^>]*>"
    r"|<\|/?DSML\|?>"
    r"|<\|redacted_reasoning\|[^>]*>"
    r"|<\|think\|[^>]*>"
    r"|</?MESSAGE_END>"
    r"|</?invoke\b[^>]*>"
    r"|<!DOCTYPE\b[^>]*>",
    re.IGNORECASE,
)

# Free-form pseudo-tool / protocol noise that indicates the model narrated an
# edit instead of calling apply_patch / structured tools.
_FREEFORM_TOOL_NOISE_RE = re.compile(
    r"(?:mutation_required_but_no_mutation_tool_attempted)"
    r"|(?:python-patch-before)"
    r"|(?:actionstarted\.\d)"
    r"|(?:apply_patch\s+suppressums)"
    r"|(?:UPDATE\s+FILE\s+commands)"
    r"|(?:Transfer Protocol Templates)",
    re.IGNORECASE,
)


def strip_leaked_thinking_markers(text: str) -> str:
    """Remove leaked think/DSML/control markers from ordinary assistant text."""
    if not text:
        return ""
    cleaned = _LEAKED_THINK_MARKERS_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def looks_like_freeform_tool_garbage(text: str) -> bool:
    """True when assistant text looks like a failed free-form tool protocol dump."""
    if not text or not str(text).strip():
        return False
    sample = str(text)
    if _FREEFORM_TOOL_NOISE_RE.search(sample):
        return True
    # Dense think/DSML leakage after strip still leaves little usable prose.
    stripped = strip_leaked_thinking_markers(sample)
    if not stripped:
        return True
    marker_hits = len(_LEAKED_THINK_MARKERS_RE.findall(sample))
    if marker_hits >= 2 and len(stripped) < max(40, len(sample) // 4):
        return True
    return False


def sanitize_assistant_visible_text(text: str) -> str:
    """Clean assistant text for history and user-facing summaries."""
    cleaned = strip_leaked_thinking_markers(text)
    if looks_like_freeform_tool_garbage(text):
        # Prefer a short diagnostic over multi-KB protocol soup.
        return (
            "Prior model output was invalid free-form tool/thinking text and was redacted. "
            "Continue with structured tools only (apply_patch for edits)."
        )
    return cleaned


__all__ = [
    "looks_like_freeform_tool_garbage",
    "sanitize_assistant_visible_text",
    "strip_leaked_thinking_markers",
]
