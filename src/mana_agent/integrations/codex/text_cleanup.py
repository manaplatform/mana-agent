"""Deterministic cleanup of leaked model control markers.

DeepSeek / NVIDIA tool multi-turn loops (and some Codex coding turns) sometimes
emit free-form think/DSML/tool-invocation markup into ordinary assistant
``content`` instead of structured tool_calls. Those markers poison multi-turn
history and user-facing Codex summaries.
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

# Tool-call / agent-protocol markup that models dump into prose when structured
# tool routing fails (e.g. ultracall, parameter XML, namespaced pseudo-tags).
_LEAKED_TOOL_MARKUP_RE = re.compile(
    # Full parameter / function / tool_use blocks when both ends exist.
    r"<parameter\b[^>]*>[\s\S]*?</parameter>"
    r"|</?parameter\b[^>]*>"
    r"|</?function(?:_call|_calls|_response)?\b[^>]*>"
    r"|</?tool_(?:call|calls|use|result|response|name|args?)\b[^>]*>"
    r"|</?tool_call_argument\b[^>]*>"
    r"|</?cmd\b[^>]*>"
    r"|</?operator\b[^>]*>"
    r"|</?minion\b[^>]*>"
    # Namespaced pseudo-tags: <danke:ultracall_calls...>, </ison:endpoint-start>
    r"|</?[a-z][\w.-]*:[a-z][\w.:-]*\b[^>]*>"
    # Broken open tags that never close cleanly: <danke:ultracall_calls{...
    r"|<[a-z][\w.-]*:[a-z][\w.:-]*\{[^<\n]{0,400}"
    # HTML fragments sometimes woven into the same soup.
    r"|</?span\b[^>]*>"
    r"|</?badge\b[^>]*>"
    # XML-ish attribute dumps that are not real prose.
    r"""|(?:^|[\s"'`])(?:name|cmd|command|path)="[^"]{0,200}"(?=[\s"'`<>]|$)""",
    re.IGNORECASE | re.MULTILINE,
)

# Free-form pseudo-tool / protocol noise that indicates the model narrated an
# edit (or dumped a broken tool invocation) instead of structured tools.
_FREEFORM_TOOL_NOISE_RE = re.compile(
    r"(?:mutation_required_but_no_mutation_tool_attempted)"
    r"|(?:python-patch-before)"
    r"|(?:actionstarted\.\d)"
    r"|(?:apply_patch\s+suppressums)"
    r"|(?:UPDATE\s+FILE\s+commands)"
    r"|(?:Transfer Protocol Templates)"
    r"|(?:ultracall[_-]?calls?)"
    r"|(?:tools?\s+invocation\s+syntax)"
    r"|(?:garbage\s+output\s+was\s+produced)"
    r"|(?:max_output_tokens)"
    r"|(?:<parameter\b)"
    r"|(?:</?danke:)"
    r"|(?:</?ison:)"
    r"|(?:birdswithering)"
    r"|(?:function_calls?\s*\{)"
    r"|(?:tool_calls?\s*\{)"
    # Tool names only when wrapped as free-form XML/pseudo-calls, not prose.
    r"|(?:<\s*(?:run_terminal_command|run_shell_command|exec_command)\b)"
    r"|(?:invoke\s+name\s*=\s*[\"'](?:run_terminal_command|run_shell_command|exec_command))",
    re.IGNORECASE,
)

_REDACTED_TOOL_GARBAGE = (
    "Prior model output was invalid free-form tool/thinking text and was redacted. "
    "Continue with a structured tool registered for this turn."
)


def strip_leaked_thinking_markers(text: str) -> str:
    """Remove leaked think/DSML/control markers from ordinary assistant text."""
    if not text:
        return ""
    cleaned = _LEAKED_THINK_MARKERS_RE.sub("", text)
    cleaned = _LEAKED_TOOL_MARKUP_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def looks_like_freeform_tool_garbage(text: str) -> bool:
    """True when assistant text looks like a failed free-form tool protocol dump."""
    if not text or not str(text).strip():
        return False
    sample = str(text)
    if _FREEFORM_TOOL_NOISE_RE.search(sample):
        return True
    # Dense think/DSML/tool markup after strip still leaves little usable prose.
    stripped = strip_leaked_thinking_markers(sample)
    if not stripped:
        return True
    marker_hits = len(_LEAKED_THINK_MARKERS_RE.findall(sample)) + len(
        _LEAKED_TOOL_MARKUP_RE.findall(sample)
    )
    if marker_hits >= 2 and len(stripped) < max(40, len(sample) // 4):
        return True
    # High angle-bracket density is almost never legitimate user-facing prose
    # for coding summaries (code fences are rare in agentMessage text).
    angle = sample.count("<") + sample.count(">")
    if angle >= 8 and angle / max(len(sample), 1) >= 0.04 and len(stripped) < len(sample) // 2:
        return True
    # Broken tool-XML leftovers: many unmatched angle brackets + punctuation soup.
    if angle >= 6 and re.search(r"[;{}=]{2,}", sample) and len(stripped) < max(60, len(sample) // 3):
        return True
    return False


def sanitize_assistant_visible_text(text: str) -> str:
    """Clean assistant text for history and user-facing summaries."""
    cleaned = strip_leaked_thinking_markers(text)
    if looks_like_freeform_tool_garbage(text):
        # Prefer a short diagnostic over multi-KB protocol soup.
        return _REDACTED_TOOL_GARBAGE
    return cleaned


__all__ = [
    "looks_like_freeform_tool_garbage",
    "sanitize_assistant_visible_text",
    "strip_leaked_thinking_markers",
]
