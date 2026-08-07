"""Safe embedding of dynamic data into same-origin dashboard HTML documents."""

from __future__ import annotations

import html
import json
import re
from typing import Any


# Conversation / surface identifiers accepted for dashboard HTML documents.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:@-]{1,160}$")


def require_safe_id(value: str, *, field: str = "id") -> str:
    """Reject identifiers that cannot be safely reflected into HTML/JS config."""
    text = str(value or "").strip()
    if not text or not _SAFE_ID_RE.fullmatch(text):
        raise ValueError(f"Invalid {field}.")
    return text


def script_json(data: Any) -> str:
    """Serialize data for embedding inside a ``<script>`` element.

    Uses JSON encoding plus Unicode escapes for HTML-sensitive characters so a
    reflected payload cannot break out of the script context.
    """
    # Default separators preserve spaces so existing tests and logs stay readable.
    payload = json.dumps(data, ensure_ascii=True)
    return (
        payload.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def html_text(value: str) -> str:
    """Escape a string for HTML text or attribute context."""
    return html.escape(str(value or ""), quote=True)


__all__ = ["html_text", "require_safe_id", "script_json"]
