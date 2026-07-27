"""
mana_agent.utils.redaction

Helpers for stripping secrets out of payloads before they are logged.

Secrets must never reach log files or stderr. Worker init payloads in
particular carry the OpenAI ``api_key``, so every structured payload that is
logged should be passed through :func:`redact_secrets` first, and any raw JSON
line should go through :func:`redact_json_line`.
"""

from __future__ import annotations

import json
import re
from typing import Any

REDACTED = "***REDACTED***"

# Keys whose values must always be redacted (compared case-insensitively).
SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "authorization",
        "token",
        "access_token",
        "refresh_token",
        "password",
        "secret",
        "cookie",
        "set-cookie",
    }
)


def _is_sensitive_key(key: Any) -> bool:
    return isinstance(key, str) and key.strip().lower() in SENSITIVE_KEYS


def redact_secrets(value: Any) -> Any:
    """Recursively redact sensitive values in dicts/lists.

    - For dicts, any key matching :data:`SENSITIVE_KEYS` (case-insensitively)
      has its value replaced with ``***REDACTED***``; remaining values are
      redacted recursively.
    - For lists/tuples, each element is redacted recursively.
    - String values also redact bearer credentials and OpenAI-style keys.
    - All other values are returned unchanged.

    The input is never mutated; a redacted copy is returned.
    """
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(key):
                redacted[key] = REDACTED
            else:
                redacted[key] = redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    if isinstance(value, str):
        redacted = _BEARER_RE.sub(f"Bearer {REDACTED}", value)
        return _OPENAI_KEY_RE.sub(REDACTED, redacted)
    return value


# Matches "<sensitive_key>": "<value>" inside a JSON-ish string.
_JSON_KV_RE = re.compile(
    r'("(?:' + "|".join(re.escape(k) for k in sorted(SENSITIVE_KEYS)) + r')"\s*:\s*)"[^"]*"',
    re.IGNORECASE,
)
# Common bearer-token / OpenAI-style key shapes that may appear in free text.
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]+")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9._\-]+")


def redact_json_line(line: str) -> str:
    """Redact secrets from a raw (possibly JSON) log line.

    Tries to parse the line as JSON and redact it structurally; if that fails
    (partial line, non-JSON text) falls back to regex-based redaction so
    secrets are still scrubbed.
    """
    text = line if isinstance(line, str) else str(line)
    stripped = text.strip()
    if stripped and stripped[0] in "{[":
        try:
            parsed = json.loads(stripped)
        except (ValueError, TypeError):
            parsed = None
        if parsed is not None:
            return json.dumps(redact_secrets(parsed), ensure_ascii=False)

    redacted = _JSON_KV_RE.sub(lambda m: f'{m.group(1)}"{REDACTED}"', text)
    redacted = _BEARER_RE.sub(f"Bearer {REDACTED}", redacted)
    redacted = _OPENAI_KEY_RE.sub(REDACTED, redacted)
    return redacted
