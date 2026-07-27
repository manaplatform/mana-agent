from __future__ import annotations

import re
from typing import Any

from mana_agent.utils.redaction import REDACTED, redact_secrets

_SECRET_KEYS = re.compile(r"(?i)^(?:secret|token|credential|private[_-]?key|authorization|raw[_-]?value|secret[_-]?value|detected[_-]?secret)$")
_SECRET_PATTERNS = (
    re.compile(r"\b(?:ghp|github_pat|ghs|ghu)_[A-Za-z0-9_]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(?:password|secret|token)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)\b[A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PRIVATE_KEY|API_KEY)[A-Z0-9_]*\s*=\s*[^\s]+"),
)


def sanitize_event_context(value: Any, *, secret_alert: bool = False) -> Any:
    """Bound and redact untrusted webhook data before durable storage or prompts."""
    value = redact_secrets(value)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if _SECRET_KEYS.search(str(key)):
                result[str(key)] = REDACTED
            else:
                result[str(key)] = sanitize_event_context(item, secret_alert=secret_alert)
        return result
    if isinstance(value, list):
        return [sanitize_event_context(item, secret_alert=secret_alert) for item in value[:100]]
    if isinstance(value, str):
        text = value[:20_000]
        for pattern in _SECRET_PATTERNS:
            text = pattern.sub(REDACTED, text)
        return text
    return value
