"""Centralized secret removal for doctor output and exception details."""
from __future__ import annotations

import re
from typing import Any

_PATTERNS = (
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(bearer\s+)[^\s,]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)((?:api[_-]?key|token|password|secret|client_secret)\s*[:=]\s*)[^\s,]+"), r"\1[REDACTED]"),
    (re.compile(r"([?&](?:api[_-]?key|token|access_token|key)=)[^&#\s]+", re.I), r"\1[REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"), "[REDACTED]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{12,}\b", re.I), "[REDACTED]"),
)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): redact("[REDACTED]" if _is_secret_key(str(key)) else item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    if not isinstance(value, str):
        return value
    for pattern, replacement in _PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def _is_secret_key(key: str) -> bool:
    return any(part in key.lower() for part in ("token", "secret", "password", "api_key", "credential", "authorization"))
