from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

_SECRET_KEY = re.compile(
    r"(?i)(api[_-]?key|authorization|cookie|password|passwd|secret|private[_-]?key|(?:access|refresh|oauth|github)?[_-]?token)$"
)
_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bgh[opusr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]+=*"),
    re.compile(r"(?i)(https?://[^:/\s]+:)[^@\s]+@"),
    re.compile(r"(?i)((?:password|passwd|api[_-]?key|access[_-]?token|refresh[_-]?token|oauth[_-]?token|github[_-]?token)\s*(?:=|:|\s)\s*)[^\s,;]+"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def secret_hash(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def redact_text(value: str, extra_patterns: Iterable[str] = ()) -> str:
    text = str(value)
    for pattern in (*_PATTERNS, *(re.compile(item) for item in extra_patterns)):
        text = pattern.sub(lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]", text)
    return text


def redact(value: Any, extra_patterns: Iterable[str] = ()) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _SECRET_KEY.search(str(key)) else redact(item, extra_patterns)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [redact(item, extra_patterns) for item in value]
    if isinstance(value, str):
        return redact_text(value, extra_patterns)
    return value


def redacted_json(value: Any, extra_patterns: Iterable[str] = ()) -> str:
    return json.dumps(redact(value, extra_patterns), sort_keys=True, default=str)
