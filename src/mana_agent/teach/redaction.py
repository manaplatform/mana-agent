"""Conservative secret and personal-data scanning."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "token": re.compile(r"\b(?:sk|ghp|github_pat|xox[baprs])[-_A-Za-z0-9]{16,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "home_path": re.compile(r"(?:/Users/|/home/|[A-Z]:\\\\Users\\\\)[^/\\\\\s]+", re.I),
    "cookie": re.compile(r"\b(?:session|cookie|csrf)[_-]?(?:id|token)?\s*[:=]\s*\S+", re.I),
}


class Redactor:
    def scan(self, value: Any) -> list[str]:
        findings: set[str] = set()
        for text in _strings(value):
            for name, pattern in PATTERNS.items():
                if pattern.search(text):
                    findings.add(name)
            if "password" in text.lower() or "secret" in text.lower():
                findings.add("secret_field")
        return sorted(findings)

    def redact(self, value: Any) -> tuple[Any, list[str]]:
        findings = self.scan(value)
        return _redact_value(value), findings


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if any(marker in str(key).lower() for marker in ("password", "secret", "token", "cookie", "card")):
                output[key] = "{{ secret }}"
            else:
                output[key] = _redact_value(item)
        return output
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        text = value
        for name, pattern in PATTERNS.items():
            replacement = "{{ email }}" if name == "email" else f"{{{{ redacted_{name} }}}}"
            text = pattern.sub(replacement, text)
        return text
    return value


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, Path):
        yield str(value)
