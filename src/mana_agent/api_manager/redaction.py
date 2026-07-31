"""API-specific structural redaction."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


SENSITIVE_NAMES = re.compile(
    r"(authorization|proxy-authorization|api[-_]?key|token|secret|password|passwd|cookie|set-cookie|client[-_]?secret)",
    re.IGNORECASE,
)


def is_sensitive_name(name: str) -> bool:
    return bool(SENSITIVE_NAMES.search(str(name or "")))


def redact_mapping(value: Any, *, secret_values: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if is_sensitive_name(str(key)) else redact_mapping(item, secret_values=secret_values)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_mapping(item, secret_values=secret_values) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_mapping(item, secret_values=secret_values) for item in value)
    text = str(value) if isinstance(value, str) else value
    if isinstance(text, str):
        for secret in sorted((item for item in secret_values if item), key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")
    return text


def redact_url(url: str, *, sensitive_query_names: tuple[str, ...] = ()) -> str:
    parsed = urlsplit(url)
    explicitly_sensitive = {name.lower() for name in sensitive_query_names}
    query = [
        (name, "[REDACTED]" if is_sensitive_name(name) or name.lower() in explicitly_sensitive else value)
        for name, value in parse_qsl(parsed.query, keep_blank_values=True)
    ]
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, urlencode(query, doseq=True), ""))

