"""Fail-closed protocol configuration validation."""

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlsplit


def validate_protocol_configuration(values: dict[str, Any]) -> None:
    host = str(values.get("MANA_A2A_HOST") or "127.0.0.1")
    public_url = str(values.get("MANA_A2A_PUBLIC_BASE_URL") or "")
    enabled = bool(values.get("MANA_A2A_SERVER_ENABLED", False))
    if enabled and not str(values.get("MANA_A2A_SERVER_TOKEN") or "").strip():
        raise ValueError("A2A server requires a bearer token or secret reference.")
    try:
        local_host = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        local_host = False
    if enabled and not local_host:
        parsed = urlsplit(public_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("Non-local A2A servers require an HTTPS public base URL.")
    if bool(values.get("MANA_A2A_PUSH_NOTIFICATIONS", False)):
        raise ValueError("A2A push notifications are not implemented and cannot be enabled.")
    for key in (
        "MANA_A2A_PORT",
        "MANA_A2A_TASK_RETENTION_DAYS",
        "MANA_A2A_MAX_REQUEST_BYTES",
        "MANA_A2A_MAX_ARTIFACT_BYTES",
        "MANA_A2A_MAX_CONCURRENT_TASKS",
        "MANA_A2A_MAX_DELEGATION_DEPTH",
        "MANA_ACP_SESSION_RETENTION_DAYS",
    ):
        if int(values.get(key) or 0) <= 0:
            raise ValueError(f"{key} must be greater than zero.")
