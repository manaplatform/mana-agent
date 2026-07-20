"""Deterministic validation after a model-selected protocol action."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

from mana_agent.utils.redaction import redact_secrets

from .exceptions import ProtocolPolicyError


@dataclass(frozen=True, slots=True)
class ProtocolSecurityPolicy:
    workspace_roots: tuple[Path, ...]
    allow_local_urls: bool = False
    max_request_bytes: int = 1_048_576
    max_artifact_bytes: int = 10_485_760
    allowed_content_types: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"text/plain", "text/markdown", "application/json", "text/x-diff"}
        )
    )

    @classmethod
    def for_workspace(
        cls,
        root: str | Path,
        additional_roots: Iterable[str | Path] = (),
        **kwargs: Any,
    ) -> "ProtocolSecurityPolicy":
        roots = tuple(dict.fromkeys(Path(item).expanduser().resolve() for item in (root, *additional_roots)))
        return cls(workspace_roots=roots, **kwargs)

    def validate_path(self, value: str | Path) -> Path:
        raw = Path(value).expanduser()
        if not raw.is_absolute():
            raise ProtocolPolicyError("Protocol paths must be absolute.")
        resolved = raw.resolve()
        if not any(resolved == root or root in resolved.parents for root in self.workspace_roots):
            raise ProtocolPolicyError("Path is outside the approved workspace roots.")
        return resolved

    def validate_size(self, size: int, *, artifact: bool = False) -> None:
        limit = self.max_artifact_bytes if artifact else self.max_request_bytes
        if int(size) > limit:
            kind = "artifact" if artifact else "request"
            raise ProtocolPolicyError(f"Protocol {kind} exceeds the configured size limit.")

    def validate_content_type(self, value: str) -> str:
        content_type = str(value or "").partition(";")[0].strip().lower()
        if content_type not in self.allowed_content_types:
            raise ProtocolPolicyError("Artifact content type is not allowed.")
        return content_type

    def validate_remote_url(self, value: str, *, production: bool = True) -> str:
        parsed = urlsplit(str(value or ""))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ProtocolPolicyError("Remote URL must be an HTTP(S) URL without embedded credentials.")
        if production and parsed.scheme != "https" and not self.allow_local_urls:
            raise ProtocolPolicyError("Remote A2A endpoints require HTTPS.")
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
        except socket.gaierror as exc:
            raise ProtocolPolicyError("Remote endpoint hostname could not be resolved.") from exc
        for address in addresses:
            ip = ipaddress.ip_address(address)
            unsafe = ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
            if unsafe and not self.allow_local_urls:
                raise ProtocolPolicyError("Remote endpoint resolves to a non-public address.")
        return parsed.geturl()


def redact_protocol_value(value: Any) -> Any:
    """Redact secrets before values reach logs, events, or task metadata."""
    return redact_secrets(value)
