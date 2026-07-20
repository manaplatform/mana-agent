"""Authentication helpers shared by network protocol adapters."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from .exceptions import ProtocolAuthenticationError


@dataclass(frozen=True, slots=True)
class CallerIdentity:
    caller_id: str
    authenticated: bool = True
    tenant: str = ""


def require_bearer_token(authorization: str | None, expected_token: str) -> CallerIdentity:
    expected = str(expected_token or "")
    if not expected:
        raise ProtocolAuthenticationError("A2A server authentication is not configured.")
    scheme, _, supplied = str(authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not supplied or not hmac.compare_digest(supplied, expected):
        raise ProtocolAuthenticationError("Invalid or missing bearer token.")
    digest = hashlib.sha256(supplied.encode("utf-8")).hexdigest()[:20]
    return CallerIdentity(caller_id=f"bearer:{digest}")
