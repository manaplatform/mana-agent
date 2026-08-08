"""Shared types for the Mana Responses compatibility bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ResponsesBridgeError(RuntimeError):
    """Protocol conversion or local bridge failure (not an upstream provider error)."""

    def __init__(self, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = int(status_code)


class UpstreamProviderError(RuntimeError):
    """Error attributed to the upstream Chat Completions provider."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        error_kind: str = "provider_error",
    ) -> None:
        super().__init__(message)
        self.provider = str(provider or "unknown")
        self.status_code = status_code
        self.error_kind = str(error_kind or "provider_error")


@dataclass(frozen=True, slots=True)
class BridgeUpstreamConfig:
    """Upstream Chat Completions target used by one bridge instance."""

    provider: str
    display_name: str
    api_key: str = field(repr=False)
    base_url: str
    model: str
    headers: dict[str, str] = field(default_factory=dict)
    request_overrides: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 600.0

    def public_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "display_name": self.display_name,
            "base_url": self.base_url,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "has_request_overrides": bool(self.request_overrides),
        }


__all__ = [
    "BridgeUpstreamConfig",
    "ResponsesBridgeError",
    "UpstreamProviderError",
]
