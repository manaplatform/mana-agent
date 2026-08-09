"""Shared types for the Mana Responses compatibility bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mana_agent.integrations.provider_failure import ProviderFailure, ProviderFailureKind


class ResponsesBridgeError(RuntimeError):
    """Protocol conversion or local bridge failure (not an upstream provider error)."""

    def __init__(self, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = int(status_code)


class UpstreamProviderError(RuntimeError):
    """Error attributed to the upstream Chat Completions provider.

    Always carries a typed :class:`ProviderFailure` so callers can apply the
    correct recovery policy instead of treating every failure as a reconnectable
    stream disconnect.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        error_kind: str = "provider_error",
        failure: ProviderFailure | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = str(provider or "unknown")
        self.status_code = status_code
        self.error_kind = str(error_kind or "provider_error")
        if failure is not None:
            self.failure = failure
            self.status_code = failure.http_status if failure.http_status is not None else status_code
            self.error_kind = failure.kind.value
            self.provider = failure.provider or self.provider
        else:
            try:
                kind = ProviderFailureKind(self.error_kind)
            except ValueError:
                kind = ProviderFailureKind.PROVIDER_ERROR
            self.failure = ProviderFailure(
                kind=kind,
                provider=self.provider,
                http_status=self.status_code,
                retryable=False,
                safe_message=str(message),
                error_code=f"upstream_{kind.value}",
            )

    @property
    def retryable(self) -> bool:
        return bool(self.failure.retryable)

    @property
    def kind(self) -> ProviderFailureKind:
        return self.failure.kind


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
    # Limit only the period before an upstream streaming response is accepted.
    # A provider that never sends response headers otherwise leaves the Codex
    # turn in a misleading "started" state for the full stream timeout.
    stream_open_timeout_seconds: float = 45.0
    # Bridge owns at most one transport attempt per Codex request. Nested retries
    # (Codex × bridge × HTTP client) are forbidden — see provider_failure module.
    transport_max_attempts: int = 1

    def public_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "display_name": self.display_name,
            "base_url": self.base_url,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "stream_open_timeout_seconds": self.stream_open_timeout_seconds,
            "has_request_overrides": bool(self.request_overrides),
            "transport_max_attempts": self.transport_max_attempts,
        }


__all__ = [
    "BridgeUpstreamConfig",
    "ResponsesBridgeError",
    "UpstreamProviderError",
]
