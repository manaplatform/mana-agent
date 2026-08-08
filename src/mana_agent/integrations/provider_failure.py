"""Shared provider failure classification, backoff, and retry ownership.

Retry ownership contract (do not multiply retries across layers):

* HTTP connection / request establishment (DNS, connect timeout, refused):
  owned by the Responses bridge / provider transport only.
* Codex Responses stream reconnect after a successful HTTP 200 SSE start:
  owned by Codex ``stream_max_retries`` only when the disconnect is
  resumable and no side-effect boundary was crossed.
* Task-level recovery (checkpoint resume, replan, reassignment):
  owned by the Resilient Execution Supervisor only.
* Model / configuration / invalid-request failures:
  no automatic retry at any layer.

Nested multiplication such as Codex×Bridge×HTTP-client retries of the same
payload is forbidden. Layers below the owner return a classified failure and
stop.
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping
from urllib.parse import urlparse

from mana_agent.utils.redaction import redact_secrets

logger = logging.getLogger(__name__)

# Bounded provider body snippets for logs and diagnostics (4–8 KiB).
BODY_SNIPPET_LIMIT = 8 * 1024
DEFAULT_BASE_DELAY_SECONDS = 0.5
DEFAULT_MAX_DELAY_SECONDS = 30.0
DEFAULT_RETRY_AFTER_CEILING_SECONDS = 120.0
DEFAULT_CIRCUIT_FAILURE_THRESHOLD = 5
DEFAULT_CIRCUIT_OPEN_SECONDS = 30.0

_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]+")
_API_KEY_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|token|secret|password|credential)\s*[:=]\s*\S+"
)
_NVAPI_RE = re.compile(r"\bnvapi-[A-Za-z0-9._\-]+")
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9._\-]+")


class ProviderFailureKind(str, Enum):
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_RETIRED = "model_retired"
    RATE_LIMITED = "rate_limited"
    PROVIDER_OVERLOADED = "provider_overloaded"
    PROVIDER_ERROR = "provider_error"
    CONNECT_TIMEOUT = "connect_timeout"
    READ_TIMEOUT = "read_timeout"
    CONNECTION_RESET = "connection_reset"
    DNS_FAILURE = "dns_failure"
    STREAM_INTERRUPTED = "stream_interrupted"
    PROTOCOL_ERROR = "protocol_error"
    CANCELLED = "cancelled"
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN = "unknown"


class RetryOwner(str, Enum):
    """Which layer may retry for a given failure class."""

    NONE = "none"
    TRANSPORT = "transport"  # bridge / HTTP client connection establishment
    CODEX_STREAM = "codex_stream"  # Codex stream reconnect only
    SUPERVISOR = "supervisor"  # task-level recovery


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# Non-retryable HTTP statuses: identical payload retries waste quota.
_NON_RETRYABLE_HTTP: dict[int, ProviderFailureKind] = {
    400: ProviderFailureKind.INVALID_REQUEST,
    401: ProviderFailureKind.AUTHENTICATION,
    403: ProviderFailureKind.PERMISSION,
    404: ProviderFailureKind.MODEL_NOT_FOUND,
    410: ProviderFailureKind.MODEL_RETIRED,
    413: ProviderFailureKind.INVALID_REQUEST,
    422: ProviderFailureKind.INVALID_REQUEST,
}

# Transient HTTP statuses: retry only when replay is safe.
_RETRYABLE_HTTP: dict[int, ProviderFailureKind] = {
    408: ProviderFailureKind.READ_TIMEOUT,
    429: ProviderFailureKind.RATE_LIMITED,
    500: ProviderFailureKind.PROVIDER_ERROR,
    502: ProviderFailureKind.PROVIDER_ERROR,
    503: ProviderFailureKind.PROVIDER_OVERLOADED,
    504: ProviderFailureKind.PROVIDER_ERROR,
}


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    """Typed classification of a provider / bridge transport failure."""

    kind: ProviderFailureKind
    provider: str
    model: str = ""
    http_status: int | None = None
    provider_request_id: str | None = None
    retryable: bool = False
    retry_after: float | None = None
    safe_message: str = ""
    upstream_body_snippet: str = field(repr=False, default="")
    attempt: int = 1
    max_attempts: int = 1
    operation: str = ""
    endpoint: str = ""
    received_stream_data: bool = False
    tool_side_effects: bool = False
    retry_owner: RetryOwner = RetryOwner.NONE
    backoff_ms: int | None = None
    error_code: str = ""

    def with_attempt(self, attempt: int, *, max_attempts: int | None = None) -> "ProviderFailure":
        return replace(
            self,
            attempt=int(attempt),
            max_attempts=int(max_attempts if max_attempts is not None else self.max_attempts),
        )

    def with_backoff_ms(self, backoff_ms: int | None) -> "ProviderFailure":
        return replace(self, backoff_ms=None if backoff_ms is None else int(backoff_ms))

    def as_log_fields(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "failure_kind": self.kind.value,
            "http_status": self.http_status,
            "retryable": self.retryable,
            "backoff_ms": self.backoff_ms,
            "retry_after_ms": (
                int(self.retry_after * 1000) if self.retry_after is not None else None
            ),
            "received_stream_data": self.received_stream_data,
            "tool_side_effects": self.tool_side_effects,
            "retry_owner": self.retry_owner.value,
            "provider_request_id": self.provider_request_id,
            "error_code": self.error_code or None,
        }

    def user_status(self) -> str:
        """Human-readable status for Mana-controlled UI surfaces."""
        name = (self.provider or "provider").upper() if self.provider == "nvidia" else (
            self.provider or "Provider"
        )
        if self.provider and self.provider.lower() == "nvidia":
            name = "NVIDIA"
        elif self.provider:
            name = self.provider
        else:
            name = "Provider"

        if self.kind is ProviderFailureKind.CANCELLED:
            return f"{name} request cancelled."
        if self.kind is ProviderFailureKind.CIRCUIT_OPEN:
            return f"{name} temporarily unavailable (circuit open). Not probing yet."
        if not self.retryable:
            if self.kind is ProviderFailureKind.MODEL_RETIRED:
                return (
                    f"The selected {name} model has been retired"
                    f"{f' (HTTP {self.http_status})' if self.http_status else ''}. "
                    "Refresh the model catalog and choose another model."
                )
            if self.kind is ProviderFailureKind.AUTHENTICATION:
                return (
                    f"{name} authentication failed"
                    f"{f' (HTTP {self.http_status})' if self.http_status else ''}. Not retrying."
                )
            if self.kind is ProviderFailureKind.PERMISSION:
                return (
                    f"{name} denied permission"
                    f"{f' (HTTP {self.http_status})' if self.http_status else ''}. Not retrying."
                )
            if self.kind is ProviderFailureKind.MODEL_NOT_FOUND:
                return (
                    f"{name} model or endpoint not found"
                    f"{f' (HTTP {self.http_status})' if self.http_status else ''}. Not retrying."
                )
            if self.kind is ProviderFailureKind.INVALID_REQUEST:
                return (
                    f"{name} rejected the request"
                    f"{f' (HTTP {self.http_status})' if self.http_status else ''}. Not retrying."
                )
            return self.safe_message or f"{name} request failed. Not retrying."

        delay_s = None
        if self.backoff_ms is not None:
            delay_s = self.backoff_ms / 1000.0
        elif self.retry_after is not None:
            delay_s = self.retry_after
        attempt_part = f" ({self.attempt}/{self.max_attempts})"
        delay_part = f" in {delay_s:.1f}s" if delay_s is not None else ""
        if self.kind is ProviderFailureKind.RATE_LIMITED:
            return f"{name} rate limit reached. Retrying{delay_part}{attempt_part}."
        if self.kind is ProviderFailureKind.PROVIDER_OVERLOADED:
            status = f" (HTTP {self.http_status})" if self.http_status else ""
            return f"{name} temporarily unavailable{status}. Retrying{delay_part}{attempt_part}."
        if self.kind in {
            ProviderFailureKind.CONNECTION_RESET,
            ProviderFailureKind.DNS_FAILURE,
            ProviderFailureKind.CONNECT_TIMEOUT,
            ProviderFailureKind.STREAM_INTERRUPTED,
        }:
            return f"{name} connection interrupted. Retrying{delay_part}{attempt_part}."
        return f"{name} temporary failure. Retrying{delay_part}{attempt_part}."


def sanitize_provider_body(body: str | bytes | None, *, limit: int = BODY_SNIPPET_LIMIT) -> str:
    """Return a bounded, secret-redacted upstream body snippet."""
    if body is None:
        return ""
    if isinstance(body, bytes):
        text = body.decode("utf-8", errors="replace")
    else:
        text = str(body)
    text = redact_secrets(text) if isinstance(text, str) else str(text)
    if not isinstance(text, str):
        text = str(text)
    text = _BEARER_RE.sub("Bearer ***REDACTED***", text)
    text = _NVAPI_RE.sub("***REDACTED***", text)
    text = _SK_RE.sub("***REDACTED***", text)
    text = _API_KEY_RE.sub(lambda m: m.group(0).split("=")[0].split(":")[0] + "=***REDACTED***", text)
    # Collapse extreme whitespace for log friendliness while keeping structure.
    text = text.replace("\x00", "")
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def parse_retry_after(
    value: str | None,
    *,
    ceiling_seconds: float = DEFAULT_RETRY_AFTER_CEILING_SECONDS,
    now: float | None = None,
) -> float | None:
    """Parse Retry-After as delta-seconds or HTTP-date; clamp to ceiling."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    delay: float | None = None
    try:
        delay = float(raw)
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(raw)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        reference = now if now is not None else time.time()
        delay = parsed.timestamp() - reference
    if delay is None or delay < 0:
        return None
    return min(float(ceiling_seconds), float(delay))


def full_jitter_backoff_seconds(
    attempt: int,
    *,
    base_delay: float = DEFAULT_BASE_DELAY_SECONDS,
    max_delay: float = DEFAULT_MAX_DELAY_SECONDS,
    rng: random.Random | None = None,
) -> float:
    """Full-jitter exponential backoff.

    ``delay = uniform(0, min(max_delay, base_delay * 2**attempt))``

    ``attempt`` is 0-indexed for the first retry wait (attempt 0 → ~0–0.5s).
    """
    if attempt < 0:
        attempt = 0
    cap = min(float(max_delay), float(base_delay) * (2 ** attempt))
    if cap <= 0:
        return 0.0
    generator = rng or random
    return float(generator.uniform(0.0, cap))


def choose_backoff_seconds(
    failure: ProviderFailure,
    *,
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY_SECONDS,
    max_delay: float = DEFAULT_MAX_DELAY_SECONDS,
    retry_after_ceiling: float = DEFAULT_RETRY_AFTER_CEILING_SECONDS,
    rng: random.Random | None = None,
) -> float:
    """Prefer Retry-After for 429/503; otherwise full-jitter exponential backoff."""
    if failure.retry_after is not None and failure.kind in {
        ProviderFailureKind.RATE_LIMITED,
        ProviderFailureKind.PROVIDER_OVERLOADED,
    }:
        return min(float(retry_after_ceiling), max(0.0, float(failure.retry_after)))
    if failure.http_status in {429, 503} and failure.retry_after is not None:
        return min(float(retry_after_ceiling), max(0.0, float(failure.retry_after)))
    return full_jitter_backoff_seconds(
        attempt, base_delay=base_delay, max_delay=max_delay, rng=rng
    )


async def cancellation_aware_sleep(
    delay_seconds: float,
    *,
    cancel_event: asyncio.Event | None = None,
) -> bool:
    """Sleep with cancellation. Returns False if cancelled before completion."""
    delay = max(0.0, float(delay_seconds))
    if delay <= 0:
        return not (cancel_event is not None and cancel_event.is_set())
    if cancel_event is None:
        try:
            await asyncio.sleep(delay)
            return True
        except asyncio.CancelledError:
            raise
    try:
        await asyncio.wait_for(cancel_event.wait(), timeout=delay)
        return False
    except asyncio.TimeoutError:
        return not cancel_event.is_set()
    except asyncio.CancelledError:
        raise


def classify_http_status(
    status_code: int,
    *,
    provider: str,
    model: str = "",
    body: str | bytes | None = None,
    headers: Mapping[str, str] | None = None,
    operation: str = "",
    endpoint: str = "",
    attempt: int = 1,
    max_attempts: int = 1,
    received_stream_data: bool = False,
    tool_side_effects: bool = False,
    display_name: str | None = None,
) -> ProviderFailure:
    """Classify an upstream HTTP status into a typed failure."""
    status = int(status_code)
    headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    request_id = (
        headers.get("x-request-id")
        or headers.get("x-nvidia-request-id")
        or headers.get("request-id")
        or None
    )
    snippet = sanitize_provider_body(body)
    retry_after = parse_retry_after(headers.get("retry-after"))

    if status in _NON_RETRYABLE_HTTP:
        kind = _NON_RETRYABLE_HTTP[status]
        retryable = False
        owner = RetryOwner.NONE
    elif status in _RETRYABLE_HTTP:
        kind = _RETRYABLE_HTTP[status]
        # Mid-stream after side effects: transport may not safely replay.
        if received_stream_data or tool_side_effects:
            retryable = False
            owner = RetryOwner.SUPERVISOR if tool_side_effects else RetryOwner.CODEX_STREAM
            # Never mark tool-side-effect failures as automatically retryable.
            if tool_side_effects:
                retryable = False
                owner = RetryOwner.SUPERVISOR
            elif received_stream_data:
                # Partial stream without tool side effects may be stream-reconnect owned.
                retryable = True
                owner = RetryOwner.CODEX_STREAM
        else:
            retryable = True
            owner = RetryOwner.TRANSPORT
    elif 400 <= status < 500:
        kind = ProviderFailureKind.INVALID_REQUEST
        retryable = False
        owner = RetryOwner.NONE
    elif status >= 500:
        kind = ProviderFailureKind.PROVIDER_ERROR
        retryable = not tool_side_effects
        owner = RetryOwner.TRANSPORT if retryable else RetryOwner.SUPERVISOR
    else:
        kind = ProviderFailureKind.UNKNOWN
        retryable = False
        owner = RetryOwner.NONE

    # Never auto-retry after tools executed; side effects must not duplicate.
    if tool_side_effects:
        retryable = False
        owner = RetryOwner.SUPERVISOR

    name = display_name or provider or "provider"
    safe_message = _safe_http_message(
        name=name,
        kind=kind,
        status=status,
        model=model,
        snippet=snippet,
    )
    return ProviderFailure(
        kind=kind,
        provider=str(provider or "unknown"),
        model=str(model or ""),
        http_status=status,
        provider_request_id=request_id,
        retryable=retryable,
        retry_after=retry_after,
        safe_message=safe_message,
        upstream_body_snippet=snippet,
        attempt=attempt,
        max_attempts=max_attempts,
        operation=operation,
        endpoint=endpoint,
        received_stream_data=received_stream_data,
        tool_side_effects=tool_side_effects,
        retry_owner=owner,
        error_code=f"upstream_{kind.value}",
    )


def classify_transport_exception(
    exc: BaseException,
    *,
    provider: str,
    model: str = "",
    operation: str = "",
    endpoint: str = "",
    attempt: int = 1,
    max_attempts: int = 1,
    received_stream_data: bool = False,
    tool_side_effects: bool = False,
    display_name: str | None = None,
) -> ProviderFailure:
    """Classify network / timeout / protocol exceptions."""
    name = display_name or provider or "provider"
    type_name = type(exc).__name__
    message = str(exc or type_name)
    lowered = f"{type_name} {message}".lower()

    if isinstance(exc, asyncio.CancelledError) or "cancelled" in lowered:
        kind = ProviderFailureKind.CANCELLED
        retryable = False
        owner = RetryOwner.NONE
        safe = f"{name} request cancelled."
    elif "timeout" in lowered and ("connect" in lowered or "connection" in lowered):
        kind = ProviderFailureKind.CONNECT_TIMEOUT
        retryable = not tool_side_effects
        owner = RetryOwner.TRANSPORT if retryable else RetryOwner.SUPERVISOR
        safe = f"{name} connection timed out."
    elif "timeout" in lowered or "timed out" in lowered or "readtimeout" in type_name.lower():
        kind = ProviderFailureKind.READ_TIMEOUT
        retryable = not tool_side_effects and not received_stream_data
        if tool_side_effects:
            owner = RetryOwner.SUPERVISOR
        elif received_stream_data:
            retryable = True
            owner = RetryOwner.CODEX_STREAM
        else:
            owner = RetryOwner.TRANSPORT
        safe = f"{name} request timed out."
    elif any(token in lowered for token in ("name or service not known", "nodename nor servname", "getaddrinfo", "dns")):
        kind = ProviderFailureKind.DNS_FAILURE
        retryable = not tool_side_effects
        owner = RetryOwner.TRANSPORT if retryable else RetryOwner.SUPERVISOR
        safe = f"{name} DNS resolution failed."
    elif any(
        token in lowered
        for token in (
            "connection reset",
            "connection aborted",
            "broken pipe",
            "remoteprotocolerror",
            "server disconnected",
            "incomplete chunked read",
            "connection refused",
        )
    ):
        kind = ProviderFailureKind.CONNECTION_RESET
        if tool_side_effects:
            retryable = False
            owner = RetryOwner.SUPERVISOR
        elif received_stream_data:
            retryable = True
            owner = RetryOwner.CODEX_STREAM
        else:
            retryable = True
            owner = RetryOwner.TRANSPORT
        safe = f"{name} connection was reset."
    elif "protocol" in lowered or "json" in lowered:
        kind = ProviderFailureKind.PROTOCOL_ERROR
        retryable = False
        owner = RetryOwner.NONE
        safe = f"{name} returned a protocol error."
    else:
        kind = ProviderFailureKind.CONNECTION_RESET
        retryable = not tool_side_effects and not received_stream_data
        owner = RetryOwner.TRANSPORT if retryable else (
            RetryOwner.SUPERVISOR if tool_side_effects else RetryOwner.CODEX_STREAM
        )
        if received_stream_data and not tool_side_effects:
            kind = ProviderFailureKind.STREAM_INTERRUPTED
            retryable = True
            owner = RetryOwner.CODEX_STREAM
        safe = f"{name} network error."

    if tool_side_effects:
        retryable = False
        owner = RetryOwner.SUPERVISOR

    return ProviderFailure(
        kind=kind,
        provider=str(provider or "unknown"),
        model=str(model or ""),
        http_status=None,
        retryable=retryable,
        safe_message=safe,
        upstream_body_snippet=sanitize_provider_body(message, limit=512),
        attempt=attempt,
        max_attempts=max_attempts,
        operation=operation,
        endpoint=endpoint,
        received_stream_data=received_stream_data,
        tool_side_effects=tool_side_effects,
        retry_owner=owner,
        error_code=f"upstream_{kind.value}",
    )


def classify_stream_interrupt(
    *,
    provider: str,
    model: str = "",
    received_stream_data: bool,
    tool_side_effects: bool,
    operation: str = "chat_completion_stream",
    attempt: int = 1,
    max_attempts: int = 1,
    display_name: str | None = None,
    detail: str = "",
) -> ProviderFailure:
    """Classify an unexpected SSE/socket close after HTTP 200 was accepted."""
    name = display_name or provider or "provider"
    if tool_side_effects:
        return ProviderFailure(
            kind=ProviderFailureKind.STREAM_INTERRUPTED,
            provider=str(provider or "unknown"),
            model=str(model or ""),
            retryable=False,
            safe_message=(
                f"{name} stream interrupted after tool side effects. "
                "Automatic whole-turn replay is disabled to avoid duplicate actions."
            ),
            upstream_body_snippet=sanitize_provider_body(detail, limit=512),
            attempt=attempt,
            max_attempts=max_attempts,
            operation=operation,
            received_stream_data=received_stream_data,
            tool_side_effects=True,
            retry_owner=RetryOwner.SUPERVISOR,
            error_code="upstream_stream_interrupted",
        )
    if received_stream_data:
        return ProviderFailure(
            kind=ProviderFailureKind.STREAM_INTERRUPTED,
            provider=str(provider or "unknown"),
            model=str(model or ""),
            retryable=True,
            safe_message=f"{name} stream interrupted after partial response.",
            upstream_body_snippet=sanitize_provider_body(detail, limit=512),
            attempt=attempt,
            max_attempts=max_attempts,
            operation=operation,
            received_stream_data=True,
            tool_side_effects=False,
            retry_owner=RetryOwner.CODEX_STREAM,
            error_code="upstream_stream_interrupted",
        )
    return ProviderFailure(
        kind=ProviderFailureKind.STREAM_INTERRUPTED,
        provider=str(provider or "unknown"),
        model=str(model or ""),
        retryable=True,
        safe_message=f"{name} stream disconnected before any data.",
        upstream_body_snippet=sanitize_provider_body(detail, limit=512),
        attempt=attempt,
        max_attempts=max_attempts,
        operation=operation,
        received_stream_data=False,
        tool_side_effects=False,
        retry_owner=RetryOwner.TRANSPORT,
        error_code="upstream_stream_interrupted",
    )


def _safe_http_message(
    *,
    name: str,
    kind: ProviderFailureKind,
    status: int,
    model: str,
    snippet: str,
) -> str:
    model_part = f" model={model}" if model else ""
    base = {
        ProviderFailureKind.AUTHENTICATION: (
            f"{name} authentication failed (HTTP {status}).{model_part}"
        ),
        ProviderFailureKind.PERMISSION: (
            f"{name} permission denied (HTTP {status}).{model_part}"
        ),
        ProviderFailureKind.MODEL_NOT_FOUND: (
            f"{name} model or endpoint not found (HTTP {status}).{model_part}"
        ),
        ProviderFailureKind.MODEL_RETIRED: (
            f"{name} model has been retired (HTTP {status}).{model_part} "
            "Refresh the model catalog and choose another model."
        ),
        ProviderFailureKind.INVALID_REQUEST: (
            f"{name} rejected the request (HTTP {status}).{model_part}"
        ),
        ProviderFailureKind.RATE_LIMITED: (
            f"{name} rate limit or quota exceeded (HTTP {status}).{model_part}"
        ),
        ProviderFailureKind.PROVIDER_OVERLOADED: (
            f"{name} temporarily unavailable (HTTP {status}).{model_part}"
        ),
        ProviderFailureKind.PROVIDER_ERROR: (
            f"{name} service failure (HTTP {status}).{model_part}"
        ),
        ProviderFailureKind.READ_TIMEOUT: (
            f"{name} request timed out (HTTP {status}).{model_part}"
        ),
    }.get(kind, f"{name} request failed (HTTP {status}).{model_part}")

    # Preserve provider validation diagnostics (sanitized) for non-retryable cases.
    if snippet and kind in {
        ProviderFailureKind.INVALID_REQUEST,
        ProviderFailureKind.MODEL_NOT_FOUND,
        ProviderFailureKind.MODEL_RETIRED,
        ProviderFailureKind.AUTHENTICATION,
        ProviderFailureKind.PERMISSION,
    }:
        # Prefer a compact single-line diagnostic.
        compact = " ".join(snippet.split())
        if len(compact) > 600:
            compact = compact[:597] + "..."
        return f"{base} Diagnostic: {compact}"
    return base


def log_provider_failure(failure: ProviderFailure, *, level: int = logging.ERROR) -> None:
    fields = failure.as_log_fields()
    logger.log(
        level,
        "provider_failure provider=%s model=%s operation=%s attempt=%s max_attempts=%s "
        "failure_kind=%s http_status=%s retryable=%s backoff_ms=%s retry_after_ms=%s "
        "received_stream_data=%s tool_side_effects=%s retry_owner=%s "
        "provider_request_id=%s body_snippet=%r",
        fields["provider"],
        fields["model"],
        fields["operation"],
        fields["attempt"],
        fields["max_attempts"],
        fields["failure_kind"],
        fields["http_status"],
        fields["retryable"],
        fields["backoff_ms"],
        fields["retry_after_ms"],
        fields["received_stream_data"],
        fields["tool_side_effects"],
        fields["retry_owner"],
        fields["provider_request_id"],
        failure.upstream_body_snippet[:512] if failure.upstream_body_snippet else "",
    )


def circuit_scope_key(*, provider: str, endpoint: str) -> str:
    """Scope circuit breakers to provider + endpoint, never globally."""
    host = ""
    try:
        host = urlparse(endpoint).netloc or endpoint
    except Exception:
        host = endpoint or ""
    return f"{str(provider or 'unknown').strip().lower()}|{str(host).strip().lower()}"


class ProviderCircuitBreaker:
    """Per provider+endpoint circuit breaker for *transient* failures only.

    Configuration/request failures (400/401/403/404/410/422) must not open the
    circuit — those require config changes, not a health cooldown.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = DEFAULT_CIRCUIT_FAILURE_THRESHOLD,
        open_seconds: float = DEFAULT_CIRCUIT_OPEN_SECONDS,
        half_open_max_probes: int = 1,
        clock: Any = None,
    ) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.open_seconds = max(0.1, float(open_seconds))
        self.half_open_max_probes = max(1, int(half_open_max_probes))
        self._clock = clock or time.monotonic
        self._states: dict[str, dict[str, Any]] = {}

    def _entry(self, key: str) -> dict[str, Any]:
        entry = self._states.get(key)
        if entry is None:
            entry = {
                "state": CircuitState.CLOSED,
                "failure_count": 0,
                "opened_at": None,
                "half_open_probes": 0,
            }
            self._states[key] = entry
        return entry

    def state(self, key: str) -> CircuitState:
        entry = self._entry(key)
        self._maybe_transition(entry)
        return entry["state"]

    def allow_request(self, key: str) -> bool:
        entry = self._entry(key)
        self._maybe_transition(entry)
        state = entry["state"]
        if state is CircuitState.CLOSED:
            return True
        if state is CircuitState.OPEN:
            return False
        # half-open: permit limited probes
        if entry["half_open_probes"] < self.half_open_max_probes:
            entry["half_open_probes"] += 1
            return True
        return False

    def record_success(self, key: str) -> CircuitState:
        entry = self._entry(key)
        entry["state"] = CircuitState.CLOSED
        entry["failure_count"] = 0
        entry["opened_at"] = None
        entry["half_open_probes"] = 0
        return entry["state"]

    def record_failure(self, key: str, failure: ProviderFailure) -> CircuitState:
        """Record only transient provider-health failures."""
        if not self._counts_toward_circuit(failure):
            return self.state(key)
        entry = self._entry(key)
        if entry["state"] is CircuitState.HALF_OPEN:
            entry["state"] = CircuitState.OPEN
            entry["opened_at"] = self._clock()
            entry["half_open_probes"] = 0
            entry["failure_count"] = self.failure_threshold
            return entry["state"]
        entry["failure_count"] = int(entry["failure_count"]) + 1
        if entry["failure_count"] >= self.failure_threshold:
            entry["state"] = CircuitState.OPEN
            entry["opened_at"] = self._clock()
        return entry["state"]

    def _maybe_transition(self, entry: dict[str, Any]) -> None:
        if entry["state"] is not CircuitState.OPEN:
            return
        opened_at = entry.get("opened_at")
        if opened_at is None:
            return
        if self._clock() - float(opened_at) >= self.open_seconds:
            entry["state"] = CircuitState.HALF_OPEN
            entry["half_open_probes"] = 0

    @staticmethod
    def _counts_toward_circuit(failure: ProviderFailure) -> bool:
        if failure.kind in {
            ProviderFailureKind.INVALID_REQUEST,
            ProviderFailureKind.AUTHENTICATION,
            ProviderFailureKind.PERMISSION,
            ProviderFailureKind.MODEL_NOT_FOUND,
            ProviderFailureKind.MODEL_RETIRED,
            ProviderFailureKind.CANCELLED,
            ProviderFailureKind.PROTOCOL_ERROR,
        }:
            return False
        if failure.http_status in {400, 401, 403, 404, 410, 413, 422}:
            return False
        return failure.retryable or failure.kind in {
            ProviderFailureKind.RATE_LIMITED,
            ProviderFailureKind.PROVIDER_OVERLOADED,
            ProviderFailureKind.PROVIDER_ERROR,
            ProviderFailureKind.CONNECT_TIMEOUT,
            ProviderFailureKind.READ_TIMEOUT,
            ProviderFailureKind.CONNECTION_RESET,
            ProviderFailureKind.DNS_FAILURE,
            ProviderFailureKind.STREAM_INTERRUPTED,
        }


# Process-wide breaker for bridge/transport scopes.
PROVIDER_CIRCUIT_BREAKER = ProviderCircuitBreaker()


__all__ = [
    "BODY_SNIPPET_LIMIT",
    "DEFAULT_BASE_DELAY_SECONDS",
    "DEFAULT_MAX_DELAY_SECONDS",
    "PROVIDER_CIRCUIT_BREAKER",
    "CircuitState",
    "ProviderCircuitBreaker",
    "ProviderFailure",
    "ProviderFailureKind",
    "RetryOwner",
    "cancellation_aware_sleep",
    "choose_backoff_seconds",
    "circuit_scope_key",
    "classify_http_status",
    "classify_stream_interrupt",
    "classify_transport_exception",
    "full_jitter_backoff_seconds",
    "log_provider_failure",
    "parse_retry_after",
    "sanitize_provider_body",
]
