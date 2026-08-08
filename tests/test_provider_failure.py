"""Unit tests for shared provider failure classification and backoff."""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone

import pytest

from mana_agent.integrations.provider_failure import (
    ProviderCircuitBreaker,
    ProviderFailure,
    ProviderFailureKind,
    RetryOwner,
    cancellation_aware_sleep,
    choose_backoff_seconds,
    classify_http_status,
    classify_stream_interrupt,
    classify_transport_exception,
    full_jitter_backoff_seconds,
    parse_retry_after,
    sanitize_provider_body,
)


def test_http_400_is_invalid_request_not_retryable() -> None:
    failure = classify_http_status(
        400,
        provider="nvidia",
        model="deepseek-ai/deepseek-v4-flash",
        body='{"error":{"message":"messages sequence invalid"}}',
        operation="chat_completion_stream",
    )
    assert failure.kind is ProviderFailureKind.INVALID_REQUEST
    assert failure.retryable is False
    assert failure.retry_owner is RetryOwner.NONE
    assert failure.http_status == 400
    assert "messages sequence invalid" in failure.safe_message
    assert "Bearer" not in failure.upstream_body_snippet or "***REDACTED***" in failure.upstream_body_snippet
    assert "Reconnecting" not in failure.user_status()
    assert "Not retrying" in failure.user_status()


def test_http_401_403_404_410_422_never_retry() -> None:
    cases = {
        401: ProviderFailureKind.AUTHENTICATION,
        403: ProviderFailureKind.PERMISSION,
        404: ProviderFailureKind.MODEL_NOT_FOUND,
        410: ProviderFailureKind.MODEL_RETIRED,
        422: ProviderFailureKind.INVALID_REQUEST,
    }
    for status, kind in cases.items():
        failure = classify_http_status(status, provider="nvidia", model="m")
        assert failure.kind is kind
        assert failure.retryable is False
        assert failure.retry_owner is RetryOwner.NONE


def test_http_429_retry_after_seconds() -> None:
    failure = classify_http_status(
        429,
        provider="nvidia",
        headers={"Retry-After": "8", "x-request-id": "req-1"},
    )
    assert failure.kind is ProviderFailureKind.RATE_LIMITED
    assert failure.retryable is True
    assert failure.retry_after == 8.0
    assert failure.provider_request_id == "req-1"
    delay = choose_backoff_seconds(failure, attempt=0)
    assert delay == 8.0


def test_retry_after_http_date() -> None:
    future = datetime(2030, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    # 10 seconds after the reference epoch used below.
    header = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    delay = parse_retry_after(header, now=future.timestamp() - 10)
    assert delay == pytest.approx(10.0, abs=0.01)


def test_http_503_uses_jitter_when_no_retry_after() -> None:
    failure = classify_http_status(503, provider="nvidia")
    assert failure.kind is ProviderFailureKind.PROVIDER_OVERLOADED
    assert failure.retryable is True
    rng = random.Random(0)
    delays = {choose_backoff_seconds(failure, attempt=i, rng=rng) for i in range(5)}
    assert all(0 <= d <= 30 for d in delays)
    # Full jitter should not pin every worker to the same delay sequence forever
    # when the RNG advances; with fixed seed values still vary by attempt cap.
    assert full_jitter_backoff_seconds(0, rng=random.Random(1)) != full_jitter_backoff_seconds(
        4, rng=random.Random(1)
    )


def test_tool_side_effects_block_automatic_retry() -> None:
    failure = classify_http_status(
        503,
        provider="nvidia",
        received_stream_data=True,
        tool_side_effects=True,
    )
    assert failure.retryable is False
    assert failure.retry_owner is RetryOwner.SUPERVISOR

    interrupt = classify_stream_interrupt(
        provider="nvidia",
        received_stream_data=True,
        tool_side_effects=True,
    )
    assert interrupt.retryable is False
    assert "duplicate" in interrupt.safe_message.lower() or "side effect" in interrupt.safe_message.lower()


def test_partial_stream_interrupt_is_codex_owned() -> None:
    failure = classify_stream_interrupt(
        provider="nvidia",
        received_stream_data=True,
        tool_side_effects=False,
    )
    assert failure.kind is ProviderFailureKind.STREAM_INTERRUPTED
    assert failure.retryable is True
    assert failure.retry_owner is RetryOwner.CODEX_STREAM


def test_pre_data_stream_interrupt_is_transport_owned() -> None:
    failure = classify_stream_interrupt(
        provider="nvidia",
        received_stream_data=False,
        tool_side_effects=False,
    )
    assert failure.retryable is True
    assert failure.retry_owner is RetryOwner.TRANSPORT


def test_dns_and_connect_failures_are_retryable() -> None:
    dns = classify_transport_exception(
        OSError("nodename nor servname provided, or not known"),
        provider="nvidia",
    )
    assert dns.kind is ProviderFailureKind.DNS_FAILURE
    assert dns.retryable is True

    reset = classify_transport_exception(
        ConnectionResetError("Connection reset by peer"),
        provider="nvidia",
    )
    assert reset.kind is ProviderFailureKind.CONNECTION_RESET
    assert reset.retryable is True


def test_sanitize_strips_secrets() -> None:
    raw = 'Authorization: Bearer sk-secret-abc nvapi-super-secret {"api_key":"x"}'
    cleaned = sanitize_provider_body(raw)
    assert "sk-secret" not in cleaned
    assert "nvapi-super" not in cleaned
    assert "***REDACTED***" in cleaned


def test_cancellation_interrupts_backoff() -> None:
    async def _run() -> None:
        cancel = asyncio.Event()
        cancel.set()
        started = asyncio.get_running_loop().time()
        ok = await cancellation_aware_sleep(30.0, cancel_event=cancel)
        elapsed = asyncio.get_running_loop().time() - started
        assert ok is False
        assert elapsed < 1.0

    asyncio.run(_run())


def test_circuit_breaker_ignores_invalid_request() -> None:
    breaker = ProviderCircuitBreaker(failure_threshold=2, open_seconds=60)
    key = "nvidia|integrate.api.nvidia.com"
    bad = classify_http_status(400, provider="nvidia")
    for _ in range(5):
        breaker.record_failure(key, bad)
    assert breaker.allow_request(key) is True

    transient = classify_http_status(503, provider="nvidia")
    breaker.record_failure(key, transient)
    breaker.record_failure(key, transient)
    assert breaker.allow_request(key) is False


def test_retry_budget_fields_on_failure() -> None:
    failure = ProviderFailure(
        kind=ProviderFailureKind.RATE_LIMITED,
        provider="nvidia",
        model="m",
        http_status=429,
        retryable=True,
        attempt=2,
        max_attempts=5,
        backoff_ms=1840,
        operation="chat_completion_stream",
    )
    fields = failure.as_log_fields()
    assert fields["attempt"] == 2
    assert fields["max_attempts"] == 5
    assert fields["backoff_ms"] == 1840
    assert fields["failure_kind"] == "rate_limited"
    assert "rate limit" in failure.user_status().lower()
