"""Probe helpers and safety enforcement for synthetic checks."""

from __future__ import annotations

import time
from typing import Awaitable, Callable

from .models import (
    HealthReasonCode,
    ProbeCategory,
    ProbeOutcome,
    ProbeResult,
    SyntheticProbeMode,
    utc_now,
)


async def timed_probe(
    category: ProbeCategory,
    coro_factory: Callable[[], Awaitable[ProbeResult]],
) -> ProbeResult:
    started = time.perf_counter()
    try:
        result = await coro_factory()
    except Exception as exc:  # intentional: probe failures become structured results
        return ProbeResult(
            category=category,
            outcome=ProbeOutcome.FAILED,
            reason_code=HealthReasonCode.PROBE_FAILED,
            latency_ms=(time.perf_counter() - started) * 1000,
            message=str(exc)[:500],
            checked_at=utc_now(),
            details={"exception_type": type(exc).__name__},
        )
    if result.latency_ms is None:
        result = result.model_copy(update={"latency_ms": (time.perf_counter() - started) * 1000})
    return result


def assert_probe_safety(
    category: ProbeCategory,
    mode: SyntheticProbeMode,
    *,
    active_probe_allowed: bool = False,
    test_channel: str = "",
) -> ProbeResult | None:
    """Return a SKIPPED result when the requested probe is unsafe; else None."""
    if mode is SyntheticProbeMode.PASSIVE:
        return None
    if mode is SyntheticProbeMode.ACTIVE and not active_probe_allowed:
        return ProbeResult(
            category=category,
            outcome=ProbeOutcome.SKIPPED,
            reason_code=HealthReasonCode.NONE,
            message="Active synthetic probes are disabled; using passive checks only",
            checked_at=utc_now(),
        )
    if mode is SyntheticProbeMode.TEST_CHANNEL and not test_channel.strip():
        return ProbeResult(
            category=category,
            outcome=ProbeOutcome.SKIPPED,
            reason_code=HealthReasonCode.NONE,
            message="No test_channel configured for synthetic probe mode test_channel",
            checked_at=utc_now(),
        )
    if category in {ProbeCategory.EGRESS, ProbeCategory.ACKNOWLEDGEMENT} and mode is SyntheticProbeMode.PASSIVE:
        return ProbeResult(
            category=category,
            outcome=ProbeOutcome.SKIPPED,
            reason_code=HealthReasonCode.NONE,
            message="Passive mode cannot perform egress/ack synthetic probes",
            checked_at=utc_now(),
        )
    return None


def passed(category: ProbeCategory, *, latency_ms: float | None = None, message: str = "ok") -> ProbeResult:
    return ProbeResult(
        category=category,
        outcome=ProbeOutcome.PASSED,
        latency_ms=latency_ms,
        message=message,
        checked_at=utc_now(),
    )


def failed(
    category: ProbeCategory,
    reason_code: HealthReasonCode,
    message: str,
    *,
    latency_ms: float | None = None,
    details: dict | None = None,
) -> ProbeResult:
    return ProbeResult(
        category=category,
        outcome=ProbeOutcome.FAILED,
        reason_code=reason_code,
        latency_ms=latency_ms,
        message=message,
        checked_at=utc_now(),
        details=details or {},
    )


def rate_limited(category: ProbeCategory, message: str = "rate limited", *, retry_after: float | None = None) -> ProbeResult:
    return ProbeResult(
        category=category,
        outcome=ProbeOutcome.RATE_LIMITED,
        reason_code=HealthReasonCode.RATE_LIMITED,
        message=message,
        checked_at=utc_now(),
        details={"retry_after": retry_after} if retry_after is not None else {},
    )
