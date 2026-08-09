"""Idempotent recovery orchestration with exponential backoff and jitter."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

from .models import (
    HealthReasonCode,
    RecoveryActionKind,
    RecoveryAttempt,
    utc_now,
)


class BackoffPolicy:
    def __init__(
        self,
        *,
        initial_delay: float = 1.0,
        maximum_delay: float = 60.0,
        maximum_attempts: int = 8,
        reset_after_success: bool = True,
    ) -> None:
        self.initial_delay = max(0.1, initial_delay)
        self.maximum_delay = max(self.initial_delay, maximum_delay)
        self.maximum_attempts = max(0, maximum_attempts)
        self.reset_after_success = reset_after_success

    def delay_seconds(self, attempt_number: int, *, connector_id: str, action: str) -> float:
        n = max(1, attempt_number)
        exponential = min(self.maximum_delay, self.initial_delay * (2 ** (n - 1)))
        seed = f"{connector_id}:{action}:{n}".encode("utf-8")
        jitter_ratio = int.from_bytes(hashlib.sha256(seed).digest()[:2], "big") / 65535
        return min(self.maximum_delay, exponential * (0.75 + 0.5 * jitter_ratio))

    def exhausted(self, attempt_number: int) -> bool:
        if self.maximum_attempts <= 0:
            return True
        return attempt_number > self.maximum_attempts


# Recoveries that only touch local transport state (no external config mutation).
LOCAL_SAFE_ACTIONS = frozenset(
    {
        RecoveryActionKind.TRANSPORT_RECONNECT,
        RecoveryActionKind.WEBSOCKET_RECREATE,
        RecoveryActionKind.TOKEN_REFRESH,
        RecoveryActionKind.CLIENT_RECREATE,
        RecoveryActionKind.POLLER_RESTART,
        RecoveryActionKind.CONSUMER_RESTART,
        RecoveryActionKind.CONNECTION_POOL_RESET,
        RecoveryActionKind.NONE,
    }
)

# Recoveries that mutate provider-side configuration and need transactional policy.
TRANSACTIONAL_ACTIONS = frozenset(
    {
        RecoveryActionKind.SUBSCRIPTION_RENEW,
        RecoveryActionKind.WEBHOOK_REREGISTER,
    }
)

# Auth failures must not loop forever on reconnect.
AUTH_TERMINAL_CODES = frozenset(
    {
        HealthReasonCode.AUTH_REVOKED,
        HealthReasonCode.AUTH_EXPIRED,
        HealthReasonCode.TOKEN_REFRESH_FAILED,
    }
)


def select_recovery_action(
    reason_code: HealthReasonCode,
    available: list[RecoveryActionKind],
) -> RecoveryActionKind:
    preferred: dict[HealthReasonCode, list[RecoveryActionKind]] = {
        HealthReasonCode.AUTH_EXPIRED: [RecoveryActionKind.TOKEN_REFRESH],
        HealthReasonCode.TOKEN_REFRESH_FAILED: [RecoveryActionKind.TOKEN_REFRESH],
        HealthReasonCode.CONNECTION_REFUSED: [
            RecoveryActionKind.TRANSPORT_RECONNECT,
            RecoveryActionKind.CLIENT_RECREATE,
            RecoveryActionKind.POLLER_RESTART,
        ],
        HealthReasonCode.CONNECTION_TIMEOUT: [
            RecoveryActionKind.TRANSPORT_RECONNECT,
            RecoveryActionKind.WEBSOCKET_RECREATE,
        ],
        HealthReasonCode.INGRESS_STALLED: [
            RecoveryActionKind.POLLER_RESTART,
            RecoveryActionKind.CONSUMER_RESTART,
            RecoveryActionKind.TRANSPORT_RECONNECT,
        ],
        HealthReasonCode.SUBSCRIPTION_MISSING: [RecoveryActionKind.SUBSCRIPTION_RENEW],
        HealthReasonCode.SUBSCRIPTION_EXPIRED: [RecoveryActionKind.SUBSCRIPTION_RENEW],
        HealthReasonCode.WEBHOOK_UNREACHABLE: [RecoveryActionKind.WEBHOOK_REREGISTER],
        HealthReasonCode.EGRESS_FAILED: [
            RecoveryActionKind.CLIENT_RECREATE,
            RecoveryActionKind.CONNECTION_POOL_RESET,
        ],
        HealthReasonCode.RECONNECT_FAILED: [
            RecoveryActionKind.CLIENT_RECREATE,
            RecoveryActionKind.TRANSPORT_RECONNECT,
        ],
        HealthReasonCode.PROBE_FAILED: [
            RecoveryActionKind.CLIENT_RECREATE,
            RecoveryActionKind.TRANSPORT_RECONNECT,
        ],
    }
    for action in preferred.get(reason_code, [RecoveryActionKind.TRANSPORT_RECONNECT]):
        if action in available:
            return action
    for action in available:
        if action is not RecoveryActionKind.NONE:
            return action
    return RecoveryActionKind.NONE


def requires_transactional_policy(action: RecoveryActionKind) -> bool:
    return action in TRANSACTIONAL_ACTIONS


def is_auth_terminal(reason_code: HealthReasonCode) -> bool:
    return reason_code in AUTH_TERMINAL_CODES


def build_recovery_attempt(
    *,
    connector_id: str,
    action: RecoveryActionKind,
    attempt_number: int,
    reason_code: HealthReasonCode,
    message: str = "",
    clock=utc_now,
) -> RecoveryAttempt:
    return RecoveryAttempt(
        connector_id=connector_id,
        action=action,
        started_at=clock(),
        attempt_number=attempt_number,
        reason_code=reason_code,
        message=message,
        requires_transactional_policy=requires_transactional_policy(action),
        requires_human=is_auth_terminal(reason_code) and action is RecoveryActionKind.TOKEN_REFRESH,
    )


def next_recovery_time(
    policy: BackoffPolicy,
    *,
    connector_id: str,
    action: RecoveryActionKind,
    attempt_number: int,
    now: datetime | None = None,
) -> datetime:
    delay = policy.delay_seconds(attempt_number, connector_id=connector_id, action=action.value)
    return (now or utc_now()) + timedelta(seconds=delay)
