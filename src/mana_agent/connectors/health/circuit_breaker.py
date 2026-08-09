"""Connector-level circuit breaker (closed / open / half-open)."""

from __future__ import annotations

from datetime import datetime, timedelta

from .models import CircuitState, utc_now


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        open_seconds: float = 30.0,
        half_open_max_probes: int = 1,
        clock=utc_now,
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.open_seconds = max(0.1, open_seconds)
        self.half_open_max_probes = max(1, half_open_max_probes)
        self.clock = clock
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.opened_at: datetime | None = None
        self.half_open_probes = 0

    def allow_probe(self) -> bool:
        now = self.clock()
        if self.state is CircuitState.CLOSED:
            return True
        if self.state is CircuitState.OPEN:
            if self.opened_at is None:
                return False
            if now - self.opened_at >= timedelta(seconds=self.open_seconds):
                self.state = CircuitState.HALF_OPEN
                self.half_open_probes = 0
                return True
            return False
        # half-open
        if self.half_open_probes < self.half_open_max_probes:
            self.half_open_probes += 1
            return True
        return False

    def record_success(self) -> CircuitState:
        self.failure_count = 0
        self.half_open_probes = 0
        self.opened_at = None
        self.state = CircuitState.CLOSED
        return self.state

    def record_failure(self) -> CircuitState:
        self.failure_count += 1
        if self.state is CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.opened_at = self.clock()
            self.half_open_probes = 0
            return self.state
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = self.clock()
        return self.state

    def force_open(self) -> CircuitState:
        self.state = CircuitState.OPEN
        self.opened_at = self.clock()
        return self.state

    def snapshot(self) -> dict:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "half_open_probes": self.half_open_probes,
        }

    def restore(
        self,
        *,
        state: CircuitState,
        failure_count: int = 0,
        opened_at: datetime | None = None,
    ) -> None:
        self.state = state
        self.failure_count = max(0, failure_count)
        self.opened_at = opened_at
        self.half_open_probes = 0
