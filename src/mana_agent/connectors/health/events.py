"""Structured health events with deduplicated alert routing."""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Callable

from mana_agent.utils.redaction import redact_secrets

from .models import (
    AlertSeverity,
    AlertSeverityByState,
    ConnectorHealthChanged,
    ConnectorHealthState,
    HealthReasonCode,
    utc_now,
)

logger = logging.getLogger(__name__)

EventSink = Callable[[str, dict], None]


class HealthEventRouter:
    """Emit structured health events; suppress repeated identical alerts."""

    def __init__(
        self,
        *,
        event_sink: EventSink | None = None,
        dedupe_window_seconds: float = 300.0,
        clock=utc_now,
    ) -> None:
        self.event_sink = event_sink
        self.dedupe_window = timedelta(seconds=max(1.0, dedupe_window_seconds))
        self.clock = clock
        self._recent: deque[tuple[str, datetime]] = deque(maxlen=500)

    def emit_log(self, event_type: str, **fields) -> None:
        safe = redact_secrets(fields)
        logger.info("%s %s", event_type, safe)
        if self.event_sink is not None:
            self.event_sink(event_type, safe)

    def health_changed(
        self,
        *,
        connector_id: str,
        previous_state: ConnectorHealthState,
        state: ConnectorHealthState,
        reason_code: HealthReasonCode,
        incident_id: str = "",
        message: str = "",
    ) -> ConnectorHealthChanged | None:
        severity = AlertSeverityByState.get(state, AlertSeverity.INFO)
        dedupe_key = f"{connector_id}:{state.value}:{reason_code.value}"
        if previous_state is state and self._is_duplicate(dedupe_key):
            return None
        if previous_state is state and state in {
            ConnectorHealthState.HEALTHY,
            ConnectorHealthState.DISABLED,
        }:
            return None
        event = ConnectorHealthChanged(
            connector_id=connector_id,
            previous_state=previous_state,
            state=state,
            reason_code=reason_code,
            incident_id=incident_id,
            severity=severity,
            occurred_at=self.clock(),
            message=message,
            dedupe_key=dedupe_key,
        )
        self._remember(dedupe_key)
        self.emit_log(
            "connector.health.changed",
            connector_id=connector_id,
            previous_state=previous_state.value,
            state=state.value,
            reason_code=reason_code.value,
            incident_id=incident_id,
            severity=severity.value,
            message=message,
        )
        if state is ConnectorHealthState.DEGRADED:
            self.emit_log("connector.health.degraded", connector_id=connector_id, reason_code=reason_code.value)
        return event

    def _is_duplicate(self, key: str) -> bool:
        now = self.clock()
        while self._recent and now - self._recent[0][1] > self.dedupe_window:
            self._recent.popleft()
        return any(item_key == key for item_key, _ in self._recent)

    def _remember(self, key: str) -> None:
        self._recent.append((key, self.clock()))
