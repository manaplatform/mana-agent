"""Incident lifecycle: open, timeline, recover, retain history."""

from __future__ import annotations

from datetime import datetime

from .models import (
    ConnectorHealthState,
    ConnectorIncident,
    HealthReasonCode,
    IncidentEvent,
    utc_now,
)
from .storage import ConnectorHealthStore


class IncidentTracker:
    def __init__(self, store: ConnectorHealthStore, *, clock=utc_now) -> None:
        self.store = store
        self.clock = clock
        self._open: dict[str, ConnectorIncident] = {}

    def load_open(self) -> None:
        for incident in self.store.list_incidents(include_open=True, include_closed=False, limit=1000):
            if incident.open:
                self._open[incident.connector_id] = incident

    def current(self, connector_id: str) -> ConnectorIncident | None:
        return self._open.get(connector_id)

    def ensure_open(
        self,
        *,
        connector_id: str,
        state: ConnectorHealthState,
        reason_code: HealthReasonCode,
        message: str = "",
    ) -> ConnectorIncident:
        existing = self._open.get(connector_id)
        if existing is not None and existing.open:
            self.append(
                existing,
                event_type=f"CONNECTOR_{state.value.upper()}",
                reason_code=reason_code,
                message=message,
            )
            return existing
        incident = ConnectorIncident(
            connector_id=connector_id,
            started_at=self.clock(),
            opening_state=state,
            opening_reason=reason_code,
        )
        incident.events.append(
            IncidentEvent(
                incident_id=incident.incident_id,
                connector_id=connector_id,
                event_type=f"CONNECTOR_{state.value.upper()}",
                occurred_at=self.clock(),
                reason_code=reason_code,
                message=message or f"Incident opened in state {state.value}",
            )
        )
        self._open[connector_id] = incident
        self.store.save_incident(incident)
        return incident

    def append(
        self,
        incident: ConnectorIncident,
        *,
        event_type: str,
        reason_code: HealthReasonCode = HealthReasonCode.NONE,
        message: str = "",
        recovery_attempt_id: str = "",
        details: dict | None = None,
    ) -> IncidentEvent:
        event = IncidentEvent(
            incident_id=incident.incident_id,
            connector_id=incident.connector_id,
            event_type=event_type,
            occurred_at=self.clock(),
            reason_code=reason_code,
            recovery_attempt_id=recovery_attempt_id,
            message=message,
            details=details or {},
        )
        incident.events.append(event)
        if recovery_attempt_id and recovery_attempt_id not in incident.recovery_attempt_ids:
            incident.recovery_attempt_ids.append(recovery_attempt_id)
        self.store.save_incident(incident)
        return event

    def close(
        self,
        connector_id: str,
        *,
        closing_state: ConnectorHealthState = ConnectorHealthState.HEALTHY,
        message: str = "CONNECTOR_HEALTHY",
    ) -> ConnectorIncident | None:
        incident = self._open.pop(connector_id, None)
        if incident is None:
            return None
        incident.ended_at = self.clock()
        incident.closing_state = closing_state
        incident.recovered = closing_state is ConnectorHealthState.HEALTHY
        incident.events.append(
            IncidentEvent(
                incident_id=incident.incident_id,
                connector_id=connector_id,
                event_type="CONNECTOR_HEALTHY" if incident.recovered else f"CONNECTOR_{closing_state.value.upper()}",
                occurred_at=incident.ended_at,
                message=message,
            )
        )
        self.store.save_incident(incident)
        return incident

    def timeline(self, connector_id: str, *, limit: int = 50) -> list[IncidentEvent]:
        events: list[IncidentEvent] = []
        for incident in self.store.list_incidents(connector_id=connector_id, limit=20):
            events.extend(incident.events)
        events.sort(key=lambda item: item.occurred_at)
        return events[-limit:]
