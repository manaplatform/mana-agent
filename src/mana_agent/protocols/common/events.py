"""User-safe protocol observability events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .security import redact_protocol_value


@dataclass(frozen=True, slots=True)
class ProtocolEvent:
    protocol: str
    event_type: str
    correlation_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def emit_protocol_event(
    sink: Callable[[ProtocolEvent], None] | None,
    *,
    protocol: str,
    event_type: str,
    correlation_id: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    if sink is not None:
        sink(
            ProtocolEvent(
                protocol=protocol,
                event_type=event_type,
                correlation_id=correlation_id,
                metadata=dict(redact_protocol_value(metadata or {})),
            )
        )
