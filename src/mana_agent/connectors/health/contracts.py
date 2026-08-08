"""Protocol every instrumented connector implements for health probes."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .models import (
    ConnectorHealthCapabilities,
    DeliveryReceipt,
    PathSignals,
    ProbeCategory,
    ProbeResult,
    RecoveryActionKind,
    SyntheticProbeMode,
)


@runtime_checkable
class HealthProbeable(Protocol):
    """Connector-facing health contract.

    Process/gateway aliveness is reported through ``PathSignals.runtime_alive``
    and never alone decides the connector is healthy.
    """

    @property
    def connector_id(self) -> str: ...

    @property
    def connector_type(self) -> str: ...

    def health_capabilities(self) -> ConnectorHealthCapabilities: ...

    def supported_probe_categories(self) -> list[ProbeCategory]: ...

    def synthetic_probe_mode(self) -> SyntheticProbeMode: ...

    def collect_signals(self) -> PathSignals: ...

    async def run_probe(self, category: ProbeCategory, *, mode: SyntheticProbeMode) -> ProbeResult: ...

    def list_recovery_actions(self, reason_codes: list[str]) -> list[RecoveryActionKind]: ...

    async def execute_recovery(self, action: RecoveryActionKind) -> ProbeResult: ...

    def recent_delivery_receipts(self, *, limit: int = 20) -> list[DeliveryReceipt]: ...

    def is_enabled(self) -> bool: ...

    def describe(self) -> dict[str, Any]: ...


class RecoveryPolicyGate(Protocol):
    """Optional policy gate for recoveries that mutate external state."""

    def authorize_recovery(
        self,
        *,
        connector_id: str,
        action: RecoveryActionKind,
        reason: str,
    ) -> bool: ...
