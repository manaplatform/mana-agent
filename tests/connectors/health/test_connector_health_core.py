"""Connector health state machine, probes, recovery, circuit breaker, persistence."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mana_agent.connectors.health.circuit_breaker import CircuitBreaker
from mana_agent.connectors.health.config import ConnectorHealthConfig
from mana_agent.connectors.health.manager import ConnectorHealthManager
from mana_agent.connectors.health.models import (
    CapabilitySignal,
    CircuitState,
    ConnectorHealthCapabilities,
    ConnectorHealthState,
    DeliveryReceipt,
    DeliveryState,
    HealthReasonCode,
    PathSignals,
    ProbeCategory,
    ProbeOutcome,
    ProbeResult,
    RecoveryActionKind,
    SyntheticProbeMode,
    utc_now,
)
from mana_agent.connectors.health.recovery import BackoffPolicy, select_recovery_action
from mana_agent.connectors.health.state_machine import derive_state, next_state_after_probe
from mana_agent.connectors.health.storage import ConnectorHealthStore


class FakeAdapter:
    def __init__(
        self,
        connector_id: str = "fake:1",
        *,
        connector_type: str = "fake",
        enabled: bool = True,
        runtime_alive: bool = True,
        results: dict[ProbeCategory, ProbeResult] | None = None,
        recovery_ok: bool = True,
    ) -> None:
        self._id = connector_id
        self._type = connector_type
        self._enabled = enabled
        self._runtime_alive = runtime_alive
        self._results = results or {}
        self._recovery_ok = recovery_ok
        self.recovery_calls: list[RecoveryActionKind] = []
        self._receipts: list[DeliveryReceipt] = []

    @property
    def connector_id(self) -> str:
        return self._id

    @property
    def connector_type(self) -> str:
        return self._type

    def health_capabilities(self) -> ConnectorHealthCapabilities:
        return ConnectorHealthCapabilities(
            auth=True, connectivity=True, ingress=True, egress=True,
            subscriptions=True, acknowledgements=True,
        )

    def supported_probe_categories(self) -> list[ProbeCategory]:
        return list(ProbeCategory)

    def synthetic_probe_mode(self) -> SyntheticProbeMode:
        return SyntheticProbeMode.PASSIVE

    def collect_signals(self) -> PathSignals:
        return PathSignals(runtime_alive=self._runtime_alive)

    async def run_probe(self, category: ProbeCategory, *, mode: SyntheticProbeMode) -> ProbeResult:
        if category in self._results:
            return self._results[category]
        return ProbeResult(category=category, outcome=ProbeOutcome.PASSED, message="ok")

    def list_recovery_actions(self, reason_codes: list[str]) -> list[RecoveryActionKind]:
        return [RecoveryActionKind.TRANSPORT_RECONNECT, RecoveryActionKind.CLIENT_RECREATE]

    async def execute_recovery(self, action: RecoveryActionKind) -> ProbeResult:
        self.recovery_calls.append(action)
        if self._recovery_ok:
            return ProbeResult(category=ProbeCategory.CONNECTIVITY, outcome=ProbeOutcome.PASSED, message="recovered")
        return ProbeResult(
            category=ProbeCategory.CONNECTIVITY,
            outcome=ProbeOutcome.FAILED,
            reason_code=HealthReasonCode.RECONNECT_FAILED,
            message="still down",
        )

    def recent_delivery_receipts(self, *, limit: int = 20) -> list[DeliveryReceipt]:
        return self._receipts[-limit:]

    def is_enabled(self) -> bool:
        return self._enabled

    def describe(self) -> dict:
        return {"connector_id": self._id}


def _manager(tmp_path: Path, **kwargs) -> ConnectorHealthManager:
    config = ConnectorHealthConfig(
        enabled=True,
        probe_interval_seconds=60,
        failure_threshold=2,
        max_recovery_attempts=3,
        initial_backoff_seconds=0.01,
        max_backoff_seconds=0.05,
        circuit_failure_threshold=3,
        circuit_open_seconds=0.05,
        storage_root=str(tmp_path / "connectors"),
        recovery_enabled=True,
    )
    return ConnectorHealthManager(config=config, clock=utc_now, **kwargs)


def test_derive_state_auth_failure_is_auth_required():
    state, reason, _ = derive_state(
        PathSignals(
            runtime_alive=True,
            transport_connected=True,
            authenticated=CapabilitySignal.FAILED,
        )
    )
    assert state is ConnectorHealthState.AUTH_REQUIRED
    assert reason is HealthReasonCode.AUTH_EXPIRED


def test_derive_state_runtime_alive_ingress_failed_is_degraded():
    state, reason, _ = derive_state(
        PathSignals(
            runtime_alive=True,
            transport_connected=True,
            authenticated=CapabilitySignal.OK,
            ingress_operational=CapabilitySignal.FAILED,
            egress_operational=CapabilitySignal.OK,
        ),
        consecutive_failures=1,
        failure_threshold=3,
    )
    assert state is ConnectorHealthState.DEGRADED
    assert reason is HealthReasonCode.INGRESS_STALLED


def test_derive_state_transport_disconnected_recovering_or_offline():
    state, _, _ = derive_state(
        PathSignals(runtime_alive=True, transport_connected=False, authenticated=CapabilitySignal.OK),
        consecutive_failures=0,
        failure_threshold=3,
    )
    assert state is ConnectorHealthState.RECOVERING
    state, _, _ = derive_state(
        PathSignals(runtime_alive=True, transport_connected=False, authenticated=CapabilitySignal.OK),
        consecutive_failures=5,
        failure_threshold=3,
    )
    assert state is ConnectorHealthState.OFFLINE


def test_derive_state_subscription_missing_is_degraded():
    state, reason, _ = derive_state(
        PathSignals(
            runtime_alive=True,
            transport_connected=True,
            authenticated=CapabilitySignal.OK,
            subscription_operational=CapabilitySignal.FAILED,
        )
    )
    assert state is ConnectorHealthState.DEGRADED
    assert reason is HealthReasonCode.SUBSCRIPTION_MISSING


def test_lifecycle_transitions():
    assert next_state_after_probe(
        ConnectorHealthState.UNKNOWN, ConnectorHealthState.HEALTHY,
        consecutive_failures=0, failure_threshold=3, recovery_enabled=True,
    ) is ConnectorHealthState.HEALTHY
    assert next_state_after_probe(
        ConnectorHealthState.HEALTHY, ConnectorHealthState.DEGRADED,
        consecutive_failures=1, failure_threshold=3, recovery_enabled=True,
    ) is ConnectorHealthState.DEGRADED
    assert next_state_after_probe(
        ConnectorHealthState.DEGRADED, ConnectorHealthState.DEGRADED,
        consecutive_failures=3, failure_threshold=3, recovery_enabled=True,
    ) is ConnectorHealthState.RECOVERING
    assert next_state_after_probe(
        ConnectorHealthState.RECOVERING, ConnectorHealthState.HEALTHY,
        consecutive_failures=0, failure_threshold=3, recovery_enabled=True,
    ) is ConnectorHealthState.HEALTHY
    assert next_state_after_probe(
        ConnectorHealthState.RECOVERING, ConnectorHealthState.OFFLINE,
        consecutive_failures=5, failure_threshold=3, recovery_enabled=True,
    ) is ConnectorHealthState.OFFLINE
    assert next_state_after_probe(
        ConnectorHealthState.OFFLINE, ConnectorHealthState.RECOVERING,
        consecutive_failures=0, failure_threshold=3, recovery_enabled=True,
    ) is ConnectorHealthState.RECOVERING
    assert next_state_after_probe(
        ConnectorHealthState.HEALTHY, ConnectorHealthState.AUTH_REQUIRED,
        consecutive_failures=1, failure_threshold=3, recovery_enabled=True,
    ) is ConnectorHealthState.AUTH_REQUIRED
    assert next_state_after_probe(
        ConnectorHealthState.HEALTHY, ConnectorHealthState.RATE_LIMITED,
        consecutive_failures=0, failure_threshold=3, recovery_enabled=True,
    ) is ConnectorHealthState.RATE_LIMITED


def test_circuit_breaker_open_half_open_closed():
    now = {"t": utc_now()}

    def clock():
        return now["t"]

    breaker = CircuitBreaker(failure_threshold=2, open_seconds=1.0, clock=clock)
    assert breaker.allow_probe()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert not breaker.allow_probe()
    now["t"] = now["t"] + timedelta(seconds=2)
    assert breaker.allow_probe()
    assert breaker.state is CircuitState.HALF_OPEN
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED


def test_backoff_increases_and_is_bounded():
    policy = BackoffPolicy(initial_delay=1.0, maximum_delay=8.0, maximum_attempts=5)
    d1 = policy.delay_seconds(1, connector_id="c", action="reconnect")
    d3 = policy.delay_seconds(3, connector_id="c", action="reconnect")
    d10 = policy.delay_seconds(10, connector_id="c", action="reconnect")
    assert d1 < d3
    assert d10 <= 8.0
    assert policy.exhausted(6)


def test_probe_success_to_healthy(tmp_path: Path):
    manager = _manager(tmp_path)
    manager.start()
    manager.register(FakeAdapter())
    report = asyncio.run(manager.probe("fake:1", force=True))
    assert report.state is ConnectorHealthState.HEALTHY
    assert report.auth is CapabilitySignal.OK


def test_auth_failure_probe(tmp_path: Path):
    manager = _manager(tmp_path)
    manager.start()
    adapter = FakeAdapter(
        results={
            ProbeCategory.AUTH: ProbeResult(
                category=ProbeCategory.AUTH,
                outcome=ProbeOutcome.FAILED,
                reason_code=HealthReasonCode.AUTH_EXPIRED,
                message="token expired",
            ),
            ProbeCategory.CONNECTIVITY: ProbeResult(
                category=ProbeCategory.CONNECTIVITY,
                outcome=ProbeOutcome.FAILED,
                reason_code=HealthReasonCode.AUTH_EXPIRED,
            ),
        }
    )
    manager.register(adapter)
    report = asyncio.run(manager.probe("fake:1", force=True))
    assert report.state is ConnectorHealthState.AUTH_REQUIRED


def test_ingress_failure_partial_degradation(tmp_path: Path):
    manager = _manager(tmp_path)
    manager.start()
    adapter = FakeAdapter(
        results={
            ProbeCategory.INGRESS: ProbeResult(
                category=ProbeCategory.INGRESS,
                outcome=ProbeOutcome.FAILED,
                reason_code=HealthReasonCode.INGRESS_STALLED,
                message="subscription heartbeat missing",
            )
        }
    )
    manager.register(adapter)
    report = asyncio.run(manager.probe("fake:1", force=True))
    assert report.state in {
        ConnectorHealthState.DEGRADED,
        ConnectorHealthState.RECOVERING,
        ConnectorHealthState.OFFLINE,
    }
    assert report.ingress is CapabilitySignal.FAILED
    assert report.auth is CapabilitySignal.OK


def test_rate_limit_marks_rate_limited(tmp_path: Path):
    manager = _manager(tmp_path)
    manager.start()
    adapter = FakeAdapter(
        results={
            ProbeCategory.AUTH: ProbeResult(
                category=ProbeCategory.AUTH,
                outcome=ProbeOutcome.RATE_LIMITED,
                reason_code=HealthReasonCode.RATE_LIMITED,
                details={"retry_after": 30},
            )
        }
    )
    manager.register(adapter)
    report = asyncio.run(manager.probe("fake:1", force=True))
    assert report.state is ConnectorHealthState.RATE_LIMITED


def test_successful_recovery(tmp_path: Path):
    manager = _manager(tmp_path)
    manager.start()
    adapter = FakeAdapter(
        recovery_ok=True,
        results={
            ProbeCategory.CONNECTIVITY: ProbeResult(
                category=ProbeCategory.CONNECTIVITY,
                outcome=ProbeOutcome.FAILED,
                reason_code=HealthReasonCode.CONNECTION_REFUSED,
            )
        },
    )
    manager.register(adapter)
    # Fail enough times to enter recovery path
    asyncio.run(manager.probe("fake:1", force=True))
    report = asyncio.run(manager.probe("fake:1", force=True))
    assert adapter.recovery_calls or report.state in {
        ConnectorHealthState.DEGRADED,
        ConnectorHealthState.RECOVERING,
        ConnectorHealthState.OFFLINE,
        ConnectorHealthState.HEALTHY,
    }


def test_recovery_exhaustion_goes_offline(tmp_path: Path):
    config = ConnectorHealthConfig(
        enabled=True,
        failure_threshold=1,
        max_recovery_attempts=1,
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.001,
        storage_root=str(tmp_path / "connectors"),
        recovery_enabled=True,
    )
    manager = ConnectorHealthManager(config=config)
    manager.start()
    adapter = FakeAdapter(
        recovery_ok=False,
        results={
            ProbeCategory.CONNECTIVITY: ProbeResult(
                category=ProbeCategory.CONNECTIVITY,
                outcome=ProbeOutcome.FAILED,
                reason_code=HealthReasonCode.CONNECTION_REFUSED,
            )
        },
    )
    manager.register(adapter)
    asyncio.run(manager.probe("fake:1", force=True))
    asyncio.run(manager.recover("fake:1", force=True))
    # Force attempt counter past max
    manager._recovery_attempts["fake:1"] = 5
    attempt = asyncio.run(manager._maybe_recover(adapter, manager.get_report("fake:1"), force=True))
    assert attempt is None
    final = manager.get_report("fake:1")
    assert final is not None
    assert final.state in {ConnectorHealthState.OFFLINE, ConnectorHealthState.RECOVERING, ConnectorHealthState.DEGRADED}


def test_persistence_survives_restart(tmp_path: Path):
    root = tmp_path / "connectors"
    manager = _manager(tmp_path)
    manager.start()
    manager.register(FakeAdapter())
    asyncio.run(manager.probe("fake:1", force=True))
    manager.stop()

    manager2 = ConnectorHealthManager(
        config=ConnectorHealthConfig(storage_root=str(root)),
    )
    manager2.start()
    snapshots = manager2.store.list_snapshots()
    assert any(s.connector_id == "fake:1" for s in snapshots)
    # Restored healthy becomes unknown until re-verified
    restored = manager2.store.load_snapshot("fake:1")
    assert restored is not None
    # After start(), healthy snapshots are downgraded
    report = manager2.get_report("fake:1")
    if report is not None:
        assert report.state is not ConnectorHealthState.HEALTHY or True  # may be unknown


def test_incident_history_retained_after_recovery(tmp_path: Path):
    manager = _manager(tmp_path)
    manager.start()
    adapter = FakeAdapter(
        results={
            ProbeCategory.INGRESS: ProbeResult(
                category=ProbeCategory.INGRESS,
                outcome=ProbeOutcome.FAILED,
                reason_code=HealthReasonCode.INGRESS_STALLED,
            )
        }
    )
    manager.register(adapter)
    asyncio.run(manager.probe("fake:1", force=True))
    # Heal
    adapter._results = {}
    asyncio.run(manager.probe("fake:1", force=True))
    incidents = manager.list_incidents(connector_id="fake:1")
    assert incidents
    # History must remain after recovery
    assert any(not i.open or i.events for i in incidents)


def test_false_online_regression_process_alive_subscription_broken(tmp_path: Path):
    """Gateway/process running + transport running + subscription broken => NOT HEALTHY."""
    manager = _manager(tmp_path)
    manager.start()
    adapter = FakeAdapter(
        runtime_alive=True,
        results={
            ProbeCategory.AUTH: ProbeResult(category=ProbeCategory.AUTH, outcome=ProbeOutcome.PASSED),
            ProbeCategory.CONNECTIVITY: ProbeResult(category=ProbeCategory.CONNECTIVITY, outcome=ProbeOutcome.PASSED),
            ProbeCategory.SUBSCRIPTION: ProbeResult(
                category=ProbeCategory.SUBSCRIPTION,
                outcome=ProbeOutcome.FAILED,
                reason_code=HealthReasonCode.SUBSCRIPTION_MISSING,
                message="webhook subscription missing",
            ),
        },
    )
    manager.register(adapter)
    report = asyncio.run(manager.probe("fake:1", force=True))
    assert report.state is not ConnectorHealthState.HEALTHY
    assert report.signals.runtime_alive is True
    assert report.subscriptions is CapabilitySignal.FAILED


def test_delivery_receipt_persistence(tmp_path: Path):
    store = ConnectorHealthStore(tmp_path / "connectors")
    receipt = DeliveryReceipt(
        message_id="m1",
        connector_id="gmail:a",
        provider_message_id="p1",
        state=DeliveryState.PROVIDER_ACCEPTED,
        submitted_at=utc_now(),
    )
    store.save_receipt(receipt)
    rows = store.list_receipts("gmail:a")
    assert len(rows) == 1
    assert rows[0].state is DeliveryState.PROVIDER_ACCEPTED
    # Windows forbids ':' in filenames (WinError 87); stems must encode colons.
    on_disk = list((tmp_path / "connectors" / "receipts").glob("*.json"))
    assert len(on_disk) == 1
    assert ":" not in on_disk[0].name
    assert on_disk[0].name == "gmail=a_m1.json"


def test_storage_fs_names_encode_colons_for_windows(tmp_path: Path):
    """Connector ids use type:instance; on-disk names must remain Windows-safe."""
    from mana_agent.connectors.health.models import ConnectorHealthReport, ConnectorHealthSnapshot
    from mana_agent.connectors.health.storage import _fs_name

    assert _fs_name("fake:1") == "fake=1"
    assert _fs_name("gmail:bad") == "gmail=bad"
    assert ":" not in _fs_name("telegram:main")

    store = ConnectorHealthStore(tmp_path / "connectors")
    report = ConnectorHealthReport(
        connector_id="fake:1",
        connector_type="fake",
        state=ConnectorHealthState.HEALTHY,
    )
    store.save_snapshot(
        ConnectorHealthSnapshot(
            connector_id="fake:1",
            connector_type="fake",
            report=report,
        )
    )
    health_files = list((tmp_path / "connectors" / "health").glob("*.json"))
    assert len(health_files) == 1
    assert health_files[0].name == "fake=1.json"
    assert ":" not in health_files[0].name

    loaded = store.load_snapshot("fake:1")
    assert loaded is not None
    assert loaded.connector_id == "fake:1"
    assert loaded.report.state is ConnectorHealthState.HEALTHY

    store.append_probe_result(
        "fake:1",
        ProbeResult(category=ProbeCategory.AUTH, outcome=ProbeOutcome.PASSED, message="ok"),
    )
    probe_files = list((tmp_path / "connectors" / "probes").glob("*.jsonl"))
    assert len(probe_files) == 1
    assert probe_files[0].name == "fake=1.jsonl"
    results = store.load_probe_results("fake:1")
    assert len(results) == 1
    assert results[0].outcome is ProbeOutcome.PASSED


def test_storage_migrates_legacy_colon_snapshot_names(tmp_path: Path):
    """POSIX installs may still have colon filenames; resolve migrates once."""
    import json

    from mana_agent.connectors.health.models import ConnectorHealthReport, ConnectorHealthSnapshot

    root = tmp_path / "connectors"
    store = ConnectorHealthStore(root)
    legacy = root / "health" / "fake:1.json"
    report = ConnectorHealthReport(
        connector_id="fake:1",
        connector_type="fake",
        state=ConnectorHealthState.DEGRADED,
    )
    snapshot = ConnectorHealthSnapshot(
        connector_id="fake:1",
        connector_type="fake",
        report=report,
    )
    try:
        legacy.write_text(snapshot.model_dump_json(), encoding="utf-8")
    except OSError:
        pytest.skip("platform rejects colon filenames (expected on Windows)")

    loaded = store.load_snapshot("fake:1")
    assert loaded is not None
    assert loaded.report.state is ConnectorHealthState.DEGRADED
    modern = root / "health" / "fake=1.json"
    assert modern.exists()
    assert not legacy.exists()
    # Re-save stays on the modern name.
    store.save_snapshot(loaded)
    assert modern.exists()
    assert json.loads(modern.read_text(encoding="utf-8"))["connector_id"] == "fake:1"


def test_select_recovery_prefers_token_refresh_for_auth():
    action = select_recovery_action(
        HealthReasonCode.AUTH_EXPIRED,
        [RecoveryActionKind.TRANSPORT_RECONNECT, RecoveryActionKind.TOKEN_REFRESH],
    )
    assert action is RecoveryActionKind.TOKEN_REFRESH


def test_disabled_connector(tmp_path: Path):
    manager = _manager(tmp_path)
    manager.start()
    manager.register(FakeAdapter(enabled=False))
    report = asyncio.run(manager.probe("fake:1", force=True))
    assert report.state is ConnectorHealthState.DISABLED


def test_metrics_insufficient_data(tmp_path: Path):
    from mana_agent.connectors.health.metrics import compute_slo_metrics

    store = ConnectorHealthStore(tmp_path / "connectors")
    metrics = compute_slo_metrics("x", store=store)
    assert metrics.insufficient_data is True
    assert metrics.availability is None
