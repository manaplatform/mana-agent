"""Supervisor, HITL, transactional, and real-adapter integration tests."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from mana_agent.connectors.health.adapters.gmail import GmailHealthAdapter
from mana_agent.connectors.health.adapters.telegram import TelegramHealthAdapter
from mana_agent.connectors.health.config import ConnectorHealthConfig
from mana_agent.connectors.health.hitl_bridge import ConnectorHitlBridge, build_auth_intervention_request
from mana_agent.connectors.health.manager import ConnectorHealthManager
from mana_agent.connectors.health.models import (
    ConnectorHealthState,
    DeliveryReceipt,
    DeliveryState,
    HealthReasonCode,
    ProbeCategory,
    ProbeOutcome,
    ProbeResult,
    RecoveryActionKind,
    SyntheticProbeMode,
    utc_now,
)
from mana_agent.connectors.health.supervisor_bridge import ConnectorSupervisorBridge
from mana_agent.connectors.health.transactional_bridge import ConnectorTransactionalBridge
from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.models import (
    CheckpointRecord,
    ExecutionState,
    SideEffectClassification,
    TaskRecord,
)
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor


class _HealthyProvider:
    async def health_check(self):
        from mana_agent.connectors.email.models import ProviderHealth

        return ProviderHealth(healthy=True, checked_at=utc_now())

    async def connect(self):
        return None

    async def search_messages(self, query, cursor=None):
        return type("R", (), {"messages": []})()


class _AuthFailProvider:
    async def health_check(self):
        raise RuntimeError("AuthenticationRequired: credentials expired")

    async def connect(self):
        raise RuntimeError("AuthenticationRequired: credentials expired")

    async def search_messages(self, query, cursor=None):
        raise RuntimeError("AuthenticationRequired: credentials expired")


class _IngressFailProvider(_HealthyProvider):
    async def search_messages(self, query, cursor=None):
        raise RuntimeError("provider timeout")


def test_gmail_adapter_healthy_auth_and_connectivity():
    adapter = GmailHealthAdapter(
        account_id="acc1",
        provider_factory=lambda: _HealthyProvider(),
        runtime_alive=True,
    )
    auth = asyncio.run(adapter.run_probe(ProbeCategory.AUTH, mode=SyntheticProbeMode.SAFE_ENDPOINT))
    conn = asyncio.run(adapter.run_probe(ProbeCategory.CONNECTIVITY, mode=SyntheticProbeMode.SAFE_ENDPOINT))
    assert auth.outcome is ProbeOutcome.PASSED
    assert conn.outcome is ProbeOutcome.PASSED


def test_gmail_adapter_auth_failure():
    adapter = GmailHealthAdapter(
        account_id="acc1",
        provider_factory=lambda: _AuthFailProvider(),
        runtime_alive=True,
    )
    result = asyncio.run(adapter.run_probe(ProbeCategory.AUTH, mode=SyntheticProbeMode.SAFE_ENDPOINT))
    assert result.outcome is ProbeOutcome.FAILED
    assert result.reason_code in {
        HealthReasonCode.AUTH_EXPIRED,
        HealthReasonCode.PROBE_FAILED,
    }


def test_gmail_adapter_never_sends_active_egress_by_default():
    adapter = GmailHealthAdapter(
        account_id="acc1",
        provider_factory=lambda: _HealthyProvider(),
        synthetic_mode=SyntheticProbeMode.ACTIVE,
    )
    result = adapter._probe_egress(SyntheticProbeMode.ACTIVE)
    assert result.outcome is ProbeOutcome.SKIPPED


def test_gmail_adapter_egress_from_receipts():
    adapter = GmailHealthAdapter(
        account_id="acc1",
        provider_factory=lambda: _HealthyProvider(),
    )
    adapter.record_delivery(
        DeliveryReceipt(
            message_id="m1",
            connector_id=adapter.connector_id,
            state=DeliveryState.PROVIDER_ACCEPTED,
            submitted_at=utc_now(),
        )
    )
    result = adapter._probe_egress(SyntheticProbeMode.PASSIVE)
    assert result.outcome is ProbeOutcome.PASSED


def test_telegram_false_online_runtime_alive_ingress_broken():
    adapter = TelegramHealthAdapter(
        enabled=True,
        transport="polling",
        runtime_alive=True,
        client_factory=lambda: type("C", (), {"get_me": lambda self: asyncio.sleep(0, result=type("I", (), {"id": 1, "username": "bot"})())})(),
        status_provider=lambda: {"running": False, "last_error": "poller dead"},
    )
    # Force auth ok signals
    adapter._last_auth_ok = True
    adapter._last_transport_ok = False
    result = adapter._probe_ingress()
    assert result.outcome is ProbeOutcome.FAILED
    assert result.reason_code in {
        HealthReasonCode.INGRESS_STALLED,
        HealthReasonCode.PROCESS_ONLY_ALIVE,
    }


def test_telegram_webhook_subscription_missing():
    async def webhook_info():
        return {"url": "", "last_error_message": None}

    adapter = TelegramHealthAdapter(
        enabled=True,
        transport="webhook",
        runtime_alive=True,
        webhook_info_provider=webhook_info,
    )
    result = asyncio.run(adapter.run_probe(ProbeCategory.SUBSCRIPTION, mode=SyntheticProbeMode.SAFE_ENDPOINT))
    assert result.outcome is ProbeOutcome.FAILED
    assert result.reason_code is HealthReasonCode.SUBSCRIPTION_MISSING


def test_supervisor_pauses_and_resumes_once(tmp_path: Path):
    supervisor = ExecutionSupervisor(
        ExecutionSupervisorConfig(root=tmp_path / "execution", startup_recovery=False)
    )
    task = supervisor.create_task(
        task_id="task_1",
        routing_decision_id="decision_test",
        side_effect_classification=SideEffectClassification.IDEMPOTENT,
        idempotency_key="idem-1",
        workspace_path=str(tmp_path),
    )
    # Move to RUNNING so connector suspension is allowed
    def mark_running(t: TaskRecord) -> None:
        t.state = ExecutionState.RUNNING
        t.required_connector_ids = ["gmail:acc"]
        t.checkpoint_id = "checkpoint_1"

    supervisor.store.update_task(task.task_id, mark_running)
    checkpoint = CheckpointRecord(
        checkpoint_id="checkpoint_1",
        task_id=task.task_id,
        attempt_id="attempt_1",
        state_version=0,
    )
    supervisor.store.save_checkpoint(checkpoint)

    dependents = [{"task_id": task.task_id, "checkpoint_id": "checkpoint_1"}]
    bridge = ConnectorSupervisorBridge(
        supervisor=supervisor,
        list_dependent_tasks=lambda cid: dependents if cid == "gmail:acc" else [],
    )
    paused = bridge.on_health_change(
        "gmail:acc",
        ConnectorHealthState.HEALTHY,
        ConnectorHealthState.OFFLINE,
    )
    assert paused == [task.task_id]
    updated = supervisor.store.get_task(task.task_id)
    assert updated.state is ExecutionState.WAITING
    assert updated.waiting_reason == "waiting_for_connector"
    assert updated.waiting_connector_id == "gmail:acc"

    resumed = bridge.on_health_change(
        "gmail:acc",
        ConnectorHealthState.OFFLINE,
        ConnectorHealthState.HEALTHY,
    )
    assert resumed == [task.task_id]
    final = supervisor.store.get_task(task.task_id)
    assert final.state is ExecutionState.QUEUED
    assert final.waiting_connector_id == ""

    # Exactly once: second resume is a no-op
    resumed_again = bridge.resume_dependents("gmail:acc")
    assert resumed_again == []


def test_hitl_auth_intervention_payload_has_no_secrets():
    payload = build_auth_intervention_request(
        connector_id="gmail:acc",
        connector_type="gmail",
        reason_code=HealthReasonCode.AUTH_REVOKED,
        message="token=super-secret-value was revoked",
    )
    text = str(payload)
    assert "super-secret-value" not in text or "***" in text
    assert payload["minimal_context"]["state"] == "auth_required"
    assert "Reconnect" in payload["title"]


def test_hitl_bridge_creates_intervention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    created: list[Any] = []

    class FakeInbox:
        def create(self, request):
            created.append(request)
            return type("Item", (), {"inbox_item_id": "inbox_1"})()

    bridge = ConnectorHitlBridge(inbox_service=FakeInbox())
    inbox_id = bridge.request_auth_intervention(
        connector_id="gmail:acc",
        connector_type="gmail",
        reason_code=HealthReasonCode.AUTH_REVOKED,
        message="token revoked",
    )
    assert inbox_id == "inbox_1"
    assert created
    # Deduped
    assert bridge.request_auth_intervention(
        connector_id="gmail:acc",
        connector_type="gmail",
        reason_code=HealthReasonCode.AUTH_REVOKED,
        message="token revoked again",
    ) == "inbox_1"


def test_transactional_bridge_blocks_webhook_without_gateway():
    bridge = ConnectorTransactionalBridge(action_gateway=None)
    assert bridge.authorize("telegram", RecoveryActionKind.TRANSPORT_RECONNECT, "x") is True
    assert bridge.authorize("telegram", RecoveryActionKind.WEBHOOK_REREGISTER, "x") is False


def test_transactional_bridge_allows_when_gateway_approves():
    class GW:
        def evaluate(self, proposal):
            return True

    bridge = ConnectorTransactionalBridge(action_gateway=GW())
    assert bridge.authorize("telegram", RecoveryActionKind.WEBHOOK_REREGISTER, "missing") is True


def test_manager_hitl_on_auth_required(tmp_path: Path):
    hitl_calls: list[tuple] = []

    def hitl(connector_id, reason, message):
        hitl_calls.append((connector_id, reason, message))
        return "inbox_x"

    manager = ConnectorHealthManager(
        config=ConnectorHealthConfig(storage_root=str(tmp_path / "connectors"), recovery_enabled=True),
        hitl_callback=hitl,
    )
    manager.start()

    class AuthAdapter:
        connector_id = "gmail:bad"
        connector_type = "gmail"

        def health_capabilities(self):
            from mana_agent.connectors.health.models import ConnectorHealthCapabilities
            return ConnectorHealthCapabilities(auth=True, connectivity=True)

        def supported_probe_categories(self):
            return [ProbeCategory.AUTH, ProbeCategory.CONNECTIVITY]

        def synthetic_probe_mode(self):
            return SyntheticProbeMode.PASSIVE

        def collect_signals(self):
            from mana_agent.connectors.health.models import PathSignals
            return PathSignals(runtime_alive=True)

        async def run_probe(self, category, *, mode):
            return ProbeResult(
                category=category,
                outcome=ProbeOutcome.FAILED,
                reason_code=HealthReasonCode.AUTH_REVOKED,
                message="revoked",
            )

        def list_recovery_actions(self, reason_codes):
            return []

        async def execute_recovery(self, action):
            return ProbeResult(category=ProbeCategory.AUTH, outcome=ProbeOutcome.FAILED)

        def recent_delivery_receipts(self, *, limit=20):
            return []

        def is_enabled(self):
            return True

        def describe(self):
            return {}

    manager.register(AuthAdapter())
    report = asyncio.run(manager.probe("gmail:bad", force=True))
    assert report.state is ConnectorHealthState.AUTH_REQUIRED
    assert hitl_calls


def test_format_status_report_readable():
    from mana_agent.connectors.health import format_status_report
    from mana_agent.connectors.health.models import ConnectorHealthReport, CapabilitySignal

    text = format_status_report(
        ConnectorHealthReport(
            connector_id="telegram",
            connector_type="telegram",
            state=ConnectorHealthState.DEGRADED,
            auth=CapabilitySignal.OK,
            ingress=CapabilitySignal.FAILED,
            egress=CapabilitySignal.OK,
            reason_code=HealthReasonCode.INGRESS_STALLED,
            message="subscription heartbeat missing",
        )
    )
    assert "DEGRADED" in text
    assert "FAILED" in text


def test_cli_status_probes_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Registration alone stays UNKNOWN; status must probe to leave STARTUP_PENDING."""
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    from typer.testing import CliRunner

    from mana_agent.commands.cli_internal import app
    from mana_agent.commands import connectors_cli  # noqa: F401
    from mana_agent.connectors.health.manager import ConnectorHealthManager
    from mana_agent.connectors.health.models import CapabilitySignal, PathSignals

    class AlwaysHealthy:
        connector_id = "fake:status"
        connector_type = "fake"

        def health_capabilities(self):
            from mana_agent.connectors.health.models import ConnectorHealthCapabilities
            return ConnectorHealthCapabilities(auth=True, connectivity=True)

        def supported_probe_categories(self):
            return [ProbeCategory.AUTH, ProbeCategory.CONNECTIVITY]

        def synthetic_probe_mode(self):
            return SyntheticProbeMode.PASSIVE

        def collect_signals(self):
            return PathSignals(runtime_alive=True, transport_connected=True, authenticated=CapabilitySignal.OK)

        async def run_probe(self, category, *, mode):
            return ProbeResult(category=category, outcome=ProbeOutcome.PASSED, message="ok")

        def list_recovery_actions(self, reason_codes):
            return []

        async def execute_recovery(self, action):
            return ProbeResult(category=ProbeCategory.AUTH, outcome=ProbeOutcome.PASSED)

        def recent_delivery_receipts(self, *, limit=20):
            return []

        def is_enabled(self):
            return True

        def describe(self):
            return {}

    def fake_bootstrap(**kwargs):
        from mana_agent.connectors.health import reset_health_manager, get_health_manager
        from mana_agent.connectors.health.config import ConnectorHealthConfig

        reset_health_manager()
        manager = ConnectorHealthManager(
            config=ConnectorHealthConfig(storage_root=str(tmp_path / "connectors"))
        )
        manager.start()
        manager.register(AlwaysHealthy())
        return manager

    monkeypatch.setattr(connectors_cli, "bootstrap_health_manager", fake_bootstrap)
    monkeypatch.setattr(connectors_cli, "reset_health_manager", lambda: None)

    # Cached-only stays UNKNOWN
    no_probe = CliRunner().invoke(app, ["connectors", "status", "--no-probe"])
    assert no_probe.exit_code == 0
    assert "UNKNOWN" in no_probe.output or "STARTUP_PENDING" in no_probe.output or "Registered" in no_probe.output

    probed = CliRunner().invoke(app, ["connectors", "status"])
    assert probed.exit_code == 0
    assert "HEALTHY" in probed.output
