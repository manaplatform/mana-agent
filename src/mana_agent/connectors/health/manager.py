"""Central connector health manager: probes, recovery, circuit breakers, incidents."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Callable

from .circuit_breaker import CircuitBreaker
from .config import ConnectorHealthConfig, load_connector_health_config
from .contracts import HealthProbeable
from .events import HealthEventRouter
from .incidents import IncidentTracker
from .metrics import compute_slo_metrics
from .models import (
    CapabilitySignal,
    CircuitState,
    ConnectorHealthReport,
    ConnectorHealthSnapshot,
    ConnectorHealthState,
    DeliveryReceipt,
    HealthReasonCode,
    PathSignals,
    ProbeCategory,
    ProbeOutcome,
    ProbeResult,
    RecoveryActionKind,
    RecoveryAttempt,
    SyntheticProbeMode,
    UnavailableStates,
    utc_now,
)
from .probes import assert_probe_safety, timed_probe
from .recovery import (
    BackoffPolicy,
    build_recovery_attempt,
    is_auth_terminal,
    next_recovery_time,
    select_recovery_action,
)
from .state_machine import derive_state, next_state_after_probe
from .storage import ConnectorHealthStore

logger = logging.getLogger(__name__)

HitlCallback = Callable[[str, HealthReasonCode, str], str | None]
TransactionalCallback = Callable[[str, RecoveryActionKind, str], bool]
SupervisorCallback = Callable[[str, ConnectorHealthState, ConnectorHealthState], None]


class ConnectorHealthManager:
    """Registers connectors, schedules probes, aggregates state, and recovers safely.

    All routine health detection and recovery is deterministic — no LLM calls.
    """

    def __init__(
        self,
        *,
        config: ConnectorHealthConfig | None = None,
        store: ConnectorHealthStore | None = None,
        event_router: HealthEventRouter | None = None,
        clock=utc_now,
        hitl_callback: HitlCallback | None = None,
        transactional_callback: TransactionalCallback | None = None,
        supervisor_callback: SupervisorCallback | None = None,
    ) -> None:
        self.config = config or load_connector_health_config()
        self.store = store or ConnectorHealthStore(self.config.resolved_storage_root())
        self.events = event_router or HealthEventRouter()
        self.clock = clock
        self.hitl_callback = hitl_callback
        self.transactional_callback = transactional_callback
        self.supervisor_callback = supervisor_callback
        self.incidents = IncidentTracker(self.store, clock=clock)
        self._adapters: dict[str, HealthProbeable] = {}
        self._breakers: dict[str, CircuitBreaker] = {}
        self._reports: dict[str, ConnectorHealthReport] = {}
        self._recovery_attempts: dict[str, int] = {}
        self._next_recovery_at: dict[str, datetime] = {}
        self._rate_limited_until: dict[str, datetime] = {}
        self._shutting_down = False
        self._started = False
        self._lock = threading.RLock()
        self._scheduler_task: asyncio.Task | None = None
        self._backoff = BackoffPolicy(
            initial_delay=self.config.initial_backoff_seconds,
            maximum_delay=self.config.max_backoff_seconds,
            maximum_attempts=self.config.max_recovery_attempts,
            reset_after_success=self.config.reset_after_success,
        )

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Load persisted state and mark connectors unknown until first probe."""
        with self._lock:
            if self._started:
                return
            self.incidents.load_open()
            for snapshot in self.store.list_snapshots():
                report = snapshot.report
                # Never trust a previous healthy mark without re-verification.
                if report.state is ConnectorHealthState.HEALTHY:
                    report = report.model_copy(
                        update={
                            "state": ConnectorHealthState.UNKNOWN,
                            "reason_code": HealthReasonCode.STARTUP_PENDING,
                            "message": "Restored after restart; awaiting health verification",
                        }
                    )
                self._reports[snapshot.connector_id] = report
                breaker = self._breaker_for(snapshot.connector_id)
                breaker.restore(
                    state=snapshot.circuit_state,
                    failure_count=snapshot.circuit_failure_count,
                    opened_at=snapshot.circuit_opened_at,
                )
                self._recovery_attempts[snapshot.connector_id] = snapshot.recovery_attempts
                if snapshot.next_recovery_at:
                    self._next_recovery_at[snapshot.connector_id] = snapshot.next_recovery_at
                if snapshot.rate_limited_until:
                    self._rate_limited_until[snapshot.connector_id] = snapshot.rate_limited_until
            self._started = True
            self.store.append_event("connector.health.manager.started", {"connectors": list(self._reports)})
            self.events.emit_log("connector.health.manager.started", connectors=list(self._adapters))

    def stop(self) -> None:
        """Clean shutdown: stop probes, persist final state, avoid false incidents."""
        with self._lock:
            self._shutting_down = True
            if self._scheduler_task is not None:
                self._scheduler_task.cancel()
                self._scheduler_task = None
            for connector_id, report in list(self._reports.items()):
                final = report.model_copy(
                    update={
                        "reason_code": HealthReasonCode.SHUTTING_DOWN,
                        "message": "Clean shutdown",
                    }
                )
                self._persist(connector_id, final, shutting_down=True)
            self.store.rotate(
                incident_retention_days=self.config.incident_retention_days,
                probe_retention_days=self.config.probe_log_retention_days,
            )
            self._started = False
            self.events.emit_log("connector.health.manager.stopped")

    async def start_background(self) -> None:
        self.start()
        if not self.config.enabled:
            return
        if self._scheduler_task is None or self._scheduler_task.done():
            self._scheduler_task = asyncio.create_task(self._scheduler_loop(), name="connector-health")

    # --- registration ------------------------------------------------------

    def register(self, adapter: HealthProbeable) -> None:
        with self._lock:
            self._adapters[adapter.connector_id] = adapter
            self._breaker_for(adapter.connector_id)
            if adapter.connector_id not in self._reports:
                self._reports[adapter.connector_id] = ConnectorHealthReport(
                    connector_id=adapter.connector_id,
                    connector_type=adapter.connector_type,
                    state=ConnectorHealthState.UNKNOWN
                    if adapter.is_enabled()
                    else ConnectorHealthState.DISABLED,
                    reason_code=HealthReasonCode.STARTUP_PENDING
                    if adapter.is_enabled()
                    else HealthReasonCode.DISABLED,
                    message="Registered; awaiting health verification"
                    if adapter.is_enabled()
                    else "Connector is disabled",
                    disabled=not adapter.is_enabled(),
                    synthetic_probe_mode=adapter.synthetic_probe_mode(),
                )
                self._persist(adapter.connector_id, self._reports[adapter.connector_id])

    def unregister(self, connector_id: str) -> None:
        with self._lock:
            self._adapters.pop(connector_id, None)

    def get_adapter(self, connector_id: str) -> HealthProbeable | None:
        return self._adapters.get(connector_id)

    # --- queries -----------------------------------------------------------

    def status(self, connector_id: str | None = None) -> list[ConnectorHealthReport]:
        with self._lock:
            if connector_id:
                report = self._reports.get(connector_id)
                return [report] if report else []
            return [self._reports[key] for key in sorted(self._reports)]

    def get_report(self, connector_id: str) -> ConnectorHealthReport | None:
        return self._reports.get(connector_id)

    def is_available_for(
        self,
        connector_id: str,
        *,
        require_ingress: bool = False,
        require_egress: bool = False,
    ) -> bool:
        report = self._reports.get(connector_id)
        if report is None:
            return False
        if report.state in UnavailableStates:
            return False
        if report.state is ConnectorHealthState.DISABLED:
            return False
        if require_ingress and report.ingress is CapabilitySignal.FAILED:
            return False
        if require_egress and report.egress is CapabilitySignal.FAILED:
            return False
        if report.state is ConnectorHealthState.DEGRADED:
            if require_ingress and report.ingress not in {CapabilitySignal.OK, CapabilitySignal.NOT_APPLICABLE}:
                return False
            if require_egress and report.egress not in {CapabilitySignal.OK, CapabilitySignal.NOT_APPLICABLE}:
                return False
        return report.state in {
            ConnectorHealthState.HEALTHY,
            ConnectorHealthState.DEGRADED,
            ConnectorHealthState.RATE_LIMITED,
        }

    def list_incidents(self, connector_id: str | None = None, *, limit: int = 50):
        return self.store.list_incidents(connector_id=connector_id, limit=limit)

    def metrics(self, connector_id: str):
        return compute_slo_metrics(connector_id, store=self.store, now=self.clock())

    def record_delivery(self, receipt: DeliveryReceipt) -> None:
        self.store.save_receipt(receipt)

    # --- probing -----------------------------------------------------------

    async def probe(self, connector_id: str, *, force: bool = False) -> ConnectorHealthReport:
        adapter = self._adapters.get(connector_id)
        if adapter is None:
            raise KeyError(f"connector not registered: {connector_id}")
        if self._shutting_down:
            report = self._reports.get(connector_id) or ConnectorHealthReport(
                connector_id=connector_id,
                connector_type=adapter.connector_type,
                state=ConnectorHealthState.DISABLED,
                reason_code=HealthReasonCode.SHUTTING_DOWN,
            )
            return report

        if not adapter.is_enabled():
            report = ConnectorHealthReport(
                connector_id=connector_id,
                connector_type=adapter.connector_type,
                state=ConnectorHealthState.DISABLED,
                reason_code=HealthReasonCode.DISABLED,
                message="Connector is disabled",
                disabled=True,
                checked_at=self.clock(),
            )
            return self._commit(connector_id, report, previous=self._reports.get(connector_id))

        breaker = self._breaker_for(connector_id)
        if not force and not breaker.allow_probe():
            previous = self._reports.get(connector_id)
            report = (previous or ConnectorHealthReport(
                connector_id=connector_id,
                connector_type=adapter.connector_type,
            )).model_copy(
                update={
                    "state": ConnectorHealthState.OFFLINE,
                    "circuit_state": CircuitState.OPEN,
                    "reason_code": HealthReasonCode.CIRCUIT_OPEN,
                    "message": "Circuit open; probe skipped",
                    "checked_at": self.clock(),
                }
            )
            return self._commit(connector_id, report, previous=previous)

        rate_until = self._rate_limited_until.get(connector_id)
        if rate_until and rate_until > self.clock() and not force:
            previous = self._reports.get(connector_id)
            report = (previous or ConnectorHealthReport(
                connector_id=connector_id,
                connector_type=adapter.connector_type,
            )).model_copy(
                update={
                    "state": ConnectorHealthState.RATE_LIMITED,
                    "reason_code": HealthReasonCode.RATE_LIMITED,
                    "message": f"Rate limited until {rate_until.isoformat()}",
                    "checked_at": self.clock(),
                }
            )
            return self._commit(connector_id, report, previous=previous)

        mode = adapter.synthetic_probe_mode()
        if mode is SyntheticProbeMode.ACTIVE and not self.config.active_probe_allowed:
            mode = SyntheticProbeMode.PASSIVE

        self.events.emit_log("connector.health.probe.started", connector_id=connector_id)
        categories = adapter.supported_probe_categories()
        results: list[ProbeResult] = []
        for category in categories:
            safety = assert_probe_safety(
                category,
                mode,
                active_probe_allowed=self.config.active_probe_allowed,
                test_channel=self.config.test_channel,
            )
            if safety is not None and category in {ProbeCategory.EGRESS, ProbeCategory.ACKNOWLEDGEMENT}:
                results.append(safety)
                continue
            result = await timed_probe(category, lambda c=category: adapter.run_probe(c, mode=mode))
            results.append(result)
            self.store.append_probe_result(connector_id, result)
            if result.outcome is ProbeOutcome.PASSED:
                self.events.emit_log(
                    "connector.health.probe.passed",
                    connector_id=connector_id,
                    category=category.value,
                )
            elif result.outcome is ProbeOutcome.FAILED:
                self.events.emit_log(
                    "connector.health.probe.failed",
                    connector_id=connector_id,
                    category=category.value,
                    reason_code=result.reason_code.value,
                )
            elif result.outcome is ProbeOutcome.RATE_LIMITED:
                retry_after = result.details.get("retry_after")
                seconds = float(retry_after) if retry_after is not None else self.config.probe_interval_seconds * self.config.rate_limit_probe_multiplier
                self._rate_limited_until[connector_id] = self.clock() + timedelta(seconds=seconds)

        signals = self._signals_from_results(adapter, results)
        previous = self._reports.get(connector_id)
        consecutive = (previous.consecutive_failures if previous else 0)
        any_failed = any(r.outcome is ProbeOutcome.FAILED for r in results)
        any_rate = any(r.outcome is ProbeOutcome.RATE_LIMITED for r in results)
        if any_failed:
            consecutive += 1
            breaker.record_failure()
            self.events.emit_log(
                "connector.circuit.opened" if breaker.state is CircuitState.OPEN else "connector.health.probe.failed",
                connector_id=connector_id,
                circuit=breaker.state.value,
            )
            if breaker.state is CircuitState.OPEN:
                self.events.emit_log("connector.circuit.opened", connector_id=connector_id)
        elif any_rate:
            pass
        else:
            consecutive = 0
            prev_circuit = breaker.state
            breaker.record_success()
            if prev_circuit is not CircuitState.CLOSED and breaker.state is CircuitState.CLOSED:
                self.events.emit_log("connector.circuit.closed", connector_id=connector_id)
            if self.config.reset_after_success:
                self._recovery_attempts[connector_id] = 0

        rate_limited = any_rate or (
            self._rate_limited_until.get(connector_id) is not None
            and self._rate_limited_until[connector_id] > self.clock()
        )
        recovering = (previous.state is ConnectorHealthState.RECOVERING) if previous else False
        derived, reason, message = derive_state(
            signals,
            disabled=not adapter.is_enabled(),
            rate_limited=rate_limited,
            circuit_open=breaker.state is CircuitState.OPEN,
            recovering=recovering,
            consecutive_failures=consecutive,
            failure_threshold=self.config.failure_threshold,
            shutting_down=self._shutting_down,
        )
        # Prefer specific probe reason when present
        for result in results:
            if result.outcome is ProbeOutcome.FAILED and result.reason_code is not HealthReasonCode.NONE:
                reason = result.reason_code
                message = result.message or message
                break
            if result.outcome is ProbeOutcome.RATE_LIMITED:
                reason = HealthReasonCode.RATE_LIMITED
                message = result.message or message
                break

        current_state = previous.state if previous else ConnectorHealthState.UNKNOWN
        state = next_state_after_probe(
            current_state,
            derived,
            consecutive_failures=consecutive,
            failure_threshold=self.config.failure_threshold,
            recovery_enabled=self.config.recovery_enabled,
        )

        latencies = [r.latency_ms for r in results if r.latency_ms is not None]
        latency = sum(latencies) / len(latencies) if latencies else None
        last_healthy = previous.last_healthy_at if previous else None
        last_failure = previous.last_failure_at if previous else None
        if state is ConnectorHealthState.HEALTHY:
            last_healthy = self.clock()
        if any_failed:
            last_failure = self.clock()

        report = ConnectorHealthReport(
            connector_id=connector_id,
            connector_type=adapter.connector_type,
            state=state,
            checked_at=self.clock(),
            last_healthy_at=last_healthy,
            last_failure_at=last_failure,
            consecutive_failures=consecutive,
            latency_ms=latency,
            signals=signals,
            auth=signals.authenticated,
            ingress=signals.ingress_operational,
            egress=signals.egress_operational,
            subscriptions=signals.subscription_operational,
            acknowledgements=signals.acknowledgements_operational,
            reconnect=CapabilitySignal.OK if state is ConnectorHealthState.HEALTHY else CapabilitySignal.UNKNOWN,
            circuit_state=breaker.state,
            reason_code=reason,
            message=message,
            synthetic_probe_mode=mode,
            probe_results=results,
            disabled=False,
        )
        report = report.model_copy(update={"metrics": compute_slo_metrics(connector_id, store=self.store, now=self.clock())})
        committed = self._commit(connector_id, report, previous=previous)

        if (
            self.config.recovery_enabled
            and committed.state in {ConnectorHealthState.DEGRADED, ConnectorHealthState.RECOVERING, ConnectorHealthState.OFFLINE}
            and not is_auth_terminal(committed.reason_code)
        ):
            await self._maybe_recover(adapter, committed)
        elif committed.state is ConnectorHealthState.AUTH_REQUIRED:
            await self._handle_auth_required(adapter, committed)

        return self._reports[connector_id]

    async def probe_all(self, *, force: bool = False) -> list[ConnectorHealthReport]:
        reports: list[ConnectorHealthReport] = []
        for connector_id in list(self._adapters):
            reports.append(await self.probe(connector_id, force=force))
        return reports

    async def recover(self, connector_id: str, *, force: bool = False) -> ConnectorHealthReport:
        adapter = self._adapters.get(connector_id)
        if adapter is None:
            raise KeyError(f"connector not registered: {connector_id}")
        report = self._reports.get(connector_id)
        if report is None:
            report = await self.probe(connector_id, force=True)
        await self._maybe_recover(adapter, report, force=force)
        return await self.probe(connector_id, force=True)

    # --- internal ----------------------------------------------------------

    def _breaker_for(self, connector_id: str) -> CircuitBreaker:
        if connector_id not in self._breakers:
            self._breakers[connector_id] = CircuitBreaker(
                failure_threshold=self.config.circuit_failure_threshold,
                open_seconds=self.config.circuit_open_seconds,
                half_open_max_probes=self.config.circuit_half_open_max_probes,
                clock=self.clock,
            )
        return self._breakers[connector_id]

    def _signals_from_results(self, adapter: HealthProbeable, results: list[ProbeResult]) -> PathSignals:
        base = adapter.collect_signals()
        by_cat = {r.category: r for r in results}

        def map_outcome(category: ProbeCategory, default: CapabilitySignal) -> CapabilitySignal:
            result = by_cat.get(category)
            if result is None:
                return default
            if result.outcome is ProbeOutcome.PASSED:
                return CapabilitySignal.OK
            if result.outcome is ProbeOutcome.FAILED:
                return CapabilitySignal.FAILED
            if result.outcome is ProbeOutcome.RATE_LIMITED:
                return CapabilitySignal.DEGRADED
            if result.outcome is ProbeOutcome.UNSUPPORTED:
                return CapabilitySignal.NOT_APPLICABLE
            return default

        caps = adapter.health_capabilities()
        return PathSignals(
            runtime_alive=base.runtime_alive,
            transport_connected=(
                map_outcome(ProbeCategory.CONNECTIVITY, CapabilitySignal.UNKNOWN) is CapabilitySignal.OK
                or base.transport_connected
            ),
            authenticated=map_outcome(
                ProbeCategory.AUTH,
                base.authenticated if caps.auth else CapabilitySignal.NOT_APPLICABLE,
            ),
            ingress_operational=map_outcome(
                ProbeCategory.INGRESS,
                base.ingress_operational if caps.ingress else CapabilitySignal.NOT_APPLICABLE,
            ),
            egress_operational=map_outcome(
                ProbeCategory.EGRESS,
                base.egress_operational if caps.egress else CapabilitySignal.NOT_APPLICABLE,
            ),
            subscription_operational=map_outcome(
                ProbeCategory.SUBSCRIPTION,
                base.subscription_operational if caps.subscriptions else CapabilitySignal.NOT_APPLICABLE,
            ),
            acknowledgements_operational=map_outcome(
                ProbeCategory.ACKNOWLEDGEMENT,
                base.acknowledgements_operational if caps.acknowledgements else CapabilitySignal.NOT_APPLICABLE,
            ),
        )

    def _commit(
        self,
        connector_id: str,
        report: ConnectorHealthReport,
        *,
        previous: ConnectorHealthReport | None,
    ) -> ConnectorHealthReport:
        previous_state = previous.state if previous else ConnectorHealthState.UNKNOWN
        if report.state not in {
            ConnectorHealthState.HEALTHY,
            ConnectorHealthState.DISABLED,
            ConnectorHealthState.UNKNOWN,
        }:
            incident = self.incidents.ensure_open(
                connector_id=connector_id,
                state=report.state,
                reason_code=report.reason_code,
                message=report.message,
            )
            report = report.model_copy(update={"current_incident_id": incident.incident_id})
        elif report.state is ConnectorHealthState.HEALTHY and previous_state not in {
            ConnectorHealthState.HEALTHY,
            ConnectorHealthState.UNKNOWN,
            ConnectorHealthState.DISABLED,
        }:
            closed = self.incidents.close(connector_id, closing_state=ConnectorHealthState.HEALTHY)
            if closed:
                report = report.model_copy(update={"current_incident_id": closed.incident_id})

        self._reports[connector_id] = report
        self._persist(connector_id, report)
        if previous_state is not report.state:
            event = self.events.health_changed(
                connector_id=connector_id,
                previous_state=previous_state,
                state=report.state,
                reason_code=report.reason_code,
                incident_id=report.current_incident_id,
                message=report.message,
            )
            if event is not None:
                self.store.append_event(
                    "connector.health.changed",
                    event.model_dump(mode="json"),
                )
            if self.supervisor_callback is not None:
                try:
                    self.supervisor_callback(connector_id, previous_state, report.state)
                except Exception:
                    logger.exception("supervisor health callback failed for %s", connector_id)
        return report

    def _persist(
        self,
        connector_id: str,
        report: ConnectorHealthReport,
        *,
        shutting_down: bool = False,
    ) -> None:
        breaker = self._breaker_for(connector_id)
        snapshot = ConnectorHealthSnapshot(
            connector_id=connector_id,
            connector_type=report.connector_type,
            report=report,
            circuit_state=breaker.state,
            circuit_opened_at=breaker.opened_at,
            circuit_failure_count=breaker.failure_count,
            recovery_attempts=self._recovery_attempts.get(connector_id, 0),
            next_probe_at=self.clock() + timedelta(seconds=self.config.probe_interval_seconds),
            next_recovery_at=self._next_recovery_at.get(connector_id),
            rate_limited_until=self._rate_limited_until.get(connector_id),
            shutting_down=shutting_down,
            updated_at=self.clock(),
        )
        self.store.save_snapshot(snapshot)

    async def _maybe_recover(
        self,
        adapter: HealthProbeable,
        report: ConnectorHealthReport,
        *,
        force: bool = False,
    ) -> RecoveryAttempt | None:
        connector_id = adapter.connector_id
        attempt_number = self._recovery_attempts.get(connector_id, 0) + 1
        # Maximum attempts always applies (including force). Force only bypasses
        # the backoff wait so operators are not stuck behind a timer.
        if self._backoff.exhausted(attempt_number):
            self.events.emit_log(
                "connector.recovery.failed",
                connector_id=connector_id,
                reason="retry_exhaustion",
            )
            offline = report.model_copy(
                update={
                    "state": ConnectorHealthState.OFFLINE,
                    "reason_code": HealthReasonCode.RECONNECT_FAILED,
                    "message": "Recovery attempts exhausted",
                }
            )
            self._commit(connector_id, offline, previous=report)
            return None
        not_before = self._next_recovery_at.get(connector_id)
        if not force and not_before and not_before > self.clock():
            return None

        action = select_recovery_action(
            report.reason_code,
            adapter.list_recovery_actions([report.reason_code.value]),
        )
        if action is RecoveryActionKind.NONE:
            return None

        attempt = build_recovery_attempt(
            connector_id=connector_id,
            action=action,
            attempt_number=attempt_number,
            reason_code=report.reason_code,
            clock=self.clock,
        )
        if attempt.requires_transactional_policy:
            allowed = True
            if self.transactional_callback is not None:
                allowed = bool(
                    self.transactional_callback(connector_id, action, report.reason_code.value)
                )
            if not allowed:
                attempt = attempt.model_copy(
                    update={
                        "finished_at": self.clock(),
                        "success": False,
                        "message": "Recovery blocked pending transactional policy approval",
                        "requires_human": True,
                    }
                )
                self._recovery_attempts[connector_id] = attempt_number
                self._next_recovery_at[connector_id] = next_recovery_time(
                    self._backoff,
                    connector_id=connector_id,
                    action=action,
                    attempt_number=attempt_number,
                    now=self.clock(),
                )
                incident = self.incidents.current(connector_id)
                if incident:
                    self.incidents.append(
                        incident,
                        event_type="connector.recovery.blocked_policy",
                        recovery_attempt_id=attempt.recovery_attempt_id,
                        reason_code=report.reason_code,
                        message=attempt.message,
                    )
                return attempt

        recovering = report.model_copy(
            update={
                "state": ConnectorHealthState.RECOVERING,
                "recovery_attempt": attempt,
                "reason_code": HealthReasonCode.RECOVERY_IN_PROGRESS,
                "message": f"Recovery started: {action.value}",
            }
        )
        self._commit(connector_id, recovering, previous=report)
        self.events.emit_log(
            "connector.recovery.started",
            connector_id=connector_id,
            action=action.value,
            attempt=attempt_number,
        )
        incident = self.incidents.current(connector_id) or self.incidents.ensure_open(
            connector_id=connector_id,
            state=ConnectorHealthState.RECOVERING,
            reason_code=report.reason_code,
        )
        self.incidents.append(
            incident,
            event_type="reconnect_attempt_started",
            recovery_attempt_id=attempt.recovery_attempt_id,
            reason_code=report.reason_code,
            message=f"recovery action {action.value}",
        )

        try:
            result = await adapter.execute_recovery(action)
            success = result.outcome is ProbeOutcome.PASSED
        except Exception as exc:
            success = False
            result = ProbeResult(
                category=ProbeCategory.CONNECTIVITY,
                outcome=ProbeOutcome.FAILED,
                reason_code=HealthReasonCode.RECONNECT_FAILED,
                message=str(exc)[:500],
            )

        attempt = attempt.model_copy(
            update={
                "finished_at": self.clock(),
                "success": success,
                "message": result.message,
                "reason_code": result.reason_code if not success else HealthReasonCode.NONE,
            }
        )
        self._recovery_attempts[connector_id] = 0 if success and self.config.reset_after_success else attempt_number
        if not success:
            self._next_recovery_at[connector_id] = next_recovery_time(
                self._backoff,
                connector_id=connector_id,
                action=action,
                attempt_number=attempt_number,
                now=self.clock(),
            )
            self.events.emit_log(
                "connector.recovery.retry" if not self._backoff.exhausted(attempt_number + 1) else "connector.recovery.failed",
                connector_id=connector_id,
                action=action.value,
                attempt=attempt_number,
            )
            self.incidents.append(
                incident,
                event_type="connector.recovery.failed",
                recovery_attempt_id=attempt.recovery_attempt_id,
                reason_code=result.reason_code,
                message=result.message,
            )
        else:
            self.events.emit_log(
                "connector.recovery.succeeded",
                connector_id=connector_id,
                action=action.value,
            )
            self.incidents.append(
                incident,
                event_type="connector.recovery.succeeded",
                recovery_attempt_id=attempt.recovery_attempt_id,
                message=result.message,
            )
        return attempt

    async def _handle_auth_required(self, adapter: HealthProbeable, report: ConnectorHealthReport) -> None:
        # Attempt a single safe token refresh when available; otherwise escalate.
        actions = adapter.list_recovery_actions([report.reason_code.value])
        if RecoveryActionKind.TOKEN_REFRESH in actions:
            result = await adapter.execute_recovery(RecoveryActionKind.TOKEN_REFRESH)
            if result.outcome is ProbeOutcome.PASSED:
                healthy_probe = await self.probe(adapter.connector_id, force=True)
                if healthy_probe.state is ConnectorHealthState.HEALTHY:
                    return
            auth_report = report.model_copy(
                update={
                    "state": ConnectorHealthState.AUTH_REQUIRED,
                    "reason_code": HealthReasonCode.TOKEN_REFRESH_FAILED
                    if result.outcome is ProbeOutcome.FAILED
                    else report.reason_code,
                    "message": result.message or report.message,
                }
            )
            self._commit(adapter.connector_id, auth_report, previous=report)
        if self.hitl_callback is not None:
            try:
                inbox_id = self.hitl_callback(
                    adapter.connector_id,
                    report.reason_code,
                    report.message,
                )
                if inbox_id:
                    incident = self.incidents.current(adapter.connector_id)
                    if incident:
                        self.incidents.append(
                            incident,
                            event_type="connector.hitl.requested",
                            message=f"inbox_item_id={inbox_id}",
                            reason_code=report.reason_code,
                        )
            except Exception:
                logger.exception("HITL callback failed for %s", adapter.connector_id)

    async def _scheduler_loop(self) -> None:
        while not self._shutting_down and self.config.enabled:
            try:
                await self.probe_all(force=False)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("connector health scheduler iteration failed")
            await asyncio.sleep(self.config.probe_interval_seconds)


_MANAGER: ConnectorHealthManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_health_manager(**kwargs: Any) -> ConnectorHealthManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = ConnectorHealthManager(**kwargs)
            _MANAGER.start()
        return _MANAGER


def reset_health_manager() -> None:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is not None:
            try:
                _MANAGER.stop()
            except Exception:
                pass
        _MANAGER = None
