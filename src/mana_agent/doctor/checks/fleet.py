"""Read-only Fleet configuration, state, identity, and provider checks."""

from __future__ import annotations

from mana_agent.config.settings import Settings
from mana_agent.doctor.models import DoctorContext, DoctorFinding, Severity
from mana_agent.execution.config import ExecutionConfig, build_provider_registry
from mana_agent.fleet import FleetConfig, FleetRegistry, FleetStore
from mana_agent.fleet.health import effective_status


def configuration(_context: DoctorContext) -> list[DoctorFinding]:
    try:
        settings = Settings()
        config = FleetConfig.from_settings(settings)
        store = FleetStore(config.root)
        registry = FleetRegistry(store, config)
        providers = set(build_provider_registry(ExecutionConfig.from_settings(settings)).names())
    except Exception as exc:
        return [DoctorFinding(
            "fleet/configuration", Severity.ERROR, "Fleet configuration is invalid",
            str(exc), "Correct the reported MANA_FLEET_* or execution-provider setting.",
        )]
    if not config.enabled:
        return [DoctorFinding(
            "fleet/configuration", Severity.INFO, "Fleet is disabled",
            "Existing local Eval, execution, SSH, and reverse-worker behavior is unchanged.",
        )]
    findings: list[DoctorFinding] = []
    for worker in registry.list():
        status = effective_status(
            worker,
            heartbeat_timeout_seconds=config.heartbeat_timeout_seconds,
            capability_ttl_seconds=config.capability_ttl_seconds,
        )
        if status.value != "connected":
            findings.append(DoctorFinding(
                "fleet/configuration", Severity.WARNING,
                f"Fleet worker {worker.worker_id} is {status.value}",
                "The worker will not receive new verification jobs.",
                "Check worker service, transport, identity, heartbeat, and capability freshness.",
            ))
        missing = set(worker.capabilities.execution_providers) - providers
        if missing:
            findings.append(DoctorFinding(
                "fleet/configuration", Severity.WARNING,
                f"Fleet worker {worker.worker_id} uses unregistered providers",
                ", ".join(sorted(missing)),
                "Enable and register the exact provider; no alternate provider will be selected.",
            ))
    if not registry.list():
        findings.append(DoctorFinding(
            "fleet/configuration", Severity.WARNING, "Fleet has no workers",
            "Fleet is enabled but no authenticated capability inventory is registered.",
            "Enroll a worker and run `mana-agent fleet doctor`.",
        ))
    return findings
