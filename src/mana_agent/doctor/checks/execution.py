from __future__ import annotations

from mana_agent.config.settings import Settings
from mana_agent.doctor.models import DoctorContext, DoctorFinding, Severity
from mana_agent.execution.config import ExecutionConfig, build_provider_registry
from mana_agent.execution.manager import run_sync


def providers(context: DoctorContext) -> list[DoctorFinding]:
    del context
    try:
        config = ExecutionConfig.from_settings(Settings())
        registry = build_provider_registry(config)
    except Exception as exc:
        return [DoctorFinding("execution/providers", Severity.ERROR, "Execution provider configuration", str(exc), "Correct MANA_EXECUTION_* configuration; no fallback provider will be used.")]
    findings: list[DoctorFinding] = []
    for name in registry.names():
        if not config.providers[name].enabled:
            findings.append(DoctorFinding("execution/providers", Severity.INFO, f"Execution provider: {name}", "Disabled by configuration."))
            continue
        health = run_sync(registry.get(name).healthcheck())
        findings.append(DoctorFinding(
            "execution/providers", Severity.INFO if health.available else Severity.ERROR,
            f"Execution provider: {name}", health.message or ("Available." if health.available else "Unavailable."),
            None if health.available else "Install or configure the provider, or disable it explicitly.",
        ))
    return findings
