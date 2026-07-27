from __future__ import annotations

from mana_agent.config.settings import Settings
from mana_agent.doctor.models import DoctorContext, DoctorFinding, Severity
from mana_agent.integrations.codex.config import CodexSettings


def binary(context: DoctorContext) -> list[DoctorFinding]:
    del context
    settings = CodexSettings.from_mana_settings(Settings())
    if not settings.enabled:
        return []
    from shutil import which
    from pathlib import Path
    resolved = which(settings.codex_bin)
    if not resolved and not (Path(settings.codex_bin).is_file() and Path(settings.codex_bin).stat().st_mode & 0o111):
        return [DoctorFinding("integrations/codex-binary", Severity.WARNING, "Codex executable unavailable", f"Configured MANA_CODEX_BIN could not be resolved: {settings.codex_bin}", "Install Codex or configure MANA_CODEX_BIN. No fallback coding backend will be used.")]
    return [DoctorFinding("integrations/codex-binary", Severity.INFO, "Codex executable", resolved or settings.codex_bin)]


def protocol(context: DoctorContext) -> list[DoctorFinding]:
    """Run the existing read-only Codex preflight only in deep mode."""
    settings = CodexSettings.from_mana_settings(Settings())
    if not settings.enabled:
        return []
    from mana_agent.integrations.codex.health import check_codex_health

    report = check_codex_health(settings, context.repository)
    if report.healthy:
        return [DoctorFinding("integrations/codex-protocol", Severity.INFO, "Codex app-server preflight", f"{report.version} supports the app-server protocol.")]
    return [DoctorFinding("integrations/codex-protocol", Severity.WARNING, "Codex app-server preflight failed", "; ".join(report.errors), "Correct MANA_CODEX_BIN or the reported Codex installation issue. No fallback coding backend will be used.", details={"version": report.version})]
