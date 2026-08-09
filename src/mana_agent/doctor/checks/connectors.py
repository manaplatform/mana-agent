"""Doctor checks for connector credentials, health manager, and path signals."""

from __future__ import annotations

from mana_agent.doctor.models import DoctorContext, DoctorFinding, Severity


def connector_health(_context: DoctorContext) -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []
    try:
        from mana_agent.connectors.health import (
            ConnectorHealthState,
            bootstrap_health_manager,
            reset_health_manager,
        )

        reset_health_manager()
        manager = bootstrap_health_manager()
    except Exception as exc:
        return [
            DoctorFinding(
                "connectors/health",
                Severity.ERROR,
                "Connector health manager failed to start",
                str(exc),
                "Inspect ~/.mana/connectors and managed configuration.",
            )
        ]

    reports = manager.status()
    if not reports:
        findings.append(
            DoctorFinding(
                "connectors/health",
                Severity.INFO,
                "No connectors registered for health monitoring",
                "Gmail accounts and Telegram appear unconfigured.",
            )
        )
        return findings

    findings.append(
        DoctorFinding(
            "connectors/health",
            Severity.INFO,
            "Connector health manager is available",
            f"Tracking {len(reports)} connector(s) under the Mana data directory.",
        )
    )

    for report in reports:
        if report.state is ConnectorHealthState.DISABLED:
            continue
        if report.state is ConnectorHealthState.AUTH_REQUIRED:
            findings.append(
                DoctorFinding(
                    "connectors/health",
                    Severity.ERROR,
                    f"Connector {report.connector_id} requires authentication",
                    report.message or report.reason_code.value,
                    "Reconnect the account (e.g. mana-agent connector email reconnect ...).",
                    details={
                        "connector_id": report.connector_id,
                        "state": report.state.value,
                        "reason_code": report.reason_code.value,
                    },
                )
            )
        elif report.state in {
            ConnectorHealthState.OFFLINE,
            ConnectorHealthState.DEGRADED,
            ConnectorHealthState.RECOVERING,
        }:
            findings.append(
                DoctorFinding(
                    "connectors/health",
                    Severity.WARNING,
                    f"Connector {report.connector_id} is {report.state.value}",
                    report.message or report.reason_code.value,
                    "Run `mana-agent connectors health <name>` and inspect incidents.",
                    details={
                        "connector_id": report.connector_id,
                        "state": report.state.value,
                        "auth": report.auth.value,
                        "ingress": report.ingress.value,
                        "egress": report.egress.value,
                        "runtime_alive": report.signals.runtime_alive,
                        "transport_connected": report.signals.transport_connected,
                    },
                )
            )
        elif report.state is ConnectorHealthState.UNKNOWN:
            findings.append(
                DoctorFinding(
                    "connectors/health",
                    Severity.WARNING,
                    f"Connector {report.connector_id} has not been health-verified",
                    "Startup state is unknown until a successful probe completes.",
                    "Run `mana-agent connectors health <name>`.",
                )
            )
        # False-online regression: runtime alive without verified path
        if report.signals.runtime_alive and report.state is ConnectorHealthState.HEALTHY:
            if report.auth.value == "failed" or report.ingress.value == "failed":
                findings.append(
                    DoctorFinding(
                        "connectors/health",
                        Severity.ERROR,
                        f"Connector {report.connector_id} false-online signal",
                        "Runtime is alive but a required path signal failed while state claims healthy.",
                        "This is a health derivation bug; open an issue with the health report JSON.",
                    )
                )
    return findings


def connector_credentials(_context: DoctorContext) -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []
    try:
        from mana_agent.connectors.email.config import load_accounts

        accounts = load_accounts()
    except Exception as exc:
        findings.append(
            DoctorFinding(
                "connectors/credentials",
                Severity.WARNING,
                "Unable to load email accounts",
                str(exc),
            )
        )
        accounts = []
    for account in accounts:
        if account.provider == "gmail" and not account.secret_ref:
            findings.append(
                DoctorFinding(
                    "connectors/credentials",
                    Severity.ERROR,
                    f"Gmail account {account.id} is missing credentials",
                    "secret_ref is empty",
                    "Reconnect the Gmail account.",
                )
            )
        elif account.provider == "gmail":
            findings.append(
                DoctorFinding(
                    "connectors/credentials",
                    Severity.INFO,
                    f"Gmail account {account.id} has a credential reference",
                    account.address.address if account.address else account.id,
                )
            )
    try:
        from mana_agent.connectors.telegram.config import load_telegram_config

        config = load_telegram_config()
        if config.enabled:
            token_present = bool(getattr(config, "bot_token", "") or config.bot_token_secret_ref)
            if not token_present:
                findings.append(
                    DoctorFinding(
                        "connectors/credentials",
                        Severity.ERROR,
                        "Telegram is enabled without a bot token",
                        "Configure the bot token via keyring or environment.",
                        "Run mana-agent connector telegram setup.",
                    )
                )
            else:
                findings.append(
                    DoctorFinding(
                        "connectors/credentials",
                        Severity.INFO,
                        "Telegram bot token reference is present",
                        f"transport={config.effective_transport}",
                    )
                )
    except Exception as exc:
        findings.append(
            DoctorFinding(
                "connectors/credentials",
                Severity.WARNING,
                "Unable to inspect Telegram configuration",
                str(exc),
            )
        )
    if not findings:
        findings.append(
            DoctorFinding(
                "connectors/credentials",
                Severity.INFO,
                "No connector credentials configured",
                "Optional connectors can be added when needed.",
            )
        )
    return findings
