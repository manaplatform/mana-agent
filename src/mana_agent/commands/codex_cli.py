"""Operational commands for the optional Codex coding backend."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from mana_agent.config.settings import Settings
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.health import check_codex_health
from mana_agent.integrations.codex.provider import (
    CodexCredential, CodexCredentialStore, CodexExecutionMode, CodexUsageStore, CredentialKind,
)

codex_app = typer.Typer(help="Inspect the optional Codex coding backend.", no_args_is_help=True)


def _settings() -> CodexSettings:
    return CodexSettings.from_mana_settings(Settings())


@codex_app.command("status")
def codex_status(
    root: str | None = typer.Option(None, "--root-dir", "--repo", help="Repository to validate."),
) -> None:
    """Report enablement, executable version, and repository access."""

    repository = Path(root or ".").expanduser().resolve()
    report = check_codex_health(_settings(), repository)
    payload = report.model_dump(mode="json")
    store = CodexCredentialStore()
    usage = CodexUsageStore()
    payload["api"] = {"credential": _credential_status(store.load(CredentialKind.API)), "usage": _usage_status(usage.load(CodexExecutionMode.API))}
    payload["subscription"] = {"credential": _credential_status(store.load(CredentialKind.SUBSCRIPTION)), "usage": _usage_status(usage.load(CodexExecutionMode.SUBSCRIPTION))}
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _credential_status(value):
    if value is None:
        return {"authenticated": False}
    return {"authenticated": value.authenticated and not value.expired, "account_identity": value.account_identity, "expires_at": value.expires_at, "refresh_state": value.refresh_state}


def _usage_status(value):
    return None if value is None else value.as_dict()


@codex_app.command("auth")
def codex_auth(
    mode: str = typer.Option(..., "--mode", help="api or subscription"),
    reference: str = typer.Option(..., "--reference", help="Secret backend reference; never the raw credential."),
    account: str = typer.Option("", "--account", help="Non-secret account identity for subscription status."),
) -> None:
    """Register a Codex credential reference without storing token material."""
    try:
        execution_mode = CodexExecutionMode(mode)
    except ValueError as exc:
        raise typer.BadParameter("mode must be api or subscription") from exc
    kind = CredentialKind.API if execution_mode is CodexExecutionMode.API else CredentialKind.SUBSCRIPTION
    CodexCredentialStore().save(CodexCredential(kind, reference.strip(), account_identity=account.strip(), authenticated=True))
    typer.echo(f"Registered {execution_mode.value} Codex credential reference {reference.strip()}.")


@codex_app.command("doctor")
def codex_doctor(
    root: str | None = typer.Option(None, "--root-dir", "--repo", help="Repository to validate."),
) -> None:
    """Run the read-only health check and exit non-zero when unavailable."""

    repository = Path(root or ".").expanduser().resolve()
    report = check_codex_health(_settings(), repository)
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    if not report.healthy:
        raise typer.Exit(code=1)


__all__ = ["codex_app"]
