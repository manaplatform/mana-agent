"""Operational commands for the optional Codex coding backend."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from mana_agent.config.settings import Settings
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.health import check_codex_health
from mana_agent.integrations.codex.provider import (
    CodexAuthenticationService, CodexCredential, CodexCredentialStore, CodexExecutionMode, CodexUsageStore, CredentialKind,
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
    api_usage = usage.load(CodexExecutionMode.API)
    subscription_usage = usage.load(CodexExecutionMode.SUBSCRIPTION)
    payload["api"] = {"status": _credential_status(store.load(CredentialKind.API)), "usage": _usage_status(api_usage), "cost": api_usage.estimated_cost_usd if api_usage else None}
    subscription_status = CodexAuthenticationService(store).status()
    payload["subscription"] = {"account": subscription_status.account_identity, "authenticated": subscription_status.authenticated, "status": subscription_status.status.value, "quota_remaining": subscription_usage.quota_remaining if subscription_usage else None, "reset_at": subscription_usage.reset_at if subscription_usage else None, "usage": _usage_status(subscription_usage)}
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _credential_status(value):
    if value is None:
        return {"authenticated": False}
    return {"authenticated": value.authenticated and not value.expired, "account_identity": value.account_identity, "expires_at": value.expires_at, "refresh_state": value.refresh_state, "status": "expired" if value.expired else "authenticated" if value.authenticated else "invalid"}


def _usage_status(value):
    return {"capacity_status": "unknown", "available": False, "capacity_score": 0.0} if value is None else value.as_dict() | {"capacity_score": round(value.quota_health, 3)}


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
    normalized_reference = reference.strip()
    if not normalized_reference.startswith(("env://", "mana-secret://")):
        raise typer.BadParameter("reference must be env://<name> or mana-secret://<id>; raw secrets are not accepted")
    CodexCredentialStore().save(CodexCredential(kind, normalized_reference, account_identity=account.strip(), authenticated=True))
    typer.echo(f"Registered {execution_mode.value} Codex credential reference {normalized_reference}.")


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
