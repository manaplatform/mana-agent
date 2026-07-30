"""Read-oriented CLI for the enrolled server registry and audit trail."""

from __future__ import annotations

import json

import typer

from mana_agent.remote_execution.profiles import get_profile
from mana_agent.server.credentials import register_key_path
from mana_agent.server.models import ServerDefinition
from mana_agent.server.service import ServerManagementService


server_app = typer.Typer(
    help="Inspect and manage explicitly enrolled Linux servers.",
    no_args_is_help=True,
)


def _safe_server(server) -> dict[str, object]:
    return server.model_dump(
        mode="json",
        exclude={"credential_ref", "known_hosts_file", "host_key_fingerprint"},
    )


@server_app.command("enroll")
def server_enroll(
    profile: str,
    server_id: str = typer.Option(..., "--server-id"),
    mode: str = typer.Option("inspect_only", "--mode"),
    capability: list[str] | None = typer.Option(None, "--capability"),
    provider: str = typer.Option("ssh", "--provider"),
) -> None:
    """Enroll a host-key-pinned SSH profile without copying private-key material."""
    if mode not in {"inspect_only", "managed_admin", "trusted_admin"}:
        raise typer.BadParameter("mode must be inspect_only, managed_admin, or trusted_admin")
    ssh = get_profile(profile)
    if not ssh.host_key_fingerprint or not ssh.known_hosts_path().is_file():
        raise typer.BadParameter(
            "SSH profile must have an explicitly trusted host key. Run `mana-agent ssh trust-host` first."
        )
    credential_ref = None
    auth_method = "ssh_agent"
    if ssh.identity_file:
        credential_ref = register_key_path(server_id, ssh.identity_file)
        auth_method = "ssh_key"
    caps = set(capability or ["inspect"])
    ServerManagementService().registry.add(
        ServerDefinition(
            server_id=server_id,
            name=ssh.name,
            host=ssh.host,
            port=ssh.port,
            username=ssh.user,
            auth_method=auth_method,
            credential_ref=credential_ref,
            provider=provider,
            allowed_capabilities=caps,
            mode=mode,
            host_key_fingerprint=ssh.host_key_fingerprint,
            known_hosts_file=str(ssh.known_hosts_path()),
            connect_timeout_seconds=ssh.connect_timeout_seconds,
        )
    )
    typer.echo(f"Enrolled server {ssh.name!r} ({server_id}) in {mode} mode.")


@server_app.command("list")
def server_list() -> None:
    """List enrolled servers without credential or host-key details."""
    typer.echo(json.dumps([_safe_server(item) for item in ServerManagementService().list_servers()], indent=2, default=str))


@server_app.command("status")
def server_status(server: str) -> None:
    """Show enrollment, permission mode, provider, and last connection state."""
    typer.echo(json.dumps(_safe_server(ServerManagementService().server(server)), indent=2, default=str))


@server_app.command("inspect")
def server_inspect(server: str) -> None:
    """Show enrollment metadata; live health uses model-selected server_inspect."""
    enrolled = ServerManagementService().server(server)
    payload = _safe_server(enrolled)
    payload["live_health"] = (
        "Use chat to request live inspection. It requires a validated ServerActionDecision "
        "and will not fall back to an unreviewed command bundle."
    )
    typer.echo(json.dumps(payload, indent=2, default=str))


@server_app.command("logs")
def server_logs(server: str, limit: int = typer.Option(100, "--limit", min=1, max=10_000)) -> None:
    """Read redacted server audit events."""
    typer.echo(json.dumps(ServerManagementService().logs(server, limit=limit), indent=2, default=str))


@server_app.command("remove")
def server_remove(server: str, yes: bool = typer.Option(False, "--yes")) -> None:
    """Remove registry metadata only; this never deletes provider infrastructure."""
    service = ServerManagementService()
    enrolled = service.server(server)
    if not yes and not typer.confirm(
        f"Remove enrollment {enrolled.name!r} ({enrolled.server_id})? The remote server will not be deleted.",
        default=False,
    ):
        raise typer.Abort()
    removed = service.remove_server(enrolled.server_id)
    typer.echo(f"Removed server enrollment {removed.name!r} ({removed.server_id}).")
