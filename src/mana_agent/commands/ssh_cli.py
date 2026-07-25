"""Direct SSH profile commands backed by the local OpenSSH client."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import uuid
from pathlib import Path

import typer

from mana_agent.remote_execution.models import RemoteCommand, RemoteExecutionRequest
from mana_agent.remote_execution.profiles import (
    SSHProfile,
    find_ssh_executable,
    get_profile,
    inspect_host_key,
    list_profiles,
    remove_profile,
    save_profile,
    trust_scanned_host_key,
)
from mana_agent.remote_execution.providers.local_ssh import LocalSSHProvider


ssh_app = typer.Typer(help="Manage direct SSH targets; private keys remain in the system SSH agent or filesystem.", no_args_is_help=True)


def _profile(
    name: str, host: str, user: str, port: int, identity: str | None, use_agent: bool,
    timeout: int, ssh_only: bool,
) -> SSHProfile:
    if bool(identity) == bool(use_agent):
        raise typer.BadParameter("Specify exactly one of --identity or --use-agent.")
    return SSHProfile(
        name=name, host=host, user=user, port=port, identity_file=identity,
        use_agent=use_agent, connect_timeout_seconds=timeout, ssh_only=ssh_only,
    )


def _request(profile: SSHProfile, command: list[str]) -> RemoteExecutionRequest:
    if not command:
        raise typer.BadParameter("Provide a remote command after `--`.")
    return RemoteExecutionRequest(
        job_id=f"ssh_{uuid.uuid4().hex}", session_id="cli", provider="remote-ssh",
        target=profile.target(), authentication=profile.authentication(), command=RemoteCommand(argv=command),
        timeout_seconds=profile.connect_timeout_seconds, connect_timeout_seconds=profile.connect_timeout_seconds,
        known_hosts_file=str(profile.known_hosts_path()),
    )


async def _run(profile: SSHProfile, command: list[str]) -> int:
    request = _request(profile, command)
    def emit(event) -> None:
        if event.kind in {"stdout", "stderr"}:
            typer.echo(str(event.data.get("chunk", "")), nl=False, err=event.kind == "stderr")
        elif event.kind in {"connection_started", "connection_closed"}:
            typer.echo(f"SSH {event.kind.replace('_', ' ')}: {profile.user}@{profile.host}:{profile.port}", err=True)
    code, _, _ = await LocalSSHProvider().execute(request, emit, asyncio.Event())
    return code


@ssh_app.command("add")
def add_profile(
    name: str, host: str = typer.Option(..., "--host"), user: str = typer.Option(..., "--user"),
    port: int = typer.Option(22, "--port"), identity: str | None = typer.Option(None, "--identity"),
    use_agent: bool = typer.Option(False, "--use-agent"), timeout: int = typer.Option(15, "--connect-timeout"),
    ssh_only: bool = typer.Option(False, "--ssh-only", help="Never suggest or bootstrap a managed worker for this target."),
) -> None:
    """Store non-secret connection metadata; no private-key data is read."""
    save_profile(_profile(name, host, user, port, identity, use_agent, timeout, ssh_only))
    typer.echo(f"Saved SSH target {name!r}.")


@ssh_app.command("list")
def profiles_list() -> None:
    rows = [profile.model_dump(mode="json", exclude={"identity_file", "host_key_fingerprint"}) for profile in list_profiles()]
    typer.echo(json.dumps(rows, indent=2, default=str))


@ssh_app.command("show")
def profiles_show(name: str) -> None:
    typer.echo(json.dumps(get_profile(name).model_dump(mode="json"), indent=2, default=str))


@ssh_app.command("edit")
def profiles_edit(
    name: str, host: str | None = typer.Option(None, "--host"), user: str | None = typer.Option(None, "--user"),
    port: int | None = typer.Option(None, "--port"), identity: str | None = typer.Option(None, "--identity"),
    use_agent: bool | None = typer.Option(None, "--use-agent/--no-use-agent"), timeout: int | None = typer.Option(None, "--connect-timeout"),
    ssh_only: bool | None = typer.Option(None, "--ssh-only/--allow-worker-bootstrap"),
) -> None:
    profile = get_profile(name)
    values = profile.model_dump()
    for key, value in {"host": host, "user": user, "port": port, "connect_timeout_seconds": timeout, "ssh_only": ssh_only}.items():
        if value is not None:
            values[key] = value
    if identity is not None:
        values.update(identity_file=identity, use_agent=False)
    elif use_agent is not None:
        values.update(use_agent=use_agent, identity_file=None if use_agent else values["identity_file"])
    save_profile(SSHProfile.model_validate(values))
    typer.echo(f"Updated SSH target {name!r}.")


@ssh_app.command("remove")
def profiles_remove(name: str, yes: bool = typer.Option(False, "--yes")) -> None:
    if not yes and not typer.confirm(f"Remove SSH profile {name!r}?", default=False):
        raise typer.Abort()
    remove_profile(name)
    typer.echo(f"Removed SSH target {name!r}.")


@ssh_app.command("trust-host")
def trust_host(name: str, yes: bool = typer.Option(False, "--yes", help="Confirm the displayed exact fingerprint.")) -> None:
    profile = get_profile(name)
    key_line, fingerprint = inspect_host_key(profile, timeout=profile.connect_timeout_seconds)
    typer.echo(f"Host: {profile.host}:{profile.port}\nFingerprint: {fingerprint}")
    if not yes and not typer.confirm("Trust exactly this host key?", default=False):
        raise typer.Abort()
    trust_scanned_host_key(profile, key_line, fingerprint)
    typer.echo("Host key trusted. Changed keys will continue to fail secure verification.")


@ssh_app.command("test")
def profile_test(name: str) -> None:
    profile = get_profile(name)
    code = asyncio.run(_run(profile, ["true"]))
    if code:
        raise typer.Exit(code=code)
    typer.echo("SSH connection and harmless remote command succeeded.")


@ssh_app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def profile_run(ctx: typer.Context, name: str) -> None:
    code = asyncio.run(_run(get_profile(name), list(ctx.args)))
    if code:
        raise typer.Exit(code=code)


def _scp_argv(profile: SSHProfile, source: str, destination: str) -> list[str]:
    scp = find_ssh_executable("scp")
    argv = [scp, "-P", str(profile.port), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", f"ConnectTimeout={profile.connect_timeout_seconds}"]
    if profile.identity_file:
        argv.extend(["-i", str(Path(profile.identity_file).expanduser())])
    argv.extend([source, destination])
    return argv


@ssh_app.command("upload")
def upload(name: str, local_path: Path, remote_path: str, yes: bool = typer.Option(False, "--yes")) -> None:
    profile = get_profile(name)
    local = local_path.expanduser().resolve()
    if not local.is_file():
        raise typer.BadParameter("local_path must be an existing regular file")
    if not remote_path.startswith("/"):
        raise typer.BadParameter("remote_path must be an absolute remote path")
    if not yes and not typer.confirm(f"Upload {local.name} to {profile.name}:{remote_path}?", default=False):
        raise typer.Abort()
    raise typer.Exit(subprocess.run(_scp_argv(profile, str(local), f"{profile.user}@{profile.host}:{remote_path}"), check=False).returncode)


@ssh_app.command("download")
def download(name: str, remote_path: str, local_path: Path, yes: bool = typer.Option(False, "--yes")) -> None:
    profile = get_profile(name)
    local = local_path.expanduser().resolve()
    if not remote_path.startswith("/"):
        raise typer.BadParameter("remote_path must be an absolute remote path")
    if local.exists() and not yes and not typer.confirm(f"Overwrite {local}?", default=False):
        raise typer.Abort()
    local.parent.mkdir(parents=True, exist_ok=True)
    raise typer.Exit(subprocess.run(_scp_argv(profile, f"{profile.user}@{profile.host}:{remote_path}", str(local)), check=False).returncode)


@ssh_app.command("logs")
def logs(name: str, lines: int = typer.Option(100, "--lines", min=1, max=10000)) -> None:
    code = asyncio.run(_run(get_profile(name), ["journalctl", "-n", str(lines), "--no-pager"]))
    if code:
        raise typer.Exit(code=code)


@ssh_app.command("doctor")
def doctor(name: str) -> None:
    profile = get_profile(name)
    findings = {
        "ssh_executable": bool(shutil.which("ssh")), "host": profile.host, "port": profile.port,
        "identity_path_exists": not profile.identity_file or Path(profile.identity_file).expanduser().is_file(),
        "ssh_agent_mode": profile.use_agent, "known_hosts_exists": profile.known_hosts_path().exists(),
    }
    typer.echo(json.dumps(findings, indent=2))
    if not findings["ssh_executable"] or not findings["identity_path_exists"]:
        raise typer.Exit(code=1)
    code = asyncio.run(_run(profile, ["uname", "-s"]))
    if code:
        raise typer.Exit(code=code)
