"""Worker lifecycle commands.  Persistent changes are explicit and never chat-only."""

from __future__ import annotations

import asyncio
import json
import platform
import socket
import urllib.error
import urllib.request
import os
from pathlib import Path
from urllib.parse import urlparse

import typer

from mana_agent.remote_execution.credentials import CredentialStore, generate_identity
from mana_agent.remote_execution.daemon import ReverseWorkerDaemon, WorkerRuntimeConfig, load_worker_config, write_worker_config
from mana_agent.remote_execution.installer import MacOSInstaller, WorkerServiceError
from mana_agent.remote_execution.installers import LinuxSystemdInstaller, WindowsTaskSchedulerInstaller

worker_app = typer.Typer(help="Install and operate this machine as an outbound reverse worker.", no_args_is_help=True)
enrollment_app = typer.Typer(help="Create or use short-lived worker enrollment payloads.", no_args_is_help=True)
workers_app = typer.Typer(help="Manage coordinator workers through its authenticated API.", no_args_is_help=True)
worker_app.add_typer(enrollment_app, name="enrollment")


def _state_dir(state_dir: str | None) -> Path:
    if state_dir:
        return Path(state_dir).expanduser().resolve()
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library/Application Support/ManaAgent"
    if system == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "ManaAgent/state"
    return Path.home() / ".local/share/mana-agent"


def _installer():
    system = platform.system()
    if system == "Darwin":
        return MacOSInstaller(home=Path.home())
    if system == "Linux":
        return LinuxSystemdInstaller(home=Path.home())
    if system == "Windows":
        return WindowsTaskSchedulerInstaller()
    raise typer.BadParameter(f"worker service installation is unsupported on {system}")


def _coordinator_call(coordinator: str, path: str, *, method: str = "GET", payload: dict | None = None,
                      api_token: str = "") -> object:
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if body else {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    request = urllib.request.Request(coordinator.rstrip("/") + path, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            response_body = response.read(1_048_577)
    except urllib.error.HTTPError as exc:
        error_body = exc.read(16_385)
        detail = _coordinator_error_detail(error_body[:16_384], fallback=exc.reason)
        typer.echo(f"Coordinator request failed (HTTP {exc.code}): {detail}", err=True)
        raise typer.Exit(1) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        detail = str(getattr(exc, "reason", exc))[-2_000:]
        typer.echo(f"Coordinator connection failed: {detail}", err=True)
        raise typer.Exit(1) from exc
    if len(response_body) > 1_048_576:
        typer.echo("Coordinator response exceeded the 1 MiB safety limit.", err=True)
        raise typer.Exit(1)
    try:
        return json.loads(response_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        typer.echo("Coordinator returned an invalid JSON response.", err=True)
        raise typer.Exit(1) from exc


def _coordinator_error_detail(body: bytes, *, fallback: str) -> str:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        text = body.decode("utf-8", errors="replace").strip()
        return text[-2_000:] or str(fallback)
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("error")
        if detail:
            return str(detail)[-2_000:]
    return str(fallback)


def _enroll(
    coordinator: str,
    token: str,
    worker_id: str,
    name: str,
    state_dir: Path,
    *,
    allow_insecure_http: bool = False,
) -> WorkerRuntimeConfig:
    identity = generate_identity(worker_id)
    registration = ReverseWorkerDaemon.registration(worker_id, name or socket.gethostname(), identity.public_key_pem)
    payload = json.dumps({"token": token, "registration": registration.model_dump(mode="json")}).encode()
    parsed = coordinator.rstrip("/")
    scheme = urlparse(parsed).scheme.lower()
    if scheme != "https" and not (allow_insecure_http and scheme == "http"):
        raise typer.BadParameter(
            "coordinator must use HTTPS; pass --allow-insecure-http to explicitly use HTTP"
        )
    request = urllib.request.Request(parsed + "/api/v1/workers/enroll", data=payload,
                                    headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.loads(response.read())
    except Exception as exc:
        raise typer.BadParameter("worker enrollment failed; check the coordinator URL, TLS, and token") from exc
    CredentialStore(state_dir).save(type(identity)(worker_id=str(result["worker_id"]), credential=str(result["credential"]), private_key_pem=identity.private_key_pem))
    return WorkerRuntimeConfig(coordinator_url=coordinator, worker_id=str(result["worker_id"]), name=name or registration.display_name,
                               state_dir=state_dir, allow_insecure_http=allow_insecure_http)


@worker_app.command("install")
def install_worker(coordinator: str = typer.Option("", "--coordinator"), token: str = typer.Option("", "--token", hide_input=True),
                   worker_id: str = typer.Option(..., "--worker-id", help="Worker ID returned by `worker enrollment create`."),
                   name: str = typer.Option("", "--name"), state_dir: str | None = typer.Option(None, "--state-dir"),
                   allow_insecure_http: bool = typer.Option(
                       False,
                       "--allow-insecure-http",
                       "--insecure-local-development",
                       help="Allow unencrypted HTTP/ws transport; credentials may be exposed in transit.",
                   )) -> None:
    """Enroll and install an owner-scoped worker service; never stores the bootstrap token."""
    if not coordinator:
        coordinator = typer.prompt("Coordinator URL")
    if not token:
        token = typer.prompt("Short-lived enrollment token", hide_input=True)
    root = _state_dir(state_dir)
    config = _enroll(
        coordinator,
        token,
        worker_id,
        name,
        root,
        allow_insecure_http=allow_insecure_http,
    )
    write_worker_config(config)
    _installer().install()
    typer.echo(f"Worker {config.worker_id} enrolled and service installed. Run `mana-agent worker doctor` to verify connectivity.")


@worker_app.command("run", hidden=True)
def run_worker(state_dir: str | None = typer.Option(None, "--state-dir")) -> None:
    """Service entrypoint; intended to run independently from the TUI."""
    asyncio.run(ReverseWorkerDaemon(load_worker_config(_state_dir(state_dir))).run())


@worker_app.command("uninstall")
def uninstall_worker(yes: bool = typer.Option(False, "--yes"), state_dir: str | None = typer.Option(None, "--state-dir")) -> None:
    if not yes and not typer.confirm("Remove this worker service and its local identity?", default=False):
        raise typer.Abort()
    _installer().uninstall()
    root = _state_dir(state_dir)
    worker_id = ""
    if (root / "worker.json").exists():
        worker_id = str(json.loads((root / "worker.json").read_text(encoding="utf-8")).get("worker_id", ""))
    CredentialStore(root).delete(worker_id or None)
    (root / "worker.json").unlink(missing_ok=True)
    typer.echo("Worker service and local identity removed.")


def _service_control(action: str) -> None:
    installer = _installer()
    control = getattr(installer, action, None)
    if control is None:
        raise typer.BadParameter(
            f"worker service {action} is not implemented for {platform.system()}"
        )
    try:
        control()
    except WorkerServiceError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Worker {action} requested.")


for _name in ("start", "stop", "restart"):
    worker_app.command(_name)(lambda action=_name: _service_control(action))


@worker_app.command("status")
def worker_status(state_dir: str | None = typer.Option(None, "--state-dir")) -> None:
    root = _state_dir(state_dir)
    config_exists, identity = (root / "worker.json").exists(), CredentialStore(root).load()
    service = _installer().status()
    typer.echo(json.dumps({"configured": config_exists, "identity_present": bool(identity), "service_running": service}, indent=2))


@worker_app.command("logs")
def worker_logs(lines: int = typer.Option(100, "--lines", min=1, max=10000)) -> None:
    installer = _installer()
    if hasattr(installer, "logs"):
        typer.echo("\n".join(installer.logs().splitlines()[-lines:]))
        return
    path = Path.home() / "Library/Logs/ManaAgent/worker.err.log"
    typer.echo("\n".join(path.read_text(errors="replace").splitlines()[-lines:]) if path.exists() else "No worker log file exists yet.")


@worker_app.command("doctor")
def worker_doctor(repair: bool = typer.Option(False, "--repair"), state_dir: str | None = typer.Option(None, "--state-dir")) -> None:
    root = _state_dir(state_dir)
    findings = {"config": (root / "worker.json").exists(), "identity": CredentialStore(root).load() is not None,
                "state_writable": root.exists() and __import__('os').access(root, __import__('os').W_OK),
                "service": _installer().status()}
    typer.echo(json.dumps(findings, indent=2))
    if repair and findings["config"] and not findings["service"]:
        _installer().install()
        typer.echo("Worker service repaired.")
    if not all(findings.values()):
        raise typer.Exit(1)


@worker_app.command("reconnect")
def reconnect_worker() -> None:
    installer = _installer()
    if hasattr(installer, "reconnect"):
        installer.reconnect()
    else:
        _service_control("restart")


@worker_app.command("rotate-identity")
def rotate_identity() -> None:
    raise typer.BadParameter("identity rotation must be requested by the authenticated coordinator; re-enroll this worker with a new short-lived token.")


@enrollment_app.command("create")
def create_enrollment(coordinator: str = typer.Option(..., "--coordinator"),
                      worker_id: str = typer.Option("", "--worker-id", help="Optional stable ID; generated by the coordinator when omitted."),
                      name: str = typer.Option("", "--name"), expires_in: int = typer.Option(900, "--expires-in"),
                      api_token: str = typer.Option("", "--api-token", hide_input=True)) -> None:
    """Create a sensitive, short-lived coordinator enrollment payload."""
    payload = {"name": name, "expires_in_seconds": expires_in}
    if worker_id:
        payload["worker_id"] = worker_id
    result = _coordinator_call(coordinator, "/api/v1/workers/enrollments", method="POST",
                               payload=payload, api_token=api_token)
    typer.echo(json.dumps(result, indent=2))


@workers_app.command("list")
def list_workers(coordinator: str = typer.Option(..., "--coordinator")) -> None:
    typer.echo(json.dumps(_coordinator_call(coordinator, "/api/v1/workers"), indent=2, default=str))


@workers_app.command("show")
def show_worker(worker_id: str, coordinator: str = typer.Option(..., "--coordinator")) -> None:
    typer.echo(json.dumps(_coordinator_call(coordinator, f"/api/v1/workers/{worker_id}"), indent=2, default=str))


@workers_app.command("revoke")
def revoke_worker(worker_id: str, coordinator: str = typer.Option(..., "--coordinator"),
                  api_token: str = typer.Option("", "--api-token", hide_input=True)) -> None:
    typer.echo(json.dumps(_coordinator_call(coordinator, f"/api/v1/workers/{worker_id}/revoke", method="POST", payload={}, api_token=api_token)))


@workers_app.command("rotate-identity")
def coordinator_rotate_identity(worker_id: str, coordinator: str = typer.Option(..., "--coordinator"),
                                api_token: str = typer.Option("", "--api-token", hide_input=True)) -> None:
    typer.echo(json.dumps(_coordinator_call(coordinator, f"/api/v1/workers/{worker_id}/rotate-identity", method="POST", payload={}, api_token=api_token)))
