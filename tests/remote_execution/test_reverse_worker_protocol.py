from __future__ import annotations

import base64
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from mana_agent.api.app import create_app
from mana_agent.config.user_config import save_user_config
from mana_agent.fleet.models import WorkerCapabilities as FleetWorkerCapabilities
from mana_agent.fleet.models import WorkerIdentity as FleetWorkerIdentity
from mana_agent.fleet.models import WorkerLabels
from mana_agent.remote_execution.credentials import CredentialStore, generate_identity
from mana_agent.remote_execution.daemon import ReverseWorkerDaemon, WorkerRuntimeConfig, load_worker_config, write_worker_config
from mana_agent.remote_execution.gateway import EnrollmentRequest, WorkerGateway, WorkerGatewayConfig, build_worker_router
from mana_agent.remote_execution.installer import LABEL, MacOSInstaller, WorkerServiceError, launchagent_payload
from mana_agent.remote_execution.installers.linux import LinuxSystemdInstaller, UNIT_NAME
from mana_agent.remote_execution.protocol import WorkerMessage


def test_protocol_rejects_unknown_version_and_oversized_frame() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        WorkerMessage.parse_frame('{"protocol_version":2,"type":"worker.hello"}')
    with pytest.raises(ValueError, match="size"):
        WorkerMessage.parse_frame(b"x" * 1_048_577)


def test_identity_fallback_is_owner_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "keyring", None)
    store = CredentialStore(tmp_path)
    identity = generate_identity("worker-1", "opaque-credential")
    store.save(identity)
    assert store.load("worker-1") == identity
    if os.name == "posix":
        assert store.path.stat().st_mode & 0o077 == 0


def test_launchagent_contains_no_credentials(tmp_path: Path) -> None:
    payload = launchagent_payload(executable="/usr/local/bin/mana-agent", state_dir=tmp_path, log_dir=tmp_path)
    rendered = str(payload)
    assert payload["Label"] == LABEL
    assert "token" not in rendered.lower()
    assert "credential" not in rendered.lower()


def test_macos_worker_start_requires_an_installed_service(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(argv, **kwargs):  # noqa: ANN001
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    installer = MacOSInstaller(home=tmp_path, runner=runner, user_id=501)

    with pytest.raises(WorkerServiceError, match="worker install"):
        installer.start()

    assert calls == []


def test_macos_worker_start_bootstraps_an_unloaded_service(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(argv, **kwargs):  # noqa: ANN001
        calls.append(argv)
        returncode = 113 if argv[1] == "print" else 0
        return subprocess.CompletedProcess(argv, returncode, "", "")

    installer = MacOSInstaller(home=tmp_path, runner=runner, user_id=501)
    installer.paths.plist.parent.mkdir(parents=True)
    installer.paths.plist.touch()

    installer.start()

    assert [call[1] for call in calls] == ["print", "bootstrap", "kickstart"]


def test_macos_worker_control_reports_launchctl_error(tmp_path: Path) -> None:
    def runner(argv, **kwargs):  # noqa: ANN001
        if argv[1] == "print":
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise subprocess.CalledProcessError(113, argv, stderr="Could not find specified service")

    installer = MacOSInstaller(home=tmp_path, runner=runner, user_id=501)
    installer.paths.plist.parent.mkdir(parents=True)
    installer.paths.plist.touch()

    with pytest.raises(WorkerServiceError, match="Could not find specified service"):
        installer.start()


def test_worker_http_coordinator_requires_explicit_opt_in(tmp_path: Path) -> None:
    default_config = WorkerRuntimeConfig(
        coordinator_url="http://coordinator.internal:8000",
        worker_id="worker-1",
        name="worker",
        state_dir=tmp_path,
    )
    insecure_config = WorkerRuntimeConfig(
        coordinator_url="http://coordinator.internal:8000",
        worker_id="worker-1",
        name="worker",
        state_dir=tmp_path,
        allow_insecure_http=True,
    )

    with pytest.raises(ValueError, match="allow_insecure_http"):
        _ = default_config.websocket_url

    assert insecure_config.websocket_url == (
        "ws://coordinator.internal:8000/api/v1/workers/connect"
    )


def test_worker_https_coordinator_uses_secure_websocket(tmp_path: Path) -> None:
    config = WorkerRuntimeConfig(
        coordinator_url="https://coordinator.example.com",
        worker_id="worker-1",
        name="worker",
        state_dir=tmp_path,
    )

    assert config.websocket_url == (
        "wss://coordinator.example.com/api/v1/workers/connect"
    )


def test_worker_http_opt_in_persists_in_runtime_config(tmp_path: Path) -> None:
    config = WorkerRuntimeConfig(
        coordinator_url="http://coordinator.internal:8000",
        worker_id="worker-1",
        name="worker",
        state_dir=tmp_path,
        allow_insecure_http=True,
    )

    write_worker_config(config)

    assert load_worker_config(tmp_path) == config


def test_worker_gateway_http_requires_explicit_opt_in() -> None:
    default_config = WorkerGatewayConfig(
        enabled=True,
        public_url="http://coordinator.internal:8000",
    )
    insecure_config = WorkerGatewayConfig(
        enabled=True,
        public_url="http://coordinator.internal:8000",
        allow_insecure_http=True,
    )

    with pytest.raises(ValueError, match="MANA_WORKER_GATEWAY_ALLOW_INSECURE_HTTP"):
        default_config.validate_public_url()

    insecure_config.validate_public_url()


def test_linux_worker_lifecycle_uses_systemd_user_service(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(argv, **kwargs):  # noqa: ANN001
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    installer = LinuxSystemdInstaller(home=tmp_path, runner=runner)
    installer.unit_path.parent.mkdir(parents=True)
    installer.unit_path.touch()

    installer.start()
    installer.stop()
    installer.restart()

    assert calls == [
        ["systemctl", "--user", "start", UNIT_NAME],
        ["systemctl", "--user", "stop", UNIT_NAME],
        ["systemctl", "--user", "restart", UNIT_NAME],
    ]


def test_linux_worker_lifecycle_requires_installation(tmp_path: Path) -> None:
    installer = LinuxSystemdInstaller(home=tmp_path)

    with pytest.raises(WorkerServiceError, match="worker install"):
        installer.start()


def test_linux_worker_lifecycle_reports_systemctl_error(tmp_path: Path) -> None:
    def runner(argv, **kwargs):  # noqa: ANN001
        raise subprocess.CalledProcessError(
            1,
            argv,
            stderr="Failed to connect to bus",
        )

    installer = LinuxSystemdInstaller(home=tmp_path, runner=runner)
    installer.unit_path.parent.mkdir(parents=True)
    installer.unit_path.touch()

    with pytest.raises(WorkerServiceError, match="Failed to connect to bus"):
        installer.start()


def test_coordinator_generates_worker_id_and_complete_http_install_command(monkeypatch) -> None:
    monkeypatch.delenv("MANA_API_TOKEN", raising=False)
    gateway = WorkerGateway(WorkerGatewayConfig(
        enabled=True,
        public_url="http://coordinator.internal:8000",
        allow_insecure_http=True,
    ))
    app = FastAPI()
    app.include_router(build_worker_router(gateway))

    response = TestClient(app).post(
        "/api/v1/workers/enrollments",
        json={"name": "ubuntu-worker"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["worker_id"].startswith("worker_")
    assert f"--worker-id {payload['worker_id']}" in payload["install_command"]
    assert "--allow-insecure-http" in payload["install_command"]


def test_disabled_worker_gateway_returns_actionable_enrollment_error(monkeypatch) -> None:
    monkeypatch.delenv("MANA_API_TOKEN", raising=False)
    app = FastAPI()
    app.include_router(build_worker_router(WorkerGateway(WorkerGatewayConfig(
        enabled=False,
        public_url="http://coordinator.internal:8000",
        allow_insecure_http=True,
    ))))

    response = TestClient(app).post(
        "/api/v1/workers/enrollments",
        json={"name": "ubuntu-worker"},
    )

    assert response.status_code == 400
    assert "MANA_WORKER_GATEWAY_ENABLED=true" in response.json()["detail"]


def test_api_worker_gateway_reads_user_config_with_environment_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    monkeypatch.setenv("MANA_WORKER_GATEWAY_ENABLED", "false")
    monkeypatch.setenv("MANA_WORKER_GATEWAY_PUBLIC_URL", "https://environment.invalid")
    save_user_config({
        "MANA_WORKER_GATEWAY_ENABLED": True,
        "MANA_WORKER_GATEWAY_PUBLIC_URL": "http://coordinator.internal:8000",
        "MANA_WORKER_GATEWAY_ALLOW_INSECURE_HTTP": True,
    }, merge=False)

    app = create_app(telegram_config=type(
        "TelegramConfig",
        (),
        {"enabled": False, "effective_transport": "polling"},
    )())

    config = app.state.worker_gateway.config
    assert config.enabled is True
    assert config.public_url == "http://coordinator.internal:8000"
    assert config.allow_insecure_http is True


def test_api_worker_gateway_preserves_environment_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "empty-mana"))
    monkeypatch.setenv("MANA_WORKER_GATEWAY_ENABLED", "true")
    monkeypatch.setenv(
        "MANA_WORKER_GATEWAY_PUBLIC_URL",
        "http://environment-coordinator.internal:8000",
    )
    monkeypatch.setenv("MANA_WORKER_GATEWAY_ALLOW_INSECURE_HTTP", "true")

    app = create_app(telegram_config=type(
        "TelegramConfig",
        (),
        {"enabled": False, "effective_transport": "polling"},
    )())

    config = app.state.worker_gateway.config
    assert config.enabled is True
    assert config.public_url == "http://environment-coordinator.internal:8000"
    assert config.allow_insecure_http is True


def test_standalone_api_accepts_authenticated_worker_capabilities(tmp_path: Path, monkeypatch) -> None:
    """The ASGI app must provide Fleet storage even without a ChatGateway."""
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    app = create_app(telegram_config=type(
        "TelegramConfig",
        (),
        {"enabled": False, "effective_transport": "polling"},
    )())
    gateway = app.state.worker_gateway
    assert app.state.fleet_registry is gateway.fleet_registry

    worker_id = "worker-standalone"
    identity = generate_identity(worker_id)
    registration = ReverseWorkerDaemon.registration(
        worker_id, "standalone", identity.public_key_pem,
    )
    token = gateway.registry.issue_enrolment_token(worker_id)
    enrollment = gateway.enroll(EnrollmentRequest(token=token, registration=registration))
    inventory = FleetWorkerCapabilities(
        worker_id=worker_id,
        platform="linux",
        architecture="x86_64",
        labels=WorkerLabels(),
        last_probe_at=datetime.now(timezone.utc),
    )
    capability_message = WorkerMessage(
        type="worker.capabilities",
        worker_id=worker_id,
        payload={"inventory": inventory.model_dump(mode="json")},
    )
    private_key = serialization.load_pem_private_key(
        identity.private_key_pem.encode(), password=None,
    )
    capability_message.signature = base64.b64encode(
        private_key.sign(capability_message.signing_bytes())
    ).decode()

    with pytest.raises(WebSocketDisconnect):
        with TestClient(app).websocket_connect("/api/v1/workers/connect") as websocket:
            websocket.send_text(WorkerMessage(
                type="worker.hello",
                worker_id=worker_id,
                payload={"credential": enrollment.credential},
            ).model_dump_json())
            assert WorkerMessage.parse_frame(websocket.receive_text()).type == "worker.authenticated"
            websocket.send_text(capability_message.model_dump_json())

    worker = gateway.fleet_registry.require(worker_id)
    assert worker.capabilities == inventory
    assert worker.identity == FleetWorkerIdentity(
        worker_id=worker_id,
        identity_fingerprint=worker.identity.identity_fingerprint,
        authenticated=True,
        credential_status="valid",
    )
