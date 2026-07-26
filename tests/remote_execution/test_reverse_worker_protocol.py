from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from mana_agent.remote_execution.credentials import CredentialStore, generate_identity
from mana_agent.remote_execution.daemon import WorkerRuntimeConfig, load_worker_config, write_worker_config
from mana_agent.remote_execution.gateway import WorkerGatewayConfig
from mana_agent.remote_execution.installer import LABEL, MacOSInstaller, WorkerServiceError, launchagent_payload
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

    installer = MacOSInstaller(home=tmp_path, runner=runner)

    with pytest.raises(WorkerServiceError, match="worker install"):
        installer.start()

    assert calls == []


def test_macos_worker_start_bootstraps_an_unloaded_service(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(argv, **kwargs):  # noqa: ANN001
        calls.append(argv)
        returncode = 113 if argv[1] == "print" else 0
        return subprocess.CompletedProcess(argv, returncode, "", "")

    installer = MacOSInstaller(home=tmp_path, runner=runner)
    installer.paths.plist.parent.mkdir(parents=True)
    installer.paths.plist.touch()

    installer.start()

    assert [call[1] for call in calls] == ["print", "bootstrap", "kickstart"]


def test_macos_worker_control_reports_launchctl_error(tmp_path: Path) -> None:
    def runner(argv, **kwargs):  # noqa: ANN001
        if argv[1] == "print":
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise subprocess.CalledProcessError(113, argv, stderr="Could not find specified service")

    installer = MacOSInstaller(home=tmp_path, runner=runner)
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

    with pytest.raises(ValueError, match="allow_insecure_http"):
        default_config.validate_public_url()

    insecure_config.validate_public_url()
