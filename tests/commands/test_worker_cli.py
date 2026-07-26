import json

import pytest
import typer
from typer.testing import CliRunner

from mana_agent.commands import worker_cli
from mana_agent.commands.worker_cli import _enroll, worker_app
from mana_agent.remote_execution.installer import WorkerServiceError


def test_worker_start_reports_service_error_without_traceback(monkeypatch) -> None:
    class MissingInstaller:
        def __init__(self, **kwargs) -> None:
            pass

        def start(self) -> None:
            raise WorkerServiceError("Worker service is not installed. Run `mana-agent worker install` first.")

    monkeypatch.setattr(worker_cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(worker_cli, "MacOSInstaller", MissingInstaller)

    result = CliRunner().invoke(worker_app, ["start"])

    assert result.exit_code == 1
    assert "Worker service is not installed" in result.output
    assert "Traceback" not in result.output


def test_worker_enrollment_rejects_http_without_explicit_opt_in(tmp_path) -> None:
    with pytest.raises(typer.BadParameter, match="--allow-insecure-http"):
        _enroll(
            "http://coordinator.internal:8000",
            "enrollment-token",
            "worker-1",
            "worker",
            tmp_path,
        )


def test_worker_enrollment_allows_explicit_http(tmp_path, monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

        def read(self) -> bytes:
            return json.dumps({
                "worker_id": "worker-1",
                "credential": "worker-credential",
            }).encode()

    requests = []
    monkeypatch.setattr(
        worker_cli.urllib.request,
        "urlopen",
        lambda request, timeout: requests.append(request) or Response(),
    )
    monkeypatch.setattr(worker_cli.CredentialStore, "save", lambda self, identity: None)

    config = _enroll(
        "http://coordinator.internal:8000",
        "enrollment-token",
        "worker-1",
        "worker",
        tmp_path,
        allow_insecure_http=True,
    )

    assert requests[0].full_url == "http://coordinator.internal:8000/api/v1/workers/enroll"
    assert config.allow_insecure_http is True
    assert config.websocket_url == "ws://coordinator.internal:8000/api/v1/workers/connect"


def test_linux_worker_start_uses_platform_installer(monkeypatch) -> None:
    calls = []

    class LinuxInstaller:
        def start(self) -> None:
            calls.append("start")

    monkeypatch.setattr(worker_cli, "_installer", lambda: LinuxInstaller())
    monkeypatch.setattr(worker_cli.platform, "system", lambda: "Linux")

    result = CliRunner().invoke(worker_app, ["start"])

    assert result.exit_code == 0
    assert calls == ["start"]
    assert "Worker start requested" in result.output


def test_enrollment_create_lets_coordinator_generate_worker_id(monkeypatch) -> None:
    calls = []

    def coordinator_call(coordinator, path, **kwargs):  # noqa: ANN001
        calls.append((coordinator, path, kwargs))
        return {
            "worker_id": "worker_generated",
            "token": "sensitive-token",
            "install_command": "mana-agent worker install --worker-id worker_generated",
        }

    monkeypatch.setattr(worker_cli, "_coordinator_call", coordinator_call)

    result = CliRunner().invoke(
        worker_app,
        ["enrollment", "create", "--coordinator", "http://coordinator.internal:8000"],
    )

    assert result.exit_code == 0
    assert '"worker_id": "worker_generated"' in result.output
    assert "worker_id" not in calls[0][2]["payload"]
