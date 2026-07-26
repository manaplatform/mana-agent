"""Regression coverage for persistent automation schedule deployment."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mana_agent.automations import service
from mana_agent.automations.service import (
    AutomationValidationError,
    ScheduleDefinition,
    deploy_local,
    deployment_status,
    render_workflow,
)
from mana_agent.commands.cli import app


def _schedule(*, target: str = "local") -> ScheduleDefinition:
    return ScheduleDefinition.create(
        name="Nightly report",
        action="analyze",
        cron="0 2 * * *",
        targets=[target],
    )


def test_schedule_validation_rejects_invalid_cron_and_custom_command() -> None:
    with pytest.raises(AutomationValidationError, match="five-field"):
        ScheduleDefinition.create(name="Bad", action="analyze", cron="daily", targets=["local"])
    with pytest.raises(AutomationValidationError, match="require a command"):
        ScheduleDefinition.create(name="Bad", action="custom", cron="0 2 * * *", targets=["local"])


def test_local_deployment_reconciles_one_tagged_crontab_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service.shutil, "which", lambda name: "/usr/bin/crontab" if name == "crontab" else None)
    writes: list[str] = []

    def runner(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args == ["crontab", "-l"]:
            return subprocess.CompletedProcess(args, 0, "5 1 * * * echo prior # mana-agent:sch_deadbeef0000\n", "")
        assert args == ["crontab", "-"]
        writes.append(str(kwargs["input"]))
        return subprocess.CompletedProcess(args, 0, "", "")

    schedule = _schedule()
    deployed = deploy_local(schedule, tmp_path, runner=runner)

    assert deployed["status"] == "deployed"
    assert len(writes) == 1
    assert service.managed_cron_marker(schedule) in writes[0]
    assert "mana-agent automation execute --action analyze" in writes[0]


def test_generated_workflow_has_schedule_dispatch_and_artifact_upload() -> None:
    workflow = render_workflow(_schedule(target="github"))

    assert 'cron: "0 2 * * *"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "MANA_HOME: ${{ runner.temp }}/mana" in workflow
    assert "path: ${{ runner.temp }}/mana/" in workflow


def test_custom_command_is_rendered_only_after_explicit_validation(tmp_path: Path) -> None:
    schedule = ScheduleDefinition.create(
        name="Custom backup",
        action="custom",
        cron="15 4 * * 1",
        targets=["local"],
        command="./scripts/backup.sh",
    )

    assert service.schedule_command(schedule, tmp_path) == "./scripts/backup.sh"


def test_fleet_schedule_requires_explicit_matrix_and_uses_fleet_command(tmp_path: Path) -> None:
    with pytest.raises(AutomationValidationError, match="platforms"):
        ScheduleDefinition.create(
            name="Fleet", action="fleet-verify", cron="0 2 * * *",
            targets=["local"], action_config={},
        )
    schedule = ScheduleDefinition.create(
        name="Fleet", action="fleet-verify", cron="0 2 * * *",
        targets=["local"],
        action_config={
            "platforms": ["linux", "windows"],
            "commands": ["python -m pytest -q"],
            "maximum_workers": 2,
            "timeout": 900,
        },
    )
    command = service.schedule_command(schedule, tmp_path)
    assert "mana-agent fleet verify" in command
    assert "--platform linux" in command
    assert "--platform windows" in command
    assert "--command 'python -m pytest -q'" in command


def test_status_reports_local_drift_without_default_action(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service.shutil, "which", lambda name: "/usr/bin/crontab" if name == "crontab" else None)

    def runner(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, "", "")

    status = deployment_status(_schedule(), tmp_path, runner=runner)
    assert status["targets"]["local"]["status"] == "drifted"


def test_automation_cli_lists_empty_schedule_store(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["automation", "list", "--root-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert result.stdout.strip() == "[]"
