"""Regression coverage for canonical durable automations."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from mana_agent.automations import service
from mana_agent.automations.service import (
    AgentPromptJob,
    AutomationDefinition,
    AutomationService,
    CronTrigger,
    DeploymentState,
    IntervalTrigger,
    OnceTrigger,
    calculate_next_run,
    create_automation,
    deploy_local_cron,
    execute_automation,
    render_workflow,
)
from mana_agent.commands.cli import app


@pytest.fixture(autouse=True)
def isolated_mana_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana-home"))


def _automation(root: Path, *, trigger=None, target_runtime: str = "local") -> AutomationDefinition:
    trigger = trigger or CronTrigger(expression="0 2 * * *", timezone="UTC")
    return create_automation(
        root,
        name="Nightly report",
        trigger=trigger,
        job=AgentPromptJob(prompt="Analyze this repository."),
        timezone_name="UTC",
        target_runtime=target_runtime,  # type: ignore[arg-type]
        deploy=False,
    )


def test_interval_is_exact_across_midnight_restart_and_dst() -> None:
    anchor = datetime(2026, 3, 7, 23, 30, tzinfo=timezone.utc)
    trigger = IntervalTrigger(every_seconds=18_000, anchor_at=anchor)
    first = calculate_next_run(trigger, after=anchor)
    assert first == datetime(2026, 3, 8, 4, 30, tzinfo=timezone.utc)
    assert calculate_next_run(trigger, after=first) == datetime(2026, 3, 8, 9, 30, tzinfo=timezone.utc)
    # A restart/delay skips to the next exact anchor multiple; no wall-clock drift.
    assert calculate_next_run(
        trigger, after=datetime(2026, 3, 9, 1, 0, tzinfo=timezone.utc)
    ) == datetime(2026, 3, 9, 5, 30, tzinfo=timezone.utc)


def test_calendar_cron_preserves_timezone() -> None:
    trigger = CronTrigger(expression="0 9 * * 1-5", timezone="Asia/Tehran")
    after = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)
    result = calculate_next_run(trigger, after=after)
    assert result is not None
    assert result.astimezone(service.ZoneInfo("Asia/Tehran")).hour == 9
    assert result.astimezone(service.ZoneInfo("Asia/Tehran")).weekday() == 0


def test_windows_store_lock_acquires_and_releases_the_same_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, int]] = []

    def locking(fd: int, mode: int, nbytes: int) -> None:
        calls.append((mode, nbytes, service.os.lseek(fd, 0, service.os.SEEK_CUR)))

    fake_msvcrt = SimpleNamespace(LK_LOCK=1, LK_UNLCK=2, locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    with (tmp_path / "automation.lock").open("a+b") as handle:
        service._acquire_windows_file_lock(handle)
        service._release_windows_file_lock(handle)

    assert [(mode, nbytes, position) for mode, nbytes, position in calls] == [
        (fake_msvcrt.LK_LOCK, 1, 0),
        (fake_msvcrt.LK_UNLCK, 1, 0),
    ]


def test_create_rejects_a_past_one_time_trigger(tmp_path: Path) -> None:
    with pytest.raises(
        service.AutomationValidationError,
        match="run_at must be strictly in the future",
    ):
        create_automation(
            tmp_path,
            name="Expired one-time task",
            trigger=OnceTrigger(
                run_at=datetime.now(timezone.utc) - timedelta(minutes=1),
                timezone="UTC",
            ),
            job=AgentPromptJob(prompt="Do not run an expired task."),
            timezone_name="UTC",
            deploy=False,
        )

    assert AutomationService(tmp_path).list() == []


def test_local_deployment_uses_id_executor_and_one_managed_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    automation = _automation(tmp_path)
    monkeypatch.setattr(service.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(service, "ensure_automation_runtime", lambda: Path("/runtime/mana-agent"))
    writes: list[str] = []

    def runner(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args == ["crontab", "-l"]:
            return subprocess.CompletedProcess(args, 0, "5 1 * * * old # mana-agent:sch_old00000000\n", "")
        writes.append(str(kwargs["input"]))
        return subprocess.CompletedProcess(args, 0, "", "")

    state = deploy_local_cron(automation, tmp_path, runner=runner)
    assert state.status == "deployed"
    assert "--automation-id " + automation.id in writes[0]
    assert "--action" not in writes[0]


def test_interval_backend_wakes_minutely_without_converting_to_cron(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    automation = _automation(
        tmp_path,
        trigger=IntervalTrigger(every_seconds=18_000, anchor_at=datetime.now(timezone.utc)),
    )
    monkeypatch.setattr(service.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(service, "ensure_automation_runtime", lambda: Path("/runtime/mana-agent"))
    writes: list[str] = []

    def runner(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args == ["crontab", "-l"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        writes.append(str(kwargs["input"]))
        return subprocess.CompletedProcess(args, 0, "", "")

    deploy_local_cron(automation, tmp_path, runner=runner)
    assert writes[0].startswith("* * * * *")
    assert "0 */5 * * *" not in writes[0]


def test_executor_reloads_persisted_definition_and_duplicate_claim_runs_once(tmp_path: Path) -> None:
    automation = _automation(
        tmp_path,
        trigger=OnceTrigger(run_at=datetime.now(timezone.utc) + timedelta(hours=1), timezone="UTC"),
    )
    calls: list[str] = []

    def execute(record: AutomationDefinition, root: Path) -> dict[str, object]:
        calls.append(record.id)
        return {"ok": True, "root": str(root)}

    first = execute_automation(tmp_path, automation.id, force=True, job_executor=execute)
    second = execute_automation(tmp_path, automation.id, force=False, job_executor=execute)
    assert first["executed"] is True
    assert second["executed"] is False
    assert calls == [automation.id]
    reloaded = AutomationService(tmp_path).get(automation.id)
    assert reloaded.recent_execution and reloaded.recent_execution.status == "succeeded"
    assert reloaded.next_run_at is None
    assert AutomationService(tmp_path).status(automation.id)["recent_runs"]


def test_restart_reloads_definition_and_next_run(tmp_path: Path) -> None:
    automation = _automation(tmp_path)
    restarted = AutomationService(tmp_path).get(automation.id)
    assert restarted.id == automation.id
    assert restarted.next_run_at == automation.next_run_at


def test_retry_enabled_automation_requires_explicit_side_effect_safety(tmp_path: Path) -> None:
    with pytest.raises(
        service.AutomationValidationError,
        match="retry-safe side-effect classification",
    ):
        create_automation(
            tmp_path,
            name="Unsafe retry",
            trigger=CronTrigger(expression="0 2 * * *", timezone="UTC"),
            job=AgentPromptJob(prompt="Perform an external action."),
            timezone_name="UTC",
            retry_policy={"maximum_attempts": 2},
            deploy=False,
        )

    safe = create_automation(
        tmp_path,
        name="Read-only retry",
        trigger=CronTrigger(expression="0 3 * * *", timezone="UTC"),
        job=AgentPromptJob(prompt="Read and summarize."),
        timezone_name="UTC",
        retry_policy={"maximum_attempts": 2},
        side_effect_classification="read_only",
        deploy=False,
    )
    assert safe.side_effect_classification == "read_only"


def test_legacy_migration_preserves_ids_runs_invalid_records_and_is_idempotent(tmp_path: Path) -> None:
    path = service.config_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "version": 2,
        "schedules": [{
            "id": "sch_deadbeef0000", "name": "Nightly", "action": "analyze",
            "cron": "0 2 * * *", "targets": ["local"], "enabled": True,
        }],
        "automations": [
            {"id": "aut_legacy-good", "name": "Report", "action": "daily_report", "cron": "0 8 * * *"},
            {"id": "aut_broken-one", "name": "Broken"},
        ],
        "runs": [{"id": "old-run", "automation_id": "sch_deadbeef0000", "status": "succeeded"}],
    }), encoding="utf-8")
    first = service.load_config(tmp_path)
    second = service.load_config(tmp_path)
    assert {row["id"] for row in first["automations"]} == {"sch_deadbeef0000", "aut_legacy-good"}
    assert first["runs"][0]["id"] == "old-run"
    assert first["migration_errors"][0]["id"] == "aut_broken-one"
    assert first == second
    assert "schedules" not in first


def test_missing_persistent_backend_is_truthfully_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    automation = _automation(tmp_path)
    monkeypatch.setattr(service.platform, "system", lambda: "Linux")
    monkeypatch.setattr(service.shutil, "which", lambda name: None)
    reconciled = service.reconcile_deployment(tmp_path, automation.id)
    assert reconciled.deployment.status == "blocked"
    assert "crontab" in reconciled.deployment.blocked_reason


def test_delete_removes_store_and_deployment_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    automation = _automation(tmp_path, target_runtime="github")
    automation.deployment = DeploymentState(status="deployed", backend="github-actions")
    service.upsert_automation(tmp_path, automation)
    path = service.workflow_path(tmp_path, automation)
    path.parent.mkdir(parents=True)
    path.write_text("managed", encoding="utf-8")
    AutomationService(tmp_path).delete(automation.id)
    assert not path.exists()
    assert AutomationService(tmp_path).list() == []


def test_github_workflow_is_id_based_and_repository_safe(tmp_path: Path) -> None:
    automation = _automation(tmp_path, target_runtime="github")
    workflow = render_workflow(automation, tmp_path)
    assert "--automation-id " + automation.id in workflow
    assert "workflow_dispatch:" in workflow


def test_platform_persistent_scheduler_adapters_are_id_based(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    automation = _automation(tmp_path)
    monkeypatch.setattr(service, "ensure_automation_runtime", lambda: Path("/runtime/mana-agent"))
    calls: list[list[str]] = []

    def runner(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    launch_path = tmp_path / "LaunchAgents" / f"{automation.id}.plist"
    monkeypatch.setattr(service, "_launchd_path", lambda _item: launch_path)
    monkeypatch.setattr(service.os, "getuid", lambda: 501, raising=False)
    launch = service.deploy_launchd(automation, tmp_path, runner=runner)
    assert launch.status == "deployed"
    launch_content = launch_path.read_text(encoding="utf-8")
    assert automation.id in launch_content
    assert "StandardOutPath" in launch_content
    assert "StandardErrorPath" in launch_content

    unit_dir = tmp_path / "systemd"
    monkeypatch.setattr(
        service,
        "_systemd_paths",
        lambda _item: (
            unit_dir / f"{automation.id}.service",
            unit_dir / f"{automation.id}.timer",
        ),
    )
    systemd = service.deploy_systemd(automation, tmp_path, runner=runner)
    assert systemd.status == "deployed"
    assert "--automation-id " + automation.id in (unit_dir / f"{automation.id}.service").read_text()

    windows = service.deploy_windows(automation, tmp_path, runner=runner)
    assert windows.status == "deployed"
    assert any("ManaAgent-" + automation.id in command for command in calls)


def test_automation_runtime_copies_the_active_venv_beneath_mana_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source-venv"
    source_site_packages = source / "lib" / "python3.12" / "site-packages"
    source_site_packages.mkdir(parents=True)
    (source / "pyvenv.cfg").write_text("home = /python\n", encoding="utf-8")
    (source / "bin").mkdir()
    (source / "bin" / "mana-agent").write_text("#!/source/python\n", encoding="utf-8")
    package_root = tmp_path / "source-package" / "mana_agent"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("VERSION = 'test'\n", encoding="utf-8")
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana-home"))
    monkeypatch.setattr(service, "_source_virtualenv", lambda: source)
    monkeypatch.setattr(service, "_runtime_fingerprint", lambda _source: "snapshot-1")
    monkeypatch.setattr(service, "__file__", str(package_root / "automations" / "service.py"))

    executable = service.ensure_automation_runtime()

    assert executable == tmp_path / "mana-home" / "automations" / "runtime" / "bin" / "mana-agent"
    assert executable.is_file()
    assert "#!" + str(executable.parent / "python") in executable.read_text(encoding="utf-8")
    assert (executable.parents[1] / "lib" / "python3.12" / "site-packages" / "mana_agent" / "__init__.py").is_file()


def test_disabling_launchd_boots_out_before_removing_the_plist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    automation = _automation(tmp_path)
    automation.enabled = False
    launch_path = tmp_path / "LaunchAgents" / f"{automation.id}.plist"
    launch_path.parent.mkdir(parents=True)
    launch_path.write_text("plist", encoding="utf-8")
    monkeypatch.setattr(service, "_launchd_path", lambda _item: launch_path)
    monkeypatch.setattr(service.os, "getuid", lambda: 501, raising=False)

    def runner(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert args[:2] == ["launchctl", "bootout"]
        assert launch_path.exists()
        return subprocess.CompletedProcess(args, 0, "", "")

    state = service.deploy_launchd(automation, tmp_path, runner=runner)
    assert state.status == "disabled"
    assert not launch_path.exists()


def test_launchd_health_reports_a_recorded_executor_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    automation = _automation(tmp_path)
    automation.deployment = DeploymentState(status="deployed", backend="launchd")
    service.upsert_automation(tmp_path, automation)
    launch_path = tmp_path / "LaunchAgents" / f"{automation.id}.plist"
    launch_path.parent.mkdir(parents=True)
    launch_path.write_text("plist", encoding="utf-8")
    monkeypatch.setattr(service, "_launchd_path", lambda _item: launch_path)
    monkeypatch.setattr(service.os, "getuid", lambda: 501, raising=False)

    def runner(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert args[:2] == ["launchctl", "print"]
        return subprocess.CompletedProcess(args, 0, "last exit code = 1", "")

    assert service.deployment_status(automation, tmp_path, runner=runner)["deployment_healthy"] is False


def test_connector_execution_fails_if_gateway_mutates_automations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    automation = create_automation(
        tmp_path,
        name="Gmail task",
        trigger=OnceTrigger(run_at=datetime.now(timezone.utc) + timedelta(hours=1), timezone="UTC"),
        job={
            "type": "connector_action", "connector": "gmail", "action": "check_inbox",
            "arguments": {"account": "account@example.com"}, "prompt": "Check the inbox.",
        },
        timezone_name="UTC", deploy=False,
    )

    class Gateway:
        def __init__(self, _root: Path) -> None:
            pass

        def send(self, _session_id: str, prompt: str) -> str:
            assert "Immediately execute" in prompt
            assert "do not create" in prompt
            duplicate = automation.model_copy(update={"id": "aut_unexpectedmutation"})
            service.upsert_automation(tmp_path, duplicate)
            return "Incorrectly created an automation."

        def close_session(self, _session_id: str) -> None:
            pass

    monkeypatch.setattr("mana_agent.gateway.AgentChatGateway", Gateway)
    result = service._execute_job(automation, tmp_path)
    assert result["ok"] is False
    assert "modify durable automation definitions" in str(result["error"])


def test_secret_values_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="credential reference"):
        create_automation(
            tmp_path,
            name="Unsafe",
            trigger=CronTrigger(expression="0 2 * * *", timezone="UTC"),
            job={"type": "connector_action", "connector": "gmail", "action": "search", "arguments": {"access_token": "plain"}},
            timezone_name="UTC",
            deploy=False,
        )


def test_cli_is_management_only(tmp_path: Path) -> None:
    runner = CliRunner()
    root_help = runner.invoke(app, ["--help"])
    help_result = runner.invoke(app, ["automation", "--help"])
    assert root_help.exit_code == 0 and "cron" not in root_help.stdout.lower()
    assert help_result.exit_code == 0
    assert "create" not in help_result.stdout.lower()
    assert "run-now" not in help_result.stdout.lower()
    assert "list" in help_result.stdout and "status" in help_result.stdout and "delete" in help_result.stdout
    listed = runner.invoke(app, ["automation", "list", "--root-dir", str(tmp_path)])
    assert listed.exit_code == 0 and listed.stdout.strip() == "[]"
