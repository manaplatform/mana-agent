from __future__ import annotations

import json

from typer.testing import CliRunner

from mana_agent.commands.cli import app
from mana_agent.doctor.models import DoctorFinding, Severity
from mana_agent.doctor.runner import DoctorReport
from mana_agent.doctor.registry import select_checks
from mana_agent.doctor.redaction import redact


def _healthy_report() -> DoctorReport:
    return DoctorReport([DoctorFinding("runtime/python-version", Severity.INFO, "Python", "healthy")], [], 1, 0, False)


def test_doctor_help() -> None:
    result = CliRunner().invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    assert "--deep" in result.output


def test_doctor_json_is_machine_readable(monkeypatch) -> None:
    monkeypatch.setattr("mana_agent.commands.cli.run_doctor", lambda **_: _healthy_report())
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert "\x1b" not in result.output


def test_doctor_unresolved_finding_returns_one(monkeypatch) -> None:
    report = DoctorReport([DoctorFinding("config/schema", Severity.ERROR, "Bad config", "bad")], [], 1, 0, False)
    monkeypatch.setattr("mana_agent.commands.cli.run_doctor", lambda **_: report)
    assert CliRunner().invoke(app, ["doctor"]).exit_code == 1


def test_doctor_unknown_check_returns_two() -> None:
    result = CliRunner().invoke(app, ["doctor", "--only", "unknown/check"])
    assert result.exit_code == 2
    assert "Unknown doctor check" in result.output


def test_doctor_skip_is_forwarded(monkeypatch) -> None:
    captured: list[dict] = []
    def fake_run(**kwargs):
        captured.append(kwargs)
        return _healthy_report()
    monkeypatch.setattr("mana_agent.commands.cli.run_doctor", fake_run)
    assert CliRunner().invoke(app, ["doctor", "--skip", "config/schema"]).exit_code == 0
    assert captured[0]["skip"] == ["config/schema"]


def test_deep_check_is_skipped_in_normal_mode() -> None:
    normal, unknown = select_checks([], [], deep=False)
    deep, _ = select_checks([], [], deep=True)
    assert not unknown
    assert "integrations/codex-protocol" not in {check.check_id for check in normal}
    assert "integrations/codex-protocol" in {check.check_id for check in deep}


def test_doctor_redaction_removes_credentials() -> None:
    value = redact({"api_key": "not-visible", "message": "Bearer token-value https://x.test/?token=secret sk-abcdefghijklmnop ghp_abcdefghijklmnop"})
    assert "not-visible" not in str(value)
    assert "secret" not in str(value)
    assert "sk-abcdefghijklmnop" not in str(value)
    assert "ghp_abcdefghijklmnop" not in str(value)
