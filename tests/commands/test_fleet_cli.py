from typer.testing import CliRunner

from mana_agent.commands.cli import app


def test_fleet_help_and_verify_help_are_registered() -> None:
    runner = CliRunner()
    root = runner.invoke(app, ["fleet", "--help"])
    verify = runner.invoke(app, ["fleet", "verify", "--help"])
    assert root.exit_code == 0
    assert "cross-platform" in root.output
    assert verify.exit_code == 0
    assert "--platform" in verify.output
    assert "--command" in verify.output
