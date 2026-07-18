from __future__ import annotations

from types import SimpleNamespace

import typer
from typer.testing import CliRunner

from mana_agent.commands import cli_internal, main_cli
from mana_agent.commands.cli import app
from mana_agent.workspaces.service import WorkspaceService


def test_root_chat_creates_one_session_before_routing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana-home"))
    routed_session_ids: list[str | None] = []

    class FakeMainAgent:
        def __init__(self, _root, **kwargs) -> None:
            routed_session_ids.append(kwargs.get("session_id"))

        def run_user_request(self, _request: str, *, entrypoint: str = "chat"):
            return SimpleNamespace(task_id="task_test", entrypoint=entrypoint)

    def frontend_chat(
        root_dir: str = typer.Option(".", "--root-dir"),
        model: str | None = typer.Option(None, "--model"),
    ) -> None:
        del model
        session = WorkspaceService().open_chat_session(root_dir)
        cli_internal._record_multi_agent_request(
            root_dir,
            "chat command",
            entrypoint="chat",
            session_id=session.session_id,
        )

    chat_command = next(item for item in app.registered_commands if item.name == "chat")
    monkeypatch.setattr(chat_command, "callback", frontend_chat)
    monkeypatch.setattr(cli_internal, "MainAgent", FakeMainAgent)
    monkeypatch.setattr(main_cli, "ensure_setup", lambda **_kwargs: None)
    monkeypatch.setattr(main_cli, "render_banner", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(
        app,
        ["--chat", "--repo", str(tmp_path), "--no-banner"],
    )

    assert result.exit_code == 0, result.output
    sessions = WorkspaceService().store.list_sessions()
    assert len(sessions) == 1
    assert routed_session_ids == [sessions[0].session_id]
