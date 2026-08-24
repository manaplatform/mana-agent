from __future__ import annotations

import typer
from typer.testing import CliRunner

from mana_agent.commands import cli_internal, main_cli
from mana_agent.commands.cli import app
from mana_agent.workspaces.service import WorkspaceService


def test_root_chat_creates_one_session_without_shadow_main_agent_route(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana-home"))
    main_agent_created = False

    class FakeMainAgent:
        def __init__(self, _root, **_kwargs) -> None:
            nonlocal main_agent_created
            main_agent_created = True
            raise AssertionError(
                "Gateway-owned chat must not create a shadow MainAgent lifecycle."
            )

    def frontend_chat(
        root_dir: str = typer.Option(".", "--root-dir"),
        model: str | None = typer.Option(None, "--model"),
    ) -> None:
        del model
        session = WorkspaceService().open_chat_session(root_dir)

        # Chat is owned end-to-end by AgentChatGateway. This legacy pre-route
        # boundary must deliberately no-op instead of creating a second
        # MainAgent/TaskBoard lifecycle.
        task_id = cli_internal._record_multi_agent_request(
            root_dir,
            "chat command",
            entrypoint="chat",
            session_id=session.session_id,
        )
        assert task_id == ""

    chat_command = next(
        item for item in app.registered_commands if item.name == "chat"
    )
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
    assert main_agent_created is False


def test_chat_pre_route_is_noop_even_with_command_scope(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana-home"))
    main_agent_created = False

    class FakeMainAgent:
        def __init__(self, _root, **_kwargs) -> None:
            nonlocal main_agent_created
            main_agent_created = True
            raise AssertionError(
                "Chat pre-routing must not instantiate MainAgent."
            )

    monkeypatch.setattr(cli_internal, "MainAgent", FakeMainAgent)

    task_id = cli_internal._record_multi_agent_request(
        tmp_path,
        "hello",
        entrypoint="chat",
        command_scope=True,
        session_id="session_test",
    )

    assert task_id == ""
    assert main_agent_created is False
