"""Tests for auto-chat tool catalog emission into the chat CLI/TUI."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from mana_agent.chat.events import AssistantMessageEvent
from mana_agent.chat.history import ChatHistory
from mana_agent.cli.chat_ui import ChatUIState, render_startup_header
from mana_agent.cli.events import make_event
from mana_agent.commands.ui_helpers import _render_direct_command
from mana_agent.tools.catalog import (
    format_tool_catalog_plain,
    format_tool_catalog_summary,
    group_tool_catalog,
    list_auto_chat_tools,
)


def test_list_auto_chat_tools_includes_core_connectors() -> None:
    tools = list_auto_chat_tools(include_mcp_discovery=False)
    names = {entry.name for entry in tools}

    assert "web_search" in names
    assert "email_read" in names
    assert "email_search" in names
    assert "repo_search" in names
    assert "read_file" in names
    # MCP capability is always visible even without configured providers.
    assert any(entry.category == "mcp" for entry in tools)

    by_name = {entry.name: entry for entry in tools}
    assert "email" in by_name["email_read"].description.lower() or "message" in by_name[
        "email_read"
    ].description.lower()
    assert by_name["web_search"].category == "search"


def test_tool_catalog_grouping_and_summary() -> None:
    tools = list_auto_chat_tools(include_mcp_discovery=False)
    grouped = group_tool_catalog(tools)
    categories = [category for category, _items in grouped]
    assert "search" in categories
    assert "email" in categories

    summary = format_tool_catalog_summary(tools)
    assert "available" in summary
    assert "email" in summary
    assert "search" in summary

    plain = format_tool_catalog_plain(tools)
    assert "Auto-chat tools" in plain
    assert "email_read" in plain
    assert "web_search" in plain

    compact = format_tool_catalog_plain(tools, max_per_category=2)
    assert "Auto-chat tools" in compact
    assert "… +" in compact


def test_chat_ui_state_loads_available_tools(tmp_path: Path) -> None:
    state = ChatUIState(
        repo_root=tmp_path,
        provider="openai",
        model="gpt-test",
        ui_mode="plain",
    )
    names = {entry.name for entry in state.available_tools}
    assert "web_search" in names
    assert "email_read" in names
    assert len(state.available_tools) >= 10


def test_startup_header_emits_auto_tools(tmp_path: Path) -> None:
    console = Console(file=StringIO(), force_terminal=True, width=120, record=True)
    state = ChatUIState(
        repo_root=tmp_path,
        provider="openai",
        model="gpt-test",
        tools_enabled=True,
        memory_enabled=True,
        skills_status="indexed",
        ui_mode="rich",
    )
    render_startup_header(console, state)
    rendered = console.export_text()

    assert "Mana-Agent" in rendered
    assert "auto tools" in rendered.lower() or "available" in rendered.lower()
    # Catalog is printed by default (name + description), not only via /tools.
    assert "web_search" in rendered
    assert "email_read" in rendered
    assert "Auto-chat tools" in rendered
    assert any(event.type == "session.tools" for event in state.events)


def test_tui_welcome_hides_available_tools_catalog(monkeypatch) -> None:
    from mana_agent.tui.app import ManaChatApp

    history = ChatHistory()
    app = ManaChatApp(history=history, repo_root=Path.cwd(), model="gpt-test")
    monkeypatch.setattr(app, "call_after_refresh", lambda callback: None)

    app.on_mount()

    messages = [
        event.content
        for event in history.get_events()
        if isinstance(event, AssistantMessageEvent)
    ]
    assert len(messages) == 1
    assert "mana-agent" in messages[0]
    assert "Available auto-chat tools" not in messages[0]
    assert "web_search" not in messages[0]
    assert "email_read" not in messages[0]


def test_tools_command_lists_available_catalog(tmp_path: Path) -> None:
    console = Console(record=True, width=100)
    state = ChatUIState(
        repo_root=tmp_path,
        provider="openai",
        model="gpt-test",
        ui_mode="rich",
        log_path=tmp_path / "chat.log",
    )
    state.record_event(
        make_event(
            "tool_done",
            title="read_file",
            message="Read README.md",
            status="success",
            metadata={"tool_name": "read_file", "path": "README.md"},
        ).finish(status="success")
    )

    answer = _render_direct_command(
        console,
        "/tools",
        project_root=tmp_path,
        index_available=True,
        coding_agent_active=True,
        tool_worker_active=True,
        ui_state=state,
        raw_question="/tools",
    )
    rendered = console.export_text()

    assert "available tools" in answer.lower()
    assert "email_read" in rendered
    assert "web_search" in rendered
    assert "read_file" in rendered  # recent activity still shown
