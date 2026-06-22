from __future__ import annotations

from pathlib import Path

from rich.console import Console

from mana_analyzer.commands.ui_helpers import (
    _classify_direct_command,
    _extract_exact_search_query,
    _render_direct_command,
)


def test_classify_direct_command_matches_known_tokens() -> None:
    assert _classify_direct_command("ping") == "ping"
    assert _classify_direct_command("PING!") == "ping"
    assert _classify_direct_command("  status  ") == "status"
    assert _classify_direct_command("help") == "help"
    assert _classify_direct_command("hi") == "hi"
    assert _classify_direct_command("hello") == "hello"


def test_classify_direct_command_ignores_real_questions() -> None:
    assert _classify_direct_command("help me fix the parser") is None
    assert _classify_direct_command("how do I ping the server?") is None
    assert _classify_direct_command("") is None


def test_render_ping_returns_pong_without_index() -> None:
    console = Console(record=True, force_terminal=False)
    answer = _render_direct_command(
        console,
        "ping",
        project_root=Path("/tmp/proj"),
        index_available=False,
        coding_agent_active=False,
        tool_worker_active=False,
    )
    assert answer == "pong"


def test_render_status_reports_index_and_agent_state() -> None:
    console = Console(record=True, force_terminal=False)
    answer = _render_direct_command(
        console,
        "status",
        project_root=Path("/tmp/proj"),
        index_available=False,
        coding_agent_active=True,
        tool_worker_active=True,
    )
    assert "/tmp/proj" in answer
    assert "missing" in answer  # semantic index missing -> fallback messaging
    assert "coding agent: active" in answer
    assert "tool worker: active" in answer


def test_extract_exact_search_query() -> None:
    assert _extract_exact_search_query("grep parse_config") == "parse_config"
    assert _extract_exact_search_query('search for "foo bar"') == "foo bar"
    assert _extract_exact_search_query("where is login") == "login"
    assert _extract_exact_search_query("how does login work?") is None


def test_extract_exact_search_query_ignores_edit_requests() -> None:
    assert _extract_exact_search_query("find all models and update docs/models.md") is None
