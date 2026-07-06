from __future__ import annotations

from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from mana_agent.commands import cli
from mana_agent.cli.chat_ui import ChatUIState
from mana_agent.cli.events import make_event
from mana_agent.cli.fullscreen_chat import _conversation_text
from mana_agent.commands.chat_cli import _should_use_coding_agent_turn
from mana_agent.commands.ui_helpers import (
    ChatLog,
    ChatLogRenderer,
    LiveToolActivity,
    _render_direct_command,
    _looks_like_edit_request,
    _looks_like_plan_trigger_request,
    _run_with_live_buffer,
    _use_live_tool_activity,
    emit_tool_event,
    set_active_chat_ui_state,
    set_active_tool_activity,
)
from mana_agent.multi_agent.runtime.coding_agent import CodingAgent

runner = CliRunner()


class DummySettings:
    openai_api_key = "test"
    openai_base_url = None
    openai_chat_model = "fake"
    openai_tool_worker_model = None
    openai_coding_planner_model = None
    openai_embed_model = "fake"
    default_top_k = 8
    coding_flow_max_turns = 5
    coding_flow_max_tasks = 20
    coding_plan_max_steps = 8
    coding_search_budget = 4
    coding_read_budget = 6
    coding_require_read_files = 2


def test_chat_intent_helpers_do_not_treat_plain_text_as_plan_or_edit() -> None:
    assert not _looks_like_plan_trigger_request("test")
    assert not _looks_like_edit_request("test")


def test_chat_intent_helpers_detect_plan_and_edit_requests() -> None:
    assert _looks_like_plan_trigger_request("give me an implementation plan for auth")
    assert _looks_like_plan_trigger_request("execute the plan")
    assert _looks_like_edit_request("fix src/mana_agent/commands/chat_cli.py")
    assert _looks_like_edit_request("build the missing auth module")
    assert _looks_like_edit_request("implement this")


def test_coding_agent_mode_routes_general_analysis_turns_to_coding_agent() -> None:
    assert _should_use_coding_agent_turn(
        coding_agent_available=True,
        agent_tools=True,
        edit_request=False,
        plan_trigger_request=False,
        force_plan_only_response=False,
        has_pending_prechecklist=False,
        coding_agent_is_custom=False,
    )
    assert not _should_use_coding_agent_turn(
        coding_agent_available=False,
        agent_tools=True,
        edit_request=True,
        plan_trigger_request=False,
        force_plan_only_response=False,
        has_pending_prechecklist=False,
        coding_agent_is_custom=False,
    )


def test_direct_ui_command_accepts_fullscreen_mode() -> None:
    console = Console(record=True)
    state = ChatUIState(
        repo_root=Path.cwd(),
        provider="openai",
        model="gpt-test",
        ui_mode="plain",
    )

    answer = _render_direct_command(
        console,
        "/ui",
        project_root=Path.cwd(),
        index_available=False,
        coding_agent_active=False,
        tool_worker_active=False,
        ui_state=state,
        raw_question="/ui fullscreen",
    )

    assert answer == "ui mode: fullscreen"
    assert state.ui_mode == "fullscreen"


def test_render_turn_summary_and_transparency_sections() -> None:
    summary = cli._render_turn_summary(
        answer="Decision: Use deterministic fallback checklist.",
        sources_count=2,
        warnings_count=1,
        tool_steps=3,
        changed_files_count=2,
        has_diff=True,
    )
    assert "Summary" in summary
    assert "Changed files: 2" in summary
    assert "Diff: yes" in summary

    turn = cli.ChatTurnTelemetry(
        turn_index=1,
        timestamp="2026-02-27T10:00:00",
        question="implement flow updates",
        answer_text="Decision: Use deterministic fallback checklist.",
        sources=[],
        warnings=["patch-only loop detected"],
        trace=[
            {
                "tool_name": "semantic_search",
                "status": "ok",
                "duration_ms": 2.1,
                "args_summary": "query=flow",
            }
        ],
        decisions=[{"decision": "Use deterministic fallback checklist", "rationale": "Planner parse failed"}],
        changed_files=["src/mana_agent/commands/cli.py"],
        has_diff=True,
    )

    console = Console(record=True)
    cli._render_turn_transparency(console, turn=turn, history=[turn])
    rendered = console.export_text()
    assert "Summary" in rendered
    assert "Steps" in rendered
    assert "Decisions" in rendered
    assert "History" in rendered
    assert "Session History" in rendered
    assert "10:00:00" in rendered


def test_tool_activity_buffer_renders_one_box_for_managed_request() -> None:
    console = Console(record=True)

    def _call(callbacks):
        _ = callbacks
        for tool_name in ("list_tools", "read_file"):
            emit_tool_event("start", tool_name, args="{}")
            emit_tool_event("end", tool_name, duration=0.0)
        return {"ok": True}

    result, debug_tail = _run_with_live_buffer(
        console,
        spinner_text="Coding…",
        fn=_call,
        callbacks=[],
    )

    assert result == {"ok": True}
    assert debug_tail == ""
    rendered = console.export_text()
    assert "Tool activity" not in rendered
    assert "─ thinking " not in rendered
    assert rendered.count("─ tools ") == 1
    assert "list_tools" in rendered
    assert "read_file" in rendered


def test_tool_activity_live_is_disabled_for_recorded_output() -> None:
    console = Console(record=True)

    assert _use_live_tool_activity(console) is False


def test_chat_log_renderer_shows_timeline_roles_and_updates_tool_rows() -> None:
    chat_log = ChatLog()
    chat_log.add_user("use logo.png or https://api.manadev.net/v1/projects/abcdef")
    chat_log.add_thinking("Locating logo references, patching, then verifying changes.")
    chat_log.start_tool("repo_search", tool_args='{"query":"logo.png"}', tool_call_id="call-1")
    chat_log.finish_tool("repo_search", duration=1.6, tool_call_id="call-1")
    chat_log.start_tool("apply_patch", tool_args='{"patch":"very long patch body"}', tool_call_id="call-2")
    chat_log.fail_tool("apply_patch", error="hunk not found", tool_call_id="call-2")
    chat_log.add_assistant("Updated README.md and verified the change.")
    chat_log.add_error("[INFO] hidden internal log line")
    chat_log.add_error("clean warning")

    console = Console(record=True, width=100)
    console.print(ChatLogRenderer(chat_log))
    rendered = console.export_text()

    assert "user" in rendered
    assert "thinking" in rendered
    assert rendered.count("repo_search") == 1
    assert "✓" in rendered and "repo_search" in rendered and "logo.png" in rendered
    assert "✗" in rendered and "apply_patch" in rendered and "hunk not found" in rendered
    assert "assistant" in rendered
    assert "Updated README.md" in rendered
    assert "https://api.manadev.net/v1/projects/abcdef" not in rendered
    assert "[INFO]" not in rendered
    assert "clean warning" in rendered


def test_fullscreen_conversation_text_prefers_answer_history() -> None:
    state = ChatUIState(repo_root=Path.cwd(), provider="openai", model="fake", ui_mode="fullscreen")
    turn_id = "turn-test"
    state.start_turn(turn_id)
    state.record_event(
        make_event(
            "agent.decision",
            title="Agent decision",
            message="Handled by direct chat command without repository tool routing.",
            status="success",
            session_id=state.session_id,
            turn_id=turn_id,
            step_id="05",
        ).finish(status="success")
    )
    state.finish_turn(turn_id)
    state.add_conversation_turn("help", "Available commands: /help, /clear, /exit")

    rendered = _conversation_text(state)

    assert "you: help" in rendered
    assert "assistant: Available commands" in rendered
    assert "Handled by direct chat command" not in rendered


def test_tool_activity_can_use_live_without_fallback_duplicate(monkeypatch) -> None:
    live_entries = 0
    live_transient_values: list[object] = []

    class _FakeLive:
        def __init__(self, *args, **kwargs) -> None:
            _ = args
            live_transient_values.append(kwargs.get("transient"))

        def __enter__(self):
            nonlocal live_entries
            live_entries += 1
            return self

        def __exit__(self, *exc_info) -> None:
            _ = exc_info

    monkeypatch.setenv("MANA_LIVE_TOOL_ACTIVITY", "1")
    monkeypatch.setattr("mana_agent.commands.ui_helpers.Live", _FakeLive)

    console = Console(record=True)

    def _call(callbacks):
        _ = callbacks
        emit_tool_event("start", "list_tools", args="{}")
        emit_tool_event("end", "list_tools", duration=0.0)
        return {"ok": True}

    result, _debug_tail = _run_with_live_buffer(
        console,
        spinner_text="Coding…",
        fn=_call,
        callbacks=[],
    )

    assert result == {"ok": True}
    assert live_entries == 1
    assert live_transient_values == [True]
    rendered = console.export_text()
    assert "Tool activity" not in rendered
    assert rendered.count("─ tools ") == 1


def test_fullscreen_mode_suppresses_legacy_tool_activity_box() -> None:
    console = Console(record=True)
    state = ChatUIState(
        repo_root=Path.cwd(),
        provider="openai",
        model="gpt-test",
        ui_mode="fullscreen",
    )
    state.tracker.start_turn("turn-1")
    set_active_chat_ui_state(state)

    def _call(callbacks):
        _ = callbacks
        emit_tool_event("start", "read_file", args='{"path":"README.md"}')
        emit_tool_event("end", "read_file", duration=0.0)
        return {"ok": True}

    try:
        result, _debug_tail = _run_with_live_buffer(
            console,
            spinner_text="Coding…",
            fn=_call,
            callbacks=[],
        )
    finally:
        set_active_chat_ui_state(None)

    assert result == {"ok": True}
    assert state.tool_runs
    rendered = console.export_text()
    assert "─ tools " not in rendered
    assert "read_file" not in rendered


def test_tool_activity_can_share_one_box_across_request_cycles() -> None:
    console = Console(record=True)
    activity = LiveToolActivity(spinner_text="Coding…")
    set_active_tool_activity(activity)
    try:
        for tool_name in ("list_tools", "read_file"):

            def _call(callbacks, tool_name=tool_name):
                _ = callbacks
                emit_tool_event("start", tool_name, args="{}")
                emit_tool_event("end", tool_name, duration=0.0)
                return {"tool": tool_name}

            _run_with_live_buffer(
                console,
                spinner_text="Coding…",
                fn=_call,
                callbacks=[],
                activity=activity,
                manage_live=False,
            )
    finally:
        set_active_tool_activity(None)
        console.print(activity)

    rendered = console.export_text()
    assert "Tool activity" not in rendered
    assert rendered.count("─ tools ") == 1
    assert "list_tools" in rendered
    assert "read_file" in rendered


def test_worker_request_error_renders_one_tool_activity_box_without_tool_call() -> None:
    console = Console(record=True)

    def _call(callbacks):
        _ = callbacks
        CodingAgent._log_worker_event(
            {
                "name": "worker_request_start",
                "data": {
                    "tool": "tool_worker",
                    "args": "Generate the full content for a new .gitignore file",
                },
            }
        )
        CodingAgent._log_worker_event(
            {
                "name": "worker_request_error",
                "data": {
                    "tool": "tool_worker",
                    "duration_seconds": 0.1,
                    "error": "tools_only_violation: tools-only mode requires at least one successful tool call",
                },
            }
        )
        return {"status": "warning"}

    result, _debug_tail = _run_with_live_buffer(
        console,
        spinner_text="Coding…",
        fn=_call,
        callbacks=[],
    )

    assert result == {"status": "warning"}
    rendered = console.export_text()
    assert "Tool activity" not in rendered
    assert rendered.count("─ tools ") == 1
    assert "tool_worker" in rendered
    assert "tools_only_violation" in rendered


def test_tool_activity_collapses_duplicate_tool_worker_rows() -> None:
    console = Console(record=True)

    def _call(callbacks):
        _ = callbacks
        for _index in range(3):
            emit_tool_event(
                "start",
                "tool_worker",
                args="Read the content of Front/.gitignore",
            )
            emit_tool_event("end", "tool_worker", duration=0.1)
        return {"ok": True}

    _run_with_live_buffer(
        console,
        spinner_text="Coding…",
        fn=_call,
        callbacks=[],
    )

    rendered = console.export_text()
    assert "Tool activity" not in rendered
    assert rendered.count("─ tools ") == 1
    assert rendered.count("tool_worker") == 1
    assert "Read the content of Front/.gitignore" in rendered


def test_tool_activity_keeps_repeated_inner_tool_rows_visible() -> None:
    console = Console(record=True)

    def _call(callbacks):
        _ = callbacks
        for event_id in ("read-1", "read-2"):
            emit_tool_event("start", "read_file", args='{"path":"Front/.gitignore"}', event_id=event_id)
            emit_tool_event("end", "read_file", duration=0.0, event_id=event_id)
        return {"ok": True}

    _run_with_live_buffer(
        console,
        spinner_text="Coding…",
        fn=_call,
        callbacks=[],
    )

    rendered = console.export_text()
    assert rendered.count("read_file") == 2


def test_tool_activity_failed_tool_error_is_not_summarized() -> None:
    console = Console(record=True, width=100)
    long_error = (
        "1 validation error for apply_patch patch\n"
        "Input should be a valid string, got dict instead\n"
        "See https://errors.pydantic.dev/ for details"
    )

    def _call(callbacks):
        _ = callbacks
        emit_tool_event("start", "apply_patch", args="{}")
        emit_tool_event("error", "apply_patch", error=long_error)
        return {"status": "warning"}

    _run_with_live_buffer(
        console,
        spinner_text="Coding…",
        fn=_call,
        callbacks=[],
    )

    rendered = console.export_text()
    assert "apply_patch" in rendered
    assert "Input should be a valid string" in rendered
    assert "https://errors.pydantic.dev/" not in rendered


def test_render_turn_transparency_preserves_multiline_command_preview() -> None:
    answer = (
        "Command surface:\n"
        "- `mana-agent` console script -> `mana_agent.commands.cli:app`\n\n"
        "Detected CLI subcommands:\n"
        "- `mana-agent ask`\n"
        "- `mana-agent chat`\n"
    )
    turn = cli.ChatTurnTelemetry(
        turn_index=1,
        timestamp="2026-06-18T00:34:27",
        question="all commands of this project?",
        answer_text=answer,
        sources=[object()] * 11,
    )

    console = Console(record=True, width=100)
    cli._render_turn_transparency(console, turn=turn, history=[turn])
    rendered = console.export_text()
    assert "Command surface:" in rendered
    assert "mana-agent ask" in rendered
    assert "Sources" in rendered
    assert "11" in rendered
    assert "00:34:27" in rendered


def test_render_coding_sections_contains_expected_blocks() -> None:
    console = Console(record=True)
    cli._render_coding_sections(
        console,
        {
            "plan": {
                "objective": "Ship flow command",
                "steps": [{"status": "in_progress", "title": "Wire command"}],
            },
            "progress": {"phase": "edit", "why": "working", "budgets": {"search_used": 1, "search_budget": 4, "read_used": 2, "read_budget": 6, "read_files_observed": 2, "required_read_files": 2}},
            "checklist": {"done": 1, "pending": 1, "blocked": 0, "total": 2},
            "actions_taken": [{"tool_name": "read_file", "status": "ok", "duration_ms": 1.2, "args_summary": "path=cli.py"}],
            "changed_files": ["src/mana_agent/commands/cli.py"],
            "static_analysis": {"finding_count": 0},
            "next_step": "Run targeted tests.",
            "warnings": ["planner fallback: deterministic checklist"],
        },
    )
    rendered = console.export_text()
    assert "Plan" in rendered
    assert "Progress" in rendered
    assert "Checklist" in rendered
    assert "Actions Taken" in rendered
    assert "Files Changed" in rendered
    assert "Verification" in rendered
    assert "Next Step" in rendered


def test_index_command_is_retired(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["index", str(tmp_path)])
    assert result.exit_code != 0
