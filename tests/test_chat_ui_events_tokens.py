from __future__ import annotations

import json
import os
from pathlib import Path

from rich.console import Console

from mana_agent.cli.chat_ui import ChatUIState, compact_path, default_ui_mode, render_startup_header, render_status
from mana_agent.cli.events import make_event
from mana_agent.cli.menu import MenuOption, select_option
from mana_agent.cli.renderers import EventRenderer, InlineChatRenderer
from mana_agent.telemetry.tokens import TokenUsageTracker, token_usage_from_provider


def _render_to_text(renderable) -> str:
    console = Console(record=True, width=100)
    console.print(renderable)
    return console.export_text()


def test_token_usage_tracker_records_exact_provider_usage() -> None:
    tracker = TokenUsageTracker()
    tracker.start_turn("turn-1")
    usage = tracker.record_model_call(
        "call-1",
        usage={
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 20,
            "input_token_details": {"cached_tokens": 3, "cache_creation_tokens": 2},
            "output_token_details": {"reasoning_tokens": 6},
        },
        provider="openai",
        model="gpt-test",
        agent_id="main",
        step_id="05",
    )

    assert usage.estimated is False
    assert usage.input_tokens == 10
    assert usage.output_tokens == 4
    assert usage.reasoning_tokens == 6
    assert tracker.by_turn["turn-1"].total_tokens == 20
    assert tracker.by_provider_model["openai:gpt-test"].cached_input_tokens == 3


def test_token_usage_tracker_marks_missing_provider_usage_estimated() -> None:
    tracker = TokenUsageTracker()
    usage = tracker.record_model_call("call-1", usage=None, estimated_text="hello world")

    assert usage.estimated is True
    assert usage.total_tokens > 0


def test_provider_usage_object_normalizes_openai_style_fields() -> None:
    class Usage:
        input_tokens = 7
        output_tokens = 5
        total_tokens = 12
        input_token_details = {"cached_tokens": 2}
        output_token_details = {"reasoning_tokens": 1}

    usage = token_usage_from_provider(Usage())

    assert usage.input_tokens == 7
    assert usage.output_tokens == 5
    assert usage.cached_input_tokens == 2
    assert usage.reasoning_tokens == 1
    assert usage.estimated is False


def test_event_schema_contains_required_fields() -> None:
    event = make_event(
        "tool.finished",
        title="read_file",
        message="Read src/app.py",
        status="success",
        session_id="sess-1",
        turn_id="turn-1",
        agent_id="main",
        step_id="08",
        metadata={"path": "src/app.py"},
    ).finish(status="success")
    data = event.as_dict()

    for key in (
        "event_id",
        "parent_event_id",
        "session_id",
        "turn_id",
        "agent_id",
        "subagent_id",
        "step_id",
        "type",
        "status",
        "title",
        "summary",
        "started_at",
        "ended_at",
        "duration_ms",
        "token_usage",
        "metadata",
    ):
        assert key in data
    assert data["type"] == "tool.finished"
    assert data["kind"] == "tool"
    assert data["id"] == data["event_id"]
    assert data["details"]["path"] == "src/app.py"
    assert data["summary"] == "Read src/app.py"
    assert data["token_usage"] is None


def test_agent_event_aliases_and_append_only_status_update(tmp_path: Path) -> None:
    state = ChatUIState(repo_root=tmp_path, provider="openai", model="gpt-test", ui_mode="plain")
    event = make_event(
        "tool_started",
        title="repo_search",
        message="searching",
        status="running",
        metadata={"tool_name": "repo_search"},
    )

    assert event.kind == "tool"
    assert event.type == "tool.started"
    event.details = {"tool_name": "repo_search", "target": "chat tui"}
    event.parent_id = "parent-1"
    state.record_event(event)
    state.update_event_status(event.id, status="success", message="8 matches", details={"result_summary": "8 matches"})

    assert len(state.events) == 1
    assert len(state.normalized_events) == 1
    assert state.events[-1].id == event.id
    assert state.events[-1].parent_id == "parent-1"
    assert state.events[-1].status == "success"
    assert state.session_path is not None and state.session_path.exists()
    persisted = state.session_path.read_text(encoding="utf-8")
    assert '"kind": "tool"' in persisted


def test_timeline_normalizes_updates_and_hides_raw_event_names(tmp_path: Path) -> None:
    state = ChatUIState(repo_root=tmp_path, provider="openai", model="gpt-test", ui_mode="plain")
    event = state.record_event(
        make_event(
            "plan_step_started",
            title="Routing decision",
            message="Routing decision is running.",
            status="running",
        )
    )
    state.update_event_status(event.id, status="completed", message="Routing complete")

    rendered = _render_to_text(state.renderer.render_timeline(state.normalized_events))

    assert rendered.count("Routing complete") == 1
    assert "plan_step_started" not in rendered
    assert "Completed" not in rendered
    assert "✓" in rendered or "ok" in rendered


def test_timeline_truncates_long_reasoning_summaries() -> None:
    event = make_event(
        "thinking_summary",
        title="Reasoning",
        message="Internal analysis " * 30,
        status="success",
    ).finish(status="success")
    rendered = _render_to_text(EventRenderer(mode="plain").render_timeline([event]))

    assert "Reasoning updated" in rendered
    assert "Internal analysis Internal analysis" not in rendered


def test_event_renderer_modes_render_without_raw_json_noise() -> None:
    event = make_event(
        "agent.decision",
        title="Agent decision",
        message="Decision summary: inspect CLI renderer before editing.",
        status="success",
        step_id="05",
    ).finish(status="success")

    rich_text = _render_to_text(EventRenderer(mode="rich").render_event(event))
    compact_text = _render_to_text(EventRenderer(mode="compact").render_event(event))
    plain_text = str(EventRenderer(mode="plain").render_event(event))
    json_text = str(EventRenderer(mode="json").render_event(event))

    assert "Agent decision" in rich_text
    assert "Agent decision" in compact_text
    assert "inspect CLI renderer" in plain_text
    assert json.loads(json_text)["type"] == "agent.decision"
    assert EventRenderer.normalize_mode("fullscreen") == "rich"


def test_event_renderer_renders_inline_status_and_tabs() -> None:
    events = [
        make_event(
            "file_read",
            title="README.md",
            status="success",
            metadata={"path": "README.md"},
        ).finish(status="success"),
        make_event(
            "patch_applied",
            title="src/app.py",
            message="Updated UI",
            status="success",
            metadata={"path": "src/app.py", "insertions": 3, "deletions": 1},
        ).finish(status="success"),
        make_event(
            "test_done",
            title="pytest tests/test_chat_ui.py",
            status="success",
            metadata={"command": "pytest tests/test_chat_ui.py", "result_summary": "passed"},
        ).finish(status="success"),
    ]
    renderer = EventRenderer(mode="plain")

    inline = str(renderer.render_inline_status(events))
    timeline = _render_to_text(renderer.render_timeline(events))
    files = _render_to_text(renderer.render_files(events))
    diff = _render_to_text(renderer.render_diff(events))
    tests = _render_to_text(renderer.render_tests(events))

    assert "files read 1" in inline
    assert "files changed 1" in inline
    assert "Timeline" in timeline
    assert "README.md" in files
    assert "src/app.py" in diff
    assert "pytest tests/test_chat_ui.py" in tests


def test_default_ui_mode_keeps_non_tty_plain(monkeypatch) -> None:
    monkeypatch.delenv("MANA_CHAT_UI", raising=False)
    monkeypatch.delenv("CI", raising=False)
    console = Console(record=True, width=140)

    assert default_ui_mode(console) == "plain"


def test_env_ui_mode_rejects_fullscreen() -> None:
    console = Console(record=True, width=100)
    old = os.environ.get("MANA_CHAT_UI")
    try:
        os.environ["MANA_CHAT_UI"] = "fullscreen"
        assert default_ui_mode(console) == "plain"
    finally:
        if old is None:
            os.environ.pop("MANA_CHAT_UI", None)
        else:
            os.environ["MANA_CHAT_UI"] = old


def test_default_ui_mode_is_terminal_native_not_fullscreen(monkeypatch) -> None:
    monkeypatch.delenv("MANA_CHAT_UI", raising=False)
    monkeypatch.delenv("CI", raising=False)
    console = Console(force_terminal=True, width=140)

    assert default_ui_mode(console) in {"rich", "compact"}
    assert default_ui_mode(console) != "fullscreen"


def test_inline_renderer_collapses_repeated_events_and_avoids_timeline_panel() -> None:
    console = Console(record=True, width=100)
    renderer = InlineChatRenderer(console, mode="rich")
    event = make_event(
        "RoutingStarted",
        title="routing request",
        message="Running.",
        status="running",
    )

    renderer.render_event(event)
    renderer.render_event(event)

    rendered = console.export_text()
    assert rendered.count("routing request") == 1
    assert "Timeline" not in rendered
    assert "╭" not in rendered


def test_inline_renderer_renders_tool_and_subagent_events_compactly() -> None:
    console = Console(record=True, width=100)
    renderer = InlineChatRenderer(console, mode="rich")
    renderer.render_event(
        make_event(
            "ToolStarted",
            title="repo_search",
            status="running",
            metadata={"tool_name": "repo_search", "args_summary": '"openclaw"'},
        )
    )
    renderer.render_event(
        make_event(
            "ToolCompleted",
            title="repo_search",
            message="12 matches",
            status="success",
            metadata={"tool_name": "repo_search", "result_summary": "12 matches"},
        ).finish(status="success")
    )
    renderer.render_event(
        make_event(
            "SubagentCreated",
            title="coding agent",
            status="success",
            subagent_id="coding-agent-0002",
            metadata={"role": "refactor chat renderer"},
        ).finish(status="success")
    )
    renderer.render_event(
        make_event(
            "SubagentCompleted",
            title="coding agent",
            status="success",
            subagent_id="coding-agent-0002",
        ).finish(status="success")
    )

    rendered = console.export_text()
    # Running ("ToolStarted") tool events are suppressed in the InlineChatRenderer transcript
    # (in-progress progress is handled by LiveToolActivity); only terminal state produces a line.
    assert '→ tool repo_search "openclaw"' not in rendered
    assert "✓ repo_search 12 matches" in rendered
    assert "↳ subagent coding-agent-0002 created: refactor chat renderer" in rendered
    assert "  ✓ coding-agent-0002 completed" in rendered
    assert "{" not in rendered


def test_inline_renderer_final_response_appears_once() -> None:
    console = Console(record=True, width=100)
    renderer = InlineChatRenderer(console, mode="rich")

    renderer.render_final("Final **answer**.")

    rendered = console.export_text()
    assert rendered.count("Final answer.") == 1
    assert "Timeline" not in rendered


def test_tools_and_subagents_render_from_events_only() -> None:
    tool_event = make_event(
        "tool.failed",
        title="run_tests",
        message="pytest failed",
        status="failed",
        step_id="09",
        agent_id="subagent_tool_worker_0001",
        subagent_id="subagent_tool_worker_0001",
        metadata={"tool_name": "run_tests", "args_summary": "pytest tests/test_cli_ui.py", "result_summary": "1 failed"},
    ).finish(status="failed")
    subagent_event = make_event(
        "subagent.finished",
        title="test-runner",
        message="Tests passed",
        status="success",
        agent_id="main",
        subagent_id="A-003",
        metadata={"role": "test-runner", "current_step": "verification"},
    ).finish(status="success")
    renderer = EventRenderer(mode="rich")

    tool_text = _render_to_text(renderer.render_tool_activity([tool_event]))
    subagent_text = _render_to_text(renderer.render_subagents([subagent_event]))

    assert "run_tests" in tool_text
    assert "subagent_tool_worker_0001" in tool_text
    assert "pytest" in tool_text
    assert "tests/test_cli_ui.py" in tool_text
    assert "A-003" in subagent_text
    assert "test-runner" in subagent_text


def test_tool_activity_keeps_nested_subagent_events_with_shared_step_id() -> None:
    events = [
        make_event(
            "tool.started",
            title="tool_worker",
            message="Run planner-selected repository discovery for rendering",
            status="running",
            turn_id="turn-1",
            step_id="07",
            agent_id="subagent_tool_worker_0001",
            subagent_id="subagent_tool_worker_0001",
            metadata={
                "tool_name": "tool_worker",
                "args_summary": "Run planner-selected repository discovery for rendering",
                "agent_role": "tool_worker",
                "model_level": "MODEL_LEVEL_1_FAST_TOOL",
                "resolved_model": "fast-model",
            },
        ),
        make_event(
            "tool.finished",
            title="ls",
            status="success",
            turn_id="turn-1",
            step_id="07",
            agent_id="subagent_tool_worker_0001",
            subagent_id="subagent_tool_worker_0001",
            metadata={"tool_name": "ls", "agent_role": "tool_worker"},
        ),
        make_event(
            "tool.finished",
            title="list_files",
            status="success",
            turn_id="turn-1",
            step_id="07",
            agent_id="subagent_tool_worker_0001",
            subagent_id="subagent_tool_worker_0001",
            metadata={"tool_name": "list_files", "agent_role": "tool_worker"},
        ),
        make_event(
            "tool.finished",
            title="read_file",
            status="success",
            turn_id="turn-1",
            step_id="07",
            agent_id="subagent_tool_worker_0001",
            subagent_id="subagent_tool_worker_0001",
            metadata={"tool_name": "read_file", "agent_role": "tool_worker"},
        ),
    ]
    for duration, event in zip((0, 376, 944, 3), events):
        event.duration_ms = duration
    renderer = EventRenderer(mode="rich")

    rich_text = _render_to_text(renderer.render_tool_activity(events))
    compact_text = _render_to_text(EventRenderer(mode="compact").render_tool_activity(events))
    subagent_text = _render_to_text(renderer.render_subagents(events))

    for expected in ("subagent_tool_worker_0001", "tool_worker", "ls", "list_files", "read_file"):
        assert expected in rich_text
        assert expected in compact_text
    # The subagent ID may be truncated in the rendered table under the test console width (100).
    # Check for a stable prefix that survives column truncation.
    assert "subagent_tool" in subagent_text
    assert "MODEL_LEVEL_1_FAST_TOOL" in subagent_text
    assert "tool_worker" in subagent_text
    assert "MODEL_LEVEL_1_FAST_TOOL" in rich_text
    assert "fast-model" in rich_text
    assert "list_files ✓ 944ms" in compact_text


def test_chat_ui_startup_header_and_token_command_render() -> None:
    console = Console(record=True, width=100)
    state = ChatUIState(
        repo_root=Path.cwd(),
        provider="openai",
        model="gpt-test",
        tools_enabled=True,
        memory_enabled=True,
        skills_status="indexed",
        ui_mode="rich",
    )
    render_startup_header(console, state)
    state.tracker.start_turn("turn-1")
    state.tracker.record_tool_result("tool-1", "some result text", turn_id="turn-1")
    console.print(state.renderer.render_tokens(state.tracker))
    rendered = console.export_text()

    assert "Mana-Agent" in rendered
    assert "/tokens" in rendered
    assert "Token usage" in rendered
    assert "~" in rendered
    assert "Chat Mode" not in rendered
    assert "[INFO]" not in rendered


def test_arrow_menu_helper_fallback_accepts_number_and_alias() -> None:
    options = [
        MenuOption("chat", "Chat with repo", ("1", "c")),
        MenuOption("exit", "Exit", ("2", "q")),
    ]

    assert select_option(title="Menu", text="Pick", options=options, input_func=lambda _p: "1") == "chat"
    assert select_option(title="Menu", text="Pick", options=options, input_func=lambda _p: "q") == "exit"


def test_startup_header_is_compact_and_uses_clean_prompt() -> None:
    console = Console(record=True, width=120)
    state = ChatUIState(
        repo_root=Path.cwd(),
        provider="openai",
        model="gpt-test",
        tools_enabled=True,
        memory_enabled=True,
        skills_status="indexed",
        ui_mode="rich",
    )

    render_startup_header(console, state)
    rendered = console.export_text()

    assert "mana ❯" not in rendered
    assert "Mana-Agent" in rendered
    assert "Ready. Ask for code changes" in rendered
    assert len([line for line in rendered.splitlines() if line.strip()]) <= 9
    assert not any("╭" in line or "┌" in line for line in rendered.splitlines())


def test_status_full_includes_trace_and_log_paths(tmp_path: Path) -> None:
    state = ChatUIState(
        repo_root=Path.cwd(),
        provider="openai",
        model="gpt-test",
        skills_status="indexed",
        ui_mode="plain",
        log_path=tmp_path / "chat.log",
    )

    compact = _render_to_text(render_status(state, full=False))
    full = _render_to_text(render_status(state, full=True))

    assert "trace path" not in compact
    assert "trace path" in full
    assert "log path" in full


def test_renderer_trace_logs_and_path_truncation(tmp_path: Path) -> None:
    log_path = tmp_path / "chat.log"
    log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    state = ChatUIState(repo_root=Path.cwd(), provider="openai", model="gpt-test", log_path=log_path)

    rendered = _render_to_text(state.renderer.render_log_lines(["two", "three"]))

    assert "Trace logs" in rendered
    assert "three" in rendered
    assert compact_path("/Users/ah/Documents/mana-agent/src/mana_agent/commands/chat_ui.py", width=42).startswith("/")
    assert len(compact_path("/Users/ah/Documents/mana-agent/src/mana_agent/commands/chat_ui.py", width=42)) <= 42
