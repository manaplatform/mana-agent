"""Unit and regression tests for normalized chat turn execution trace.

Verifies:
- Chronological progression through execution phases (routing, context, coding, tool, model, completion)
- In-place active step updates without duplicate rows
- Subtle running transitions and terminal state completions
- Association of coding and tool command details with specific steps
- Error, timeout, and cancellation cleanup of running steps
- Idempotent replay and reconnect safety
- Session isolation across /new
- Generic fallback for unknown future runtime phases
"""

from __future__ import annotations

import asyncio
import pytest

from mana_agent.chat.execution_trace import (
    ExecutionStep,
    ExecutionTrace,
    classify_runtime_phase,
)
from mana_agent.chat.events import (
    CodingActivityEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
    AssistantMessageEvent,
)
from mana_agent.chat.history import ChatHistory
from mana_agent.tui.app import ManaChatApp
from mana_agent.tui.widgets.execution_panel import ExecutionPanel
from mana_agent.tui.widgets.chat_log import ChatLog


def test_classify_runtime_phase_known_and_generic_fallback() -> None:
    # Routing
    phase, title = classify_runtime_phase("routing_started")
    assert phase == "routing"
    assert title == "Routing"

    phase, title = classify_runtime_phase("agent.routing")
    assert phase == "routing"

    # Context
    phase, title = classify_runtime_phase("context_preparation_started")
    assert phase == "context"
    assert title == "Context preparation"

    phase, title = classify_runtime_phase("context.budget")
    assert phase == "context"

    # Coding / Codex
    phase, title = classify_runtime_phase("coding_started", {"backend": "codex"})
    assert phase == "coding"
    assert title == "Codex"

    phase, title = classify_runtime_phase("command.started")
    assert phase == "coding"

    # Search
    phase, title = classify_runtime_phase("search_started")
    assert phase == "search"
    assert title == "Searching"

    phase, title = classify_runtime_phase("tool.started", {"tool_name": "repo_search"})
    assert phase == "search"
    assert title == "Searching"

    phase, title = classify_runtime_phase("tool.started", {"tool_name": "web_search"})
    assert phase == "search"
    assert title == "Searching"

    # Tool (non-search)
    phase, title = classify_runtime_phase("tool.started", {"tool_name": "bash"})
    assert phase == "tool"
    assert title == "Tool: bash"

    # Model
    phase, title = classify_runtime_phase("model_execution_started")
    assert phase == "model"
    assert title == "Model execution"

    phase, title = classify_runtime_phase("assistant.delta")
    assert phase == "model"

    # Completion
    phase, title = classify_runtime_phase("turn.finished")
    assert phase == "completion"
    assert title == "Completed"

    # Failure
    phase, title = classify_runtime_phase("turn.cancelled")
    assert phase == "failure"
    assert title == "Cancelled"

    phase, title = classify_runtime_phase("error")
    assert phase == "failure"
    assert title == "Failed"

    # Generic fallback for future runtime phases
    phase, title = classify_runtime_phase("audit_check.started")
    assert phase == "audit"
    assert title == "Audit"

    phase, title = classify_runtime_phase("deep_verification_step")
    assert phase == "deep"
    assert title == "Deep"


def test_execution_trace_chronological_progression_and_in_place_updates() -> None:
    trace = ExecutionTrace("turn-101")

    # 1. Routing phase starts
    s1 = trace.apply_event({
        "event_id": "ev-1",
        "event_type": "routing_started",
        "status": "running",
    })
    assert s1 is not None
    assert s1.phase == "routing"
    assert s1.status == "running"
    assert trace.active_step_id == "turn-101:routing"
    assert len(trace.steps) == 1

    # 1b. Routing completes with duration
    s1_done = trace.apply_event({
        "event_id": "ev-2",
        "event_type": "routing_completed",
        "status": "success",
        "duration_ms": 35,
    })
    assert len(trace.steps) == 1  # Updated in place, no duplicate row
    assert s1_done.status == "completed"
    assert s1_done.duration_ms == 35

    # 2. Context preparation starts
    s2 = trace.apply_event({
        "event_id": "ev-3",
        "event_type": "context_preparation_started",
        "status": "running",
    })
    assert len(trace.steps) == 2
    assert s2.phase == "context"
    assert s2.status == "running"

    # 2b. Context preparation completes
    trace.apply_event({
        "event_id": "ev-4",
        "event_type": "context_retrieval_completed",
        "status": "success",
        "duration_ms": 80,
    })
    assert len(trace.steps) == 2
    assert s2.status == "completed"
    assert s2.duration_ms == 80

    # 3. Coding step starts
    s3 = trace.apply_event({
        "event_id": "ev-5",
        "event_type": "coding_started",
        "status": "running",
        "metadata": {"backend": "codex"},
    })
    assert len(trace.steps) == 3
    assert s3.phase == "coding"
    assert s3.title == "Codex"
    assert s3.status == "running"

    # 3b. Sub-events (commands) within coding step
    trace.apply_event({
        "event_id": "ev-6",
        "event_type": "command.started",
        "status": "running",
        "metadata": {"command": "git status"},
    })
    assert len(trace.steps) == 3  # Sub-event attached to coding step, not extra step
    assert len(s3.sub_events) == 2

    # 3c. Coding finishes
    trace.apply_event({
        "event_id": "ev-7",
        "event_type": "coding.terminal",
        "status": "success",
        "duration_ms": 1200,
    })
    assert len(trace.steps) == 3
    assert s3.status == "completed"
    assert s3.duration_ms == 1200

    # 4. Tool step starts
    s4 = trace.apply_event({
        "event_id": "ev-8",
        "tool_call_id": "call-read-file",
        "event_type": "tool.started",
        "status": "running",
        "metadata": {"tool_name": "read_file", "path": "main.py"},
    })
    assert len(trace.steps) == 4
    assert s4.phase == "tool"
    assert s4.title == "Tool: read_file"
    assert s4.status == "running"

    # 4b. Tool finishes
    trace.apply_event({
        "event_id": "ev-9",
        "tool_call_id": "call-read-file",
        "event_type": "tool.finished",
        "status": "success",
        "duration_ms": 45,
        "detail": "read 120 lines",
    })
    assert len(trace.steps) == 4
    assert s4.status == "completed"
    assert s4.duration_ms == 45

    # 5. Model execution starts
    s5 = trace.apply_event({
        "event_id": "ev-10",
        "event_type": "assistant.started",
        "status": "running",
    })
    assert len(trace.steps) == 5
    assert s5.phase == "model"
    assert s5.status == "running"

    # 6. Turn completes
    s6 = trace.apply_event({
        "event_id": "ev-11",
        "event_type": "turn.finished",
        "status": "success",
    })
    assert trace.is_completed is True
    assert s5.status == "unverified"  # Root completion does not prove an unobserved model outcome
    assert s6.status == "completed"
    assert len(trace.steps) == 6


def test_execution_trace_error_and_cancellation_cleanup() -> None:
    trace = ExecutionTrace("turn-fail")

    # Start routing
    trace.apply_event({
        "event_id": "r-1",
        "event_type": "routing_started",
        "status": "running",
    })
    assert trace.active_step_id == "turn-fail:routing"

    # Mark failed
    trace.mark_failed("Network timeout connecting to model")
    assert trace.is_failed is True
    assert trace.active_step_id is None
    step = trace.get_step("turn-fail:routing")
    assert step is not None
    assert step.status == "failed"
    assert "timeout" in step.detail

    # Test cancellation on another trace
    cancel_trace = ExecutionTrace("turn-cancel")
    cancel_trace.apply_event({
        "event_id": "c-1",
        "event_type": "coding_started",
        "status": "running",
    })
    cancel_trace.mark_failed("Interrupted by user", cancelled=True)
    assert cancel_trace.is_cancelled is True
    assert cancel_trace.active_step_id is None
    step_cancel = cancel_trace.get_step("turn-cancel:coding")
    assert step_cancel is not None
    assert step_cancel.status == "cancelled"


def test_execution_trace_idempotent_replay() -> None:
    trace = ExecutionTrace("turn-replay")
    event = {
        "event_id": "idemp-1",
        "event_type": "context_preparation_started",
        "status": "running",
    }

    # Applying the identical event multiple times produces exactly one step
    trace.apply_event(event)
    trace.apply_event(event)
    trace.apply_event(event)
    assert len(trace.steps) == 1
    assert len(trace.steps[0].sub_events) == 1


def test_execution_stream_exposes_live_state_and_terminal_outcome() -> None:
    panel = ExecutionPanel(turn_id="turn-live")
    panel.update_event({"type": "command.started", "operation_id": "one", "command": "git status", "status": "running"})
    assert panel.live_label() == "Running command"
    panel.update_event({"type": "turn.finished", "status": "success"})
    assert panel.live_label() == ""
    assert panel._terminal
    assert panel._status(panel.events[0]) == "unverified"


def test_chat_log_places_panel_under_user_message_and_clears_on_new() -> None:
    history = ChatHistory()
    app = ManaChatApp(history=history)

    async def run() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            # Add user message
            user_ev = UserMessageEvent(content="List all tests", turn_id="turn-user-1")
            history.add(user_ev)
            await pilot.pause()

            chat_log = app.query_one(ChatLog)
            assert chat_log.get_execution_panel("turn-user-1") is None

            # Post tool activity
            call_ev = ToolCallEvent(
                turn_id="turn-user-1",
                call_id="c-1",
                tool_name="list_dir",
                args={"path": "."},
            )
            history.add(call_ev)
            await pilot.pause()

            # Panel is mounted directly under user message upon activity
            panel = chat_log.get_execution_panel("turn-user-1")
            assert panel is not None
            assert panel in chat_log.children
            user_msg = app.query_one(".user-message-row")
            assert list(chat_log.children).index(panel) == list(chat_log.children).index(user_msg) + 1

            assert panel.events[0]["tool_name"] == "list_dir"
            assert panel.events[0]["operation_id"] == "c-1"

            # Post assistant message
            asst_ev = AssistantMessageEvent(
                content="Here are the tests",
                turn_id="turn-user-1",
            )
            history.add(asst_ev)
            await pilot.pause()
            assert not panel._terminal
            history.add(CodingActivityEvent(activity={"type": "turn.finished", "status": "success"}, turn_id="turn-user-1"))
            await pilot.pause()
            assert panel._terminal

            # Clear log on /new
            chat_log.clear_log()
            for _ in range(5):
                if not list(chat_log.query(".user-message-row, AssistantReply, ExecutionPanel")):
                    break
                await pilot.pause(0.05)
            assert not list(chat_log.query(".user-message-row, AssistantReply, ExecutionPanel"))
            assert chat_log.get_execution_panel("turn-user-1") is None

    asyncio.run(run())


def test_execution_trace_and_activity_boxes_handle_string_args_without_error() -> None:
    """Verify that string (JSON-encoded) args in ToolCallEvent do not trigger AttributeError."""
    trace = ExecutionTrace(turn_id="turn-str-args")
    trace.apply_event({
        "event_id": "tc-string-args",
        "event_type": "coding.progress",
        "title": "Code: git_status",
        "tool_name": "git_status",
        "tool_call_id": "call_123",
        "status": "running",
        "detail": '{"repo_path": "/Users/ah/Documents/mana-agent"}',
        "backend": "coding",
        "payload": {
            "args": '{"repo_path": "/Users/ah/Documents/mana-agent", "path": "src/mana_agent/app.py"}',
        },
    })
    assert len(trace.coding_events) == 1
    assert trace.changed_files == []  # Inspection arguments are not mutation evidence


def test_operation_expansion_survives_completion_and_sibling_activity() -> None:
    import asyncio
    from textual.app import App, ComposeResult
    from mana_agent.tui.widgets.operation_row import OperationRow

    class Host(App):
        def compose(self) -> ComposeResult:
            yield ExecutionPanel(turn_id="turn-open-test")

    async def run() -> None:
        app = Host()
        async with app.run_test() as pilot:
            panel = app.query_one(ExecutionPanel)
            panel.update_event({"type": "tool.started", "tool_name": "web_search",
                                "tool_call_id": "c1", "status": "running"})
            await pilot.pause(.12)
            row = panel.query_one(OperationRow)
            row.collapsed = False
            panel.update_event({"type": "tool.finished", "tool_name": "web_search",
                                "tool_call_id": "c1", "status": "success", "detail": "found 5 links"})
            panel.update_event({"type": "tool.started", "tool_name": "fetch_page",
                                "tool_call_id": "c2", "status": "running"})
            await pilot.pause(.12)
            siblings = list(panel.query(OperationRow))
            assert siblings[0] is row and not row.collapsed
            assert "found 5 links" in row.output.text
            assert "fetch_page" in siblings[1].title and siblings[1].collapsed

            row.collapsed = True
            panel.update_event({"type": "tool.finished", "tool_name": "fetch_page",
                                "tool_call_id": "c2", "status": "success"})
            await pilot.pause(.12)
            assert row.collapsed

    asyncio.run(run())


def test_lifecycle_events_do_not_create_duplicate_steps() -> None:
    """Backend selection and turn start events must not appear as task execution steps."""
    trace = ExecutionTrace("turn-lifecycle")

    step1 = trace.apply_event({
        "event_id": "ev-backend",
        "event_type": "backend.selected",
        "backend": "internal",
        "title": "Internal backend selected",
        "summary": "Executing coding task via Mana-Agent Native Coding Engine",
    })
    assert step1 is None
    assert len(trace.steps) == 0

    step2 = trace.apply_event({
        "event_id": "ev-turn-start",
        "event_type": "turn.started",
        "backend": "internal",
        "title": "Internal coding turn started",
    })
    assert step2 is None
    assert len(trace.steps) == 0


def test_tool_execution_start_and_finish_update_step_in_place() -> None:
    """Tool start and completion events correlate by call_id and update the step in place."""
    trace = ExecutionTrace("turn-tool-update")

    # Start git_status
    step_start = trace.apply_event({
        "event_id": "internal-1-started",
        "event_type": "tool.call.started",
        "tool_name": "git_status",
        "title": "git_status",
        "status": "running",
        "summary": '{"repo_path": "/workspace", "short": true}',
        "payload": {"call_id": "internal-1"},
    })
    assert step_start is not None
    assert len(trace.steps) == 1
    assert step_start.status == "running"
    assert step_start.title == "git_status"

    # Finish git_status
    step_done = trace.apply_event({
        "event_id": "internal-1-completed",
        "event_type": "tool.call.completed",
        "tool_name": "git_status",
        "title": "git_status",
        "status": "success",
        "duration_ms": 114,
        "payload": {"call_id": "internal-1"},
    })
    assert len(trace.steps) == 1
    assert step_done is step_start
    assert step_done.status == "completed"
    assert step_done.duration_ms == 114

    # Start git_create_branch
    step_b_start = trace.apply_event({
        "event_id": "internal-2-started",
        "event_type": "tool.call.started",
        "tool_name": "git_create_branch",
        "title": "git_create_branch",
        "status": "running",
        "payload": {"call_id": "internal-2"},
    })
    assert len(trace.steps) == 2
    assert step_b_start.status == "running"

    # Fail git_create_branch (approval required)
    step_b_fail = trace.apply_event({
        "event_id": "internal-2-failed",
        "event_type": "tool.call.failed",
        "tool_name": "git_create_branch",
        "title": "git_create_branch",
        "status": "failed",
        "duration_ms": 777,
        "detail": "The action requires approval bound to its exact preview and policy decision.",
        "payload": {"call_id": "internal-2"},
    })
    assert len(trace.steps) == 2
    assert step_b_fail is step_b_start
    assert step_b_fail.status == "failed"
    assert step_b_fail.duration_ms == 777
    assert "approval bound" in step_b_fail.detail


def test_execution_stream_preserves_main_and_coding_provenance() -> None:
    panel = ExecutionPanel(turn_id="turn-live-stream")
    panel.update_event({
        "event_id": "main-plan", "type": "agent.planning", "status": "running",
        "operation_id": "planning", "agent_id": "MainAgent", "title": "Reviewing requirements",
    })
    panel.update_event({
        "event_id": "internal-read", "type": "tool.call.started", "status": "running",
        "operation_id": "read", "backend": "internal", "tool_name": "read_file",
        "agent_id": "CodingAgent", "title": "Reading implementation",
    })
    assert [row["event_id"] for row in panel.events] == ["main-plan", "internal-read"]
    assert [row["agent_id"] for row in panel.events] == ["MainAgent", "CodingAgent"]
    assert not panel.earlier.display and not panel.newer.display
