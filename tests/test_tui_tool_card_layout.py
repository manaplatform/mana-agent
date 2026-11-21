"""Regression coverage for Textual inline operation sizing and selectable transcript widgets."""

from __future__ import annotations

import asyncio

from mana_agent.chat.events import (
    AssistantMessageEvent,
    CodingActivityEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from mana_agent.chat.history import ChatHistory
from mana_agent.tui.app import ManaChatApp
from mana_agent.tui.widgets.selectable_text import SelectableText
from mana_agent.tui.widgets.execution_panel import ExecutionPanel
from mana_agent.tui.widgets.operation_row import OperationRow
from mana_agent.tui.widgets.chat_log import AssistantReply


def _run(coroutine) -> None:  # noqa: ANN001
    asyncio.run(coroutine)


def _rendered_source(widget: SelectableText) -> str:
    return widget.text


def test_message_widgets_support_textual_mouse_selection_and_copy(wait_for_tui_widgets) -> None:
    history = ChatHistory()
    app = ManaChatApp(history=history)

    async def run() -> None:
        async with app.run_test() as pilot:
            history.add(UserMessageEvent(content="copy this user message"))
            history.add(
                AssistantMessageEvent(
                    content="# Copy this markdown\n\n```python\nvalue = 1\n```"
                )
            )
            await wait_for_tui_widgets(pilot, ".user-message")
            await wait_for_tui_widgets(pilot, ".assistant-message")

            chat_log = app.query_one("#chat-log")
            user_message = chat_log.query_one(".user-message", SelectableText)
            reply = chat_log.query_one(AssistantReply)
            assert reply.markdown.display
            reply.copy_mode.press()
            await wait_for_tui_widgets(pilot, ".assistant-message > SelectableText")
            assistant_message = reply.source

            assert app.mouse_captured is None
            assert user_message.can_focus
            assert assistant_message.can_focus
            user_message.selection = ((0, 0), (0, 4))
            copied: list[str] = []
            app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
            user_message.focus()
            await pilot.press("ctrl+c")
            assert copied == ["copy"]

    _run(run())


def test_read_only_messages_do_not_paint_an_inner_cursor_line() -> None:
    """Chat cards own the border; selectable text must not add a second box."""
    assert ".text-area--cursor-line" in SelectableText.DEFAULT_CSS
    assert "background: transparent" in SelectableText.DEFAULT_CSS


def test_activity_updates_one_turn_scoped_row_live(wait_for_tui_widgets) -> None:
    history = ChatHistory()
    app = ManaChatApp(history=history)
    event = {"operation_id": "cmd", "backend": "codex", "command": "pytest -q"}

    async def run() -> None:
        async with app.run_test(size=(60, 24)) as pilot:
            app.chat_log.set_active_turn("turn")
            history.add(CodingActivityEvent(turn_id="turn", activity={
                **event, "type": "command.started", "status": "running",
            }))
            await wait_for_tui_widgets(pilot, "ExecutionPanel OperationRow")
            panel = app.query_one(ExecutionPanel)
            row = app.query_one(OperationRow)
            assert "Running pytest -q" in row.title
            assert "Running command" in str(app.chat_log.live.renderable)

            history.add(CodingActivityEvent(turn_id="turn", activity={
                **event, "type": "command.finished", "status": "success", "duration_ms": 12,
            }))
            await wait_for_tui_widgets(
                pilot, "ExecutionPanel OperationRow",
                ready=lambda: row.event["type"] == "command.finished",
            )
            assert list(app.query(ExecutionPanel)) == [panel]
            assert list(app.query(OperationRow)) == [row]
            assert "Ran pytest -q" in row.title
            row.collapsed = False
            await wait_for_tui_widgets(
                pilot, "OperationRow OperationOutput", ready=lambda: bool(row.output.text),
            )
            assert "Duration (ms): 12" in row.output.text

    _run(run())


def test_history_messages_wrap_at_available_card_width_after_replay_append_and_resize(
    wait_for_tui_widgets,
) -> None:
    """Dynamically mounted history TextAreas must not retain a one-column wrap."""
    history = ChatHistory()
    history.add(UserMessageEvent(content="short user message"))
    history.add(
        AssistantMessageEvent(
            content=(
                "# پاسخ ✅\n\n"
                "متن فارسی و Unicode remain intact beside emoji.\n\n"
                "Visit [the docs](https://example.test/docs).\n\n"
                "```python\nresult = 'markdown code remains selectable'\n```\n\n"
                "- a markdown list item with enough ordinary prose to wrap at the card edge"
            )
        )
    )
    app = ManaChatApp(history=history)

    async def run() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            await wait_for_tui_widgets(pilot, ".user-message")
            await wait_for_tui_widgets(pilot, ".assistant-message")
            chat_log = app.query_one("#chat-log")
            user_message = chat_log.query_one(".user-message", SelectableText)
            reply = chat_log.query_one(AssistantReply)
            assert reply.markdown.display
            reply.copy_mode.press()
            await wait_for_tui_widgets(pilot, ".assistant-message > SelectableText")
            assistant_message = reply.source

            # Replayed short text has the full message-card width rather than one
            # character per row. Read-only messages do not reserve an editable
            # cursor cell; card borders and padding remain part of the box model.
            assert user_message.wrapped_document.height == 1
            assert (
                user_message.wrap_width
                == user_message.scrollable_content_region.width
            )
            assert user_message.region.width == user_message.content_size.width
            assert user_message.size.width > 1
            assert (
                assistant_message.wrap_width
                == assistant_message.scrollable_content_region.width
            )
            assert assistant_message.wrapped_document.height < len(
                assistant_message.text
            )
            assert "متن فارسی" in assistant_message.text
            assert "https://example.test/docs" in assistant_message.text
            assert "```python" in assistant_message.text

            history.add(AssistantMessageEvent(content="short assistant message"))
            await wait_for_tui_widgets(pilot, ".assistant-message", count=2)
            await wait_for_tui_widgets(pilot, ".assistant-message:last-of-type > Markdown")
            live_assistant = list(chat_log.query(AssistantReply))[-1].markdown
            assert live_assistant.region.height >= 1

            wide_wrap_width = assistant_message.wrap_width
            await pilot.resize_terminal(48, 24)
            await pilot.pause()
            assert assistant_message.wrap_width < wide_wrap_width
            assert assistant_message.wrapped_document.height > 1
            assert user_message.wrapped_document.height == 1

            await pilot.resize_terminal(120, 24)
            await pilot.pause()
            assert assistant_message.wrap_width > wide_wrap_width
            assert assistant_message.wrapped_document.height < len(
                assistant_message.text
            )
            assert live_assistant.region.width > 1

    _run(run())


def test_expanded_operation_keeps_streaming_output_and_reflows(wait_for_tui_widgets) -> None:
    history = ChatHistory()
    app = ManaChatApp(history=history)
    correlation = {"operation_id": "check", "command": "pytest -q"}

    async def run() -> None:
        async with app.run_test(size=(90, 35)) as pilot:
            history.add(CodingActivityEvent(turn_id="turn", activity={
                **correlation, "type": "command.started", "status": "running", "metadata": {"cwd": "/workspace"},
            }))
            for offset, content in ((0, "first line\n"), (11, "second line\n")):
                activity = {
                    **correlation, "type": "command.output", "status": "running", "output_preview": content,
                    "metadata": {"stream_channel": "stdout", "stream_offset": offset, "content_length": len(content)},
                }
                history.add(CodingActivityEvent(turn_id="turn", activity=activity))
                history.add(CodingActivityEvent(turn_id="turn", activity=activity))
            await wait_for_tui_widgets(pilot, "ExecutionPanel OperationRow")

            row = app.query_one(OperationRow)
            row.collapsed = False
            await wait_for_tui_widgets(
                pilot, "OperationRow OperationOutput", ready=lambda: bool(row.output.text),
            )
            assert "$ pytest -q" in row.output.text
            assert "cwd: /workspace" in row.output.text
            assert row.output.text.count("first line") == 1
            assert row.output.text.count("second line") == 1
            wide = row.output.wrap_width
            await pilot.resize_terminal(48, 24)
            await pilot.pause()
            assert row.output.wrap_width < wide

            history.add(CodingActivityEvent(turn_id="turn", activity={
                **correlation, "type": "command.finished", "status": "failed", "duration_ms": 0,
                "metadata": {"exit_code": 1, "error": "pytest failed"},
            }))
            await wait_for_tui_widgets(
                pilot, "ExecutionPanel OperationRow",
                ready=lambda: row.event["type"] == "command.finished",
            )
            assert app.query_one(OperationRow) is row
            assert "Exit code: 1" in row.output.text
            assert "Duration (ms): 0" in row.output.text
            assert "pytest failed" in row.output.text
            assert "first line" in row.output.text

    _run(run())


def test_is_coding_tool_classification() -> None:
    from mana_agent.chat.execution_trace import is_coding_tool

    # Coding tools
    assert is_coding_tool("edit_file") is True
    assert is_coding_tool("multi_edit_file") is True
    assert is_coding_tool("apply_patch") is True
    assert is_coding_tool("apply_patch_batch") is True
    assert is_coding_tool("write_file") is True
    assert is_coding_tool("create_file") is True
    assert is_coding_tool("delete_file") is True
    assert is_coding_tool("read_file") is True
    assert is_coding_tool("repo_batch_read") is True
    assert is_coding_tool("shell") is True
    assert is_coding_tool("run_command") is True
    assert is_coding_tool("verify_project") is True
    assert is_coding_tool("git") is True
    assert is_coding_tool("git_status") is True
    assert is_coding_tool("git.diff") is True
    assert is_coding_tool("git_commit") is True

    # Non-coding tools
    assert is_coding_tool("web_search") is False
    assert is_coding_tool("github_search") is False
    assert is_coding_tool("repo_search") is False
    assert is_coding_tool("email_read") is False
    assert is_coding_tool("email_send") is False
    assert is_coding_tool("generate_image") is False
    assert is_coding_tool("api_request_execute") is False
    assert is_coding_tool("approval_request") is False


def test_runtime_and_legacy_tools_share_one_stream_and_keep_concurrent_attempts(
    wait_for_tui_widgets,
) -> None:
    history = ChatHistory()
    app = ManaChatApp(history=history)

    async def run() -> None:
        async with app.run_test(size=(90, 28)) as pilot:
            for call_id, name in (("read", "read_file"), ("search", "web_search")):
                history.add(ToolCallEvent(turn_id="turn", call_id=call_id, tool_name=name))
            await wait_for_tui_widgets(pilot, "ExecutionPanel OperationRow", count=2)
            panel = app.query_one(ExecutionPanel)
            initial_rows = {row.event.get("tool_name"): row for row in panel.query(OperationRow)}
            assert initial_rows.keys() == {"read_file", "web_search"}
            original = initial_rows["read_file"]
            assert "read_file" in original.title
            for attempt in ("a1", "a2"):
                history.add(CodingActivityEvent(turn_id="turn", activity={
                    "type": "tool.completed", "operation_id": "read", "tool_name": "read_file",
                    "task_id": "task", "attempt_id": attempt, "status": "success", "path": "src/settings.py",
                }))
            history.add(ToolResultEvent(turn_id="turn", call_id="read", tool_name="read_file", success=True))
            await wait_for_tui_widgets(pilot, "ExecutionPanel OperationRow", count=3)
            assert app.query_one(ExecutionPanel) is panel
            rows = list(panel.query(OperationRow))
            assert len(rows) == 3
            assert original in rows
            assert initial_rows["web_search"] in rows
            read_attempts = {
                row.event.get("attempt_id")
                for row in rows
                if row.event.get("tool_name") == "read_file"
            }
            assert read_attempts == {"a1", "a2"}
            assert not panel.changed_files

    _run(run())


def test_finish_shows_observed_changes_and_assistant_reply(wait_for_tui_widgets) -> None:
    history = ChatHistory()
    app = ManaChatApp(history=history)

    async def run() -> None:
        async with app.run_test(size=(80, 30)) as pilot:
            history.add(UserMessageEvent(content="Adjust the settings.", turn_id="turn"))
            history.add(CodingActivityEvent(turn_id="turn", activity={
                "type": "file.changed", "operation_id": "edit", "status": "success", "path": "src/settings.py",
                "metadata": {"added_lines": 1, "removed_lines": 1,
                             "diff": "--- a/src/settings.py\n+++ b/src/settings.py\n@@ -1 +1 @@\n-old\n+new"},
            }))
            history.add(CodingActivityEvent(turn_id="turn", activity={"type": "turn.finished", "status": "success"}))
            history.add(AssistantMessageEvent(content="Updated `src/settings.py`.", turn_id="turn"))
            await wait_for_tui_widgets(pilot, "ExecutionPanel > .changed-files")
            panel = app.query_one(ExecutionPanel)
            await wait_for_tui_widgets(pilot, "ExecutionPanel OperationRow")
            assert "Changed 1 file" in str(panel.summary.renderable)
            row = panel.query_one(OperationRow)
            assert "Edited src/settings.py (+1 -1)" in row.title
            row.collapsed = False
            await wait_for_tui_widgets(
                pilot, "OperationRow OperationOutput", ready=lambda: bool(row.output.text),
            )
            await wait_for_tui_widgets(pilot, "AssistantReply > Markdown")
            assert "@@ -1 +1 @@\n-old\n+new" in row.output.text
            assert app.query_one(AssistantReply).markdown.display

    _run(run())
