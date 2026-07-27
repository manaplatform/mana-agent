"""Regression coverage for Textual tool-card sizing and passive text widgets."""

from __future__ import annotations

import asyncio

from textual.widgets import Collapsible

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
from mana_agent.tui.widgets.tool_card import ToolCard


def _run(coroutine) -> None:  # noqa: ANN001
    asyncio.run(coroutine)


def _rendered_source(widget: SelectableText) -> str:
    return widget.text


def test_message_widgets_support_textual_mouse_selection_and_copy() -> None:
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
            await pilot.pause()

            chat_log = app.query_one("#chat-log")
            user_message = chat_log.query_one(".user-message", SelectableText)
            assistant_message = chat_log.query_one(".assistant-message", SelectableText)

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


def test_coding_activity_updates_one_turn_scoped_panel_live() -> None:
    history = ChatHistory()
    app = ManaChatApp(history=history)

    async def run() -> None:
        async with app.run_test(size=(60, 24)) as pilot:
            history.add(CodingActivityEvent(
                turn_id="frontend-turn",
                activity={
                    "event_id": "coding-start",
                    "event_type": "tool.call.started",
                    "backend": "codex",
                    "model": "gpt-test",
                    "status": "running",
                    "title": "pytest -q",
                },
            ))
            await pilot.pause()
            panel = app.query_one(ExecutionPanel)
            assert panel.activity_log is not None and "pytest -q" in panel.activity_log.text

            history.add(CodingActivityEvent(
                turn_id="frontend-turn",
                activity={
                    "event_id": "coding-done",
                    "event_type": "tool.call.completed",
                    "backend": "codex",
                    "status": "success",
                    "title": "pytest -q",
                    "duration_ms": 12,
                    "token_usage": {"total_tokens": 25},
                },
            ))
            await pilot.pause()
            assert len(app.query(ExecutionPanel)) == 1
            assert panel.footer is not None and "25 tokens" in str(panel.footer.render())
            history.add(CodingActivityEvent(
                turn_id="frontend-turn",
                activity={
                    "event_id": "turn-done",
                    "event_type": "turn.completed",
                    "backend": "codex",
                    "status": "success",
                    "title": "Coding turn completed",
                },
            ))
            await pilot.pause()
            assert panel.details is not None and panel.details.collapsed
            await pilot.resize_terminal(42, 24)
            await pilot.pause()
            assert panel.size.width > 1

    _run(run())


def test_history_messages_wrap_at_available_card_width_after_replay_append_and_resize() -> (
    None
):
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
            await pilot.pause()
            chat_log = app.query_one("#chat-log")
            user_message = chat_log.query_one(".user-message", SelectableText)
            assistant_message = chat_log.query_one(".assistant-message", SelectableText)

            # Replayed short text has the full message-card width rather than one
            # character per row. Card borders and padding remain part of the box model.
            assert user_message.wrapped_document.height == 1
            assert user_message.wrap_width == user_message.content_size.width - 1
            assert user_message.region.width > user_message.content_size.width
            assert user_message.size.width > 1
            assert (
                assistant_message.wrap_width == assistant_message.content_size.width - 1
            )
            assert assistant_message.wrapped_document.height < len(
                assistant_message.text
            )
            assert "متن فارسی" in assistant_message.text
            assert "https://example.test/docs" in assistant_message.text
            assert "```python" in assistant_message.text

            history.add(AssistantMessageEvent(content="short assistant message"))
            await pilot.pause()
            # A dynamic mount is a two-phase operation in Textual: the first
            # idle cycle lays out the child and posts its Resize event, and the
            # second processes that event and rewraps the TextArea. Windows'
            # Proactor loop does not collapse both phases into one cycle.
            await pilot.pause()
            live_assistant = list(chat_log.query(".assistant-message"))[-1]
            assert live_assistant.wrapped_document.height == 1
            assert live_assistant.wrap_width == live_assistant.content_size.width - 1

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
            assert live_assistant.wrapped_document.height == 1

    _run(run())


def test_tool_card_remeasures_after_toggle_live_updates_and_resize() -> None:
    history = ChatHistory()
    app = ManaChatApp(history=history)
    call = ToolCallEvent(
        tool_name="run_tests", args={"command": "pytest"}, call_id="layout-card"
    )

    async def run() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            history.add(call)
            await pilot.pause()
            card = app.query_one(ToolCard)
            collapsible = card.query_one(Collapsible)
            collapsed_height = card.size.height

            collapsible.collapsed = False
            await pilot.pause()
            expanded_height = card.size.height
            assert expanded_height > collapsed_height

            history.add(
                ToolResultEvent(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    success=True,
                    result="\n".join(
                        f"line {number}: detailed live output" for number in range(40)
                    ),
                    summary="running output",
                )
            )
            await pilot.pause()
            result_body = card.query_one(".tool-result-body", SelectableText)
            assert "line 39: detailed live output" in _rendered_source(result_body)
            assert card.size.height >= expanded_height
            assert app.query_one("#chat-log").virtual_size.height >= card.size.height

            collapsible.collapsed = True
            await pilot.pause()
            assert card.size.height == collapsed_height

            collapsible.collapsed = False
            await pilot.pause()
            assert card.size.height >= expanded_height

            await pilot.resize_terminal(70, 24)
            await pilot.pause()
            assert "line 39: detailed live output" in _rendered_source(result_body)
            assert app.query_one("#chat-log").virtual_size.height >= card.size.height

            history.add(
                ToolResultEvent(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    success=False,
                    error="Traceback (most recent call last):\n  File 'tool.py', line 1\nRuntimeError: failed",
                    summary="failed",
                )
            )
            await pilot.pause()
            assert "RuntimeError: failed" in _rendered_source(result_body)
            assert card.has_class("tool-result-error") is False
            assert result_body.has_class("tool-result-error")

    _run(run())
