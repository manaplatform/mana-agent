"""Regression coverage for initial chat-message width and Textual reflow."""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from mana_agent.chat.events import UserMessageEvent
from mana_agent.chat.history import ChatHistory
from mana_agent.tui.app import ManaChatApp
from mana_agent.tui.widgets.chat_log import ChatLog
from mana_agent.tui.widgets.selectable_text import SelectableText


def _run(coroutine) -> None:  # noqa: ANN001
    asyncio.run(coroutine)


def _message(app: App, index: int = 0) -> SelectableText:
    return list(app.query(SelectableText).filter(".user-message"))[index]


class _PanelLayoutApp(App):
    """Small layout host that changes the ChatLog parent's available width."""

    CSS = """
    Horizontal { height: 1fr; }
    #chat-log { width: 1fr; }
    #side-panel { display: none; width: 30; }
    """

    def __init__(self, history: ChatHistory) -> None:
        super().__init__()
        self.history = history

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield ChatLog(history=self.history, id="chat-log")
            yield Static("tool details", id="side-panel")


def test_short_message_uses_full_initial_card_width() -> None:
    """A line equal to the card width must not lose a cell to an edit cursor."""
    history = ChatHistory()
    history.add(UserMessageEvent(content="x" * 90))
    app = ManaChatApp(history=history)

    async def run() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            message = _message(app)
            assert message.scrollable_content_region.width == 90
            assert message.wrap_width == 90
            assert message.wrapped_document.height == 1

    _run(run())


def test_long_message_wraps_at_the_initial_container_width() -> None:
    history = ChatHistory()
    history.add(UserMessageEvent(content="x" * 91))
    app = ManaChatApp(history=history)

    async def run() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            message = _message(app)
            assert message.wrap_width == message.scrollable_content_region.width
            assert message.wrapped_document.height == 2
            assert message.size.height == 2

    _run(run())


def test_existing_message_reflows_when_terminal_narrows() -> None:
    history = ChatHistory()
    history.add(UserMessageEvent(content="x" * 90))
    app = ManaChatApp(history=history)

    async def run() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            message = _message(app)
            assert message.wrapped_document.height == 1
            await pilot.resize_terminal(70, 24)
            await pilot.pause()
            assert message.wrap_width == message.scrollable_content_region.width
            assert message.wrapped_document.height == 2
            assert message.size.height == 2

    _run(run())


def test_existing_message_reflows_when_terminal_widens() -> None:
    history = ChatHistory()
    history.add(UserMessageEvent(content="x" * 90))
    app = ManaChatApp(history=history)

    async def run() -> None:
        async with app.run_test(size=(70, 24)) as pilot:
            await pilot.pause()
            message = _message(app)
            assert message.wrapped_document.height == 2
            await pilot.resize_terminal(100, 24)
            await pilot.pause()
            assert message.wrap_width == message.scrollable_content_region.width
            assert message.wrapped_document.height == 1
            assert message.size.height == 1

    _run(run())


def test_new_message_uses_the_current_width_without_a_resize() -> None:
    history = ChatHistory()
    history.add(UserMessageEvent(content="first"))
    app = ManaChatApp(history=history)

    async def run() -> None:
        async with app.run_test(size=(70, 24)) as pilot:
            await pilot.pause()
            history.add(UserMessageEvent(content="x" * 60))
            await pilot.pause()
            message = _message(app, 1)
            assert message.wrap_width == message.scrollable_content_region.width
            assert message.wrapped_document.height == 1
            assert message.size.height == 1

    _run(run())


def test_message_reflows_when_a_surrounding_panel_changes_width() -> None:
    history = ChatHistory()
    history.add(UserMessageEvent(content="x" * 70))
    app = _PanelLayoutApp(history)

    async def run() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            message = _message(app)
            panel = app.query_one("#side-panel", Static)
            initial_width = message.scrollable_content_region.width
            assert message.wrapped_document.height == 1

            panel.styles.display = "block"
            app.refresh(layout=True)
            await pilot.pause()
            assert message.scrollable_content_region.width < initial_width
            assert message.wrap_width == message.scrollable_content_region.width
            assert message.wrapped_document.height == 2

            panel.styles.display = "none"
            app.refresh(layout=True)
            await pilot.pause()
            assert message.scrollable_content_region.width == initial_width
            assert message.wrapped_document.height == 1

    _run(run())


def test_consecutive_messages_do_not_retain_another_widgets_wrap_width() -> None:
    history = ChatHistory()
    history.add(UserMessageEvent(content="x" * 90))
    app = ManaChatApp(history=history)

    async def run() -> None:
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause()
            first = _message(app)
            assert first.wrapped_document.height == 1

            await pilot.resize_terminal(70, 24)
            await pilot.pause()
            history.add(UserMessageEvent(content="x" * 60))
            await pilot.pause()
            second = _message(app, 1)

            assert first.wrap_width == first.scrollable_content_region.width
            assert first.wrapped_document.height == 2
            assert second.wrap_width == second.scrollable_content_region.width
            assert second.wrapped_document.height == 1
            assert first.wrap_width == second.wrap_width

    _run(run())
