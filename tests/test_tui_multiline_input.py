"""Regression coverage for the multiline Textual chat composer."""

from __future__ import annotations

import asyncio

from mana_agent.chat.events import UserMessageEvent
from mana_agent.chat.history import ChatHistory
from mana_agent.tui.app import ManaChatApp
from mana_agent.tui.widgets.message_input import MessageInput


def test_multiline_composer_submits_exact_content_and_resets() -> None:
    history = ChatHistory()
    app = ManaChatApp(history=history)

    async def run() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            composer = app.query_one(MessageInput)
            composer.value = "first line"
            await pilot.pause()
            await pilot.press("shift+enter")
            composer.insert("second line")
            assert composer.text == "first line\nsecond line"

            await pilot.press("enter")
            await pilot.pause()

            submitted = [event for event in history.get_events() if isinstance(event, UserMessageEvent)]
            assert [event.content for event in submitted] == ["first line\nsecond line"]
            assert composer.text == ""
            assert composer.outer_size.height == MessageInput.MIN_HEIGHT

    asyncio.run(run())


def test_multiline_composer_grows_to_cap_and_does_not_submit_newline_shortcuts() -> None:
    history = ChatHistory()
    app = ManaChatApp(history=history)

    async def run() -> None:
        async with app.run_test(size=(40, 24)) as pilot:
            composer = app.query_one(MessageInput)
            composer.value = "\n".join(f"line {number}" for number in range(20))
            await pilot.pause()

            assert composer.outer_size.height == MessageInput.MAX_HEIGHT
            assert app.query_one("#input-bar").outer_size.height == MessageInput.MAX_HEIGHT + 2
            assert not [event for event in history.get_events() if isinstance(event, UserMessageEvent)]

            composer.reset()
            await pilot.pause()
            assert composer.outer_size.height == MessageInput.MIN_HEIGHT
            assert app.query_one("#input-bar").outer_size.height == MessageInput.MIN_HEIGHT + 2

    asyncio.run(run())


def test_empty_message_is_not_submitted_and_existing_clear_shortcut_still_works() -> None:
    history = ChatHistory()
    app = ManaChatApp(history=history)

    async def run() -> None:
        async with app.run_test() as pilot:
            composer = app.query_one(MessageInput)
            composer.value = " \n\t\n"
            await pilot.press("enter")
            await pilot.pause()
            assert not [event for event in history.get_events() if isinstance(event, UserMessageEvent)]

            await pilot.press("ctrl+l")
            await pilot.pause()
            assert app.status_text == "Log cleared (history preserved)"

    asyncio.run(run())
