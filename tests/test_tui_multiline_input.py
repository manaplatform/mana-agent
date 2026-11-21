"""Regression coverage for the multiline Textual chat composer."""

from __future__ import annotations

import asyncio

from mana_agent.chat.events import UserMessageEvent
from mana_agent.chat.history import ChatHistory
from mana_agent.tui.app import ManaChatApp
from mana_agent.tui.widgets.message_input import MessageInput


def test_empty_composer_hides_placeholder_on_click_and_typing() -> None:
    app = ManaChatApp(history=ChatHistory())

    async def run() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            composer = app.query_one(MessageInput)
            hint = "Ask Mana anything..."
            assert hint in composer.render_line(0).text
            assert app.focused is composer

            await pilot.click("#chat-input")
            assert hint not in composer.render_line(0).text
            assert composer.text == ""

            app.set_focus(None)
            await pilot.pause()
            assert hint in composer.render_line(0).text

            composer.focus()
            await pilot.press("m")
            assert composer.text == "m"
            assert hint not in composer.render_line(0).text
            await pilot.press("backspace")
            assert composer.text == ""
            assert hint not in composer.render_line(0).text

    asyncio.run(run())


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
            assert app.query_one("#input-bar").outer_size.height == MessageInput.MAX_HEIGHT + 1
            assert not [event for event in history.get_events() if isinstance(event, UserMessageEvent)]

            composer.reset()
            await pilot.pause()
            assert composer.outer_size.height == MessageInput.MIN_HEIGHT
            assert app.query_one("#input-bar").outer_size.height == MessageInput.MIN_HEIGHT + 1

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


def test_active_composer_steers_and_esc_cancels_the_same_session(tmp_path) -> None:
    from types import SimpleNamespace

    from mana_agent.chat.events import CodingActivityEvent
    from mana_agent.chat_commands.registry import build_default_registry

    class Gateway:
        config = SimpleNamespace(session_id="tty-session")
        command_registry = build_default_registry()

        def __init__(self) -> None:
            self.steered = []
            self.cancelled = []

        def create_session(self, *, frontend, session_id):
            return session_id

        def submit_steering(self, *, session_id, text, scope, request_id, execution_id):
            self.steered.append((session_id, text, scope, execution_id))
            return {"request_id": request_id, "status": "queued"}

        def cancel(self, session_id, *, source):
            self.cancelled.append((session_id, source))
            return True

    gateway = Gateway()
    history = ChatHistory()
    app = ManaChatApp(history=history, repo_root=tmp_path, gateway=gateway)

    async def run() -> None:
        async with app.run_test(size=(58, 24)) as pilot:
            user = UserMessageEvent(content="Check the repository.", turn_id="live-turn")
            history.add(user)
            app._turn_in_progress = True
            app._current_frontend_turn_id = user.turn_id
            app.chat_log.set_active_turn(user.turn_id)
            app._refresh_command_menu()
            composer = app.query_one(MessageInput)
            composer.value = "/res"
            assert not composer.menu.display
            composer.value = "/stat"
            assert "Show the active chat status" in str(composer.menu.renderable)
            await pilot.press("tab")
            assert composer.value == "/status "

            composer.value = "Keep the existing implementation."
            await pilot.press("enter")
            await pilot.pause(.1)
            assert gateway.steered == [("tty-session", "Keep the existing implementation.", "execution", "live-turn")]
            assert len(app.query("#live-status")) == 1
            assert any(event.is_steering for event in history.get_events() if isinstance(event, UserMessageEvent))

            composer.value = "Unsent follow-up"
            await pilot.press("escape")
            await pilot.pause()
            assert gateway.cancelled == [("tty-session", "tui_escape")]
            assert "Interrupt requested" in str(app.chat_log.live.renderable)
            assert composer.value == "Unsent follow-up"
            history.add(CodingActivityEvent(turn_id=user.turn_id, activity={"type": "turn.cancelled", "status": "cancelled"}))
            await pilot.pause(.12)
            panel = app.chat_log.get_execution_panel(user.turn_id)
            assert str(panel.summary.renderable) == "Interrupted"

    asyncio.run(run())
