from __future__ import annotations

import asyncio
from typing import Any

from textual.app import App, ComposeResult
from textual.widgets import Button, Input, ListView, Static

from mana_agent.tui.session_management import SessionAction, SessionPickerScreen


def _run(coroutine) -> None:
    asyncio.run(coroutine)


class _SessionPickerApp(App[SessionAction | None]):
    def __init__(self, sessions: list[dict[str, Any]]) -> None:
        super().__init__()
        self.sessions = sessions
        self.result_action: SessionAction | None = None

    def on_mount(self) -> None:
        def _handle_result(action: SessionAction | None) -> None:
            self.result_action = action
            self.exit(action)

        self.push_screen(SessionPickerScreen(self.sessions), _handle_result)


def test_session_picker_screen_renders_and_filters() -> None:
    sessions = [
        {
            "session_id": "sess-12345",
            "short_id": "sess-123",
            "title": "Chat Alpha",
            "status": "active",
            "message_count": 3,
            "current": True,
        },
        {
            "session_id": "sess-67890",
            "short_id": "sess-678",
            "title": "Chat Beta",
            "status": "closed",
            "message_count": 5,
            "current": False,
        },
    ]
    app = _SessionPickerApp(sessions)

    async def run() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SessionPickerScreen)
            list_view = screen.query_one("#session-list", ListView)
            assert len(list_view.children) == 2

            # Filter sessions
            search_input = screen.query_one("#session-search", Input)
            search_input.value = "Beta"
            await pilot.pause()
            assert len(list_view.children) == 1

    _run(run())


def test_session_picker_screen_switch_action() -> None:
    sessions = [
        {
            "session_id": "sess-12345",
            "short_id": "sess-123",
            "title": "Chat Alpha",
            "status": "active",
            "message_count": 3,
            "current": False,
        },
    ]
    app = _SessionPickerApp(sessions)

    async def run() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SessionPickerScreen)
            list_view = screen.query_one("#session-list", ListView)
            list_view.index = 0
            await pilot.pause()

            switch_button = screen.query_one("#session-switch", Button)
            switch_button.press()
            await pilot.pause()

    _run(run())
    assert app.result_action == SessionAction(action="switch", session_id="sess-12345")


def test_session_picker_screen_rename_action() -> None:
    sessions = [
        {
            "session_id": "sess-12345",
            "short_id": "sess-123",
            "title": "Chat Alpha",
            "status": "active",
            "message_count": 3,
            "current": False,
        },
    ]
    app = _SessionPickerApp(sessions)

    async def run() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SessionPickerScreen)
            list_view = screen.query_one("#session-list", ListView)
            list_view.index = 0
            await pilot.pause()

            title_input = screen.query_one("#session-title", Input)
            title_input.value = "Updated Title"
            await pilot.pause()

            rename_button = screen.query_one("#session-rename", Button)
            rename_button.press()
            await pilot.pause()

    _run(run())
    assert app.result_action == SessionAction(action="rename", session_id="sess-12345", title="Updated Title")


def test_session_picker_screen_delete_requires_confirmation() -> None:
    sessions = [
        {
            "session_id": "sess-12345",
            "short_id": "sess-123",
            "title": "Chat Alpha",
            "status": "active",
            "message_count": 3,
            "current": False,
        },
    ]
    app = _SessionPickerApp(sessions)

    async def run() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SessionPickerScreen)
            list_view = screen.query_one("#session-list", ListView)
            list_view.index = 0
            await pilot.pause()

            delete_button = screen.query_one("#session-delete", Button)
            delete_button.press()
            await pilot.pause()

            note = screen.query_one("#session-note", Static)
            assert "Press Delete again to confirm" in str(note.renderable)
            assert app.result_action is None

            delete_button.press()
            await pilot.pause()

    _run(run())
    assert app.result_action == SessionAction(action="delete", session_id="sess-12345")


def test_session_picker_screen_archived_cannot_switch() -> None:
    sessions = [
        {
            "session_id": "sess-archived",
            "short_id": "sess-arc",
            "title": "Old Archived Chat",
            "status": "archived",
            "message_count": 10,
            "current": False,
        },
    ]
    app = _SessionPickerApp(sessions)

    async def run() -> None:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SessionPickerScreen)
            list_view = screen.query_one("#session-list", ListView)
            list_view.index = 0
            await pilot.pause()

            switch_button = screen.query_one("#session-switch", Button)
            switch_button.press()
            await pilot.pause()

            note = screen.query_one("#session-note", Static)
            assert "Archived chats cannot be opened" in str(note.renderable)
            assert app.result_action is None

    _run(run())
