"""TUI auto-chat tool event emission / replay."""

from __future__ import annotations

import asyncio
from pathlib import Path

from types import SimpleNamespace

from mana_agent.chat.events import ToolCallEvent, ToolResultEvent, UserMessageEvent
from mana_agent.chat.history import ChatHistory
from mana_agent.chat_commands.models import CommandResult
from mana_agent.gateway.turn_engine import ChatTurnResult, _serialize_tool_traces
from mana_agent.tui.app import ManaChatApp
from mana_agent.tui.widgets.message_input import MessageInput


def test_serialize_tool_traces_from_response_objects() -> None:
    class Trace:
        def to_dict(self):
            return {
                "tool_name": "web_search",
                "args_summary": "query=x",
                "duration_ms": 1.0,
                "status": "ok",
                "output_preview": "hit",
            }

    class Resp:
        trace = [Trace()]

    rows = _serialize_tool_traces(Resp())
    assert rows[0]["tool_name"] == "web_search"


def test_tui_replays_auto_chat_traces_into_history(tmp_path: Path) -> None:
    history = ChatHistory()
    app = ManaChatApp(history=history, repo_root=tmp_path, model="gpt-test")
    result = ChatTurnResult(
        answer="done",
        auto_chat_mode="answer_only",
        used_coding_agent=False,
        payload={
            "route": "auto_chat",
            "trace": [
                {
                    "tool_name": "email_read",
                    "args_summary": "message_ref=x",
                    "status": "ok",
                    "output_preview": "hello",
                    "duration_ms": 5,
                }
            ],
        },
        trace=[
            {
                "tool_name": "email_read",
                "args_summary": "message_ref=x",
                "status": "ok",
                "output_preview": "hello",
                "duration_ms": 5,
            }
        ],
    )
    app._replay_tool_traces_from_result(result, turn_id="turn-auto")
    tools = [e for e in history.get_events() if isinstance(e, (ToolCallEvent, ToolResultEvent))]
    assert any(isinstance(e, ToolCallEvent) and e.tool_name == "email_read" for e in tools)
    assert any(isinstance(e, ToolResultEvent) and e.tool_name == "email_read" and e.success for e in tools)


def test_tool_emit_bridge_writes_history(tmp_path: Path) -> None:
    history = ChatHistory()
    app = ManaChatApp(history=history, repo_root=tmp_path, model="gpt-test")
    app._tool_cid_map = {}
    bridge = app._make_tool_emit_bridge(original_emit=lambda *a, **k: None, turn_id="t1")
    bridge("start", "web_search", args='{"query":"x"}', event_id="e1")
    bridge("end", "web_search", duration=0.01, event_id="e1")
    names = [e.tool_name for e in history.get_events() if isinstance(e, (ToolCallEvent, ToolResultEvent))]
    assert names.count("web_search") == 2


def test_tui_new_conversation_uses_gateway_boundary_and_clears_visible_history(tmp_path: Path) -> None:
    class Gateway:
        config = SimpleNamespace(session_id="session_old")

        def create_session(self, *, frontend: str, session_id: str | None = None) -> str:
            assert frontend == "tui"
            return session_id or "session_old"

        def start_new_conversation(self, session_id: str, *, frontend: str) -> str:
            assert (session_id, frontend) == ("session_old", "tui")
            return "session_new"

    history = ChatHistory()
    history.add(UserMessageEvent(content="Remember one = b."))
    app = ManaChatApp(history=history, repo_root=tmp_path, model="gpt-test", gateway=Gateway())

    new_session = app._start_new_conversation()

    assert new_session == "session_new"
    assert app._gateway_session_id == "session_new"
    events = history.get_events()
    assert events == []


def test_tui_new_command_deletes_active_session_and_clears_mounted_log(tmp_path: Path) -> None:
    class Gateway:
        config = SimpleNamespace(session_id="session_old")

        def __init__(self) -> None:
            self.deleted_session = ""

        def create_session(self, *, frontend: str, session_id: str | None = None) -> str:
            assert frontend == "tui"
            return session_id or "session_old"

        def dispatch_command(
            self, text: str, *, session_id: str, frontend: str
        ) -> CommandResult:
            assert (text, session_id, frontend) == ("/new", "session_old", "tui")
            self.deleted_session = session_id
            return CommandResult(
                status="success",
                message="",
                data={"session_id": "session_new"},
                events=[
                    {"type": "timeline.replace", "messages": []},
                    {"type": "session.activated", "session_id": "session_new"},
                ],
            )

    gateway = Gateway()
    history = ChatHistory()
    history.add(UserMessageEvent(content="old visible history"))
    app = ManaChatApp(
        history=history, repo_root=tmp_path, model="gpt-test", gateway=gateway
    )

    async def run() -> None:
        async with app.run_test() as pilot:
            assert app.chat_log is not None
            await pilot.pause()
            assert len(app.chat_log.children) == 1
            composer = app.query_one(MessageInput)
            composer.value = "/new"
            await pilot.press("enter")
            await pilot.pause()

            assert gateway.deleted_session == "session_old"
            assert app._gateway_session_id == "session_new"
            assert history.get_events() == []
            assert list(app.chat_log.children) == []

    asyncio.run(run())
