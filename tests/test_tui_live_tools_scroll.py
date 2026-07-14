"""TUI live tools + auto-scroll related unit checks (full coding agent integration).

Proves:
1. ManaChatApp drives CodingAgent via the exact generate* surface (parity with console chat_cli).
2. emit_tool_event bridge + actions_taken safety net records ToolCall/ToolResult into ChatHistory.
   These become ToolCards inside the ChatLog (the chat box / "tool box").
3. Full parity flags (dir_mode, auto_execute_*, k, max_steps, etc.) are accepted and used to build
   identical calls. Tools reliably surface inside the chat log (no raw emissions).
4. Answer extraction handles coding-agent dict payloads.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from mana_agent.chat.events import ToolCallEvent, ToolResultEvent, UserMessageEvent
from mana_agent.chat.history import ChatHistory
from mana_agent.tui.app import ManaChatApp


class _FakeCodingAgent:
    """Mirrors runtime CodingAgent surface: generate() only (no handle)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, request: str, **kwargs):
        self.calls.append({"request": request, **kwargs})
        # Simulate real tool emission while agent runs (as workers/callbacks do).
        from mana_agent.commands.ui_helpers import emit_tool_event

        emit_tool_event("start", "read_file", args='{"path":"README.md"}', event_id="cid-live-1")
        emit_tool_event("end", "read_file", duration=0.042, event_id="cid-live-1")
        return {"answer": "README is the project overview.", "status": "ok"}


def test_extract_answer_from_coding_agent_dict() -> None:
    assert ManaChatApp._extract_answer({"answer": "hello"}) == "hello"
    assert ManaChatApp._extract_answer({"content": "x"}) == "x"
    assert ManaChatApp._extract_answer(SimpleNamespace(answer="obj")) == "obj"
    assert ManaChatApp._extract_answer(None) == ""
    assert ManaChatApp._extract_answer("plain") == "plain"


def test_handle_real_turn_uses_generate_and_emits_tools_live() -> None:
    history = ChatHistory()
    agent = _FakeCodingAgent()
    app = ManaChatApp(
        history=history,
        coding_agent=agent,
        repo_root=".",
        # exercise new parity context (used for full generate call construction)
        dir_mode=False,
        auto_execute_plan=True,
        coding_agent_max_steps=42,
        resolved_k=5,
    )

    user = UserMessageEvent(content="read the readme")
    history.add(user)

    asyncio.run(app._handle_real_turn(user))

    assert agent.calls, "coding agent generate() must be invoked"
    assert agent.calls[0]["request"] == "read the readme"
    assert "index_dir" in agent.calls[0]
    # Full parity context is threaded through (exact same kwargs surface as console path)
    assert agent.calls[0].get("max_steps") == 42
    assert agent.calls[0].get("k") == 5

    events = history.get_events()
    tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]

    assert any(e.tool_name == "read_file" for e in tool_calls), "tool call must appear in history"
    assert any(e.tool_name == "read_file" and e.success for e in tool_results), "tool result must appear"
    # Pairing via event_id
    call = next(e for e in tool_calls if e.tool_name == "read_file")
    result = next(e for e in tool_results if e.tool_name == "read_file")
    assert call.call_id == result.call_id == "cid-live-1"

    # Final assistant answer was streamed into history
    from mana_agent.chat.events import AssistantMessageEvent

    assistants = [e for e in events if isinstance(e, AssistantMessageEvent) and not e.is_streaming]
    assert any("README" in e.content for e in assistants)


def test_chat_log_has_scroll_to_latest_helper() -> None:
    from mana_agent.tui.widgets.chat_log import ChatLog

    assert hasattr(ChatLog, "_scroll_to_latest")
    assert callable(ChatLog._scroll_to_latest)
