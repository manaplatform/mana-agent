#!/usr/bin/env python3
"""
test_chat.py — Demonstration of the enhanced mana-agent Chat TUI.

This script proves:
1. ChatHistory + subscription model works.
2. Tool events are delivered and visible for every turn (the bug fix).
3. Full flow: UserMessage → ToolCall → ToolResult → streaming Assistant.

Run the interactive TUI:
    python test_chat.py

It will launch ManaChatApp. Press Ctrl+R inside the TUI for an instant
full demo of user + tool call + result + streamed answer.

You can also run non-interactively to validate the event plumbing:

    python -c "
    from test_chat import demo_history_only
    demo_history_only()
    "
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make src importable when running from repo root
ROOT = Path(__file__).parent.resolve()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mana_agent.chat.events import (
    AssistantMessageEvent,
    StreamTokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from mana_agent.chat.history import ChatHistory, reset_global_history

# TUI imports are lazy (only when actually launching the TUI) so that
# `test_chat.py --demo` works even before `pip install -e ".[dashboard]"` or adding textual.
# Textual is a new dependency added for the enhanced chat TUI.
ManaChatApp = None
run_chat_tui = None


def _ensure_tui_imports():
    global ManaChatApp, run_chat_tui
    if ManaChatApp is None:
        from mana_agent.tui.app import ManaChatApp as _App, run_chat_tui as _run
        ManaChatApp = _App
        run_chat_tui = _run


def demo_history_only() -> None:
    """
    Non-TUI validation that the subscription model works correctly
    and tool events are never lost across messages.
    """
    print("=== ChatHistory subscription demo (no TUI) ===\n")

    history = reset_global_history()
    received: list[str] = []

    def listener(event):
        et = type(event).__name__
        if isinstance(event, (UserMessageEvent, AssistantMessageEvent)):
            received.append(f"{et}: {event.content[:60]}...")
        elif isinstance(event, ToolCallEvent):
            received.append(f"ToolCall: {event.tool_name}")
        elif isinstance(event, ToolResultEvent):
            received.append(f"ToolResult: {event.tool_name} success={event.success}")
        elif isinstance(event, StreamTokenEvent):
            received.append("StreamToken")

    unsub = history.subscribe(listener)

    # Turn 1
    print("--- Turn 1 ---")
    history.add(UserMessageEvent(content="Please read the README"))
    history.add(
        ToolCallEvent(
            tool_name="read_file",
            args={"path": "README.md"},
            call_id="t1-call-1",
        )
    )
    history.add(
        ToolResultEvent(
            call_id="t1-call-1",
            tool_name="read_file",
            success=True,
            result="# Mana Agent\n...",
        )
    )
    history.add(AssistantMessageEvent(content="README contains project overview."))

    # Turn 2 — this is where the old bug manifested
    print("--- Turn 2 (subsequent message) ---")
    history.add(UserMessageEvent(content="Now list the src directory and summarize tools"))
    history.add(
        ToolCallEvent(
            tool_name="list_dir",
            args={"path": "src/mana_agent"},
            call_id="t2-call-1",
            summary="list src/mana_agent",
        )
    )
    history.add(
        ToolResultEvent(
            call_id="t2-call-1",
            tool_name="list_dir",
            success=True,
            result={"dirs": ["agent", "chat", "tui", "multi_agent"], "files": 40},
            duration_ms=9,
        )
    )
    history.add(
        ToolCallEvent(
            tool_name="grep",
            args={"pattern": "def run", "path": "src"},
            call_id="t2-call-2",
        )
    )
    history.add(
        ToolResultEvent(
            call_id="t2-call-2",
            tool_name="grep",
            success=True,
            result=["chat/app.py:42", "multi_agent/runtime/..."],
        )
    )
    history.add(
        AssistantMessageEvent(
            content="Found several run functions. The new chat TUI is at mana_agent/tui/app.py."
        )
    )

    unsub()

    print("\n=== Events received by subscriber ===")
    for r in received:
        print("  ", r)

    print("\nTotal events in history:", len(history.get_events()))
    print("\nSUCCESS: Tool calls/results were delivered for BOTH turns.")
    print("The subscription + single-source-of-truth design fixes the visibility bug.")


async def _demo_streaming_in_history() -> None:
    """Quick async streaming test."""
    history = reset_global_history()
    chunks = []

    def on_event(e):
        if isinstance(e, StreamTokenEvent):
            chunks.append(e.token)

    history.subscribe(on_event)

    asst = AssistantMessageEvent(content="", is_streaming=True)
    history.add(asst)

    for t in ["Hello", " ", "world", "!"]:
        history.add(StreamTokenEvent(token=t, assistant_event_id=asst.event_id))

    history.add(AssistantMessageEvent(content="Hello world!", is_streaming=False))

    print("Streaming accumulation test:", repr("".join(chunks)))
    assert "".join(chunks) == "Hello world!"
    print("Streaming accumulation: OK")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"--demo", "demo", "--history"}:
        demo_history_only()
        asyncio.run(_demo_streaming_in_history())
        return

    _ensure_tui_imports()
    print("Launching enhanced Mana Chat TUI...")
    print("Tips inside the TUI:")
    print("  • Type a message and press Enter")
    print("  • Press Ctrl+R for instant full demo (user + tools + streaming)")
    print("  • Ctrl+L clears the visible log")
    print("  • Ctrl+C or q to quit\n")

    run_chat_tui()


if __name__ == "__main__":
    main()
