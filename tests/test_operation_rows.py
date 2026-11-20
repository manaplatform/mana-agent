from __future__ import annotations

import asyncio
from textual.app import App, ComposeResult
from mana_agent.tui.widgets.execution_panel import ExecutionPanel
from mana_agent.tui.widgets.operation_row import OperationRow


def event(kind, sequence, **changes):
    return {"type": kind, "event_id": f"event-{sequence}", "sequence": sequence,
        "task_id": "task", "agent_id": "worker", "parent_agent_id": "coding",
        "operation_id": "operation", "operation_attempt_id": "invocation", "status": "running",
        "tool_name": "run_command", "metadata": {"command": "python streams.py", "cwd": "/workspace"}, **changes}


def test_streaming_keeps_expanded_operation_row_and_command_identity():
    async def scenario():
        class Application(App):
            def compose(self) -> ComposeResult:
                yield ExecutionPanel(turn_id="test")
        app = Application()
        async with app.run_test() as pilot:
            box = app.query_one(ExecutionPanel)
            box.update_event(event("tool.started", 1))
            await pilot.pause(.15)
            row = box.query_one(OperationRow)
            row.collapsed = False
            box.update_event(event("tool.stdout", 2, output_preview="progress\n", metadata={
                "stream_channel": "stdout", "stream_offset": 0, "content_length": 9}))
            await pilot.pause(.15)
            assert box.query_one(OperationRow) is row
            assert not row.collapsed
            assert "progress" in row.output.text
            assert "python streams.py" in row.output.text
            assert "cwd: /workspace" in row.output.text
            assert len(box.query(OperationRow)) == 1
    asyncio.run(scenario())


def test_history_keeps_reader_position_and_bounds_mounted_rows() -> None:
    from mana_agent.chat.events import AssistantMessageEvent, CodingActivityEvent, UserMessageEvent
    from mana_agent.chat.history import ChatHistory
    from mana_agent.tui.app import ManaChatApp

    history = ChatHistory()
    app = ManaChatApp(history=history)

    async def scenario():
        async with app.run_test(size=(60, 21)) as pilot:
            log = app.chat_log
            for index in range(25):
                history.add(AssistantMessageEvent(content=f"Previous reply {index}"))
            history.add(UserMessageEvent(content="Show recent commands.", turn_id="test"))
            await pilot.pause(.15)
            assert log.max_scroll_y > 10
            log.scroll_to(y=2, animate=False, immediate=True)
            await pilot.pause()
            original_y = log.scroll_y
            for index in range(100):
                history.add(CodingActivityEvent(turn_id="test", activity={
                    "type": "command.finished", "status": "success", "command": f"command {index}",
                    "operation_id": f"command-{index}", "sequence": index + 1,
                }))
            await pilot.pause(.2)
            panel = log.get_execution_panel("test")
            assert len(panel.query(OperationRow)) == panel.PAGE_SIZE
            assert panel.earlier.display
            assert log.scroll_y == original_y

            app.action_latest_activity()
            await pilot.pause()
            assert log.max_scroll_y - log.scroll_y <= 2
            panel.earlier.press()
            await pilot.pause(.15)
            assert "command 79" in " ".join(row.title for row in panel.query(OperationRow))

    asyncio.run(scenario())


def test_retained_output_uses_scoped_offsets_and_bounds_each_page(monkeypatch) -> None:
    from mana_agent.services import operation_trace

    calls = []

    def read(service, conversation, execution, operation, attempt, *, operation_attempt_id, channel, offset, limit):
        calls.append((conversation, execution, operation, attempt, operation_attempt_id, channel, offset, limit))
        if channel != "stdout":
            return {"content": "", "next_offset": 0, "has_more": False, "trace_complete": True}
        return {"content": "first page" if not offset else "second page",
                "next_offset": offset + 100, "has_more": offset == 0, "trace_complete": True}

    monkeypatch.setattr(operation_trace, "operation_content", read)

    class Host(App):
        def compose(self) -> ComposeResult:
            yield OperationRow(event("command.finished", 1, conversation_id="conversation", execution_id="execution", repository_id="repo",
                                     attempt_id="attempt", status="success"))

    async def scenario():
        app = Host()
        async with app.run_test() as pilot:
            row = app.query_one(OperationRow)
            row.collapsed = False
            await pilot.pause()
            await row.load_retained()
            assert "first page" in row.output.text
            await row.load_retained()
            assert "second page" in row.output.text and "first page" not in row.output.text
            assert calls[-1] == ("conversation", "execution", "operation", "attempt", "invocation", "stdout", 100, 16000)
            assert row.retained.disabled

    asyncio.run(scenario())
