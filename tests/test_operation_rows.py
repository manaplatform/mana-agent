from __future__ import annotations

import asyncio
from textual.app import App, ComposeResult
from mana_agent.tui.widgets.activity_boxes import CodingActivityBox


def event(kind, sequence, **changes):
    return {"type": kind, "event_id": f"event-{sequence}", "sequence": sequence,
        "task_id": "task", "agent_id": "worker", "parent_agent_id": "coding",
        "operation_id": "operation", "operation_attempt_id": "invocation", "status": "running",
        "tool_name": "run_command", "metadata": {"command": "python streams.py", "cwd": "/workspace"}, **changes}


def test_streaming_keeps_expanded_operation_row_and_command_identity():
    async def scenario():
        class Application(App):
            def compose(self) -> ComposeResult:
                yield CodingActivityBox(turn_id="test")
        app = Application()
        async with app.run_test() as pilot:
            box = app.query_one(CodingActivityBox)
            box.update_event(event("tool.started", 1))
            await pilot.pause(.15)
            row = next(iter(box.operation_rows.values()))
            row.collapsed = False
            box.update_event(event("tool.stdout", 2, output_preview="progress\n", metadata={
                "stream_channel": "stdout", "stream_offset": 0, "content_length": 9}))
            await pilot.pause(.15)
            assert next(iter(box.operation_rows.values())) is row
            assert not row.collapsed
            assert "progress" in row.output.text
            assert "python streams.py" in row.output.text
            assert "cwd: /workspace" in row.output.text
            assert len(box.operation_rows) == 1
    asyncio.run(scenario())
