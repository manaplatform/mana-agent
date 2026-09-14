"""Tests for TUI attachment bar widget and event handling."""

from __future__ import annotations

from pathlib import Path

from mana_agent.chat.attachments import ChatAttachment
from mana_agent.chat.events import UserMessageEvent
from mana_agent.tui.widgets.attachment_bar import AttachmentBar
from mana_agent.tui.widgets.chat_log import ChatLog


def test_tui_attachment_bar_state_manipulation() -> None:
    bar = AttachmentBar()
    assert bar.attachments == []
    assert bar.count == 0

    att1 = ChatAttachment(
        attachment_id="att-1",
        filename="diagram.png",
        mime_type="image/png",
        size_bytes=1024,
        category="image",
        storage_path="/tmp/diagram.png",
    )
    att2 = ChatAttachment(
        attachment_id="att-2",
        filename="report.pdf",
        mime_type="application/pdf",
        size_bytes=2048,
        category="document",
        storage_path="/tmp/report.pdf",
    )

    # Add attachments
    bar.add_attachment(att1)
    bar.add_attachment(att2)
    assert len(bar.attachments) == 2
    assert bar.count == 2

    # Remove by name
    removed = bar.remove_by_name("diagram.png")
    assert removed is not None
    assert removed.attachment_id == "att-1"
    assert len(bar.attachments) == 1
    assert bar.attachments[0].attachment_id == "att-2"

    # Remove by index
    removed_idx = bar.remove_attachment(0)
    assert removed_idx is not None
    assert removed_idx.attachment_id == "att-2"
    assert len(bar.attachments) == 0

    # Clear all
    bar.add_attachment(att1)
    cleared = bar.clear_attachments()
    assert len(cleared) == 1
    assert len(bar.attachments) == 0


def test_user_message_event_with_attachments() -> None:
    att_dict = {
        "attachment_id": "att-10",
        "filename": "code.py",
        "mime_type": "text/x-python",
        "size_bytes": 128,
        "category": "code",
    }
    # UserMessageEvent normalizes list to tuple
    event = UserMessageEvent(
        content="Check this python script",
        attachments=[att_dict],
    )
    assert event.content == "Check this python script"
    assert isinstance(event.attachments, tuple)
    assert len(event.attachments) == 1
    assert event.attachments[0]["filename"] == "code.py"

    # Message without attachments defaults to empty tuple
    text_only_event = UserMessageEvent(content="Hello")
    assert text_only_event.attachments == ()
