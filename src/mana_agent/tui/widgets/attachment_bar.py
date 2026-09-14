"""Attachment bar widget displaying pending attachments above the chat composer."""

from __future__ import annotations

from typing import Any

from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Static

from mana_agent.chat.attachments import ChatAttachment, format_size


class AttachmentChip(Static):
    """Small compact attachment pill/box."""

    DEFAULT_CSS = """
    AttachmentChip {
        width: auto;
        height: 1;
        background: #312e81;
        color: #e0e7ff;
        padding: 0 1;
        margin-right: 1;
        text-style: bold;
    }
    AttachmentChip:hover {
        background: #4338ca;
        color: #ffffff;
    }
    """

    def __init__(self, attachment: ChatAttachment, index: int, **kwargs: Any) -> None:
        self.attachment = attachment
        self.index = index
        label = f"📎 {attachment.filename} ({attachment.format_size()}) ✕"
        super().__init__(label, **kwargs)

    def on_click(self) -> None:
        if self.parent and hasattr(self.parent, "remove_attachment"):
            self.parent.remove_attachment(self.index)


class AttachmentBar(Horizontal):
    """Bar displaying pending attachments as small compact boxes above the composer."""

    DEFAULT_CSS = """
    AttachmentBar {
        height: 1;
        width: 100%;
        background: transparent;
        padding: 0 1;
        margin: 0;
        overflow-x: auto;
        display: none;
    }
    """

    count: reactive[int] = reactive(0)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._attachments: list[ChatAttachment] = []

    @property
    def attachments(self) -> list[ChatAttachment]:
        return list(self._attachments)

    def add_attachment(self, attachment: ChatAttachment) -> None:
        self._attachments.append(attachment)
        self._refresh_view()

    def remove_attachment(self, index: int) -> ChatAttachment | None:
        if 0 <= index < len(self._attachments):
            removed = self._attachments.pop(index)
            self._refresh_view()
            if self.app and hasattr(self.app, "notify"):
                try:
                    self.app.notify(f"Removed attachment: {removed.filename}", severity="information")
                except Exception:
                    pass
            return removed
        return None

    def remove_by_name(self, name: str) -> ChatAttachment | None:
        target = name.strip().lower()
        for idx, att in enumerate(self._attachments):
            if att.filename.lower() == target:
                return self.remove_attachment(idx)
        return None

    def clear_attachments(self) -> list[ChatAttachment]:
        removed = list(self._attachments)
        self._attachments.clear()
        self._refresh_view()
        return removed

    def _refresh_view(self) -> None:
        self.count = len(self._attachments)
        if not self._attachments:
            self.styles.display = "none"
            self.remove_children()
            if self.app and getattr(self.app, "input", None):
                try:
                    self.app.input._report_height(force=True)
                except Exception:
                    pass
            return

        self.styles.display = "block"
        self.remove_children()

        for idx, att in enumerate(self._attachments):
            self.mount(AttachmentChip(att, idx))

        if self.app and getattr(self.app, "input", None):
            try:
                self.app.input._report_height(force=True)
            except Exception:
                pass
