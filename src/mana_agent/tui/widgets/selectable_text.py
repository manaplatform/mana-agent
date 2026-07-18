"""Read-only text that supports Textual mouse selection and keyboard copying."""

from __future__ import annotations

from textual.binding import Binding
from textual.widgets import TextArea


class SelectableText(TextArea):
    """A content-sized, read-only ``TextArea`` for chat and tool output.

    ``Static`` and ``Markdown`` are passive renderers, so Textual cannot create a
    selection when users drag across their text. ``TextArea`` owns that gesture,
    draws the selection, and exposes the selected source for a normal copy action.
    """

    DEFAULT_CSS = """
    SelectableText {
        width: 1fr;
        height: auto;
        min-height: 1;
        border: none;
        padding: 0;
        background: transparent;
    }
    SelectableText:focus {
        border: none;
    }
    """

    BINDINGS = [
        *TextArea.BINDINGS,
        Binding("ctrl+c", "copy_selection", "Copy selection", show=False, priority=True),
        Binding("ctrl+a", "select_all", "Select all", show=False, priority=True),
    ]

    def __init__(self, text: str = "", **kwargs: object) -> None:
        super().__init__(text, read_only=True, show_line_numbers=False, **kwargs)

    def action_copy_selection(self) -> None:
        """Copy the selected source text using Textual's terminal clipboard API."""
        if self.selected_text:
            self.app.copy_to_clipboard(self.selected_text)
        else:
            self.app.action_quit()
