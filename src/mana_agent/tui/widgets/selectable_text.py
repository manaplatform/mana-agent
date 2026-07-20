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
        Binding(
            "ctrl+c", "copy_selection", "Copy selection", show=False, priority=True
        ),
        Binding("ctrl+a", "select_all", "Select all", show=False, priority=True),
    ]

    def __init__(self, text: str = "", **kwargs: object) -> None:
        super().__init__(text, read_only=True, show_line_numbers=False, **kwargs)

    def on_mount(self) -> None:
        """Rewrap dynamically mounted cards after their parent has laid out.

        On Windows a child can be queried before the first paint, so relying on
        ``render_line`` alone leaves the initial one-column document observable
        to callers. ``call_after_refresh`` runs after the mount/layout cycle on
        every supported Textual backend and retains the render-time guard below.
        """
        self.call_after_refresh(self._rewrap_after_layout)

    def _rewrap_after_layout(self) -> None:
        wrap_width = self.wrap_width
        if wrap_width > 0 and self.wrapped_document._width != wrap_width:
            self._rewrap_and_refresh_virtual_size()

    def render_line(self, y: int):  # noqa: ANN201
        """Ensure TextArea wraps only after this card has a real content width.

        Message cards are mounted dynamically.  Textual may create their
        ``TextArea`` document before the containing ``VerticalScroll`` has been
        laid out, which gives the initial document a negative wrap width.  The
        base widget does not necessarily receive a later resize event, so it
        preserves that one-column wrapping.  Rendering is the first point at
        which the final content width is guaranteed to be available.
        """
        wrap_width = self.wrap_width
        if wrap_width > 0 and self.wrapped_document._width != wrap_width:
            self._rewrap_and_refresh_virtual_size()
        return super().render_line(y)

    def action_copy_selection(self) -> None:
        """Copy the selected source text using Textual's terminal clipboard API."""
        if self.selected_text:
            self.app.copy_to_clipboard(self.selected_text)
        else:
            self.app.action_quit()
