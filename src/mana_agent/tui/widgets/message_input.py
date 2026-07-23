"""Multiline message composer used by the chat TUI."""

from __future__ import annotations

import os

from textual import events
from textual.message import Message
from textual.widgets import TextArea


class MessageInput(TextArea):
    """A compact, growing ``TextArea`` with chat-oriented key bindings.

    Terminals do not consistently report Shift+Enter, so Ctrl+J and Alt+Enter
    deliberately provide portable ways to insert a newline.
    """

    MIN_HEIGHT = 3
    MAX_HEIGHT = 8
    NEWLINE_KEYS = frozenset({"shift+enter", "alt+enter", "ctrl+j"})

    class Submitted(Message):
        """Posted when Enter submits the current composer contents."""

        def __init__(self, message_input: "MessageInput") -> None:
            self.message_input = message_input
            super().__init__()

        @property
        def value(self) -> str:
            return self.message_input.text

    class HeightChanged(Message):
        """Posted after content or wrapping changes the desired input height."""

        def __init__(self, message_input: "MessageInput", height: int) -> None:
            self.message_input = message_input
            self.height = height
            super().__init__()

    def __init__(self, **kwargs: object) -> None:
        super().__init__(
            "",
            soft_wrap=True,
            show_line_numbers=False,
            tab_behavior="focus",
            **kwargs,
        )
        self._reported_height = self.MIN_HEIGHT
        self._command_completions: tuple[str, ...] = ()

    def set_command_completions(self, names: list[str] | tuple[str, ...]) -> None:
        self._command_completions = tuple(sorted({f"/{name.lstrip('/')}" for name in names}))

    @property
    def value(self) -> str:
        """Compatibility alias for callers that previously used ``Input.value``."""
        return self.text

    @value.setter
    def value(self, value: str) -> None:
        self.load_text(value)
        lines = value.split("\n")
        self.cursor_location = (len(lines) - 1, len(lines[-1]))
        # ``load_text`` posts ``Changed`` asynchronously. On Windows' Proactor
        # loop that event can lag behind a test or caller's next layout cycle,
        # leaving explicit multiline assignments at the one-line height. The
        # new text is already authoritative for explicit line counting, so
        # report it synchronously; the queued refresh still refines wrapping.
        self._report_height()

    async def _on_key(self, event: events.Key) -> None:
        """Send on Enter and insert newlines only on explicit alternate keys."""
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self))
            return
        if event.key in self.NEWLINE_KEYS:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if event.key == "tab" and self.text.startswith("/") and " " not in self.text:
            matches = [item for item in self._command_completions if item.startswith(self.text)]
            if matches:
                completion = os.path.commonprefix(matches)
                if len(matches) == 1:
                    completion += " "
                self.value = completion
                event.stop()
                event.prevent_default()
                return
        await super()._on_key(event)

    def on_text_area_changed(self, _: TextArea.Changed) -> None:
        self.call_after_refresh(self._report_height)

    def on_resize(self, _: events.Resize) -> None:
        self.call_after_refresh(self._report_height)

    def reset(self) -> None:
        """Clear the composer and restore its compact height."""
        self.clear()
        self._report_height(force=True)

    def _report_height(self, *, force: bool = False) -> None:
        # ``virtual_size.height`` accounts for both explicit newlines and
        # Textual's soft wrapping. TextArea handles scrolling once constrained.
        # Windows' event loop may dispatch ``Changed`` before Textual refreshes
        # the virtual document.  Explicit newlines are already authoritative,
        # so include them rather than briefly retaining the previous one-line
        # virtual height. Soft-wrapped lines are still supplied by Textual.
        explicit_lines = self.text.count("\n") + 1
        wrapped_lines = max(1, explicit_lines, self.virtual_size.height)
        height = min(self.MAX_HEIGHT, max(self.MIN_HEIGHT, wrapped_lines + 2))
        if force or height != self._reported_height:
            self._reported_height = height
            self.styles.height = height
            self.post_message(self.HeightChanged(self, height))
