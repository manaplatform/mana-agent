"""Growing composer with registry-backed command completion."""

from __future__ import annotations

from rich.text import Text
from textual import events
from textual.message import Message
from textual.widgets import Static, TextArea

from mana_agent.tui.presentation import glyph


class MessageInput(TextArea):
    MIN_HEIGHT = 1
    MAX_HEIGHT = 8
    NEWLINE_KEYS = frozenset({"shift+enter", "alt+enter", "ctrl+j"})
    MENU_SIZE = 6

    class Submitted(Message):
        def __init__(self, message_input: "MessageInput") -> None:
            super().__init__()
            self.message_input = message_input
            self.value = message_input.text

    def __init__(self, **kwargs: object) -> None:
        super().__init__("", soft_wrap=True, show_line_numbers=False, tab_behavior="focus", **kwargs)
        self._reported_height = self.MIN_HEIGHT
        self.styles.height = self.MIN_HEIGHT
        self.menu: Static | None = None
        self._commands: list[tuple[str, str, tuple[str, ...]]] = []
        self._choices: list[tuple[str, str, tuple[str, ...]]] = []
        self._selected = 0
        self._editing = False

    def get_line(self, line_index: int) -> Text:
        if line_index == 0 and not self.text and not self._editing:
            return Text("Ask Mana anything...", style="dim")
        return super().get_line(line_index)

    def _begin_editing(self) -> None:
        if not self._editing:
            self._editing = True
            self.refresh()

    def on_mouse_down(self, _: events.MouseDown) -> None:
        self._begin_editing()

    def on_blur(self, _: events.Blur) -> None:
        self._editing = False
        self.refresh()

    def set_commands(self, commands: list[tuple[str, str, tuple[str, ...]]]) -> None:
        self._commands = sorted({row[0]: row for row in commands}.values())
        self._refresh_menu(reset=True)

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, value: str) -> None:
        self.load_text(value)
        lines = value.split("\n")
        self.cursor_location = (len(lines) - 1, len(lines[-1]))
        self._report_height()
        self._refresh_menu(reset=True)

    def _refresh_menu(self, *, reset: bool = False) -> None:
        if self.menu is None:
            return
        if not self.text.startswith("/") or "\n" in self.text:
            self._choices = []
        else:
            needle = self.text[1:].casefold()
            first = needle.split(maxsplit=1)[0] if needle.strip() else ""
            known = any(first in (name, *aliases) for name, _, aliases in self._commands)
            if known and any(char.isspace() for char in needle):
                self._choices = []
            else:
                self._choices = [row for row in self._commands if needle in " ".join((row[0], row[1], *row[2])).casefold()]
                self._choices.sort(key=lambda row: (not any(value.startswith(needle) for value in (row[0], *row[2])), row[0]))
        if reset or self._selected >= len(self._choices):
            self._selected = 0
        self.menu.display = bool(self._choices)
        if not self._choices:
            return
        start = self._selected // self.MENU_SIZE * self.MENU_SIZE
        lines = []
        for index, (name, description, _) in enumerate(self._choices[start:start + self.MENU_SIZE], start=start):
            marker = glyph("❯", ">") if index == self._selected else " "
            lines.append(Text(f"{marker} /{name}  {description}", no_wrap=True, overflow="ellipsis",
                              style="bold" if index == self._selected else ""))
        self.menu.update(Text("\n").join(lines))

    async def _on_key(self, event: events.Key) -> None:
        if event.is_printable or event.key in self.NEWLINE_KEYS:
            self._begin_editing()
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
        if self._choices and event.key in {"up", "down", "tab"}:
            event.stop()
            event.prevent_default()
            if event.key == "tab":
                self.value = f"/{self._choices[self._selected][0]} "
            else:
                delta = 1 if event.key == "down" else -1
                self._selected = (self._selected + delta) % len(self._choices)
                self._refresh_menu()
            return
        if event.key == "escape" and self._choices and not self.app.can_interrupt:
            self._choices = []
            if self.menu is not None:
                self.menu.display = False
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)

    def on_text_area_changed(self, _: TextArea.Changed) -> None:
        if self.has_focus and self.text:
            self._begin_editing()
        self._refresh_menu(reset=True)
        self.call_after_refresh(self._report_height)

    def on_resize(self, _: events.Resize) -> None:
        self.call_after_refresh(self._report_height)

    def reset(self) -> None:
        self.clear()
        self._report_height(force=True)
        self._refresh_menu(reset=True)

    def _report_height(self, *, force: bool = False) -> None:
        explicit_lines = self.text.count("\n") + 1
        wrapped_lines = max(explicit_lines, self.wrapped_document.height)
        height = min(self.MAX_HEIGHT, max(self.MIN_HEIGHT, wrapped_lines))
        if force or height != self._reported_height:
            self._reported_height = height
            self.styles.height = height
