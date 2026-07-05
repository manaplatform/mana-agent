"""Interactive chat input box built on prompt_toolkit.

Provides a single entry point, :func:`read_chat_input`, that renders a modern,
multiline-capable input box where:

* **Enter** sends the message.
* **Shift+Enter** (kitty / CSI-u capable terminals), **Alt+Enter**, or **Ctrl+J**
  insert a newline so multi-line prompts are easy to compose.

The module degrades gracefully: when prompt_toolkit is unavailable or stdin is
not an interactive TTY (tests, pipes, CI), callers should fall back to plain
``input()``-based reading. Use :func:`prompt_toolkit_available` to decide.
"""

from __future__ import annotations

import sys
from typing import Optional

try:  # prompt_toolkit is an optional-at-runtime dependency
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.styles import Style

    _PTK_IMPORT_OK = True
except Exception:  # pragma: no cover - exercised only when ptk missing
    _PTK_IMPORT_OK = False


# Raw escape sequences that various terminals emit for Shift+Enter. Mapping them
# to Ctrl+J means "Shift+Enter == insert newline" without needing a dedicated
# key constant (prompt_toolkit has none for Shift+Enter).
_SHIFT_ENTER_SEQUENCES = (
    "\x1b[13;2u",   # kitty keyboard protocol (kitty, WezTerm, foot, …)
    "\x1b[27;2;13~",  # xterm modifyOtherKeys
)


def _register_shift_enter_sequences() -> None:
    if not _PTK_IMPORT_OK:
        return
    # These are Shift-specific sequences, so mapping them straight to Ctrl+J
    # (our newline key) is always correct — override any stock mapping.
    for seq in _SHIFT_ENTER_SEQUENCES:
        ANSI_SEQUENCES[seq] = Keys.ControlJ


_register_shift_enter_sequences()


_STYLE = None
_SESSION: "Optional[PromptSession]" = None


def prompt_toolkit_available() -> bool:
    """Return True when the rich prompt_toolkit input box can be used."""
    if not _PTK_IMPORT_OK:
        return False
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        return False


def _build_key_bindings() -> "KeyBindings":
    bindings = KeyBindings()

    @bindings.add("enter")
    def _submit(event) -> None:
        """Enter sends the message (submits the buffer)."""
        event.current_buffer.validate_and_handle()

    @bindings.add("c-j")
    def _newline_ctrl_j(event) -> None:
        """Ctrl+J (and Shift+Enter via the remap above) inserts a newline."""
        event.current_buffer.insert_text("\n")

    @bindings.add("escape", "enter")
    def _newline_alt_enter(event) -> None:
        """Alt+Enter inserts a newline (reliable fallback everywhere)."""
        event.current_buffer.insert_text("\n")

    return bindings


def _build_style() -> "Style":
    return Style.from_dict(
        {
            "prompt": "bold #5fd7ff",
            "continuation": "#5f87af",
            "bottom-toolbar": "#9e9e9e bg:#1c1c1c",
            "bottom-toolbar.key": "bold #5fd7ff bg:#1c1c1c",
        }
    )


def _bottom_toolbar() -> "HTML":
    return HTML(
        "  <b>Enter</b> send"
        "   ·   <b>Shift+Enter</b> / <b>Alt+Enter</b> / <b>Ctrl+J</b> newline"
        "   ·   <b>Ctrl+C</b> cancel   ·   <b>Ctrl+D</b> exit"
    )


def _prompt_continuation(width, line_number, is_soft_wrap):  # noqa: ANN001
    # Aligns wrapped/continuation lines under the first-line glyph.
    return HTML('<continuation>%s</continuation>' % ("·".rjust(max(width, 1))))


def _get_session() -> "PromptSession":
    global _SESSION, _STYLE
    if _SESSION is None:
        _STYLE = _build_style()
        _SESSION = PromptSession(
            history=InMemoryHistory(),
            multiline=True,
            key_bindings=_build_key_bindings(),
            auto_suggest=AutoSuggestFromHistory(),
            style=_STYLE,
            prompt_continuation=_prompt_continuation,
            bottom_toolbar=_bottom_toolbar,
            mouse_support=False,
        )
    return _SESSION


def read_chat_input(message: str = "mana ❯ ") -> str:
    """Read one (possibly multi-line) chat message via prompt_toolkit.

    Raises EOFError on Ctrl+D and KeyboardInterrupt on Ctrl+C, mirroring the
    semantics of the built-in ``input()`` the caller falls back to.
    """
    session = _get_session()
    text = session.prompt(HTML('<prompt>%s</prompt>' % message))
    return (text or "").strip()
