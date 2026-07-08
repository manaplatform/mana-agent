from __future__ import annotations

from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import input_dialog, yes_no_dialog

from mana_agent.tui.menu import NonInteractivePromptError, is_interactive


def text_input(title: str, text: str, *, default: str = "") -> str:
    if not is_interactive():
        raise NonInteractivePromptError("Text input requested in a non-TTY session.")
    value = input_dialog(title=title, text=text, default=default).run()
    if value is None:
        raise KeyboardInterrupt
    return str(value)


def secret_input(title: str, text: str) -> str:
    if not is_interactive():
        raise NonInteractivePromptError("Secret input requested in a non-TTY session.")
    _ = title
    return str(prompt(f"{text} ", is_password=True) or "")


def confirm(title: str, text: str, *, default: bool = True) -> bool:
    if not is_interactive():
        raise NonInteractivePromptError("Confirm prompt requested in a non-TTY session.")
    value = yes_no_dialog(title=title, text=text).run()
    if value is None:
        return default
    return bool(value)
