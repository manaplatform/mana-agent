from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass

from prompt_toolkit.shortcuts import radiolist_dialog


@dataclass(frozen=True, slots=True)
class MenuOption:
    value: str
    label: str
    aliases: tuple[str, ...] = ()


class NonInteractivePromptError(RuntimeError):
    pass


def is_interactive() -> bool:
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def select_option(
    *,
    title: str,
    text: str,
    options: Iterable[MenuOption],
    default: str | None = None,
    input_func=None,
) -> str:
    items = list(options)
    if not items:
        raise ValueError("At least one menu option is required.")
    if input_func is not None or not is_interactive():
        if input_func is None:
            input_func = input
            prompt = "\n".join([title, text, *(f"{index}. {option.label}" for index, option in enumerate(items, start=1))]) + "\n> "
        else:
            prompt = ""
        raw = str(input_func(prompt) or "").strip().lower()
        for option in items:
            choices = {option.value.lower(), *(alias.lower() for alias in option.aliases)}
            if raw in choices:
                return option.value
        for index, option in enumerate(items, start=1):
            if raw == str(index):
                return option.value
        return raw
    if not is_interactive():
        raise NonInteractivePromptError("Interactive menu requested in a non-TTY session.")
    selected = radiolist_dialog(
        title=title,
        text=text,
        values=[(option.value, option.label) for option in items],
        default=default or items[0].value,
    ).run()
    if selected is None:
        raise KeyboardInterrupt
    return str(selected)
