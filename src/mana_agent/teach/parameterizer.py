"""Conservative input inference from demonstrated values and explanations."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from .models import FlowInput, RecordedEvent, TeachSession


DATE_PATTERN = re.compile(r"\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]20\d{2})\b")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class Parameterizer:
    def infer(
        self, session: TeachSession, events: list[RecordedEvent]
    ) -> tuple[dict[str, FlowInput], dict[str, str]]:
        inputs: dict[str, FlowInput] = {}
        replacements: dict[str, str] = {}
        hint = " ".join(item.text.lower() for item in session.explanations)
        for event in events:
            value = event.data.get("value")
            if not isinstance(value, str) or not value or event.sensitive:
                continue
            if DATE_PATTERN.search(value) and any(word in hint for word in ("date", "week", "month", "changes")):
                inputs.setdefault("date", FlowInput(type="date", description="Date demonstrated in the recording."))
                replacements[value] = "{{ date }}"
            elif EMAIL_PATTERN.fullmatch(value) and any(word in hint for word in ("recipient", "email", "changes")):
                inputs.setdefault("recipient", FlowInput(type="email", description="Message recipient."))
                replacements[value] = "{{ recipient }}"
            elif _looks_like_path(value) and any(word in hint for word in ("file", "path", "folder", "changes")):
                inputs.setdefault("file_path", FlowInput(type="path", description="File selected during the workflow."))
                replacements[value] = "{{ file_path }}"
        return inputs, replacements


def _looks_like_path(value: str) -> bool:
    return value.startswith(("/", "~")) or bool(Path(value).suffix)


def replace_values(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: replace_values(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_values(item, replacements) for item in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, date):
        return value.isoformat()
    return value
