"""Explicit native prompt templates; formatting does not invoke inference."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from mana_agent.llm.messages import Message

@dataclass(frozen=True)
class MessageTemplate:
    parts: tuple[tuple[str, str], ...]

    @classmethod
    def from_messages(cls, parts: list[tuple[str, str]]) -> MessageTemplate:
        return cls(tuple(parts))

    def render(self, values: dict[str, Any]) -> list[Message]:
        return [Message(role={"human": "user", "ai": "assistant"}.get(role, role),
                        content=template.format_map(values)) for role, template in self.parts]
