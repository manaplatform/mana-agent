"""Model-aware tokenizer resolution with explicit estimation confidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


class Tokenizer(Protocol):
    name: str
    confidence: str

    def count(self, value: Any) -> int: ...


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


@dataclass(frozen=True, slots=True)
class CharacterFallbackTokenizer:
    """Conservative estimator used only when no model tokenizer is registered."""

    characters_per_token: int = 3
    name: str = "character-fallback"
    confidence: str = "low"

    def count(self, value: Any) -> int:
        text = _render(value)
        return 0 if not text else max(1, (len(text) + self.characters_per_token - 1) // self.characters_per_token)


@dataclass(frozen=True, slots=True)
class TiktokenTokenizer:
    encoding_name: str
    confidence: str = "medium"

    @property
    def name(self) -> str:
        return f"tiktoken:{self.encoding_name}"

    def count(self, value: Any) -> int:
        import tiktoken

        encoder = tiktoken.get_encoding(self.encoding_name)
        return len(encoder.encode(_render(value), disallowed_special=()))


class TokenizerRegistry:
    def __init__(self, *, fallback_characters_per_token: int = 3) -> None:
        self._fallback = CharacterFallbackTokenizer(max(1, int(fallback_characters_per_token)))

    def resolve(self, tokenizer: str | None) -> tuple[Tokenizer, tuple[str, ...]]:
        name = str(tokenizer or "").strip()
        if name.startswith("tiktoken:"):
            selected = TiktokenTokenizer(name.split(":", 1)[1])
            try:
                selected.count("")
                return selected, ()
            except (ImportError, KeyError, ValueError):
                return self._fallback, (f"configured tokenizer {name!r} is unavailable",)
        if name:
            return self._fallback, (f"configured tokenizer {name!r} is unsupported",)
        return self._fallback, ("model tokenizer metadata is unavailable",)


__all__ = ["CharacterFallbackTokenizer", "TiktokenTokenizer", "Tokenizer", "TokenizerRegistry"]
