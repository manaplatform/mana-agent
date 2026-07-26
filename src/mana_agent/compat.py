"""Small standard-library compatibility imports for supported Python versions."""

from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - exercised on Python 3.10 CI
    from enum import Enum

    class StrEnum(str, Enum):
        """Python 3.10-compatible subset used by Mana's explicit-value enums."""

        def __str__(self) -> str:
            return str(self.value)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib

__all__ = ["StrEnum", "tomllib"]
