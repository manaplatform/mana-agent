"""Validated configuration-level coding backend selection.

Selection happens once, before a coding turn starts.  Runtime failures are
therefore never interpreted as permission to try another backend.
"""

from __future__ import annotations

from typing import Literal, NamedTuple


CodingBackendName = Literal["codex"]


class CodingBackendConfigurationError(ValueError):
    """Raised when coding backend settings cannot be executed safely."""


class CodingBackendSelection(NamedTuple):
    backend: CodingBackendName
    codex_enabled: bool
    source: Literal["explicit", "migration-default"]


def resolve_coding_backend(settings: object) -> CodingBackendSelection:
    """Resolve and validate the configured backend without probing Codex.

    Codex is the single authoritative coding engine: Coding -> Codex.
    No internal coding engine can be selected or invoked.
    """

    configured = str(getattr(settings, "mana_coding_backend", "") or "").strip().lower()
    codex_enabled = bool(getattr(settings, "mana_codex_enabled", True))

    if configured == "internal":
        raise CodingBackendConfigurationError(
            "Internal coding engine has been removed. Codex is the single authoritative "
            "coding engine. No internal coding engine can be selected or invoked."
        )

    if not codex_enabled:
        raise CodingBackendConfigurationError(
            "Codex is the single authoritative coding engine but MANA_CODEX_ENABLED is false. "
            "Enable Codex to execute coding tasks."
        )

    if configured and configured != "codex":
        raise CodingBackendConfigurationError(
            f"MANA_CODING_BACKEND must be 'codex' (got '{configured}'). No coding backend was started."
        )

    return CodingBackendSelection("codex", codex_enabled, "explicit" if configured else "migration-default")


__all__ = [
    "CodingBackendConfigurationError",
    "CodingBackendName",
    "CodingBackendSelection",
    "resolve_coding_backend",
]
