"""Validated configuration-level coding backend selection.

Selection happens once, before a coding turn starts.  Runtime failures are
therefore never interpreted as permission to try another backend.
"""

from __future__ import annotations

from typing import Literal, NamedTuple


CodingBackendName = Literal["codex", "internal"]


class CodingBackendConfigurationError(ValueError):
    """Raised when coding backend settings cannot be executed safely."""


class CodingBackendSelection(NamedTuple):
    backend: CodingBackendName
    codex_enabled: bool
    source: Literal["explicit", "migration-default"]


def resolve_coding_backend(settings: object) -> CodingBackendSelection:
    """Resolve and validate the configured backend without probing Codex.

    Existing configurations predate ``MANA_CODING_BACKEND``.  Their documented
    migration rule is: select Codex while it is enabled, otherwise select the
    internal runtime.  Once the setting is explicit, contradictory settings are
    rejected rather than silently rewritten.
    """

    configured = str(getattr(settings, "mana_coding_backend", "") or "").strip().lower()
    codex_enabled = bool(getattr(settings, "mana_codex_enabled", True))
    if not configured:
        return CodingBackendSelection(
            "codex" if codex_enabled else "internal",
            codex_enabled,
            "migration-default",
        )
    if configured not in {"codex", "internal"}:
        raise CodingBackendConfigurationError(
            "MANA_CODING_BACKEND must be 'codex' or 'internal'. No coding backend was started."
        )
    if configured == "codex" and not codex_enabled:
        raise CodingBackendConfigurationError(
            "MANA_CODING_BACKEND is 'codex' but MANA_CODEX_ENABLED is false. "
            "Enable Codex or explicitly select the internal backend. No coding backend was started."
        )
    return CodingBackendSelection(configured, codex_enabled, "explicit")  # type: ignore[arg-type]


__all__ = [
    "CodingBackendConfigurationError",
    "CodingBackendName",
    "CodingBackendSelection",
    "resolve_coding_backend",
]
