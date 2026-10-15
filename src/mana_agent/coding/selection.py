"""Validated configuration-level coding backend selection.

Selection happens once, before a coding turn starts.  Runtime failures are
therefore never interpreted as permission to try another backend.
"""

from __future__ import annotations

from typing import Literal, NamedTuple


CodingBackendName = Literal["codex", "external", "internal"]


class CodingBackendConfigurationError(ValueError):
    """Raised when coding backend settings cannot be executed safely."""


class CodingBackendSelection(NamedTuple):
    backend: CodingBackendName
    codex_enabled: bool
    source: Literal["explicit", "migration-default"]


def resolve_coding_backend(settings: object) -> CodingBackendSelection:
    """Resolve and validate the configured backend without probing Codex.

    Supports 'external' (delegated to Codex) and 'internal' (Mana-Agent native).
    Defaults to 'external' for existing configurations.
    """

    configured = str(getattr(settings, "mana_coding_backend", "") or "").strip().lower()
    codex_enabled = bool(getattr(settings, "mana_codex_enabled", True))
    internal_enabled = bool(getattr(settings, "mana_internal_coding_enabled", True))

    if configured == "internal":
        if not internal_enabled:
            raise CodingBackendConfigurationError(
                "Internal coding engine is configured but MANA_INTERNAL_CODING_ENABLED is false."
            )
        return CodingBackendSelection("internal", codex_enabled, "explicit")

    if configured in {"", "external", "codex"}:
        if not codex_enabled:
            raise CodingBackendConfigurationError(
                "External coding engine (Codex) is selected but MANA_CODEX_ENABLED is false. "
                "Enable Codex to execute external coding tasks or configure MANA_CODING_BACKEND='internal'."
            )
        resolved_backend: CodingBackendName = "external" if configured == "external" else "codex"
        return CodingBackendSelection(
            resolved_backend,
            codex_enabled,
            "explicit" if configured else "migration-default",
        )

    raise CodingBackendConfigurationError(
        f"MANA_CODING_BACKEND must be 'internal', 'external', or 'codex' (got '{configured}'). No coding backend was started."
    )


__all__ = [
    "CodingBackendConfigurationError",
    "CodingBackendName",
    "CodingBackendSelection",
    "resolve_coding_backend",
]
