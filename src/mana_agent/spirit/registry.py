from __future__ import annotations

from typing import Any, Mapping

from mana_agent.spirit.defaults import DEFAULT_MANA_SPIRIT
from mana_agent.spirit.errors import SpiritResolutionError
from mana_agent.spirit.schema import (
    DEFAULT_SPIRIT_ID,
    DEFAULT_SPIRIT_VERSION,
    Spirit,
    SpiritRef,
    SpiritSettings,
)


class SpiritRegistry:
    """Versioned built-in Spirits. Unknown identifiers fail closed."""

    def __init__(self, spirits: tuple[Spirit, ...] | None = None) -> None:
        catalog = spirits if spirits is not None else (DEFAULT_MANA_SPIRIT,)
        self._spirits: dict[tuple[str, int], Spirit] = {}
        for spirit in catalog:
            self.register(spirit)

    def register(self, spirit: Spirit) -> None:
        self._spirits[(spirit.id, spirit.version)] = spirit

    def get(self, spirit_id: str, version: int) -> Spirit:
        key = (str(spirit_id or "").strip().lower(), int(version))
        spirit = self._spirits.get(key)
        if spirit is None:
            raise SpiritResolutionError(
                "Spirit resolution failed: "
                f"unknown spirit {key[0]!r} version {key[1]}. "
                "No fallback spirit was selected.",
                spirit_id=key[0],
                spirit_version=key[1],
            )
        return spirit

    def default(self) -> Spirit:
        return self.get(DEFAULT_SPIRIT_ID, DEFAULT_SPIRIT_VERSION)


registry = SpiritRegistry()


def default_mana_spirit() -> Spirit:
    return registry.default()


def default_spirit_ref() -> SpiritRef:
    return default_mana_spirit().ref()


def spirit_settings_from(settings: Any | None) -> SpiritSettings:
    if settings is None:
        return SpiritSettings()
    if isinstance(settings, SpiritSettings):
        return settings
    if isinstance(settings, Mapping):
        if "spirit" in settings:
            candidate = settings.get("spirit")
        elif "id" in settings or "version" in settings:
            candidate = settings
        else:
            candidate = None
    else:
        candidate = getattr(settings, "spirit", None)
    if isinstance(candidate, str) and candidate.strip():
        return SpiritSettings(id=candidate.strip())
    if isinstance(candidate, Mapping) and candidate:
        return SpiritSettings.model_validate(candidate)
    return SpiritSettings()


def resolve_spirit(
    *,
    spirit_id: str | None = None,
    spirit_version: int | None = None,
    settings: Any | None = None,
) -> Spirit:
    selected_id = str(spirit_id or "").strip().lower()
    selected_version = int(spirit_version) if spirit_version else 0
    if not selected_id:
        configured = spirit_settings_from(settings)
        selected_id = configured.id
        selected_version = selected_version or configured.version
    if not selected_id:
        return default_mana_spirit()
    return registry.get(selected_id, selected_version or DEFAULT_SPIRIT_VERSION)


def resolve_configured_spirit() -> Spirit:
    from mana_agent.config.settings import Settings

    return resolve_spirit(settings=Settings())
