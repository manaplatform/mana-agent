"""Typed, fail-closed Live Canvas configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


WIRE_VERSION = "v0.9"
IMPLEMENTATION_VERSION = "v0.9.1"
MANA_CATALOG_ID = "https://mana-agent.dev/a2ui/catalogs/core/v1/catalog.json"


@dataclass(frozen=True, slots=True)
class CanvasConfig:
    enabled: bool = True
    protocol_versions: tuple[str, ...] = (WIRE_VERSION,)
    default_protocol_version: str = WIRE_VERSION
    allowed_catalogs: tuple[str, ...] = (MANA_CATALOG_ID,)
    accept_inline_catalogs: bool = False
    max_active_surfaces_per_session: int = 16
    max_components_per_surface: int = 250
    max_event_payload_bytes: int = 262_144
    max_component_depth: int = 24
    snapshot_interval: int = 20
    surface_expiry_seconds: int = 86_400
    action_timeout_seconds: int = 900
    validation_retry_limit: int = 1
    max_updates_per_second: int = 20
    websocket_queue_size: int = 256
    allowed_image_schemes: tuple[str, ...] = ("https",)
    allowed_artifact_schemes: tuple[str, ...] = ("https", "artifact")
    developer_diagnostics: bool = False

    @classmethod
    def from_settings(cls, settings: Any) -> "CanvasConfig":
        versions = _csv(
            getattr(settings, "mana_canvas_protocol_versions", WIRE_VERSION)
        )
        catalogs = _csv(
            getattr(settings, "mana_canvas_allowed_catalogs", MANA_CATALOG_ID)
        )
        config = cls(
            enabled=bool(getattr(settings, "mana_canvas_enabled", True)),
            protocol_versions=versions,
            default_protocol_version=str(
                getattr(settings, "mana_canvas_default_protocol_version", WIRE_VERSION)
            ).strip(),
            allowed_catalogs=catalogs,
            accept_inline_catalogs=bool(
                getattr(settings, "mana_canvas_accept_inline_catalogs", False)
            ),
            max_active_surfaces_per_session=int(
                getattr(settings, "mana_canvas_max_active_surfaces", 16)
            ),
            max_components_per_surface=int(
                getattr(settings, "mana_canvas_max_components", 250)
            ),
            max_event_payload_bytes=int(
                getattr(settings, "mana_canvas_max_event_bytes", 262_144)
            ),
            max_component_depth=int(getattr(settings, "mana_canvas_max_depth", 24)),
            snapshot_interval=int(
                getattr(settings, "mana_canvas_snapshot_interval", 20)
            ),
            surface_expiry_seconds=int(
                getattr(settings, "mana_canvas_surface_expiry_seconds", 86_400)
            ),
            action_timeout_seconds=int(
                getattr(settings, "mana_canvas_action_timeout_seconds", 900)
            ),
            validation_retry_limit=int(
                getattr(settings, "mana_canvas_validation_retry_limit", 1)
            ),
            max_updates_per_second=int(
                getattr(settings, "mana_canvas_max_updates_per_second", 20)
            ),
            websocket_queue_size=int(
                getattr(settings, "mana_canvas_websocket_queue_size", 256)
            ),
            allowed_image_schemes=_csv(
                getattr(settings, "mana_canvas_allowed_image_schemes", "https")
            ),
            allowed_artifact_schemes=_csv(
                getattr(
                    settings, "mana_canvas_allowed_artifact_schemes", "https,artifact"
                )
            ),
            developer_diagnostics=bool(
                getattr(settings, "mana_canvas_developer_diagnostics", False)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if (
            not self.protocol_versions
            or self.default_protocol_version not in self.protocol_versions
        ):
            raise ValueError("Canvas default protocol version must be enabled.")
        if any(version != WIRE_VERSION for version in self.protocol_versions):
            raise ValueError(
                f"Unsupported Canvas protocol version; supported: {WIRE_VERSION}."
            )
        if not self.allowed_catalogs:
            raise ValueError("Canvas requires at least one allowlisted catalog.")
        positive = {
            "max_active_surfaces_per_session": self.max_active_surfaces_per_session,
            "max_components_per_surface": self.max_components_per_surface,
            "max_event_payload_bytes": self.max_event_payload_bytes,
            "max_component_depth": self.max_component_depth,
            "snapshot_interval": self.snapshot_interval,
            "surface_expiry_seconds": self.surface_expiry_seconds,
            "action_timeout_seconds": self.action_timeout_seconds,
            "max_updates_per_second": self.max_updates_per_second,
            "websocket_queue_size": self.websocket_queue_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"Canvas {name} must be greater than zero.")
        if self.validation_retry_limit < 0 or self.validation_retry_limit > 3:
            raise ValueError("Canvas validation_retry_limit must be between 0 and 3.")
        if self.max_event_payload_bytes > 2_097_152:
            raise ValueError("Canvas event payload limit cannot exceed 2 MiB.")
        valid_schemes = {"https", "artifact"}
        if not set(self.allowed_image_schemes).issubset({"https"}):
            raise ValueError("Canvas images only support the HTTPS URL scheme.")
        if not set(self.allowed_artifact_schemes).issubset(valid_schemes):
            raise ValueError(
                "Canvas artifact URL policy contains an unsupported scheme."
            )


def _csv(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item).strip() for item in value or () if str(item).strip())
