"""Global and per-connector health configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mana_agent.config.settings import mana_home
from mana_agent.config.user_config import load_user_config

from .models import SyntheticProbeMode


class ConnectorHealthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    probe_interval_seconds: int = Field(default=60, ge=5, le=86_400)
    failure_threshold: int = Field(default=3, ge=1, le=100)
    recovery_enabled: bool = True
    max_recovery_attempts: int = Field(default=8, ge=0, le=100)
    initial_backoff_seconds: float = Field(default=1.0, gt=0, le=3600)
    max_backoff_seconds: float = Field(default=60.0, gt=0, le=3600)
    reset_after_success: bool = True
    circuit_failure_threshold: int = Field(default=5, ge=1, le=100)
    circuit_open_seconds: float = Field(default=30.0, gt=0, le=86_400)
    circuit_half_open_max_probes: int = Field(default=1, ge=1, le=10)
    incident_retention_days: int = Field(default=30, ge=1, le=365)
    probe_log_retention_days: int = Field(default=14, ge=1, le=365)
    ack_timeout_seconds: float = Field(default=120.0, gt=0, le=86_400)
    synthetic_probe_mode: SyntheticProbeMode = SyntheticProbeMode.PASSIVE
    test_channel: str = ""
    active_probe_allowed: bool = False
    rate_limit_probe_multiplier: float = Field(default=4.0, ge=1.0, le=100.0)
    storage_root: str = ""

    @field_validator("synthetic_probe_mode", mode="before")
    @classmethod
    def coerce_mode(cls, value: Any) -> Any:
        if isinstance(value, SyntheticProbeMode):
            return value
        if isinstance(value, str):
            return SyntheticProbeMode(value.strip().lower())
        return value

    def resolved_storage_root(self):
        if self.storage_root.strip():
            from pathlib import Path

            return Path(self.storage_root).expanduser().resolve()
        return mana_home() / "connectors"


class PerConnectorHealthConfig(ConnectorHealthConfig):
    """Per-connector override; inherits global defaults when constructed via merge."""

    connector_id: str = ""


def _section(raw: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = raw
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, dict) else {}


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if value is None:
            continue
        merged[key] = value
    return merged


def load_connector_health_config(
    *,
    connector_id: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> ConnectorHealthConfig | PerConnectorHealthConfig:
    """Load global defaults, then optional per-connector and explicit overrides."""
    user = load_user_config()
    global_raw = _section(user, "connector_health")
    # Nested form: [connectors.<id>.health]
    connector_raw: dict[str, Any] = {}
    if connector_id:
        connector_raw = _section(user, "connectors", connector_id, "health")
        if not connector_raw:
            # Flat MANA-style keys are not used; also accept [connector_health.<id>]
            connector_raw = _section(user, "connector_health", connector_id)
    merged = _merge_dicts(global_raw, connector_raw)
    if overrides:
        merged = _merge_dicts(merged, overrides)
    # Normalize hyphen/underscore keys commonly found in TOML examples
    normalized: dict[str, Any] = {}
    for key, value in merged.items():
        normalized[str(key).replace("-", "_")] = value
    if connector_id:
        return PerConnectorHealthConfig(connector_id=connector_id, **{
            k: v for k, v in normalized.items() if k in PerConnectorHealthConfig.model_fields
        })
    return ConnectorHealthConfig(**{
        k: v for k, v in normalized.items() if k in ConnectorHealthConfig.model_fields
    })
