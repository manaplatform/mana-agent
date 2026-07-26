"""Fleet configuration with disabled-by-default compatibility."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mana_agent.config.settings import Settings, mana_home


class FleetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    root: Path = Field(default_factory=lambda: mana_home() / "fleet")
    max_workers_per_run: int = Field(default=4, ge=1, le=64)
    max_concurrent_jobs: int = Field(default=4, ge=1, le=256)
    capability_ttl_seconds: int = Field(default=300, ge=10, le=86_400)
    heartbeat_timeout_seconds: int = Field(default=90, ge=5, le=86_400)
    job_timeout_seconds: int = Field(default=1800, ge=1, le=86_400)
    workspace_max_lifetime_seconds: int = Field(default=3600, ge=60, le=604_800)
    max_log_bytes: int = Field(default=1_048_576, ge=1024)
    max_artifact_bytes: int = Field(default=104_857_600, ge=1024)
    retain_days: int = Field(default=30, ge=1, le=3650)
    auto_repair_enabled: bool = False
    require_trusted_label: bool = True

    @model_validator(mode="after")
    def coherent_timeouts(self) -> "FleetConfig":
        if self.heartbeat_timeout_seconds > self.capability_ttl_seconds:
            raise ValueError("fleet heartbeat timeout cannot exceed capability TTL")
        return self

    @classmethod
    def from_settings(cls, settings: Settings) -> "FleetConfig":
        return cls(
            enabled=getattr(settings, "mana_fleet_enabled", False),
            max_workers_per_run=getattr(settings, "mana_fleet_max_workers_per_run", 4),
            max_concurrent_jobs=getattr(settings, "mana_fleet_max_concurrent_jobs", 4),
            capability_ttl_seconds=getattr(settings, "mana_fleet_capability_ttl_seconds", 300),
            heartbeat_timeout_seconds=getattr(settings, "mana_fleet_heartbeat_timeout_seconds", 90),
            job_timeout_seconds=getattr(settings, "mana_fleet_job_timeout_seconds", 1800),
            workspace_max_lifetime_seconds=getattr(settings, "mana_fleet_workspace_max_lifetime_seconds", 3600),
            max_log_bytes=getattr(settings, "mana_fleet_max_log_bytes", 1_048_576),
            max_artifact_bytes=getattr(settings, "mana_fleet_max_artifact_bytes", 104_857_600),
            retain_days=getattr(settings, "mana_fleet_retain_days", 30),
            auto_repair_enabled=getattr(settings, "mana_fleet_auto_repair_enabled", False),
            require_trusted_label=getattr(settings, "mana_fleet_require_trusted_label", True),
        )
