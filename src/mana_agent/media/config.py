from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mana_agent.config.user_config import load_effective_settings
from mana_agent.media.errors import MediaConfigurationError
from mana_agent.media.models import MediaType


class MediaModalityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    provider: str = ""
    model: str = ""
    credential_ref: str = ""
    base_url: str = ""
    timeout_seconds: int = Field(default=120, ge=1, le=3600)
    max_output_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)
    max_duration_seconds: int | None = Field(default=None, ge=1, le=3600)
    defaults: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", "model", "credential_ref", "base_url")
    @classmethod
    def strip_strings(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator("max_duration_seconds", mode="before")
    @classmethod
    def normalize_unset_duration(cls, value: Any) -> Any:
        if value is None or str(value).strip().casefold() in {"", "none", "null"}:
            return None
        return value

    @model_validator(mode="after")
    def enabled_requires_selection(self) -> "MediaModalityConfig":
        if self.enabled and (not self.provider or not self.model):
            raise ValueError("enabled media configuration requires provider and model")
        return self


class MediaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: MediaModalityConfig = Field(default_factory=MediaModalityConfig)
    voice: MediaModalityConfig = Field(
        default_factory=lambda: MediaModalityConfig(timeout_seconds=120)
    )
    video: MediaModalityConfig = Field(
        default_factory=lambda: MediaModalityConfig(
            timeout_seconds=600,
            max_output_bytes=500 * 1024 * 1024,
            max_duration_seconds=120,
        )
    )
    permissions: dict[str, str] = Field(
        default_factory=lambda: {
            "media.image.generate": "allow",
            "media.voice.generate": "allow",
            "media.video.generate": "allow",
            "media.artifact.write": "allow",
            "media.status.read": "allow",
            "media.generation.cancel": "allow",
        }
    )
    artifact_retention_days: int = Field(default=30, ge=1, le=3650)

    @classmethod
    def load(cls, values: dict[str, Any] | None = None) -> "MediaConfig":
        source = values if values is not None else load_effective_settings(include_env=False)
        raw = source.get("media") if isinstance(source, dict) else {}
        return cls.model_validate(raw if isinstance(raw, dict) else {})

    def modality(self, media_type: MediaType) -> MediaModalityConfig:
        return {
            MediaType.IMAGE: self.image,
            MediaType.VOICE: self.voice,
            MediaType.VIDEO: self.video,
        }[media_type]

    def api_key(self, media_type: MediaType, values: dict[str, Any] | None = None) -> str:
        modality = self.modality(media_type)
        settings = values or load_effective_settings(include_env=False)
        reference = modality.credential_ref
        if not reference:
            from mana_agent.config.provider_registry import provider_credential_env_names

            reference = provider_credential_env_names(str(modality.provider or "openai"))[0]
        return str(settings.get(reference) or "").strip()

    def require(self, media_type: MediaType) -> MediaModalityConfig:
        modality = self.modality(media_type)
        label = media_type.value.capitalize()
        if not modality.enabled:
            raise MediaConfigurationError(
                f"media_{media_type.value}_disabled",
                f"{label} generation is disabled.",
            )
        if not modality.provider or not modality.model:
            raise MediaConfigurationError(
                f"media_{media_type.value}_not_configured",
                f"{label} generation is not configured.",
            )
        return modality
