"""Configuration for Mana-Agent Live mode."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LiveConfig(BaseModel):
    """Settings for the Live realtime voice interaction mode.

    Loaded from the ``[live]`` section of ``~/.mana/config.toml``.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    realtime_model: str = Field(
        default="gpt-4o-realtime-preview",
        description="OpenAI Realtime API model identifier.",
    )
    delegation_model: str | None = Field(
        default=None,
        description="Model identifier for delegated backend tasks (e.g. gpt-4o, gpt-5.6-luna, o3).",
    )
    voice: str = Field(
        default="alloy",
        description="OpenAI voice for spoken responses.",
    )
    vad_mode: Literal["server", "disabled"] = Field(
        default="server",
        description="Voice activity detection mode.",
    )
    vad_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Server VAD sensitivity threshold.",
    )
    turn_detection_silence_ms: int = Field(
        default=500, ge=100, le=5000,
        description="Silence duration (ms) before a user turn ends.",
    )
    input_sample_rate: int = Field(
        default=24000, ge=8000, le=48000,
        description="Audio input sample rate in Hz.",
    )
    output_sample_rate: int = Field(
        default=24000, ge=8000, le=48000,
        description="Audio output sample rate in Hz.",
    )
    audio_format: Literal["pcm16", "g711_ulaw", "g711_alaw"] = Field(
        default="pcm16",
        description="Audio encoding format for the Realtime API.",
    )
    max_response_output_tokens: int | None = Field(
        default=None, ge=1,
        description="Optional token cap for realtime responses.",
    )
    auto_reconnect: bool = Field(
        default=True,
        description="Automatically reconnect on connection loss.",
    )
    reconnect_delay_seconds: float = Field(
        default=2.0, ge=0.5, le=60.0,
        description="Base delay between reconnect attempts.",
    )
    reconnect_max_attempts: int = Field(
        default=5, ge=0, le=50,
        description="Maximum number of consecutive reconnect attempts.",
    )
    input_device: int | None = Field(
        default=None,
        description="Audio input device index (None = system default).",
    )
    output_device: int | None = Field(
        default=None,
        description="Audio output device index (None = system default).",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "model" in data:
                raw_model = data.pop("model")
                if "realtime_model" not in data and raw_model:
                    data["realtime_model"] = raw_model
            if data.get("delegation_model") == "":
                data["delegation_model"] = None
        return data

    @property
    def model(self) -> str:
        """Alias for realtime_model for backwards compatibility."""
        return self.realtime_model

    @classmethod
    def from_mana_settings(cls, settings: Any) -> LiveConfig:
        """Build LiveConfig from Mana settings / user config.

        Reads from the ``[live]`` section of the effective settings.
        """
        from mana_agent.config.user_config import load_effective_settings

        effective = load_effective_settings(include_env=False)
        raw = effective.get("live")
        if isinstance(raw, dict):
            return cls.model_validate(raw)
        return cls()

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> LiveConfig:
        """Build LiveConfig from an explicit dictionary."""
        if not values:
            return cls()
        return cls.model_validate(values)
