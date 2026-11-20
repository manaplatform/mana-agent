from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MediaType(str, Enum):
    IMAGE = "image"
    VOICE = "voice"
    VIDEO = "video"
    REALTIME = "realtime"
    TRANSCRIPTION = "transcription"


class GenerationStatus(str, Enum):
    QUEUED = "queued"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ImageGenerationRequest(StrictModel):
    prompt: str = Field(min_length=1, max_length=32_000)
    model: str = ""
    size: str = Field(default="1024x1024", pattern=r"^(auto|\d{2,5}x\d{2,5})$")
    count: int = Field(default=1, ge=1, le=4)
    quality: str = Field(default="auto", max_length=32)
    output_format: str = Field(default="png", pattern=r"^(png|jpeg|webp)$")
    background: str | None = Field(default=None, max_length=32)
    aspect_ratio: str = Field(default="", max_length=32)
    resolution: str = Field(default="", max_length=32)
    reference_artifact_ids: tuple[str, ...] = ()
    idempotency_key: str = Field(default="", max_length=200)


class VoiceGenerationRequest(StrictModel):
    text: str = Field(min_length=1, max_length=4096)
    model: str = ""
    voice: str = Field(default="alloy", min_length=1, max_length=160)
    output_format: str = Field(
        default="mp3", pattern=r"^(mp3|opus|aac|flac|wav|pcm)$"
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: str = Field(default="", max_length=4096)
    idempotency_key: str = Field(default="", max_length=200)


class VideoGenerationRequest(StrictModel):
    prompt: str = Field(min_length=1, max_length=32_000)
    model: str = ""
    duration_seconds: int = Field(default=4, ge=1, le=120)
    aspect_ratio: str = Field(default="", max_length=32)
    resolution: str = Field(
        default="720x1280", pattern=r"^(auto|\d{2,5}x\d{2,5})$"
    )
    reference_artifact_ids: tuple[str, ...] = ()
    idempotency_key: str = Field(default="", max_length=200)


class MediaArtifact(StrictModel):
    artifact_id: str = Field(min_length=1)
    local_path: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    duration_seconds: float | None = Field(default=None, ge=0)
    task_id: str = ""
    session_id: str = ""
    media_type: str = ""
    filename: str = ""
    provider: str = ""
    model: str = ""
    created_at: str = Field(default_factory=utc_now_iso)


class GenerationResult(StrictModel):
    generation_id: str = Field(min_length=1)
    media_type: MediaType
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    status: GenerationStatus
    artifacts: tuple[MediaArtifact, ...] = ()
    provider_request_id: str = ""
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    progress: float | None = Field(default=None, ge=0, le=1)
    error_code: str = ""
    error_detail: str = ""
    request: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)

    @property
    def primary_artifact(self) -> MediaArtifact | None:
        return self.artifacts[0] if self.artifacts else None


class MediaOperationDecision(StrictModel):
    operation: Literal["image.generate", "voice.generate", "video.generate", "generation.status", "generation.cancel"]
    prompt: str = Field(default="", max_length=32_000, description="Required for image.generate and video.generate.")
    text: str = Field(default="", max_length=4096, description="Required for voice.generate.")
    generation_id: str = Field(
        default="", max_length=200, description="Required for generation.status and generation.cancel.",
    )
    model: str = Field(default="", max_length=300)
    size: str = Field(default="", max_length=32)
    count: int = Field(default=1, ge=1, le=4)
    quality: str = Field(default="", max_length=32)
    output_format: str = Field(default="", max_length=16)
    background: str | None = Field(default=None, max_length=32)
    voice: str = Field(default="", max_length=160)
    speed: float | None = Field(default=None, ge=0.25, le=4.0)
    instructions: str = Field(default="", max_length=4096)
    duration_seconds: int = Field(default=0, ge=0, le=120)
    aspect_ratio: str = Field(default="", max_length=32)
    resolution: str = Field(default="", max_length=32)
    reference_artifact_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_operation_inputs(self) -> "MediaOperationDecision":
        required_input = {
            "image.generate": "prompt", "video.generate": "prompt", "voice.generate": "text",
            "generation.status": "generation_id", "generation.cancel": "generation_id",
        }[self.operation]
        if not getattr(self, required_input).strip():
            raise PydanticCustomError(
                "media_operation_input", "Provide nonblank {field} for {operation}.",
                {"operation": self.operation, "field": required_input},
            )
        return self

    @field_validator("reference_artifact_ids")
    @classmethod
    def validate_artifact_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            if not value or "/" in value or "\\" in value or ".." in value:
                raise ValueError("reference artifact IDs must be opaque identifiers")
        return values
