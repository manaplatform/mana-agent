from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from mana_agent.config.model_catalog import ModelCapability
from mana_agent.media.models import (
    GenerationStatus,
    ImageGenerationRequest,
    MediaArtifact,
    VideoGenerationRequest,
    VoiceGenerationRequest,
)


@dataclass(slots=True)
class ProviderOutput:
    provider_request_id: str
    status: GenerationStatus
    content: tuple[bytes, ...] = ()
    mime_types: tuple[str, ...] = ()
    remote_urls: tuple[str, ...] = ()
    progress: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MediaProvider(Protocol):
    provider_id: str

    def capabilities(self, model: str) -> frozenset[ModelCapability]: ...

    def generate_image(
        self,
        request: ImageGenerationRequest,
        reference_artifacts: tuple[MediaArtifact, ...] = (),
    ) -> ProviderOutput: ...

    def generate_speech(self, request: VoiceGenerationRequest) -> ProviderOutput: ...

    def generate_video(
        self,
        request: VideoGenerationRequest,
        reference_artifacts: tuple[MediaArtifact, ...] = (),
    ) -> ProviderOutput: ...

    def get_generation_status(self, provider_request_id: str) -> ProviderOutput: ...

    def cancel_generation(self, provider_request_id: str) -> ProviderOutput: ...

    def download_result(self, provider_request_id: str) -> ProviderOutput: ...
