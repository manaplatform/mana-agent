from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from mana_agent.config.provider_registry import qualify_model_id


class ModelCapability(str, Enum):
    TEXT_GENERATION = "text_generation"
    REASONING = "reasoning"
    TOOL_CALLING = "tool_calling"
    STRUCTURED_OUTPUT = "structured_output"
    CODE = "code"
    IMAGE_INPUT = "image_input"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    AUDIO_GENERATION = "audio_generation"
    VIDEO_GENERATION = "video_generation"


class ModelPurpose(str, Enum):
    AGENT = "agent"
    EMBEDDING = "embedding"


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    provider: str
    id: str
    capabilities: frozenset[ModelCapability]
    context_window: int | None = None
    source: str = "discovered"
    available: bool = True
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def qualified_id(self) -> str:
        return qualify_model_id(self.provider, self.id)

    def supports(self, purpose: ModelPurpose) -> bool:
        if purpose is ModelPurpose.EMBEDDING:
            return ModelCapability.EMBEDDING in self.capabilities
        return ModelCapability.TEXT_GENERATION in self.capabilities


# Maintained metadata takes precedence over the isolated provider-name
# normalizer below. Entries are intentionally capability-focused, not a claim
# that every model is available to every account.
_MAINTAINED: dict[str, frozenset[ModelCapability]] = {
    "gpt-4.1": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.REASONING, ModelCapability.TOOL_CALLING, ModelCapability.CODE, ModelCapability.IMAGE_INPUT}),
    "gpt-4.1-mini": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.TOOL_CALLING, ModelCapability.CODE, ModelCapability.IMAGE_INPUT}),
    "gpt-4o": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.TOOL_CALLING, ModelCapability.IMAGE_INPUT}),
    "gpt-4o-mini": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.TOOL_CALLING, ModelCapability.IMAGE_INPUT}),
    "text-embedding-3-small": frozenset({ModelCapability.EMBEDDING}),
    "text-embedding-3-large": frozenset({ModelCapability.EMBEDDING}),
    "nvidia/nv-embedqa-e5-v5": frozenset({ModelCapability.EMBEDDING}),
}

_NON_TEXT_MARKERS: tuple[tuple[ModelCapability, tuple[str, ...]], ...] = (
    (ModelCapability.EMBEDDING, ("embed", "embedding")),
    (ModelCapability.IMAGE_GENERATION, ("dall-e", "image-gen", "image_generation")),
    (ModelCapability.SPEECH_TO_TEXT, ("whisper", "transcri", "speech-to-text", "stt")),
    (ModelCapability.TEXT_TO_SPEECH, ("tts", "text-to-speech")),
    (ModelCapability.VIDEO_GENERATION, ("sora", "video-gen", "video_generation")),
    (ModelCapability.AUDIO_GENERATION, ("audio", "voice", "realtime")),
)


def normalize_capabilities(
    provider: str,
    model_id: str,
    supplied: Iterable[str | ModelCapability] | None = None,
) -> frozenset[ModelCapability]:
    """Normalize model metadata without treating unknown models as agents.

    Provider metadata wins, then maintained metadata. The final name-based
    pass is deliberately isolated and conservative: it recognizes only
    well-known non-text product categories and a small set of provider text
    families. Truly unknown models remain unclassified and require Advanced
    manual entry.
    """
    if supplied:
        parsed: set[ModelCapability] = set()
        for value in supplied:
            try:
                parsed.add(value if isinstance(value, ModelCapability) else ModelCapability(str(value)))
            except ValueError:
                continue
        if parsed:
            return frozenset(parsed)
    model = str(model_id or "").strip()
    if model in _MAINTAINED:
        return _MAINTAINED[model]
    lowered = model.lower()
    for capability, markers in _NON_TEXT_MARKERS:
        if any(marker in lowered for marker in markers):
            return frozenset({capability})
    provider_id = str(provider or "").strip().lower()
    text_family = (
        provider_id == "openai" and lowered.startswith(("gpt-", "o1", "o3", "o4"))
    ) or (
        provider_id == "nvidia" and any(marker in lowered for marker in ("llama", "nemotron", "mistral", "qwen", "deepseek"))
    )
    if text_family:
        return frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.TOOL_CALLING})
    return frozenset()


def descriptors_from_catalog(provider: str, records: Iterable[str | dict[str, Any]], *, source: str = "discovered") -> list[ModelDescriptor]:
    result: list[ModelDescriptor] = []
    for record in records:
        if isinstance(record, str):
            model_id = record
            metadata: dict[str, Any] = {}
        else:
            model_id = str(record.get("id") or "").strip()
            metadata = dict(record)
        if not model_id:
            continue
        capabilities = normalize_capabilities(provider, model_id, metadata.get("capabilities"))
        context_window = metadata.get("context_length") or metadata.get("context_window")
        try:
            context_window = int(context_window) if context_window is not None else None
        except (TypeError, ValueError):
            context_window = None
        result.append(ModelDescriptor(provider=provider, id=model_id, capabilities=capabilities, context_window=context_window, source=source, metadata=metadata))
    # Catalog endpoints can contain duplicate IDs while an upstream changes.
    deduplicated = {item.id: item for item in result}
    return sorted(deduplicated.values(), key=lambda item: item.qualified_id)


def filter_models(models: Iterable[ModelDescriptor], purpose: ModelPurpose) -> list[ModelDescriptor]:
    return [model for model in models if model.supports(purpose)]
