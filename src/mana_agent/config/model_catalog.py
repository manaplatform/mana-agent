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
    IMAGE = "image"
    VOICE = "voice"
    VIDEO = "video"
    MULTIMODAL_INPUT = "multimodal_input"


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    provider: str
    id: str
    capabilities: frozenset[ModelCapability]
    context_window: int | None = None
    max_output_tokens: int | None = None
    tokenizer: str | None = None
    source: str = "discovered"
    available: bool = True
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def qualified_id(self) -> str:
        return qualify_model_id(self.provider, self.id)

    def supports(self, purpose: ModelPurpose) -> bool:
        if purpose is ModelPurpose.EMBEDDING:
            return ModelCapability.EMBEDDING in self.capabilities
        if purpose is ModelPurpose.IMAGE:
            return ModelCapability.IMAGE_GENERATION in self.capabilities
        if purpose is ModelPurpose.VOICE:
            return bool(
                self.capabilities
                & {ModelCapability.TEXT_TO_SPEECH, ModelCapability.AUDIO_GENERATION}
            )
        if purpose is ModelPurpose.VIDEO:
            return ModelCapability.VIDEO_GENERATION in self.capabilities
        if purpose is ModelPurpose.MULTIMODAL_INPUT:
            return ModelCapability.IMAGE_INPUT in self.capabilities
        return ModelCapability.TEXT_GENERATION in self.capabilities


# Known provider context windows used when catalog endpoints omit token limits.
# Values are capability facts for accounting; they do not select models.
# (context_window, max_output_tokens)
_MAINTAINED_TOKEN_LIMITS: dict[str, tuple[int, int]] = {
    "gpt-4.1": (1_047_576, 32_768),
    "gpt-4.1-mini": (1_047_576, 32_768),
    "gpt-4.1-nano": (1_047_576, 32_768),
    "gpt-4o": (128_000, 16_384),
    "gpt-4o-mini": (128_000, 16_384),
    "gpt-5": (400_000, 128_000),
    "gpt-5-mini": (400_000, 128_000),
    "gpt-5-nano": (400_000, 128_000),
    "gpt-5.1": (400_000, 128_000),
    "gpt-5.2": (400_000, 128_000),
    "gpt-5.4": (400_000, 128_000),
    "gpt-5.5": (400_000, 128_000),
    "gpt-5.6-luna": (400_000, 128_000),
    "gpt-5.6-sol": (400_000, 128_000),
    "gpt-5.6-terra": (400_000, 128_000),
    "o3": (200_000, 100_000),
    "o3-mini": (200_000, 100_000),
    "o4-mini": (200_000, 100_000),
}


def maintained_token_limits(provider: str, model_id: str) -> tuple[int, int] | None:
    """Return maintained (context_window, max_output_tokens) when known."""
    model = str(model_id or "").strip()
    if not model:
        return None
    direct = _MAINTAINED_TOKEN_LIMITS.get(model)
    if direct is not None:
        return direct
    # Family prefixes (e.g. gpt-5.4-mini-2026-03-17) inherit parent limits.
    lowered = model.casefold()
    for key, limits in _MAINTAINED_TOKEN_LIMITS.items():
        if lowered.startswith(key.casefold()):
            return limits
    provider_id = str(provider or "").strip().casefold()
    if provider_id == "openai" and lowered.startswith("gpt-5"):
        return (400_000, 128_000)
    if provider_id == "openai" and lowered.startswith("gpt-4.1"):
        return (1_047_576, 32_768)
    if provider_id == "openai" and lowered.startswith("gpt-4o"):
        return (128_000, 16_384)
    return None


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
    "nvidia/llama-3.2-nv-embedqa-1b-v2": frozenset({ModelCapability.EMBEDDING}),
    # Known NVIDIA Build / NIM text models used as agent baselines. Tool
    # calling is OpenAI-compatible on these hosted models; unknown catalog
    # entries remain unclassified until Advanced/manual selection.
    "deepseek-ai/deepseek-v4-flash": frozenset(
        {
            ModelCapability.TEXT_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.CODE,
            ModelCapability.TOOL_CALLING,
        }
    ),
    "deepseek-ai/deepseek-v4-pro": frozenset(
        {
            ModelCapability.TEXT_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.CODE,
            ModelCapability.TOOL_CALLING,
        }
    ),
    "nvidia/nemotron-3-nano-30b-a3b": frozenset(
        {
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE,
            ModelCapability.TOOL_CALLING,
        }
    ),
    "gpt-image-1": frozenset({ModelCapability.IMAGE_GENERATION}),
    "gpt-image-1-mini": frozenset({ModelCapability.IMAGE_GENERATION}),
    "dall-e-2": frozenset({ModelCapability.IMAGE_GENERATION}),
    "dall-e-3": frozenset({ModelCapability.IMAGE_GENERATION}),
    "tts-1": frozenset({ModelCapability.TEXT_TO_SPEECH}),
    "tts-1-hd": frozenset({ModelCapability.TEXT_TO_SPEECH}),
    "gpt-4o-mini-tts": frozenset({ModelCapability.TEXT_TO_SPEECH}),
    "sora-2": frozenset({ModelCapability.VIDEO_GENERATION}),
    "sora-2-pro": frozenset({ModelCapability.VIDEO_GENERATION}),
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
    # Dated / build-suffixed ids (e.g. deepseek-ai/deepseek-v4-flash-0731)
    # inherit the maintained family entry when they share the same prefix.
    lowered = model.lower()
    for key, caps in _MAINTAINED.items():
        key_l = key.lower()
        if lowered.startswith(key_l) and (
            len(lowered) == len(key_l) or lowered[len(key_l)] in "-._/"
        ):
            return caps
    for capability, markers in _NON_TEXT_MARKERS:
        if any(marker in lowered for marker in markers):
            return frozenset({capability})
    provider_id = str(provider or "").strip().lower()
    # Conservative name-based text detection only. Do not invent tool-calling
    # or reasoning capability solely because a model is an LLM; unknown models
    # remain unclassified and stay usable via Advanced/manual entry.
    text_family = (
        provider_id == "openai" and lowered.startswith(("gpt-", "o1", "o3", "o4"))
    ) or (
        provider_id == "nvidia"
        and any(
            marker in lowered
            for marker in (
                "llama",
                "nemotron",
                "mistral",
                "mixtral",
                "qwen",
                "deepseek",
                "kimi",
                "moonshot",
                "gemma",
                "phi-",
                "codellama",
                "yi-",
            )
        )
    )
    if text_family:
        if provider_id == "openai":
            return frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.TOOL_CALLING})
        # NVIDIA DeepSeek V4 family is agent-capable with tools on NIM even when
        # the exact build suffix is not listed in _MAINTAINED.
        if provider_id == "nvidia" and "deepseek" in lowered:
            return frozenset(
                {
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.REASONING,
                    ModelCapability.CODE,
                    ModelCapability.TOOL_CALLING,
                }
            )
        return frozenset({ModelCapability.TEXT_GENERATION})
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
        max_output_tokens = metadata.get("max_output_tokens") or metadata.get("max_completion_tokens")
        try:
            context_window = int(context_window) if context_window is not None else None
        except (TypeError, ValueError):
            context_window = None
        try:
            max_output_tokens = int(max_output_tokens) if max_output_tokens is not None else None
        except (TypeError, ValueError):
            max_output_tokens = None
        if context_window is None or max_output_tokens is None:
            maintained = maintained_token_limits(provider, model_id)
            if maintained is not None:
                context_window = context_window or maintained[0]
                max_output_tokens = max_output_tokens or maintained[1]
        result.append(ModelDescriptor(
            provider=provider,
            id=model_id,
            capabilities=capabilities,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            tokenizer=str(metadata.get("tokenizer") or "") or None,
            source=source,
            metadata=metadata,
        ))
    # Catalog endpoints can contain duplicate IDs while an upstream changes.
    deduplicated = {item.id: item for item in result}
    return sorted(deduplicated.values(), key=lambda item: item.qualified_id)


def filter_models(models: Iterable[ModelDescriptor], purpose: ModelPurpose) -> list[ModelDescriptor]:
    return [model for model in models if model.supports(purpose)]


def search_models(
    models: Iterable[ModelDescriptor],
    *,
    purpose: ModelPurpose,
    query: str = "",
) -> list[ModelDescriptor]:
    """Capability-first filtering with an optional user-visible search term."""
    compatible = filter_models(models, purpose)
    needle = str(query or "").strip().casefold()
    if not needle:
        return compatible
    return [
        model
        for model in compatible
        if needle in model.id.casefold()
        or needle in model.provider.casefold()
        or needle in model.qualified_id.casefold()
    ]
