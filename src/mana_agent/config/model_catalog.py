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
    IMAGE_EDITING = "image_editing"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    AUDIO_GENERATION = "audio_generation"
    VIDEO_GENERATION = "video_generation"
    REALTIME = "realtime"
    AUDIO_INPUT = "audio_input"
    VIDEO_INPUT = "video_input"


class ModelPurpose(str, Enum):
    AGENT = "agent"
    EMBEDDING = "embedding"
    IMAGE = "image"
    IMAGE_EDIT = "image_edit"
    VOICE = "voice"
    VIDEO = "video"
    REALTIME = "realtime"
    TRANSCRIPTION = "transcription"
    MULTIMODAL_INPUT = "multimodal_input"
    AUDIO_INPUT = "audio_input"
    VIDEO_INPUT = "video_input"


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
        if not self.available:
            return False
        if purpose is ModelPurpose.EMBEDDING:
            return ModelCapability.EMBEDDING in self.capabilities
        if purpose is ModelPurpose.IMAGE:
            return ModelCapability.IMAGE_GENERATION in self.capabilities
        if purpose is ModelPurpose.IMAGE_EDIT:
            return ModelCapability.IMAGE_EDITING in self.capabilities
        if purpose is ModelPurpose.VOICE:
            return bool(
                self.capabilities
                & {ModelCapability.TEXT_TO_SPEECH, ModelCapability.AUDIO_GENERATION}
            )
        if purpose is ModelPurpose.VIDEO:
            return ModelCapability.VIDEO_GENERATION in self.capabilities
        if purpose is ModelPurpose.REALTIME:
            return ModelCapability.REALTIME in self.capabilities
        if purpose is ModelPurpose.TRANSCRIPTION:
            return ModelCapability.SPEECH_TO_TEXT in self.capabilities
        if purpose is ModelPurpose.MULTIMODAL_INPUT:
            return ModelCapability.IMAGE_INPUT in self.capabilities
        if purpose is ModelPurpose.AUDIO_INPUT:
            return bool(
                self.capabilities
                & {ModelCapability.AUDIO_INPUT, ModelCapability.SPEECH_TO_TEXT}
            )
        if purpose is ModelPurpose.VIDEO_INPUT:
            return ModelCapability.VIDEO_INPUT in self.capabilities
        if purpose is ModelPurpose.AGENT:
            return (
                ModelCapability.TEXT_GENERATION in self.capabilities
                and ModelCapability.REALTIME not in self.capabilities
            )
        return ModelCapability.TEXT_GENERATION in self.capabilities


def _is_model_available(metadata: dict[str, Any]) -> bool:
    if not metadata:
        return True
    if metadata.get("available") is False:
        return False
    if metadata.get("deprecated") is True:
        return False
    status = str(metadata.get("status") or "").strip().lower()
    if status in {"deprecated", "shutdown", "disabled", "inactive"}:
        return False
    shutdown = metadata.get("shutdown_date") or metadata.get("deprecation_date")
    if shutdown is not None:
        import time
        from datetime import datetime, timezone

        if isinstance(shutdown, (int, float)):
            if shutdown < time.time():
                return False
        elif isinstance(shutdown, datetime):
            dt = shutdown
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt < datetime.now(timezone.utc):
                return False
        elif isinstance(shutdown, str) and shutdown.strip():
            raw_str = shutdown.strip()
            try:
                ts = float(raw_str)
                if ts < time.time():
                    return False
            except ValueError:
                try:
                    iso_str = raw_str[:-1] + "+00:00" if raw_str.endswith(("Z", "z")) else raw_str
                    dt = datetime.fromisoformat(iso_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt < datetime.now(timezone.utc):
                        return False
                except Exception:
                    pass
    return True


def cached_token_limits(provider: str, model_id: str) -> tuple[int, int] | None:
    if not provider or not model_id:
        return None
    from mana_agent.llm.registry import resolve_profile
    profile = resolve_profile(provider, model_id)
    if profile.context_window and profile.max_output_tokens:
        return profile.context_window, profile.max_output_tokens
    return None


def normalize_capabilities(provider: str, model_id: str, supplied=None) -> frozenset[ModelCapability]:
    if supplied is not None:
        result = set()
        for item in supplied:
            try:
                result.add(ModelCapability(item))
            except ValueError:
                continue
        return frozenset(result)
    from mana_agent.llm.registry import resolve_profile
    profile = resolve_profile(provider, model_id)
    return frozenset(_profile_capabilities(profile))


def parse_model_metadata_text(text: str) -> dict[str, Any]:
    import json
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _profile_capabilities(profile) -> set[ModelCapability]:
    result = set()
    mapping = {"output_text": ModelCapability.TEXT_GENERATION, "tool_calling": ModelCapability.TOOL_CALLING,
        "native_structured_output": ModelCapability.STRUCTURED_OUTPUT, "reasoning": ModelCapability.REASONING,
        "input_image": ModelCapability.IMAGE_INPUT, "input_audio": ModelCapability.AUDIO_INPUT,
        "input_video": ModelCapability.VIDEO_INPUT, "image_generation": ModelCapability.IMAGE_GENERATION,
        "image_editing": ModelCapability.IMAGE_EDITING, "speech_to_text": ModelCapability.SPEECH_TO_TEXT,
        "text_to_speech": ModelCapability.TEXT_TO_SPEECH, "output_image": ModelCapability.IMAGE_GENERATION,
        "output_audio": ModelCapability.AUDIO_GENERATION, "output_video": ModelCapability.VIDEO_GENERATION,
        "realtime": ModelCapability.REALTIME, "embedding": ModelCapability.EMBEDDING}
    for name, capability in mapping.items():
        if profile.supports(name):
            result.add(capability)
    return result


def extract_capabilities_from_record(item: dict[str, Any]) -> set[ModelCapability]:
    from mana_agent.llm.capabilities import ModelProfile, metadata_facts
    facts = metadata_facts(item, source="provider")
    profile = ModelProfile(provider="metadata", model_id=str(item.get("id") or "metadata"), facts=facts)
    result = _profile_capabilities(profile)
    for name in item.get("capabilities", []) if isinstance(item.get("capabilities"), list) else []:
        try:
            result.add(ModelCapability(name))
        except ValueError:
            continue
    return result


def descriptors_from_catalog(provider: str, records: Iterable[str | dict[str, Any]], *, source: str = "discovered") -> list[ModelDescriptor]:
    from mana_agent.config.provider_registry import PROVIDERS
    from mana_agent.llm.registry import GLOBAL_CAPABILITY_REGISTRY
    result = []
    provider = PROVIDERS.get(provider).id
    for record in records:
        metadata = dict(record) if isinstance(record, dict) else {"id": record}
        model_id = str(metadata.get("id") or "")
        if not model_id:
            continue
        profile = GLOBAL_CAPABILITY_REGISTRY.get(provider, model_id, metadata.get("_mana_catalog_binding", ""))
        capabilities = _profile_capabilities(profile) if profile else extract_capabilities_from_record(metadata)
        limits = metadata.get("limit") or {}
        result.append(ModelDescriptor(provider=provider, id=model_id, capabilities=frozenset(capabilities),
            context_window=profile.context_window if profile else metadata.get("context_window", metadata.get("context_length", limits.get("context"))),
            max_output_tokens=profile.max_output_tokens if profile else metadata.get("max_output_tokens", limits.get("output")),
            tokenizer=metadata.get("tokenizer"), source=source, available=_is_model_available(metadata), metadata=metadata))
    return sorted(result, key=lambda item: item.id)

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
