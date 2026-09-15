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


# Known provider context windows used when catalog endpoints omit token limits.
# Values are capability facts for accounting; they do not select models.
# (context_window, max_output_tokens)
_MAINTAINED_TOKEN_LIMITS: dict[str, tuple[int, int]] = {
    "gpt-4.1": (1_047_576, 32_768),
    "gpt-4.1-mini": (1_047_576, 32_768),
    "gpt-4.1-nano": (1_047_576, 32_768),
    "gpt-4o": (128_000, 16_384),
    "gpt-4o-mini": (128_000, 16_384),
    "gpt-4o-realtime-preview": (128_000, 4_096),
    "gpt-live": (128_000, 4_096),
    "gpt-live-1": (128_000, 4_096),
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
    "gpt-6": (1_050_000, 128_000),
    "gpt-6-astra": (1_050_000, 128_000),
    "astra": (1_050_000, 128_000),
    "openai/gpt-6": (1_050_000, 128_000),
    "openai/gpt-6-astra": (1_050_000, 128_000),
    "openai/astra": (1_050_000, 128_000),
    "o3": (200_000, 100_000),
    "o3-mini": (200_000, 100_000),
    "o4-mini": (200_000, 100_000),
    # x-AI Grok 4.6 (500k context window)
    "x-ai/grok-4.6": (500_000, 65_536),
    # NVIDIA NIM / integrate.api DeepSeek V4 (1M context; max_tokens soft cap 65_536).
    "deepseek-ai/deepseek-v4-flash": (1_000_000, 65_536),
    "deepseek-ai/deepseek-v4-pro": (1_000_000, 65_536),
    # OpenRouter multi-tenant catalog defaults
    "deepseek/deepseek-v4-flash": (1_000_000, 65_536),
    "deepseek/deepseek-r1": (163_840, 65_536),
    "google/gemini-2.5-pro": (1_000_000, 16_384),
    "google/gemini-2.5-flash": (1_000_000, 16_384),
    "qwen/qwen-2.5-coder-32b-instruct": (128_000, 32_768),
    "mistralai/mistral-large-2411": (128_000, 32_768),
    "meta-llama/llama-3.3-70b-instruct": (128_000, 16_384),
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
    if "astra" in lowered:
        return (1_050_000, 128_000)
    if provider_id == "openai" and (lowered.startswith("gpt-6") or "astra" in lowered):
        return (1_050_000, 128_000)
    if "grok" in lowered or "x-ai" in lowered:
        return (500_000, 65_536)
    if "claude" in lowered or provider_id == "anthropic":
        return (200_000, 16_384)
    if "gemini" in lowered or provider_id == "google":
        return (1_000_000, 16_384)
    if provider_id == "openai" and lowered.startswith("gpt-5"):
        return (400_000, 128_000)
    if provider_id == "openai" and lowered.startswith("gpt-4.1"):
        return (1_047_576, 32_768)
    if provider_id == "openai" and lowered.startswith("gpt-4o"):
        return (128_000, 16_384)
    if "deepseek" in lowered:
        return (1_000_000, 65_536)
    if "qwen" in lowered:
        return (128_000, 32_768)
    if "mistral" in lowered or "mixtral" in lowered:
        return (128_000, 32_768)
    if "llama" in lowered:
        return (128_000, 16_384)
    if "command" in lowered or "cohere" in lowered:
        return (128_000, 8_192)
    return None


# Maintained metadata takes precedence over the isolated provider-name
# normalizer below. Entries are intentionally capability-focused, not a claim
# that every model is available to every account.
_MAINTAINED: dict[str, frozenset[ModelCapability]] = {
    "gpt-4.1": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.REASONING, ModelCapability.TOOL_CALLING, ModelCapability.CODE, ModelCapability.IMAGE_INPUT}),
    "gpt-4.1-mini": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.TOOL_CALLING, ModelCapability.CODE, ModelCapability.IMAGE_INPUT}),
    "gpt-4o": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.TOOL_CALLING, ModelCapability.IMAGE_INPUT}),
    "gpt-4o-mini": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.TOOL_CALLING, ModelCapability.IMAGE_INPUT}),
    "chatgpt-4o-latest": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.TOOL_CALLING, ModelCapability.IMAGE_INPUT}),
    "gpt-4-turbo": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.TOOL_CALLING, ModelCapability.IMAGE_INPUT}),
    "gpt-4-vision-preview": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.TOOL_CALLING, ModelCapability.IMAGE_INPUT}),
    "gpt-4.5": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.REASONING, ModelCapability.TOOL_CALLING, ModelCapability.CODE, ModelCapability.IMAGE_INPUT}),
    "gpt-4.5-preview": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.REASONING, ModelCapability.TOOL_CALLING, ModelCapability.CODE, ModelCapability.IMAGE_INPUT}),
    "gpt-5": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.REASONING, ModelCapability.TOOL_CALLING, ModelCapability.CODE, ModelCapability.IMAGE_INPUT}),
    "gpt-5-mini": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.TOOL_CALLING, ModelCapability.CODE, ModelCapability.IMAGE_INPUT}),
    "o1": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.REASONING, ModelCapability.TOOL_CALLING, ModelCapability.CODE, ModelCapability.IMAGE_INPUT}),
    "o1-preview": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.REASONING, ModelCapability.TOOL_CALLING, ModelCapability.CODE, ModelCapability.IMAGE_INPUT}),
    "o3": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.REASONING, ModelCapability.TOOL_CALLING, ModelCapability.CODE, ModelCapability.IMAGE_INPUT}),
    "gpt-6": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.REASONING, ModelCapability.TOOL_CALLING, ModelCapability.STRUCTURED_OUTPUT, ModelCapability.CODE, ModelCapability.IMAGE_INPUT}),
    "gpt-6-astra": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.REASONING, ModelCapability.TOOL_CALLING, ModelCapability.STRUCTURED_OUTPUT, ModelCapability.CODE, ModelCapability.IMAGE_INPUT}),
    "astra": frozenset({ModelCapability.TEXT_GENERATION, ModelCapability.REASONING, ModelCapability.TOOL_CALLING, ModelCapability.STRUCTURED_OUTPUT, ModelCapability.CODE, ModelCapability.IMAGE_INPUT}),
    "text-embedding-3-small": frozenset({ModelCapability.EMBEDDING}),
    "text-embedding-3-large": frozenset({ModelCapability.EMBEDDING}),
    "nvidia/nv-embedqa-e5-v5": frozenset({ModelCapability.EMBEDDING}),
    "nvidia/llama-3.2-nv-embedqa-1b-v2": frozenset({ModelCapability.EMBEDDING}),
    "x-ai/grok-4.6": frozenset(
        {
            ModelCapability.TEXT_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.TOOL_CALLING,
            ModelCapability.CODE,
            ModelCapability.STRUCTURED_OUTPUT,
        }
    ),
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
    "deepseek/deepseek-v4-flash": frozenset(
        {
            ModelCapability.TEXT_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.CODE,
            ModelCapability.TOOL_CALLING,
        }
    ),
    "deepseek/deepseek-r1": frozenset(
        {
            ModelCapability.TEXT_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.CODE,
            ModelCapability.TOOL_CALLING,
        }
    ),
    "google/gemini-2.5-pro": frozenset(
        {
            ModelCapability.TEXT_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.CODE,
            ModelCapability.TOOL_CALLING,
            ModelCapability.IMAGE_INPUT,
        }
    ),
    "google/gemini-2.5-flash": frozenset(
        {
            ModelCapability.TEXT_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.CODE,
            ModelCapability.TOOL_CALLING,
            ModelCapability.IMAGE_INPUT,
        }
    ),
    "qwen/qwen-2.5-coder-32b-instruct": frozenset(
        {
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE,
            ModelCapability.TOOL_CALLING,
        }
    ),
    "mistralai/mistral-large-2411": frozenset(
        {
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE,
            ModelCapability.TOOL_CALLING,
        }
    ),
    "meta-llama/llama-3.3-70b-instruct": frozenset(
        {
            ModelCapability.TEXT_GENERATION,
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
    "gpt-image-1": frozenset({ModelCapability.IMAGE_GENERATION, ModelCapability.IMAGE_EDITING}),
    "gpt-image-1-mini": frozenset({ModelCapability.IMAGE_GENERATION, ModelCapability.IMAGE_EDITING}),
    "dall-e-2": frozenset({ModelCapability.IMAGE_GENERATION, ModelCapability.IMAGE_EDITING}),
    "dall-e-3": frozenset({ModelCapability.IMAGE_GENERATION}),
    "tts-1": frozenset({ModelCapability.TEXT_TO_SPEECH, ModelCapability.AUDIO_GENERATION}),
    "tts-1-hd": frozenset({ModelCapability.TEXT_TO_SPEECH, ModelCapability.AUDIO_GENERATION}),
    "gpt-4o-mini-tts": frozenset({ModelCapability.TEXT_TO_SPEECH, ModelCapability.AUDIO_GENERATION}),
    "sora-2": frozenset({ModelCapability.VIDEO_GENERATION}),
    "sora-2-pro": frozenset({ModelCapability.VIDEO_GENERATION}),
    "gpt-4o-realtime-preview": frozenset({ModelCapability.REALTIME, ModelCapability.TEXT_GENERATION, ModelCapability.TOOL_CALLING}),
    "gpt-live": frozenset({ModelCapability.REALTIME, ModelCapability.TEXT_GENERATION, ModelCapability.TOOL_CALLING}),
    "gpt-live-1": frozenset({ModelCapability.REALTIME, ModelCapability.TEXT_GENERATION, ModelCapability.TOOL_CALLING}),
}

_NON_TEXT_MARKERS: tuple[tuple[ModelCapability, tuple[str, ...]], ...] = (
    (ModelCapability.EMBEDDING, ("embed", "embedding")),
    (ModelCapability.IMAGE_GENERATION, ("dall-e", "image-gen", "image_generation", "gpt-image-")),
    (ModelCapability.SPEECH_TO_TEXT, ("whisper", "transcri", "speech-to-text", "stt")),
    (ModelCapability.TEXT_TO_SPEECH, ("tts", "text-to-speech")),
    (ModelCapability.VIDEO_GENERATION, ("sora", "video-gen", "video_generation")),
    (ModelCapability.AUDIO_GENERATION, ("audio", "voice")),
)


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
    lowered = model.lower()
    provider_id = str(provider or "").strip().lower()

    if lowered.startswith("gpt-image-"):
        return frozenset({ModelCapability.IMAGE_GENERATION, ModelCapability.IMAGE_EDITING})
    if lowered == "dall-e-2":
        return frozenset({ModelCapability.IMAGE_GENERATION, ModelCapability.IMAGE_EDITING})
    if lowered == "dall-e-3":
        return frozenset({ModelCapability.IMAGE_GENERATION})
    if "sora" in lowered or (provider_id in {"openai", "openrouter"} and "video" in lowered):
        return frozenset({ModelCapability.VIDEO_GENERATION})
    if (
        "realtime" in lowered
        or "gpt-live" in lowered
        or (provider_id == "openai" and ("-live" in lowered or lowered.startswith("gpt-live")))
    ):
        return frozenset({ModelCapability.REALTIME, ModelCapability.TEXT_GENERATION, ModelCapability.TOOL_CALLING})

    if model in _MAINTAINED:
        return _MAINTAINED[model]
    # Dated / build-suffixed ids (e.g. deepseek-ai/deepseek-v4-flash-0731)
    # inherit the maintained family entry when they share the same prefix.
    for key, caps in _MAINTAINED.items():
        key_l = key.lower()
        if lowered.startswith(key_l) and (
            len(lowered) == len(key_l) or lowered[len(key_l)] in "-._/"
        ):
            return caps
    for capability, markers in _NON_TEXT_MARKERS:
        if any(marker in lowered for marker in markers):
            if capability is ModelCapability.IMAGE_GENERATION and "gpt-image-" in lowered:
                return frozenset({ModelCapability.IMAGE_GENERATION, ModelCapability.IMAGE_EDITING})
            if capability is ModelCapability.TEXT_TO_SPEECH:
                return frozenset({ModelCapability.TEXT_TO_SPEECH, ModelCapability.AUDIO_GENERATION})
            return frozenset({capability})
    if provider_id == "openrouter":
        if "grok" in lowered:
            return frozenset(
                {
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.REASONING,
                    ModelCapability.CODE,
                    ModelCapability.TOOL_CALLING,
                }
            )
        if any(marker in lowered for marker in ("claude", "gpt-", "o1", "o3", "o4", "gemini", "deepseek", "qwen", "mistral", "llama")):
            caps_set = {
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CODE,
                ModelCapability.TOOL_CALLING,
            }
            if any(
                m in lowered
                for m in (
                    "4o",
                    "4.1",
                    "4.5",
                    "vision",
                    "claude-3",
                    "gemini-1.5",
                    "gemini-2",
                    "gpt-4-turbo",
                    "gpt-5",
                    "o1",
                    "o3",
                    "o4",
                )
            ):
                caps_set.add(ModelCapability.IMAGE_INPUT)
            return frozenset(caps_set)
    # Conservative name-based text detection only. Do not invent tool-calling
    # or reasoning capability solely because a model is an LLM; unknown models
    # remain unclassified and stay usable via Advanced/manual entry.
    text_family = (
        provider_id == "openai" and lowered.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-"))
    ) or (
        provider_id == "groq" and ("llama" in lowered or "mixtral" in lowered or "gemma" in lowered)
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
            base = {ModelCapability.TEXT_GENERATION, ModelCapability.TOOL_CALLING}
            if any(
                marker in lowered
                for marker in ("4o", "4.1", "4.5", "gpt-4-turbo", "vision", "gpt-5", "gpt-6", "o1", "o3", "o4")
            ):
                base.add(ModelCapability.IMAGE_INPUT)
            return frozenset(base)
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


def parse_model_metadata_text(text: str) -> dict[str, Any]:
    """Parse text-formatted model documentation, model cards, or platform tables into structured metadata."""
    if not text or not isinstance(text, str):
        return {}

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return {}

    result: dict[str, Any] = {
        "modalities": {},
        "endpoints": [],
        "features": {},
        "tools": {},
    }

    current_section: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        line_lower = line.lower().rstrip(":")

        if line_lower == "modalities":
            current_section = "modalities"
            i += 1
            continue
        elif line_lower == "endpoints":
            current_section = "endpoints"
            i += 1
            continue
        elif line_lower == "features":
            current_section = "features"
            i += 1
            continue
        elif line_lower == "tools":
            current_section = "tools"
            i += 1
            continue
        elif current_section is None:
            for section in ("modalities", "endpoints", "features", "tools"):
                if line_lower.startswith(section):
                    current_section = section
                    break
            i += 1
            continue

        if current_section == "modalities":
            if line_lower in {"endpoints", "features", "tools"}:
                current_section = line_lower
                i += 1
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                result["modalities"][key.strip()] = val.strip()
                i += 1
            elif i + 1 < len(lines) and lines[i + 1].lower() in {
                "input and output",
                "input only",
                "output only",
                "not supported",
                "supported",
                "input",
                "output",
                "none",
            }:
                result["modalities"][line] = lines[i + 1]
                i += 2
            else:
                result["modalities"][line] = "supported"
                i += 1

        elif current_section == "endpoints":
            if line_lower in {"features", "tools", "modalities"}:
                current_section = line_lower
                i += 1
                continue
            if line.startswith(("v1/", "/", "http://", "https://")):
                result["endpoints"].append(line)
                i += 1
            elif i + 1 < len(lines) and lines[i + 1].startswith(("v1/", "/", "http://", "https://")):
                result["endpoints"].append(lines[i + 1])
                i += 2
            else:
                result["endpoints"].append(line)
                i += 1

        elif current_section == "features":
            if line_lower in {"endpoints", "tools", "modalities"}:
                current_section = line_lower
                i += 1
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                result["features"][key.strip()] = val.strip()
                i += 1
            elif i + 1 < len(lines) and lines[i + 1].lower() in {
                "supported",
                "not supported",
                "yes",
                "no",
                "true",
                "false",
            }:
                result["features"][line] = lines[i + 1]
                i += 2
            else:
                result["features"][line] = "supported"
                i += 1

        elif current_section == "tools":
            if line_lower in {"endpoints", "features", "modalities"}:
                current_section = line_lower
                i += 1
                continue
            if "supported by this model" in line_lower:
                i += 1
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                result["tools"][key.strip()] = val.strip()
                i += 1
            elif i + 1 < len(lines) and lines[i + 1].lower() in {
                "supported",
                "not supported",
                "yes",
                "no",
                "true",
                "false",
            }:
                result["tools"][line] = lines[i + 1]
                i += 2
            else:
                result["tools"][line] = "supported"
                i += 1
        else:
            i += 1

    return result


def extract_capabilities_from_record(item: dict[str, Any]) -> set[ModelCapability]:
    """Extract standard ModelCapability values from any provider API model record."""
    capabilities: set[ModelCapability] = set()

    # Check for embedded raw text card or documentation
    raw_doc = item.get("raw_metadata") or item.get("description") or item.get("model_card")
    if isinstance(raw_doc, str) and any(
        kw in raw_doc.lower() for kw in ("modalities", "endpoints", "features", "tools")
    ):
        parsed_doc = parse_model_metadata_text(raw_doc)
        for section in ("modalities", "features", "tools"):
            if parsed_doc.get(section) and not item.get(section):
                item[section] = parsed_doc[section]
        if parsed_doc.get("endpoints") and not item.get("endpoints"):
            item["endpoints"] = parsed_doc["endpoints"]

    # 1. Explicit capabilities supplied in record
    supplied = item.get("capabilities")
    if isinstance(supplied, (list, set, tuple)):
        for value in supplied:
            if isinstance(value, ModelCapability):
                capabilities.add(value)
                continue
            text = str(value or "").strip().lower().replace("-", "_")
            try:
                capabilities.add(ModelCapability(text))
            except ValueError:
                if text in {"vision", "image"}:
                    capabilities.add(ModelCapability.IMAGE_INPUT)
                elif text in {"voice", "audio"}:
                    capabilities.add(ModelCapability.AUDIO_INPUT)
                    capabilities.add(ModelCapability.SPEECH_TO_TEXT)
                elif text == "video":
                    capabilities.add(ModelCapability.VIDEO_INPUT)
    elif isinstance(supplied, dict):
        for key, val in supplied.items():
            if val:
                text = str(key).strip().lower().replace("-", "_")
                try:
                    capabilities.add(ModelCapability(text))
                except ValueError:
                    if text in {"vision", "image"}:
                        capabilities.add(ModelCapability.IMAGE_INPUT)

    # 2. Supported parameters
    supported = item.get("supported_parameters") if isinstance(item.get("supported_parameters"), list) else []
    supp_lower = {str(val).lower() for val in supported}
    if any(p in supp_lower for p in ("tools", "tool_choice", "parallel_tool_calls")):
        capabilities.add(ModelCapability.TOOL_CALLING)
    if any("structured" in p or "response_format" in p for p in supp_lower):
        capabilities.add(ModelCapability.STRUCTURED_OUTPUT)
    if any("reasoning" in p or "thinking" in p for p in supp_lower):
        capabilities.add(ModelCapability.REASONING)

    # 3. Modalities (dict, list, or string)
    raw_mods = item.get("modalities")
    arch = item.get("architecture") if isinstance(item.get("architecture"), dict) else {}
    if not raw_mods and isinstance(arch, dict):
        raw_mods = arch.get("modalities")

    if isinstance(raw_mods, dict):
        for mod_name, support_val in raw_mods.items():
            m_name = str(mod_name).lower().strip()
            s_val = str(support_val).lower().strip()
            if s_val in {"not supported", "unsupported", "none", "false", "no", "0"}:
                continue
            if "image" in m_name or "vision" in m_name:
                if "input" in s_val or s_val in {"supported", "yes", "true", "1"}:
                    capabilities.add(ModelCapability.IMAGE_INPUT)
            if "audio" in m_name or "voice" in m_name or "speech" in m_name:
                if "input" in s_val or s_val in {"supported", "yes", "true", "1"}:
                    capabilities.add(ModelCapability.AUDIO_INPUT)
                    capabilities.add(ModelCapability.SPEECH_TO_TEXT)
                if "output" in s_val:
                    capabilities.add(ModelCapability.TEXT_TO_SPEECH)
                    capabilities.add(ModelCapability.AUDIO_GENERATION)
            if "video" in m_name:
                if "input" in s_val or s_val in {"supported", "yes", "true", "1"}:
                    capabilities.add(ModelCapability.VIDEO_INPUT)
                if "output" in s_val:
                    capabilities.add(ModelCapability.VIDEO_GENERATION)
            if "text" in m_name:
                capabilities.add(ModelCapability.TEXT_GENERATION)
    elif isinstance(raw_mods, (list, tuple, set)):
        for m in raw_mods:
            m_l = str(m).lower().strip()
            if m_l in {"image", "vision", "image_url"}:
                capabilities.add(ModelCapability.IMAGE_INPUT)
            if m_l in {"audio", "voice"}:
                capabilities.add(ModelCapability.AUDIO_INPUT)
                capabilities.add(ModelCapability.SPEECH_TO_TEXT)
            if m_l in {"video"}:
                capabilities.add(ModelCapability.VIDEO_INPUT)
            if m_l in {"text"}:
                capabilities.add(ModelCapability.TEXT_GENERATION)

    # 4. Input modalities (architecture or top-level)
    in_mods = arch.get("input_modalities") if isinstance(arch, dict) else None
    if not in_mods:
        in_mods = item.get("input_modalities")

    if isinstance(in_mods, dict):
        for mod_name, support_val in in_mods.items():
            m_name = str(mod_name).lower().strip()
            s_val = str(support_val).lower().strip()
            if s_val in {"not supported", "unsupported", "none", "false", "no", "0"}:
                continue
            if "image" in m_name or "vision" in m_name:
                capabilities.add(ModelCapability.IMAGE_INPUT)
            if "audio" in m_name or "voice" in m_name:
                capabilities.add(ModelCapability.AUDIO_INPUT)
                capabilities.add(ModelCapability.SPEECH_TO_TEXT)
            if "video" in m_name:
                capabilities.add(ModelCapability.VIDEO_INPUT)
            if "text" in m_name:
                capabilities.add(ModelCapability.TEXT_GENERATION)
    elif isinstance(in_mods, (list, tuple, set)):
        mods_lower = {str(m).lower() for m in in_mods}
        if mods_lower & {"image", "image_url", "vision"}:
            capabilities.add(ModelCapability.IMAGE_INPUT)
        if mods_lower & {"audio", "voice"}:
            capabilities.add(ModelCapability.AUDIO_INPUT)
            capabilities.add(ModelCapability.SPEECH_TO_TEXT)
        if mods_lower & {"video"}:
            capabilities.add(ModelCapability.VIDEO_INPUT)
        if mods_lower & {"text"}:
            capabilities.add(ModelCapability.TEXT_GENERATION)

    # 5. Output modalities (architecture or top-level)
    out_mods = arch.get("output_modalities") if isinstance(arch, dict) else None
    if not out_mods:
        out_mods = item.get("output_modalities")
    if isinstance(out_mods, (list, tuple, set)):
        out_lower = {str(m).lower() for m in out_mods}
        if out_lower & {"text"}:
            capabilities.add(ModelCapability.TEXT_GENERATION)
        if out_lower & {"image"}:
            capabilities.add(ModelCapability.IMAGE_GENERATION)
        if out_lower & {"audio", "voice", "speech"}:
            capabilities.add(ModelCapability.TEXT_TO_SPEECH)
            capabilities.add(ModelCapability.AUDIO_GENERATION)
        if out_lower & {"video"}:
            capabilities.add(ModelCapability.VIDEO_GENERATION)

    # 6. Features (dict or list)
    feats = item.get("features")
    if isinstance(feats, dict):
        for feat_name, support_val in feats.items():
            f_l = str(feat_name).lower().strip()
            s_l = str(support_val).lower().strip()
            if s_l in {"not supported", "unsupported", "false", "no", "0"}:
                continue
            if "function" in f_l or "tool" in f_l:
                capabilities.add(ModelCapability.TOOL_CALLING)
            if "structured" in f_l or "response_format" in f_l:
                capabilities.add(ModelCapability.STRUCTURED_OUTPUT)
            if "reasoning" in f_l or "thinking" in f_l:
                capabilities.add(ModelCapability.REASONING)
            if "streaming" in f_l:
                capabilities.add(ModelCapability.TEXT_GENERATION)
    elif isinstance(feats, (list, tuple, set)):
        for f in feats:
            f_l = str(f).lower().strip()
            if "function" in f_l or "tool" in f_l:
                capabilities.add(ModelCapability.TOOL_CALLING)
            if "structured" in f_l or "response_format" in f_l:
                capabilities.add(ModelCapability.STRUCTURED_OUTPUT)
            if "reasoning" in f_l or "thinking" in f_l:
                capabilities.add(ModelCapability.REASONING)

    # 7. Tools (dict or list)
    tools = item.get("tools")
    if isinstance(tools, dict):
        for tool_name, support_val in tools.items():
            t_l = str(tool_name).lower().strip()
            s_l = str(support_val).lower().strip()
            if s_l in {"not supported", "unsupported", "false", "no", "0"}:
                continue
            capabilities.add(ModelCapability.TOOL_CALLING)
            if "code" in t_l or "interpreter" in t_l or "patch" in t_l:
                capabilities.add(ModelCapability.CODE)
            if "image generation" in t_l:
                capabilities.add(ModelCapability.IMAGE_GENERATION)
    elif isinstance(tools, (list, tuple, set)) and len(tools) > 0:
        capabilities.add(ModelCapability.TOOL_CALLING)

    # 8. Endpoints (list or dict)
    endpoints = item.get("endpoints")
    ep_paths: list[str] = []
    if isinstance(endpoints, dict):
        ep_paths = [str(v) for v in endpoints.values()] + [str(k) for k in endpoints.keys()]
    elif isinstance(endpoints, (list, tuple, set)):
        ep_paths = [str(e) for e in endpoints]
    for ep in ep_paths:
        ep_l = ep.lower().strip()
        if "chat/completions" in ep_l or "responses" in ep_l or "chat" in ep_l:
            capabilities.add(ModelCapability.TEXT_GENERATION)
        if "realtime" in ep_l or "live" in ep_l:
            capabilities.add(ModelCapability.REALTIME)
        if "images/generations" in ep_l:
            capabilities.add(ModelCapability.IMAGE_GENERATION)
        if "images/edits" in ep_l:
            capabilities.add(ModelCapability.IMAGE_EDITING)
        if "audio/speech" in ep_l:
            capabilities.add(ModelCapability.TEXT_TO_SPEECH)
        if "audio/transcriptions" in ep_l:
            capabilities.add(ModelCapability.SPEECH_TO_TEXT)
        if "embeddings" in ep_l:
            capabilities.add(ModelCapability.EMBEDDING)
        if "videos" in ep_l:
            capabilities.add(ModelCapability.VIDEO_GENERATION)

    # 9. Modality string (e.g. "text+image->text", "text+audio->text")
    modality_str = str(arch.get("modality") or item.get("modality") or "").lower()
    if "image" in modality_str:
        capabilities.add(ModelCapability.IMAGE_INPUT)
    if "audio" in modality_str:
        capabilities.add(ModelCapability.AUDIO_INPUT)
        capabilities.add(ModelCapability.SPEECH_TO_TEXT)
    if "video" in modality_str:
        capabilities.add(ModelCapability.VIDEO_INPUT)

    return capabilities


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
        extracted_caps = extract_capabilities_from_record(metadata)
        capabilities = normalize_capabilities(
            provider, model_id, extracted_caps or metadata.get("capabilities")
        )
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
        available = _is_model_available(metadata)
        result.append(ModelDescriptor(
            provider=provider,
            id=model_id,
            capabilities=capabilities,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            tokenizer=str(metadata.get("tokenizer") or "") or None,
            source=source,
            available=available,
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
