"""Execution router for dynamically selecting between Realtime API and OpenAI Batch API."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from mana_agent.batch.models import (
    BatchConfig,
    BatchExecutionMode,
    BatchItemRequest,
)

logger = logging.getLogger(__name__)

SUPPORTED_BATCH_ENDPOINTS = {
    "/v1/responses",
    "/v1/chat/completions",
    "/v1/embeddings",
}

def parse_batch_support_from_metadata(metadata: Any, endpoint: str = "/v1/responses") -> bool | None:
    """Project only explicit Batch fields; other endpoints and text imply nothing."""
    if hasattr(metadata, "supports_batch") and getattr(metadata, "supports_batch") is not None:
        return metadata.supports_batch
    parsed_record: dict[str, Any] = {}
    if isinstance(metadata, str):
        from mana_agent.config.model_catalog import parse_model_metadata_text
        parsed_record = parse_model_metadata_text(metadata)
    elif isinstance(metadata, dict):
        parsed_record = dict(metadata)
    elif hasattr(metadata, "metadata") and isinstance(metadata.metadata, dict):
        parsed_record = dict(metadata.metadata)
    elif hasattr(metadata, "to_dict") and callable(metadata.to_dict):
        record = metadata.to_dict()
        if isinstance(record, dict):
            parsed_record = record
    if not parsed_record:
        return None

    nested_metadata = parsed_record.get("metadata") or parsed_record.get("model_metadata")
    if isinstance(nested_metadata, dict):
        parsed_record = {**nested_metadata, **parsed_record}

    for key in (
        "batch_supported",
        "supports_batch",
        "supports_batch_api",
        "batch_api_supported",
        "batch",
    ):
        if key in parsed_record:
            val = parsed_record[key]
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)) and val in {0, 1}:
                return bool(val)
            if isinstance(val, str):
                s = val.strip().lower()
                if any(term in s for term in ("unsupported", "not supported", "unavailable", "disabled", "no", "false", "0")):
                    return False
                if any(term in s for term in ("supported", "available", "yes", "true", "1", "enabled")):
                    return True

    features = parsed_record.get("features")
    if isinstance(features, dict):
        for feat, val in features.items():
            if "batch" in str(feat).lower():
                if val is True or val is None:
                    return True
                if isinstance(val, str):
                    s = val.lower()
                    if any(term in s for term in ("unsupported", "not supported", "disabled", "no")):
                        return False
                    if any(term in s for term in ("supported", "available", "yes", "true")):
                        return True
    elif isinstance(features, (list, tuple, set)):
        for feature in features:
            feature_text = str(feature).lower().strip()
            if "batch" in feature_text:
                if any(term in feature_text for term in ("unsupported", "not supported", "disabled", "no")):
                    return False
                return True

    endpoints = parsed_record.get("endpoints") or parsed_record.get("supported_endpoints") or parsed_record.get("api_endpoints")
    if endpoints:
        from urllib.parse import urlsplit
        ep_list = list(endpoints.values()) if isinstance(endpoints, dict) else list(endpoints) if isinstance(endpoints, (list, tuple, set)) else [str(endpoints)]
        clean_eps = set()
        for ep in ep_list:
            s = str(ep).lower().strip()
            if "://" in s:
                s = urlsplit(s).path
            s = s.split("?", 1)[0].rstrip("/")
            if not s.startswith("/"):
                s = "/" + s
            clean_eps.add(s)
        if any(path == "/v1/batches" or path.endswith("/v1/batches") for path in clean_eps):
            return True
        clean_target = str(endpoint).lower().strip()
        if "://" in clean_target:
            clean_target = urlsplit(clean_target).path
        clean_target = clean_target.split("?", 1)[0].rstrip("/")
        if not clean_target.startswith("/"):
            clean_target = "/" + clean_target
        if clean_target in clean_eps:
            return True
        if clean_eps:
            return False

    from mana_agent.llm.capabilities import ModelProfile, metadata_facts
    profile = ModelProfile(provider="metadata", model_id="metadata", facts=metadata_facts(parsed_record, source="provider"))
    if profile.supports_batch is not None:
        if profile.supports_batch is not True:
            return profile.supports_batch
        if profile.supported_batch_endpoints and endpoint not in profile.supported_batch_endpoints:
            return False
        return True
    return None


def is_model_supported_for_batch(model: str, endpoint: str = "/v1/responses", *, metadata: Any = None, provider: str = "openai") -> bool:
    if not isinstance(model, str) or not model.strip():
        return False
    if metadata is not None:
        explicit = parse_batch_support_from_metadata(metadata, endpoint)
        if explicit is not None:
            return explicit
    from mana_agent.llm.registry import resolve_profile
    profile = resolve_profile(provider, model)
    if profile.supports_batch is True and endpoint in profile.supported_batch_endpoints:
        return True
    if profile.supports_batch is False:
        return False
    if provider in {"openai", "azure_openai", "azure"}:
        try:
            from mana_agent.config.model_catalog import normalize_capabilities, ModelCapability
            caps = normalize_capabilities(provider, model)
            if caps:
                if ModelCapability.REALTIME in caps and ModelCapability.TEXT_GENERATION not in caps:
                    return False
                if ModelCapability.REALTIME in caps and "realtime" in model.lower():
                    return False
                if caps & {
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.EMBEDDING,
                    ModelCapability.CODE,
                    ModelCapability.REASONING,
                }:
                    return True
        except Exception:
            pass
        m = model.strip().lower()
        if any(unsupported in m for unsupported in ["audio", "image", "realtime", "whisper", "tts", "dall-e"]):
            return False
        return True
    return False


def is_openai_batch_supported(
    model: str,
    endpoint: str = "/v1/responses",
    *,
    metadata: Any = None,
    provider: str = "openai",
) -> bool:
    if not model or not isinstance(model, str):
        return False

    target_model = model.strip()
    target_provider = (provider or "openai").strip().lower()

    if "/" in target_model:
        prefix, suffix = target_model.split("/", 1)
        if prefix.lower() in {"openai", "azure_openai", "azure"}:
            target_provider = prefix.lower()
            target_model = suffix
        else:
            return False

    if target_provider not in {"openai", "azure_openai", "azure"}:
        return False

    m_lower = target_model.lower()
    if any(unsupported in m_lower for unsupported in ("audio", "realtime", "whisper", "tts", "dall-e", "vision-preview")):
        return False

    return is_model_supported_for_batch(target_model, endpoint=endpoint, metadata=metadata, provider=target_provider)


@dataclass
class BatchDecision:
    """Decision outcome explaining why a batch or realtime execution was chosen."""

    mode: BatchExecutionMode
    reason: str
    eligible_items: list[BatchItemRequest] = field(default_factory=list)
    ineligible_items: list[BatchItemRequest] = field(default_factory=list)
    discount_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.mode == BatchExecutionMode.BATCH and self.discount_factor == 1.0:
            self.discount_factor = 0.5

    @property
    def is_batch(self) -> bool:
        return self.mode == BatchExecutionMode.BATCH


class BatchExecutionRouter:
    """Evaluates incoming model workloads and selects Realtime vs OpenAI Batch API."""

    def __init__(self, config: BatchConfig | None = None) -> None:
        self.config = config or BatchConfig()

    def route(
        self,
        items: Sequence[BatchItemRequest],
        *,
        model: str = "",
        model_metadata: dict[str, Any] | str | Any = None,
        mode_override: str | None = None,
        requires_streaming: bool = False,
        is_interactive: bool = False,
        provider: str = "openai",
        has_intra_dependencies: bool = False,
    ) -> BatchDecision:
        """Validate an explicit remote choice; ordinary requests remain native."""
        if mode_override != BatchExecutionMode.BATCH.value:
            return BatchDecision(mode=BatchExecutionMode.REALTIME,
                reason="OpenAI Batch is disabled." if not self.config.enabled else
                       "No explicit OpenAI Batch selection; native internal/parallel execution remains selected.",
                ineligible_items=list(items))
        if not items:
            raise ValueError("An explicit OpenAI Batch cannot be empty")
        from mana_agent.batch.validation import validate_batch_selection
        target_model = model or str(items[0].body.get("model") or "")
        validate_batch_selection(self.config, model=target_model, provider=provider, endpoint=items[0].url, items=items)
        if model_metadata is not None and parse_batch_support_from_metadata(model_metadata, endpoint=items[0].url) is False:
            raise ValueError("Selected model metadata explicitly rejects this Batch endpoint")
        if requires_streaming or has_intra_dependencies:
            raise ValueError("Explicit OpenAI Batch requires non-streaming independent requests; use separate dependent waves")
        return BatchDecision(mode=BatchExecutionMode.BATCH, reason="Explicit OpenAI Batch selected and validated.",
                             eligible_items=list(items), discount_factor=0.5)


__all__ = [
    "BatchDecision",
    "BatchExecutionRouter",
    "SUPPORTED_BATCH_ENDPOINTS",
    "is_model_supported_for_batch",
    "is_openai_batch_supported",
    "parse_batch_support_from_metadata",
]
