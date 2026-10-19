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

# Retained as a historical alias; dynamic metadata parsing is preferred.
SUPPORTED_BATCH_MODEL_FAMILIES = (
    "gpt-6",
    "gpt-5.6",
    "gpt-5",
    "gpt-4o",
    "gpt-4.1",
    "gpt-4",
    "gpt-3.5",
    "o1",
    "o3",
    "o4",
    "luna",
    "astra",
    "text-embedding",
    "chatgpt-4o",
)

UNSUPPORTED_BATCH_MODEL_KEYWORDS = (
    "realtime",
    "whisper",
    "tts",
    "dall-e",
    "vision-preview",
)


def _metadata_support_value(value: Any) -> bool | None:
    """Normalize common boolean/status values used in provider metadata."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if not isinstance(value, str):
        return None

    normalized = value.lower().strip().replace("_", " ").replace("-", " ")
    if any(term in normalized for term in ("not supported", "unsupported", "unavailable", "disabled")):
        return False
    if normalized in {"no", "false", "0", "none"}:
        return False
    if normalized in {"supported", "yes", "true", "1", "available", "enabled"}:
        return True
    return None


def parse_batch_support_from_metadata(
    metadata: dict[str, Any] | str | Any,
    endpoint: str = "/v1/responses",
) -> bool | None:
    """Parse batch API support directly from model metadata, model cards, or catalog records."""
    if metadata is None:
        return None

    parsed_record: dict[str, Any] = {}
    if isinstance(metadata, str):
        # Raw model card or platform documentation text
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

    # Catalog records commonly wrap provider-specific fields under metadata.
    nested_metadata = parsed_record.get("metadata") or parsed_record.get("model_metadata")
    if isinstance(nested_metadata, dict):
        parsed_record = {**nested_metadata, **parsed_record}

    # Provider catalogs use several names for this explicit capability.
    for key in (
        "batch_supported",
        "supports_batch",
        "supports_batch_api",
        "batch_api_supported",
        "batch",
    ):
        if key in parsed_record:
            support = _metadata_support_value(parsed_record[key])
            if support is not None:
                return support

    # 2. Check features in metadata. Lists may contain status text as well as names.
    features = parsed_record.get("features")
    if isinstance(features, dict):
        for feat, val in features.items():
            if "batch" in str(feat).lower():
                support = _metadata_support_value(val)
                if support is not None:
                    return support
                # A feature flag with no explicit value means it is listed as supported.
                if val is True or val is None:
                    return True
    elif isinstance(features, (list, tuple, set)):
        for feature in features:
            feature_text = str(feature).lower().strip()
            if "batch" in feature_text:
                support = _metadata_support_value(feature_text)
                if support is not None:
                    return support
                return True

    # 3. Check endpoint/API lists in metadata.
    endpoints = parsed_record.get("endpoints")
    if endpoints is None:
        endpoints = parsed_record.get("supported_endpoints") or parsed_record.get("api_endpoints")
    ep_paths: list[str] = []
    if isinstance(endpoints, dict):
        ep_paths = [str(v) for v in endpoints.values()] + [str(k) for k in endpoints.keys()]
    elif isinstance(endpoints, (list, tuple, set)):
        ep_paths = [str(e) for e in endpoints]
    elif isinstance(endpoints, str):
        ep_paths = [endpoints]

    def normalize_endpoint(value: str) -> str:
        value = value.strip().lower()
        if "://" in value:
            value = urlsplit(value).path
        value = value.split("?", 1)[0].rstrip("/")
        return value if value.startswith("/") else f"/{value}"

    clean_eps = {normalize_endpoint(value) for value in ep_paths if value.strip()}
    if any(path == "/v1/batches" or path.endswith("/v1/batches") for path in clean_eps):
        return True

    clean_target = normalize_endpoint(endpoint)
    if clean_target in clean_eps:
        return True
    if clean_eps:
        # Endpoint metadata is authoritative: a model that only lists a different
        # endpoint (for example, a preview route) must not be inferred compatible.
        return False

    # 4. Check capabilities from parsed record
    try:
        from mana_agent.config.model_catalog import ModelCapability, extract_capabilities_from_record

        caps = extract_capabilities_from_record(parsed_record)
        if caps:
            if ModelCapability.REALTIME in caps and ModelCapability.TEXT_GENERATION not in caps:
                return False
            if any(
                c in caps
                for c in (
                    ModelCapability.SPEECH_TO_TEXT,
                    ModelCapability.TEXT_TO_SPEECH,
                    ModelCapability.IMAGE_GENERATION,
                    ModelCapability.VIDEO_GENERATION,
                )
            ) and not (
                caps
                & {
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.EMBEDDING,
                    ModelCapability.CODE,
                    ModelCapability.REASONING,
                }
            ):
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

    return None


def is_model_supported_for_batch(
    model: str,
    endpoint: str = "/v1/responses",
    *,
    metadata: dict[str, Any] | str | Any = None,
    provider: str = "openai",
) -> bool:
    """Determine whether a model supports the OpenAI Batch API.

    Dynamically parses support from model metadata (model cards, platform tables,
    catalog records, or provider descriptors) without relying on hardcoded model lists.
    """
    if not model or not isinstance(model, str):
        return True  # If model is unspecified, default to supported for standard OpenAI endpoints

    # 1. Parse directly from explicitly supplied metadata if present
    if metadata is not None:
        parsed_support = parse_batch_support_from_metadata(metadata, endpoint=endpoint)
        if parsed_support is not None:
            return parsed_support

    # 2. Extract and parse metadata dynamically from model capability descriptor
    try:
        from mana_agent.config.model_discovery import GLOBAL_CAPABILITY_REGISTRY

        prof = GLOBAL_CAPABILITY_REGISTRY.get(provider, model)
        if prof and prof.supports_batch is not None:
            return prof.supports_batch
    except Exception:
        pass

    try:
        from mana_agent.config.model_capabilities import resolve_model_capability

        desc = resolve_model_capability(provider, model)
        if desc and desc.metadata:
            parsed_support = parse_batch_support_from_metadata(desc.metadata, endpoint=endpoint)
            if parsed_support is not None:
                return parsed_support
    except Exception:
        pass

    # 3. Parse from normalized model capabilities in catalog
    try:
        from mana_agent.config.model_catalog import ModelCapability, normalize_capabilities

        caps = normalize_capabilities(provider, model)
        if caps:
            # Realtime-only models cannot run on Batch API
            if ModelCapability.REALTIME in caps and ModelCapability.TEXT_GENERATION not in caps:
                return False
            if ModelCapability.REALTIME in caps and "realtime" in model.lower():
                return False
            # Pure audio, image generation, or video models cannot run on batch responses
            if any(
                c in caps
                for c in (
                    ModelCapability.SPEECH_TO_TEXT,
                    ModelCapability.TEXT_TO_SPEECH,
                    ModelCapability.IMAGE_GENERATION,
                    ModelCapability.VIDEO_GENERATION,
                )
            ) and not (
                caps
                & {
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.EMBEDDING,
                    ModelCapability.CODE,
                    ModelCapability.REASONING,
                }
            ):
                return False
            # Text generation, code, reasoning, and embedding models support Batch API
            if caps & {
                ModelCapability.TEXT_GENERATION,
                ModelCapability.EMBEDDING,
                ModelCapability.CODE,
                ModelCapability.REASONING,
            }:
                return True
    except Exception:
        pass

    # 4. Fallback for unclassified custom models: reject obvious non-batch indicators
    m = model.strip().lower()
    for keyword in UNSUPPORTED_BATCH_MODEL_KEYWORDS:
        if keyword in m:
            return False

    return not any(unsupported in m for unsupported in ["audio", "image", "realtime"])


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
        """Evaluate a group of requests and decide whether to route to OpenAI Batch API."""
        if not self.config.enabled:
            return BatchDecision(
                mode=BatchExecutionMode.REALTIME,
                reason="OpenAI Batch API is disabled in configuration.",
                ineligible_items=list(items),
                discount_factor=1.0,
            )

        effective_mode_str = (mode_override or self.config.mode).lower()
        if effective_mode_str == BatchExecutionMode.REALTIME.value:
            return BatchDecision(
                mode=BatchExecutionMode.REALTIME,
                reason="Batch execution mode explicitly configured as realtime.",
                ineligible_items=list(items),
                discount_factor=1.0,
            )

        # Provider constraint: Batch API is OpenAI-specific
        if provider.lower() not in {"openai", "azure_openai"}:
            return BatchDecision(
                mode=BatchExecutionMode.REALTIME,
                reason=f"Provider '{provider}' does not support OpenAI Batch API.",
                ineligible_items=list(items),
                discount_factor=1.0,
            )

        # Determine target model and metadata
        target_model = model or ""
        target_metadata = model_metadata
        if items:
            for it in items:
                if not isinstance(it.body, dict):
                    continue
                item_model = it.body.get("model")
                if not target_model and item_model:
                    target_model = str(item_model)
                # Metadata can live on the request body even when callers also
                # supply `model=` explicitly; only use metadata for that model.
                if target_metadata is None and (not item_model or str(item_model) == target_model):
                    target_metadata = it.body.get("model_metadata") or it.body.get("metadata")

        # Model support check via parsed metadata
        if target_model and not is_model_supported_for_batch(
            target_model,
            metadata=target_metadata,
            provider=provider,
        ):
            return BatchDecision(
                mode=BatchExecutionMode.REALTIME,
                reason=f"Model '{target_model}' is not supported by OpenAI Batch API based on metadata.",
                ineligible_items=list(items),
                discount_factor=1.0,
            )

        # Realtime requirements: streaming or hard intra-batch dependencies
        if requires_streaming:
            return BatchDecision(
                mode=BatchExecutionMode.REALTIME,
                reason="Workload requires real-time streaming.",
                ineligible_items=list(items),
                discount_factor=1.0,
            )

        if is_interactive and effective_mode_str != BatchExecutionMode.BATCH.value and not self.config.always_enable_if_supported:
            return BatchDecision(
                mode=BatchExecutionMode.REALTIME,
                reason="Interactive user conversational turn requires immediate response.",
                ineligible_items=list(items),
                discount_factor=1.0,
            )

        if has_intra_dependencies:
            return BatchDecision(
                mode=BatchExecutionMode.REALTIME,
                reason="Workload items contain intra-batch dependencies and cannot execute asynchronously as a single batch.",
                ineligible_items=list(items),
                discount_factor=1.0,
            )

        # Validate item count
        if not items:
            return BatchDecision(
                mode=BatchExecutionMode.REALTIME,
                reason="No request items provided.",
                ineligible_items=[],
                discount_factor=1.0,
            )

        # Filter items by supported endpoints
        eligible: list[BatchItemRequest] = []
        ineligible: list[BatchItemRequest] = []
        for it in items:
            normalized_url = (it.url or "/v1/responses").lower().rstrip("/")
            if normalized_url in SUPPORTED_BATCH_ENDPOINTS:
                eligible.append(it)
            else:
                ineligible.append(it)

        if not eligible:
            return BatchDecision(
                mode=BatchExecutionMode.REALTIME,
                reason="No items target supported OpenAI Batch API endpoints (/v1/responses, /v1/chat/completions, /v1/embeddings).",
                eligible_items=[],
                ineligible_items=list(items),
                discount_factor=1.0,
            )

        # Check always_enable_if_supported
        if self.config.always_enable_if_supported or effective_mode_str == BatchExecutionMode.BATCH.value:
            logger.info(
                "BatchExecutionRouter routed %d requests to OpenAI Batch API (always_enable_if_supported=True, 50%% discount)",
                len(eligible),
            )
            return BatchDecision(
                mode=BatchExecutionMode.BATCH,
                reason=f"OpenAI Batch API enabled for supported model '{target_model or 'openai'}' ({len(eligible)} requests, 50% token discount).",
                eligible_items=eligible,
                ineligible_items=ineligible,
                discount_factor=0.5,
            )

        # Check min_requests threshold if always_enable_if_supported is disabled
        min_reqs = self.config.min_requests
        if len(eligible) < min_reqs:
            return BatchDecision(
                mode=BatchExecutionMode.REALTIME,
                reason=f"Eligible request count ({len(eligible)}) is below min_requests threshold ({min_reqs}).",
                eligible_items=eligible,
                ineligible_items=ineligible + (eligible if len(eligible) < min_reqs else []),
                discount_factor=1.0,
            )

        logger.info(
            "BatchExecutionRouter routed %d requests to OpenAI Batch API (50%% cost discount). Reason: eligible bulk workload",
            len(eligible),
        )

        return BatchDecision(
            mode=BatchExecutionMode.BATCH,
            reason=f"Eligible for OpenAI Batch API: {len(eligible)} independent requests queued (50% token cost discount).",
            eligible_items=eligible,
            ineligible_items=ineligible,
            discount_factor=0.5,
        )


__all__ = [
    "BatchDecision",
    "BatchExecutionRouter",
    "SUPPORTED_BATCH_ENDPOINTS",
    "SUPPORTED_BATCH_MODEL_FAMILIES",
    "is_model_supported_for_batch",
    "parse_batch_support_from_metadata",
]
