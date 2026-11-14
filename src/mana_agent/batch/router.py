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
    from mana_agent.llm.capabilities import ModelProfile, metadata_facts
    if hasattr(metadata, "supports_batch"):
        return metadata.supports_batch
    if isinstance(metadata, str):
        from mana_agent.config.model_catalog import parse_model_metadata_text
        metadata = parse_model_metadata_text(metadata)
    if not isinstance(metadata, dict):
        return None
    profile = ModelProfile(provider="metadata", model_id="metadata", facts=metadata_facts(metadata, source="provider"))
    if profile.supports_batch is not True:
        return profile.supports_batch
    if profile.supported_batch_endpoints and endpoint not in profile.supported_batch_endpoints:
        return False
    return True


def is_model_supported_for_batch(model: str, endpoint: str = "/v1/responses", *, metadata: Any = None, provider: str = "openai") -> bool:
    if not isinstance(model, str) or not model.strip():
        return False
    if metadata is not None:
        explicit = parse_batch_support_from_metadata(metadata, endpoint)
        if explicit is not None:
            return explicit
    from mana_agent.llm.registry import resolve_profile
    profile = resolve_profile(provider, model)
    return profile.supports_batch is True and endpoint in profile.supported_batch_endpoints


def is_openai_batch_supported(model: str, endpoint: str = "/v1/responses", *, metadata: Any = None, provider: str = "openai") -> bool:
    from mana_agent.config.provider_registry import split_qualified_model_id
    selected_provider, model_id = split_qualified_model_id(model, default_provider=provider)
    return selected_provider == "openai" and is_model_supported_for_batch(model_id, endpoint, metadata=metadata, provider=selected_provider)


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
