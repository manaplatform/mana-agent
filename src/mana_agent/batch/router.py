"""Execution router for dynamically selecting between Realtime API and OpenAI Batch API."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

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

SUPPORTED_BATCH_MODEL_FAMILIES = (
    "gpt-5",
    "gpt-4o",
    "gpt-4",
    "gpt-3.5",
    "o1",
    "o3",
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


def is_model_supported_for_batch(model: str, endpoint: str = "/v1/responses") -> bool:
    """Determine whether a model supports the OpenAI Batch API."""
    if not model or not isinstance(model, str):
        return True  # If model is unspecified, default to supported for standard OpenAI endpoints
    m = model.strip().lower()

    for keyword in UNSUPPORTED_BATCH_MODEL_KEYWORDS:
        if keyword in m:
            return False

    for family in SUPPORTED_BATCH_MODEL_FAMILIES:
        if family in m:
            return True

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

        # Determine target model
        target_model = model or ""
        if not target_model and items:
            for it in items:
                m = it.body.get("model") if isinstance(it.body, dict) else ""
                if m:
                    target_model = str(m)
                    break

        # Model support check
        if target_model and not is_model_supported_for_batch(target_model):
            return BatchDecision(
                mode=BatchExecutionMode.REALTIME,
                reason=f"Model '{target_model}' is not supported by OpenAI Batch API.",
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
