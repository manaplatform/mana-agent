"""Intelligent ExecutionModeRouter dynamically selecting between realtime, parallel, and batch execution."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mana_agent.batch.models import (
    BatchConfig,
    BatchExecutionMode,
    BatchItemRequest,
)
from mana_agent.batch.router import (
    SUPPORTED_BATCH_ENDPOINTS,
    BatchDecision,
    is_model_supported_for_batch,
)
from mana_agent.config.model_discovery import (
    GLOBAL_CAPABILITY_REGISTRY,
    ModelCapabilityRegistry,
)

logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    """Operational execution mode."""

    REALTIME = "realtime"
    PARALLEL = "parallel"
    BATCH = "batch"
    AUTO = "auto"


@dataclass
class ExecutionRoutingDecision:
    """Detailed decision outcome explaining the chosen execution mode and reason."""

    selected_mode: ExecutionMode
    reason: str
    eligible_items: list[BatchItemRequest] = field(default_factory=list)
    ineligible_items: list[BatchItemRequest] = field(default_factory=list)
    discount_factor: float = 1.0
    requires_approval: bool = False
    is_latency_sensitive: bool = False

    @property
    def is_batch(self) -> bool:
        return self.selected_mode == ExecutionMode.BATCH

    @property
    def is_parallel(self) -> bool:
        return self.selected_mode == ExecutionMode.PARALLEL

    @property
    def is_realtime(self) -> bool:
        return self.selected_mode == ExecutionMode.REALTIME


class ExecutionModeRouter:
    """Intelligent router selecting among realtime, parallel tool execution, and Batch API."""

    def __init__(
        self,
        config: BatchConfig | None = None,
        registry: ModelCapabilityRegistry | None = None,
    ) -> None:
        self.config = config or BatchConfig.from_settings()
        self.registry = registry or GLOBAL_CAPABILITY_REGISTRY

    def route(
        self,
        items: Sequence[BatchItemRequest],
        *,
        model: str = "",
        model_metadata: dict[str, Any] | str | Any = None,
        mode_override: str | None = None,
        requires_streaming: bool = False,
        is_interactive: bool = False,
        is_latency_sensitive: bool = False,
        has_intra_dependencies: bool = False,
        has_parallel_tool_calls: bool = False,
        requires_approval: bool = False,
        provider: str = "openai",
    ) -> ExecutionRoutingDecision:
        """Evaluate workload requirements and select the appropriate execution mechanism."""
        # 1. Normalize provider and model
        target_model = model or ""
        target_provider = provider.lower()
        if target_model and "/" in target_model:
            p_prefix, m_suffix = target_model.split("/", 1)
            if p_prefix.lower() in {"openai", "azure_openai"}:
                target_provider = p_prefix.lower()
                target_model = m_suffix

        # Extract model from items if not supplied
        if not target_model and items:
            for it in items:
                if isinstance(it.body, dict) and it.body.get("model"):
                    target_model = str(it.body["model"])
                    break

        configured_mode = (mode_override or self.config.mode).lower()

        # Explicit overrides
        if configured_mode == ExecutionMode.REALTIME.value:
            return ExecutionRoutingDecision(
                selected_mode=ExecutionMode.REALTIME,
                reason="Execution mode explicitly set to realtime.",
                ineligible_items=list(items),
                discount_factor=1.0,
            )

        if configured_mode == ExecutionMode.PARALLEL.value:
            return ExecutionRoutingDecision(
                selected_mode=ExecutionMode.PARALLEL,
                reason="Execution mode explicitly set to parallel tool execution.",
                ineligible_items=list(items),
                discount_factor=1.0,
            )

        # Provider constraint: Batch API is OpenAI-compatible
        if target_provider not in {"openai", "azure_openai"}:
            return ExecutionRoutingDecision(
                selected_mode=ExecutionMode.REALTIME,
                reason=f"Provider '{target_provider}' does not support OpenAI Batch API.",
                ineligible_items=list(items),
                discount_factor=1.0,
            )

        # Model batch support check
        model_batch_ok = is_model_supported_for_batch(
            target_model,
            metadata=model_metadata,
            provider=target_provider,
        )

        if not model_batch_ok:
            return ExecutionRoutingDecision(
                selected_mode=ExecutionMode.REALTIME,
                reason=f"Model '{target_model}' does not support OpenAI Batch API.",
                ineligible_items=list(items),
                discount_factor=1.0,
            )

        # Latency-sensitive constraints: never schedule latency-sensitive work in batch
        if is_latency_sensitive:
            mode = ExecutionMode.PARALLEL if has_parallel_tool_calls else ExecutionMode.REALTIME
            return ExecutionRoutingDecision(
                selected_mode=mode,
                reason="Task is latency-sensitive; routed to realtime/parallel execution to guarantee immediate latency.",
                ineligible_items=list(items),
                discount_factor=1.0,
                is_latency_sensitive=True,
            )

        if requires_streaming:
            return ExecutionRoutingDecision(
                selected_mode=ExecutionMode.REALTIME,
                reason="Workload requires real-time streaming output.",
                ineligible_items=list(items),
                discount_factor=1.0,
            )

        if has_intra_dependencies:
            return ExecutionRoutingDecision(
                selected_mode=ExecutionMode.REALTIME,
                reason="Requests have intra-batch dependencies and must execute sequentially.",
                ineligible_items=list(items),
                discount_factor=1.0,
            )

        # Check interactive turn constraint when always_enable_if_supported is disabled
        if is_interactive and configured_mode != ExecutionMode.BATCH.value and not self.config.always_enable_if_supported:
            mode = ExecutionMode.PARALLEL if has_parallel_tool_calls else ExecutionMode.REALTIME
            return ExecutionRoutingDecision(
                selected_mode=mode,
                reason="Interactive conversational turn requires immediate response.",
                ineligible_items=list(items),
                discount_factor=1.0,
            )

        # Filter items by supported endpoints
        eligible: list[BatchItemRequest] = []
        ineligible: list[BatchItemRequest] = []
        for it in items:
            norm_url = (it.url or "/v1/responses").lower().rstrip("/")
            if norm_url in SUPPORTED_BATCH_ENDPOINTS:
                eligible.append(it)
            else:
                ineligible.append(it)

        if not eligible:
            mode = ExecutionMode.PARALLEL if has_parallel_tool_calls else ExecutionMode.REALTIME
            return ExecutionRoutingDecision(
                selected_mode=mode,
                reason="No items target supported OpenAI Batch API endpoints (/v1/responses, /v1/chat/completions, /v1/embeddings).",
                eligible_items=[],
                ineligible_items=list(items),
                discount_factor=1.0,
            )

        # When always_enable_if_supported is active or batch mode explicitly requested
        if self.config.always_enable_if_supported or configured_mode == ExecutionMode.BATCH.value:
            logger.info(
                "ExecutionModeRouter selected Batch API for %d items on model '%s' (always_enable_if_supported=True, 50%% discount)",
                len(eligible),
                target_model,
            )
            return ExecutionRoutingDecision(
                selected_mode=ExecutionMode.BATCH,
                reason=f"OpenAI Batch API selected for supported model '{target_model}' ({len(eligible)} requests, 50% token discount).",
                eligible_items=eligible,
                ineligible_items=ineligible,
                discount_factor=0.5,
                requires_approval=requires_approval,
            )

        # Check request count threshold for bulk workloads
        if len(eligible) >= self.config.min_requests:
            return ExecutionRoutingDecision(
                selected_mode=ExecutionMode.BATCH,
                reason=f"Eligible bulk workload: {len(eligible)} independent requests routed to OpenAI Batch API (50% token discount).",
                eligible_items=eligible,
                ineligible_items=ineligible,
                discount_factor=0.5,
                requires_approval=requires_approval,
            )

        # Fallback to parallel or realtime
        fallback_mode = ExecutionMode.PARALLEL if (has_parallel_tool_calls or len(eligible) > 1) else ExecutionMode.REALTIME
        return ExecutionRoutingDecision(
            selected_mode=fallback_mode,
            reason=f"Request count ({len(eligible)}) is below batch threshold ({self.config.min_requests}).",
            eligible_items=eligible,
            ineligible_items=ineligible,
            discount_factor=1.0,
        )


__all__ = [
    "ExecutionMode",
    "ExecutionModeRouter",
    "ExecutionRoutingDecision",
]
