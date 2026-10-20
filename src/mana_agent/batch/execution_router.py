"""Intelligent ExecutionModeRouter dynamically selecting between realtime, parallel, and batch execution."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mana_agent.batch.models import (
    BatchConfig,
    BatchItemRequest,
)
from mana_agent.batch.router import (
    parse_batch_support_from_metadata,
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
        if mode_override == ExecutionMode.BATCH.value:
            from mana_agent.batch.validation import validate_batch_selection
            if parse_batch_support_from_metadata(model_metadata) is False:
                raise ValueError("Selected model metadata explicitly disables OpenAI Batch")
            validate_batch_selection(self.config, model=target_model, provider=target_provider,
                                     endpoint=items[0].url if items else self.config.endpoint, items=items)
            if requires_streaming or has_intra_dependencies:
                raise ValueError("Explicit OpenAI Batch requires non-streaming independent requests; split dependent waves")
            return ExecutionRoutingDecision(
                selected_mode=ExecutionMode.BATCH, reason="Explicit user-selected OpenAI Batch execution.",
                eligible_items=list(items), discount_factor=0.5, requires_approval=requires_approval,
            )
        return ExecutionRoutingDecision(
            selected_mode=(ExecutionMode.REALTIME if mode_override == "realtime" else
                           ExecutionMode.PARALLEL if configured_mode == "parallel" or has_parallel_tool_calls or len(items) > 1 else ExecutionMode.REALTIME),
            reason="No explicit OpenAI Batch selection; native internal/parallel execution remains selected.",
            ineligible_items=list(items),
        )



__all__ = [
    "ExecutionMode",
    "ExecutionModeRouter",
    "ExecutionRoutingDecision",
]
