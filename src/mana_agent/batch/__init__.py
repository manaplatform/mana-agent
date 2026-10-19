"""Official OpenAI Batch API integration for Mana-Agent.

Provides native support for /v1/batches, /v1/files (purpose=batch), automatic routing
for bulk asynchronous model workloads, 50% cost accounting discount, durable SQLite state
machine, crash resumption, and revision fencing.
"""

from __future__ import annotations

from mana_agent.batch.client import OpenAIBatchClient
from mana_agent.batch.manager import BatchLifecycleManager
from mana_agent.batch.models import (
    BatchConfig,
    BatchExecutionMode,
    BatchItemRequest,
    BatchItemResult,
    BatchJobRecord,
    BatchLifecycleStatus,
)
from mana_agent.batch.router import BatchDecision, BatchExecutionRouter, is_model_supported_for_batch
from mana_agent.batch.store import BatchJobStore

__all__ = [
    "BatchConfig",
    "BatchDecision",
    "BatchExecutionMode",
    "BatchExecutionRouter",
    "BatchItemRequest",
    "BatchItemResult",
    "BatchJobRecord",
    "BatchLifecycleManager",
    "BatchLifecycleStatus",
    "BatchJobStore",
    "OpenAIBatchClient",
    "is_model_supported_for_batch",
]
