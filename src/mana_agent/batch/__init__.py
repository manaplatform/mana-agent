"""Official OpenAI Batch API integration for Mana-Agent.

Provides native support for /v1/batches, /v1/files (purpose=batch), explicit task selection
for asynchronous model workloads, 50% cost accounting discount, durable SQLite state
machine, crash resumption, and revision fencing.
"""

from __future__ import annotations

from mana_agent.batch.builder import BatchLimits, BatchRequestBuilder
from mana_agent.batch.client import OpenAIBatchClient
from mana_agent.batch.router import (
    BatchDecision,
    BatchExecutionRouter,
    is_model_supported_for_batch,
    is_openai_batch_supported,
    parse_batch_support_from_metadata,
)
from mana_agent.batch.execution_router import (
    ExecutionMode,
    ExecutionModeRouter,
    ExecutionRoutingDecision,
)
from mana_agent.batch.manager import BatchLifecycleManager
from mana_agent.batch.models import (
    BatchConfig,
    BatchExecutionMode,
    BatchItemRequest,
    BatchItemResult,
    BatchJobRecord,
    BatchLifecycleStatus,
)
from mana_agent.batch.processor import BatchResultProcessor, ProcessedBatchResults
from mana_agent.batch.service import BatchExecutionService
from mana_agent.batch.store import BatchJobStore
from mana_agent.batch.tools import (
    batch_cancel_job,
    batch_check_status,
    batch_create_job,
    batch_download_results,
    batch_get_status,
    batch_list_jobs,
    batch_retrieve_results,
    batch_submit_requests,
    build_batch_langchain_tools,
)

__all__ = [
    "BatchConfig",
    "BatchDecision",
    "BatchExecutionMode",
    "BatchExecutionRouter",
    "BatchExecutionService",
    "BatchItemRequest",
    "BatchItemResult",
    "BatchJobRecord",
    "BatchLifecycleManager",
    "BatchLifecycleStatus",
    "BatchLimits",
    "BatchRequestBuilder",
    "BatchResultProcessor",
    "BatchJobStore",
    "ExecutionMode",
    "ExecutionModeRouter",
    "ExecutionRoutingDecision",
    "OpenAIBatchClient",
    "ProcessedBatchResults",
    "batch_cancel_job",
    "batch_check_status",
    "batch_create_job",
    "batch_download_results",
    "batch_get_status",
    "batch_list_jobs",
    "batch_retrieve_results",
    "batch_submit_requests",
    "build_batch_langchain_tools",
    "is_model_supported_for_batch",
    "is_openai_batch_supported",
    "parse_batch_support_from_metadata",
]
