"""Model-callable tools and LangChain tool wrappers for OpenAI Batch API operations."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from pydantic import BaseModel, Field

try:
    from langchain_core.tools import StructuredTool
except ImportError:
    from langchain.tools import StructuredTool  # type: ignore[no-redef]

from mana_agent.batch.manager import BatchLifecycleManager
from mana_agent.batch.models import BatchItemRequest

logger = logging.getLogger(__name__)

# Shared manager instance
_shared_manager: BatchLifecycleManager | None = None


def get_batch_manager() -> BatchLifecycleManager:
    global _shared_manager
    if _shared_manager is None:
        _shared_manager = BatchLifecycleManager()
    return _shared_manager


# --- Input Schemas ---


class BatchItemInput(BaseModel):
    custom_id: str = Field(description="Unique string identifier for this request item.")
    body: dict[str, Any] = Field(description="Request payload conforming to target endpoint (e.g. {'model': 'gpt-6-luna', 'input': '...'}).")
    url: str = Field(default="/v1/responses", description="Target endpoint: /v1/responses, /v1/chat/completions, or /v1/embeddings.")
    method: str = Field(default="POST", description="HTTP method (POST).")


class BatchCreateJobInput(BaseModel):
    requests: list[dict[str, Any]] = Field(
        description="List of request items, each with 'custom_id', 'body', and optional 'url'."
    )
    endpoint: str = Field(
        default="/v1/responses",
        description="Target OpenAI Batch API endpoint: /v1/responses, /v1/chat/completions, or /v1/embeddings.",
    )
    completion_window: str = Field(
        default="24h",
        description="Completion window for the batch job (currently '24h').",
    )
    model: str = Field(
        default="",
        description="Optional model ID (e.g. 'gpt-6-luna', 'gpt-5.6-luna', 'gpt-4o'). If specified, injected into body.",
    )
    task_id: str = Field(default="", description="Originating task ID for traceability.")
    idempotency_key: str = Field(default="", description="Deduplication key to prevent duplicate batch submission.")
    metadata: dict[str, str] | None = Field(default=None, description="Optional metadata key-value pairs.")
    mode: str = Field(default="auto", description="Execution mode: 'auto' (default), 'openai', or 'internal'.")
    provider: str = Field(default="openai", description="Explicit target inference provider.")


class BatchGetStatusInput(BaseModel):
    batch_id: str = Field(description="The OpenAI or internal batch ID (e.g. 'batch_abc123', 'batch_int_...').")


class BatchListJobsInput(BaseModel):
    limit: int = Field(default=20, description="Maximum number of batch jobs to return.")
    status: str = Field(default="", description="Filter by status (e.g. 'completed', 'in_progress', 'validating').")
    task_id: str = Field(default="", description="Filter by originating task ID.")
    sync_platform: bool = Field(default=False, description="Explicitly synchronize jobs from OpenAI; ordinary listing uses local state.")


class BatchCancelJobInput(BaseModel):
    batch_id: str = Field(description="The batch ID to cancel.")


class BatchDownloadResultsInput(BaseModel):
    batch_id: str = Field(description="The batch ID whose output results should be downloaded.")


# --- Tool Implementations ---


def batch_create_job(
    requests: Sequence[dict[str, Any]],
    *,
    endpoint: str = "/v1/responses",
    completion_window: str = "24h",
    model: str = "",
    task_id: str = "",
    run_id: str = "",
    idempotency_key: str = "",
    metadata: dict[str, str] | None = None,
    mode: str = "auto",
    provider: str = "openai",
    manager: BatchLifecycleManager | None = None,
) -> dict[str, Any]:
    """Submit a batch of requests for asynchronous execution via OpenAI Batch API or internal engine."""
    mgr = manager or get_batch_manager()
    if not requests:
        return {"ok": False, "error": "Requests list cannot be empty.", "error_code": "EMPTY_REQUESTS"}

    items: list[BatchItemRequest] = []
    target_model = str(model or "").strip()
    if not target_model:
        for raw in requests:
            if isinstance(raw, dict) and isinstance(raw.get("body"), dict):
                m = raw["body"].get("model")
                if m:
                    target_model = str(m).strip()
                    break
    if not target_model:
        return {"ok": False, "error": "A selected model is required", "error_code": "MODEL_REQUIRED"}

    for idx, raw in enumerate(requests):
        if not isinstance(raw, dict):
            return {"ok": False, "error": "Every Batch request must be an object", "error_code": "INVALID_REQUEST"}
        cid = str(raw.get("custom_id") or f"req-{idx}").strip()
        body = dict(raw.get("body") or {})
        if target_model and "model" not in body:
            body["model"] = target_model

        items.append(
            BatchItemRequest(
                custom_id=cid,
                body=body,
                url=str(raw.get("url") or endpoint),
                method=str(raw.get("method") or "POST").upper(),
                origin_task_id=task_id,
            )
        )

    try:
        record = mgr.submit_batch(
            items,
            model=target_model,
            endpoint=endpoint,
            completion_window=completion_window,
            task_id=task_id,
            run_id=run_id,
            idempotency_key=idempotency_key,
            metadata=metadata,
            mode=mode,
            provider=provider,
        )
        is_internal = bool(record.metadata.get("engine") == "internal" or record.metadata.get("mode") == "internal")
        return {
            "ok": True,
            "batch_id": record.batch_id,
            "status": record.status,
            "mode": "internal" if is_internal else "openai",
            "engine": "internal" if is_internal else "openai",
            "cost_estimate_known": record.metadata.get("cost_estimate_known") == "true",
            "total_requests": record.total_requests,
            "input_file_id": record.input_file_id,
            "created_at": record.created_at,
            "discount": "none (internal)" if is_internal else "50%",
            "discount_factor": 1.0 if is_internal else 0.5,
            "message": (
                f"Executed internal batch job {record.batch_id} with {record.total_requests} requests."
                if is_internal
                else f"Submitted OpenAI Batch API job {record.batch_id} with {record.total_requests} requests (50% cost discount)."
            ),
        }
    except Exception as exc:
        logger.warning("batch_create_job failed: %s", exc)
        return {"ok": False, "error": str(exc), "error_code": "BATCH_SUBMISSION_FAILED"}


def batch_get_status(
    batch_id: str,
    *,
    manager: BatchLifecycleManager | None = None,
) -> dict[str, Any]:
    """Retrieve current execution status and request progress for an OpenAI batch job."""
    mgr = manager or get_batch_manager()
    clean_id = str(batch_id or "").strip()
    if not clean_id:
        return {"ok": False, "error": "batch_id is required.", "error_code": "MISSING_BATCH_ID"}

    try:
        record = mgr.poll_batch(clean_id)
        return {
            "ok": True,
            "batch_id": record.batch_id,
            "status": record.status,
            "total_requests": record.total_requests,
            "completed_requests": record.completed_requests,
            "failed_requests": record.failed_requests,
            "output_file_id": record.output_file_id,
            "error_file_id": record.error_file_id,
            "in_progress_at": record.in_progress_at,
            "completed_at": record.completed_at,
            "failed_at": record.failed_at,
            "cost_estimate_usd": record.cost_estimate_usd,
        }
    except Exception as exc:
        # Fallback to local store
        local = mgr.store.get_job(clean_id)
        if local:
            return {
                "ok": True,
                "batch_id": local.batch_id,
                "status": local.status,
                "total_requests": local.total_requests,
                "completed_requests": local.completed_requests,
                "failed_requests": local.failed_requests,
                "output_file_id": local.output_file_id,
                "error_file_id": local.error_file_id,
                "cached": True,
            }
        return {"ok": False, "error": str(exc), "error_code": "BATCH_STATUS_FAILED"}


def batch_list_jobs(
    *,
    limit: int = 20,
    status: str = "",
    task_id: str = "",
    sync_platform: bool = False,
    manager: BatchLifecycleManager | None = None,
) -> dict[str, Any]:
    """List recent OpenAI batch jobs with their lifecycle status and counts."""
    mgr = manager or get_batch_manager()
    try:
        if sync_platform and not task_id:
            try:
                platform_jobs = mgr.client.list_batches(limit=limit)
                for pj in platform_jobs:
                    mgr.store.save_job(pj)
            except Exception as sync_exc:
                raise RuntimeError("Explicit OpenAI Batch synchronization failed") from sync_exc

        jobs = mgr.store.list_jobs(task_id=task_id, limit=limit)
        if status:
            s_clean = status.strip().lower()
            jobs = [j for j in jobs if j.status.lower() == s_clean]

        return {
            "ok": True,
            "total": len(jobs),
            "jobs": [
                {
                    "batch_id": j.batch_id,
                    "status": j.status,
                    "model": j.model,
                    "total_requests": j.total_requests,
                    "completed_requests": j.completed_requests,
                    "failed_requests": j.failed_requests,
                    "created_at": j.created_at,
                    "output_file_id": j.output_file_id,
                }
                for j in jobs
            ],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "error_code": "BATCH_LIST_FAILED"}


def batch_cancel_job(
    batch_id: str,
    *,
    manager: BatchLifecycleManager | None = None,
) -> dict[str, Any]:
    """Cancel an active or queued OpenAI batch job."""
    mgr = manager or get_batch_manager()
    clean_id = str(batch_id or "").strip()
    if not clean_id:
        return {"ok": False, "error": "batch_id is required.", "error_code": "MISSING_BATCH_ID"}

    try:
        record = mgr.cancel_batch(clean_id)
        return {
            "ok": True,
            "batch_id": record.batch_id,
            "status": record.status,
            "cancelled_at": record.cancelled_at,
            "message": f"Batch {clean_id} cancellation requested.",
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "error_code": "BATCH_CANCEL_FAILED"}


def batch_download_results(
    batch_id: str,
    *,
    manager: BatchLifecycleManager | None = None,
) -> dict[str, Any]:
    """Download, parse, and return output and error results for a completed batch job."""
    mgr = manager or get_batch_manager()
    clean_id = str(batch_id or "").strip()
    if not clean_id:
        return {"ok": False, "error": "batch_id is required.", "error_code": "MISSING_BATCH_ID"}

    try:
        record, results = mgr.finalize_batch(clean_id)
        return {
            "ok": True,
            "batch_id": record.batch_id,
            "status": record.status,
            "total_results": len(results),
            "completed_requests": record.completed_requests,
            "failed_requests": record.failed_requests,
            "cost_estimate_usd": record.cost_estimate_usd,
            "results": [
                {
                    "custom_id": r.custom_id,
                    "status_code": r.status_code,
                    "response": r.response,
                    "error": r.error,
                }
                for r in results
            ],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "error_code": "BATCH_DOWNLOAD_FAILED"}


# Aliases
batch_submit_requests = batch_create_job
batch_check_status = batch_get_status
batch_retrieve_results = batch_download_results


def build_batch_langchain_tools(manager: BatchLifecycleManager | None = None) -> list[StructuredTool]:
    """Build LangChain StructuredTool instances for inclusion in agent tool surfaces."""
    mgr = manager or get_batch_manager()

    def _create(
        requests: list[dict[str, Any]],
        endpoint: str = "/v1/responses",
        completion_window: str = "24h",
        model: str = "",
        task_id: str = "",
        idempotency_key: str = "",
        mode: str = "auto",
        provider: str = "openai",
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return batch_create_job(
            requests=requests,
            endpoint=endpoint,
            completion_window=completion_window,
            model=model,
            task_id=task_id,
            idempotency_key=idempotency_key,
            mode=mode,
            provider=provider,
            metadata=metadata,
            manager=mgr,
        )

    def _status(batch_id: str) -> dict[str, Any]:
        return batch_get_status(batch_id=batch_id, manager=mgr)

    def _list(limit: int = 20, status: str = "", task_id: str = "", sync_platform: bool = False) -> dict[str, Any]:
        return batch_list_jobs(limit=limit, status=status, task_id=task_id, sync_platform=sync_platform, manager=mgr)

    def _cancel(batch_id: str) -> dict[str, Any]:
        return batch_cancel_job(batch_id=batch_id, manager=mgr)

    def _download(batch_id: str) -> dict[str, Any]:
        return batch_download_results(batch_id=batch_id, manager=mgr)

    metadata = {"category": "batch_operations", "provider": "openai"}

    return [
        StructuredTool.from_function(
            func=_create,
            name="batch_create_job",
            description=(
                "Submit a batch of independent model requests for asynchronous execution. "
                "Supports OpenAI Batch API (50% cost discount) and internal batch engine."
            ),
            args_schema=BatchCreateJobInput,
            metadata=metadata,
        ),
        StructuredTool.from_function(
            func=_status,
            name="batch_get_status",
            description="Check the execution status and request counters of a batch job.",
            args_schema=BatchGetStatusInput,
            metadata=metadata,
        ),
        StructuredTool.from_function(
            func=_list,
            name="batch_list_jobs",
            description="List recent batch jobs and their current execution statuses.",
            args_schema=BatchListJobsInput,
            metadata=metadata,
        ),
        StructuredTool.from_function(
            func=_cancel,
            name="batch_cancel_job",
            description="Cancel an active or queued batch job.",
            args_schema=BatchCancelJobInput,
            metadata=metadata,
        ),
        StructuredTool.from_function(
            func=_download,
            name="batch_download_results",
            description="Download and parse results (outputs and errors) for a completed batch job.",
            args_schema=BatchDownloadResultsInput,
            metadata=metadata,
        ),
        StructuredTool.from_function(
            func=_create,
            name="batch_submit_requests",
            description=(
                "Submit a batch of requests for asynchronous execution (alias for batch_create_job). "
                "Supports OpenAI Batch API and internal batch execution."
            ),
            args_schema=BatchCreateJobInput,
            metadata=metadata,
        ),
        StructuredTool.from_function(
            func=_status,
            name="batch_check_status",
            description="Check the execution status and request counters of a batch job (alias for batch_get_status).",
            args_schema=BatchGetStatusInput,
            metadata=metadata,
        ),
        StructuredTool.from_function(
            func=_download,
            name="batch_retrieve_results",
            description="Retrieve and parse results for a completed batch job (alias for batch_download_results).",
            args_schema=BatchDownloadResultsInput,
            metadata=metadata,
        ),
    ]


__all__ = [
    "BatchCancelJobInput",
    "BatchCreateJobInput",
    "BatchDownloadResultsInput",
    "BatchGetStatusInput",
    "BatchItemInput",
    "BatchListJobsInput",
    "batch_cancel_job",
    "batch_check_status",
    "batch_create_job",
    "batch_download_results",
    "batch_get_status",
    "batch_list_jobs",
    "batch_retrieve_results",
    "batch_submit_requests",
    "build_batch_langchain_tools",
    "get_batch_manager",
]
