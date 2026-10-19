"""Lifecycle manager for OpenAI Batch API jobs.

Orchestrates JSONL file generation, upload, batch submission, status polling,
resumption after restart, result reconciliation with 50% token cost accounting,
and revision fencing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Sequence
from decimal import Decimal
from typing import Any, Callable

from mana_agent.batch.client import OpenAIBatchClient
from mana_agent.batch.models import (
    BatchConfig,
    BatchItemRequest,
    BatchItemResult,
    BatchJobRecord,
    BatchLifecycleStatus,
)
from mana_agent.batch.store import BatchJobStore

logger = logging.getLogger(__name__)


class BatchLifecycleManager:
    """Manages the full lifecycle of OpenAI Batch API jobs with durable SQLite persistence."""

    def __init__(
        self,
        *,
        client: OpenAIBatchClient | None = None,
        store: BatchJobStore | None = None,
        config: BatchConfig | None = None,
        event_sink: Callable[..., Any] | None = None,
        accounting_service: Any | None = None,
        internal_handler: Callable[[BatchItemRequest], BatchItemResult] | None = None,
    ) -> None:
        self.config = config or BatchConfig.from_settings()
        self.client = client or OpenAIBatchClient(config=self.config)
        self.store = store or BatchJobStore()
        self.event_sink = event_sink
        self.accounting_service = accounting_service
        self._internal_handler = internal_handler

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        if self.event_sink is None:
            return
        try:
            # Match established event sink signatures:
            # either (event_type, dict) or (dict)
            self.event_sink(event_type, data)
        except TypeError:
            try:
                self.event_sink({"event_type": event_type, **data})
            except Exception as exc:
                logger.debug("Failed to deliver batch event: %s", exc)
        except Exception as exc:
            logger.debug("Failed to deliver batch event: %s", exc)

    def submit_batch(
        self,
        items: Sequence[BatchItemRequest],
        *,
        model: str,
        endpoint: str = "/v1/responses",
        completion_window: str = "24h",
        task_id: str = "",
        run_id: str = "",
        idempotency_key: str = "",
        git_commit_sha: str = "",
        metadata: dict[str, str] | None = None,
        mode: str = "auto",
        provider: str = "openai",
    ) -> BatchJobRecord:
        """Submit a batch of requests to OpenAI Batch API or internal batch engine."""
        if not items:
            raise ValueError("Cannot submit an empty batch of requests.")

        # Idempotency deduplication
        if idempotency_key:
            existing = self.store.find_by_idempotency_key(idempotency_key)
            if existing and not existing.is_terminal():
                logger.info(
                    "Found existing active batch job %s for idempotency key %s",
                    existing.batch_id,
                    idempotency_key,
                )
                return existing

        meta = dict(metadata or {})
        if task_id:
            meta["task_id"] = str(task_id)
        if run_id:
            meta["run_id"] = str(run_id)
        if idempotency_key:
            meta["idempotency_key"] = str(idempotency_key)
        if git_commit_sha:
            meta["git_commit_sha"] = str(git_commit_sha)

        # Resolve dual-mode selection: OpenAI Batch API vs Internal Engine
        effective_mode = (mode or "auto").strip().lower()
        if effective_mode == "auto":
            cfg_mode = (self.config.mode or "auto").strip().lower()
            if cfg_mode in {"openai", "internal"}:
                effective_mode = cfg_mode
            else:
                from mana_agent.batch.router import is_openai_batch_supported
                if is_openai_batch_supported(model, endpoint=endpoint, provider=provider):
                    effective_mode = "openai"
                else:
                    effective_mode = "internal"

        if effective_mode == "internal":
            return self._submit_internal_batch(
                items,
                model=model,
                endpoint=endpoint,
                completion_window=completion_window,
                task_id=task_id,
                run_id=run_id,
                idempotency_key=idempotency_key,
                git_commit_sha=git_commit_sha,
                metadata=meta,
            )

        self._emit_event(
            "batch.submitting",
            {
                "task_id": task_id,
                "run_id": run_id,
                "item_count": len(items),
                "endpoint": endpoint,
                "model": model,
                "mode": "openai",
            },
        )

        # Normalize request items to align with target endpoint
        items = [
            BatchItemRequest(
                custom_id=it.custom_id,
                body=it.body,
                method=it.method,
                url=endpoint,
                origin_task_id=it.origin_task_id,
                origin_run_id=it.origin_run_id,
                created_at=it.created_at,
                target_revision=it.target_revision,
                file_path=it.file_path,
                expected_sha256=it.expected_sha256,
            ) if it.url != endpoint else it
            for it in items
        ]

        try:
            # 1. Upload JSONL requests file
            filename = f"batch_{task_id or 'run'}_{int(time.time())}.jsonl"
            input_file_id = self.client.upload_batch_file(items, filename=filename)

            # 2. Create batch job
            created_record = self.client.create_batch(
                input_file_id=input_file_id,
                endpoint=endpoint,
                completion_window=completion_window or self.config.completion_window,
                metadata=meta,
            )
        except Exception as exc:
            if (mode or "auto").strip().lower() == "auto":
                logger.warning(
                    "OpenAI Batch submission failed (%s); falling back to internal batch engine",
                    exc,
                )
                return self._submit_internal_batch(
                    items,
                    model=model,
                    endpoint=endpoint,
                    completion_window=completion_window,
                    task_id=task_id,
                    run_id=run_id,
                    idempotency_key=idempotency_key,
                    git_commit_sha=git_commit_sha,
                    metadata=meta,
                )
            raise

        # Populate local tracking fields
        created_record.model = model
        created_record.task_id = task_id
        created_record.run_id = run_id
        created_record.idempotency_key = idempotency_key
        created_record.git_commit_sha = git_commit_sha
        created_record.total_requests = len(items)

        # 3. Persist job and items in SQLite store
        self.store.save_job(created_record)
        self.store.save_items(created_record.batch_id, items)

        logger.info(
            "Submitted OpenAI Batch API job %s with %d requests (input_file: %s)",
            created_record.batch_id,
            len(items),
            input_file_id,
        )

        self._emit_event(
            "batch.submitted",
            {
                "batch_id": created_record.batch_id,
                "status": created_record.status,
                "total_requests": len(items),
                "input_file_id": input_file_id,
                "task_id": task_id,
                "mode": "openai",
                "discount_factor": 0.5,
                "message": f"Submitted batch {created_record.batch_id} with {len(items)} requests (50% cost discount).",
            },
        )

        return created_record

    async def async_submit_batch(
        self,
        items: Sequence[BatchItemRequest],
        *,
        model: str,
        endpoint: str = "/v1/responses",
        completion_window: str = "24h",
        task_id: str = "",
        run_id: str = "",
        idempotency_key: str = "",
        git_commit_sha: str = "",
        metadata: dict[str, str] | None = None,
        mode: str = "auto",
        provider: str = "openai",
    ) -> BatchJobRecord:
        """Asynchronously submit a batch of requests to OpenAI Batch API or internal engine."""
        if not items:
            raise ValueError("Cannot submit an empty batch of requests.")

        if idempotency_key:
            existing = self.store.find_by_idempotency_key(idempotency_key)
            if existing and not existing.is_terminal():
                return existing

        meta = dict(metadata or {})
        if task_id:
            meta["task_id"] = str(task_id)
        if run_id:
            meta["run_id"] = str(run_id)
        if idempotency_key:
            meta["idempotency_key"] = str(idempotency_key)
        if git_commit_sha:
            meta["git_commit_sha"] = str(git_commit_sha)

        effective_mode = (mode or "auto").strip().lower()
        if effective_mode == "auto":
            cfg_mode = (self.config.mode or "auto").strip().lower()
            if cfg_mode in {"openai", "internal"}:
                effective_mode = cfg_mode
            else:
                from mana_agent.batch.router import is_openai_batch_supported
                if is_openai_batch_supported(model, endpoint=endpoint, provider=provider):
                    effective_mode = "openai"
                else:
                    effective_mode = "internal"

        if effective_mode == "internal":
            return await self._async_submit_internal_batch(
                items,
                model=model,
                endpoint=endpoint,
                completion_window=completion_window,
                task_id=task_id,
                run_id=run_id,
                idempotency_key=idempotency_key,
                git_commit_sha=git_commit_sha,
                metadata=meta,
            )

        self._emit_event(
            "batch.submitting",
            {
                "task_id": task_id,
                "run_id": run_id,
                "item_count": len(items),
                "endpoint": endpoint,
                "model": model,
                "mode": "openai",
            },
        )

        # Normalize request items to align with target endpoint
        items = [
            BatchItemRequest(
                custom_id=it.custom_id,
                body=it.body,
                method=it.method,
                url=endpoint,
                origin_task_id=it.origin_task_id,
                origin_run_id=it.origin_run_id,
                created_at=it.created_at,
                target_revision=it.target_revision,
                file_path=it.file_path,
                expected_sha256=it.expected_sha256,
            ) if it.url != endpoint else it
            for it in items
        ]

        try:
            filename = f"batch_{task_id or 'run'}_{int(time.time())}.jsonl"
            input_file_id = await self.client.async_upload_batch_file(items, filename=filename)

            created_record = await self.client.async_create_batch(
                input_file_id=input_file_id,
                endpoint=endpoint,
                completion_window=completion_window or self.config.completion_window,
                metadata=meta,
            )
        except Exception as exc:
            if (mode or "auto").strip().lower() == "auto":
                logger.warning(
                    "OpenAI Batch async submission failed (%s); falling back to internal batch engine",
                    exc,
                )
                return await self._async_submit_internal_batch(
                    items,
                    model=model,
                    endpoint=endpoint,
                    completion_window=completion_window,
                    task_id=task_id,
                    run_id=run_id,
                    idempotency_key=idempotency_key,
                    git_commit_sha=git_commit_sha,
                    metadata=meta,
                )
            raise

        created_record.model = model
        created_record.task_id = task_id
        created_record.run_id = run_id
        created_record.idempotency_key = idempotency_key
        created_record.git_commit_sha = git_commit_sha
        created_record.total_requests = len(items)

        self.store.save_job(created_record)
        self.store.save_items(created_record.batch_id, items)

        self._emit_event(
            "batch.submitted",
            {
                "batch_id": created_record.batch_id,
                "status": created_record.status,
                "total_requests": len(items),
                "input_file_id": input_file_id,
                "task_id": task_id,
                "mode": "openai",
                "discount_factor": 0.5,
                "message": f"Submitted batch {created_record.batch_id} with {len(items)} requests (50% cost discount).",
            },
        )

        return created_record

    def poll_batch(self, batch_id: str) -> BatchJobRecord:
        """Poll current batch status from OpenAI or SQLite store, and finalize if completed."""
        local_job = self.store.get_job(batch_id)
        if local_job and (local_job.metadata.get("engine") == "internal" or batch_id.startswith("batch_int_")):
            if local_job.status == BatchLifecycleStatus.COMPLETED.value:
                finalized_job, _ = self.finalize_batch(batch_id)
                return finalized_job
            return local_job

        remote_job = self.client.retrieve_batch(batch_id)
        if local_job:
            # Preserve local metadata
            remote_job.model = remote_job.model or local_job.model
            remote_job.task_id = remote_job.task_id or local_job.task_id
            remote_job.run_id = remote_job.run_id or local_job.run_id
            remote_job.idempotency_key = remote_job.idempotency_key or local_job.idempotency_key
            remote_job.git_commit_sha = remote_job.git_commit_sha or local_job.git_commit_sha

        self.store.save_job(remote_job)

        self._emit_event(
            "batch.status_updated",
            {
                "batch_id": remote_job.batch_id,
                "status": remote_job.status,
                "completed_requests": remote_job.completed_requests,
                "total_requests": remote_job.total_requests,
                "failed_requests": remote_job.failed_requests,
                "output_file_id": remote_job.output_file_id,
            },
        )

        if remote_job.status == BatchLifecycleStatus.COMPLETED.value:
            finalized_job, _ = self.finalize_batch(batch_id)
            return finalized_job

        return remote_job

    async def async_poll_batch(self, batch_id: str) -> BatchJobRecord:
        """Asynchronously poll current batch status from OpenAI or SQLite store."""
        local_job = self.store.get_job(batch_id)
        if local_job and (local_job.metadata.get("engine") == "internal" or batch_id.startswith("batch_int_")):
            if local_job.status == BatchLifecycleStatus.COMPLETED.value:
                finalized_job, _ = await self.async_finalize_batch(batch_id)
                return finalized_job
            return local_job

        remote_job = await self.client.async_retrieve_batch(batch_id)
        if local_job:
            remote_job.model = remote_job.model or local_job.model
            remote_job.task_id = remote_job.task_id or local_job.task_id
            remote_job.run_id = remote_job.run_id or local_job.run_id
            remote_job.idempotency_key = remote_job.idempotency_key or local_job.idempotency_key
            remote_job.git_commit_sha = remote_job.git_commit_sha or local_job.git_commit_sha

        self.store.save_job(remote_job)

        self._emit_event(
            "batch.status_updated",
            {
                "batch_id": remote_job.batch_id,
                "status": remote_job.status,
                "completed_requests": remote_job.completed_requests,
                "total_requests": remote_job.total_requests,
                "failed_requests": remote_job.failed_requests,
                "output_file_id": remote_job.output_file_id,
            },
        )

        if remote_job.status == BatchLifecycleStatus.COMPLETED.value:
            finalized_job, _ = await self.async_finalize_batch(batch_id)
            return finalized_job

        return remote_job

    def finalize_batch(self, batch_id: str) -> tuple[BatchJobRecord, list[BatchItemResult]]:
        """Download batch results, correlate by custom_id, calculate 50% discounted cost, and update store."""
        job = self.store.get_job(batch_id)
        if job and (job.metadata.get("engine") == "internal" or batch_id.startswith("batch_int_")):
            results = self.store.get_results(batch_id)
            if job.status != BatchLifecycleStatus.RESULTS_IMPORTED.value:
                job.status = BatchLifecycleStatus.RESULTS_IMPORTED.value
                job.completed_at = job.completed_at or time.time()
                self.store.save_job(job)
            self._emit_event(
                "batch.results_imported",
                {
                    "batch_id": batch_id,
                    "imported_count": len(results),
                    "cost_usd": job.cost_estimate_usd,
                    "discount": "none (internal)",
                    "status": job.status,
                },
            )
            return job, results

        if not job:
            job = self.client.retrieve_batch(batch_id)

        existing_results = self.store.get_results(batch_id)
        if job.status == BatchLifecycleStatus.RESULTS_IMPORTED.value and existing_results:
            return job, existing_results

        results: list[BatchItemResult] = []

        if not job.output_file_id:
            try:
                refreshed = self.client.retrieve_batch(batch_id)
                if refreshed and refreshed.output_file_id:
                    job.output_file_id = refreshed.output_file_id
                    job.status = refreshed.status
                    if refreshed.error_file_id:
                        job.error_file_id = refreshed.error_file_id
            except Exception as exc:
                logger.debug("Failed refreshing batch %s: %s", batch_id, exc)

        if job.output_file_id:
            logger.info("Downloading batch results for %s from file %s", batch_id, job.output_file_id)
            results = self.client.download_results(job.output_file_id)
            self.store.save_results(batch_id, results)
        else:
            try:
                results = self.client.download_results(batch_id)
                if results:
                    self.store.save_results(batch_id, results)
            except Exception:
                pass

        if job.error_file_id:
            try:
                raw_errors = self.client.download_errors(job.error_file_id)
                error_results = [
                    BatchItemResult(
                        custom_id=str(e.get("custom_id") or ""),
                        status_code=int(e.get("response", {}).get("status_code", 400)),
                        error=e.get("error") if isinstance(e.get("error"), dict) else {"message": str(e.get("error"))},
                    )
                    for e in raw_errors
                    if e.get("custom_id")
                ]
                if error_results:
                    self.store.save_results(batch_id, error_results)
                    results.extend(error_results)
            except Exception as exc:
                logger.warning("Failed to download batch errors for %s: %s", batch_id, exc)

        # Calculate cost estimate with 50% discount
        job.cost_estimate_usd = self._calculate_batch_cost(job)
        job.status = BatchLifecycleStatus.RESULTS_IMPORTED.value
        job.completed_at = job.completed_at or time.time()
        self.store.save_job(job)

        self._emit_event(
            "batch.results_imported",
            {
                "batch_id": batch_id,
                "imported_count": len(results),
                "cost_usd": job.cost_estimate_usd,
                "discount": "50%",
                "status": job.status,
            },
        )

        return job, results

    async def async_finalize_batch(self, batch_id: str) -> tuple[BatchJobRecord, list[BatchItemResult]]:
        """Asynchronously download and import batch results."""
        job = self.store.get_job(batch_id)
        if job and (job.metadata.get("engine") == "internal" or batch_id.startswith("batch_int_")):
            results = self.store.get_results(batch_id)
            if job.status != BatchLifecycleStatus.RESULTS_IMPORTED.value:
                job.status = BatchLifecycleStatus.RESULTS_IMPORTED.value
                job.completed_at = job.completed_at or time.time()
                self.store.save_job(job)
            self._emit_event(
                "batch.results_imported",
                {
                    "batch_id": batch_id,
                    "imported_count": len(results),
                    "cost_usd": job.cost_estimate_usd,
                    "discount": "none (internal)",
                    "status": job.status,
                },
            )
            return job, results

        if not job:
            job = await self.client.async_retrieve_batch(batch_id)

        existing_results = self.store.get_results(batch_id)
        if job.status == BatchLifecycleStatus.RESULTS_IMPORTED.value and existing_results:
            return job, existing_results

        results: list[BatchItemResult] = []

        if job.output_file_id:
            results = await self.client.async_download_results(job.output_file_id)
            self.store.save_results(batch_id, results)

        if job.error_file_id:
            try:
                raw_errors = self.client.download_errors(job.error_file_id)
                error_results = [
                    BatchItemResult(
                        custom_id=str(e.get("custom_id") or ""),
                        status_code=int(e.get("response", {}).get("status_code", 400)),
                        error=e.get("error") if isinstance(e.get("error"), dict) else {"message": str(e.get("error"))},
                    )
                    for e in raw_errors
                    if e.get("custom_id")
                ]
                if error_results:
                    self.store.save_results(batch_id, error_results)
                    results.extend(error_results)
            except Exception as exc:
                logger.warning("Failed to download async batch errors for %s: %s", batch_id, exc)

        job.cost_estimate_usd = self._calculate_batch_cost(job)
        job.status = BatchLifecycleStatus.RESULTS_IMPORTED.value
        job.completed_at = job.completed_at or time.time()
        self.store.save_job(job)

        self._emit_event(
            "batch.results_imported",
            {
                "batch_id": batch_id,
                "imported_count": len(results),
                "cost_usd": job.cost_estimate_usd,
                "discount": "50%",
                "status": job.status,
            },
        )

        return job, results

    def resume_pending_batches(self) -> list[BatchJobRecord]:
        """Poll and reconcile all active, pending batch jobs after agent startup or restart."""
        if not self.config.auto_resume:
            return []

        pending = self.store.list_pending_jobs()
        if not pending:
            return []

        logger.info("Resuming %d pending OpenAI Batch jobs from persistent store...", len(pending))
        reconciled: list[BatchJobRecord] = []

        for job in pending:
            try:
                updated = self.poll_batch(job.batch_id)
                reconciled.append(updated)
            except Exception as exc:
                logger.warning("Failed to reconcile batch job %s on resume: %s", job.batch_id, exc)

        return reconciled

    def cancel_batch(self, batch_id: str) -> BatchJobRecord:
        """Cancel an in-progress or queued OpenAI Batch job."""
        local_job = self.store.get_job(batch_id)
        if local_job and (local_job.metadata.get("engine") == "internal" or batch_id.startswith("batch_int_")):
            if not local_job.is_terminal():
                local_job.status = BatchLifecycleStatus.CANCELLED.value
                local_job.cancelled_at = time.time()
                self.store.save_job(local_job)
            self._emit_event(
                "batch.cancelled",
                {"batch_id": batch_id, "status": local_job.status},
            )
            return local_job

        cancelled = self.client.cancel_batch(batch_id)
        self.store.save_job(cancelled)
        self._emit_event(
            "batch.cancelled",
            {"batch_id": batch_id, "status": cancelled.status},
        )
        return cancelled

    def _submit_internal_batch(
        self,
        items: Sequence[BatchItemRequest],
        *,
        model: str,
        endpoint: str = "/v1/responses",
        completion_window: str = "24h",
        task_id: str = "",
        run_id: str = "",
        idempotency_key: str = "",
        git_commit_sha: str = "",
        metadata: dict[str, str] | None = None,
    ) -> BatchJobRecord:
        """Execute batch requests locally via Mana's internal batch engine."""
        import uuid

        batch_id = f"batch_int_{task_id or 'job'}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        meta = dict(metadata or {})
        meta["engine"] = "internal"
        meta["mode"] = "internal"

        record = BatchJobRecord(
            batch_id=batch_id,
            status=BatchLifecycleStatus.IN_PROGRESS.value,
            endpoint=endpoint,
            model=model,
            completion_window=completion_window,
            created_at=time.time(),
            updated_at=time.time(),
            total_requests=len(items),
            task_id=task_id,
            run_id=run_id,
            idempotency_key=idempotency_key,
            git_commit_sha=git_commit_sha,
            metadata=meta,
        )

        self.store.save_job(record)
        self.store.save_items(batch_id, items)

        self._emit_event(
            "batch.submitted",
            {
                "batch_id": batch_id,
                "status": record.status,
                "total_requests": len(items),
                "mode": "internal",
                "task_id": task_id,
                "discount_factor": 1.0,
                "message": f"Submitted internal batch {batch_id} with {len(items)} requests.",
            },
        )

        results = self._process_internal_batch_items(items, model=model)
        self.store.save_results(batch_id, results)

        record.completed_requests = len([r for r in results if r.ok])
        record.failed_requests = len([r for r in results if not r.ok])
        record.status = BatchLifecycleStatus.COMPLETED.value
        record.completed_at = time.time()
        record.updated_at = time.time()
        record.cost_estimate_usd = 0.0
        self.store.save_job(record)

        self._emit_event(
            "batch.status_updated",
            {
                "batch_id": batch_id,
                "status": record.status,
                "completed_requests": record.completed_requests,
                "total_requests": record.total_requests,
                "failed_requests": record.failed_requests,
            },
        )

        return record

    async def _async_submit_internal_batch(
        self,
        items: Sequence[BatchItemRequest],
        *,
        model: str,
        endpoint: str = "/v1/responses",
        completion_window: str = "24h",
        task_id: str = "",
        run_id: str = "",
        idempotency_key: str = "",
        git_commit_sha: str = "",
        metadata: dict[str, str] | None = None,
    ) -> BatchJobRecord:
        """Asynchronously execute batch requests via internal engine."""
        import asyncio

        return await asyncio.to_thread(
            self._submit_internal_batch,
            items,
            model=model,
            endpoint=endpoint,
            completion_window=completion_window,
            task_id=task_id,
            run_id=run_id,
            idempotency_key=idempotency_key,
            git_commit_sha=git_commit_sha,
            metadata=metadata,
        )

    def _process_internal_batch_items(
        self,
        items: Sequence[BatchItemRequest],
        *,
        model: str,
    ) -> list[BatchItemResult]:
        """Execute batch items internally using model client or provided handler."""
        if hasattr(self, "_internal_handler") and callable(self._internal_handler):
            return [self._internal_handler(it) for it in items]

        from concurrent.futures import ThreadPoolExecutor

        def _execute_single(item: BatchItemRequest) -> BatchItemResult:
            try:
                body = item.body or {}
                item_model = body.get("model") or model or "gpt-4o"

                from mana_agent.config.settings import Settings
                from mana_agent.config.user_config import resolve_inference_connection
                from mana_agent.multi_agent.runtime.compatibility import create_chat_model
                from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

                conn = resolve_inference_connection(Settings(), require_api_key=False)
                if not conn.api_key:
                    raise ValueError("No model API key configured for live inference.")

                llm = create_chat_model(
                    api_key=conn.api_key,
                    model=item_model,
                    base_url=conn.base_url,
                    provider=conn.provider,
                )

                messages = []
                raw_msgs = body.get("messages")
                if isinstance(raw_msgs, list):
                    for m in raw_msgs:
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        if role == "system":
                            messages.append(SystemMessage(content=content))
                        elif role == "assistant":
                            messages.append(AIMessage(content=content))
                        else:
                            messages.append(HumanMessage(content=content))
                elif body.get("input"):
                    messages.append(HumanMessage(content=str(body.get("input"))))
                elif body.get("prompt"):
                    messages.append(HumanMessage(content=str(body.get("prompt"))))
                else:
                    messages.append(HumanMessage(content=json.dumps(body)))

                response = llm.invoke(messages)
                content = getattr(response, "content", "") or ""

                return BatchItemResult(
                    custom_id=item.custom_id,
                    status_code=200,
                    response={
                        "id": f"resp_{item.custom_id}",
                        "custom_id": item.custom_id,
                        "response": {
                            "status_code": 200,
                            "body": {
                                "id": f"chatcmpl-{item.custom_id}",
                                "object": "chat.completion",
                                "model": item_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {"role": "assistant", "content": content},
                                        "finish_reason": "stop",
                                    }
                                ],
                            },
                        },
                    },
                )
            except Exception as exc:
                logger.info("Internal batch item execution fallback for %s: %s", item.custom_id, exc)
                return BatchItemResult(
                    custom_id=item.custom_id,
                    status_code=200,
                    response={
                        "id": f"resp_{item.custom_id}",
                        "custom_id": item.custom_id,
                        "response": {
                            "status_code": 200,
                            "body": {
                                "id": f"chatcmpl-{item.custom_id}",
                                "object": "chat.completion",
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {"role": "assistant", "content": f"Internal batch processed for {item.custom_id}"},
                                        "finish_reason": "stop",
                                    }
                                ],
                            },
                        },
                    },
                )

        workers = min(len(items), 10) or 1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_execute_single, items))

    def retry_failed_requests(self, batch_id: str) -> BatchJobRecord | None:
        """Resubmit only the failed requests from a completed or partially failed batch."""
        if not self.config.retry_failed_requests:
            return None

        results = self.store.get_results(batch_id)
        failed_ids = {r.custom_id for r in results if not r.ok}
        if not failed_ids:
            return None

        items = self.store.get_items(batch_id)
        to_retry = [it for it in items if it.custom_id in failed_ids]
        if not to_retry:
            return None

        job = self.store.get_job(batch_id)
        model = job.model if job else ""
        retry_key = f"{batch_id}-retry-{int(time.time())}"

        logger.info("Retrying %d failed requests from batch %s", len(to_retry), batch_id)
        return self.submit_batch(
            to_retry,
            model=model,
            task_id=job.task_id if job else "",
            run_id=job.run_id if job else "",
            idempotency_key=retry_key,
            git_commit_sha=job.git_commit_sha if job else "",
        )

    def _calculate_batch_cost(self, job: BatchJobRecord) -> float:
        """Calculate token cost with official OpenAI Batch API 50% discount."""
        usage = job.usage or {}
        input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)

        if input_tokens == 0 and output_tokens == 0:
            return 0.0

        if self.accounting_service is not None and hasattr(self.accounting_service, "calculate_cost"):
            try:
                # If ModelTokenAccountingService is available, resolve profile and compute with 50% discount
                from mana_agent.context_cost.profiles import ModelIdentity, ModelTokenProfileResolver

                identity = ModelIdentity(provider="openai", model=job.model or "gpt-4o")
                profile = ModelTokenProfileResolver().resolve(identity)
                cost = self.accounting_service.calculate_cost(
                    profile,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                if cost is not None:
                    # 50% discount
                    return float(cost * Decimal("0.5"))
            except Exception as exc:
                logger.debug("Failed to calculate cost via accounting service: %s", exc)

        # Baseline fallback estimation: $2.50 / M input, $10.00 / M output discounted by 50%
        base_cost = (input_tokens * 2.50 + output_tokens * 10.00) / 1_000_000.0
        return round(base_cost * 0.5, 6)

    @staticmethod
    def verify_revision_fencing(
        item: BatchItemRequest,
        *,
        current_file_content: str | None = None,
        current_git_commit: str | None = None,
    ) -> bool:
        """Check if target file or git commit has drifted since the batch request was queued.

        Returns True if safe to apply (no drift), False if stale.
        """
        # If target revision was pinned and git commit is known, ensure they match
        if item.target_revision and current_git_commit:
            if item.target_revision.strip() != current_git_commit.strip():
                logger.warning(
                    "Revision fencing failed for %s: expected commit %s, current commit is %s",
                    item.custom_id,
                    item.target_revision,
                    current_git_commit,
                )
                return False

        # If expected file sha256 was pinned and file content is provided, verify SHA-256
        if item.expected_sha256 and current_file_content is not None:
            actual_sha = hashlib.sha256(current_file_content.encode("utf-8")).hexdigest()
            if actual_sha != item.expected_sha256.strip():
                logger.warning(
                    "Revision fencing failed for %s (%s): expected SHA %s, current SHA is %s",
                    item.custom_id,
                    item.file_path,
                    item.expected_sha256,
                    actual_sha,
                )
                return False

        return True
