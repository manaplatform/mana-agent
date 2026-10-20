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
        if completion_window != "24h":
            raise ValueError("OpenAI Batch completion_window must be 24h")

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
            effective_mode = "internal"
        if effective_mode == "internal":
            return self._submit_internal_batch(
                items, model=model, endpoint=endpoint, completion_window=completion_window,
                task_id=task_id, run_id=run_id, idempotency_key=idempotency_key,
                git_commit_sha=git_commit_sha, metadata={**meta, "provider": provider},
            )
        if effective_mode not in {"openai", "batch"}:
            raise ValueError("Batch submission mode must be openai or explicitly internal")
        from mana_agent.batch.policy import current_execution_policy
        selected_policy = current_execution_policy()
        if selected_policy is not None and not selected_policy.required:
            raise ValueError("This task has no explicit OpenAI Batch requirement; remote submission is not authorized")
        from mana_agent.batch.validation import validate_batch_selection
        from mana_agent.batch.operations import BatchOperationStore
        validate_batch_selection(self.config, model=model, provider=provider, endpoint=endpoint, items=items)
        if isinstance(self.client, OpenAIBatchClient):
            self.client._get_client()  # Validate credentials before claiming a submission intent; no network call.
        operations = BatchOperationStore(self.store)
        fingerprint_payload = items[0].body if len(items) == 1 else [it.to_jsonl_row() for it in items]
        fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if idempotency_key:
            claimed = operations.prepare(idempotency_key, fingerprint, task_id, {
                "items": [it.to_jsonl_row() for it in items], "model": model, "metadata": meta,
            })
            if not claimed:
                operation = operations.get(idempotency_key)
                if operation and operation["batch_id"]:
                    existing = self.store.get_job(operation["batch_id"])
                    if existing:
                        return existing
                matches = [job for job in self.client.list_batches(limit=100)
                           if job.metadata.get("idempotency_key") == idempotency_key]
                if len(matches) != 1:
                    raise RuntimeError("Batch submission outcome is uncertain; reconcile the original operation before retrying")
                recovered = matches[0]
                recovered.model, recovered.task_id = model, task_id
                recovered.idempotency_key = idempotency_key
                recovered.total_requests = len(items)
                self.store.save_job(recovered)
                self.store.save_items(recovered.batch_id, items)
                operations.update(idempotency_key, state="submitted", batch_id=recovered.batch_id)
                return recovered

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

        try:
            # 1. Upload JSONL requests file
            filename = f"batch_{task_id or 'run'}_{int(time.time())}.jsonl"
            input_file_id = self.client.upload_batch_file(items, filename=filename)

            if idempotency_key:
                operations.update(idempotency_key, state="uploaded", input_file_id=input_file_id)
            # 2. Create batch job
            created_record = self.client.create_batch(
                input_file_id=input_file_id,
                endpoint=endpoint,
                completion_window=completion_window or self.config.completion_window,
                metadata=meta,
            )
        except Exception as exc:
            from mana_agent.utils.redaction import redact_secrets
            detail = str(redact_secrets(str(exc)))
            logger.error("OpenAI Batch submission failed on OpenAI platform: %s", detail)
            raise RuntimeError(f"OpenAI Batch API creation failed on OpenAI platform: {detail}") from exc

        # Populate local tracking fields
        created_record.metadata.update(meta)
        created_record.model = model
        created_record.task_id = task_id
        created_record.run_id = run_id
        created_record.idempotency_key = idempotency_key
        created_record.git_commit_sha = git_commit_sha
        created_record.total_requests = len(items)

        # 3. Persist job and items in SQLite store
        self.store.save_job(created_record)
        self.store.save_items(created_record.batch_id, items)
        if idempotency_key:
            operations.update(idempotency_key, state="submitted", batch_id=created_record.batch_id)

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
        """Use the same durable submission contract for sync and async callers."""
        import asyncio
        return await asyncio.to_thread(
            self.submit_batch, items, model=model, endpoint=endpoint,
            completion_window=completion_window, task_id=task_id, run_id=run_id,
            idempotency_key=idempotency_key, git_commit_sha=git_commit_sha,
            metadata=metadata, mode=mode, provider=provider,
        )

    def poll_batch(self, batch_id: str) -> BatchJobRecord:
        """Poll current batch status from OpenAI or SQLite store, and finalize if completed."""
        local_job = self.store.get_job(batch_id)
        if local_job and (local_job.metadata.get("engine") == "internal" or batch_id.startswith("batch_int_")):
            if local_job.status == BatchLifecycleStatus.COMPLETED.value:
                finalized_job, _ = self.finalize_batch(batch_id)
                return finalized_job
            return local_job

        if local_job and (local_job.status in {"results_imported", "verified"} or local_job.metadata.get("results_imported") == "true"):
            return local_job
        remote_job = self.client.retrieve_batch(batch_id)
        if local_job:
            remote_job.metadata = {**local_job.metadata, **remote_job.metadata}
            # Preserve local metadata
            remote_job.model = remote_job.model or local_job.model
            remote_job.task_id = remote_job.task_id or local_job.task_id
            remote_job.run_id = remote_job.run_id or local_job.run_id
            remote_job.idempotency_key = remote_job.idempotency_key or local_job.idempotency_key
            remote_job.git_commit_sha = remote_job.git_commit_sha or local_job.git_commit_sha

        self.store.save_job(remote_job)

        if local_job is None or (local_job.status, local_job.completed_requests, local_job.failed_requests) != (remote_job.status, remote_job.completed_requests, remote_job.failed_requests):
            self._emit_event(
                "batch.status_updated",
                {
                    "batch_id": remote_job.batch_id,
                    "task_id": remote_job.task_id,
                    "status": remote_job.status,
                    "completed_requests": remote_job.completed_requests,
                    "total_requests": remote_job.total_requests,
                    "failed_requests": remote_job.failed_requests,
                    "output_file_id": remote_job.output_file_id,
                },
            )

        if remote_job.status in {"completed", "failed", "expired", "cancelled"}:
            finalized_job, _ = self.finalize_batch(batch_id)
            return finalized_job

        return remote_job

    async def async_poll_batch(self, batch_id: str) -> BatchJobRecord:
        import asyncio
        return await asyncio.to_thread(self.poll_batch, batch_id)

    def finalize_batch(self, batch_id: str) -> tuple[BatchJobRecord, list[BatchItemResult]]:
        """Import only terminal, fully correlated results; import is not verification."""
        job = self.store.get_job(batch_id) or self.client.retrieve_batch(batch_id)
        if job.status in {"results_imported", "verified"} or job.metadata.get("results_imported") == "true":
            return job, self.store.get_results(batch_id)
        if job.status not in {"completed", "expired", "cancelled", "failed"}:
            raise RuntimeError(f"Batch {batch_id} is still {job.status}; results are not ready")
        if job.metadata.get("engine") == "internal":
            results = self.store.get_results(batch_id)
        else:
            if job.status == "completed" and not job.output_file_id and not job.error_file_id:
                refreshed = self.client.retrieve_batch(batch_id)
                job.output_file_id, job.error_file_id = refreshed.output_file_id, refreshed.error_file_id
            results = self.client.download_results(job.output_file_id) if job.output_file_id else []
            if job.error_file_id:
                results += [BatchItemResult.from_jsonl_row(row) for row in self.client.download_errors(job.error_file_id)]
        expected = {item.custom_id for item in self.store.get_items(batch_id)}
        actual = [result.custom_id for result in results]
        if len(actual) != len(set(actual)) or any(cid not in expected for cid in actual):
            raise ValueError("Batch output contains duplicate or unknown custom_id values")
        missing = expected - set(actual)
        if missing and job.status == "completed":
            raise ValueError("Completed Batch output is missing expected custom_id values")
        results += [BatchItemResult(custom_id=cid, status_code=400,
                    error={"code": job.status, "message": "No result returned before terminal batch status"}) for cid in missing]
        usage: dict[str, int] = {}
        for result in results:
            item_usage = result.response_body.get("usage") or {}
            for key in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] = usage.get(key, 0) + int(item_usage.get(key) or 0)
        if any(usage.values()):
            job.usage = usage
        job.cost_estimate_usd = self._calculate_batch_cost(job)
        job.metadata["results_imported"] = "true"
        if job.status == "completed":
            job.status = BatchLifecycleStatus.RESULTS_IMPORTED.value
        job.completed_at = job.completed_at or time.time()
        self.store.import_results(job, results)
        self._emit_event("batch.results_imported", {
            "batch_id": batch_id, "imported_count": len(results), "status": job.status,
            "task_id": job.task_id, "cost_usd": job.cost_estimate_usd,
        })
        return job, results

    async def async_finalize_batch(self, batch_id: str) -> tuple[BatchJobRecord, list[BatchItemResult]]:
        import asyncio
        return await asyncio.to_thread(self.finalize_batch, batch_id)

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
        if local_job:
            cancelled.metadata = {**local_job.metadata, **cancelled.metadata}
            for name in ("model", "task_id", "run_id", "idempotency_key", "git_commit_sha"):
                if not getattr(cancelled, name):
                    setattr(cancelled, name, getattr(local_job, name))
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

        results = self._process_internal_batch_items(items, model=model, provider=meta.get("provider", "openai"))
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
        provider: str = "openai",
    ) -> list[BatchItemResult]:
        """Execute batch items internally using model client or provided handler."""
        if hasattr(self, "_internal_handler") and callable(self._internal_handler):
            return [self._internal_handler(it) for it in items]

        from concurrent.futures import ThreadPoolExecutor

        def _execute_single(item: BatchItemRequest) -> BatchItemResult:
            try:
                body = item.body or {}
                item_model = body.get("model") or model
                if not item_model:
                    raise ValueError("Internal batch inference requires a selected model")
                unsupported = set(body) - {"model", "messages", "input", "prompt", "metadata"}
                if unsupported:
                    raise ValueError(f"Internal batch request options are unsupported: {sorted(unsupported)}; use the native tool loop for tool-bearing requests")

                from mana_agent.config.inference_provider import resolve_inference_connection
                from mana_agent.config.settings import Settings
                from mana_agent.multi_agent.runtime.compatibility import create_chat_model

                conn = resolve_inference_connection(Settings(), provider=provider, require_api_key=False)
                if not conn.api_key:
                    raise ValueError("No model API key configured for live inference.")

                llm = create_chat_model(
                    api_key=conn.api_key,
                    model=item_model,
                    base_url=conn.base_url,
                    provider=conn.provider,
                )

                from langchain_core.messages import convert_to_messages
                raw_messages = body.get("messages")
                if raw_messages is None:
                    raw_input = body.get("input", body.get("prompt"))
                    raw_messages = raw_input if isinstance(raw_input, list) else [{"role": "user", "content": raw_input}]
                if not raw_messages or any(not isinstance(message, dict) or message.get("content") is None for message in raw_messages):
                    raise ValueError("Internal batch requires valid conversation messages or text input")
                response = llm.invoke(convert_to_messages(raw_messages))
                content = getattr(response, "content", "") or ""
                if not content:
                    raise ValueError("Internal batch model returned no output")
                usage = getattr(response, "usage_metadata", None) or {}
                return BatchItemResult(custom_id=item.custom_id, status_code=200, response_body={
                    "id": f"resp_{item.custom_id}", "model": item_model, "usage": usage,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                })
            except Exception as exc:
                from mana_agent.utils.redaction import redact_secrets
                logger.warning("Internal batch item %s failed: %s", item.custom_id, type(exc).__name__)
                return BatchItemResult(custom_id=item.custom_id, status_code=500,
                    error={"code": "INTERNAL_BATCH_INFERENCE_FAILED", "message": str(redact_secrets(str(exc)))})

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
            mode="internal" if job and job.metadata.get("engine") == "internal" else "openai",
        )

    def _calculate_batch_cost(self, job: BatchJobRecord) -> float:
        """Use selected-model prices; the Batch discount applies only to official jobs."""
        usage = job.usage or {}
        input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)

        if input_tokens == 0 and output_tokens == 0:
            job.metadata["cost_estimate_known"] = "false"
            return 0.0

        from mana_agent.context_cost.profiles import ModelIdentity, ModelTokenProfileResolver
        try:
            profile = ModelTokenProfileResolver().resolve(ModelIdentity(provider=job.metadata.get("provider", "openai"), model=job.model or "gpt-4o"))
            if profile.input_price_per_million is not None and profile.output_price_per_million is not None:
                job.metadata["cost_estimate_known"] = "true"
                return float((Decimal(input_tokens) * profile.input_price_per_million
                    + Decimal(output_tokens) * profile.output_price_per_million)
                    * (Decimal("1") if job.metadata.get("engine") == "internal" else Decimal("0.5")) / Decimal(1_000_000))
        except ValueError:
            pass

        # Baseline fallback estimation: $2.50 / M input, $10.00 / M output discounted by 50%
        job.metadata["cost_estimate_known"] = "true"
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
        if item.target_revision and not current_git_commit:
            return False
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
        if item.expected_sha256 and current_file_content is None:
            return False
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
