"""Batch result processor correlating results by custom_id and enforcing revision fencing."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from mana_agent.batch.models import (
    BatchItemRequest,
    BatchItemResult,
    BatchJobRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessedBatchResults:
    """Outcome of processing output and error files for a batch job."""

    batch_id: str
    total_results: int
    successful_results: list[BatchItemResult]
    failed_results: list[BatchItemResult]
    results_by_custom_id: dict[str, BatchItemResult]
    estimated_cost_usd: float = 0.0
    discount_factor: float = 0.5


class BatchResultProcessor:
    """Parses, correlates, and validates OpenAI Batch API results."""

    def parse_output_jsonl(self, jsonl_content: str) -> list[BatchItemResult]:
        """Parse raw output JSONL text into structured BatchItemResult instances.

        Matches results strictly by custom_id, never JSONL line order.
        """
        results: list[BatchItemResult] = []
        if not jsonl_content or not jsonl_content.strip():
            return results

        for line in jsonl_content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception as exc:
                raise ValueError("Malformed Batch output JSONL") from exc

            results.append(BatchItemResult.from_jsonl_row(data))

        return results

    def parse_error_jsonl(self, jsonl_content: str) -> list[BatchItemResult]:
        """Parse raw error JSONL text into structured BatchItemResult instances."""
        error_results: list[BatchItemResult] = []
        if not jsonl_content or not jsonl_content.strip():
            return error_results

        for line in jsonl_content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception as exc:
                raise ValueError("Malformed Batch error JSONL") from exc

            error_results.append(BatchItemResult.from_jsonl_row(data))

        return error_results

    def correlate_results(
        self,
        *,
        batch_id: str,
        output_jsonl: str = "",
        error_jsonl: str = "",
        original_items: Sequence[BatchItemRequest] | None = None,
    ) -> ProcessedBatchResults:
        """Correlate all returned rows with original requests by custom_id."""
        outputs = self.parse_output_jsonl(output_jsonl)
        errors = self.parse_error_jsonl(error_jsonl)

        results_by_id: dict[str, BatchItemResult] = {}
        for r in outputs:
            if not r.custom_id or r.custom_id in results_by_id:
                raise ValueError("Empty or duplicate custom_id in Batch output")
            if r.custom_id:
                results_by_id[r.custom_id] = r
        for e in errors:
            if not e.custom_id or e.custom_id in results_by_id:
                raise ValueError("Empty or duplicate custom_id in Batch errors")
            if e.custom_id:
                results_by_id[e.custom_id] = e

        if original_items is not None:
            expected = {item.custom_id for item in original_items}
            if set(results_by_id) != expected:
                raise ValueError("Batch output custom_id set does not match original requests")
        successful = [r for r in results_by_id.values() if r.ok]
        failed = [r for r in results_by_id.values() if not r.ok]

        return ProcessedBatchResults(
            batch_id=batch_id,
            total_results=len(results_by_id),
            successful_results=successful,
            failed_results=failed,
            results_by_custom_id=results_by_id,
            discount_factor=0.5,
        )

    def extract_failed_items_for_retry(
        self,
        original_items: Sequence[BatchItemRequest],
        processed: ProcessedBatchResults,
    ) -> list[BatchItemRequest]:
        """Idempotently extract only failed or missing requests for targeted retry.

        Never resubmits requests that succeeded in the initial run.
        """
        successful_cids = {r.custom_id for r in processed.successful_results}
        retry_items: list[BatchItemRequest] = []

        for item in original_items:
            if item.custom_id not in successful_cids:
                retry_items.append(item)

        logger.info(
            "Batch %s retry filter: %d/%d requests eligible for retry (successful excluded)",
            processed.batch_id,
            len(retry_items),
            len(original_items),
        )
        return retry_items

    @staticmethod
    def verify_revision_fencing(
        item: BatchItemRequest,
        *,
        current_file_content: str | None = None,
        current_git_commit: str = "",
    ) -> bool:
        """Verify that workspace state has not drifted since the batch request was created."""
        import hashlib

        if item.expected_sha256 and current_file_content is None:
            return False
        if item.target_revision and not current_git_commit:
            return False
        if item.expected_sha256 and current_file_content is not None:
            actual_sha = hashlib.sha256(current_file_content.encode("utf-8")).hexdigest()
            if actual_sha != item.expected_sha256:
                logger.warning(
                    "Revision fencing: file '%s' sha256 mismatch (expected %s, got %s)",
                    item.file_path,
                    item.expected_sha256[:12],
                    actual_sha[:12],
                )
                return False

        if item.target_revision and current_git_commit:
            if item.target_revision != current_git_commit:
                logger.warning(
                    "Revision fencing: git commit mismatch (expected %s, current %s)",
                    item.target_revision[:8],
                    current_git_commit[:8],
                )
                return False

        return True


__all__ = [
    "BatchResultProcessor",
    "ProcessedBatchResults",
]
