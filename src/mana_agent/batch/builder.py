"""Batch request generator and validator conforming to OpenAI Batch API specifications."""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from mana_agent.batch.models import BatchItemRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchLimits:
    """Documented limits for OpenAI Batch API (versioned and updateable)."""

    max_requests_per_batch: int = 50_000
    max_file_size_mb: int = 200
    completion_window: str = "24h"
    upload_purpose: str = "batch"
    version: str = "2026.10.1"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


DEFAULT_BATCH_LIMITS = BatchLimits()


class BatchRequestBuilder:
    """Validates and builds JSONL payloads for the OpenAI Batch API."""

    def __init__(self, limits: BatchLimits | None = None) -> None:
        self.limits = limits or DEFAULT_BATCH_LIMITS

    def build_item(
        self,
        *,
        custom_id: str,
        body: dict[str, Any],
        url: str = "/v1/responses",
        method: str = "POST",
        model: str = "",
        origin_task_id: str = "",
        file_path: str = "",
        expected_sha256: str = "",
        target_revision: str = "",
    ) -> BatchItemRequest:
        """Create and validate a single batch request item."""
        cid = str(custom_id or "").strip()
        if not cid:
            raise ValueError("custom_id is required and cannot be empty.")

        item_body = dict(body)
        if model and "model" not in item_body:
            item_body["model"] = model

        return BatchItemRequest(
            custom_id=cid,
            body=item_body,
            method=method.upper(),
            url=url,
            origin_task_id=origin_task_id,
            file_path=file_path,
            expected_sha256=expected_sha256,
            target_revision=target_revision,
        )

    def group_by_model_and_endpoint(
        self, items: Sequence[BatchItemRequest]
    ) -> dict[tuple[str, str], list[BatchItemRequest]]:
        """Group request items by compatible target model and endpoint URL."""
        groups: dict[tuple[str, str], list[BatchItemRequest]] = {}
        for it in items:
            m = str((it.body or {}).get("model") or "unknown").strip()
            u = str(it.url or "/v1/responses").strip().lower().rstrip("/")
            key = (m, u)
            if key not in groups:
                groups[key] = []
            groups[key].append(it)
        return groups

    def validate_and_serialize_jsonl(self, items: Sequence[BatchItemRequest]) -> bytes:
        """Validate all items, ensure uniqueness of custom_id, and generate JSONL bytes."""
        if not items:
            raise ValueError("Cannot serialize an empty batch request list.")

        if len(items) > self.limits.max_requests_per_batch:
            raise ValueError(
                f"Batch exceeds maximum limit of {self.limits.max_requests_per_batch} requests (got {len(items)})."
            )

        seen_custom_ids: set[str] = set()
        lines: list[str] = []

        for idx, item in enumerate(items):
            cid = str(item.custom_id or "").strip()
            if not cid:
                raise ValueError(f"Batch item at index {idx} has an empty 'custom_id'.")
            if cid in seen_custom_ids:
                raise ValueError(f"Duplicate custom_id found in batch request: '{cid}'")
            seen_custom_ids.add(cid)

            # Ensure model is present in body
            if not isinstance(item.body, dict) or not item.body.get("model"):
                raise ValueError(f"Batch item '{cid}' must specify a 'model' inside the request body.")

            row = {
                "custom_id": cid,
                "method": item.method.upper(),
                "url": item.url,
                "body": item.body,
            }
            lines.append(json.dumps(row, ensure_ascii=False))

        raw_bytes = ("\n".join(lines) + "\n").encode("utf-8")
        if len(raw_bytes) > self.limits.max_file_size_bytes:
            size_mb = len(raw_bytes) / 1024 / 1024
            raise ValueError(
                f"Batch file exceeds size limit of {self.limits.max_file_size_mb} MB (got {size_mb:.2f} MB)."
            )

        return raw_bytes


__all__ = [
    "BatchLimits",
    "BatchRequestBuilder",
    "DEFAULT_BATCH_LIMITS",
]
