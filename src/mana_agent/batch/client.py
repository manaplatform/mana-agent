"""Native client for OpenAI Batch API (/v1/batches and /v1/files)."""

from __future__ import annotations

import io
import json
import logging
import os
import time
from collections.abc import Sequence
from typing import Any

from mana_agent.batch.models import (
    BatchConfig,
    BatchItemRequest,
    BatchItemResult,
    BatchJobRecord,
    BatchLifecycleStatus,
)

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB
MAX_REQUESTS_PER_BATCH = 50_000


def _extract_field(obj: Any, key: str, default: Any = None) -> Any:
    """Safely extract field from a Pydantic object, class instance, or dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        val = obj.get(key)
        return val if val is not None else default
    val = getattr(obj, key, None)
    return val if val is not None else default


def _extract_dict(obj: Any) -> dict[str, Any]:
    """Safely convert a model, mapping, or object into a python dictionary."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    res: dict[str, Any] = {}
    for k in dir(obj):
        if not k.startswith("_"):
            v = getattr(obj, k, None)
            if not callable(v):
                res[k] = v
    return res


class OpenAIBatchClient:
    """Client for uploading, creating, monitoring, and retrieving OpenAI Batch jobs."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        async_client: Any | None = None,
        config: BatchConfig | None = None,
    ) -> None:
        self.config = config or BatchConfig.from_settings()
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")

        if not self._api_key:
            try:
                from mana_agent.config.inference_provider import resolve_inference_connection
                from mana_agent.config.settings import Settings

                settings = Settings()
                if getattr(settings, "openai_api_key", None):
                    self._api_key = str(settings.openai_api_key).strip()
                if not self._base_url and getattr(settings, "openai_base_url", None):
                    self._base_url = str(settings.openai_base_url).strip()

                if not self._api_key:
                    conn = resolve_inference_connection(settings, provider="openai", require_api_key=False)
                    if conn.api_key:
                        self._api_key = conn.api_key
                    if not self._base_url and conn.base_url:
                        self._base_url = conn.base_url
            except Exception as exc:
                logger.debug("Failed to resolve OpenAI inference connection for batch client: %s", exc)

        self._client = client
        self._async_client = async_client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import openai

        if not self._api_key:
            raise ValueError("OpenAI Batch requires an OpenAI platform API key")
        kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = openai.OpenAI(**kwargs)
        return self._client

    def _get_async_client(self) -> Any:
        if self._async_client is not None:
            return self._async_client
        import openai

        if not self._api_key:
            raise ValueError("OpenAI Batch requires an OpenAI platform API key")
        kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._async_client = openai.AsyncOpenAI(**kwargs)
        return self._async_client

    def serialize_requests_to_jsonl(self, items: Sequence[BatchItemRequest]) -> bytes:
        """Validate and serialize item requests into JSONL bytes conforming to OpenAI Batch spec."""
        if not items:
            raise ValueError("Cannot create a batch with 0 requests.")
        if len(items) > self.config.max_requests_per_batch:
            raise ValueError(
                f"Batch exceeds maximum limit of {self.config.max_requests_per_batch} requests: got {len(items)}"
            )

        seen_custom_ids: set[str] = set()
        lines: list[str] = []

        for item in items:
            if not item.custom_id or not str(item.custom_id).strip():
                raise ValueError("Every batch request must have a non-empty 'custom_id'.")
            if item.custom_id in seen_custom_ids:
                raise ValueError(f"Duplicate custom_id found in batch request: {item.custom_id}")
            seen_custom_ids.add(item.custom_id)

            row = item.to_jsonl_row()
            lines.append(json.dumps(row, ensure_ascii=False))

        raw_bytes = ("\n".join(lines) + "\n").encode("utf-8")
        max_bytes = self.config.max_file_size_mb * 1024 * 1024
        if len(raw_bytes) > max_bytes:
            raise ValueError(
                f"Batch file exceeds size limit of {self.config.max_file_size_mb} MB (got {len(raw_bytes) / 1024 / 1024:.2f} MB)"
            )

        return raw_bytes

    def upload_batch_file(self, items: Sequence[BatchItemRequest], *, filename: str = "batch_requests.jsonl") -> str:
        """Upload batch requests to OpenAI Files API with purpose='batch'."""
        raw_bytes = self.serialize_requests_to_jsonl(items)
        client = self._get_client()

        file_obj = client.files.create(
            file=(filename, io.BytesIO(raw_bytes), "application/jsonl"),
            purpose="batch",
        )
        file_id = _extract_field(file_obj, "id", "")
        logger.info("Uploaded OpenAI batch file %s (%d bytes, %d requests)", file_id, len(raw_bytes), len(items))
        return str(file_id)

    async def async_upload_batch_file(self, items: Sequence[BatchItemRequest], *, filename: str = "batch_requests.jsonl") -> str:
        """Asynchronously upload batch requests to OpenAI Files API with purpose='batch'."""
        raw_bytes = self.serialize_requests_to_jsonl(items)
        client = self._get_async_client()

        file_obj = await client.files.create(
            file=(filename, io.BytesIO(raw_bytes), "application/jsonl"),
            purpose="batch",
        )
        file_id = _extract_field(file_obj, "id", "")
        return str(file_id)

    def create_batch(
        self,
        input_file_id: str,
        *,
        endpoint: str = "/v1/responses",
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
    ) -> BatchJobRecord:
        """Submit a new batch job to OpenAI Batch API."""
        client = self._get_client()
        batch = client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata or {},
        )
        return self._convert_sdk_batch(batch)

    async def async_create_batch(
        self,
        input_file_id: str,
        *,
        endpoint: str = "/v1/responses",
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
    ) -> BatchJobRecord:
        """Asynchronously submit a new batch job to OpenAI Batch API."""
        client = self._get_async_client()
        batch = await client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata or {},
        )
        return self._convert_sdk_batch(batch)

    def retrieve_batch(self, batch_id: str) -> BatchJobRecord:
        """Retrieve real-time status of an OpenAI Batch job."""
        client = self._get_client()
        batch = client.batches.retrieve(batch_id)
        return self._convert_sdk_batch(batch)

    async def async_retrieve_batch(self, batch_id: str) -> BatchJobRecord:
        """Asynchronously retrieve real-time status of an OpenAI Batch job."""
        client = self._get_async_client()
        batch = await client.batches.retrieve(batch_id)
        return self._convert_sdk_batch(batch)

    def cancel_batch(self, batch_id: str) -> BatchJobRecord:
        """Cancel an in-progress or queued OpenAI Batch job."""
        client = self._get_client()
        batch = client.batches.cancel(batch_id)
        return self._convert_sdk_batch(batch)

    async def async_cancel_batch(self, batch_id: str) -> BatchJobRecord:
        """Asynchronously cancel an in-progress or queued OpenAI Batch job."""
        client = self._get_async_client()
        batch = await client.batches.cancel(batch_id)
        return self._convert_sdk_batch(batch)

    def list_batches(self, limit: int = 20) -> list[BatchJobRecord]:
        """List recent batches from OpenAI API."""
        client = self._get_client()
        response = client.batches.list(limit=limit)
        items = getattr(response, "data", response)
        return [self._convert_sdk_batch(b) for b in items]

    def download_results(self, output_file_id: str) -> list[BatchItemResult]:
        """Download output file content and parse each JSONL result indexed by custom_id."""
        client = self._get_client()
        resp = client.files.content(output_file_id)
        content = resp.text if hasattr(resp, "text") else (resp.read().decode("utf-8") if hasattr(resp, "read") else str(resp))
        return self._parse_jsonl_results(content)

    async def async_download_results(self, output_file_id: str) -> list[BatchItemResult]:
        """Asynchronously download output file content and parse each JSONL result."""
        client = self._get_async_client()
        resp = await client.files.content(output_file_id)
        text = resp.text if hasattr(resp, "text") else (resp.read().decode("utf-8") if hasattr(resp, "read") else str(resp))
        return self._parse_jsonl_results(text)

    def download_errors(self, error_file_id: str) -> list[dict[str, Any]]:
        """Download error file content and parse request-level failure records."""
        client = self._get_client()
        resp = client.files.content(error_file_id)
        content = resp.text if hasattr(resp, "text") else (resp.read().decode("utf-8") if hasattr(resp, "read") else str(resp))
        return [json.loads(line) for line in content.splitlines() if line.strip()]

    def _parse_jsonl_results(self, raw_text: str) -> list[BatchItemResult]:
        results: list[BatchItemResult] = []
        for line in raw_text.splitlines():
            clean = line.strip()
            if not clean:
                continue
            try:
                row = json.loads(clean)
                results.append(BatchItemResult.from_jsonl_row(row))
            except Exception as exc:
                raise ValueError("Malformed Batch result JSONL") from exc
        return results

    def _convert_sdk_batch(self, batch: Any) -> BatchJobRecord:
        """Normalize an OpenAI SDK Batch object or dictionary into BatchJobRecord."""
        b_id = _extract_field(batch, "id", "")
        endpoint = _extract_field(batch, "endpoint", "/v1/responses")
        status_raw = _extract_field(batch, "status", "created")
        input_file_id = _extract_field(batch, "input_file_id", "")
        output_file_id = _extract_field(batch, "output_file_id", "")
        error_file_id = _extract_field(batch, "error_file_id", "")
        window = _extract_field(batch, "completion_window", "24h")
        created_at = _extract_field(batch, "created_at", time.time())
        completed_at = _extract_field(batch, "completed_at")
        expires_at = _extract_field(batch, "expires_at")
        metadata = _extract_dict(_extract_field(batch, "metadata"))
        usage = _extract_dict(_extract_field(batch, "usage"))

        counts = _extract_field(batch, "request_counts")
        total = _extract_field(counts, "total", 0) or 0
        completed = _extract_field(counts, "completed", 0) or 0
        failed = _extract_field(counts, "failed", 0) or 0

        # Map status to canonical BatchLifecycleStatus
        status_map = {
            "validating": BatchLifecycleStatus.VALIDATING.value,
            "failed": BatchLifecycleStatus.FAILED.value,
            "in_progress": BatchLifecycleStatus.IN_PROGRESS.value,
            "finalizing": BatchLifecycleStatus.FINALIZING.value,
            "completed": BatchLifecycleStatus.COMPLETED.value,
            "expired": BatchLifecycleStatus.EXPIRED.value,
            "cancelling": BatchLifecycleStatus.CANCELLING.value,
            "cancelled": BatchLifecycleStatus.CANCELLED.value,
        }
        status = status_map.get(str(status_raw).lower(), str(status_raw).lower())

        return BatchJobRecord(
            batch_id=str(b_id or ""),
            status=status,
            endpoint=str(endpoint or "/v1/responses"),
            input_file_id=str(input_file_id or ""),
            output_file_id=str(output_file_id or ""),
            error_file_id=str(error_file_id or ""),
            completion_window=str(window or "24h"),
            created_at=float(created_at or time.time()),
            updated_at=time.time(),
            completed_at=float(completed_at) if completed_at is not None else None,
            expires_at=float(expires_at) if expires_at is not None else None,
            total_requests=int(total),
            completed_requests=int(completed),
            failed_requests=int(failed),
            task_id=str(metadata.get("task_id") or ""),
            run_id=str(metadata.get("run_id") or ""),
            metadata={str(k): str(v) for k, v in metadata.items()},
            usage=usage,
        )
