"""BatchExecutionService coordinating OpenAI Files and Batches API interactions."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from mana_agent.batch.builder import BatchRequestBuilder
from mana_agent.batch.client import OpenAIBatchClient
from mana_agent.batch.models import (
    BatchConfig,
    BatchItemRequest,
    BatchJobRecord,
)

logger = logging.getLogger(__name__)


class BatchExecutionService:
    """Official OpenAI Batch API execution service using official SDK endpoints."""

    def __init__(
        self,
        client: OpenAIBatchClient | None = None,
        builder: BatchRequestBuilder | None = None,
        config: BatchConfig | None = None,
    ) -> None:
        self.config = config or BatchConfig()
        self.client = client or OpenAIBatchClient(config=self.config)
        self.builder = builder or BatchRequestBuilder()

    def upload_requests_file(
        self,
        items: Sequence[BatchItemRequest],
        *,
        filename: str = "batch_requests.jsonl",
    ) -> str:
        """Serialize items to JSONL and upload to POST /v1/files with purpose='batch'."""
        raw_bytes = self.builder.validate_and_serialize_jsonl(items)
        client = self.client._get_client()

        import io

        file_obj = client.files.create(
            file=(filename, io.BytesIO(raw_bytes), "application/jsonl"),
            purpose=self.builder.limits.upload_purpose,
        )
        file_id = getattr(file_obj, "id", None) or (file_obj.get("id") if isinstance(file_obj, dict) else "")
        logger.info("Uploaded OpenAI batch file %s (%d requests, %d bytes)", file_id, len(items), len(raw_bytes))
        return str(file_id)

    async def async_upload_requests_file(
        self,
        items: Sequence[BatchItemRequest],
        *,
        filename: str = "batch_requests.jsonl",
    ) -> str:
        """Asynchronously upload to POST /v1/files with purpose='batch'."""
        raw_bytes = self.builder.validate_and_serialize_jsonl(items)
        client = self.client._get_async_client()

        import io

        file_obj = await client.files.create(
            file=(filename, io.BytesIO(raw_bytes), "application/jsonl"),
            purpose=self.builder.limits.upload_purpose,
        )
        file_id = getattr(file_obj, "id", None) or (file_obj.get("id") if isinstance(file_obj, dict) else "")
        return str(file_id)

    def create_batch(
        self,
        input_file_id: str,
        *,
        endpoint: str = "/v1/responses",
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
    ) -> BatchJobRecord:
        """Create and submit a batch job via POST /v1/batches."""
        return self.client.create_batch(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata,
        )

    async def async_create_batch(
        self,
        input_file_id: str,
        *,
        endpoint: str = "/v1/responses",
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
    ) -> BatchJobRecord:
        """Asynchronously create batch via POST /v1/batches."""
        return await self.client.async_create_batch(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata,
        )

    def get_batch(self, batch_id: str) -> BatchJobRecord:
        """Retrieve batch status via GET /v1/batches/{batch_id}."""
        return self.client.retrieve_batch(batch_id)

    async def async_get_batch(self, batch_id: str) -> BatchJobRecord:
        """Asynchronously retrieve status via GET /v1/batches/{batch_id}."""
        return await self.client.async_retrieve_batch(batch_id)

    def list_batches(self, limit: int = 20) -> list[BatchJobRecord]:
        """List batches via GET /v1/batches."""
        return self.client.list_batches(limit=limit)

    async def async_list_batches(self, limit: int = 20) -> list[BatchJobRecord]:
        """Asynchronously list batches via GET /v1/batches."""
        return await self.client.async_list_batches(limit=limit)

    def cancel_batch(self, batch_id: str) -> BatchJobRecord:
        """Cancel batch via POST /v1/batches/{batch_id}/cancel."""
        return self.client.cancel_batch(batch_id)

    async def async_cancel_batch(self, batch_id: str) -> BatchJobRecord:
        """Asynchronously cancel batch via POST /v1/batches/{batch_id}/cancel."""
        return await self.client.async_cancel_batch(batch_id)

    def download_file_content(self, file_id: str) -> str:
        """Download raw file content via GET /v1/files/{file_id}/content."""
        client = self.client._get_client()
        resp = client.files.content(file_id)
        if hasattr(resp, "text"):
            return resp.text
        if hasattr(resp, "read"):
            content = resp.read()
            return content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
        return str(resp)

    async def async_download_file_content(self, file_id: str) -> str:
        """Asynchronously download raw file content via GET /v1/files/{file_id}/content."""
        client = self.client._get_async_client()
        resp = await client.files.content(file_id)
        if hasattr(resp, "text"):
            return resp.text
        if hasattr(resp, "read"):
            content = await resp.read()
            return content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
        return str(resp)


__all__ = [
    "BatchExecutionService",
]
