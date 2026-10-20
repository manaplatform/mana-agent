"""Unit tests for OpenAIBatchClient."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mana_agent.batch.client import OpenAIBatchClient
from mana_agent.batch.models import (
    BatchConfig,
    BatchItemRequest,
    BatchLifecycleStatus,
)


def test_serialize_requests_to_jsonl_valid() -> None:
    client = OpenAIBatchClient(config=BatchConfig())
    items = [
        BatchItemRequest(custom_id="req-1", body={"model": "gpt-4o", "input": "test 1"}),
        BatchItemRequest(custom_id="req-2", body={"model": "gpt-4o", "input": "test 2"}),
    ]

    raw_bytes = client.serialize_requests_to_jsonl(items)
    lines = raw_bytes.decode("utf-8").strip().split("\n")
    assert len(lines) == 2

    row1 = json.loads(lines[0])
    assert row1["custom_id"] == "req-1"
    assert row1["method"] == "POST"
    assert row1["url"] == "/v1/responses"
    assert row1["body"]["input"] == "test 1"

    row2 = json.loads(lines[1])
    assert row2["custom_id"] == "req-2"


def test_serialize_requests_empty_raises() -> None:
    client = OpenAIBatchClient(config=BatchConfig())
    with pytest.raises(ValueError, match="Cannot create a batch with 0 requests"):
        client.serialize_requests_to_jsonl([])


def test_serialize_requests_duplicate_custom_id_raises() -> None:
    client = OpenAIBatchClient(config=BatchConfig())
    items = [
        BatchItemRequest(custom_id="same-id", body={"input": "a"}),
        BatchItemRequest(custom_id="same-id", body={"input": "b"}),
    ]
    with pytest.raises(ValueError, match="Duplicate custom_id"):
        client.serialize_requests_to_jsonl(items)


def test_serialize_requests_empty_custom_id_raises() -> None:
    client = OpenAIBatchClient(config=BatchConfig())
    items = [BatchItemRequest(custom_id="", body={"input": "a"})]
    with pytest.raises(ValueError, match="non-empty 'custom_id'"):
        client.serialize_requests_to_jsonl(items)


def test_serialize_requests_exceeds_size_limit() -> None:
    client = OpenAIBatchClient(config=BatchConfig(max_file_size_mb=0))  # 0 MB limit
    items = [BatchItemRequest(custom_id="req-1", body={"input": "hello"})]
    with pytest.raises(ValueError, match="exceeds size limit"):
        client.serialize_requests_to_jsonl(items)


def test_upload_batch_file_calls_files_api() -> None:
    mock_sdk = MagicMock()
    mock_sdk.files.create.return_value = SimpleNamespace(id="file-batch-123")

    client = OpenAIBatchClient(client=mock_sdk)
    items = [BatchItemRequest(custom_id="r1", body={"input": "data"})]

    file_id = client.upload_batch_file(items)
    assert file_id == "file-batch-123"

    mock_sdk.files.create.assert_called_once()
    _, kwargs = mock_sdk.files.create.call_args
    assert kwargs["purpose"] == "batch"


def test_create_batch_calls_batches_api() -> None:
    mock_sdk = MagicMock()
    mock_sdk.batches.create.return_value = {
        "id": "batch_abc",
        "endpoint": "/v1/responses",
        "status": "validating",
        "input_file_id": "file-123",
        "completion_window": "24h",
        "created_at": 1700000000,
        "metadata": {"task_id": "t-1"},
    }

    client = OpenAIBatchClient(client=mock_sdk)
    record = client.create_batch("file-123", endpoint="/v1/responses", metadata={"task_id": "t-1"})

    assert record.batch_id == "batch_abc"
    assert record.status == BatchLifecycleStatus.VALIDATING.value
    assert record.task_id == "t-1"
    mock_sdk.batches.create.assert_called_once_with(
        input_file_id="file-123",
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"task_id": "t-1"},
    )


def test_retrieve_and_cancel_batch() -> None:
    mock_sdk = MagicMock()
    mock_sdk.batches.retrieve.return_value = {
        "id": "batch_xyz",
        "status": "in_progress",
        "request_counts": {"total": 10, "completed": 7, "failed": 1},
    }
    mock_sdk.batches.cancel.return_value = {
        "id": "batch_xyz",
        "status": "cancelled",
    }

    client = OpenAIBatchClient(client=mock_sdk)
    rec = client.retrieve_batch("batch_xyz")
    assert rec.status == BatchLifecycleStatus.IN_PROGRESS.value
    assert rec.total_requests == 10
    assert rec.completed_requests == 7
    assert rec.failed_requests == 1

    cancelled = client.cancel_batch("batch_xyz")
    assert cancelled.status == BatchLifecycleStatus.CANCELLED.value


def test_download_results_out_of_order_parsing() -> None:
    mock_sdk = MagicMock()
    # OpenAI Batch output lines arrive in arbitrary order
    jsonl_output = (
        json.dumps({
            "custom_id": "item-2",
            "response": {"status_code": 200, "body": {"output_text": "result 2"}},
        })
        + "\n"
        + json.dumps({
            "custom_id": "item-1",
            "response": {"status_code": 200, "body": {"output_text": "result 1"}},
        })
        + "\n"
        + json.dumps({
            "custom_id": "item-3",
            "error": {"code": "rate_limit_exceeded", "message": "Too many requests"},
            "response": {"status_code": 429},
        })
    )

    mock_sdk.files.content.return_value = SimpleNamespace(text=jsonl_output)

    client = OpenAIBatchClient(client=mock_sdk)
    results = client.download_results("file-output-999")

    assert len(results) == 3
    res_map = {r.custom_id: r for r in results}

    assert res_map["item-1"].ok
    assert res_map["item-1"].response_body["output_text"] == "result 1"

    assert res_map["item-2"].ok
    assert res_map["item-2"].response_body["output_text"] == "result 2"

    assert not res_map["item-3"].ok
    assert res_map["item-3"].status_code == 429
    assert res_map["item-3"].error["code"] == "rate_limit_exceeded"


def test_convert_sdk_pydantic_batch_object() -> None:
    from openai.types import Batch

    data = {
        "id": "batch_6ad664f8b64c819087e566052d9a83fd",
        "completion_window": "24h",
        "created_at": 1792435448,
        "endpoint": "/v1/responses",
        "input_file_id": "file-123",
        "object": "batch",
        "status": "validating",
        "output_file_id": None,
        "error_file_id": None,
        "metadata": {"task_id": "live-task-01"},
        "request_counts": {"total": 2, "completed": 0, "failed": 0},
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
    }
    batch = Batch.model_validate(data)
    client = OpenAIBatchClient()
    record = client._convert_sdk_batch(batch)

    assert record.batch_id == "batch_6ad664f8b64c819087e566052d9a83fd"
    assert record.status == BatchLifecycleStatus.VALIDATING.value
    assert record.output_file_id == ""
    assert record.error_file_id == ""
    assert record.total_requests == 2
    assert record.task_id == "live-task-01"
    assert record.usage["input_tokens"] == 0
