"""Unit tests for BatchJobStore and BatchLifecycleManager."""

from __future__ import annotations

import hashlib
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mana_agent.batch.client import OpenAIBatchClient
from mana_agent.batch.manager import BatchLifecycleManager
from mana_agent.batch.models import (
    BatchConfig,
    BatchItemRequest,
    BatchItemResult,
    BatchJobRecord,
    BatchLifecycleStatus,
)
from mana_agent.batch.store import BatchJobStore


def test_store_job_crud_and_idempotency() -> None:
    store = BatchJobStore(":memory:")

    job = BatchJobRecord(
        batch_id="batch-001",
        status=BatchLifecycleStatus.CREATED.value,
        endpoint="/v1/responses",
        model="gpt-4o",
        idempotency_key="idemp-key-1",
        task_id="task-123",
        total_requests=5,
    )
    store.save_job(job)

    fetched = store.get_job("batch-001")
    assert fetched is not None
    assert fetched.batch_id == "batch-001"
    assert fetched.status == BatchLifecycleStatus.CREATED.value
    assert fetched.idempotency_key == "idemp-key-1"
    assert fetched.total_requests == 5

    # Check idempotency lookup
    found = store.find_by_idempotency_key("idemp-key-1")
    assert found is not None
    assert found.batch_id == "batch-001"

    assert store.find_by_idempotency_key("non-existent") is None

    # Update job
    job.status = BatchLifecycleStatus.IN_PROGRESS.value
    job.completed_requests = 3
    store.save_job(job)

    updated = store.get_job("batch-001")
    assert updated is not None
    assert updated.status == BatchLifecycleStatus.IN_PROGRESS.value
    assert updated.completed_requests == 3


def test_store_items_and_results() -> None:
    store = BatchJobStore(":memory:")
    batch_id = "batch-items-test"

    items = [
        BatchItemRequest(custom_id="req-1", body={"input": "test 1"}, file_path="foo.py"),
        BatchItemRequest(custom_id="req-2", body={"input": "test 2"}, file_path="bar.py"),
    ]
    store.save_items(batch_id, items)

    loaded_items = store.get_items(batch_id)
    assert len(loaded_items) == 2
    assert {it.custom_id for it in loaded_items} == {"req-1", "req-2"}

    # Save results
    results = [
        BatchItemResult(custom_id="req-1", status_code=200, response_body={"output_text": "hello"}),
        BatchItemResult(custom_id="req-2", status_code=400, error={"message": "fail"}),
    ]
    store.save_results(batch_id, results)

    loaded_results = store.get_results(batch_id)
    assert len(loaded_results) == 2
    res_map = {r.custom_id: r for r in loaded_results}
    assert res_map["req-1"].ok
    assert res_map["req-1"].response_body["output_text"] == "hello"
    assert not res_map["req-2"].ok
    assert res_map["req-2"].error["message"] == "fail"


def test_manager_submit_and_idempotency_dedup() -> None:
    mock_client = MagicMock(spec=OpenAIBatchClient)
    mock_client.config = BatchConfig()
    mock_client.upload_batch_file.return_value = "file-input-1"
    mock_client.create_batch.return_value = BatchJobRecord(
        batch_id="batch_real_1",
        status=BatchLifecycleStatus.VALIDATING.value,
        endpoint="/v1/responses",
        input_file_id="file-input-1",
    )

    events: list[tuple[str, dict]] = []
    store = BatchJobStore(":memory:")
    manager = BatchLifecycleManager(
        client=mock_client,
        store=store,
        event_sink=lambda et, d: events.append((et, d)),
    )

    items = [
        BatchItemRequest(custom_id="i1", body={"input": "a"}),
        BatchItemRequest(custom_id="i2", body={"input": "b"}),
    ]

    # First submission
    record1 = manager.submit_batch(
        items,
        model="gpt-4o",
        task_id="t-100",
        idempotency_key="unique-token-xyz",
    )
    assert record1.batch_id == "batch_real_1"
    assert record1.status == BatchLifecycleStatus.VALIDATING.value
    assert mock_client.upload_batch_file.call_count == 1
    assert mock_client.create_batch.call_count == 1

    # Second submission with same idempotency key returns existing job without calling API again
    record2 = manager.submit_batch(
        items,
        model="gpt-4o",
        task_id="t-100",
        idempotency_key="unique-token-xyz",
    )
    assert record2.batch_id == "batch_real_1"
    assert mock_client.upload_batch_file.call_count == 1
    assert mock_client.create_batch.call_count == 1


def test_manager_poll_and_auto_finalize() -> None:
    mock_client = MagicMock(spec=OpenAIBatchClient)
    mock_client.config = BatchConfig()
    store = BatchJobStore(":memory:")

    # Initial job in store
    store.save_job(
        BatchJobRecord(
            batch_id="batch_poll_test",
            status=BatchLifecycleStatus.IN_PROGRESS.value,
            model="gpt-4o",
            task_id="t-poll",
        )
    )

    # When polled, OpenAI API returns status='completed' with output_file_id
    mock_client.retrieve_batch.return_value = BatchJobRecord(
        batch_id="batch_poll_test",
        status=BatchLifecycleStatus.COMPLETED.value,
        output_file_id="file-out-99",
        usage={"prompt_tokens": 1000, "completion_tokens": 500},
    )

    mock_client.download_results.return_value = [
        BatchItemResult(custom_id="c1", status_code=200, response_body={"output_text": "done"}),
    ]

    manager = BatchLifecycleManager(client=mock_client, store=store)
    result = manager.poll_batch("batch_poll_test")

    # Should have automatically finalized to RESULTS_IMPORTED
    assert result.status == BatchLifecycleStatus.RESULTS_IMPORTED.value
    assert result.cost_estimate_usd > 0
    # Usage was 1000 in, 500 out -> baseline cost discounted by 50%
    expected_cost = round(((1000 * 2.50 + 500 * 10.00) / 1_000_000.0) * 0.5, 6)
    assert result.cost_estimate_usd == expected_cost

    saved = store.get_job("batch_poll_test")
    assert saved is not None
    assert saved.status == BatchLifecycleStatus.RESULTS_IMPORTED.value


def test_manager_resume_pending_batches() -> None:
    mock_client = MagicMock(spec=OpenAIBatchClient)
    mock_client.config = BatchConfig()
    store = BatchJobStore(":memory:")

    # 2 pending jobs and 1 terminal job
    store.save_job(BatchJobRecord(batch_id="b1", status=BatchLifecycleStatus.IN_PROGRESS.value))
    store.save_job(BatchJobRecord(batch_id="b2", status=BatchLifecycleStatus.VALIDATING.value))
    store.save_job(BatchJobRecord(batch_id="b3", status=BatchLifecycleStatus.COMPLETED.value))

    mock_client.retrieve_batch.side_effect = lambda b_id: BatchJobRecord(
        batch_id=b_id,
        status=BatchLifecycleStatus.IN_PROGRESS.value,
    )

    manager = BatchLifecycleManager(client=mock_client, store=store)
    reconciled = manager.resume_pending_batches()

    assert len(reconciled) == 2
    assert {r.batch_id for r in reconciled} == {"b1", "b2"}


def test_revision_fencing_verification() -> None:
    content = "def add(a, b):\n    return a + b\n"
    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    commit = "a1b2c3d4e5f6"

    item = BatchItemRequest(
        custom_id="i1",
        body={"input": "test"},
        file_path="math.py",
        expected_sha256=sha,
        target_revision=commit,
    )

    # Content and commit match -> True
    assert BatchLifecycleManager.verify_revision_fencing(
        item,
        current_file_content=content,
        current_git_commit=commit,
    )

    # Content changed -> False (drift detected)
    drifted_content = "def add(a, b):\n    return a + b + 1\n"
    assert not BatchLifecycleManager.verify_revision_fencing(
        item,
        current_file_content=drifted_content,
        current_git_commit=commit,
    )

    # Commit changed -> False
    assert not BatchLifecycleManager.verify_revision_fencing(
        item,
        current_file_content=content,
        current_git_commit="999999999999",
    )


def test_lifecycle_with_real_sdk_batch_models() -> None:
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
        "metadata": {"task_id": "live-task-01", "idempotency_key": "live-idemp-01"},
        "request_counts": {"total": 2, "completed": 0, "failed": 0},
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
    }
    batch_validating = Batch.model_validate(data)

    data_in_prog = dict(data, status="in_progress")
    batch_in_prog = Batch.model_validate(data_in_prog)

    data_cancelled = dict(data, status="cancelled")
    batch_cancelled = Batch.model_validate(data_cancelled)

    mock_sdk = MagicMock()
    mock_sdk.files.create.return_value = SimpleNamespace(id="file-123")
    mock_sdk.batches.create.return_value = batch_validating
    mock_sdk.batches.retrieve.return_value = batch_in_prog
    mock_sdk.batches.cancel.return_value = batch_cancelled

    client = OpenAIBatchClient(client=mock_sdk)
    store = BatchJobStore(":memory:")
    manager = BatchLifecycleManager(client=client, store=store)

    items = [
        BatchItemRequest(custom_id="c1", body={"model": "gpt-4o-mini", "input": "test 1"}),
        BatchItemRequest(custom_id="c2", body={"model": "gpt-4o-mini", "input": "test 2"}),
    ]

    # 1. Submit
    rec = manager.submit_batch(
        items,
        model="gpt-4o-mini",
        task_id="live-task-01",
        idempotency_key="live-idemp-01",
    )
    assert rec.batch_id == "batch_6ad664f8b64c819087e566052d9a83fd"
    assert rec.status == BatchLifecycleStatus.VALIDATING.value
    assert rec.output_file_id == ""

    # 2. Poll
    polled = manager.poll_batch(rec.batch_id)
    assert polled.status == BatchLifecycleStatus.IN_PROGRESS.value

    # 3. Cancel
    cancelled = manager.cancel_batch(rec.batch_id)
    assert cancelled.status == BatchLifecycleStatus.CANCELLED.value
