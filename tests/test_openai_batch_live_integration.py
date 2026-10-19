"""Live integration tests for OpenAI Batch API against real endpoint.

Requires OPENAI_API_KEY and MANA_ENABLE_LIVE_OPENAI_BATCH_TEST=1.
"""

from __future__ import annotations

import os
import pytest

from mana_agent.batch.client import OpenAIBatchClient
from mana_agent.batch.manager import BatchLifecycleManager
from mana_agent.batch.models import (
    BatchConfig,
    BatchItemRequest,
    BatchLifecycleStatus,
)
from mana_agent.batch.store import BatchJobStore

LIVE_BATCH_ENABLED = bool(
    os.getenv("MANA_ENABLE_LIVE_OPENAI_BATCH_TEST") and os.getenv("OPENAI_API_KEY")
)


@pytest.mark.skipif(
    not LIVE_BATCH_ENABLED,
    reason="Live OpenAI Batch API tests require MANA_ENABLE_LIVE_OPENAI_BATCH_TEST=1 and OPENAI_API_KEY.",
)
def test_live_openai_batch_submission_and_status() -> None:
    config = BatchConfig(endpoint="/v1/responses", completion_window="24h")
    client = OpenAIBatchClient(config=config)
    store = BatchJobStore(":memory:")
    manager = BatchLifecycleManager(client=client, store=store, config=config)

    items = [
        BatchItemRequest(
            custom_id="live-test-1",
            body={"model": "gpt-4o-mini", "input": "Reply with 'pong 1'"},
            url="/v1/responses",
        ),
        BatchItemRequest(
            custom_id="live-test-2",
            body={"model": "gpt-4o-mini", "input": "Reply with 'pong 2'"},
            url="/v1/responses",
        ),
    ]

    record = manager.submit_batch(
        items,
        model="gpt-4o-mini",
        task_id="live-task-01",
        idempotency_key="live-idemp-01",
    )

    assert record.batch_id.startswith("batch_")
    assert record.status in {
        BatchLifecycleStatus.VALIDATING.value,
        BatchLifecycleStatus.IN_PROGRESS.value,
    }

    # Retrieve live status
    polled = manager.poll_batch(record.batch_id)
    assert polled.batch_id == record.batch_id

    # Cancel to avoid consuming unnecessary quota
    cancelled = manager.cancel_batch(record.batch_id)
    assert cancelled.status in {
        BatchLifecycleStatus.CANCELLING.value,
        BatchLifecycleStatus.CANCELLED.value,
    }
