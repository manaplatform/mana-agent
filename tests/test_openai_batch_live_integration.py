"""Live integration tests for OpenAI Batch API against real endpoint.

Requires OPENAI_API_KEY and MANA_ENABLE_LIVE_OPENAI_BATCH_TEST=1.
"""

from __future__ import annotations

import os
import time
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
    endpoint = os.getenv("MANA_BATCH_TEST_ENDPOINT", "/v1/responses")
    config = BatchConfig(endpoint=endpoint, completion_window="24h")
    client = OpenAIBatchClient(config=config)
    store = BatchJobStore(":memory:")
    manager = BatchLifecycleManager(client=client, store=store, config=config)

    model = (
        os.getenv("MANA_BATCH_TEST_MODEL")
        or os.getenv("MANA_PRIMARY_MODEL")
        or os.getenv("MANA_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-6-luna"
    )
    model = model.removeprefix("openai/")

    body_1 = (
        {"model": model, "input": "Reply with 'pong 1'"}
        if endpoint == "/v1/responses"
        else {"model": model, "messages": [{"role": "user", "content": "Reply with 'pong 1'"}]}
    )
    body_2 = (
        {"model": model, "input": "Reply with 'pong 2'"}
        if endpoint == "/v1/responses"
        else {"model": model, "messages": [{"role": "user", "content": "Reply with 'pong 2'"}]}
    )

    items = [
        BatchItemRequest(
            custom_id="live-test-1",
            body=body_1,
            url=endpoint,
        ),
        BatchItemRequest(
            custom_id="live-test-2",
            body=body_2,
            url=endpoint,
        ),
    ]

    record = manager.submit_batch(
        items,
        mode="openai",
        model=model,
        endpoint=endpoint,
        task_id=f"live-task-luna-{int(time.time())}",
        idempotency_key=f"live-idemp-luna-{int(time.time())}",
    )

    assert record.batch_id.startswith("batch_")
    assert record.status in {
        BatchLifecycleStatus.VALIDATING.value,
        BatchLifecycleStatus.IN_PROGRESS.value,
    }

    # Retrieve live status
    polled = manager.poll_batch(record.batch_id)
    assert polled.batch_id == record.batch_id

    # Do not cancel by default so the batch actually processes and is usable on OpenAI.
    # Cancel only if MANA_BATCH_CANCEL_LIVE=1 is explicitly requested.
    if os.getenv("MANA_BATCH_CANCEL_LIVE", "0") == "1":
        cancelled = manager.cancel_batch(record.batch_id)
        assert cancelled.status in {
            BatchLifecycleStatus.CANCELLING.value,
            BatchLifecycleStatus.CANCELLED.value,
        }
