"""Unit tests for BatchExecutionRouter."""

from __future__ import annotations

import pytest

from mana_agent.batch.models import (
    BatchConfig,
    BatchExecutionMode,
    BatchItemRequest,
)
from mana_agent.batch.router import BatchExecutionRouter, is_model_supported_for_batch


def test_model_support_check() -> None:
    assert is_model_supported_for_batch("gpt-5")
    assert is_model_supported_for_batch("gpt-5.4")
    assert is_model_supported_for_batch("gpt-4o")
    assert is_model_supported_for_batch("gpt-4o-mini")
    assert is_model_supported_for_batch("o1")
    assert is_model_supported_for_batch("o1-mini")
    assert is_model_supported_for_batch("gpt-4-turbo")
    assert is_model_supported_for_batch("gpt-3.5-turbo")
    assert is_model_supported_for_batch("text-embedding-3-small")

    # Unsupported models
    assert not is_model_supported_for_batch("gpt-4o-realtime-preview")
    assert not is_model_supported_for_batch("whisper-1")
    assert not is_model_supported_for_batch("dall-e-3")


def test_router_disabled_by_config() -> None:
    router = BatchExecutionRouter(BatchConfig(enabled=False))
    items = [
        BatchItemRequest(custom_id="1", body={"input": "a"}),
        BatchItemRequest(custom_id="2", body={"input": "b"}),
    ]
    decision = router.route(items)
    assert decision.mode == BatchExecutionMode.REALTIME
    assert "disabled" in decision.reason.lower()
    assert decision.discount_factor == 1.0


def test_router_explicit_realtime_mode() -> None:
    router = BatchExecutionRouter(BatchConfig(mode="realtime"))
    items = [
        BatchItemRequest(custom_id="1", body={"input": "a"}),
        BatchItemRequest(custom_id="2", body={"input": "b"}),
    ]
    decision = router.route(items)
    assert decision.mode == BatchExecutionMode.REALTIME
    assert "realtime" in decision.reason.lower()


def test_router_provider_constraint() -> None:
    router = BatchExecutionRouter(BatchConfig())
    items = [
        BatchItemRequest(custom_id="1", body={"input": "a"}),
        BatchItemRequest(custom_id="2", body={"input": "b"}),
    ]
    decision = router.route(items, provider="anthropic")
    assert decision.mode == BatchExecutionMode.REALTIME
    assert "anthropic" in decision.reason.lower()


def test_router_unsupported_model() -> None:
    router = BatchExecutionRouter(BatchConfig())
    items = [
        BatchItemRequest(custom_id="1", body={"model": "gpt-4o-realtime-preview", "input": "a"}),
    ]
    decision = router.route(items, model="gpt-4o-realtime-preview")
    assert decision.mode == BatchExecutionMode.REALTIME
    assert "not supported" in decision.reason.lower()


def test_router_always_enable_if_supported() -> None:
    router = BatchExecutionRouter(BatchConfig(always_enable_if_supported=True, min_requests=5))
    # Even a single request with supported model routes to batch
    items = [
        BatchItemRequest(custom_id="1", body={"model": "gpt-4o-mini", "input": "a"}, url="/v1/responses"),
    ]
    decision = router.route(items, model="gpt-4o-mini")
    assert decision.mode == BatchExecutionMode.BATCH
    assert decision.is_batch
    assert decision.discount_factor == 0.5
    assert "enabled" in decision.reason.lower()


def test_router_realtime_constraints() -> None:
    router = BatchExecutionRouter(BatchConfig(always_enable_if_supported=False))
    items = [
        BatchItemRequest(custom_id="1", body={"input": "a"}),
        BatchItemRequest(custom_id="2", body={"input": "b"}),
    ]

    # Streaming required
    decision_stream = router.route(items, requires_streaming=True)
    assert decision_stream.mode == BatchExecutionMode.REALTIME
    assert "streaming" in decision_stream.reason.lower()

    # Dependencies between requests
    decision_deps = router.route(items, has_intra_dependencies=True)
    assert decision_deps.mode == BatchExecutionMode.REALTIME
    assert "dependencies" in decision_deps.reason.lower()


def test_router_below_min_requests_threshold_when_always_enable_is_false() -> None:
    router = BatchExecutionRouter(BatchConfig(always_enable_if_supported=False, min_requests=3))
    items = [
        BatchItemRequest(custom_id="1", body={"input": "a"}),
        BatchItemRequest(custom_id="2", body={"input": "b"}),
    ]
    decision = router.route(items)
    assert decision.mode == BatchExecutionMode.REALTIME
    assert "min_requests threshold" in decision.reason


def test_router_eligible_workload_routes_to_batch() -> None:
    router = BatchExecutionRouter(BatchConfig(min_requests=2))
    items = [
        BatchItemRequest(custom_id="1", body={"input": "a"}, url="/v1/responses"),
        BatchItemRequest(custom_id="2", body={"input": "b"}, url="/v1/responses"),
        BatchItemRequest(custom_id="3", body={"input": "c"}, url="/v1/responses"),
    ]
    decision = router.route(items)
    assert decision.mode == BatchExecutionMode.BATCH
    assert decision.is_batch
    assert decision.discount_factor == 0.5
    assert len(decision.eligible_items) == 3
    assert len(decision.ineligible_items) == 0


def test_router_mode_override_forces_batch() -> None:
    router = BatchExecutionRouter(BatchConfig(always_enable_if_supported=False, min_requests=5))
    items = [BatchItemRequest(custom_id="1", body={"input": "a"}, url="/v1/responses")]
    # Force batch even below min_requests when always_enable is false
    decision = router.route(items, mode_override="batch")
    assert decision.mode == BatchExecutionMode.BATCH
    assert decision.discount_factor == 0.5
