"""Unit tests for BatchExecutionRouter."""

from __future__ import annotations

import pytest

from mana_agent.batch.models import (
    BatchConfig,
    BatchExecutionMode,
    BatchItemRequest,
)
from mana_agent.batch.router import (
    BatchExecutionRouter,
    is_model_supported_for_batch,
    parse_batch_support_from_metadata,
)


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


def test_parse_batch_support_from_metadata_dict() -> None:
    # Explicit boolean flags
    assert parse_batch_support_from_metadata({"batch_supported": True}) is True
    assert parse_batch_support_from_metadata({"batch_supported": False}) is False
    assert parse_batch_support_from_metadata({"supports_batch": True}) is True
    assert parse_batch_support_from_metadata({"supports_batch_api": "enabled"}) is True
    assert parse_batch_support_from_metadata({"batch_api_supported": "unavailable"}) is False
    assert parse_batch_support_from_metadata(
        {"metadata": {"supports_batch_api": True}}
    ) is True

    # Features dictionary
    assert parse_batch_support_from_metadata({"features": {"Batch API": "Supported"}}) is True
    assert parse_batch_support_from_metadata({"features": {"Batch": "Not supported"}}) is False
    assert parse_batch_support_from_metadata({"features": ["batch", "streaming"]}) is True
    assert parse_batch_support_from_metadata({"features": ["Batch API (not supported)"]}) is False

    # Endpoints list
    assert parse_batch_support_from_metadata({"endpoints": ["/v1/batches", "/v1/responses"]}) is True
    assert parse_batch_support_from_metadata({"supported_endpoints": ["/v1/responses"]}) is True
    assert parse_batch_support_from_metadata(
        {"endpoints": ["https://api.example.test/v1/responses"]}
    ) is True
    assert parse_batch_support_from_metadata({"endpoints": ["/v1/realtime"]}) is False
    # Endpoint matching must not accept a similarly named, unsupported route.
    assert parse_batch_support_from_metadata(
        {"endpoints": ["/v1/responses-preview"]}, endpoint="/v1/responses"
    ) is False


def test_parse_batch_support_from_metadata_text_card() -> None:
    raw_card_supported = """
Modalities
Text
Input and output
Endpoints
v1/batches
v1/responses
Features
Batch API
Supported
"""
    assert parse_batch_support_from_metadata(raw_card_supported) is True

    raw_card_unsupported = """
Features
Batch
Not supported
"""
    assert parse_batch_support_from_metadata(raw_card_unsupported) is False


def test_model_support_check_with_metadata() -> None:
    # Custom unknown model verified via passed metadata card
    card_ok = "Endpoints\nv1/batches\nFeatures\nBatch: Supported\n"
    assert is_model_supported_for_batch("custom-fine-tuned-model", metadata=card_ok) is True

    card_bad = "Features\nBatch: Not supported\n"
    assert is_model_supported_for_batch("custom-fine-tuned-model", metadata=card_bad) is False

    # Router using item metadata
    router = BatchExecutionRouter(BatchConfig(min_requests=1))
    items_ok = [
        BatchItemRequest(
            custom_id="1",
            body={"model": "custom-special", "metadata": {"batch_supported": True}},
            url="/v1/responses",
        )
    ]
    dec_ok = router.route(items_ok)
    assert dec_ok.mode == BatchExecutionMode.BATCH

    # Request metadata is still considered when the caller also passes model=.
    dec_explicit_model = router.route(items_ok, model="custom-special")
    assert dec_explicit_model.mode == BatchExecutionMode.BATCH

    items_bad = [
        BatchItemRequest(
            custom_id="1",
            body={"model": "custom-special", "metadata": {"batch_supported": False}},
            url="/v1/responses",
        )
    ]
    dec_bad = router.route(items_bad)
    assert dec_bad.mode == BatchExecutionMode.REALTIME
    assert "not supported" in dec_bad.reason.lower()

    dec_explicit_model_bad = router.route(items_bad, model="custom-special")
    assert dec_explicit_model_bad.mode == BatchExecutionMode.REALTIME
