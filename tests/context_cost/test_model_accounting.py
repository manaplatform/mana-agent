from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pytest

from mana_agent.context_cost.accounting import (
    ModelContextLimitError,
    ModelTokenAccountingService,
    TokenEstimationRequest,
)
from mana_agent.context_cost.profiles import (
    ModelIdentity,
    ModelTokenProfile,
    ModelTokenProfileResolver,
    UnknownModelProfileError,
)
from mana_agent.context_cost.store import AccountingStore
from mana_agent.context_cost.usage import normalize_provider_usage


def profile(
    model: str,
    *,
    context: int = 4_096,
    output: int = 1_024,
    input_price: Decimal | None = Decimal("2"),
    output_price: Decimal | None = Decimal("4"),
) -> ModelTokenProfile:
    return ModelTokenProfile(
        ModelIdentity("fixture", model),
        context_window=context,
        max_output_tokens=output,
        input_price_per_million=input_price,
        cached_input_price_per_million=Decimal("1") if input_price is not None else None,
        output_price_per_million=output_price,
        reasoning_price_per_million=Decimal("6") if output_price is not None else None,
        supports_usage_reporting=True,
        source="test-catalog",
        confidence="high",
    )


def service(tmp_path: Path, *profiles: ModelTokenProfile) -> ModelTokenAccountingService:
    return ModelTokenAccountingService(
        ModelTokenProfileResolver(profiles),
        store=AccountingStore(tmp_path / "accounting"),
        safety_margin_ratio=Decimal("0.05"),
    )


def request(model: str, **values) -> TokenEstimationRequest:
    defaults = {
        "model_identity": ModelIdentity("fixture", model),
        "components": {"messages": "hello world"},
        "route": "chat",
        "lane": "conversation",
        "requested_output_tokens": 64,
        "execution_kind": "standard_chat",
    }
    defaults.update(values)
    return TokenEstimationRequest(**defaults)


def test_model_context_output_pricing_and_explainable_components(tmp_path: Path) -> None:
    accounting = service(tmp_path, profile("small", context=512, output=128))
    estimate = accounting.estimate(request(
        "small",
        components={"messages": "hello", "tool_schemas": {"name": "search", "schema": "x" * 80}},
    ))
    assert estimate.profile.context_window == 512
    assert estimate.profile.max_output_tokens == 128
    assert estimate.components["tool_schemas"] > 0
    assert estimate.estimated_cost is not None
    assert estimate.estimated_cost == accounting.calculate_cost(
        estimate.profile, input_tokens=estimate.input_tokens, output_tokens=estimate.output_tokens
    )
    with pytest.raises(ModelContextLimitError, match="output capacity"):
        accounting.estimate(request("small", requested_output_tokens=129))


def test_multi_call_tool_prediction_and_effective_policy_limits(tmp_path: Path) -> None:
    accounting = service(tmp_path, profile("large", context=8_192, output=2_048))
    single = accounting.estimate(request("large", components={"messages": "x" * 200, "tool_schemas": "schema"}))
    multiple = accounting.estimate(request(
        "large",
        components={"messages": "x" * 200, "tool_schemas": "schema"},
        expected_model_calls=3,
        expected_tool_steps=2,
    ))
    assert multiple.input_tokens > single.input_tokens
    assert multiple.output_tokens == single.output_tokens * 3
    assert "follow_up_call_payload" in multiple.components
    with pytest.raises(ModelContextLimitError) as caught:
        accounting.estimate(request("large", lane_policy_limit=50))
    assert caught.value.deficit and caught.value.deficit > 0


def test_unknown_metadata_and_pricing_are_explicit(tmp_path: Path) -> None:
    strict = ModelTokenAccountingService(
        ModelTokenProfileResolver((), unknown_policy="require_metadata"),
        store=AccountingStore(tmp_path / "strict"),
    )
    with pytest.raises(UnknownModelProfileError):
        strict.estimate(request("missing"))

    unknown_price = service(
        tmp_path, profile("unpriced", input_price=None, output_price=None)
    ).estimate(request("unpriced"))
    assert unknown_price.estimated_cost is None


def test_reported_and_estimated_usage_reconcile_idempotently(tmp_path: Path) -> None:
    accounting = service(tmp_path, profile("usage"))
    reservation = accounting.reserve(request("usage"), operation_id="task:attempt:call")
    completed = accounting.reconcile(
        reservation,
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "total_tokens": 140,
            "prompt_tokens_details": {"cached_tokens": 20},
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
    )
    assert completed.actual is not None
    assert completed.actual.status == "reported"
    assert completed.actual.cached_input_tokens == 20
    assert completed.actual.reasoning_tokens == 5
    assert completed.actual.actual_cost is not None
    repeated = accounting.reconcile(reservation, usage={"total_tokens": 999})
    assert repeated.status == "reconciled"
    assert repeated.actual == completed.actual

    estimated = accounting.reserve(request("usage"), operation_id="missing-usage")
    completed_estimate = accounting.reconcile(
        estimated, final_components={"messages": "request", "response": "answer"}
    )
    assert completed_estimate.actual is not None
    assert completed_estimate.actual.status == "estimated"


def test_provider_usage_shapes_are_normalized() -> None:
    usage = normalize_provider_usage({
        "inputTokens": 10,
        "outputTokens": 4,
        "totalTokens": 14,
        "cachedInputTokens": 3,
        "reasoningTokens": 2,
    })
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens, usage.cached_input_tokens) == (10, 4, 3)


def test_fallback_model_retry_gets_fresh_reservation_and_release_is_audited(tmp_path: Path) -> None:
    accounting = service(tmp_path, profile("first"), profile("fallback"))
    first = accounting.reserve(request("first"), operation_id="attempt-1")
    fallback = accounting.reserve(request("fallback"), operation_id="attempt-2")
    assert first.reservation_id != fallback.reservation_id
    released = accounting.release(first, reason="provider failure")
    assert released.status == "released"
    assert accounting.store.get(first.reservation_id)["release_reason"] == "provider failure"


def test_parallel_duplicate_reservations_are_idempotent_and_private(tmp_path: Path) -> None:
    accounting = service(tmp_path, profile("parallel"))
    sensitive = "private-prompt-secret"
    estimation = request("parallel", components={"messages": sensitive})

    with ThreadPoolExecutor(max_workers=4) as pool:
        reservations = list(pool.map(
            lambda _: accounting.reserve(estimation, operation_id="same-operation"), range(8)
        ))

    assert len({item.reservation_id for item in reservations}) == 1
    persisted = accounting.store.get(reservations[0].reservation_id)
    assert persisted["estimate"]["components"]["messages"] > 0
    assert sensitive not in str(persisted)


def test_historical_prediction_uses_matching_reconciled_runs(tmp_path: Path) -> None:
    accounting = service(tmp_path, profile("history", context=16_384, output=4_096))
    first = accounting.reserve(request("history"), operation_id="historical-1")
    accounting.reconcile(first, usage={"input_tokens": 600, "output_tokens": 80, "total_tokens": 680})
    predicted = accounting.estimate(request("history"))
    assert predicted.input_tokens >= 600
    assert any("historical p80" in item for item in predicted.assumptions)
