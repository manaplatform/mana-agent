from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
import pytest

from mana_agent.config.settings import Settings
from mana_agent.context_cost.accounting import (
    LaneBudgetExceededError,
    ModelContextExceededError,
    ModelContextLimitError,
    ModelTokenAccountingService,
    TaskBudgetExceededError,
    TaskReservationExceededError,
    TokenEstimationRequest,
    VerificationBudgetExceededError,
)
from mana_agent.context_cost.governor import ContextCostGovernor
from mana_agent.context_cost.models import (
    ContextBudgetExceeded,
    ContextSegment,
    GovernorMode,
)
from mana_agent.context_cost.profiles import (
    ModelIdentity,
    ModelTokenProfile,
    ModelTokenProfileResolver,
)
from mana_agent.context_cost.store import AccountingStore


def make_profile(
    model: str,
    *,
    context: int = 400_000,
    output: int = 32_768,
    input_price: Decimal | None = Decimal("2"),
    output_price: Decimal | None = Decimal("4"),
) -> ModelTokenProfile:
    return ModelTokenProfile(
        ModelIdentity("test_provider", model),
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


def test_turn_budget_overflow_impossible(tmp_path: Path) -> None:
    """Critical regression: turn_budget_tokens=100000, turn_consumed_tokens=1451498 must be impossible."""
    settings = Settings(
        mana_routing_task_token_budget=1_000_000,
        mana_context_cost_log_enabled=False,
    )
    governor = ContextCostGovernor(
        settings=settings,
        mode=GovernorMode.ENFORCE,
        session_id="test_session",
    )
    profile = make_profile("test-model", context=400_000, output=32_768)
    governor.register_model_profiles([profile])

    turn_id = "turn_overflow_check"
    governor.reset_turn_accounting(turn_id, allocated_tokens=100_000)

    # 1. First call uses 60,000 tokens (fits within 100,000 turn budget)
    segments_1 = [ContextSegment(kind="prompt", content="x" * 150_000, token_estimate=50_000, protected=True)]
    call_id_1, decision_1 = governor.before_model_call(
        segments_1,
        model="test-model",
        provider="test_provider",
        turn_id=turn_id,
        task_id="task_1",
        expected_output_tokens=10_000,
    )
    assert decision_1.allowed is True
    governor.record_model_call(
        call_id_1,
        usage={"input_tokens": 50_000, "output_tokens": 10_000, "total_tokens": 60_000},
        provider="test_provider",
        model="test-model",
        turn_id=turn_id,
        task_id="task_1",
    )

    snapshot_1 = governor.accounting_snapshot(task_id="task_1", turn_id=turn_id)
    assert snapshot_1.turn_consumed_tokens == 60_000
    assert snapshot_1.turn_remaining_tokens == 40_000

    # 2. Second call needs 50,000 tokens, which would push turn tokens to 110,000 > 100,000
    segments_2 = [ContextSegment(kind="prompt", content="x" * 120_000, token_estimate=40_000, protected=True)]
    with pytest.raises(ContextBudgetExceeded) as exc_info:
        governor.before_model_call(
            segments_2,
            model="test-model",
            provider="test_provider",
            turn_id=turn_id,
            task_id="task_1",
            expected_output_tokens=10_000,
        )
    assert exc_info.value.decision.reason == "turn_budget_exhausted"


def test_large_context_fits_configured_1m_task_budget(tmp_path: Path) -> None:
    """Critical regression: required=137842, old task ceiling=100000, model context=400000 fits 1M ceiling."""
    settings = Settings(
        mana_routing_task_token_budget=1_000_000,
        mana_context_cost_log_enabled=False,
    )
    governor = ContextCostGovernor(
        settings=settings,
        mode=GovernorMode.ENFORCE,
        session_id="test_session",
    )
    profile = make_profile("large-model", context=400_000, output=32_768)
    governor.register_model_profiles([profile])

    # Required: ~137,842 tokens
    segments = [ContextSegment(kind="prompt", content="x" * 390_000, token_estimate=130_000, protected=True)]
    call_id, decision = governor.before_model_call(
        segments,
        model="large-model",
        provider="test_provider",
        task_id="task_large",
        expected_output_tokens=7_842,
    )
    assert decision.allowed is True
    assert decision.snapshot.status == "ok"


def test_provider_call_vs_task_budget_separation(tmp_path: Path) -> None:
    """Provider-call admission compares against model context, task admission compares against 1M task ceiling."""
    service = ModelTokenAccountingService(
        ModelTokenProfileResolver([make_profile("small-model", context=50_000, output=4_000)]),
        store=AccountingStore(tmp_path / "acct_test"),
    )

    # 1. Provider call exceeding model context window (60,000 > 50,000) raises ModelContextExceededError
    with pytest.raises(ModelContextExceededError) as exc_info:
        service.estimate(TokenEstimationRequest(
            model_identity=ModelIdentity("test_provider", "small-model"),
            components={"prompt": "test"},
            component_token_overrides={"prompt": 60_000},
            task_token_remaining=1_000_000,
        ))
    assert exc_info.value.error_type == "model_context_exceeded"

    # 2. Provider call within model context (30,000 < 50,000) but exceeding task token remaining raises TaskBudgetExceededError
    with pytest.raises(TaskBudgetExceededError) as exc_info:
        service.estimate(TokenEstimationRequest(
            model_identity=ModelIdentity("test_provider", "small-model"),
            components={"prompt": "test"},
            component_token_overrides={"prompt": 30_000},
            task_token_remaining=20_000,
        ))
    assert exc_info.value.error_type == "task_budget_exceeded"


def test_reservation_lifecycle_and_idempotent_reconciliation(tmp_path: Path) -> None:
    """Reservation lifecycle: reserved -> reconciled / released / cancelled, idempotent across replays."""
    service = ModelTokenAccountingService(
        ModelTokenProfileResolver([make_profile("test-model", context=400_000, output=32_768)]),
        store=AccountingStore(tmp_path / "acct_store"),
    )
    req = TokenEstimationRequest(
        model_identity=ModelIdentity("test_provider", "test-model"),
        components={"prompt": "hello"},
        requested_output_tokens=100,
    )

    # Lifecycle 1: reserved -> reconciled
    res_1 = service.reserve(req, operation_id="op_1", task_id="task_1", attempt_id="att_1")
    assert res_1.status == "reserved"
    reconciled_1 = service.reconcile(
        res_1,
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    )
    assert reconciled_1.status == "reconciled"
    assert reconciled_1.actual is not None and reconciled_1.actual.total_tokens == 30

    # Idempotent re-reconciliation does not change or double-count
    reconciled_repeat = service.reconcile(
        res_1,
        usage={"input_tokens": 999, "output_tokens": 999, "total_tokens": 1998},
    )
    assert reconciled_repeat.status == "reconciled"
    assert reconciled_repeat.actual is not None and reconciled_repeat.actual.total_tokens == 30

    # Lifecycle 2: reserved -> released
    res_2 = service.reserve(req, operation_id="op_2", task_id="task_1", attempt_id="att_2")
    released_2 = service.release(res_2, reason="provider failed")
    assert released_2.status == "released"

    # Lifecycle 3: reserved -> cancelled
    res_3 = service.reserve(req, operation_id="op_3", task_id="task_1", attempt_id="att_3")
    cancelled_3 = service.cancel(res_3, reason="user abort")
    assert cancelled_3.status == "cancelled"


def test_reservation_revisions_enforce_atomic_task_invariant(tmp_path: Path) -> None:
    """Atomic invariant: task_consumed + task_reserved <= configured_task_budget."""
    settings = Settings(
        mana_routing_task_token_budget=100_000,
        mana_context_cost_log_enabled=False,
    )
    governor = ContextCostGovernor(
        settings=settings,
        mode=GovernorMode.ENFORCE,
        session_id="test_session",
    )
    profile = make_profile("test-model", context=400_000, output=32_768)
    governor.register_model_profiles([profile])

    segments = [ContextSegment(kind="prompt", content="x" * 60_000, token_estimate=20_000, protected=True)]
    call_id, decision = governor.before_model_call(
        segments,
        model="test-model",
        provider="test_provider",
        task_id="task_rev",
        expected_output_tokens=10_000,
    )
    assert decision.allowed is True

    # 1. Revise within budget (from ~31k to ~50k) -> succeeds
    new_segments_ok = [ContextSegment(kind="prompt", content="x" * 120_000, token_estimate=40_000, protected=True)]
    revised_ok = governor.revise_reservation(call_id, new_segments=new_segments_ok, expected_output_tokens=10_000)
    assert revised_ok.status == "reserved"

    # 2. Revise exceeding task budget (to ~110k > 100k) -> raises TaskReservationExceededError
    new_segments_overflow = [ContextSegment(kind="prompt", content="x" * 285_000, token_estimate=95_000, protected=True)]
    with pytest.raises(TaskReservationExceededError) as exc_info:
        governor.revise_reservation(call_id, new_segments=new_segments_overflow, expected_output_tokens=10_000)
    assert exc_info.value.error_type == "task_reservation_exceeded"


def test_multi_call_cumulative_task_budget_exhaustion(tmp_path: Path) -> None:
    """Cumulative multi-call execution correctly respects configured task budget limit."""
    settings = Settings(
        mana_routing_task_token_budget=100_000,
        mana_context_cost_log_enabled=False,
    )
    governor = ContextCostGovernor(
        settings=settings,
        mode=GovernorMode.ENFORCE,
        session_id="test_session",
    )
    profile = make_profile("test-model", context=400_000, output=32_768)
    governor.register_model_profiles([profile])

    # Call 1: consumes 60,000 tokens
    segments_1 = [ContextSegment(kind="prompt", content="x" * 150_000, token_estimate=50_000, protected=True)]
    call_1, _ = governor.before_model_call(
        segments_1,
        model="test-model",
        provider="test_provider",
        task_id="task_cum",
        expected_output_tokens=10_000,
    )
    governor.record_model_call(
        call_1,
        usage={"input_tokens": 50_000, "output_tokens": 10_000, "total_tokens": 60_000},
        provider="test_provider",
        model="test-model",
        task_id="task_cum",
    )

    # Call 2: requests 50,000 tokens (60k + 50k = 110k > 100k) -> blocked with task_budget_exceeded
    segments_2 = [ContextSegment(kind="prompt", content="x" * 120_000, token_estimate=40_000, protected=True)]
    with pytest.raises(ContextBudgetExceeded) as exc_info:
        governor.before_model_call(
            segments_2,
            model="test-model",
            provider="test_provider",
            task_id="task_cum",
            expected_output_tokens=10_000,
        )
    assert exc_info.value.decision.reason == "task_budget_exceeded"


def test_verification_reserve_protection(tmp_path: Path) -> None:
    """Verification reserve is explicitly carved out and protected."""
    settings = Settings(
        mana_routing_task_token_budget=100_000,
        mana_routing_verification_reserve_ratio=0.20,  # 20% = 20,000 reserved
        mana_context_cost_log_enabled=False,
    )
    governor = ContextCostGovernor(
        settings=settings,
        mode=GovernorMode.ENFORCE,
        session_id="test_session",
    )
    snapshot = governor.accounting_snapshot(task_id="task_v")
    assert snapshot.verification_reserve_tokens == 20_000
    assert snapshot.task_budget_tokens == 100_000


def test_concurrency_and_thread_safety(tmp_path: Path) -> None:
    """Concurrent reservations and reconciliations maintain atomic invariants without race conditions."""
    settings = Settings(
        mana_routing_task_token_budget=1_000_000,
        mana_context_cost_log_enabled=False,
    )
    governor = ContextCostGovernor(
        settings=settings,
        mode=GovernorMode.ENFORCE,
        session_id="test_session",
    )
    profile = make_profile("test-model", context=400_000, output=32_768)
    governor.register_model_profiles([profile])

    def worker(worker_id: int) -> None:
        segments = [ContextSegment(kind="prompt", content=f"worker_{worker_id}", token_estimate=100, protected=True)]
        call_id, dec = governor.before_model_call(
            segments,
            model="test-model",
            provider="test_provider",
            task_id=f"task_{worker_id % 4}",
            step_id=f"step_{worker_id}",
            expected_output_tokens=50,
        )
        if dec.allowed:
            governor.record_model_call(
                call_id,
                usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                provider="test_provider",
                model="test-model",
                task_id=f"task_{worker_id % 4}",
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(32)))

    snapshot = governor.accounting_snapshot()
    assert snapshot.session_consumed_tokens == 32 * 150
