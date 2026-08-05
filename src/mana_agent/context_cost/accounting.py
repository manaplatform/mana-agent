"""Authoritative model-aware token estimation, reservation, and reconciliation."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping

from mana_agent.context_cost.history import HistoricalSample, HistoricalTokenPredictor, PredictionKey
from mana_agent.context_cost.profiles import ModelIdentity, ModelTokenProfile, ModelTokenProfileResolver
from mana_agent.context_cost.store import AccountingStore
from mana_agent.context_cost.tokenizers import TokenizerRegistry
from mana_agent.context_cost.usage import ActualTokenUsage, normalize_provider_usage


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _money(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


@dataclass(frozen=True, slots=True)
class TokenEstimationRequest:
    model_identity: ModelIdentity
    components: Mapping[str, Any]
    component_token_overrides: Mapping[str, int] = field(default_factory=dict)
    route: str = ""
    lane: str = ""
    expected_tool_steps: int = 0
    expected_model_calls: int = 1
    requested_output_tokens: int | None = None
    historical_prediction_enabled: bool = True
    execution_kind: str = "model_call"
    tool_count: int = 0
    task_token_remaining: int | None = None
    session_token_remaining: int | None = None
    lane_policy_limit: int | None = None


@dataclass(frozen=True, slots=True)
class TokenEstimate:
    model_identity: ModelIdentity
    profile: ModelTokenProfile
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: Decimal | None
    confidence: str
    components: Mapping[str, int]
    assumptions: tuple[str, ...]
    effective_total_limit: int
    remaining_context_tokens: int


@dataclass(frozen=True, slots=True)
class AccountingReservation:
    reservation_id: str
    parent_reservation_id: str | None
    estimate: TokenEstimate
    task_id: str = ""
    session_id: str = ""
    run_id: str = ""
    attempt_id: str = ""
    status: str = "reserved"
    created_at: str = field(default_factory=_now)
    reconciled_at: str | None = None
    actual: ActualTokenUsage | None = None
    prediction_key: PredictionKey | None = None


class ModelContextLimitError(ValueError):
    def __init__(self, message: str, *, required: int | None = None, effective_limit: int | None = None) -> None:
        super().__init__(message)
        self.required = required
        self.effective_limit = effective_limit
        self.deficit = None if required is None or effective_limit is None else max(0, required - effective_limit)


class ModelTokenAccountingService:
    def __init__(
        self,
        resolver: ModelTokenProfileResolver,
        *,
        tokenizer_registry: TokenizerRegistry | None = None,
        store: AccountingStore | None = None,
        predictor: HistoricalTokenPredictor | None = None,
        safety_margin_ratio: Decimal = Decimal("0.05"),
        default_output_ratio: Decimal = Decimal("0.20"),
        historical_prediction_enabled: bool = True,
    ) -> None:
        self.resolver = resolver
        self.tokenizers = tokenizer_registry or TokenizerRegistry()
        self.store = store or AccountingStore()
        self.predictor = predictor or HistoricalTokenPredictor()
        self.safety_margin_ratio = max(Decimal("0"), safety_margin_ratio)
        self.default_output_ratio = min(Decimal("1"), max(Decimal("0.01"), default_output_ratio))
        self.historical_prediction_enabled = historical_prediction_enabled
        self._finalized: dict[str, AccountingReservation] = {}
        if historical_prediction_enabled:
            for row in self.store.rows():
                actual = row.get("actual") if isinstance(row.get("actual"), Mapping) else {}
                key = row.get("prediction_key") if isinstance(row.get("prediction_key"), Mapping) else {}
                if row.get("status") != "reconciled" or not actual or not key:
                    continue
                try:
                    self.predictor.record(HistoricalSample(
                        PredictionKey(**key),
                        int(actual.get("input_tokens") or 0),
                        int(actual.get("output_tokens") or 0),
                    ))
                except (TypeError, ValueError):
                    continue

    def estimate(self, request: TokenEstimationRequest) -> TokenEstimate:
        profile = self.resolver.resolve(request.model_identity)
        tokenizer, tokenizer_assumptions = self.tokenizers.resolve(profile.tokenizer)
        component_tokens = {
            str(name): max(0, tokenizer.count(value))
            for name, value in request.components.items()
            if value not in (None, "", (), [], {})
        }
        component_tokens.update({
            str(name): max(0, int(value))
            for name, value in request.component_token_overrides.items()
        })
        base_input = sum(component_tokens.values())
        assumptions = list(tokenizer_assumptions)
        calls = max(1, int(request.expected_model_calls))
        if calls > 1:
            component_tokens["follow_up_call_payload"] = base_input * (calls - 1)
            assumptions.append(f"expected model calls: {calls}")
        base_input = sum(component_tokens.values())
        requested_output = request.requested_output_tokens
        if requested_output is None:
            requested_output = max(1, int(Decimal(profile.max_output_tokens) * self.default_output_ratio))
            assumptions.append("output allowance uses configured estimation policy")
        output = max(1, int(requested_output)) * calls
        key = self._prediction_key(request)
        if self.historical_prediction_enabled and request.historical_prediction_enabled:
            historical_input, historical_output, sample_count = self.predictor.predict(key)
            if historical_input is not None:
                base_input = max(base_input, historical_input)
                component_tokens["historical_input_allowance"] = max(0, historical_input - sum(component_tokens.values()))
            if historical_output is not None:
                output = max(output, historical_output)
            if sample_count:
                assumptions.append(f"historical p80 prediction from {sample_count} matching runs")
        margin = max(1, int(Decimal(base_input + output) * self.safety_margin_ratio))
        component_tokens["safety_margin"] = margin
        input_tokens = base_input + margin
        if output > profile.max_output_tokens * calls:
            raise ModelContextLimitError(
                f"requested output {output} exceeds {profile.identity.key} output capacity {profile.max_output_tokens * calls}",
                required=output,
                effective_limit=profile.max_output_tokens * calls,
            )
        total = input_tokens + output
        limits = [profile.context_window * calls]
        limits.extend(
            int(value)
            for value in (request.task_token_remaining, request.session_token_remaining, request.lane_policy_limit)
            if value is not None
        )
        effective = min(limits)
        if total > effective:
            raise ModelContextLimitError(
                f"estimated run requires {total} tokens but effective limit is {effective} for {profile.identity.key}; deficit={total - effective}",
                required=total,
                effective_limit=effective,
            )
        cost = self.calculate_cost(profile, input_tokens=input_tokens, output_tokens=output)
        confidence = "low" if tokenizer.confidence == "low" or profile.confidence == "low" else "medium" if tokenizer.confidence == "medium" else "high"
        return TokenEstimate(request.model_identity, profile, input_tokens, output, total, cost, confidence, component_tokens, tuple(assumptions), effective, effective - total)

    def reserve(
        self,
        request: TokenEstimationRequest,
        *,
        operation_id: str,
        parent_reservation_id: str | None = None,
        task_id: str = "",
        session_id: str = "",
        run_id: str = "",
        attempt_id: str = "",
    ) -> AccountingReservation:
        estimate = self.estimate(request)
        reservation_id = self._stable_id(operation_id, request.model_identity.key, attempt_id)
        reservation = AccountingReservation(
            reservation_id, parent_reservation_id, estimate, task_id, session_id, run_id,
            attempt_id, prediction_key=self._prediction_key(request),
        )
        existing = self.store.put_if_absent(reservation_id, self._reservation_dict(reservation))
        if existing.get("status") != "reserved":
            raise ValueError(f"accounting operation {operation_id!r} was already finalized")
        return reservation

    def reconcile(
        self,
        reservation: AccountingReservation,
        *,
        usage: Any = None,
        final_components: Mapping[str, Any] | None = None,
    ) -> AccountingReservation:
        stored = self.store.get(reservation.reservation_id)
        if stored is not None and stored.get("status") == "reconciled":
            finalized = self._finalized.get(reservation.reservation_id)
            if finalized is not None:
                return finalized
            raw_actual = stored.get("actual") if isinstance(stored.get("actual"), Mapping) else {}
            actual = ActualTokenUsage(
                input_tokens=int(raw_actual.get("input_tokens") or 0),
                cached_input_tokens=int(raw_actual.get("cached_input_tokens") or 0),
                output_tokens=int(raw_actual.get("output_tokens") or 0),
                reasoning_tokens=int(raw_actual.get("reasoning_tokens") or 0),
                total_tokens=int(raw_actual.get("total_tokens") or 0),
                actual_cost=(Decimal(str(raw_actual["actual_cost"])) if raw_actual.get("actual_cost") is not None else None),
                status=str(raw_actual.get("usage_status") or "estimated"),
            )
            return replace(
                reservation,
                status="reconciled",
                reconciled_at=str(stored.get("reconciled_at") or _now()),
                actual=actual,
            )
        actual = normalize_provider_usage(usage)
        if actual is None:
            tokenizer, _ = self.tokenizers.resolve(reservation.estimate.profile.tokenizer)
            values = dict(final_components or {})
            input_tokens = sum(tokenizer.count(value) for key, value in values.items() if key != "response")
            output_tokens = tokenizer.count(values.get("response", ""))
            actual = ActualTokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                status="estimated",
            )
        cost = self.calculate_cost(
            reservation.estimate.profile,
            input_tokens=max(0, actual.input_tokens - actual.cached_input_tokens),
            cached_input_tokens=actual.cached_input_tokens,
            output_tokens=actual.output_tokens,
            reasoning_tokens=actual.reasoning_tokens,
        )
        actual = replace(actual, actual_cost=cost)
        completed = replace(reservation, status="reconciled", reconciled_at=_now(), actual=actual)
        existing, finalized = self.store.update_if_status(
            reservation.reservation_id,
            expected_status="reserved",
            record=self._reservation_dict(completed),
        )
        if not finalized:
            if existing.get("status") == "reconciled":
                return self.reconcile(reservation, usage=usage, final_components=final_components)
            return replace(
                reservation,
                status=str(existing.get("status") or reservation.status),
                reconciled_at=str(existing.get("reconciled_at") or _now()),
            )
        self._finalized[reservation.reservation_id] = completed
        if reservation.prediction_key is not None:
            self.predictor.record(HistoricalSample(reservation.prediction_key, actual.input_tokens, actual.output_tokens))
        return completed

    def release(self, reservation: AccountingReservation, *, reason: str) -> AccountingReservation:
        stored = self.store.get(reservation.reservation_id)
        if stored is not None and stored.get("status") == "reconciled":
            return self.reconcile(reservation)
        if stored is not None and stored.get("status") == "released":
            return self._finalized.get(
                reservation.reservation_id,
                replace(
                    reservation,
                    status="released",
                    reconciled_at=str(stored.get("reconciled_at") or _now()),
                ),
            )
        released = replace(reservation, status="released", reconciled_at=_now())
        payload = self._reservation_dict(released)
        payload["release_reason"] = str(reason)
        existing, finalized = self.store.update_if_status(
            reservation.reservation_id,
            expected_status="reserved",
            record=payload,
        )
        if not finalized:
            if existing.get("status") == "reconciled":
                return self.reconcile(reservation)
            return replace(
                reservation,
                status=str(existing.get("status") or reservation.status),
                reconciled_at=str(existing.get("reconciled_at") or _now()),
            )
        self._finalized[reservation.reservation_id] = released
        return released

    @staticmethod
    def calculate_cost(
        profile: ModelTokenProfile,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> Decimal | None:
        if profile.input_price_per_million is None or profile.output_price_per_million is None:
            return None
        million = Decimal(1_000_000)
        cached_rate = profile.cached_input_price_per_million or profile.input_price_per_million
        reasoning_rate = profile.reasoning_price_per_million or Decimal("0")
        return (
            Decimal(max(0, input_tokens)) * profile.input_price_per_million
            + Decimal(max(0, cached_input_tokens)) * cached_rate
            + Decimal(max(0, output_tokens)) * profile.output_price_per_million
            + Decimal(max(0, reasoning_tokens)) * reasoning_rate
        ) / million

    @staticmethod
    def _stable_id(operation_id: str, model_key: str, attempt_id: str) -> str:
        payload = f"{operation_id}\0{model_key}\0{attempt_id}".encode()
        return f"reservation_{hashlib.sha256(payload).hexdigest()[:32]}"

    @staticmethod
    def _prediction_key(request: TokenEstimationRequest) -> PredictionKey:
        return PredictionKey(
            request.model_identity.provider,
            request.model_identity.model,
            request.route,
            request.lane,
            request.execution_kind,
            min(10, max(0, int(request.tool_count))),
            min(20, max(0, int(request.expected_tool_steps))),
        )

    @staticmethod
    def _reservation_dict(reservation: AccountingReservation) -> dict[str, Any]:
        estimate = reservation.estimate
        actual = reservation.actual
        return {
            "schema_version": 1,
            "reservation_id": reservation.reservation_id,
            "parent_reservation_id": reservation.parent_reservation_id,
            "task_id": reservation.task_id,
            "session_id": reservation.session_id,
            "run_id": reservation.run_id,
            "attempt_id": reservation.attempt_id,
            "provider": estimate.model_identity.provider,
            "model": estimate.model_identity.model,
            "route": reservation.prediction_key.route if reservation.prediction_key else "",
            "lane": reservation.prediction_key.lane if reservation.prediction_key else "",
            "execution_kind": reservation.prediction_key.execution_kind if reservation.prediction_key else "",
            "profile_source": estimate.profile.source,
            "tokenizer": estimate.profile.tokenizer,
            "context_window": estimate.profile.context_window,
            "max_output_tokens": estimate.profile.max_output_tokens,
            "estimate": {
                "input_tokens": estimate.input_tokens,
                "output_tokens": estimate.output_tokens,
                "total_tokens": estimate.total_tokens,
                "estimated_cost": _money(estimate.estimated_cost),
                "confidence": estimate.confidence,
                "components": dict(estimate.components),
                "assumptions": list(estimate.assumptions),
            },
            "actual": None if actual is None else {
                "input_tokens": actual.input_tokens,
                "cached_input_tokens": actual.cached_input_tokens,
                "output_tokens": actual.output_tokens,
                "reasoning_tokens": actual.reasoning_tokens,
                "total_tokens": actual.total_tokens,
                "actual_cost": _money(actual.actual_cost),
                "usage_status": actual.status,
            },
            "estimate_error": None if actual is None else actual.total_tokens - estimate.total_tokens,
            "status": reservation.status,
            "prediction_key": (None if reservation.prediction_key is None else asdict(reservation.prediction_key)),
            "created_at": reservation.created_at,
            "reconciled_at": reservation.reconciled_at,
        }


__all__ = [
    "AccountingReservation", "ModelContextLimitError", "ModelTokenAccountingService",
    "TokenEstimate", "TokenEstimationRequest",
]
