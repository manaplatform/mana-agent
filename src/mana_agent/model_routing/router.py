from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Iterable

from mana_agent.model_routing.history import InMemoryRoutingHistory, RoutingHistory
from mana_agent.model_routing.models import (
    CandidateRejection,
    LatencyClass,
    ModelProfile,
    RoutingDecision,
    RoutingFailure,
    RoutingOutcome,
    RoutingPolicy,
    RoutingRequest,
    level_value,
    sanitize_configuration,
)


@dataclass(frozen=True, slots=True)
class _Scored:
    profile: ModelProfile
    score: float
    confidence: float
    estimated_input: int
    estimated_output: int
    estimated_cost: float
    reasons: tuple[str, ...]


_FAILURES = {"provider_error", "authentication", "rate_limit", "invalid_tool_call", "unsupported_parameter", "malformed_output", "verification_failure", "timeout"}
_LATENCY = {LatencyClass.INTERACTIVE: 0, LatencyClass.STANDARD: 1, LatencyClass.BATCH: 2}


class ModelRouter:
    """Deterministic scoring policy; provider execution intentionally lives elsewhere."""

    def __init__(self, profiles: Iterable[ModelProfile], *, history: RoutingHistory | None = None, policy: RoutingPolicy | None = None) -> None:
        self.profiles = tuple(sorted(profiles, key=lambda item: item.key))
        if len({item.key for item in self.profiles}) != len(self.profiles):
            raise ValueError("model profile registry contains duplicate provider/model IDs")
        self.history = history or InMemoryRoutingHistory()
        self.policy = policy or RoutingPolicy()

    def record_outcome(self, outcome: RoutingOutcome) -> None:
        if f"{outcome.provider}/{outcome.model_id}" not in {item.key for item in self.profiles}:
            raise ValueError("routing outcome references an unregistered provider/model")
        self.history.record(outcome)

    def route(self, request: RoutingRequest) -> RoutingDecision:
        if not self.policy.enabled:
            raise RoutingFailure("Adaptive model routing is disabled. No fallback action was executed.")
        scored: list[_Scored] = []
        rejected: list[CandidateRejection] = []
        for profile in self.profiles:
            reasons = self._reject(profile, request)
            if reasons:
                rejected.append(CandidateRejection(profile.key, tuple(reasons)))
                continue
            item = self._score(profile, request)
            if item.confidence < self.policy.minimum_confidence:
                rejected.append(CandidateRejection(profile.key, (f"confidence {item.confidence:.3f} is below {self.policy.minimum_confidence:.3f}",)))
                continue
            scored.append(item)
        if not scored:
            raise RoutingFailure("No configured model satisfies the routing capability, reliability, latency, and budget constraints. No fallback action was executed.", rejected=tuple(rejected))
        scored.sort(key=lambda item: (-item.score, item.estimated_cost, item.profile.key))
        winner = scored[0]
        competition = self._competition_allowed(request, scored)
        competition_candidates = tuple(item.profile.key for item in scored[: self.policy.maximum_candidate_count]) if competition else ()
        verifier, independent = self._select_verifier(request, author=winner.profile, candidates=scored)
        all_rejected = rejected + [CandidateRejection(item.profile.key, ("lower deterministic routing score",)) for item in scored[1:]]
        return RoutingDecision(
            selected_model=winner.profile.model_id,
            provider=winner.profile.provider,
            model_configuration=sanitize_configuration(winner.profile.configuration),
            selected_role=request.role,
            routing_score=round(winner.score, 6),
            confidence=round(winner.confidence, 6),
            estimated_input_tokens=winner.estimated_input,
            estimated_output_tokens=winner.estimated_output,
            estimated_cost=round(winner.estimated_cost, 8),
            expected_latency_class=winner.profile.latency_class,
            selection_reasons=winner.reasons,
            rejected_candidates=tuple(all_rejected),
            candidate_competition=competition,
            competition_candidates=competition_candidates,
            verifier_model=verifier.key if verifier else None,
            verifier_independent=independent,
            applicable_budgets=request.budgets,
        )

    def _reject(self, profile: ModelProfile, request: RoutingRequest) -> list[str]:
        reasons: list[str] = []
        if not profile.available:
            reasons.append("model is unavailable")
        if request.role not in profile.supported_roles and "*" not in profile.supported_roles:
            reasons.append(f"role {request.role!r} is unsupported")
        missing_tools = request.required_tools - profile.supported_tools
        if request.required_tools and not profile.can_tool_call:
            reasons.append("model cannot call tools")
        elif missing_tools and "*" not in profile.supported_tools:
            reasons.append("required tools are unsupported: " + ", ".join(sorted(missing_tools)))
        capability_flags = {
            "patch": profile.can_patch, "structured_output": profile.can_structured_output,
            "tool_calls": profile.can_tool_call, "verification": profile.can_verify,
            "reasoning": bool(profile.reasoning_settings - {"none"}),
        }
        missing = sorted(item for item in request.required_capabilities if not capability_flags.get(item, False))
        if missing:
            reasons.append("required capabilities are unsupported: " + ", ".join(missing))
        required_context = request.expected_prompt_tokens + request.retrieved_context_tokens + request.expected_response_tokens
        if required_context > profile.context_window:
            reasons.append(f"required context {required_context} exceeds {profile.context_window}")
        if _LATENCY[profile.latency_class] > _LATENCY[request.latency_requirement]:
            reasons.append(f"latency class {profile.latency_class.value} exceeds {request.latency_requirement.value}")
        if self._circuit_open(profile):
            reasons.append("provider/model circuit breaker is open")
        estimated = self._estimate_cost(profile, request)
        reserve = self._verification_reserve(request)
        limits = [value for value in (request.budgets.task_cost_limit, request.budgets.session_cost_remaining) if value is not None]
        if limits and estimated > max(0.0, min(limits) - reserve) and not request.budgets.allow_controlled_override:
            reasons.append(f"estimated implementation cost {estimated:.6f} exceeds budget after verification reserve {reserve:.6f}")
        estimated_input, estimated_output = self._estimated_tokens(profile, request)
        total_tokens = estimated_input + estimated_output
        if request.budgets.task_token_limit is not None and total_tokens > request.budgets.task_token_limit:
            reasons.append(f"estimated tokens {total_tokens} exceed task limit {request.budgets.task_token_limit}")
        return reasons

    def _score(self, profile: ModelProfile, request: RoutingRequest) -> _Scored:
        history = self._similar_history(profile, request)
        successes = [item for item in history if item.accepted is not None]
        success_rate = sum(bool(item.accepted) for item in successes) / len(successes) if successes else profile.reliability_score
        verified = [item for item in history if item.verification_passed is not None]
        verification_rate = sum(bool(item.verification_passed) for item in verified) / len(verified) if verified else profile.reliability_score
        tool_rows = [item for item in history if item.tool_failures or item.accepted is not None]
        tool_reliability = 1.0 - min(1.0, sum(item.tool_failures for item in tool_rows) / max(1, len(tool_rows)))
        structured_reliability = 1.0 - min(1.0, sum(item.failure_kind == "malformed_output" for item in history) / max(1, len(history)))
        historical = (success_rate + verification_rate + tool_reliability + structured_reliability) / 4
        benchmark = profile.benchmark_scores.get(request.task_type, profile.reliability_score)
        demand = self._effective_demand(request)
        reasoning = 1.0 if profile.reasoning_settings - {"none"} else 0.55
        quality = (profile.reliability_score + benchmark + reasoning * demand) / (2 + demand)
        languages = set(request.repository.languages)
        language = 1.0 if not languages or not profile.supported_languages else len(languages & profile.supported_languages) / len(languages)
        estimated_cost = self._estimate_cost(profile, request)
        cost_scale = request.budgets.task_cost_limit or request.budgets.session_cost_remaining
        if cost_scale is not None:
            cost_score = max(0.0, 1.0 - estimated_cost / max(cost_scale, 1e-9))
        else:
            unit_cost = (
                (profile.input_cost_per_million + profile.output_cost_per_million) / 2
                if profile.input_cost_per_million or profile.output_cost_per_million
                else profile.logical_cost_per_1k_tokens
            )
            cost_score = 1.0 / (1.0 + max(0.0, unit_cost))
        latency_score = 1.0 - (_LATENCY[profile.latency_class] * 0.25)
        capability = 1.0
        weights = self.policy.weights
        score = sum((
            weights.get("capability", 0.0) * capability,
            weights.get("quality", 0.0) * (0.65 + demand) * quality,
            weights.get("history", 0.0) * (0.65 + demand) * historical,
            weights.get("language", 0.0) * language,
            weights.get("cost", 0.0) * (1.4 - demand) * cost_score,
            weights.get("latency", 0.0) * latency_score,
        ))
        penalty = self._recent_failure_penalty(profile)
        score = max(0.0, score - penalty)
        confidence = max(0.0, min(1.0, (quality * 0.45) + (historical * 0.35) + (language * 0.20) - penalty))
        reasons = (
            f"capabilities satisfy {request.role}/{request.task_type}",
            f"quality evidence={quality:.3f}, historical reliability={historical:.3f}",
            f"repository language fit={language:.3f}",
            f"estimated cost={estimated_cost:.6f}, latency={profile.latency_class.value}",
        )
        estimated_input, estimated_output = self._estimated_tokens(profile, request)
        return _Scored(profile, score, confidence, estimated_input, estimated_output, estimated_cost, reasons)

    @staticmethod
    def _base_estimated_input(request: RoutingRequest) -> int:
        historical_overhead = request.expected_tool_calls * 350
        return request.expected_prompt_tokens + request.retrieved_context_tokens + historical_overhead

    def _estimated_tokens(self, profile: ModelProfile, request: RoutingRequest) -> tuple[int, int]:
        rows = self._similar_history(profile, request)
        input_rows = [item.input_tokens for item in rows if item.input_tokens > 0]
        output_rows = [item.output_tokens for item in rows if item.output_tokens > 0]
        base_input = self._base_estimated_input(request)
        estimated_input = round((base_input + (sum(input_rows) / len(input_rows))) / 2) if input_rows else base_input
        estimated_output = round((request.expected_response_tokens + (sum(output_rows) / len(output_rows))) / 2) if output_rows else request.expected_response_tokens
        return max(1, estimated_input), max(1, estimated_output)

    def _similar_history(self, profile: ModelProfile, request: RoutingRequest):
        rows = self.history.query(provider=profile.provider, model_id=profile.model_id, task_category=request.task_type)
        languages = set(request.repository.languages)
        frameworks = set(request.repository.frameworks)
        matching = tuple(
            item for item in rows
            if (not languages or not item.repository_languages or languages.intersection(item.repository_languages))
            and (not frameworks or not item.repository_frameworks or frameworks.intersection(item.repository_frameworks))
        )
        return matching or rows

    def _estimate_cost(self, profile: ModelProfile, request: RoutingRequest) -> float:
        input_tokens, output_tokens = self._estimated_tokens(profile, request)
        monetary = (input_tokens * profile.input_cost_per_million + output_tokens * profile.output_cost_per_million) / 1_000_000
        if monetary > 0:
            return monetary
        return ((input_tokens + output_tokens) / 1_000) * profile.logical_cost_per_1k_tokens

    @staticmethod
    def _effective_demand(request: RoutingRequest) -> float:
        demand = max(level_value(request.complexity), level_value(request.risk))
        repository = request.repository
        if repository.sensitive_areas:
            demand = max(demand, 0.9)
        if repository.file_count >= 20_000 or len(repository.changed_files) >= 30:
            demand = min(1.0, demand + 0.2)
        elif repository.file_count >= 5_000 or len(repository.changed_files) >= 10:
            demand = min(1.0, demand + 0.1)
        return demand

    def _verification_reserve(self, request: RoutingRequest) -> float:
        limits = [value for value in (request.budgets.task_cost_limit, request.budgets.session_cost_remaining) if value is not None]
        if not limits:
            return 0.0
        configured = request.budgets.verification_cost_limit
        ratio_reserve = min(limits) * request.budgets.verification_reserve_ratio
        return min(ratio_reserve, configured) if configured is not None else ratio_reserve

    def _recent_failure_penalty(self, profile: ModelProfile) -> float:
        now = datetime.now(timezone.utc)
        penalty = 0.0
        for item in self.history.query(provider=profile.provider, model_id=profile.model_id):
            if item.failure_kind not in _FAILURES:
                continue
            age = max(0.0, (now - item.occurred_at).total_seconds())
            penalty += self.policy.model_failure_penalty_weight * math.exp(-age / max(1, self.policy.reliability_decay_seconds))
        for item in self._provider_history(profile.provider):
            if item.model_id == profile.model_id or item.failure_kind not in _FAILURES:
                continue
            age = max(0.0, (now - item.occurred_at).total_seconds())
            penalty += self.policy.provider_failure_penalty_weight * math.exp(-age / max(1, self.policy.reliability_decay_seconds))
        return min(0.35, penalty)

    def _circuit_open(self, profile: ModelProfile) -> bool:
        now = datetime.now(timezone.utc)
        failures = sum(
            item.failure_kind in _FAILURES and (now - item.occurred_at).total_seconds() <= self.policy.circuit_breaker_window_seconds
            for item in self._provider_history(profile.provider)
        )
        return failures >= self.policy.circuit_breaker_failures

    def _provider_history(self, provider: str):
        rows = []
        for candidate in self.profiles:
            if candidate.provider == provider:
                rows.extend(self.history.query(provider=provider, model_id=candidate.model_id))
        return tuple(rows)

    def _competition_allowed(self, request: RoutingRequest, candidates: list[_Scored]) -> bool:
        if not request.multi_candidate_permitted or len(candidates) < 2 or self.policy.maximum_candidate_count < 2:
            return False
        threshold_value = min(level_value(self.policy.competition_complexity_threshold), level_value(self.policy.competition_risk_threshold))
        threshold = self._effective_demand(request) >= threshold_value
        triggered = threshold or request.previous_verification_failed or request.explicit_competition
        if not triggered or request.latency_requirement is LatencyClass.INTERACTIVE:
            return False
        limit = request.budgets.competition_cost_limit
        projected = sum(item.estimated_cost for item in candidates[: self.policy.maximum_candidate_count]) + self._verification_reserve(request)
        return limit is None or projected <= limit or request.budgets.allow_controlled_override

    def _select_verifier(self, request: RoutingRequest, *, author: ModelProfile, candidates: list[_Scored]) -> tuple[ModelProfile | None, bool]:
        required_context = request.expected_prompt_tokens + request.retrieved_context_tokens + request.expected_response_tokens
        qualified = [
            profile for profile in self.profiles
            if profile.available
            and profile.can_verify
            and profile.can_structured_output
            and ("verifier" in profile.supported_roles or "*" in profile.supported_roles)
            and profile.context_window >= required_context
            and not self._circuit_open(profile)
            and (
                request.budgets.verification_cost_limit is None
                or self._estimate_cost(profile, request) <= request.budgets.verification_cost_limit
                or request.budgets.allow_controlled_override
            )
        ]
        languages = set(request.repository.languages)
        qualified.sort(key=lambda profile: (profile.key == author.key, -(profile.reliability_score + profile.benchmark_scores.get("verification", 0.0)), -(len(languages & profile.supported_languages) if profile.supported_languages else len(languages)), profile.key))
        if not qualified:
            return None, False
        return qualified[0], qualified[0].key != author.key
