from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Callable, Iterable
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mana_agent.context_cost.accounting import ModelContextLimitError, ModelTokenAccountingService, TokenEstimationRequest
from mana_agent.context_cost.models import ContextBudgetExceeded
from mana_agent.context_cost.profiles import ModelIdentity, ModelTokenProfileResolver, UnknownModelProfileError

from mana_agent.evals.recorder import record_current
from mana_agent.llm.errors import InferenceError
from mana_agent.model_routing.history import InMemoryRoutingHistory, RoutingHistory
from mana_agent.model_routing.models import (
    CandidateRejection,
    Complexity,
    LatencyClass,
    ModelProfile,
    NoWriteCapableModelAvailableError,
    RoutingDecision,
    RoutingFailure,
    RoutingOutcome,
    RoutingPolicy,
    RoutingRequest,
    RoutingResourceScore,
    RoutingMode,
    RiskLevel,
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
    estimated_cost: float | None
    estimate_confidence: str
    estimate_components: dict[str, int]
    estimate_assumptions: tuple[str, ...]
    reasons: tuple[str, ...]
    resource: RoutingResourceScore


_FAILURES = {"provider_error", "authentication", "rate_limit", "invalid_tool_call", "unsupported_parameter", "malformed_output", "verification_failure", "timeout"}
_LATENCY = {LatencyClass.INTERACTIVE: 0, LatencyClass.STANDARD: 1, LatencyClass.BATCH: 2}


class ModelAssignmentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate: str
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class ExecutionModeDecision(ModelAssignmentDecision):
    mode: RoutingMode


class VerificationModeDecision(ExecutionModeDecision):
    verifier: str | None = Field(default=None, description="Independent verifier ID for verification or parallel modes. Use null for single mode.")


class ModelSelectionDecision(VerificationModeDecision):
    competition_candidates: list[str] = Field(default_factory=list, description=(
        "Use only for a parallel mode. Include the selected candidate and at least one other eligible "
        "candidate. Exclude the independent verifier. For every other mode, return an empty array."
    ))


class ModelRouter:
    """Main-model selection with deterministic eligibility and authorization gates."""

    def __init__(self, profiles: Iterable[ModelProfile], *, history: RoutingHistory | None = None, policy: RoutingPolicy | None = None, resource_availability: Callable[[ModelProfile, RoutingRequest], tuple[bool, str]] | None = None, resource_scoring: Callable[[ModelProfile, RoutingRequest], RoutingResourceScore] | None = None, decision_model: Any = None) -> None:
        self.profiles = tuple(sorted(profiles, key=lambda item: item.key))
        if len({item.key for item in self.profiles}) != len(self.profiles):
            raise ValueError("model profile registry contains duplicate provider/model IDs")
        self.history = history or InMemoryRoutingHistory()
        self.policy = policy or RoutingPolicy()
        self.resource_availability = resource_availability
        self.resource_scoring = resource_scoring
        self.decision_model = decision_model
        self.context_cost_governor = None
        self.accounting = ModelTokenAccountingService(
            ModelTokenProfileResolver(self.profiles, unknown_policy="require_metadata")
        )

    def record_outcome(self, outcome: RoutingOutcome) -> None:
        if f"{outcome.provider}/{outcome.model_id}" not in {item.key for item in self.profiles}:
            raise ValueError("routing outcome references an unregistered provider/model")
        self.history.record(outcome)

    def route(self, request: RoutingRequest, *, fallback_of_decision_id: str = "", fallback_reason: str = "") -> RoutingDecision:
        if not self.policy.enabled:
            raise RoutingFailure("Adaptive model routing is disabled. No fallback action was executed.")
        if self.decision_model is not None:
            return self._route_with_model(request, fallback_of_decision_id=fallback_of_decision_id, fallback_reason=fallback_reason)
        return self._route_deterministic(request, fallback_of_decision_id=fallback_of_decision_id, fallback_reason=fallback_reason)

    def _route_with_model(self, request: RoutingRequest, *, fallback_of_decision_id: str = "", fallback_reason: str = "") -> RoutingDecision:
        candidates, verifiers, rejected = {}, {}, []
        verification_request = replace(request, role="verifier", required_tools=frozenset(),
            required_capabilities=frozenset({"verification"}), expected_model_calls=1, expected_tool_calls=0)
        for profile in self.profiles:
            resource = self._resource_score(profile, request)
            reasons = self._reject(profile, request, resource)
            if reasons:
                rejected.append(CandidateRejection(profile.key, tuple(reasons)))
            else:
                candidates[profile.key] = (profile, resource)
            if not request.assignment_only and profile.can_verify and not self._reject(profile, verification_request, resource):
                verifiers[profile.key] = profile
        if not candidates:
            is_write_task = (
                "repository_write" in request.required_tools
                or "patch" in request.required_capabilities
                or "repository_write" in request.required_capabilities
            )
            if is_write_task:
                diagnostics = {
                    "required_capabilities": sorted(request.required_capabilities),
                    "required_tools": sorted(request.required_tools),
                    "rejected_candidates": [asdict(item) for item in rejected],
                }
                raise NoWriteCapableModelAvailableError(
                    "no_write_capable_model_available: No configured model satisfies the required write and tool "
                    "capabilities. No fallback action was executed.",
                    rejected=tuple(rejected),
                    diagnostics=diagnostics,
                )
            rejection_details = "; ".join(
                f"{item.model} ({', '.join(item.reasons)})" for item in rejected
            )
            msg = "No configured model satisfies the routing capability, reliability, latency, and budget constraints."
            if rejection_details:
                msg = f"{msg} Rejected candidates: {rejection_details}."
            msg = f"{msg} No fallback action was executed."
            raise RoutingFailure(msg, rejected=tuple(rejected))
        multi_agent_permitted = (
            not request.assignment_only and self.policy.multi_agent_enabled
            and request.subagents_allowed and request.main_model_requested_multi_agent
            and request.isolation_available and not request.active_task_conflict
            and request.task_tree_depth < self.policy.maximum_task_tree_depth
        )
        parallel_permitted = (
            self.policy.parallel_execution_enabled and request.parallel_execution_allowed and request.multi_candidate_permitted
            and request.main_model_requested_parallel and request.independent_verifier_available
            and request.isolation_available and not request.active_task_conflict
            and min(request.maximum_concurrency, self.policy.maximum_concurrency) >= 2 and self.policy.maximum_candidate_count >= 2
            and any(len(candidates.keys() - {key}) >= 2 for key in verifiers)
        )
        permitted_modes = [RoutingMode.SINGLE]
        if any(candidates.keys() - {key} for key in verifiers):
            permitted_modes.append(RoutingMode.SINGLE_WITH_VERIFICATION)
        if multi_agent_permitted:
            permitted_modes.append(RoutingMode.MULTI_AGENT)
        if parallel_permitted:
            permitted_modes.append(RoutingMode.PARALLEL_CANDIDATES)
            if multi_agent_permitted:
                permitted_modes.append(RoutingMode.MULTI_AGENT_WITH_PARALLEL_CANDIDATES)
        if request.assignment_only:
            decision_schema = ModelAssignmentDecision
            instruction = "Select one model from candidates to assign to the requested role. Return its exact ID, your reason, and your confidence."
        else:
            if parallel_permitted:
                decision_schema = ModelSelectionDecision
            elif RoutingMode.SINGLE_WITH_VERIFICATION in permitted_modes:
                decision_schema = VerificationModeDecision
            else:
                decision_schema = ExecutionModeDecision
            instruction = (
                "Select an execution model from candidates and a mode from permitted_modes. "
                "Choose an independent verifier from verifier_candidates for single_with_verification and parallel modes. "
                "When the schema includes verifier, use null for single mode. Use only fields in the response schema and the offered IDs."
            )
        model = self.decision_model
        from mana_agent.llm.messages import SystemMessage, UserMessage
        offered = []
        for key, (candidate, resource) in candidates.items():
            descriptor = candidate.capability_descriptor
            if descriptor is not None and hasattr(descriptor, "facts"):
                caps = {name: descriptor.fact(name).model_dump(mode="json") for name in sorted(descriptor.facts)}
            else:
                caps = {}
            estimate = self._token_estimate(candidate, request)
            offered.append({
                "id": key,
                "capabilities": caps,
                "estimated_input_tokens": estimate.input_tokens,
                "estimated_output_tokens": estimate.output_tokens,
                "estimated_cost": None if estimate.estimated_cost is None else float(estimate.estimated_cost),
                "latency_class": candidate.latency_class.value,
                "languages": sorted(candidate.supported_languages),
                "resource": asdict(resource),
            })
        payload = {"task_id": request.task_id, "request_id": request.request_id, "task": request.task_description,
                   "role": request.role, "required_capabilities": sorted(request.required_capabilities),
                   "required_tools": sorted(request.required_tools), "budgets": asdict(request.budgets),
                   "complexity": request.complexity.value, "risk": request.risk.value,
                   "repository_languages": list(request.repository.languages),
                   "candidates": offered, "verifier_candidates": list(verifiers),
                   "permitted_modes": [mode.value for mode in permitted_modes]}
        try:
            choice = model.decide_sync([SystemMessage(instruction),
                UserMessage(json.dumps(payload, ensure_ascii=False, default=str))], decision_schema)
            if type(choice) is not decision_schema:
                choice = decision_schema.model_validate(choice.model_dump() if isinstance(choice, BaseModel) else choice, strict=True)
        except Exception as exc:
            if isinstance(exc, ValidationError):
                fields = sorted({item["loc"][0] for item in exc.errors(include_input=False, include_context=False, include_url=False)
                    if item["loc"] and item["loc"][0] in decision_schema.model_fields})
                detail = "the response did not match the selection schema"
                if fields:
                    detail += f" ({', '.join(fields)})"
            elif isinstance(exc, (InferenceError, ContextBudgetExceeded, ModelContextLimitError, UnknownModelProfileError)):
                from mana_agent.utils.redaction import redact_secrets
                detail = str(redact_secrets(str(exc)))[:300]
            else:
                detail = f"inference failed ({type(exc).__name__})"
            raise RoutingFailure(f"Main model selection failed for role {request.role}. Reason: {detail.rstrip('.')}. The router selected no fallback.") from exc
        if choice.candidate not in candidates or choice.confidence < self.policy.minimum_confidence:
            raise RoutingFailure("Main model selected an unoffered candidate or insufficient confidence")
        mode = choice.mode if isinstance(choice, ExecutionModeDecision) else RoutingMode.SINGLE
        if mode not in permitted_modes:
            raise RoutingFailure("Main model selected an execution mode outside the admitted authority")
        profile, resource = candidates[choice.candidate]
        verifier_id = choice.verifier if isinstance(choice, VerificationModeDecision) else None
        verifier = verifiers.get(verifier_id) if verifier_id is not None else None
        if verifier_id is not None and (verifier is None or verifier.key == profile.key or not verifier.can_verify):
            raise RoutingFailure("Selected verifier is unavailable, not independent, or incapable")
        competition = mode in {RoutingMode.PARALLEL_CANDIDATES, RoutingMode.MULTI_AGENT_WITH_PARALLEL_CANDIDATES}
        multi_agent = mode in {RoutingMode.MULTI_AGENT, RoutingMode.MULTI_AGENT_WITH_PARALLEL_CANDIDATES}
        competition_ids = tuple(choice.competition_candidates) if isinstance(choice, ModelSelectionDecision) else ()
        if competition and not (verifier is not None
                and 2 <= len(set(competition_ids)) == len(competition_ids) <= self.policy.maximum_candidate_count
                and all(key in candidates for key in competition_ids)
                and choice.candidate in competition_ids and verifier.key not in competition_ids):
            raise RoutingFailure("Selected competition exceeds candidate, verifier, isolation or parallel authority")
        if not competition and competition_ids:
            raise RoutingFailure("Competition candidates were supplied for a non-competition mode")
        if mode == RoutingMode.SINGLE_WITH_VERIFICATION and verifier is None:
            raise RoutingFailure("Selected verification mode requires an independent verifier")
        if mode == RoutingMode.SINGLE and verifier is not None:
            raise RoutingFailure("A verifier requires an explicit verification execution mode")
        selected_profiles = [candidates[key][0] for key in competition_ids] if competition else [profile]
        request_id = request.request_id or self._stable_id("request", request)
        decision_id = self._stable_id("decision", {"request_id": request_id, "selected": profile.key, "mode": mode.value})
        estimate = self._token_estimate(profile, request)
        return RoutingDecision(
            selected_model=profile.model_id, provider=profile.provider,
            model_configuration=sanitize_configuration(profile.configuration), selected_role=request.role,
            routing_score=1.0, confidence=round(choice.confidence, 4),
            estimated_input_tokens=estimate.input_tokens, estimated_output_tokens=estimate.output_tokens,
            estimated_cost=None if estimate.estimated_cost is None else float(estimate.estimated_cost),
            estimate_confidence=estimate.confidence, estimate_components=dict(estimate.components),
            estimate_assumptions=tuple(estimate.assumptions), model_context_window=profile.context_window,
            model_max_output_tokens=profile.max_output_tokens, expected_model_calls=request.expected_model_calls,
            expected_latency_class=profile.latency_class, selection_reasons=(choice.reason,), rejected_candidates=tuple(rejected),
            candidate_competition=competition, competition_candidates=competition_ids,
            verifier_model=verifier.key if verifier else None, verifier_independent=verifier is not None,
            applicable_budgets=request.budgets, decision_id=decision_id, request_id=request_id, task_id=request.task_id,
            routing_mode=mode, required_verification_level="independent" if verifier else "standard",
            parallel_execution_permitted=competition, multi_agent_execution_permitted=multi_agent,
            applicable_limits={"maximum_candidates": self.policy.maximum_candidate_count if competition else 1,
                "maximum_concurrency": min(request.maximum_concurrency, self.policy.maximum_concurrency),
                "maximum_task_tree_depth": self.policy.maximum_task_tree_depth}, deadline_seconds=self.policy.task_timeout_seconds,
            orchestration_reasons=(choice.reason,), resource_score=resource,
            fallback_of_decision_id=fallback_of_decision_id, fallback_reason=fallback_reason)

    def _route_deterministic(self, request: RoutingRequest, *, fallback_of_decision_id: str = "", fallback_reason: str = "") -> RoutingDecision:
        scored: list[_Scored] = []
        rejected: list[CandidateRejection] = []
        for profile in self.profiles:
            resource = self._resource_score(profile, request)
            reasons = self._reject(profile, request, resource)
            if reasons:
                rejected.append(CandidateRejection(profile.key, tuple(reasons)))
                desc = getattr(profile, "capability_descriptor", None)
                transport = getattr(desc, "transport", "") or "direct_responses"
                record_current(
                    "model.candidate.rejected",
                    {
                        "provider": profile.provider,
                        "model": profile.model_id,
                        "transport": transport,
                        "required_capabilities": sorted(request.required_capabilities),
                        "resolved_capabilities": desc.to_dict() if desc is not None else {},
                        "capability_source": desc.capability_source if desc is not None else "unknown",
                        "rejection_reason": "; ".join(reasons),
                    },
                )
                continue
            item = self._score(profile, request, resource)
            if item.confidence < self.policy.minimum_confidence:
                rejection_reason = f"confidence {item.confidence:.3f} is below {self.policy.minimum_confidence:.3f}"
                rejected.append(CandidateRejection(profile.key, (rejection_reason,)))
                desc = getattr(profile, "capability_descriptor", None)
                transport = getattr(desc, "transport", "") or "direct_responses"
                record_current(
                    "model.candidate.rejected",
                    {
                        "provider": profile.provider,
                        "model": profile.model_id,
                        "transport": transport,
                        "required_capabilities": sorted(request.required_capabilities),
                        "resolved_capabilities": desc.to_dict() if desc is not None else {},
                        "capability_source": desc.capability_source if desc is not None else "unknown",
                        "rejection_reason": rejection_reason,
                    },
                )
                continue
            scored.append(item)
        if not scored:
            is_write_task = (
                "repository_write" in request.required_tools
                or "patch" in request.required_capabilities
                or "repository_write" in request.required_capabilities
            )
            if is_write_task:
                diagnostics = {
                    "required_capabilities": sorted(request.required_capabilities),
                    "required_tools": sorted(request.required_tools),
                    "rejected_candidates": [asdict(item) for item in rejected],
                }
                raise NoWriteCapableModelAvailableError(
                    "no_write_capable_model_available: No configured model satisfies the required write and tool "
                    "capabilities. No fallback action was executed.",
                    rejected=tuple(rejected),
                    diagnostics=diagnostics,
                )
            rejection_details = "; ".join(
                f"{item.model} ({', '.join(item.reasons)})" for item in rejected
            )
            msg = "No configured model satisfies the routing capability, reliability, latency, and budget constraints."
            if rejection_details:
                msg = f"{msg} Rejected candidates: {rejection_details}."
            msg = f"{msg} No fallback action was executed."
            raise RoutingFailure(msg, rejected=tuple(rejected))
        scored.sort(key=lambda item: (-item.score, item.estimated_cost is None, item.estimated_cost or 0.0, item.profile.key))
        winner = scored[0]
        winner_desc = getattr(winner.profile, "capability_descriptor", None)
        winner_transport = getattr(winner_desc, "transport", "") or "direct_responses"
        record_current(
            "model.candidate.selected",
            {
                "provider": winner.profile.provider,
                "model": winner.profile.model_id,
                "transport": winner_transport,
                "required_capabilities": sorted(request.required_capabilities),
                "resolved_capabilities": winner_desc.to_dict() if winner_desc is not None else {},
                "capability_source": winner_desc.capability_source if winner_desc is not None else "unknown",
            },
        )
        verifier, independent = self._select_verifier(request, author=winner.profile, candidates=scored)
        competition, competition_reasons = self._competition_allowed(
            request,
            scored,
            independent_verifier=bool(verifier and independent),
        )
        multi_agent, multi_agent_reasons = self._multi_agent_allowed(request)
        competition_candidates = tuple(item.profile.key for item in scored[: self.policy.maximum_candidate_count]) if competition else ()
        if multi_agent and competition:
            routing_mode = RoutingMode.MULTI_AGENT_WITH_PARALLEL_CANDIDATES
        elif multi_agent:
            routing_mode = RoutingMode.MULTI_AGENT
        elif competition:
            routing_mode = RoutingMode.PARALLEL_CANDIDATES
        elif verifier is not None and (level_value(request.risk) >= level_value(RiskLevel.HIGH) or request.previous_verification_failed):
            routing_mode = RoutingMode.SINGLE_WITH_VERIFICATION
        else:
            routing_mode = RoutingMode.SINGLE
        all_rejected = rejected + [CandidateRejection(item.profile.key, ("lower deterministic routing score",)) for item in scored[1:]]
        request_id = request.request_id or self._stable_id("request", request)
        decision_id = self._stable_id(
            "decision",
            {"request_id": request_id, "selected": winner.profile.key, "mode": routing_mode.value},
        )
        return RoutingDecision(
            selected_model=winner.profile.model_id,
            provider=winner.profile.provider,
            model_configuration=sanitize_configuration(winner.profile.configuration),
            selected_role=request.role,
            routing_score=round(winner.score, 6),
            confidence=round(winner.confidence, 6),
            estimated_input_tokens=winner.estimated_input,
            estimated_output_tokens=winner.estimated_output,
            estimated_cost=(None if winner.estimated_cost is None else round(winner.estimated_cost, 8)),
            estimate_confidence=winner.estimate_confidence,
            estimate_components=dict(winner.estimate_components),
            estimate_assumptions=winner.estimate_assumptions,
            model_context_window=winner.profile.context_window,
            model_max_output_tokens=winner.profile.max_output_tokens,
            expected_model_calls=request.expected_model_calls,
            expected_latency_class=winner.profile.latency_class,
            selection_reasons=winner.reasons,
            rejected_candidates=tuple(all_rejected),
            candidate_competition=competition,
            competition_candidates=competition_candidates,
            verifier_model=verifier.key if verifier else None,
            verifier_independent=independent,
            applicable_budgets=request.budgets,
            decision_id=decision_id,
            request_id=request_id,
            task_id=request.task_id,
            routing_mode=routing_mode,
            required_verification_level=(
                "independent" if competition else "enhanced" if routing_mode is RoutingMode.SINGLE_WITH_VERIFICATION else "standard"
            ),
            parallel_execution_permitted=competition,
            multi_agent_execution_permitted=multi_agent,
            applicable_limits={
                "maximum_candidates": self.policy.maximum_candidate_count if competition else 1,
                "maximum_concurrency": min(request.maximum_concurrency, self.policy.maximum_concurrency),
                "maximum_task_tree_depth": self.policy.maximum_task_tree_depth,
            },
            deadline_seconds=self.policy.task_timeout_seconds,
            orchestration_reasons=tuple((*multi_agent_reasons, *competition_reasons)),
            resource_score=winner.resource,
            fallback_of_decision_id=fallback_of_decision_id,
            fallback_reason=fallback_reason,
        )

    @staticmethod
    def _stable_id(prefix: str, value: object) -> str:
        payload = json.dumps(asdict(value) if hasattr(value, "__dataclass_fields__") else value, sort_keys=True, default=str, separators=(",", ":"))
        return f"{prefix}_{hashlib.sha256(payload.encode()).hexdigest()[:20]}"

    def _resource_score(self, profile: ModelProfile, request: RoutingRequest) -> RoutingResourceScore:
        return self.resource_scoring(profile, request) if self.resource_scoring is not None else RoutingResourceScore(True, 1.0, 1.0, 1.0)

    def _reject(self, profile: ModelProfile, request: RoutingRequest, resource: RoutingResourceScore) -> list[str]:
        reasons: list[str] = []
        if not profile.available:
            reasons.append("model is unavailable")
        if self.resource_availability is not None:
            available, reason = self.resource_availability(profile, request)
            if not available:
                reasons.append(reason or "provider resource is unavailable")
        if not resource.available:
            reasons.append(resource.reason or "provider resource is unavailable")
        if request.role not in profile.supported_roles and "*" not in profile.supported_roles:
            reasons.append(f"role {request.role!r} is unsupported")
        if profile.context_window <= 0 or profile.max_output_tokens <= 0:
            reasons.append("context/output limits are UNKNOWN")
        pricing_known = bool(
            profile.configuration.get("pricing_known")
            or profile.input_cost_per_million > 0
            or profile.output_cost_per_million > 0
            or profile.logical_cost_per_1k_tokens > 0
        )
        if (request.budgets.task_cost_limit is not None or request.budgets.session_cost_remaining is not None) and not pricing_known:
            reasons.append("pricing is UNKNOWN for the cost-constrained operation")
        desc = getattr(profile, "capability_descriptor", None)
        if desc is not None and not desc.is_known and profile.source_level != "pinned":
            reasons.append(
                f"model capabilities are unknown (provider={profile.provider!r}, model={profile.model_id!r}, transport={getattr(desc, 'transport', '')!r})"
            )
        if desc is not None and (not desc.context_window or not desc.max_output_tokens or not desc.supports("input_text") or not desc.supports("output_text")):
            reasons.append("Text or token-limit evidence is absent or expired")
        missing_tools = request.required_tools - profile.supported_tools
        if request.required_tools and (not profile.can_tool_call or (desc is not None and not desc.supports_tool_calls)):
            reasons.append("model cannot call tools")
        elif missing_tools and "*" not in profile.supported_tools:
            reasons.append("required tools are unsupported: " + ", ".join(sorted(missing_tools)))
        supports_write = (
            desc.supports_repository_write if desc is not None else profile.can_patch
        )
        supports_read = (
            desc.supports_repository_read
            if desc is not None
            else ("repository_read" in profile.supported_tools or "*" in profile.supported_tools)
        )
        capability_flags = {
            "patch": profile.can_patch and supports_write,
            "structured_output": profile.can_structured_output and (desc.supports_structured_output if desc is not None else True),
            "tool_calls": profile.can_tool_call and (desc.supports_tool_calls if desc is not None else True),
            "verification": profile.can_verify,
            "reasoning": bool(desc.supports_reasoning if desc is not None else (profile.reasoning_settings - {"none"})),
            "repository_write": supports_write,
            "repository_read": supports_read,
            "shell": desc.supports_shell if desc is not None else profile.can_tool_call,
            "streaming": desc.supports_streaming if desc is not None else True,
            "parallel_tools": desc.supports_parallel_tools if desc is not None else False,
        }
        missing = sorted(item for item in request.required_capabilities if not capability_flags.get(item, False))
        if missing:
            reasons.append("required capabilities are unsupported: " + ", ".join(missing))
        try:
            estimate = self._token_estimate(profile, request)
        except (ModelContextLimitError, UnknownModelProfileError) as exc:
            reasons.append(str(exc))
            estimate = None
        if _LATENCY[profile.latency_class] > _LATENCY[request.latency_requirement]:
            reasons.append(f"latency class {profile.latency_class.value} exceeds {request.latency_requirement.value}")
        if self._circuit_open(profile):
            reasons.append("provider/model circuit breaker is open")
        estimated = None if estimate is None else estimate.estimated_cost
        reserve = self._verification_reserve(request)
        limits = [value for value in (request.budgets.task_cost_limit, request.budgets.session_cost_remaining) if value is not None]
        if limits and estimated is None and not request.budgets.allow_controlled_override:
            reasons.append("model pricing is unknown for a cost-constrained request")
        elif limits and estimated is not None and float(estimated) > max(0.0, min(limits) - reserve) and not request.budgets.allow_controlled_override:
            reasons.append(f"estimated implementation cost {estimated:.6f} exceeds budget after verification reserve {reserve:.6f}")
        if estimate is not None:
            total_tokens = estimate.total_tokens
            if request.budgets.task_token_limit is not None and total_tokens > request.budgets.task_token_limit:
                reasons.append(f"estimated tokens {total_tokens} exceed task limit {request.budgets.task_token_limit}")
        return reasons

    def _score(self, profile: ModelProfile, request: RoutingRequest, resource: RoutingResourceScore | None = None) -> _Scored:
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
        estimate = self._token_estimate(profile, request)
        estimated_cost = float(estimate.estimated_cost) if estimate.estimated_cost is not None else None
        cost_scale = request.budgets.task_cost_limit or request.budgets.session_cost_remaining
        if cost_scale is not None:
            cost_score = 0.0 if estimated_cost is None else max(0.0, 1.0 - estimated_cost / max(cost_scale, 1e-9))
        else:
            unit_cost = (
                (profile.input_cost_per_million + profile.output_cost_per_million) / 2
                if profile.input_cost_per_million or profile.output_cost_per_million
                else profile.logical_cost_per_1k_tokens
            )
            cost_score = 1.0 / (1.0 + max(0.0, unit_cost))
        latency_score = 1.0 - (_LATENCY[profile.latency_class] * 0.25)
        capability = 1.0
        resource = resource or self._resource_score(profile, request)
        weights = self.policy.weights
        score = sum((
            weights.get("capability", 0.0) * capability,
            weights.get("quality", 0.0) * (0.65 + demand) * quality,
            weights.get("history", 0.0) * (0.65 + demand) * historical,
            weights.get("language", 0.0) * language,
            weights.get("cost", 0.0) * (1.4 - demand) * cost_score,
            weights.get("latency", 0.0) * latency_score,
            weights.get("provider_capacity", 0.0) * resource.provider_capacity,
            weights.get("quota_health", 0.0) * resource.quota_health,
            weights.get("resource_confidence", 0.0) * resource.resource_confidence,
        ))
        penalty = self._recent_failure_penalty(profile)
        score = max(0.0, score - penalty)
        confidence = max(0.0, min(1.0, (quality * 0.45) + (historical * 0.35) + (language * 0.20) - penalty))
        reasons = (
            f"capabilities satisfy {request.role}/{request.task_type}",
            f"quality evidence={quality:.3f}, historical reliability={historical:.3f}",
            f"repository language fit={language:.3f}",
            f"estimated cost={'unknown' if estimated_cost is None else f'{estimated_cost:.6f}'}, latency={profile.latency_class.value}",
            f"provider capacity={resource.provider_capacity:.3f}, quota health={resource.quota_health:.3f}, resource confidence={resource.resource_confidence:.3f}",
        )
        estimated_input, estimated_output = self._estimated_tokens(profile, request)
        return _Scored(
            profile, score, confidence, estimated_input, estimated_output, estimated_cost,
            estimate.confidence, dict(estimate.components), estimate.assumptions, reasons, resource,
        )

    def _estimated_tokens(self, profile: ModelProfile, request: RoutingRequest) -> tuple[int, int]:
        estimate = self._token_estimate(profile, request)
        return estimate.input_tokens, estimate.output_tokens

    def _token_estimate(self, profile: ModelProfile, request: RoutingRequest):
        return self.accounting.estimate(TokenEstimationRequest(
            model_identity=ModelIdentity(profile.provider, profile.model_id),
            components={
                "task_description": request.task_description,
                **dict(request.estimation_components),
            },
            component_token_overrides={
                "declared_prompt_context": request.expected_prompt_tokens,
                "retrieved_context": request.retrieved_context_tokens,
            },
            route=request.task_type,
            lane=request.execution_lane,
            expected_tool_steps=request.expected_tool_calls,
            expected_model_calls=request.expected_model_calls,
            requested_output_tokens=request.expected_response_tokens or None,
            execution_kind=request.role,
            tool_count=request.expected_tool_calls,
            task_token_remaining=request.budgets.task_token_limit,
        ))

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

    def _estimate_cost(self, profile: ModelProfile, request: RoutingRequest) -> float | None:
        cost = self._token_estimate(profile, request).estimated_cost
        return None if cost is None else float(cost)

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

    def _competition_allowed(
        self,
        request: RoutingRequest,
        candidates: list[_Scored],
        *,
        independent_verifier: bool,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        if not request.multi_candidate_permitted or len(candidates) < 2 or self.policy.maximum_candidate_count < 2:
            return False, ("parallel candidates were not requested or fewer than two candidates qualified",)
        task_aware = bool(request.request_id or request.task_id)
        if task_aware:
            if not self.policy.parallel_execution_enabled or not request.parallel_execution_allowed:
                return False, ("parallel execution is disabled by gateway policy",)
            if not request.main_model_requested_parallel:
                return False, ("the main coordinating model did not request parallel candidates",)
            if not request.isolation_available:
                return False, ("isolated candidate execution is unavailable",)
            if not request.independent_verifier_available or not independent_verifier:
                return False, ("an independent qualified verifier is unavailable",)
            if request.active_task_conflict:
                return False, ("an active task ownership conflict prevents candidate execution",)
            if request.maximum_concurrency < 2:
                return False, ("task concurrency does not permit two candidates",)
            demand = max(level_value(request.complexity), level_value(request.risk))
            threshold = max(level_value(self.policy.competition_complexity_threshold), level_value(self.policy.competition_risk_threshold))
            if demand < threshold:
                return False, ("task complexity and risk do not justify parallel execution",)
        elif not request.explicit_competition:
            return False, ("parallel execution was not requested",)
        if request.latency_requirement is LatencyClass.INTERACTIVE:
            return False, ("latency requirement does not permit parallel execution",)
        evidence = self._parallel_evidence_score(request, candidates)
        if task_aware and evidence < self.policy.minimum_parallel_evidence:
            return False, (f"parallel evidence {evidence:.3f} is below {self.policy.minimum_parallel_evidence:.3f}",)
        limit = request.budgets.competition_cost_limit
        selected_costs = [item.estimated_cost for item in candidates[: self.policy.maximum_candidate_count]]
        projected = sum(item for item in selected_costs if item is not None) + self._verification_reserve(request)
        if limit is not None and any(item is None for item in selected_costs) and not request.budgets.allow_controlled_override:
            return False, ("candidate pricing is unknown for a cost-constrained competition",)
        if limit is not None and projected > limit and not request.budgets.allow_controlled_override:
            return False, (f"projected candidate and verification cost {projected:.6f} exceeds {limit:.6f}",)
        reasons.extend((
            f"parallel evidence {evidence:.3f} satisfies policy",
            f"{min(len(candidates), self.policy.maximum_candidate_count)} materially qualified candidates are available",
            "independent verification and isolated execution are available" if task_aware else "legacy validated competition request",
        ))
        return True, tuple(reasons)

    def _parallel_evidence_score(self, request: RoutingRequest, candidates: list[_Scored]) -> float:
        demand = self._effective_demand(request)
        initial_uncertainty = 1.0 - candidates[0].confidence
        failure_signal = min(1.0, request.similar_task_failures / 3)
        strategies = min(1.0, max(0, request.plausible_strategy_count - 1) / 2)
        diversity = min(1.0, len({item.profile.provider for item in candidates}) / 2)
        score = (
            0.25 * demand
            + 0.15 * initial_uncertainty
            + 0.15 * failure_signal
            + 0.15 * request.historical_result_variance
            + 0.15 * request.historical_parallel_benefit
            + 0.10 * strategies
            + 0.05 * diversity
        )
        if request.explicit_competition:
            score += 0.10
        return min(1.0, score)

    def _multi_agent_allowed(self, request: RoutingRequest) -> tuple[bool, tuple[str, ...]]:
        if not request.main_model_requested_multi_agent:
            return False, ("the main coordinating model did not request decomposition",)
        if not self.policy.multi_agent_enabled or not request.subagents_allowed:
            return False, ("multi-agent execution is disabled by gateway policy",)
        if request.task_tree_depth >= self.policy.maximum_task_tree_depth:
            return False, ("maximum task-tree depth would be exceeded",)
        if request.maximum_concurrency < 2:
            return False, ("task concurrency does not permit multiple agents",)
        if request.active_task_conflict:
            return False, ("an active ownership conflict prevents decomposition",)
        return True, ("main-model decomposition request passed gateway policy",)

    def _select_verifier(self, request: RoutingRequest, *, author: ModelProfile, candidates: list[_Scored]) -> tuple[ModelProfile | None, bool]:
        def fits(profile: ModelProfile) -> bool:
            try:
                estimate = self._token_estimate(profile, request)
            except (ModelContextLimitError, UnknownModelProfileError):
                return False
            cost = None if estimate.estimated_cost is None else float(estimate.estimated_cost)
            return (
                request.budgets.verification_cost_limit is None
                or (cost is not None and cost <= request.budgets.verification_cost_limit)
                or request.budgets.allow_controlled_override
            )

        qualified = [
            profile for profile in self.profiles
            if profile.available
            and profile.can_verify
            and profile.can_structured_output
            and ("verifier" in profile.supported_roles or "*" in profile.supported_roles)
            and not self._circuit_open(profile)
            and fits(profile)
        ]
        languages = set(request.repository.languages)
        qualified.sort(key=lambda profile: (profile.key == author.key, -(profile.reliability_score + profile.benchmark_scores.get("verification", 0.0)), -(len(languages & profile.supported_languages) if profile.supported_languages else len(languages)), profile.key))
        if not qualified:
            return None, False
        return qualified[0], qualified[0].key != author.key
