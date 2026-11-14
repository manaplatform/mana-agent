from __future__ import annotations

from dataclasses import asdict, replace
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
            raise RoutingFailure("Model routing is disabled; no fallback was selected")
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
            error = NoWriteCapableModelAvailableError if request.required_capabilities & {"patch", "repository_write"} else RoutingFailure
            raise error("No configured model has current evidence for the required capabilities and budgets. Refresh metadata, configure the model, or initiate a capability test.", rejected=tuple(rejected))
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
        if model is None:
            from mana_agent.config.settings import Settings
            from mana_agent.config.inference_provider import resolve_inference_connection
            from mana_agent.config.provider_registry import split_qualified_model_id
            from mana_agent.llm.engine import create_model
            settings = Settings()
            selected = str(settings.mana_primary_model or settings.openai_chat_model or "")
            if not selected:
                raise RoutingFailure("Configure the Main model to select execution models; no default was selected")
            provider, model_id = split_qualified_model_id(selected, default_provider=settings.mana_ai_provider)
            connection = resolve_inference_connection(settings, provider=provider)
            model = create_model(provider=provider, model=model_id, api_key=connection.api_key,
                                 base_url=connection.base_url, default_headers=connection.headers)
            model.context_cost_governor = self.context_cost_governor
        from mana_agent.llm.messages import SystemMessage, UserMessage
        offered = []
        for key, (candidate, resource) in candidates.items():
            descriptor = candidate.capability_descriptor
            estimate = self._token_estimate(candidate, request)
            offered.append({
                "id": key,
                "capabilities": {name: descriptor.fact(name).model_dump(mode="json") for name in sorted(descriptor.facts)},
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
        estimates = [self._token_estimate(item, request) for item in selected_profiles]
        if verifier is not None:
            estimates.append(self._token_estimate(verifier, verification_request))
        cost_limits = [value for value in (request.budgets.task_cost_limit, request.budgets.session_cost_remaining,
            request.budgets.competition_cost_limit if competition else None) if value is not None]
        if cost_limits and (any(item.estimated_cost is None for item in estimates) or
                sum(float(item.estimated_cost) for item in estimates) > min(cost_limits)):
            raise RoutingFailure("Selected execution models exceed the combined operation budget")
        if request.budgets.task_token_limit is not None and sum(item.total_tokens for item in estimates) > request.budgets.task_token_limit:
            raise RoutingFailure("Selected execution models exceed the combined token budget")
        estimate = self._token_estimate(profile, request)
        request_id = request.request_id or self._stable_id("request", request)
        decision_id = self._stable_id("decision", {"request_id": request_id, "choice": choice.model_dump(mode="json")})
        record_current("model.candidate.selected", {"decision_id": decision_id, "task_id": request.task_id,
            "provider": profile.provider, "model": profile.model_id, "reason": choice.reason, "selection_source": "main_model"})
        return RoutingDecision(selected_model=profile.model_id, provider=profile.provider,
            model_configuration=sanitize_configuration(profile.configuration), selected_role=request.role,
            routing_score=choice.confidence, confidence=choice.confidence,
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
        if (request.budgets.task_cost_limit is not None or request.budgets.session_cost_remaining is not None) and not profile.configuration.get("pricing_known"):
            reasons.append("pricing is UNKNOWN for the cost-constrained operation")
        desc = getattr(profile, "capability_descriptor", None)
        if desc is None or not desc.is_known:
            reasons.append(
                f"model capabilities are unknown (provider={profile.provider!r}, model={profile.model_id!r}, transport={getattr(desc, 'transport', '')!r})"
            )
        if desc is not None and (not desc.context_window or not desc.max_output_tokens or not desc.supports("input_text") or not desc.supports("output_text")):
            reasons.append("Text or token-limit evidence is absent or expired")
        missing_tools = request.required_tools - profile.supported_tools
        if request.required_tools and (not profile.can_tool_call or not desc or not desc.supports_tool_calls):
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
            "structured_output": profile.can_structured_output and bool(desc and desc.supports_structured_output),
            "tool_calls": profile.can_tool_call and bool(desc and desc.supports_tool_calls),
            "verification": profile.can_verify,
            "reasoning": bool(desc and desc.supports_reasoning),
            "repository_write": supports_write,
            "repository_read": supports_read,
            "shell": desc.supports_shell if desc is not None else profile.can_tool_call,
            "streaming": desc.supports_streaming if desc is not None else False,
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
        if limits and estimated is None:
            reasons.append("model pricing is unknown for a cost-constrained request")
        elif limits and estimated is not None and float(estimated) > max(0.0, min(limits) - reserve) and not request.budgets.allow_controlled_override:
            reasons.append(f"estimated implementation cost {estimated:.6f} exceeds budget after verification reserve {reserve:.6f}")
        if estimate is not None:
            total_tokens = estimate.total_tokens
            if request.budgets.task_token_limit is not None and total_tokens > request.budgets.task_token_limit:
                reasons.append(f"estimated tokens {total_tokens} exceed task limit {request.budgets.task_token_limit}")
        return reasons



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




    def _verification_reserve(self, request: RoutingRequest) -> float:
        limits = [value for value in (request.budgets.task_cost_limit, request.budgets.session_cost_remaining) if value is not None]
        if not limits:
            return 0.0
        configured = request.budgets.verification_cost_limit
        ratio_reserve = min(limits) * request.budgets.verification_reserve_ratio
        return min(ratio_reserve, configured) if configured is not None else ratio_reserve


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




