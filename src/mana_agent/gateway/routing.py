"""Gateway-owned authority for evidence-based model routing.

The authority is the only component that adds execution identity, persists the
request/decision pair, and publishes lifecycle events.  It deliberately wraps
the existing :class:`ModelRouter`; it is not a second routing policy.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import uuid
from typing import Any, Callable, Iterable

from mana_agent.config.model_capabilities import resolve_model_capability
from mana_agent.config.settings import Settings, mana_home
from mana_agent.config.catalog_service import ModelCatalogService
from mana_agent.config.inference_provider import resolve_inference_connection
from mana_agent.model_routing.history import JsonlRoutingHistory
from mana_agent.model_routing.models import (
    Complexity,
    ModelProfile,
    RiskLevel,
    RoutingDecision,
    RoutingFailure,
    RoutingOutcome,
    RoutingPolicy,
    RoutingRequest,
    sanitize_configuration,
)
from mana_agent.model_routing.profiles import configured_profiles, profiles_from_legacy_configuration
from mana_agent.model_routing.repository import RepositoryMetadataInspector
from mana_agent.model_routing.router import ModelRouter
from mana_agent.spirit.routing import (
    RoutedSpiritBinding,
    attach_spirit_accounting,
    bind_after_route,
    resolve_base_self,
)
from mana_agent.spirit.registry import default_spirit_ref
from mana_agent.spirit.schema import SpiritRef


class GatewayRoutingError(RoutingFailure):
    """Raised when the gateway cannot produce and persist a valid decision."""


class GatewayRoutingAuthority:
    """Own one configured router and durable decision journal per gateway."""

    def __init__(
        self,
        root: str | Path,
        *,
        settings: Settings | None = None,
        profiles: Iterable[ModelProfile] | None = None,
        event_sink: Callable[..., Any] | None = None,
        decision_path: Path | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.settings = settings or Settings()
        if not bool(getattr(self.settings, "mana_gateway_routing_enforced", True)):
            raise GatewayRoutingError(
                "Gateway-wide model routing enforcement is disabled. No model action was executed."
            )
        candidates = tuple(profiles) if profiles is not None else self._configured_profiles()
        if not candidates:
            raise GatewayRoutingError(
                "No model profiles are configured for gateway routing. No fallback model was selected."
            )
        self.policy = self._policy()
        self.history = JsonlRoutingHistory(
            mana_home() / "routing" / "outcomes.jsonl",
            retention_days=self.policy.evidence_retention_days,
        )
        self.router = ModelRouter(candidates, history=self.history, policy=self.policy)
        self.repository = RepositoryMetadataInspector().inspect(self.root)
        self.event_sink = event_sink
        self.decision_path = decision_path or (mana_home() / "routing" / "decisions.jsonl")
        self._write_lock = threading.Lock()
        self.context_cost_governor: Any | None = None
        self._spirit_by_task: dict[str, SpiritRef] = {}
        self._bindings: dict[str, RoutedSpiritBinding] = {}

    def binding_for(self, decision_id: str) -> RoutedSpiritBinding | None:
        return self._bindings.get(str(decision_id or ""))

    def _base_self_for(self, request: RoutingRequest):
        pinned = self._spirit_by_task.get(request.task_id) if request.task_id else None
        base_self = resolve_base_self(
            role=request.role,
            purpose=request.task_description,
            spirit=pinned,
            settings=self.settings,
        )
        if request.task_id:
            self._spirit_by_task[request.task_id] = base_self.ref()
        return base_self

    def route(
        self,
        request: RoutingRequest,
        *,
        fallback_of_decision_id: str = "",
        fallback_reason: str = "",
    ) -> RoutingDecision:
        """Route and persist one model-backed invocation with explicit retry lineage."""

        invocation_id = request.request_id or f"route_req_{uuid.uuid4().hex}"
        budgets = (
            self.context_cost_governor.remaining_routing_budgets(request.budgets)
            if self.context_cost_governor is not None
            else request.budgets
        )
        base_self = self._base_self_for(request)
        spirit_diag = {
            "spirit_id": base_self.spirit.id,
            "spirit_version": base_self.spirit.version,
            "agent_role": request.role,
        }
        enriched = attach_spirit_accounting(
            replace(
                request,
                request_id=invocation_id,
                repository=request.repository if request.repository.fingerprint else self.repository,
                budgets=budgets,
            ),
            base_self,
        )
        self._emit(
            "routing.requested",
            request_id=invocation_id,
            task_id=enriched.task_id,
            role=enriched.role,
            **spirit_diag,
        )
        try:
            decision = self.router.route(
                enriched,
                fallback_of_decision_id=fallback_of_decision_id,
                fallback_reason=fallback_reason,
            )
            if not decision.decision_id or decision.request_id != invocation_id:
                raise GatewayRoutingError("Router returned an invalid decision identity. No model action was executed.")
            if self.context_cost_governor is not None:
                try:
                    self.context_cost_governor.reserve_routing_children(decision)
                except ValueError as exc:
                    raise GatewayRoutingError(
                        f"Routing child budget allocation failed: {exc}. No candidate or subagent was started."
                    ) from exc
            binding = bind_after_route(base_self, decision)
            self._bindings[decision.decision_id] = binding
            self._persist(enriched, decision, spirit=base_self.ref())
        except RoutingFailure as exc:
            self._emit(
                "routing.failed",
                request_id=invocation_id,
                task_id=enriched.task_id,
                error=str(exc),
                **spirit_diag,
            )
            raise
        self._emit(
            "routing.completed",
            request_id=invocation_id,
            decision_id=decision.decision_id,
            task_id=enriched.task_id,
            provider=decision.provider,
            model=decision.selected_model,
            selected_provider=decision.provider,
            selected_model=decision.selected_model,
            routing_mode=decision.routing_mode.value,
            confidence=decision.confidence,
            **spirit_diag,
        )
        return decision

    def route_for_role(
        self,
        *,
        role: str,
        task_description: str,
        task_type: str,
        complexity: Complexity,
        risk: RiskLevel,
        **context: Any,
    ) -> RoutingDecision:
        return self.route(RoutingRequest(
            role=role,
            task_description=task_description,
            task_type=task_type,
            complexity=complexity,
            risk=risk,
            **context,
        ))

    def route_retry(
        self,
        request: RoutingRequest,
        *,
        previous_decision: RoutingDecision,
        failure_kind: str,
        verification_failed: bool = False,
    ) -> RoutingDecision:
        """Record failure evidence and require a fresh decision for a retry."""

        if previous_decision.task_id and request.task_id != previous_decision.task_id:
            raise GatewayRoutingError("Retry request does not match the failed task. No retry was executed.")
        self.router.record_outcome(RoutingOutcome(
            provider=previous_decision.provider,
            model_id=previous_decision.selected_model,
            model_configuration=previous_decision.model_configuration,
            task_category=request.task_type,
            repository_languages=request.repository.languages,
            repository_frameworks=request.repository.frameworks,
            complexity=request.complexity.value,
            risk=request.risk.value,
            routing_score=previous_decision.routing_score,
            selection_reason="; ".join(previous_decision.selection_reasons),
            estimated_cost=previous_decision.estimated_cost,
            verification_passed=False if verification_failed else None,
            accepted=False,
            retry_count=1,
            failure_kind=str(failure_kind or "provider_error"),
            decision_id=previous_decision.decision_id,
            request_id=previous_decision.request_id,
            task_id=previous_decision.task_id or request.task_id,
            fallback_of_decision_id=previous_decision.decision_id,
        ))
        self._emit(
            "routing.retry_requested",
            task_id=request.task_id,
            previous_decision_id=previous_decision.decision_id,
            failure_kind=failure_kind,
        )
        return self.route(replace(
            request,
            request_id="",
            previous_verification_failed=verification_failed,
        ), fallback_of_decision_id=previous_decision.decision_id, fallback_reason=str(failure_kind or "provider_error"))

    def latest(self, *, session_id: str = "", task_id: str = "") -> dict[str, Any] | None:
        rows = self.history_rows(limit=200)
        for row in reversed(rows):
            request = row.get("request") or {}
            if session_id and request.get("session_id") != session_id:
                continue
            if task_id and request.get("task_id") != task_id:
                continue
            return row
        return None

    def history_rows(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if not self.decision_path.exists():
            return []
        try:
            lines = self.decision_path.read_text(encoding="utf-8").splitlines()
            rows = [json.loads(line) for line in lines[-max(1, limit):] if line.strip()]
        except (OSError, json.JSONDecodeError):
            return []
        return [row for row in rows if isinstance(row, dict)]

    def health(self) -> dict[str, Any]:
        return {
            "enforced": bool(getattr(self.settings, "mana_gateway_routing_enforced", True)),
            "profiles": [profile.key for profile in self.router.profiles],
            "evidence_store_healthy": self.history.healthy(),
            "decision_store_healthy": self.decision_path.parent.is_dir() or not self.decision_path.exists(),
            "simple_default": self.policy.simple_routing_default,
            "multi_agent_enabled": self.policy.multi_agent_enabled,
            "parallel_execution_enabled": self.policy.parallel_execution_enabled,
        }

    def _persist(
        self,
        request: RoutingRequest,
        decision: RoutingDecision,
        *,
        spirit: SpiritRef | None = None,
    ) -> None:
        request_payload = asdict(request)
        request_payload["estimation_components"] = {
            str(key): "[accounted-without-content]"
            for key in request.estimation_components
        }
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request": sanitize_configuration(request_payload),
            "decision": sanitize_configuration(asdict(decision)),
            "spirit": (
                {"id": spirit.id, "version": spirit.version}
                if spirit is not None
                else default_spirit_ref().model_dump()
            ),
        }
        self.decision_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
            with self._write_lock, self.decision_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
        except OSError as exc:
            raise GatewayRoutingError(
                f"Routing decision could not be persisted: {exc}. No model action was executed."
            ) from exc

    def _configured_profiles(self) -> tuple[ModelProfile, ...]:
        explicit = configured_profiles(getattr(self.settings, "mana_model_profiles", []))
        if explicit:
            return explicit
        legacy = profiles_from_legacy_configuration(
            global_model=str(getattr(self.settings, "openai_chat_model", "gpt-4.1-mini") or "gpt-4.1-mini"),
            default_provider=str(getattr(self.settings, "mana_ai_provider", "openai") or "openai"),
            context_window=int(getattr(self.settings, "mana_context_unknown_model_context_window", 16_384)),
            max_output_tokens=int(getattr(self.settings, "mana_context_unknown_model_max_output_tokens", 4_096)),
        )
        try:
            connection = resolve_inference_connection(self.settings, require_api_key=False)
            descriptors = {
                item.id: item
                for item in ModelCatalogService().cached(provider=connection.provider, base_url=connection.base_url)
            }
        except (KeyError, ValueError):
            descriptors = {}
        resolved: list[ModelProfile] = []
        for profile in legacy:
            descriptor = descriptors.get(profile.model_id)
            if descriptor is None:
                cap_desc = resolve_model_capability(profile.provider, profile.model_id)
                resolved.append(replace(
                    profile,
                    can_patch=cap_desc.supports_repository_write if cap_desc.is_known else False,
                    can_structured_output=cap_desc.supports_structured_output if cap_desc.is_known else False,
                    can_tool_call=cap_desc.supports_tool_calls if cap_desc.is_known else False,
                    can_verify=(cap_desc.supports_tool_calls or cap_desc.supports_structured_output) if cap_desc.is_known else False,
                    supported_tools=(
                        frozenset({"*"}) if (cap_desc.is_known and cap_desc.supports_tool_calls and cap_desc.supports_repository_write)
                        else frozenset({"repository_read", "test_execution", "shell"} if (cap_desc.is_known and cap_desc.supports_tool_calls and cap_desc.supports_repository_read) else set())
                        if (cap_desc.is_known and cap_desc.supports_tool_calls)
                        else frozenset()
                    ),
                    capability_descriptor=cap_desc,
                    configuration={
                        **profile.configuration,
                        "capability_confidence": cap_desc.capability_confidence,
                        "capability_source": cap_desc.capability_source,
                        "capability_descriptor": cap_desc.to_dict(),
                    },
                ))
                continue
            metadata = dict(descriptor.metadata)
            context_window = descriptor.context_window or profile.context_window
            max_output_tokens = min(
                context_window,
                descriptor.max_output_tokens or profile.max_output_tokens,
            )
            complete_capabilities = (
                descriptor.context_window is not None
                and descriptor.max_output_tokens is not None
            )
            cap_desc = resolve_model_capability(
                profile.provider,
                profile.model_id,
                catalog_records=[metadata] if metadata else None,
            )
            resolved.append(replace(
                profile,
                context_window=context_window,
                max_output_tokens=max_output_tokens,
                tokenizer=descriptor.tokenizer,
                input_cost_per_million=float(metadata.get("input_price_per_million") or 0),
                output_cost_per_million=float(metadata.get("output_price_per_million") or 0),
                cached_input_cost_per_million=(float(metadata["cached_input_price_per_million"]) if metadata.get("cached_input_price_per_million") is not None else None),
                can_patch=cap_desc.supports_repository_write if cap_desc.is_known else False,
                can_structured_output=cap_desc.supports_structured_output if cap_desc.is_known else False,
                can_tool_call=cap_desc.supports_tool_calls if cap_desc.is_known else False,
                can_verify=(cap_desc.supports_tool_calls or cap_desc.supports_structured_output) if cap_desc.is_known else False,
                supported_tools=(
                    frozenset({"*"}) if (cap_desc.is_known and cap_desc.supports_tool_calls and cap_desc.supports_repository_write)
                    else frozenset({"repository_read", "test_execution", "shell"} if (cap_desc.is_known and cap_desc.supports_tool_calls and cap_desc.supports_repository_read) else set())
                    if (cap_desc.is_known and cap_desc.supports_tool_calls)
                    else frozenset()
                ),
                capability_descriptor=cap_desc,
                source_level=("provider-catalog" if complete_capabilities else "provider-catalog-with-configured-unknown-limit"),
                configuration={
                    **profile.configuration,
                    "token_profile_confidence": "high" if complete_capabilities else "low",
                    "capability_source": (
                        cap_desc.capability_source
                        if cap_desc.is_known
                        else ("provider-catalog" if complete_capabilities else "provider-catalog-with-configured-unknown-limit")
                    ),
                    "capability_confidence": cap_desc.capability_confidence,
                    "capability_descriptor": cap_desc.to_dict(),
                },
            ))
        return tuple(resolved)

    def _policy(self) -> RoutingPolicy:
        weights = getattr(self.settings, "mana_routing_benchmark_weights", {})
        return RoutingPolicy(
            enabled=bool(getattr(self.settings, "mana_adaptive_routing_enabled", True)),
            minimum_confidence=float(getattr(self.settings, "mana_routing_min_confidence", 0.55)),
            competition_complexity_threshold=Complexity(getattr(self.settings, "mana_routing_complexity_threshold", "high")),
            competition_risk_threshold=RiskLevel(getattr(self.settings, "mana_routing_risk_threshold", "high")),
            maximum_candidate_count=int(getattr(self.settings, "mana_routing_max_candidates", 2)),
            circuit_breaker_failures=int(getattr(self.settings, "mana_routing_circuit_breaker_failures", 3)),
            circuit_breaker_window_seconds=int(getattr(self.settings, "mana_routing_circuit_breaker_window_seconds", 900)),
            reliability_decay_seconds=int(getattr(self.settings, "mana_routing_reliability_decay_seconds", 3600)),
            model_failure_penalty_weight=float(getattr(self.settings, "mana_routing_model_failure_penalty_weight", 0.08)),
            provider_failure_penalty_weight=float(getattr(self.settings, "mana_routing_provider_failure_penalty_weight", 0.04)),
            evidence_retention_days=int(getattr(self.settings, "mana_routing_evidence_retention_days", 90)),
            simple_routing_default=bool(getattr(self.settings, "mana_routing_simple_default", True)),
            multi_agent_enabled=bool(getattr(self.settings, "mana_routing_multi_agent_enabled", False)),
            parallel_execution_enabled=bool(getattr(self.settings, "mana_routing_parallel_enabled", False)),
            minimum_parallel_evidence=float(getattr(self.settings, "mana_routing_min_parallel_evidence", 0.65)),
            maximum_task_tree_depth=int(getattr(self.settings, "mana_routing_max_task_tree_depth", 3)),
            maximum_concurrency=int(getattr(self.settings, "mana_routing_max_concurrent_tasks", 4)),
            task_timeout_seconds=int(getattr(self.settings, "mana_routing_task_timeout_seconds", 1800)),
            weights=dict(weights) if isinstance(weights, dict) and weights else RoutingPolicy().weights,
        )

    def _emit(self, event_type: str, **payload: Any) -> None:
        if not callable(self.event_sink):
            return
        try:
            self.event_sink(event_type, event_type.replace(".", " ").title(), metadata=payload)
        except TypeError:
            self.event_sink(event_type, payload)


__all__ = ["GatewayRoutingAuthority", "GatewayRoutingError"]
