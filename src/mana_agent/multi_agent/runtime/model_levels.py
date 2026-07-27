from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from mana_agent.config.user_config import get_setting
from mana_agent.model_routing.history import JsonlRoutingHistory
from mana_agent.model_routing.models import Complexity, LatencyClass, RepositoryMetadata, RiskLevel, RoutingBudgets, RoutingDecision, RoutingRequest
from mana_agent.model_routing.profiles import configured_profiles, profiles_from_legacy_configuration
from mana_agent.model_routing.router import ModelRouter
from mana_agent.multi_agent.core.types import AgentRole

MODEL_LEVEL_3_HIGH_REASONING = "MODEL_LEVEL_3_HIGH_REASONING"
MODEL_LEVEL_2_CODING = "MODEL_LEVEL_2_CODING"
MODEL_LEVEL_1_FAST_TOOL = "MODEL_LEVEL_1_FAST_TOOL"

_DEFAULT_MODEL_LEVELS = {
    AgentRole.MAIN: ("MANA_MODEL_MAIN", MODEL_LEVEL_3_HIGH_REASONING),
    AgentRole.HEAD_DECISION: ("MANA_MODEL_HEAD_DECISION", MODEL_LEVEL_3_HIGH_REASONING),
    AgentRole.PLANNER: ("MANA_MODEL_PLANNER", MODEL_LEVEL_3_HIGH_REASONING),
    AgentRole.CODING: ("MANA_MODEL_CODING", MODEL_LEVEL_2_CODING),
    AgentRole.VERIFIER: ("MANA_MODEL_VERIFIER", MODEL_LEVEL_2_CODING),
    AgentRole.REVIEWER: ("MANA_MODEL_REVIEWER", MODEL_LEVEL_3_HIGH_REASONING),
    AgentRole.TOOL: ("MANA_MODEL_TOOL", MODEL_LEVEL_1_FAST_TOOL),
    AgentRole.TOOL_WORKER: ("MANA_MODEL_TOOL_WORKER", MODEL_LEVEL_1_FAST_TOOL),
    AgentRole.RESEARCH: ("MANA_MODEL_TOOL", MODEL_LEVEL_1_FAST_TOOL),
    AgentRole.SUMMARIZER: ("MANA_MODEL_SUMMARIZER", MODEL_LEVEL_1_FAST_TOOL),
}


@dataclass(frozen=True)
class ModelLevelAssignment:
    role: AgentRole
    env_var: str
    model_level: str


def model_level_for_role(role: AgentRole) -> ModelLevelAssignment:
    env_var, default = _DEFAULT_MODEL_LEVELS[role]
    return ModelLevelAssignment(role=role, env_var=env_var, model_level=str(get_setting(env_var, default) or default))


@dataclass(frozen=True)
class ResolvedModelAssignment:
    role: AgentRole
    env_var: str
    model_level: str
    resolved_model: str
    routing_decision: RoutingDecision | None = None


def _is_symbolic_model_level(value: str) -> bool:
    return str(value or "").strip().startswith("MODEL_LEVEL_")


_ROLE_TASK = {
    AgentRole.MAIN: ("routing", Complexity.HIGH, RiskLevel.MEDIUM),
    AgentRole.HEAD_DECISION: ("routing", Complexity.HIGH, RiskLevel.MEDIUM),
    AgentRole.PLANNER: ("planning", Complexity.HIGH, RiskLevel.MEDIUM),
    AgentRole.CODING: ("coding", Complexity.MEDIUM, RiskLevel.MEDIUM),
    AgentRole.VERIFIER: ("verification", Complexity.HIGH, RiskLevel.HIGH),
    AgentRole.REVIEWER: ("review", Complexity.HIGH, RiskLevel.HIGH),
    AgentRole.TOOL: ("tool", Complexity.LOW, RiskLevel.LOW),
    AgentRole.TOOL_WORKER: ("tool", Complexity.LOW, RiskLevel.LOW),
    AgentRole.RESEARCH: ("research", Complexity.MEDIUM, RiskLevel.LOW),
    AgentRole.SUMMARIZER: ("summarization", Complexity.LOW, RiskLevel.LOW),
}


def _language_preference(value) -> frozenset[str]:
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = value or ()
    return frozenset(str(item).strip().lower() for item in values if str(item).strip())


def route_model(request: RoutingRequest, *, global_model: str, profiles=None) -> RoutingDecision:
    """Central validated entry point for gateway, agents, subagents, and retries."""
    if profiles is not None:
        candidates = tuple(profiles)
        policy = None
    else:
        from mana_agent.config.settings import Settings, mana_home

        settings = Settings()
        explicit = configured_profiles(settings.mana_model_profiles)
        candidates = explicit or profiles_from_legacy_configuration(global_model=global_model, default_provider=settings.mana_ai_provider)
        language_preferences = settings.mana_routing_language_preferences if isinstance(settings.mana_routing_language_preferences, dict) else {}
        candidates = tuple(
            replace(candidate, supported_languages=_language_preference(language_preferences.get(candidate.key, candidate.supported_languages)))
            for candidate in candidates
        )
        weights = settings.mana_routing_benchmark_weights if isinstance(settings.mana_routing_benchmark_weights, dict) else {}
        from mana_agent.model_routing.models import RoutingPolicy

        policy = RoutingPolicy(
            enabled=settings.mana_adaptive_routing_enabled,
            minimum_confidence=settings.mana_routing_min_confidence,
            competition_complexity_threshold=Complexity(settings.mana_routing_complexity_threshold),
            competition_risk_threshold=RiskLevel(settings.mana_routing_risk_threshold),
            maximum_candidate_count=settings.mana_routing_max_candidates,
            circuit_breaker_failures=settings.mana_routing_circuit_breaker_failures,
            circuit_breaker_window_seconds=settings.mana_routing_circuit_breaker_window_seconds,
            reliability_decay_seconds=settings.mana_routing_reliability_decay_seconds,
            model_failure_penalty_weight=settings.mana_routing_model_failure_penalty_weight,
            provider_failure_penalty_weight=settings.mana_routing_provider_failure_penalty_weight,
            evidence_retention_days=settings.mana_routing_evidence_retention_days,
            weights=weights or RoutingPolicy().weights,
        )
        history = JsonlRoutingHistory(mana_home() / "routing" / "outcomes.jsonl", retention_days=policy.evidence_retention_days)
    return ModelRouter(candidates, policy=policy, history=history if profiles is None else None).route(request)


def routing_budgets_from_settings(settings) -> RoutingBudgets:
    return RoutingBudgets(
        task_token_limit=getattr(settings, "mana_routing_task_token_budget", 32_000),
        task_cost_limit=getattr(settings, "mana_routing_task_cost_budget", None),
        session_cost_remaining=getattr(settings, "mana_routing_session_cost_budget", None),
        competition_cost_limit=getattr(settings, "mana_routing_competition_cost_budget", None),
        verification_cost_limit=getattr(settings, "mana_routing_verification_cost_budget", None),
        retry_cost_limit=getattr(settings, "mana_routing_retry_cost_budget", None),
        verification_reserve_ratio=getattr(settings, "mana_routing_verification_reserve_ratio", 0.15),
    )


def resolve_model_for_role(
    role: AgentRole,
    *,
    global_model: str,
    repository: RepositoryMetadata | None = None,
    task_description: str | None = None,
    routing_authority=None,
    task_id: str = "",
    parent_task_id: str | None = None,
    session_id: str = "",
    workspace_id: str = "",
    repository_id: str = "",
    execution_lane: str = "",
) -> ResolvedModelAssignment:
    env_var, default_level = _DEFAULT_MODEL_LEVELS[role]
    role_value = str(get_setting(env_var, "") or "").strip()
    configured = role_value or default_level
    configured_target = str(get_setting(configured, "") or "").strip() if _is_symbolic_model_level(configured) else role_value
    migration_target = configured_target or str(global_model or "").strip()
    task_type, complexity, risk = _ROLE_TASK[role]
    from mana_agent.config.settings import Settings
    settings = Settings()
    request = RoutingRequest(
            role=role.value,
            task_description=task_description or f"Initialize the {role.value} execution lane.",
            task_type=task_type,
            complexity=complexity,
            risk=risk,
            repository=repository or RepositoryMetadata(),
            latency_requirement=LatencyClass.STANDARD if complexity is not Complexity.LOW else LatencyClass.INTERACTIVE,
            budgets=routing_budgets_from_settings(settings),
            required_capabilities=frozenset({"structured_output"}) if role in {AgentRole.MAIN, AgentRole.HEAD_DECISION, AgentRole.PLANNER, AgentRole.REVIEWER, AgentRole.VERIFIER} else frozenset(),
            task_id=task_id,
            parent_task_id=parent_task_id,
            session_id=session_id,
            workspace_id=workspace_id,
            repository_id=repository_id,
            execution_lane=execution_lane or role.value,
        )
    decision = (
        routing_authority.route(request)
        if routing_authority is not None
        else route_model(request, global_model=migration_target)
    )
    return ResolvedModelAssignment(
        role=role,
        env_var=env_var,
        model_level=configured if _is_symbolic_model_level(configured) else default_level,
        resolved_model=decision.selected_model,
        routing_decision=decision,
    )
