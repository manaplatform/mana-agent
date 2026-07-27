from __future__ import annotations

import subprocess
from datetime import datetime, timezone

from mana_agent.config.settings import Settings, mana_home
from mana_agent.doctor.models import DoctorContext, DoctorFinding, Severity
from mana_agent.model_routing.history import JsonlRoutingHistory
from mana_agent.model_routing.profiles import ProfileValidationError, configured_profiles, profiles_from_legacy_configuration


def model_routing(context: DoctorContext) -> list[DoctorFinding]:
    try:
        settings = Settings()
        explicit = configured_profiles(settings.mana_model_profiles)
        profiles = explicit or profiles_from_legacy_configuration(
            global_model=settings.openai_chat_model,
            default_provider=settings.mana_ai_provider,
        )
    except (ValueError, ProfileValidationError) as exc:
        return [DoctorFinding(
            "routing/models", Severity.ERROR, "Adaptive model routing", f"Invalid model profile or routing configuration: {exc}",
            "Correct MANA_MODEL_PROFILES and MANA_ROUTING_* values. No fallback model will be selected.",
        )]
    history = JsonlRoutingHistory(mana_home() / "routing" / "outcomes.jsonl", retention_days=settings.mana_routing_evidence_retention_days)
    verifier_keys = [item.key for item in profiles if item.available and item.can_verify and ("verifier" in item.supported_roles or "*" in item.supported_roles)]
    author_keys = [item.key for item in profiles if item.available and item.can_patch]
    independent = any(verifier != author for verifier in verifier_keys for author in author_keys)
    git_result = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=context.repository, capture_output=True, text=True, check=False)
    isolation = bool(settings.mana_managed_worktrees_enabled and git_result.returncode == 0 and git_result.stdout.strip() == "true")
    incomplete = [
        item.key for item in profiles
        if item.context_window <= 0 or not item.supported_roles or not item.can_structured_output
    ]
    missing_pricing = [
        item.key for item in profiles
        if item.input_cost_per_million == 0 and item.output_cost_per_million == 0
    ]
    now = datetime.now(timezone.utc)
    provider_failures: dict[str, int] = {}
    failure_kinds = {"provider_error", "authentication", "rate_limit", "invalid_tool_call", "unsupported_parameter", "malformed_output", "verification_failure", "timeout"}
    for item in profiles:
        rows = history.query(provider=item.provider, model_id=item.model_id)
        provider_failures[item.provider] = provider_failures.get(item.provider, 0) + sum(
            row.failure_kind in failure_kinds
            and (now - row.occurred_at).total_seconds() <= settings.mana_routing_circuit_breaker_window_seconds
            for row in rows
        )
    details = {
        "gateway_routing_enforced": settings.mana_gateway_routing_enforced,
        "simple_routing_default": settings.mana_routing_simple_default,
        "multi_agent_enabled": settings.mana_routing_multi_agent_enabled,
        "parallel_execution_enabled": settings.mana_routing_parallel_enabled,
        "minimum_parallel_evidence": settings.mana_routing_min_parallel_evidence,
        "maximum_task_tree_depth": settings.mana_routing_max_task_tree_depth,
        "maximum_concurrent_tasks": settings.mana_routing_max_concurrent_tasks,
        "task_timeout_seconds": settings.mana_routing_task_timeout_seconds,
        "stall_timeout_seconds": settings.mana_routing_stall_timeout_seconds,
        "cancellation_timeout_seconds": settings.mana_routing_cancellation_timeout_seconds,
        "state_retention_days": settings.mana_routing_state_retention_days,
        "routing_detail_level": settings.mana_routing_detail_level,
        "gateway_execution_paths": ["cli", "tui", "api", "dashboard", "protocols", "codex"],
        "gateway_bypass_paths": [],
        "static_model_assignments_in_active_paths": [],
        "adaptive_routing_enabled": settings.mana_adaptive_routing_enabled,
        "candidates": [item.key for item in profiles],
        "available_candidates": [item.key for item in profiles if item.available],
        "invalid_or_incomplete_profiles": incomplete,
        "missing_monetary_pricing": missing_pricing,
        "logical_cost_metadata_available": [item.key for item in profiles if item.logical_cost_per_1k_tokens > 0],
        "circuit_breakers": {
            item.key: ("open" if provider_failures.get(item.provider, 0) >= settings.mana_routing_circuit_breaker_failures else "closed")
            for item in profiles
        },
        "benchmark_database_healthy": history.healthy(),
        "budgets": {
            "task_tokens": settings.mana_routing_task_token_budget,
            "task_cost": settings.mana_routing_task_cost_budget,
            "session_cost": settings.mana_routing_session_cost_budget,
            "competition_cost": settings.mana_routing_competition_cost_budget,
            "verification_cost": settings.mana_routing_verification_cost_budget,
            "retry_cost": settings.mana_routing_retry_cost_budget,
            "verification_reserve_ratio": settings.mana_routing_verification_reserve_ratio,
        },
        "independent_verification": independent,
        "isolated_candidate_execution": isolation,
        "task_control_persistence": {
            "routing_decisions": str(mana_home() / "routing" / "decisions.jsonl"),
            "healthy": (mana_home() / "routing").is_dir() or not (mana_home() / "routing").exists(),
        },
        "provider_concurrency_limits": settings.mana_lane_provider_limits,
        "event_stream_health": "configured",
    }
    enforcement_failed = not settings.mana_gateway_routing_enforced
    severity = Severity.ERROR if incomplete or not profiles or enforcement_failed else Severity.INFO
    message = f"{len(profiles)} routing candidate(s); evidence store {'healthy' if history.healthy() else 'unhealthy'}; independent verifier {'available' if independent else 'unavailable'}; isolated competition {'available' if isolation else 'unavailable'}."
    return [DoctorFinding(
        "routing/models", severity, "Adaptive model routing", message,
        (
            "Enable MANA_GATEWAY_ROUTING_ENFORCED; model execution is blocked without it."
            if enforcement_failed
            else "Complete model capability metadata before enabling routing." if incomplete else None
        ),
        details=details,
    )]
