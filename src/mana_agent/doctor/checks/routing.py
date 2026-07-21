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
    }
    severity = Severity.ERROR if incomplete or not profiles else Severity.INFO
    message = f"{len(profiles)} routing candidate(s); evidence store {'healthy' if history.healthy() else 'unhealthy'}; independent verifier {'available' if independent else 'unavailable'}; isolated competition {'available' if isolation else 'unavailable'}."
    return [DoctorFinding(
        "routing/models", severity, "Adaptive model routing", message,
        "Complete model capability metadata before enabling routing." if incomplete else None,
        details=details,
    )]
