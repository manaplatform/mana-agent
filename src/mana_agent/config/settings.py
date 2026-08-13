from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mana_agent.config.user_config import settings_source_for_pydantic


MANA_ROOT_DIRNAME = ".mana"


def mana_home() -> Path:
    """Return Mana-Agent's user-level state directory.

    Repository source trees are deliberately not used as state stores.  Tests
    and managed installations can isolate state with ``MANA_HOME``.

    Windows ``Path.resolve`` always calls ``os.getcwd`` via ``ntpath.realpath``;
    when the process CWD is gone we still return a usable absolute path.
    """

    configured = str(os.getenv("MANA_HOME") or "").strip()
    candidate = (
        Path(configured).expanduser()
        if configured
        else (Path.home() / MANA_ROOT_DIRNAME)
    )
    try:
        return candidate.resolve(strict=False)
    except (FileNotFoundError, OSError, RuntimeError):
        if candidate.is_absolute():
            return Path(os.path.normpath(candidate))
        return candidate


# Default embedding models per provider. The chat and embedding endpoints share a
# single base URL, so when no embedding model is configured explicitly we pick a
# provider-appropriate default based on that URL (an OpenAI embedding model does
# not exist on NVIDIA's API and vice versa).
OPENAI_DEFAULT_EMBED_MODEL = "text-embedding-3-small"
NVIDIA_DEFAULT_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"


def resolve_embed_model(
    base_url: str | None,
    explicit_model: str | None = None,
    *,
    provider: str | None = None,
) -> str:
    """Return the embedding model to use for the given base URL / provider.

    An explicitly configured model always wins. Otherwise the model is inferred
    from the provider or base URL: NVIDIA endpoints get an NVIDIA embedding
    model, everything else falls back to the OpenAI default.
    """
    if explicit_model and explicit_model.strip():
        return explicit_model.strip()
    provider_id = str(provider or "").strip().lower()
    if provider_id == "nvidia" or (base_url and "nvidia" in base_url.lower()):
        return NVIDIA_DEFAULT_EMBED_MODEL
    return OPENAI_DEFAULT_EMBED_MODEL


class Settings(BaseSettings):
    mana_config_schema_version: int = Field(default=2, alias="MANA_CONFIG_SCHEMA_VERSION")
    mana_ai_provider: str = Field(default="openai", alias="MANA_AI_PROVIDER")
    mana_primary_model: str = Field(
        default="openai/gpt-4.1-mini", alias="MANA_PRIMARY_MODEL"
    )
    mana_embedding_model: str = Field(
        default="openai/text-embedding-3-small", alias="MANA_EMBEDDING_MODEL"
    )
    spirit: dict[str, Any] | str = Field(default_factory=dict, alias="spirit")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    openrouter_http_referer: str = Field(
        default="https://github.com/mana-agent/mana-agent",
        alias="OPENROUTER_HTTP_REFERER",
    )
    openrouter_title: str = Field(default="Mana-Agent", alias="OPENROUTER_TITLE")
    openrouter_provider_preferences: dict[str, Any] | str = Field(
        default_factory=dict, alias="OPENROUTER_PROVIDER_PREFERENCES"
    )
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL"
    )
    openai_chat_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_CHAT_MODEL")
    openai_tool_worker_model: str | None = Field(
        default=None, alias="OPENAI_TOOL_WORKER_MODEL"
    )
    openai_coding_planner_model: str | None = Field(
        default=None, alias="OPENAI_CODING_PLANNER_MODEL"
    )
    # Left unset by default so the embedding model can be auto-selected from the
    # active base URL (see ``resolve_embed_model``). An explicit value always wins.
    openai_embed_model: str | None = Field(default=None, alias="OPENAI_EMBED_MODEL")
    mana_adaptive_routing_enabled: bool = Field(
        default=True, alias="MANA_ADAPTIVE_ROUTING_ENABLED"
    )
    mana_model_profiles: list[dict[str, Any]] | str = Field(
        default_factory=list, alias="MANA_MODEL_PROFILES"
    )
    mana_routing_complexity_threshold: str = Field(
        default="high", alias="MANA_ROUTING_COMPLEXITY_THRESHOLD"
    )
    mana_routing_risk_threshold: str = Field(
        default="high", alias="MANA_ROUTING_RISK_THRESHOLD"
    )
    mana_routing_max_candidates: int = Field(
        default=2, alias="MANA_ROUTING_MAX_CANDIDATES"
    )
    mana_routing_min_confidence: float = Field(
        default=0.55, alias="MANA_ROUTING_MIN_CONFIDENCE"
    )
    mana_routing_task_token_budget: int = Field(
        default=32_000, alias="MANA_ROUTING_TASK_TOKEN_BUDGET"
    )
    mana_routing_task_cost_budget: float | None = Field(
        default=None, alias="MANA_ROUTING_TASK_COST_BUDGET"
    )
    mana_routing_session_cost_budget: float | None = Field(
        default=None, alias="MANA_ROUTING_SESSION_COST_BUDGET"
    )
    mana_routing_competition_cost_budget: float | None = Field(
        default=None, alias="MANA_ROUTING_COMPETITION_COST_BUDGET"
    )
    mana_routing_verification_cost_budget: float | None = Field(
        default=None, alias="MANA_ROUTING_VERIFICATION_COST_BUDGET"
    )
    mana_routing_retry_cost_budget: float | None = Field(
        default=None, alias="MANA_ROUTING_RETRY_COST_BUDGET"
    )
    mana_routing_verification_reserve_ratio: float = Field(
        default=0.15, alias="MANA_ROUTING_VERIFICATION_RESERVE_RATIO"
    )
    mana_context_governor_enabled: bool = Field(
        default=True, alias="MANA_CONTEXT_GOVERNOR_ENABLED"
    )
    mana_context_governor_mode: Literal["observe", "soft", "enforce"] = Field(
        default="observe", alias="MANA_CONTEXT_GOVERNOR_MODE"
    )
    mana_context_warning_ratio: float = Field(
        default=0.70, ge=0.0, le=1.0, alias="MANA_CONTEXT_WARNING_RATIO"
    )
    mana_context_compact_ratio: float = Field(
        default=0.80, ge=0.0, le=1.0, alias="MANA_CONTEXT_COMPACT_RATIO"
    )
    mana_context_max_utilization: float = Field(
        default=0.85, ge=0.0, le=1.0, alias="MANA_CONTEXT_MAX_UTILIZATION"
    )
    mana_context_hard_limit_ratio: float = Field(
        default=0.95, ge=0.0, le=1.0, alias="MANA_CONTEXT_HARD_LIMIT_RATIO"
    )
    mana_context_response_reserve_ratio: float = Field(
        default=0.12, ge=0.0, le=1.0, alias="MANA_CONTEXT_RESPONSE_RESERVE_RATIO"
    )
    mana_context_response_reserve_tokens: int = Field(
        default=0, ge=0, alias="MANA_CONTEXT_RESPONSE_RESERVE_TOKENS"
    )
    mana_context_tool_result_max_tokens: int = Field(
        default=2_000, ge=1, alias="MANA_CONTEXT_TOOL_RESULT_MAX_TOKENS"
    )
    mana_context_history_max_tokens: int = Field(
        default=8_000, ge=1, alias="MANA_CONTEXT_HISTORY_MAX_TOKENS"
    )
    mana_context_retrieval_max_tokens: int = Field(
        default=12_000, ge=1, alias="MANA_CONTEXT_RETRIEVAL_MAX_TOKENS"
    )
    mana_context_lazy_capabilities: bool = Field(
        default=True, alias="MANA_CONTEXT_LAZY_CAPABILITIES"
    )
    mana_context_capability_idle_steps: int = Field(
        default=3, ge=1, alias="MANA_CONTEXT_CAPABILITY_IDLE_STEPS"
    )
    mana_context_artifact_retention_days: int = Field(
        default=30, ge=1, alias="MANA_CONTEXT_ARTIFACT_RETENTION_DAYS"
    )
    mana_context_cost_log_enabled: bool = Field(
        default=True, alias="MANA_CONTEXT_COST_LOG_ENABLED"
    )
    mana_context_cost_log_retention_days: int = Field(
        default=30, ge=1, alias="MANA_CONTEXT_COST_LOG_RETENTION_DAYS"
    )
    mana_context_estimation_safety_margin_ratio: float = Field(
        default=0.05, ge=0.0, le=0.5, alias="MANA_CONTEXT_ESTIMATION_SAFETY_MARGIN_RATIO"
    )
    mana_context_default_output_ratio: float = Field(
        default=0.20, gt=0.0, le=1.0, alias="MANA_CONTEXT_DEFAULT_OUTPUT_RATIO"
    )
    mana_context_historical_prediction_enabled: bool = Field(
        default=True, alias="MANA_CONTEXT_HISTORICAL_PREDICTION_ENABLED"
    )
    mana_context_unknown_model_policy: Literal["require_metadata", "conservative"] = Field(
        default="conservative", alias="MANA_CONTEXT_UNKNOWN_MODEL_POLICY"
    )
    mana_context_unknown_model_context_window: int = Field(
        default=16_384, ge=1, alias="MANA_CONTEXT_UNKNOWN_MODEL_CONTEXT_WINDOW"
    )
    mana_context_unknown_model_max_output_tokens: int = Field(
        default=4_096, ge=1, alias="MANA_CONTEXT_UNKNOWN_MODEL_MAX_OUTPUT_TOKENS"
    )
    mana_routing_benchmark_weights: dict[str, float] | str = Field(
        default_factory=dict, alias="MANA_ROUTING_BENCHMARK_WEIGHTS"
    )
    mana_routing_language_preferences: dict[str, Any] | str = Field(
        default_factory=dict, alias="MANA_ROUTING_LANGUAGE_PREFERENCES"
    )
    mana_routing_evidence_retention_days: int = Field(
        default=90, alias="MANA_ROUTING_EVIDENCE_RETENTION_DAYS"
    )
    mana_routing_circuit_breaker_failures: int = Field(
        default=3, alias="MANA_ROUTING_CIRCUIT_BREAKER_FAILURES"
    )
    mana_routing_circuit_breaker_window_seconds: int = Field(
        default=900, alias="MANA_ROUTING_CIRCUIT_BREAKER_WINDOW_SECONDS"
    )
    mana_routing_reliability_decay_seconds: int = Field(
        default=3600, alias="MANA_ROUTING_RELIABILITY_DECAY_SECONDS"
    )
    mana_routing_model_failure_penalty_weight: float = Field(
        default=0.08, alias="MANA_ROUTING_MODEL_FAILURE_PENALTY_WEIGHT"
    )
    mana_routing_provider_failure_penalty_weight: float = Field(
        default=0.04, alias="MANA_ROUTING_PROVIDER_FAILURE_PENALTY_WEIGHT"
    )
    mana_gateway_routing_enforced: bool = Field(
        default=True, alias="MANA_GATEWAY_ROUTING_ENFORCED"
    )
    mana_routing_simple_default: bool = Field(
        default=True, alias="MANA_ROUTING_SIMPLE_DEFAULT"
    )
    mana_routing_multi_agent_enabled: bool = Field(
        default=False, alias="MANA_ROUTING_MULTI_AGENT_ENABLED"
    )
    mana_routing_parallel_enabled: bool = Field(
        default=False, alias="MANA_ROUTING_PARALLEL_ENABLED"
    )
    mana_routing_min_parallel_evidence: float = Field(
        default=0.65, alias="MANA_ROUTING_MIN_PARALLEL_EVIDENCE"
    )
    mana_routing_max_task_tree_depth: int = Field(
        default=3, alias="MANA_ROUTING_MAX_TASK_TREE_DEPTH"
    )
    mana_routing_max_concurrent_tasks: int = Field(
        default=4, alias="MANA_ROUTING_MAX_CONCURRENT_TASKS"
    )
    mana_routing_task_timeout_seconds: int = Field(
        default=1800, alias="MANA_ROUTING_TASK_TIMEOUT_SECONDS"
    )
    mana_routing_stall_timeout_seconds: int = Field(
        default=300, alias="MANA_ROUTING_STALL_TIMEOUT_SECONDS"
    )
    mana_routing_cancellation_timeout_seconds: int = Field(
        default=30, alias="MANA_ROUTING_CANCELLATION_TIMEOUT_SECONDS"
    )
    mana_routing_state_retention_days: int = Field(
        default=30, alias="MANA_ROUTING_STATE_RETENTION_DAYS"
    )
    mana_routing_detail_level: str = Field(
        default="concise", alias="MANA_ROUTING_DETAIL_LEVEL"
    )
    llm_model: str | None = Field(default=None, alias="LLM_MODEL")
    mana_llm_log_file: str | None = Field(default=None, alias="MANA_LLM_LOG_FILE")
    default_top_k: int = Field(default=8, alias="DEFAULT_TOP_K")
    coding_flow_max_turns: int = Field(default=5, alias="CODING_FLOW_MAX_TURNS")
    coding_flow_max_tasks: int = Field(default=20, alias="CODING_FLOW_MAX_TASKS")
    coding_plan_max_steps: int = Field(default=8, alias="CODING_PLAN_MAX_STEPS")
    coding_search_budget: int = Field(default=4, alias="CODING_SEARCH_BUDGET")
    coding_read_budget: int = Field(default=6, alias="CODING_READ_BUDGET")
    coding_require_read_files: int = Field(default=2, alias="CODING_REQUIRE_READ_FILES")
    tool_exec_backend: str = Field(default="local", alias="TOOL_EXEC_BACKEND")
    redis_url: str = Field(default="redis://127.0.0.1:6379/0", alias="REDIS_URL")
    toolsmanager_parallel_requests: int = Field(
        default=3, alias="TOOLSMANAGER_PARALLEL_REQUESTS"
    )
    redis_queue_name: str = Field(default="mana-tools", alias="REDIS_QUEUE_NAME")
    redis_ttl_seconds: int = Field(default=86_400, alias="REDIS_TTL_SECONDS")
    mana_github_token: str = Field(default="", alias="MANA_GITHUB_TOKEN")
    mana_github_credential_source: str = Field(
        default="disabled", alias="MANA_GITHUB_CREDENTIAL_SOURCE"
    )
    mana_github_secret_ref: str = Field(default="", alias="MANA_GITHUB_SECRET_REF")
    mana_github_metadata_enabled: bool = Field(
        default=False, alias="MANA_GITHUB_METADATA_ENABLED"
    )
    mana_search_enable_web: bool = Field(default=True, alias="MANA_SEARCH_ENABLE_WEB")
    mana_search_enable_github: bool = Field(
        default=True, alias="MANA_SEARCH_ENABLE_GITHUB"
    )
    mana_search_max_results: int = Field(default=8, alias="MANA_SEARCH_MAX_RESULTS")
    mana_search_timeout_seconds: int = Field(
        default=15, alias="MANA_SEARCH_TIMEOUT_SECONDS"
    )
    mana_search_memory_ttl_days: int = Field(
        default=14, alias="MANA_SEARCH_MEMORY_TTL_DAYS"
    )
    mana_memory_mode: str = Field(default="internal", alias="MANA_MEMORY_MODE")
    mana_memory_provider: str = Field(default="mana", alias="MANA_MEMORY_PROVIDER")
    mana_memory_fallback_to_internal: bool = Field(
        default=False, alias="MANA_MEMORY_FALLBACK_TO_INTERNAL"
    )
    mana_memory_secret_ref: str = Field(default="", alias="MANA_MEMORY_SECRET_REF")
    mem0_org_id: str = Field(default="", alias="MEM0_ORG_ID")
    mem0_project_id: str = Field(default="", alias="MEM0_PROJECT_ID")
    mem0_base_url: str = Field(default="", alias="MEM0_BASE_URL")
    supermemory_base_url: str = Field(default="", alias="SUPERMEMORY_BASE_URL")
    mana_memory_timeout_seconds: int = Field(
        default=15, alias="MANA_MEMORY_TIMEOUT_SECONDS"
    )
    mana_memory_capsules_enabled: bool = Field(default=True, alias="MANA_MEMORY_CAPSULES_ENABLED")
    mana_memory_capsules_default_max_capsules: int = Field(default=12, ge=1, le=100, alias="MANA_MEMORY_CAPSULES_DEFAULT_MAX_CAPSULES")
    mana_memory_capsules_default_max_tokens: int = Field(default=4000, ge=1, alias="MANA_MEMORY_CAPSULES_DEFAULT_MAX_TOKENS")
    mana_memory_capsules_shared_writes_require_review: bool = Field(default=True, alias="MANA_MEMORY_CAPSULES_SHARED_WRITES_REQUIRE_REVIEW")
    mana_memory_capsules_organisation_scope_enabled: bool = Field(default=False, alias="MANA_MEMORY_CAPSULES_ORGANISATION_SCOPE_ENABLED")
    mana_memory_capsules_user_scope_enabled: bool = Field(default=True, alias="MANA_MEMORY_CAPSULES_USER_SCOPE_ENABLED")
    mana_memory_capsules_record_access_events: bool = Field(default=True, alias="MANA_MEMORY_CAPSULES_RECORD_ACCESS_EVENTS")
    mana_memory_capsules_quarantine_prompt_injection: bool = Field(default=True, alias="MANA_MEMORY_CAPSULES_QUARANTINE_PROMPT_INJECTION")
    mana_memory_capsules_retention_private_days: int = Field(default=7, ge=1, alias="MANA_MEMORY_CAPSULES_RETENTION_PRIVATE_DAYS")
    mana_memory_capsules_retention_parent_child_days: int = Field(default=30, ge=1, alias="MANA_MEMORY_CAPSULES_RETENTION_PARENT_CHILD_DAYS")
    mana_memory_capsules_retention_team_days: int = Field(default=90, ge=1, alias="MANA_MEMORY_CAPSULES_RETENTION_TEAM_DAYS")
    mana_memory_capsules_retention_project_days: int = Field(default=180, ge=1, alias="MANA_MEMORY_CAPSULES_RETENTION_PROJECT_DAYS")
    mana_memory_capsules_retention_organisation_days: int = Field(default=365, ge=1, alias="MANA_MEMORY_CAPSULES_RETENTION_ORGANISATION_DAYS")
    mana_web_search_provider: str = Field(default="", alias="MANA_WEB_SEARCH_PROVIDER")
    mana_web_search_api_key: str = Field(default="", alias="MANA_WEB_SEARCH_API_KEY")
    mana_web_search_endpoint: str = Field(default="", alias="MANA_WEB_SEARCH_ENDPOINT")
    mana_web_search_base_url: str = Field(default="", alias="MANA_WEB_SEARCH_BASE_URL")
    mana_web_search_engine_id: str = Field(
        default="", alias="MANA_WEB_SEARCH_ENGINE_ID"
    )
    mana_web_search_max_results: int = Field(
        default=8, alias="MANA_WEB_SEARCH_MAX_RESULTS"
    )
    mana_search_max_injected_results: int = Field(
        default=5, alias="MANA_SEARCH_MAX_INJECTED_RESULTS"
    )
    mana_search_max_summary_words: int = Field(
        default=80, alias="MANA_SEARCH_MAX_SUMMARY_WORDS"
    )
    mana_search_enable_ask_agent: bool = Field(
        default=True, alias="MANA_SEARCH_ENABLE_ASK_AGENT"
    )
    mana_workspace_allowed_roots: str = Field(
        default="", alias="MANA_WORKSPACE_ALLOWED_ROOTS"
    )
    mana_api_token: str = Field(default="", alias="MANA_API_TOKEN")
    mana_api_manager_allowed_hosts: str = Field(
        default="", alias="MANA_API_MANAGER_ALLOWED_HOSTS"
    )
    mana_api_manager_trusted_internal_hosts: str = Field(
        default="", alias="MANA_API_MANAGER_TRUSTED_INTERNAL_HOSTS"
    )
    mana_api_manager_trusted_internal_networks: str = Field(
        default="", alias="MANA_API_MANAGER_TRUSTED_INTERNAL_NETWORKS"
    )
    mana_api_manager_allow_http: bool = Field(
        default=False, alias="MANA_API_MANAGER_ALLOW_HTTP"
    )
    mana_api_manager_max_redirects: int = Field(
        default=3, alias="MANA_API_MANAGER_MAX_REDIRECTS"
    )
    mana_api_manager_max_response_bytes: int = Field(
        default=10_485_760, alias="MANA_API_MANAGER_MAX_RESPONSE_BYTES"
    )
    mana_mcp_server_token: str = Field(default="", alias="MANA_MCP_SERVER_TOKEN")
    mana_worker_gateway_enabled: bool = Field(
        default=False, alias="MANA_WORKER_GATEWAY_ENABLED"
    )
    mana_worker_gateway_public_url: str = Field(
        default="", alias="MANA_WORKER_GATEWAY_PUBLIC_URL"
    )
    mana_worker_gateway_allow_insecure_http: bool = Field(
        default=False,
        alias="MANA_WORKER_GATEWAY_ALLOW_INSECURE_HTTP",
    )
    mana_worker_gateway_local_dev: bool = Field(
        default=False, alias="MANA_WORKER_GATEWAY_LOCAL_DEV"
    )
    mana_acp_enabled: bool = Field(default=True, alias="MANA_ACP_ENABLED")
    mana_acp_allowed_roots: str = Field(default="", alias="MANA_ACP_ALLOWED_ROOTS")
    mana_acp_mcp_forwarding: bool = Field(default=True, alias="MANA_ACP_MCP_FORWARDING")
    mana_acp_session_load: bool = Field(default=True, alias="MANA_ACP_SESSION_LOAD")
    mana_acp_session_retention_days: int = Field(
        default=30, alias="MANA_ACP_SESSION_RETENTION_DAYS"
    )
    mana_a2a_server_enabled: bool = Field(
        default=False, alias="MANA_A2A_SERVER_ENABLED"
    )
    mana_a2a_host: str = Field(default="127.0.0.1", alias="MANA_A2A_HOST")
    mana_a2a_port: int = Field(default=8766, alias="MANA_A2A_PORT")
    mana_a2a_public_base_url: str = Field(default="", alias="MANA_A2A_PUBLIC_BASE_URL")
    mana_a2a_server_token: str = Field(default="", alias="MANA_A2A_SERVER_TOKEN")
    mana_a2a_enabled_skills: str = Field(default="", alias="MANA_A2A_ENABLED_SKILLS")
    mana_a2a_streaming: bool = Field(default=True, alias="MANA_A2A_STREAMING")
    mana_a2a_push_notifications: bool = Field(
        default=False, alias="MANA_A2A_PUSH_NOTIFICATIONS"
    )
    mana_a2a_task_retention_days: int = Field(
        default=30, alias="MANA_A2A_TASK_RETENTION_DAYS"
    )
    mana_a2a_max_request_bytes: int = Field(
        default=1_048_576, alias="MANA_A2A_MAX_REQUEST_BYTES"
    )
    mana_a2a_max_artifact_bytes: int = Field(
        default=10_485_760, alias="MANA_A2A_MAX_ARTIFACT_BYTES"
    )
    mana_a2a_max_concurrent_tasks: int = Field(
        default=4, alias="MANA_A2A_MAX_CONCURRENT_TASKS"
    )
    mana_a2a_delegation_enabled: bool = Field(
        default=False, alias="MANA_A2A_DELEGATION_ENABLED"
    )
    mana_a2a_max_delegation_depth: int = Field(
        default=3, alias="MANA_A2A_MAX_DELEGATION_DEPTH"
    )
    mana_canvas_enabled: bool = Field(default=True, alias="MANA_CANVAS_ENABLED")
    mana_canvas_protocol_versions: str = Field(
        default="v0.9", alias="MANA_CANVAS_PROTOCOL_VERSIONS"
    )
    mana_canvas_default_protocol_version: str = Field(
        default="v0.9", alias="MANA_CANVAS_DEFAULT_PROTOCOL_VERSION"
    )
    mana_canvas_allowed_catalogs: str = Field(
        default="https://mana-agent.dev/a2ui/catalogs/core/v1/catalog.json",
        alias="MANA_CANVAS_ALLOWED_CATALOGS",
    )
    mana_canvas_accept_inline_catalogs: bool = Field(
        default=False, alias="MANA_CANVAS_ACCEPT_INLINE_CATALOGS"
    )
    mana_canvas_allow_localhost: bool = Field(
        default=True, alias="MANA_CANVAS_ALLOW_LOCALHOST"
    )
    mana_canvas_max_active_surfaces: int = Field(
        default=16, alias="MANA_CANVAS_MAX_ACTIVE_SURFACES"
    )
    mana_canvas_max_components: int = Field(
        default=250, alias="MANA_CANVAS_MAX_COMPONENTS"
    )
    mana_canvas_max_event_bytes: int = Field(
        default=262_144, alias="MANA_CANVAS_MAX_EVENT_BYTES"
    )
    mana_canvas_max_depth: int = Field(default=24, alias="MANA_CANVAS_MAX_DEPTH")
    mana_canvas_snapshot_interval: int = Field(
        default=20, alias="MANA_CANVAS_SNAPSHOT_INTERVAL"
    )
    mana_canvas_generation_timeout_seconds: int = Field(
        default=30, alias="MANA_CANVAS_GENERATION_TIMEOUT_SECONDS"
    )
    mana_canvas_surface_expiry_seconds: int = Field(
        default=86_400, alias="MANA_CANVAS_SURFACE_EXPIRY_SECONDS"
    )
    mana_canvas_action_timeout_seconds: int = Field(
        default=900, alias="MANA_CANVAS_ACTION_TIMEOUT_SECONDS"
    )
    mana_canvas_validation_retry_limit: int = Field(
        default=1, alias="MANA_CANVAS_VALIDATION_RETRY_LIMIT"
    )
    mana_canvas_max_updates_per_second: int = Field(
        default=20, alias="MANA_CANVAS_MAX_UPDATES_PER_SECOND"
    )
    mana_canvas_websocket_queue_size: int = Field(
        default=256, alias="MANA_CANVAS_WEBSOCKET_QUEUE_SIZE"
    )
    mana_canvas_allowed_image_schemes: str = Field(
        default="https", alias="MANA_CANVAS_ALLOWED_IMAGE_SCHEMES"
    )
    mana_canvas_allowed_artifact_schemes: str = Field(
        default="https,artifact", alias="MANA_CANVAS_ALLOWED_ARTIFACT_SCHEMES"
    )
    mana_canvas_developer_diagnostics: bool = Field(
        default=False, alias="MANA_CANVAS_DEVELOPER_DIAGNOSTICS"
    )
    mana_browser_enabled: bool = Field(default=True, alias="MANA_BROWSER_ENABLED")
    mana_browser_headless: bool = Field(default=True, alias="MANA_BROWSER_HEADLESS")
    mana_browser_timeout_seconds: int = Field(
        default=30, alias="MANA_BROWSER_TIMEOUT_SECONDS"
    )
    mana_browser_persist_auth: bool = Field(
        default=False, alias="MANA_BROWSER_PERSIST_AUTH"
    )
    mana_browser_download_max_mb: int = Field(
        default=100, alias="MANA_BROWSER_DOWNLOAD_MAX_MB"
    )
    mana_browser_upload_roots: str = Field(
        default="", alias="MANA_BROWSER_UPLOAD_ROOTS"
    )
    mana_browser_artifact_dir: str = Field(
        default="", alias="MANA_BROWSER_ARTIFACT_DIR"
    )
    media: dict[str, Any] = Field(default_factory=dict, alias="media")
    mana_browser_profile_max_age_days: int = Field(
        default=30, alias="MANA_BROWSER_PROFILE_MAX_AGE_DAYS"
    )
    mana_computer_control_enabled: bool = Field(
        default=False, alias="MANA_COMPUTER_CONTROL_ENABLED"
    )
    # When enabled, coding/tool multi-agent routes allocate an isolated Git worktree
    # under ~/.mana/repositories/<repository-id>/worktrees/ instead of mutating the
    # primary checkout. Explicit merge intent is still required after review.
    mana_managed_worktrees_enabled: bool = Field(
        default=True, alias="MANA_MANAGED_WORKTREES_ENABLED"
    )
    # Non-interactive / bench mode: auto-allow transactional REQUIRE_APPROVAL
    # outcomes (shell, destructive file ops, remote git). DENY stays deny.
    mana_transactional_always_approve: bool = Field(
        default=False, alias="MANA_TRANSACTIONAL_ALWAYS_APPROVE"
    )
    # Empty preserves pre-0.0.19 configurations: Codex when enabled, internal otherwise.
    mana_coding_backend: str = Field(default="", alias="MANA_CODING_BACKEND")
    mana_codex_enabled: bool = Field(default=True, alias="MANA_CODEX_ENABLED")
    mana_codex_max_workers: int = Field(default=2, alias="MANA_CODEX_MAX_WORKERS")
    mana_codex_stream_events: bool = Field(
        default=True, alias="MANA_CODEX_STREAM_EVENTS"
    )
    mana_codex_worktree_isolation: bool = Field(
        default=False, alias="MANA_CODEX_WORKTREE_ISOLATION"
    )
    mana_codex_task_timeout_seconds: int = Field(
        default=1800, alias="MANA_CODEX_TASK_TIMEOUT_SECONDS"
    )
    mana_codex_allow_network: bool = Field(
        default=False, alias="MANA_CODEX_ALLOW_NETWORK"
    )
    mana_codex_model: str | None = Field(default=None, alias="MANA_CODEX_MODEL")
    mana_codex_bin: str = Field(default="codex", alias="MANA_CODEX_BIN")
    # full = all auto-chat tools; coding = repository/edit/verify/document/git only
    # (SWE-bench and other isolated coding runs).
    mana_auto_chat_tool_surface: str = Field(
        default="full", alias="MANA_AUTO_CHAT_TOOL_SURFACE"
    )
    mana_github_autopilot_enabled: bool = Field(
        default=False, alias="MANA_GITHUB_AUTOPILOT_ENABLED"
    )
    mana_github_app_id: str = Field(default="", alias="MANA_GITHUB_APP_ID")
    mana_github_app_private_key_path: str = Field(
        default="", alias="MANA_GITHUB_APP_PRIVATE_KEY_PATH"
    )
    mana_github_webhook_secret: str = Field(
        default="", alias="MANA_GITHUB_WEBHOOK_SECRET"
    )
    mana_github_public_webhook_url: str = Field(
        default="", alias="MANA_GITHUB_PUBLIC_WEBHOOK_URL"
    )
    mana_github_invocation_name: str = Field(
        default="@mana-agent", alias="MANA_GITHUB_INVOCATION_NAME"
    )
    mana_github_fix_label: str = Field(
        default="mana-fix", alias="MANA_GITHUB_FIX_LABEL"
    )
    mana_github_minimum_actor_permission: str = Field(
        default="write", alias="MANA_GITHUB_MINIMUM_ACTOR_PERMISSION"
    )
    mana_github_allowed_repositories: str = Field(
        default="", alias="MANA_GITHUB_ALLOWED_REPOSITORIES"
    )
    mana_github_allowed_organizations: str = Field(
        default="", alias="MANA_GITHUB_ALLOWED_ORGANIZATIONS"
    )
    mana_github_allowed_workflows: str = Field(
        default="", alias="MANA_GITHUB_ALLOWED_WORKFLOWS"
    )
    mana_github_allowed_branches: str = Field(
        default="", alias="MANA_GITHUB_ALLOWED_BRANCHES"
    )
    mana_github_actor_allowlist: str = Field(
        default="", alias="MANA_GITHUB_ACTOR_ALLOWLIST"
    )
    mana_github_security_events_enabled: bool = Field(
        default=False, alias="MANA_GITHUB_SECURITY_EVENTS_ENABLED"
    )
    mana_github_allow_bots: bool = Field(default=False, alias="MANA_GITHUB_ALLOW_BOTS")
    mana_github_worker_concurrency: int = Field(
        default=2, alias="MANA_GITHUB_WORKER_CONCURRENCY"
    )
    mana_github_maximum_job_iterations: int = Field(
        default=8, alias="MANA_GITHUB_MAXIMUM_JOB_ITERATIONS"
    )
    mana_github_maximum_job_runtime: int = Field(
        default=1800, alias="MANA_GITHUB_MAXIMUM_JOB_RUNTIME"
    )
    mana_github_maximum_changed_files: int = Field(
        default=50, alias="MANA_GITHUB_MAXIMUM_CHANGED_FILES"
    )
    mana_github_draft_pr_only: bool = Field(
        default=True, alias="MANA_GITHUB_DRAFT_PR_ONLY"
    )
    mana_github_workflow_files_write_enabled: bool = Field(
        default=False, alias="MANA_GITHUB_WORKFLOW_FILES_WRITE_ENABLED"
    )
    mana_lane_contracts: dict[str, Any] | str = Field(
        default_factory=dict, alias="MANA_LANE_CONTRACTS"
    )
    mana_lane_global_worker_limit: int = Field(
        default=8, alias="MANA_LANE_GLOBAL_WORKER_LIMIT"
    )
    mana_lane_provider_limits: dict[str, int] | str = Field(
        default_factory=dict, alias="MANA_LANE_PROVIDER_LIMITS"
    )
    mana_lane_session_token_budget: int | None = Field(
        default=None, alias="MANA_LANE_SESSION_TOKEN_BUDGET"
    )
    mana_lane_global_token_budget: int | None = Field(
        default=None, alias="MANA_LANE_GLOBAL_TOKEN_BUDGET"
    )
    # Provider-neutral task execution. Provider details are structured JSON in
    # user config and contain references to secrets, never secret values.
    mana_execution_default_provider: str = Field(
        default="local-process", alias="MANA_EXECUTION_DEFAULT_PROVIDER"
    )
    mana_execution_allowed_providers: list[str] | str = Field(
        default_factory=lambda: [
            "local-process",
            "local-docker",
            "remote-ssh",
            "kubernetes",
            "modal",
            "custom-http-runtime",
        ],
        alias="MANA_EXECUTION_ALLOWED_PROVIDERS",
    )
    mana_execution_cleanup_on_exit: bool = Field(
        default=True, alias="MANA_EXECUTION_CLEANUP_ON_EXIT"
    )
    mana_execution_idle_timeout_seconds: int = Field(
        default=900, alias="MANA_EXECUTION_IDLE_TIMEOUT_SECONDS"
    )
    mana_execution_max_lifetime_seconds: int = Field(
        default=7200, alias="MANA_EXECUTION_MAX_LIFETIME_SECONDS"
    )
    mana_execution_global_concurrency: int = Field(
        default=16, alias="MANA_EXECUTION_GLOBAL_CONCURRENCY"
    )
    mana_execution_routing: dict[str, Any] | str = Field(
        default_factory=dict, alias="MANA_EXECUTION_ROUTING"
    )
    mana_execution_providers: dict[str, Any] | str = Field(
        default_factory=dict, alias="MANA_EXECUTION_PROVIDERS"
    )
    mana_execution_supervisor_enabled: bool = Field(
        default=True, alias="MANA_EXECUTION_SUPERVISOR_ENABLED"
    )
    mana_execution_supervisor_lease_seconds: int = Field(
        default=60, alias="MANA_EXECUTION_SUPERVISOR_LEASE_SECONDS"
    )
    mana_execution_supervisor_heartbeat_seconds: int = Field(
        default=15, alias="MANA_EXECUTION_SUPERVISOR_HEARTBEAT_SECONDS"
    )
    mana_execution_supervisor_checkpoint_seconds: int = Field(
        default=60, alias="MANA_EXECUTION_SUPERVISOR_CHECKPOINT_SECONDS"
    )
    mana_execution_supervisor_retry_budget: int = Field(
        default=3, alias="MANA_EXECUTION_SUPERVISOR_RETRY_BUDGET"
    )
    mana_execution_supervisor_max_replans: int = Field(
        default=2, alias="MANA_EXECUTION_SUPERVISOR_MAX_REPLANS"
    )
    mana_execution_supervisor_max_child_depth: int = Field(
        default=5, alias="MANA_EXECUTION_SUPERVISOR_MAX_CHILD_DEPTH"
    )
    mana_execution_supervisor_max_children: int = Field(
        default=20, alias="MANA_EXECUTION_SUPERVISOR_MAX_CHILDREN"
    )
    mana_execution_supervisor_max_total_subtasks: int = Field(
        default=100, alias="MANA_EXECUTION_SUPERVISOR_MAX_TOTAL_SUBTASKS"
    )
    mana_execution_supervisor_max_concurrent_children: int = Field(
        default=4, alias="MANA_EXECUTION_SUPERVISOR_MAX_CONCURRENT_CHILDREN"
    )
    mana_execution_supervisor_startup_recovery: bool = Field(
        default=True, alias="MANA_EXECUTION_SUPERVISOR_STARTUP_RECOVERY"
    )
    mana_execution_supervisor_verify_artifacts: bool = Field(
        default=True, alias="MANA_EXECUTION_SUPERVISOR_VERIFY_ARTIFACTS"
    )
    mana_execution_supervisor_allow_unknown_retry: bool = Field(
        default=False, alias="MANA_EXECUTION_SUPERVISOR_ALLOW_UNKNOWN_RETRY"
    )
    # Distributed verification is opt-in and fail-closed when no compatible
    # authenticated worker satisfies the model-produced selection request.
    mana_fleet_enabled: bool = Field(default=False, alias="MANA_FLEET_ENABLED")
    mana_fleet_max_workers_per_run: int = Field(
        default=4, alias="MANA_FLEET_MAX_WORKERS_PER_RUN"
    )
    mana_fleet_max_concurrent_jobs: int = Field(
        default=4, alias="MANA_FLEET_MAX_CONCURRENT_JOBS"
    )
    mana_fleet_capability_ttl_seconds: int = Field(
        default=300, alias="MANA_FLEET_CAPABILITY_TTL_SECONDS"
    )
    mana_fleet_heartbeat_timeout_seconds: int = Field(
        default=90, alias="MANA_FLEET_HEARTBEAT_TIMEOUT_SECONDS"
    )
    mana_fleet_job_timeout_seconds: int = Field(
        default=1800, alias="MANA_FLEET_JOB_TIMEOUT_SECONDS"
    )
    mana_fleet_workspace_max_lifetime_seconds: int = Field(
        default=3600, alias="MANA_FLEET_WORKSPACE_MAX_LIFETIME_SECONDS"
    )
    mana_fleet_max_log_bytes: int = Field(
        default=1_048_576, alias="MANA_FLEET_MAX_LOG_BYTES"
    )
    mana_fleet_max_artifact_bytes: int = Field(
        default=104_857_600, alias="MANA_FLEET_MAX_ARTIFACT_BYTES"
    )
    mana_fleet_retain_days: int = Field(default=30, alias="MANA_FLEET_RETAIN_DAYS")
    mana_fleet_auto_repair_enabled: bool = Field(
        default=False, alias="MANA_FLEET_AUTO_REPAIR_ENABLED"
    )
    mana_fleet_require_trusted_label: bool = Field(
        default=True, alias="MANA_FLEET_REQUIRE_TRUSTED_LABEL"
    )
    # Remote SSH execution has its own policy and never modifies global OpenSSH
    # configuration. Values are structured JSON in Mana user configuration.
    mana_remote_execution: dict[str, Any] | str = Field(
        default_factory=dict, alias="MANA_REMOTE_EXECUTION"
    )

    # Mana-managed settings are intentionally repository-independent.  Loading
    # a project's ``.env`` here can silently replace the API key selected in
    # the setup wizard with an unrelated development credential.
    model_config = SettingsConfigDict(extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        def user_config_settings() -> dict[str, object]:
            return settings_source_for_pydantic()

        return (
            init_settings,
            user_config_settings,
            file_secret_settings,
        )

    @model_validator(mode="after")
    def validate_semantics(self) -> Settings:
        if self.mana_memory_mode == "external" and not self.mana_memory_provider:
            raise ValueError("External memory mode requires a memory provider to be configured.")
        return self

    def model_post_init(self, __context: object) -> None:
        _ = __context
        ratios = (
            self.mana_context_warning_ratio,
            self.mana_context_compact_ratio,
            self.mana_context_max_utilization,
            self.mana_context_hard_limit_ratio,
        )
        if tuple(sorted(ratios)) != ratios or len(set(ratios)) != len(ratios):
            raise ValueError(
                "context ratios must increase in warning, compact, maximum-utilization, hard-limit order"
            )
        if self.mana_context_unknown_model_max_output_tokens > self.mana_context_unknown_model_context_window:
            raise ValueError("unknown-model max output cannot exceed its configured context window")
        from mana_agent.canvas.config import CanvasConfig

        CanvasConfig.from_settings(self)


def default_index_dir(target_path: str | Path) -> Path:
    # Compatibility helper for callers that have not resolved a repository id.
    # The workspace registry replaces this with repository_index_dir(repo_id).
    from mana_agent.workspaces.paths import repository_id_for_path, repository_index_dir

    return repository_index_dir(repository_id_for_path(target_path))


def mana_root_dir(target_path: str | Path) -> Path:
    # Kept as a public compatibility name: generated state is now user-level.
    _ = target_path
    return mana_home()


def default_logs_dir(target_path: str | Path) -> Path:
    _ = target_path
    return mana_home() / "logs"


def default_tools_logs_dir(target_path: str | Path) -> Path:
    _ = target_path
    return mana_home() / "tools_logs"


def default_llm_logs_dir(target_path: str | Path) -> Path:
    _ = target_path
    return mana_home() / "llm_logs"


def default_diagrams_dir(target_path: str | Path) -> Path:
    _ = target_path
    return mana_home() / "diagrams"
