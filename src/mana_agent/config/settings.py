from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mana_agent.config.user_config import settings_source_for_pydantic


MANA_ROOT_DIRNAME = ".mana"


def mana_home() -> Path:
    """Return Mana-Agent's user-level state directory.

    Repository source trees are deliberately not used as state stores.  Tests
    and managed installations can isolate state with ``MANA_HOME``.
    """

    configured = str(os.getenv("MANA_HOME") or "").strip()
    return Path(configured).expanduser().resolve() if configured else (Path.home() / MANA_ROOT_DIRNAME).resolve()

# Default embedding models per provider. The chat and embedding endpoints share a
# single base URL, so when no embedding model is configured explicitly we pick a
# provider-appropriate default based on that URL (an OpenAI embedding model does
# not exist on NVIDIA's API and vice versa).
OPENAI_DEFAULT_EMBED_MODEL = "text-embedding-3-small"
NVIDIA_DEFAULT_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"


def resolve_embed_model(base_url: str | None, explicit_model: str | None = None) -> str:
    """Return the embedding model to use for the given base URL.

    An explicitly configured model always wins. Otherwise the model is inferred
    from the base URL: NVIDIA endpoints get an NVIDIA embedding model, everything
    else falls back to the OpenAI default.
    """
    if explicit_model and explicit_model.strip():
        return explicit_model.strip()
    if base_url and "nvidia" in base_url.lower():
        return NVIDIA_DEFAULT_EMBED_MODEL
    return OPENAI_DEFAULT_EMBED_MODEL


class Settings(BaseSettings):
    mana_ai_provider: str = Field(default="openai", alias="MANA_AI_PROVIDER")
    mana_primary_model: str = Field(default="openai/gpt-4.1-mini", alias="MANA_PRIMARY_MODEL")
    mana_embedding_model: str = Field(default="openai/text-embedding-3-small", alias="MANA_EMBEDDING_MODEL")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_chat_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_CHAT_MODEL")
    openai_tool_worker_model: str | None = Field(default=None, alias="OPENAI_TOOL_WORKER_MODEL")
    openai_coding_planner_model: str | None = Field(default=None, alias="OPENAI_CODING_PLANNER_MODEL")
    # Left unset by default so the embedding model can be auto-selected from the
    # active base URL (see ``resolve_embed_model``). An explicit value always wins.
    openai_embed_model: str | None = Field(default=None, alias="OPENAI_EMBED_MODEL")
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
    toolsmanager_parallel_requests: int = Field(default=3, alias="TOOLSMANAGER_PARALLEL_REQUESTS")
    redis_queue_name: str = Field(default="mana-tools", alias="REDIS_QUEUE_NAME")
    redis_ttl_seconds: int = Field(default=86_400, alias="REDIS_TTL_SECONDS")
    mana_github_token: str = Field(default="", alias="MANA_GITHUB_TOKEN")
    mana_github_credential_source: str = Field(default="disabled", alias="MANA_GITHUB_CREDENTIAL_SOURCE")
    mana_github_secret_ref: str = Field(default="", alias="MANA_GITHUB_SECRET_REF")
    mana_github_metadata_enabled: bool = Field(default=False, alias="MANA_GITHUB_METADATA_ENABLED")
    mana_search_enable_web: bool = Field(default=True, alias="MANA_SEARCH_ENABLE_WEB")
    mana_search_enable_github: bool = Field(default=True, alias="MANA_SEARCH_ENABLE_GITHUB")
    mana_search_max_results: int = Field(default=8, alias="MANA_SEARCH_MAX_RESULTS")
    mana_search_timeout_seconds: int = Field(default=15, alias="MANA_SEARCH_TIMEOUT_SECONDS")
    mana_search_memory_ttl_days: int = Field(default=14, alias="MANA_SEARCH_MEMORY_TTL_DAYS")
    mana_memory_mode: str = Field(default="internal", alias="MANA_MEMORY_MODE")
    mana_memory_provider: str = Field(default="mana", alias="MANA_MEMORY_PROVIDER")
    mana_memory_fallback_to_internal: bool = Field(default=False, alias="MANA_MEMORY_FALLBACK_TO_INTERNAL")
    mana_memory_secret_ref: str = Field(default="", alias="MANA_MEMORY_SECRET_REF")
    mem0_org_id: str = Field(default="", alias="MEM0_ORG_ID")
    mem0_project_id: str = Field(default="", alias="MEM0_PROJECT_ID")
    mem0_base_url: str = Field(default="", alias="MEM0_BASE_URL")
    mana_memory_timeout_seconds: int = Field(default=15, alias="MANA_MEMORY_TIMEOUT_SECONDS")
    mana_web_search_provider: str = Field(default="", alias="MANA_WEB_SEARCH_PROVIDER")
    mana_web_search_api_key: str = Field(default="", alias="MANA_WEB_SEARCH_API_KEY")
    mana_web_search_endpoint: str = Field(default="", alias="MANA_WEB_SEARCH_ENDPOINT")
    mana_web_search_base_url: str = Field(default="", alias="MANA_WEB_SEARCH_BASE_URL")
    mana_web_search_engine_id: str = Field(default="", alias="MANA_WEB_SEARCH_ENGINE_ID")
    mana_web_search_max_results: int = Field(default=8, alias="MANA_WEB_SEARCH_MAX_RESULTS")
    mana_search_max_injected_results: int = Field(default=5, alias="MANA_SEARCH_MAX_INJECTED_RESULTS")
    mana_search_max_summary_words: int = Field(default=80, alias="MANA_SEARCH_MAX_SUMMARY_WORDS")
    mana_search_enable_ask_agent: bool = Field(default=True, alias="MANA_SEARCH_ENABLE_ASK_AGENT")
    mana_workspace_allowed_roots: str = Field(default="", alias="MANA_WORKSPACE_ALLOWED_ROOTS")
    mana_api_token: str = Field(default="", alias="MANA_API_TOKEN")
    mana_mcp_server_token: str = Field(default="", alias="MANA_MCP_SERVER_TOKEN")
    mana_acp_enabled: bool = Field(default=True, alias="MANA_ACP_ENABLED")
    mana_acp_allowed_roots: str = Field(default="", alias="MANA_ACP_ALLOWED_ROOTS")
    mana_acp_mcp_forwarding: bool = Field(default=True, alias="MANA_ACP_MCP_FORWARDING")
    mana_acp_session_load: bool = Field(default=True, alias="MANA_ACP_SESSION_LOAD")
    mana_acp_session_retention_days: int = Field(default=30, alias="MANA_ACP_SESSION_RETENTION_DAYS")
    mana_a2a_server_enabled: bool = Field(default=False, alias="MANA_A2A_SERVER_ENABLED")
    mana_a2a_host: str = Field(default="127.0.0.1", alias="MANA_A2A_HOST")
    mana_a2a_port: int = Field(default=8766, alias="MANA_A2A_PORT")
    mana_a2a_public_base_url: str = Field(default="", alias="MANA_A2A_PUBLIC_BASE_URL")
    mana_a2a_server_token: str = Field(default="", alias="MANA_A2A_SERVER_TOKEN")
    mana_a2a_enabled_skills: str = Field(default="", alias="MANA_A2A_ENABLED_SKILLS")
    mana_a2a_streaming: bool = Field(default=True, alias="MANA_A2A_STREAMING")
    mana_a2a_push_notifications: bool = Field(default=False, alias="MANA_A2A_PUSH_NOTIFICATIONS")
    mana_a2a_task_retention_days: int = Field(default=30, alias="MANA_A2A_TASK_RETENTION_DAYS")
    mana_a2a_max_request_bytes: int = Field(default=1_048_576, alias="MANA_A2A_MAX_REQUEST_BYTES")
    mana_a2a_max_artifact_bytes: int = Field(default=10_485_760, alias="MANA_A2A_MAX_ARTIFACT_BYTES")
    mana_a2a_max_concurrent_tasks: int = Field(default=4, alias="MANA_A2A_MAX_CONCURRENT_TASKS")
    mana_a2a_delegation_enabled: bool = Field(default=False, alias="MANA_A2A_DELEGATION_ENABLED")
    mana_a2a_max_delegation_depth: int = Field(default=3, alias="MANA_A2A_MAX_DELEGATION_DEPTH")
    mana_browser_enabled: bool = Field(default=True, alias="MANA_BROWSER_ENABLED")
    mana_browser_headless: bool = Field(default=True, alias="MANA_BROWSER_HEADLESS")
    mana_browser_timeout_seconds: int = Field(default=30, alias="MANA_BROWSER_TIMEOUT_SECONDS")
    mana_browser_persist_auth: bool = Field(default=False, alias="MANA_BROWSER_PERSIST_AUTH")
    mana_browser_download_max_mb: int = Field(default=100, alias="MANA_BROWSER_DOWNLOAD_MAX_MB")
    mana_browser_upload_roots: str = Field(default="", alias="MANA_BROWSER_UPLOAD_ROOTS")
    mana_browser_artifact_dir: str = Field(default="", alias="MANA_BROWSER_ARTIFACT_DIR")
    mana_browser_profile_max_age_days: int = Field(default=30, alias="MANA_BROWSER_PROFILE_MAX_AGE_DAYS")
    # When enabled, coding/tool multi-agent routes allocate an isolated Git worktree
    # under ~/.mana/repositories/<repository-id>/worktrees/ instead of mutating the
    # primary checkout. Explicit merge intent is still required after review.
    mana_managed_worktrees_enabled: bool = Field(default=True, alias="MANA_MANAGED_WORKTREES_ENABLED")
    mana_codex_enabled: bool = Field(default=True, alias="MANA_CODEX_ENABLED")
    mana_codex_max_workers: int = Field(default=2, alias="MANA_CODEX_MAX_WORKERS")
    mana_codex_stream_events: bool = Field(default=True, alias="MANA_CODEX_STREAM_EVENTS")
    mana_codex_worktree_isolation: bool = Field(default=True, alias="MANA_CODEX_WORKTREE_ISOLATION")
    mana_codex_task_timeout_seconds: int = Field(default=1800, alias="MANA_CODEX_TASK_TIMEOUT_SECONDS")
    mana_codex_allow_network: bool = Field(default=False, alias="MANA_CODEX_ALLOW_NETWORK")
    mana_codex_model: str | None = Field(default=None, alias="MANA_CODEX_MODEL")
    mana_codex_bin: str = Field(default="codex", alias="MANA_CODEX_BIN")
    mana_github_autopilot_enabled: bool = Field(default=False, alias="MANA_GITHUB_AUTOPILOT_ENABLED")
    mana_github_app_id: str = Field(default="", alias="MANA_GITHUB_APP_ID")
    mana_github_app_private_key_path: str = Field(default="", alias="MANA_GITHUB_APP_PRIVATE_KEY_PATH")
    mana_github_webhook_secret: str = Field(default="", alias="MANA_GITHUB_WEBHOOK_SECRET")
    mana_github_public_webhook_url: str = Field(default="", alias="MANA_GITHUB_PUBLIC_WEBHOOK_URL")
    mana_github_invocation_name: str = Field(default="@mana-agent", alias="MANA_GITHUB_INVOCATION_NAME")
    mana_github_fix_label: str = Field(default="mana-fix", alias="MANA_GITHUB_FIX_LABEL")
    mana_github_minimum_actor_permission: str = Field(default="write", alias="MANA_GITHUB_MINIMUM_ACTOR_PERMISSION")
    mana_github_allowed_repositories: str = Field(default="", alias="MANA_GITHUB_ALLOWED_REPOSITORIES")
    mana_github_allowed_organizations: str = Field(default="", alias="MANA_GITHUB_ALLOWED_ORGANIZATIONS")
    mana_github_allowed_workflows: str = Field(default="", alias="MANA_GITHUB_ALLOWED_WORKFLOWS")
    mana_github_allowed_branches: str = Field(default="", alias="MANA_GITHUB_ALLOWED_BRANCHES")
    mana_github_actor_allowlist: str = Field(default="", alias="MANA_GITHUB_ACTOR_ALLOWLIST")
    mana_github_security_events_enabled: bool = Field(default=False, alias="MANA_GITHUB_SECURITY_EVENTS_ENABLED")
    mana_github_allow_bots: bool = Field(default=False, alias="MANA_GITHUB_ALLOW_BOTS")
    mana_github_worker_concurrency: int = Field(default=2, alias="MANA_GITHUB_WORKER_CONCURRENCY")
    mana_github_maximum_job_iterations: int = Field(default=8, alias="MANA_GITHUB_MAXIMUM_JOB_ITERATIONS")
    mana_github_maximum_job_runtime: int = Field(default=1800, alias="MANA_GITHUB_MAXIMUM_JOB_RUNTIME")
    mana_github_maximum_changed_files: int = Field(default=50, alias="MANA_GITHUB_MAXIMUM_CHANGED_FILES")
    mana_github_draft_pr_only: bool = Field(default=True, alias="MANA_GITHUB_DRAFT_PR_ONLY")
    mana_github_workflow_files_write_enabled: bool = Field(default=False, alias="MANA_GITHUB_WORKFLOW_FILES_WRITE_ENABLED")
    mana_lane_contracts: dict[str, Any] | str = Field(default_factory=dict, alias="MANA_LANE_CONTRACTS")
    mana_lane_global_worker_limit: int = Field(default=8, alias="MANA_LANE_GLOBAL_WORKER_LIMIT")
    mana_lane_provider_limits: dict[str, int] | str = Field(default_factory=dict, alias="MANA_LANE_PROVIDER_LIMITS")
    mana_lane_session_token_budget: int | None = Field(default=None, alias="MANA_LANE_SESSION_TOKEN_BUDGET")
    mana_lane_global_token_budget: int | None = Field(default=None, alias="MANA_LANE_GLOBAL_TOKEN_BUDGET")
    # Provider-neutral task execution. Provider details are structured JSON in
    # user config and contain references to secrets, never secret values.
    mana_execution_default_provider: str = Field(default="local-process", alias="MANA_EXECUTION_DEFAULT_PROVIDER")
    mana_execution_allowed_providers: list[str] | str = Field(
        default_factory=lambda: [
            "local-process", "local-docker", "remote-ssh", "kubernetes", "modal", "custom-http-runtime",
        ],
        alias="MANA_EXECUTION_ALLOWED_PROVIDERS",
    )
    mana_execution_cleanup_on_exit: bool = Field(default=True, alias="MANA_EXECUTION_CLEANUP_ON_EXIT")
    mana_execution_idle_timeout_seconds: int = Field(default=900, alias="MANA_EXECUTION_IDLE_TIMEOUT_SECONDS")
    mana_execution_max_lifetime_seconds: int = Field(default=7200, alias="MANA_EXECUTION_MAX_LIFETIME_SECONDS")
    mana_execution_global_concurrency: int = Field(default=16, alias="MANA_EXECUTION_GLOBAL_CONCURRENCY")
    mana_execution_routing: dict[str, Any] | str = Field(default_factory=dict, alias="MANA_EXECUTION_ROUTING")
    mana_execution_providers: dict[str, Any] | str = Field(default_factory=dict, alias="MANA_EXECUTION_PROVIDERS")

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

    def model_post_init(self, __context: object) -> None:
        _ = __context


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
