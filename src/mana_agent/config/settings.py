from __future__ import annotations

import os
from pathlib import Path

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
    mana_search_enable_web: bool = Field(default=True, alias="MANA_SEARCH_ENABLE_WEB")
    mana_search_enable_github: bool = Field(default=True, alias="MANA_SEARCH_ENABLE_GITHUB")
    mana_search_max_results: int = Field(default=8, alias="MANA_SEARCH_MAX_RESULTS")
    mana_search_timeout_seconds: int = Field(default=15, alias="MANA_SEARCH_TIMEOUT_SECONDS")
    mana_search_memory_ttl_days: int = Field(default=14, alias="MANA_SEARCH_MEMORY_TTL_DAYS")
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
    mana_browser_enabled: bool = Field(default=True, alias="MANA_BROWSER_ENABLED")
    mana_browser_headless: bool = Field(default=True, alias="MANA_BROWSER_HEADLESS")
    mana_browser_timeout_seconds: int = Field(default=30, alias="MANA_BROWSER_TIMEOUT_SECONDS")
    mana_browser_persist_auth: bool = Field(default=False, alias="MANA_BROWSER_PERSIST_AUTH")
    mana_browser_download_max_mb: int = Field(default=100, alias="MANA_BROWSER_DOWNLOAD_MAX_MB")
    mana_browser_upload_roots: str = Field(default="", alias="MANA_BROWSER_UPLOAD_ROOTS")
    mana_browser_artifact_dir: str = Field(default="", alias="MANA_BROWSER_ARTIFACT_DIR")
    mana_browser_profile_max_age_days: int = Field(default=30, alias="MANA_BROWSER_PROFILE_MAX_AGE_DAYS")

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
