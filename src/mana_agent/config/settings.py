from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mana_agent.config.user_config import settings_source_for_pydantic


MANA_ROOT_DIRNAME = ".mana"

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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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
            env_settings,
            dotenv_settings,
            user_config_settings,
            file_secret_settings,
        )

    def model_post_init(self, __context: object) -> None:
        if os.getenv("LLM_MODEL") and not os.getenv("OPENAI_CHAT_MODEL"):
            self.openai_chat_model = str(self.llm_model or self.openai_chat_model)


def default_index_dir(target_path: str | Path) -> Path:
    return mana_root_dir(target_path) / "index"


def mana_root_dir(target_path: str | Path) -> Path:
    return Path(target_path).resolve() / MANA_ROOT_DIRNAME


def default_logs_dir(target_path: str | Path) -> Path:
    return mana_root_dir(target_path) / "logs"


def default_tools_logs_dir(target_path: str | Path) -> Path:
    return mana_root_dir(target_path) / "tools_logs"


def default_llm_logs_dir(target_path: str | Path) -> Path:
    return mana_root_dir(target_path) / "llm_logs"


def default_diagrams_dir(target_path: str | Path) -> Path:
    return mana_root_dir(target_path) / "diagrams"
