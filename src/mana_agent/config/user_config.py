from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from mana_agent.workspaces.paths import mana_home

CONFIG_DIR = mana_home()
CONFIG_FILE = CONFIG_DIR / "config.toml"
SECRETS_FILE = CONFIG_DIR / "secrets.toml"
MODEL_CACHE_FILE = CONFIG_DIR / "model_cache.json"

SECRET_KEYS = {
    "OPENAI_API_KEY",
    "MANA_GITHUB_TOKEN",
    "MANA_WEB_SEARCH_API_KEY",
    "MANA_API_TOKEN",
    "MANA_MCP_SERVER_TOKEN",
    "MANA_A2A_SERVER_TOKEN",
    "MANA_GITHUB_WEBHOOK_SECRET",
}
NON_PERSISTED_SECRET_KEYS = {"MEM0_API_KEY"}


DEFAULT_USER_CONFIG: dict[str, Any] = {
    "MANA_CONFIG_SCHEMA_VERSION": 2,
    "MANA_AI_PROVIDER": "openai",
    "MANA_PRIMARY_MODEL": "openai/gpt-4.1-mini",
    "MANA_EMBEDDING_MODEL": "openai/text-embedding-3-small",
    "MANA_CONFIGURED_PROVIDERS": ["openai"],
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "OPENAI_CHAT_MODEL": "gpt-4.1-mini",
    "OPENAI_TOOL_WORKER_MODEL": "",
    "OPENAI_CODING_PLANNER_MODEL": "",
    "OPENAI_EMBED_MODEL": "",
    "MODEL_LEVEL_1_FAST_TOOL": "",
    "MODEL_LEVEL_2_CODING": "",
    "MODEL_LEVEL_3_HIGH_REASONING": "",
    "DEFAULT_TOP_K": 8,
    "MANA_LLM_LOG_FILE": "",
    "LLM_MODEL": "",
    "MANA_LLM_API_MODE": "auto",
    "MANA_LLM_REASONING_EFFORT": "",
    "MANA_LLM_SUPPORTS_RESPONSES_API": "",
    "MANA_LLM_SUPPORTS_CHAT_COMPLETIONS": "",
    "MANA_LLM_SUPPORTS_TOOLS": "",
    "MANA_LLM_SUPPORTS_REASONING": "",
    "MANA_LLM_SUPPORTS_TOOLS_WITH_CHAT_REASONING": "",
    "MANA_MODEL_MAIN": "MODEL_LEVEL_3_HIGH_REASONING",
    "MANA_MODEL_HEAD_DECISION": "MODEL_LEVEL_3_HIGH_REASONING",
    "MANA_MODEL_PLANNER": "MODEL_LEVEL_3_HIGH_REASONING",
    "MANA_MODEL_CODING": "MODEL_LEVEL_2_CODING",
    "MANA_MODEL_VERIFIER": "MODEL_LEVEL_2_CODING",
    "MANA_MODEL_REVIEWER": "MODEL_LEVEL_3_HIGH_REASONING",
    "MANA_MODEL_TOOL": "MODEL_LEVEL_1_FAST_TOOL",
    "MANA_MODEL_TOOL_WORKER": "MODEL_LEVEL_1_FAST_TOOL",
    "MANA_MODEL_SUMMARIZER": "MODEL_LEVEL_1_FAST_TOOL",
    "MANA_GITHUB_TOKEN": "",
    "MANA_GITHUB_CREDENTIAL_SOURCE": "disabled",
    "MANA_GITHUB_SECRET_REF": "",
    "MANA_GITHUB_METADATA_ENABLED": False,
    "MANA_SEARCH_ENABLE_WEB": True,
    "MANA_SEARCH_ENABLE_GITHUB": True,
    "MANA_SEARCH_MAX_RESULTS": 8,
    "MANA_SEARCH_TIMEOUT_SECONDS": 15,
    "MANA_SEARCH_MEMORY_TTL_DAYS": 14,
    "MANA_MEMORY_MODE": "internal",
    "MANA_MEMORY_PROVIDER": "mana",
    "MANA_MEMORY_FALLBACK_TO_INTERNAL": False,
    "MANA_MEMORY_SECRET_REF": "",
    "MEM0_ORG_ID": "",
    "MEM0_PROJECT_ID": "",
    "MEM0_BASE_URL": "",
    "MANA_MEMORY_TIMEOUT_SECONDS": 15,
    "MANA_WEB_SEARCH_PROVIDER": "",
    "MANA_WEB_SEARCH_API_KEY": "",
    "MANA_WEB_SEARCH_MAX_RESULTS": 8,
    "MANA_WEB_SEARCH_ENGINE_ID": "",
    "MANA_WEB_SEARCH_BASE_URL": "",
    "MANA_WEB_SEARCH_ENDPOINT": "",
    "MANA_WEB_SEARCH_QUERY_PARAM": "q",
    "MANA_SEARCH_MAX_INJECTED_RESULTS": 5,
    "MANA_SEARCH_MAX_SUMMARY_WORDS": 80,
    "MANA_SEARCH_ENABLE_ASK_AGENT": True,
    "MANA_WORKSPACE_ALLOWED_ROOTS": "",
    "MANA_API_TOKEN": "",
    "MANA_MCP_SERVER_TOKEN": "",
    "MANA_ACP_ENABLED": True,
    "MANA_ACP_ALLOWED_ROOTS": "",
    "MANA_ACP_MCP_FORWARDING": True,
    "MANA_ACP_SESSION_LOAD": True,
    "MANA_ACP_SESSION_RETENTION_DAYS": 30,
    "MANA_A2A_SERVER_ENABLED": False,
    "MANA_A2A_HOST": "127.0.0.1",
    "MANA_A2A_PORT": 8766,
    "MANA_A2A_PUBLIC_BASE_URL": "",
    "MANA_A2A_SERVER_TOKEN": "",
    "MANA_A2A_ENABLED_SKILLS": "",
    "MANA_A2A_STREAMING": True,
    "MANA_A2A_PUSH_NOTIFICATIONS": False,
    "MANA_A2A_TASK_RETENTION_DAYS": 30,
    "MANA_A2A_MAX_REQUEST_BYTES": 1048576,
    "MANA_A2A_MAX_ARTIFACT_BYTES": 10485760,
    "MANA_A2A_MAX_CONCURRENT_TASKS": 4,
    "MANA_A2A_DELEGATION_ENABLED": False,
    "MANA_A2A_MAX_DELEGATION_DEPTH": 3,
    "MANA_BROWSER_ENABLED": True,
    "MANA_BROWSER_HEADLESS": True,
    "MANA_BROWSER_TIMEOUT_SECONDS": 30,
    "MANA_BROWSER_PERSIST_AUTH": False,
    "MANA_BROWSER_DOWNLOAD_MAX_MB": 100,
    "MANA_BROWSER_UPLOAD_ROOTS": "",
    "MANA_BROWSER_ARTIFACT_DIR": "",
    "MANA_BROWSER_PROFILE_MAX_AGE_DAYS": 30,
    "MANA_CODEX_ENABLED": True,
    "MANA_CODEX_MAX_WORKERS": 2,
    "MANA_CODEX_STREAM_EVENTS": True,
    "MANA_CODEX_WORKTREE_ISOLATION": True,
    "MANA_CODEX_TASK_TIMEOUT_SECONDS": 1800,
    "MANA_CODEX_ALLOW_NETWORK": False,
    "MANA_CODEX_MODEL": "",
    "MANA_CODEX_BIN": "codex",
    "MANA_LANE_CONTRACTS": {},
    "MANA_LANE_GLOBAL_WORKER_LIMIT": 8,
    "MANA_LANE_PROVIDER_LIMITS": {},
    "MANA_LANE_SESSION_TOKEN_BUDGET": 0,
    "MANA_LANE_GLOBAL_TOKEN_BUDGET": 0,
    "MANA_EXECUTION_DEFAULT_PROVIDER": "local-process",
    "MANA_EXECUTION_ALLOWED_PROVIDERS": ["local-process", "local-docker", "remote-ssh", "kubernetes", "modal", "custom-http-runtime"],
    "MANA_EXECUTION_CLEANUP_ON_EXIT": True,
    "MANA_EXECUTION_IDLE_TIMEOUT_SECONDS": 900,
    "MANA_EXECUTION_MAX_LIFETIME_SECONDS": 7200,
    "MANA_EXECUTION_GLOBAL_CONCURRENCY": 16,
    "MANA_EXECUTION_ROUTING": {},
    "MANA_EXECUTION_PROVIDERS": {},
    "experience_to_skill": {
        "enabled": True,
        "auto_propose": True,
        "minimum_confidence": 0.80,
        "needs_attention_confidence": 0.60,
        "minimum_successful_runs": 1,
        "require_verification": True,
        "require_user_acceptance": False,
        "semantic_duplicate_threshold": 0.88,
        "retain_rejected_days": 90,
        "quarantine_on_validation_failure": True,
    },
}


FIELD_NAME_BY_ENV: dict[str, str] = {
    "MANA_AI_PROVIDER": "mana_ai_provider",
    "MANA_PRIMARY_MODEL": "mana_primary_model",
    "MANA_EMBEDDING_MODEL": "mana_embedding_model",
    "OPENAI_API_KEY": "openai_api_key",
    "OPENAI_BASE_URL": "openai_base_url",
    "OPENAI_CHAT_MODEL": "openai_chat_model",
    "OPENAI_TOOL_WORKER_MODEL": "openai_tool_worker_model",
    "OPENAI_CODING_PLANNER_MODEL": "openai_coding_planner_model",
    "OPENAI_EMBED_MODEL": "openai_embed_model",
    "DEFAULT_TOP_K": "default_top_k",
    "LLM_MODEL": "llm_model",
    "MANA_LLM_LOG_FILE": "mana_llm_log_file",
    "MANA_GITHUB_TOKEN": "mana_github_token",
    "MANA_GITHUB_CREDENTIAL_SOURCE": "mana_github_credential_source",
    "MANA_GITHUB_SECRET_REF": "mana_github_secret_ref",
    "MANA_GITHUB_METADATA_ENABLED": "mana_github_metadata_enabled",
    "MANA_SEARCH_ENABLE_WEB": "mana_search_enable_web",
    "MANA_SEARCH_ENABLE_GITHUB": "mana_search_enable_github",
    "MANA_SEARCH_MAX_RESULTS": "mana_search_max_results",
    "MANA_SEARCH_TIMEOUT_SECONDS": "mana_search_timeout_seconds",
    "MANA_SEARCH_MEMORY_TTL_DAYS": "mana_search_memory_ttl_days",
    "MANA_MEMORY_MODE": "mana_memory_mode",
    "MANA_MEMORY_PROVIDER": "mana_memory_provider",
    "MANA_MEMORY_FALLBACK_TO_INTERNAL": "mana_memory_fallback_to_internal",
    "MANA_MEMORY_SECRET_REF": "mana_memory_secret_ref",
    "MEM0_ORG_ID": "mem0_org_id",
    "MEM0_PROJECT_ID": "mem0_project_id",
    "MEM0_BASE_URL": "mem0_base_url",
    "MANA_MEMORY_TIMEOUT_SECONDS": "mana_memory_timeout_seconds",
    "MANA_WEB_SEARCH_PROVIDER": "mana_web_search_provider",
    "MANA_WEB_SEARCH_API_KEY": "mana_web_search_api_key",
    "MANA_WEB_SEARCH_MAX_RESULTS": "mana_web_search_max_results",
    "MANA_WEB_SEARCH_ENGINE_ID": "mana_web_search_engine_id",
    "MANA_WEB_SEARCH_BASE_URL": "mana_web_search_base_url",
    "MANA_WEB_SEARCH_ENDPOINT": "mana_web_search_endpoint",
    "MANA_SEARCH_MAX_INJECTED_RESULTS": "mana_search_max_injected_results",
    "MANA_SEARCH_MAX_SUMMARY_WORDS": "mana_search_max_summary_words",
    "MANA_SEARCH_ENABLE_ASK_AGENT": "mana_search_enable_ask_agent",
    "MANA_MODEL_TOOL_WORKER": "mana_model_tool_worker",
    "MANA_WORKSPACE_ALLOWED_ROOTS": "mana_workspace_allowed_roots",
    "MANA_API_TOKEN": "mana_api_token",
    "MANA_MCP_SERVER_TOKEN": "mana_mcp_server_token",
    "MANA_ACP_ENABLED": "mana_acp_enabled",
    "MANA_ACP_ALLOWED_ROOTS": "mana_acp_allowed_roots",
    "MANA_ACP_MCP_FORWARDING": "mana_acp_mcp_forwarding",
    "MANA_ACP_SESSION_LOAD": "mana_acp_session_load",
    "MANA_ACP_SESSION_RETENTION_DAYS": "mana_acp_session_retention_days",
    "MANA_A2A_SERVER_ENABLED": "mana_a2a_server_enabled",
    "MANA_A2A_HOST": "mana_a2a_host",
    "MANA_A2A_PORT": "mana_a2a_port",
    "MANA_A2A_PUBLIC_BASE_URL": "mana_a2a_public_base_url",
    "MANA_A2A_SERVER_TOKEN": "mana_a2a_server_token",
    "MANA_A2A_ENABLED_SKILLS": "mana_a2a_enabled_skills",
    "MANA_A2A_STREAMING": "mana_a2a_streaming",
    "MANA_A2A_PUSH_NOTIFICATIONS": "mana_a2a_push_notifications",
    "MANA_A2A_TASK_RETENTION_DAYS": "mana_a2a_task_retention_days",
    "MANA_A2A_MAX_REQUEST_BYTES": "mana_a2a_max_request_bytes",
    "MANA_A2A_MAX_ARTIFACT_BYTES": "mana_a2a_max_artifact_bytes",
    "MANA_A2A_MAX_CONCURRENT_TASKS": "mana_a2a_max_concurrent_tasks",
    "MANA_A2A_DELEGATION_ENABLED": "mana_a2a_delegation_enabled",
    "MANA_A2A_MAX_DELEGATION_DEPTH": "mana_a2a_max_delegation_depth",
    "MANA_BROWSER_ENABLED": "mana_browser_enabled",
    "MANA_BROWSER_HEADLESS": "mana_browser_headless",
    "MANA_BROWSER_TIMEOUT_SECONDS": "mana_browser_timeout_seconds",
    "MANA_BROWSER_PERSIST_AUTH": "mana_browser_persist_auth",
    "MANA_BROWSER_DOWNLOAD_MAX_MB": "mana_browser_download_max_mb",
    "MANA_BROWSER_UPLOAD_ROOTS": "mana_browser_upload_roots",
    "MANA_BROWSER_ARTIFACT_DIR": "mana_browser_artifact_dir",
    "MANA_BROWSER_PROFILE_MAX_AGE_DAYS": "mana_browser_profile_max_age_days",
    "MANA_CODEX_ENABLED": "mana_codex_enabled",
    "MANA_CODEX_MAX_WORKERS": "mana_codex_max_workers",
    "MANA_CODEX_STREAM_EVENTS": "mana_codex_stream_events",
    "MANA_CODEX_WORKTREE_ISOLATION": "mana_codex_worktree_isolation",
    "MANA_CODEX_TASK_TIMEOUT_SECONDS": "mana_codex_task_timeout_seconds",
    "MANA_CODEX_ALLOW_NETWORK": "mana_codex_allow_network",
    "MANA_CODEX_MODEL": "mana_codex_model",
    "MANA_CODEX_BIN": "mana_codex_bin",
    "MANA_LANE_CONTRACTS": "mana_lane_contracts",
    "MANA_LANE_GLOBAL_WORKER_LIMIT": "mana_lane_global_worker_limit",
    "MANA_LANE_PROVIDER_LIMITS": "mana_lane_provider_limits",
    "MANA_LANE_SESSION_TOKEN_BUDGET": "mana_lane_session_token_budget",
    "MANA_LANE_GLOBAL_TOKEN_BUDGET": "mana_lane_global_token_budget",
    "MANA_EXECUTION_DEFAULT_PROVIDER": "mana_execution_default_provider",
    "MANA_EXECUTION_ALLOWED_PROVIDERS": "mana_execution_allowed_providers",
    "MANA_EXECUTION_CLEANUP_ON_EXIT": "mana_execution_cleanup_on_exit",
    "MANA_EXECUTION_IDLE_TIMEOUT_SECONDS": "mana_execution_idle_timeout_seconds",
    "MANA_EXECUTION_MAX_LIFETIME_SECONDS": "mana_execution_max_lifetime_seconds",
    "MANA_EXECUTION_GLOBAL_CONCURRENCY": "mana_execution_global_concurrency",
    "MANA_EXECUTION_ROUTING": "mana_execution_routing",
    "MANA_EXECUTION_PROVIDERS": "mana_execution_providers",
}

CONFIG_WRITE_ORDER = [
    "MANA_CONFIG_SCHEMA_VERSION",
    "MANA_AI_PROVIDER",
    "MANA_PRIMARY_MODEL",
    "MANA_EMBEDDING_MODEL",
    "MANA_CONFIGURED_PROVIDERS",
    "OPENAI_BASE_URL",
    "OPENAI_CHAT_MODEL",
    "LLM_MODEL",
    "OPENAI_TOOL_WORKER_MODEL",
    "OPENAI_CODING_PLANNER_MODEL",
    "OPENAI_EMBED_MODEL",
    "MODEL_LEVEL_3_HIGH_REASONING",
    "MODEL_LEVEL_2_CODING",
    "MODEL_LEVEL_1_FAST_TOOL",
    "MANA_MODEL_MAIN",
    "MANA_MODEL_HEAD_DECISION",
    "MANA_MODEL_PLANNER",
    "MANA_MODEL_CODING",
    "MANA_MODEL_VERIFIER",
    "MANA_MODEL_REVIEWER",
    "MANA_MODEL_TOOL",
    "MANA_MODEL_TOOL_WORKER",
    "MANA_MODEL_SUMMARIZER",
    "DEFAULT_TOP_K",
    "MANA_LLM_LOG_FILE",
    "MANA_LLM_API_MODE",
    "MANA_LLM_REASONING_EFFORT",
    "MANA_LLM_SUPPORTS_RESPONSES_API",
    "MANA_LLM_SUPPORTS_CHAT_COMPLETIONS",
    "MANA_LLM_SUPPORTS_TOOLS",
    "MANA_LLM_SUPPORTS_REASONING",
    "MANA_LLM_SUPPORTS_TOOLS_WITH_CHAT_REASONING",
    "MANA_SEARCH_ENABLE_WEB",
    "MANA_SEARCH_ENABLE_GITHUB",
    "MANA_GITHUB_CREDENTIAL_SOURCE",
    "MANA_GITHUB_SECRET_REF",
    "MANA_GITHUB_METADATA_ENABLED",
    "MANA_SEARCH_MAX_RESULTS",
    "MANA_SEARCH_TIMEOUT_SECONDS",
    "MANA_SEARCH_MEMORY_TTL_DAYS",
    "MANA_MEMORY_MODE",
    "MANA_MEMORY_PROVIDER",
    "MANA_MEMORY_FALLBACK_TO_INTERNAL",
    "MANA_MEMORY_SECRET_REF",
    "MEM0_ORG_ID",
    "MEM0_PROJECT_ID",
    "MEM0_BASE_URL",
    "MANA_MEMORY_TIMEOUT_SECONDS",
    "MANA_WEB_SEARCH_PROVIDER",
    "MANA_WEB_SEARCH_MAX_RESULTS",
    "MANA_WEB_SEARCH_ENGINE_ID",
    "MANA_WEB_SEARCH_BASE_URL",
    "MANA_WEB_SEARCH_ENDPOINT",
    "MANA_WEB_SEARCH_QUERY_PARAM",
    "MANA_SEARCH_MAX_INJECTED_RESULTS",
    "MANA_SEARCH_MAX_SUMMARY_WORDS",
    "MANA_SEARCH_ENABLE_ASK_AGENT",
    "MANA_BROWSER_ENABLED",
    "MANA_ACP_ENABLED",
    "MANA_ACP_ALLOWED_ROOTS",
    "MANA_ACP_MCP_FORWARDING",
    "MANA_ACP_SESSION_LOAD",
    "MANA_ACP_SESSION_RETENTION_DAYS",
    "MANA_A2A_SERVER_ENABLED",
    "MANA_A2A_HOST",
    "MANA_A2A_PORT",
    "MANA_A2A_PUBLIC_BASE_URL",
    "MANA_A2A_ENABLED_SKILLS",
    "MANA_A2A_STREAMING",
    "MANA_A2A_PUSH_NOTIFICATIONS",
    "MANA_A2A_TASK_RETENTION_DAYS",
    "MANA_A2A_MAX_REQUEST_BYTES",
    "MANA_A2A_MAX_ARTIFACT_BYTES",
    "MANA_A2A_MAX_CONCURRENT_TASKS",
    "MANA_A2A_DELEGATION_ENABLED",
    "MANA_A2A_MAX_DELEGATION_DEPTH",
    "MANA_BROWSER_HEADLESS",
    "MANA_BROWSER_TIMEOUT_SECONDS",
    "MANA_BROWSER_PERSIST_AUTH",
    "MANA_BROWSER_DOWNLOAD_MAX_MB",
    "MANA_BROWSER_UPLOAD_ROOTS",
    "MANA_BROWSER_ARTIFACT_DIR",
    "MANA_BROWSER_PROFILE_MAX_AGE_DAYS",
    "MANA_CODEX_ENABLED",
    "MANA_CODEX_MAX_WORKERS",
    "MANA_CODEX_STREAM_EVENTS",
    "MANA_CODEX_WORKTREE_ISOLATION",
    "MANA_CODEX_TASK_TIMEOUT_SECONDS",
    "MANA_CODEX_ALLOW_NETWORK",
    "MANA_CODEX_MODEL",
    "MANA_CODEX_BIN",
    "MANA_LANE_CONTRACTS",
    "MANA_LANE_GLOBAL_WORKER_LIMIT",
    "MANA_LANE_PROVIDER_LIMITS",
    "MANA_LANE_SESSION_TOKEN_BUDGET",
    "MANA_LANE_GLOBAL_TOKEN_BUDGET",
]


class UserConfigError(RuntimeError):
    pass


def ensure_user_config_dir() -> Path:
    CONFIG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        CONFIG_DIR.chmod(0o700)
    except OSError:
        pass
    return CONFIG_DIR


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        return {}
    return dict(data)


def load_user_config() -> dict[str, Any]:
    return _read_toml(CONFIG_FILE)


def load_user_secrets() -> dict[str, Any]:
    return _read_toml(SECRETS_FILE)


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    text = str(value)
    return json.dumps(text)


def _write_toml(path: Path, values: dict[str, Any], *, mode: int = 0o600) -> None:
    ensure_user_config_dir()
    ordered_keys = [key for key in CONFIG_WRITE_ORDER if key in values]
    ordered_keys.extend(sorted(key for key in values if key not in set(ordered_keys)))
    lines = [f"{key} = {_toml_scalar(values[key])}" for key in ordered_keys if not isinstance(values[key], dict)]

    def append_tables(prefix: str, table: dict[str, Any]) -> None:
        scalar_items = [(key, value) for key, value in table.items() if not isinstance(value, dict)]
        if scalar_items:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(f"[{prefix}]")
            lines.extend(f"{key} = {_toml_scalar(value)}" for key, value in scalar_items)
        for key, value in table.items():
            if isinstance(value, dict):
                append_tables(f"{prefix}.{key}", value)

    for key in ordered_keys:
        value = values[key]
        if isinstance(value, dict):
            append_tables(key, value)
    payload = "\n".join(lines).rstrip() + "\n"
    temp_path: Path | None = None
    try:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        temp_path = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.chmod(mode)
        os.replace(temp_path, path)
    except OSError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise UserConfigError(f"Could not save {path.name}; the previous configuration was preserved.") from exc
    try:
        path.chmod(mode)
    except OSError:
        pass


def save_user_config(values: dict[str, Any], *, merge: bool = True) -> None:
    current = load_user_config() if merge else {}
    current.update(
        {
            key: value
            for key, value in values.items()
            if key not in SECRET_KEYS and key not in NON_PERSISTED_SECRET_KEYS
        }
    )
    _write_toml(CONFIG_FILE, current, mode=0o600)


def save_user_secrets(values: dict[str, Any], *, merge: bool = True) -> None:
    current = load_user_secrets() if merge else {}
    current.update({key: value for key, value in values.items() if key in SECRET_KEYS})
    _write_toml(SECRETS_FILE, current, mode=0o600)


def save_effective_user_config(values: dict[str, Any], *, merge: bool = True) -> None:
    save_user_config(values, merge=merge)
    save_user_secrets(values, merge=merge)


def has_user_config() -> bool:
    return CONFIG_FILE.exists() or SECRETS_FILE.exists()


def is_user_config_valid() -> bool:
    effective = load_effective_settings(include_env=True)
    return bool(
        str(effective.get("OPENAI_API_KEY", "") or "").strip()
        and str(effective.get("OPENAI_CHAT_MODEL", "") or "").strip()
    )


def load_effective_settings(*, include_env: bool = True) -> dict[str, Any]:
    """Load Mana-managed settings exclusively from the user configuration.

    Explicit Mana configuration wins over environment variables. Environment
    variables may fill missing values for CI and automation, but repository
    ``.env`` files are deliberately never loaded here.
    """
    values = dict(DEFAULT_USER_CONFIG)
    user_values = load_user_config()
    values.update(user_values)
    secret_values = load_user_secrets()
    values.update(secret_values)
    if include_env:
        explicit = {**user_values, **secret_values}
        for key in set(DEFAULT_USER_CONFIG) | set(FIELD_NAME_BY_ENV) | SECRET_KEYS:
            if key not in explicit and key in os.environ:
                values[key] = os.environ[key]
    if user_values.get("LLM_MODEL") and not user_values.get("OPENAI_CHAT_MODEL"):
        values["OPENAI_CHAT_MODEL"] = user_values["LLM_MODEL"]
    if not values.get("LLM_MODEL") and values.get("OPENAI_CHAT_MODEL"):
        values["LLM_MODEL"] = values["OPENAI_CHAT_MODEL"]
    return values


def settings_source_for_pydantic() -> dict[str, Any]:
    effective = load_effective_settings(include_env=False)
    return {
        env_name: effective[env_name]
        for env_name in FIELD_NAME_BY_ENV
        if env_name in effective and effective[env_name] not in (None, "")
    }


def get_setting(name: str, default: Any = None) -> Any:
    return load_effective_settings(include_env=False).get(name, default)


def mask_secret(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    if len(text) <= 4:
        return "****"
    return "*" * max(8, min(12, len(text) - 4)) + text[-4:]


def masked_config_summary() -> dict[str, Any]:
    values = load_effective_settings(include_env=True)
    return {
        key: mask_secret(value) if key in SECRET_KEYS else value
        for key, value in sorted(values.items())
    }


def validate_base_url(value: str) -> str:
    text = str(value or "").strip()
    if not (text.startswith("http://") or text.startswith("https://")):
        raise UserConfigError("Base URL must begin with http:// or https://.")
    return text.rstrip("/")


def validate_positive_int(name: str, value: Any, *, minimum: int = 1, maximum: int = 1000) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise UserConfigError(f"{name} must be an integer.") from exc
    if parsed < minimum or parsed > maximum:
        raise UserConfigError(f"{name} must be between {minimum} and {maximum}.")
    return parsed


def validate_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    raise UserConfigError(f"Invalid boolean value: {value!r}.")


def validate_model_level(value: str) -> str:
    allowed = {
        "MODEL_LEVEL_1_FAST_TOOL",
        "MODEL_LEVEL_2_CODING",
        "MODEL_LEVEL_3_HIGH_REASONING",
    }
    text = str(value or "").strip()
    if not text:
        raise UserConfigError("Model assignment cannot be empty.")
    # Advanced role mapping may store a direct (preferably provider-qualified)
    # model ID. Symbolic level names remain strictly validated.
    if text.startswith("MODEL_LEVEL_") and text not in allowed:
        raise UserConfigError(f"Model level must be one of: {', '.join(sorted(allowed))}.")
    return text


def validate_config_values(values: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(values)
    if cleaned.get("OPENAI_BASE_URL"):
        cleaned["OPENAI_BASE_URL"] = validate_base_url(str(cleaned["OPENAI_BASE_URL"]))
    for name in (
        "DEFAULT_TOP_K",
        "MANA_SEARCH_MAX_RESULTS",
        "MANA_SEARCH_TIMEOUT_SECONDS",
        "MANA_SEARCH_MEMORY_TTL_DAYS",
        "MANA_MEMORY_TIMEOUT_SECONDS",
        "MANA_WEB_SEARCH_MAX_RESULTS",
        "MANA_LANE_GLOBAL_WORKER_LIMIT",
    ):
        if name in cleaned:
            cleaned[name] = validate_positive_int(name, cleaned[name], minimum=1, maximum=1000)
    for name in ("MANA_LANE_SESSION_TOKEN_BUDGET", "MANA_LANE_GLOBAL_TOKEN_BUDGET"):
        if name in cleaned:
            value = int(cleaned[name] or 0)
            if value < 0:
                raise UserConfigError(f"{name} must be zero (unlimited) or a positive integer.")
            cleaned[name] = value
    for name in (
        "MANA_SEARCH_ENABLE_WEB",
        "MANA_SEARCH_ENABLE_GITHUB",
        "MANA_LLM_SUPPORTS_RESPONSES_API",
        "MANA_LLM_SUPPORTS_CHAT_COMPLETIONS",
        "MANA_LLM_SUPPORTS_TOOLS",
        "MANA_LLM_SUPPORTS_REASONING",
        "MANA_LLM_SUPPORTS_TOOLS_WITH_CHAT_REASONING",
        "MANA_MEMORY_FALLBACK_TO_INTERNAL",
    ):
        if name in cleaned:
            if str(cleaned[name] or "").strip():
                cleaned[name] = validate_bool(cleaned[name])
    if "MANA_MEMORY_MODE" in cleaned or "MANA_MEMORY_PROVIDER" in cleaned:
        from mana_agent.memory.config import MemoryConfig

        MemoryConfig(
            mode=str(cleaned.get("MANA_MEMORY_MODE") or "internal").lower(),
            provider=str(cleaned.get("MANA_MEMORY_PROVIDER") or "mana").lower(),
            fallback_to_internal=bool(cleaned.get("MANA_MEMORY_FALLBACK_TO_INTERNAL", False)),
            api_key=str(cleaned.get("MEM0_API_KEY") or ""),
            secret_ref=str(cleaned.get("MANA_MEMORY_SECRET_REF") or ""),
            org_id=str(cleaned.get("MEM0_ORG_ID") or ""),
            project_id=str(cleaned.get("MEM0_PROJECT_ID") or ""),
            base_url=str(cleaned.get("MEM0_BASE_URL") or ""),
            timeout_seconds=float(cleaned.get("MANA_MEMORY_TIMEOUT_SECONDS") or 15),
        ).validate()
    for name in (
        "MANA_MODEL_MAIN",
        "MANA_MODEL_HEAD_DECISION",
        "MANA_MODEL_PLANNER",
        "MANA_MODEL_CODING",
        "MANA_MODEL_VERIFIER",
        "MANA_MODEL_REVIEWER",
        "MANA_MODEL_TOOL",
        "MANA_MODEL_TOOL_WORKER",
        "MANA_MODEL_SUMMARIZER",
    ):
        if name in cleaned:
            cleaned[name] = validate_model_level(str(cleaned[name]))
    for name in (
        "MODEL_LEVEL_1_FAST_TOOL",
        "MODEL_LEVEL_2_CODING",
        "MODEL_LEVEL_3_HIGH_REASONING",
    ):
        if name in cleaned:
            cleaned[name] = str(cleaned[name] or "").strip()
    return cleaned


@dataclass(frozen=True, slots=True)
class CachedModels:
    provider: str
    base_url: str
    created_at: str
    models: list[str]


def provider_cache_key(provider: str, base_url: str) -> str:
    seed = f"{provider.strip().lower()}|{base_url.rstrip('/')}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def load_model_cache(provider: str, base_url: str) -> CachedModels | None:
    if not MODEL_CACHE_FILE.exists():
        return None
    try:
        data = json.loads(MODEL_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    item = data.get(provider_cache_key(provider, base_url))
    if not isinstance(item, dict):
        return None
    models = [str(model) for model in item.get("models", []) if str(model).strip()]
    return CachedModels(
        provider=str(item.get("provider") or provider),
        base_url=str(item.get("base_url") or base_url),
        created_at=str(item.get("created_at") or ""),
        models=models,
    )


def save_model_cache(provider: str, base_url: str, models: list[str]) -> None:
    ensure_user_config_dir()
    data: dict[str, Any] = {}
    if MODEL_CACHE_FILE.exists():
        try:
            loaded = json.loads(MODEL_CACHE_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError):
            data = {}
    data[provider_cache_key(provider, base_url)] = {
        "provider": provider,
        "base_url": base_url.rstrip("/"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": sorted(dict.fromkeys(models)),
    }
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=".model_cache.", suffix=".tmp", dir=str(CONFIG_DIR))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temp_path, MODEL_CACHE_FILE)
    except OSError:
        temp_path.unlink(missing_ok=True)
        raise
    try:
        MODEL_CACHE_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def invalidate_model_cache() -> None:
    MODEL_CACHE_FILE.unlink(missing_ok=True)


def migrate_legacy_config() -> list[str]:
    """Migrate legacy flat settings without losing unknown keys.

    The migration is idempotent. A backup is created only when secrets must be
    removed from the normal configuration file, which is the sole destructive
    schema change.
    """
    if not CONFIG_FILE.exists():
        return []
    config = load_user_config()
    secrets = load_user_secrets()
    messages: list[str] = []
    destructive = False
    for key in SECRET_KEYS:
        value = config.pop(key, None)
        if value not in (None, ""):
            secrets.setdefault(key, value)
            destructive = True
    provider = str(config.get("MANA_AI_PROVIDER") or "").strip()
    if not provider:
        base_url = str(config.get("OPENAI_BASE_URL") or "").lower()
        provider = "nvidia" if "nvidia" in base_url else "openai"
        config["MANA_AI_PROVIDER"] = provider
    from mana_agent.config.provider_registry import qualify_model_id

    chat_model = str(config.get("OPENAI_CHAT_MODEL") or config.get("LLM_MODEL") or "").strip()
    if chat_model and not config.get("MANA_PRIMARY_MODEL"):
        config["MANA_PRIMARY_MODEL"] = qualify_model_id(provider, chat_model)
    embed_model = str(config.get("OPENAI_EMBED_MODEL") or "").strip()
    if embed_model and not config.get("MANA_EMBEDDING_MODEL"):
        config["MANA_EMBEDDING_MODEL"] = qualify_model_id(provider, embed_model)
    config.setdefault("MANA_CONFIGURED_PROVIDERS", [provider])
    config["MANA_CONFIG_SCHEMA_VERSION"] = 2
    if destructive:
        backup = CONFIG_FILE.with_suffix(".toml.bak")
        if not backup.exists():
            backup.write_bytes(CONFIG_FILE.read_bytes())
            try:
                backup.chmod(0o600)
            except OSError:
                pass
        messages.append(f"Moved legacy credentials to {SECRETS_FILE.name}; backup: {backup.name}.")
    save_user_config(config, merge=False)
    if secrets:
        save_user_secrets(secrets, merge=False)
    return messages
