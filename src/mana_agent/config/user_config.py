from __future__ import annotations

import hashlib
import json
import stat
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
}


DEFAULT_USER_CONFIG: dict[str, Any] = {
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
    "MANA_SEARCH_ENABLE_WEB": True,
    "MANA_SEARCH_ENABLE_GITHUB": True,
    "MANA_SEARCH_MAX_RESULTS": 8,
    "MANA_SEARCH_TIMEOUT_SECONDS": 15,
    "MANA_SEARCH_MEMORY_TTL_DAYS": 14,
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
    "MANA_BROWSER_ENABLED": True,
    "MANA_BROWSER_HEADLESS": True,
    "MANA_BROWSER_TIMEOUT_SECONDS": 30,
    "MANA_BROWSER_PERSIST_AUTH": False,
    "MANA_BROWSER_DOWNLOAD_MAX_MB": 100,
    "MANA_BROWSER_UPLOAD_ROOTS": "",
    "MANA_BROWSER_ARTIFACT_DIR": "",
    "MANA_BROWSER_PROFILE_MAX_AGE_DAYS": 30,
}


FIELD_NAME_BY_ENV: dict[str, str] = {
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
    "MANA_SEARCH_ENABLE_WEB": "mana_search_enable_web",
    "MANA_SEARCH_ENABLE_GITHUB": "mana_search_enable_github",
    "MANA_SEARCH_MAX_RESULTS": "mana_search_max_results",
    "MANA_SEARCH_TIMEOUT_SECONDS": "mana_search_timeout_seconds",
    "MANA_SEARCH_MEMORY_TTL_DAYS": "mana_search_memory_ttl_days",
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
    "MANA_BROWSER_ENABLED": "mana_browser_enabled",
    "MANA_BROWSER_HEADLESS": "mana_browser_headless",
    "MANA_BROWSER_TIMEOUT_SECONDS": "mana_browser_timeout_seconds",
    "MANA_BROWSER_PERSIST_AUTH": "mana_browser_persist_auth",
    "MANA_BROWSER_DOWNLOAD_MAX_MB": "mana_browser_download_max_mb",
    "MANA_BROWSER_UPLOAD_ROOTS": "mana_browser_upload_roots",
    "MANA_BROWSER_ARTIFACT_DIR": "mana_browser_artifact_dir",
    "MANA_BROWSER_PROFILE_MAX_AGE_DAYS": "mana_browser_profile_max_age_days",
}

CONFIG_WRITE_ORDER = [
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
    "MANA_SEARCH_ENABLE_WEB",
    "MANA_SEARCH_ENABLE_GITHUB",
    "MANA_SEARCH_MAX_RESULTS",
    "MANA_SEARCH_TIMEOUT_SECONDS",
    "MANA_SEARCH_MEMORY_TTL_DAYS",
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
    "MANA_BROWSER_HEADLESS",
    "MANA_BROWSER_TIMEOUT_SECONDS",
    "MANA_BROWSER_PERSIST_AUTH",
    "MANA_BROWSER_DOWNLOAD_MAX_MB",
    "MANA_BROWSER_UPLOAD_ROOTS",
    "MANA_BROWSER_ARTIFACT_DIR",
    "MANA_BROWSER_PROFILE_MAX_AGE_DAYS",
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
    text = str(value)
    return json.dumps(text)


def _write_toml(path: Path, values: dict[str, Any], *, mode: int = 0o600) -> None:
    ensure_user_config_dir()
    ordered_keys = [key for key in CONFIG_WRITE_ORDER if key in values]
    ordered_keys.extend(sorted(key for key in values if key not in set(ordered_keys)))
    lines = [f"{key} = {_toml_scalar(values[key])}" for key in ordered_keys]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    try:
        path.chmod(mode)
    except OSError:
        pass


def save_user_config(values: dict[str, Any], *, merge: bool = True) -> None:
    current = load_user_config() if merge else {}
    current.update({key: value for key, value in values.items() if key not in SECRET_KEYS})
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
    return bool(str(effective.get("OPENAI_API_KEY", "") or "").strip())


def load_effective_settings(*, include_env: bool = True) -> dict[str, Any]:
    """Load Mana-managed settings exclusively from the user configuration.

    ``include_env`` remains an accepted compatibility argument for callers
    from older releases, but environment variables and repository ``.env``
    files must never override the user-selected configuration.
    """
    _ = include_env
    values = dict(DEFAULT_USER_CONFIG)
    user_values = load_user_config()
    values.update(user_values)
    values.update(load_user_secrets())
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
    if text not in allowed:
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
        "MANA_WEB_SEARCH_MAX_RESULTS",
    ):
        if name in cleaned:
            cleaned[name] = validate_positive_int(name, cleaned[name], minimum=1, maximum=1000)
    for name in ("MANA_SEARCH_ENABLE_WEB", "MANA_SEARCH_ENABLE_GITHUB"):
        if name in cleaned:
            cleaned[name] = validate_bool(cleaned[name])
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
    MODEL_CACHE_FILE.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        MODEL_CACHE_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
