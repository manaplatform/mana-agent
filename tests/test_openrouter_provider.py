from __future__ import annotations

import json
from pathlib import Path

import pytest

from mana_agent.config.catalog_service import ModelCatalogService
from mana_agent.config.inference_provider import resolve_inference_connection
from mana_agent.config.model_catalog import ModelCapability, ModelPurpose, filter_models
from mana_agent.config.provider_registry import PROVIDERS, split_qualified_model_id
from mana_agent.config.session import ConfigurationDraft
from mana_agent.config.settings import Settings
from mana_agent.config import user_config
from mana_agent.model_routing.models import Complexity, LatencyClass, RiskLevel, RoutingRequest
from mana_agent.model_routing.profiles import profiles_from_legacy_configuration
from mana_agent.model_routing.router import ModelRouter
from mana_agent.tui.model_picker import parse_openrouter_models


@pytest.fixture()
def isolated_openrouter_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_dir = tmp_path / ".mana"
    monkeypatch.setenv("MANA_HOME", str(config_dir))
    return config_dir


def test_openrouter_is_first_class_provider() -> None:
    provider = PROVIDERS.get("openrouter")
    assert provider.display_name == "OpenRouter"
    assert provider.default_base_url == "https://openrouter.ai/api/v1"
    assert provider.api_key_env == "OPENROUTER_API_KEY"
    assert split_qualified_model_id("openrouter/anthropic/claude-sonnet") == ("openrouter", "anthropic/claude-sonnet")


def test_openrouter_catalog_preserves_ids_and_capabilities() -> None:
    records = parse_openrouter_models({"data": [{
        "id": "anthropic/claude-3.7-sonnet:free",
        "name": "Claude",
        "context_length": 200000,
        "supported_parameters": ["tools", "response_format", "reasoning"],
        "architecture": {"input_modalities": ["text", "image"]},
    }]})
    models = ModelCatalogService(fetcher=lambda **_kwargs: records).refresh(
        provider="openrouter", base_url="https://openrouter.ai/api/v1", api_key="secret"
    )
    assert models[0].id == "anthropic/claude-3.7-sonnet:free"
    assert models[0].context_window == 200000
    assert {ModelCapability.TOOL_CALLING, ModelCapability.STRUCTURED_OUTPUT, ModelCapability.IMAGE_INPUT, ModelCapability.REASONING} <= models[0].capabilities
    assert filter_models(models, ModelPurpose.AGENT) == models


def test_openrouter_configuration_persists_provider_model_pair(isolated_openrouter_config: Path) -> None:
    draft = ConfigurationDraft.load()
    draft.set_secret("OPENROUTER_API_KEY", "secret")
    draft.values["OPENROUTER_BASE_URL"] = "https://openrouter.ai/api/v1"
    draft.set_models(provider="openrouter", high="openai/gpt-4.1-mini", coding="openai/gpt-4.1-mini", fast="openai/gpt-4.1-mini", embedding="openai/text-embedding-3-small")
    draft.save()
    values = user_config.load_effective_settings(include_env=False)
    assert values["MANA_AI_PROVIDER"] == "openrouter"
    assert values["MANA_PRIMARY_MODEL"] == "openrouter/openai/gpt-4.1-mini"
    assert user_config.load_user_secrets()["OPENROUTER_API_KEY"] == "secret"
    assert "secret" not in (isolated_openrouter_config / "config.toml").read_text()


def test_openrouter_connection_uses_its_key_and_attribution_headers(isolated_openrouter_config: Path) -> None:
    user_config.save_effective_user_config({"MANA_AI_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "secret"}, merge=False)
    connection = resolve_inference_connection(Settings())
    assert connection.provider == "openrouter"
    assert connection.api_key == "secret"
    assert connection.base_url == "https://openrouter.ai/api/v1"
    assert connection.headers["X-OpenRouter-Title"] == "Mana-Agent"
    assert "secret" not in json.dumps({"provider": connection.provider, "base_url": connection.base_url, "headers": connection.headers})


def test_shared_openrouter_fast_assignment_remains_eligible_for_tool_lane(
    isolated_openrouter_config: Path,
) -> None:
    user_config.save_effective_user_config(
        {
            "MANA_AI_PROVIDER": "openrouter",
            "MODEL_LEVEL_1_FAST_TOOL": "moonshotai/kimi-k3",
            "MODEL_LEVEL_2_CODING": "moonshotai/kimi-k3",
            "MODEL_LEVEL_3_HIGH_REASONING": "moonshotai/kimi-k3",
        },
        merge=False,
    )
    profile = profiles_from_legacy_configuration(default_provider="openrouter")[0]
    assert profile.latency_class is LatencyClass.INTERACTIVE
    decision = ModelRouter([profile]).route(
        RoutingRequest(
            role="tool_worker",
            task_description="Run a tool task.",
            task_type="tool",
            complexity=Complexity.LOW,
            risk=RiskLevel.LOW,
            latency_requirement=LatencyClass.INTERACTIVE,
        )
    )
    assert (decision.provider, decision.selected_model) == ("openrouter", "moonshotai/kimi-k3")
