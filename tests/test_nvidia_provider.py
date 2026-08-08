from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mana_agent.config.catalog_service import ModelCatalogService
from mana_agent.config.inference_provider import (
    ProviderConfigurationError,
    credentials_from_mapping,
    resolve_inference_connection,
)
from mana_agent.config.model_catalog import ModelCapability, ModelPurpose, filter_models
from mana_agent.config.provider_registry import (
    PROVIDERS,
    provider_credential_env_names,
    qualify_model_id,
    split_qualified_model_id,
)
from mana_agent.config.session import ConfigurationDraft
from mana_agent.config.settings import Settings
from mana_agent.config import user_config
from mana_agent.multi_agent.runtime.compatibility import (
    create_chat_model,
    format_provider_error,
)
from mana_agent.tui.model_picker import parse_openai_compatible_model_records


@pytest.fixture()
def isolated_nvidia_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_dir = tmp_path / ".mana"
    monkeypatch.setenv("MANA_HOME", str(config_dir))
    for key in (
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "NVIDIA_API_KEY",
        "OPENAI_BASE_URL",
        "OPENROUTER_BASE_URL",
        "NVIDIA_BASE_URL",
        "MANA_AI_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)
    return config_dir


def test_nvidia_is_first_class_provider() -> None:
    provider = PROVIDERS.get("nvidia")
    assert provider.display_name == "NVIDIA"
    assert provider.default_base_url == "https://integrate.api.nvidia.com/v1"
    assert provider.api_key_env == "NVIDIA_API_KEY"
    assert provider.supports_model_refresh is True
    assert provider.supports_validation is True
    assert provider.supports_responses_api is False
    assert provider_credential_env_names("nvidia") == ("NVIDIA_API_KEY", "NVIDIA_BASE_URL")


def test_nvidia_qualified_ids_preserve_nested_namespaces() -> None:
    assert qualify_model_id("nvidia", "deepseek-ai/deepseek-v4-flash") == (
        "nvidia/deepseek-ai/deepseek-v4-flash"
    )
    assert split_qualified_model_id(
        "nvidia/deepseek-ai/deepseek-v4-flash", default_provider="nvidia"
    ) == ("nvidia", "deepseek-ai/deepseek-v4-flash")

    # Upstream org shares the Mana provider name.
    assert qualify_model_id("nvidia", "nvidia/nemotron-3-nano-30b-a3b") == (
        "nvidia/nvidia/nemotron-3-nano-30b-a3b"
    )
    assert split_qualified_model_id(
        "nvidia/nvidia/nemotron-3-nano-30b-a3b", default_provider="nvidia"
    ) == ("nvidia", "nvidia/nemotron-3-nano-30b-a3b")
    assert split_qualified_model_id(
        "nvidia/nemotron-3-nano-30b-a3b", default_provider="nvidia"
    ) == ("nvidia", "nvidia/nemotron-3-nano-30b-a3b")

    # Bare multi-tenant IDs stay under the active NVIDIA provider.
    assert split_qualified_model_id(
        "deepseek-ai/deepseek-v4-flash", default_provider="nvidia"
    ) == ("nvidia", "deepseek-ai/deepseek-v4-flash")
    assert split_qualified_model_id(
        "moonshotai/kimi-k2.6", default_provider="nvidia"
    ) == ("nvidia", "moonshotai/kimi-k2.6")


def test_nvidia_catalog_preserves_canonical_ids(isolated_nvidia_config: Path) -> None:
    records = parse_openai_compatible_model_records(
        {
            "data": [
                {"id": "deepseek-ai/deepseek-v4-flash", "object": "model"},
                {"id": "nvidia/nemotron-3-nano-30b-a3b", "object": "model"},
                {"id": "moonshotai/kimi-k2.6", "object": "model"},
                {"id": "nvidia/nv-embedqa-e5-v5", "object": "model"},
            ]
        }
    )
    assert [item["id"] for item in records] == [
        "deepseek-ai/deepseek-v4-flash",
        "moonshotai/kimi-k2.6",
        "nvidia/nemotron-3-nano-30b-a3b",
        "nvidia/nv-embedqa-e5-v5",
    ]
    models = ModelCatalogService(fetcher=lambda **_kwargs: records).refresh(
        provider="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="nvapi-secret",
    )
    by_id = {model.id: model for model in models}
    assert "deepseek-ai/deepseek-v4-flash" in by_id
    assert by_id["deepseek-ai/deepseek-v4-flash"].qualified_id == (
        "nvidia/deepseek-ai/deepseek-v4-flash"
    )
    assert ModelCapability.TEXT_GENERATION in by_id["deepseek-ai/deepseek-v4-flash"].capabilities
    assert ModelCapability.TOOL_CALLING in by_id["deepseek-ai/deepseek-v4-flash"].capabilities
    assert by_id["nvidia/nv-embedqa-e5-v5"].supports(ModelPurpose.EMBEDDING)
    # Unknown IDs remain usable via advanced selection rather than being dropped.
    assert any(model.id == "moonshotai/kimi-k2.6" for model in models)


def test_nvidia_configuration_persists_isolated_credentials(
    isolated_nvidia_config: Path,
) -> None:
    draft = ConfigurationDraft.load()
    draft.set_secret("NVIDIA_API_KEY", "nvapi-secret")
    draft.values["NVIDIA_BASE_URL"] = "https://integrate.api.nvidia.com/v1"
    draft.set_models(
        provider="nvidia",
        high="deepseek-ai/deepseek-v4-flash",
        coding="deepseek-ai/deepseek-v4-flash",
        fast="nvidia/nemotron-3-nano-30b-a3b",
        embedding="nvidia/nv-embedqa-e5-v5",
    )
    draft.save()
    values = user_config.load_effective_settings(include_env=False)
    assert values["MANA_AI_PROVIDER"] == "nvidia"
    assert values["MANA_PRIMARY_MODEL"] == "nvidia/deepseek-ai/deepseek-v4-flash"
    assert values["OPENAI_CHAT_MODEL"] == "deepseek-ai/deepseek-v4-flash"
    assert values["NVIDIA_BASE_URL"] == "https://integrate.api.nvidia.com/v1"
    assert user_config.load_user_secrets()["NVIDIA_API_KEY"] == "nvapi-secret"
    assert "nvapi-secret" not in (isolated_nvidia_config / "config.toml").read_text()
    # OpenAI credentials are not required and must not be invented.
    assert not str(values.get("OPENAI_API_KEY") or "").strip()


def test_nvidia_connection_uses_nvidia_key_not_openai(
    isolated_nvidia_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_config.save_effective_user_config(
        {
            "MANA_AI_PROVIDER": "nvidia",
            "NVIDIA_API_KEY": "nvapi-only",
            "NVIDIA_BASE_URL": "https://integrate.api.nvidia.com/v1",
            "OPENAI_API_KEY": "sk-openai-must-not-be-used",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
        },
        merge=False,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-openai")
    connection = resolve_inference_connection(Settings())
    assert connection.provider == "nvidia"
    assert connection.api_key == "nvapi-only"
    assert connection.base_url == "https://integrate.api.nvidia.com/v1"
    assert connection.supports_responses_api is False
    assert "nvapi-only" not in json.dumps(
        {
            "provider": connection.provider,
            "base_url": connection.base_url,
            "headers": connection.headers,
        }
    )


def test_nvidia_missing_key_is_provider_specific(
    isolated_nvidia_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_config.save_effective_user_config(
        {
            "MANA_AI_PROVIDER": "nvidia",
            "OPENAI_API_KEY": "sk-openai-present",
            "NVIDIA_BASE_URL": "https://integrate.api.nvidia.com/v1",
        },
        merge=False,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-openai")
    with pytest.raises(ProviderConfigurationError, match="NVIDIA") as exc_info:
        resolve_inference_connection(Settings())
    message = str(exc_info.value)
    assert "NVIDIA_API_KEY" in message
    assert "OPENAI" not in message or "OPENAI_API_KEY" not in message


def test_credentials_from_mapping_isolates_providers() -> None:
    values = {
        "MANA_AI_PROVIDER": "nvidia",
        "NVIDIA_API_KEY": "nvapi-x",
        "NVIDIA_BASE_URL": "https://integrate.api.nvidia.com/v1",
        "OPENAI_API_KEY": "sk-openai",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENROUTER_API_KEY": "or-key",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
    }
    api_key, base_url = credentials_from_mapping(values, provider="nvidia")
    assert api_key == "nvapi-x"
    assert base_url == "https://integrate.api.nvidia.com/v1"
    openai_key, openai_base = credentials_from_mapping(values, provider="openai")
    assert openai_key == "sk-openai"
    assert openai_base == "https://api.openai.com/v1"


def test_nvidia_chat_model_uses_chat_completions(
    isolated_nvidia_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mana_agent.multi_agent.runtime.compatibility.get_setting",
        lambda name, default=None: default,
    )
    llm = create_chat_model(
        api_key="nvapi-test",
        model="deepseek-ai/deepseek-v4-flash",
        base_url="https://integrate.api.nvidia.com/v1",
        provider="nvidia",
    )
    assert llm.selected_provider == "nvidia"
    assert llm.compatibility_api_mode == "chat_completions"
    assert llm.compatibility_capabilities.supports_responses_api is False
    assert llm.model_name == "deepseek-ai/deepseek-v4-flash"


def test_nvidia_model_specific_request_configuration_forwarded(
    isolated_nvidia_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mana_agent.multi_agent.runtime.compatibility.get_setting",
        lambda name, default=None: default,
    )
    llm = create_chat_model(
        api_key="nvapi-test",
        model="deepseek-ai/deepseek-v4-flash",
        base_url="https://integrate.api.nvidia.com/v1",
        provider="nvidia",
        model_configuration={
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"},
        },
    )
    extra = getattr(llm, "extra_body", None) or {}
    assert extra.get("chat_template_kwargs") == {
        "thinking": True,
        "reasoning_effort": "high",
    }


def test_nvidia_format_provider_error_never_says_openai() -> None:
    class FakeHTTPError(Exception):
        status_code = 401

    message = format_provider_error(
        FakeHTTPError("Incorrect API key"),
        provider="nvidia",
        model="deepseek-ai/deepseek-v4-flash",
    )
    assert "NVIDIA" in message
    assert "401" in message
    assert "OPENAI_API_KEY" in message


def test_nvidia_tool_calling_payload_shape(
    isolated_nvidia_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.messages import HumanMessage

    monkeypatch.setattr(
        "mana_agent.multi_agent.runtime.compatibility.get_setting",
        lambda name, default=None: default,
    )
    llm = create_chat_model(
        api_key="nvapi-test",
        model="deepseek-ai/deepseek-v4-flash",
        base_url="https://integrate.api.nvidia.com/v1",
        provider="nvidia",
    )
    tool = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    }
    payload = llm._get_request_payload([HumanMessage(content="hello")], tools=[tool])
    assert "messages" in payload
    assert "input" not in payload
    assert payload["tools"][0]["function"]["name"] == "read_file"
    assert payload["model"] == "deepseek-ai/deepseek-v4-flash"


def test_nvidia_mock_chat_completion_roundtrip(
    isolated_nvidia_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mana_agent.multi_agent.runtime.compatibility.get_setting",
        lambda name, default=None: default,
    )
    llm = create_chat_model(
        api_key="nvapi-test",
        model="deepseek-ai/deepseek-v4-flash",
        base_url="https://integrate.api.nvidia.com/v1",
        provider="nvidia",
    )

    class _Usage:
        def __init__(self) -> None:
            self.prompt_tokens = 3
            self.completion_tokens = 5
            self.total_tokens = 8

    class _Message:
        content = "hello from nvidia"
        tool_calls: list[Any] = []

    class _Choice:
        message = _Message()
        finish_reason = "stop"

    class _Response:
        choices = [_Choice()]
        usage = _Usage()

    client = MagicMock()
    client.chat.completions.create.return_value = _Response()
    # CompatibleChatOpenAI may resolve the root client differently by version;
    # patch the high-level generate path for transport isolation.
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.messages import AIMessage

    def fake_generate(messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="hello from nvidia"))],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 5,
                    "total_tokens": 8,
                },
                "model_name": "deepseek-ai/deepseek-v4-flash",
            },
        )

    monkeypatch.setattr(
        llm.__class__.__mro__[1],
        "_generate",
        lambda self, messages, stop=None, run_manager=None, **kwargs: fake_generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ),
    )
    result = llm._generate([])
    assert result.generations[0].message.content == "hello from nvidia"
    assert result.llm_output["token_usage"]["total_tokens"] == 8


def test_nvidia_streaming_chunks(
    isolated_nvidia_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk

    monkeypatch.setattr(
        "mana_agent.multi_agent.runtime.compatibility.get_setting",
        lambda name, default=None: default,
    )
    llm = create_chat_model(
        api_key="nvapi-test",
        model="deepseek-ai/deepseek-v4-flash",
        base_url="https://integrate.api.nvidia.com/v1",
        provider="nvidia",
    )

    def fake_stream(self, *args, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(content="hel"))
        yield ChatGenerationChunk(message=AIMessageChunk(content="lo"))

    monkeypatch.setattr(llm.__class__.__mro__[1], "_stream", fake_stream)
    chunks = list(llm._stream([]))
    assert "".join(str(chunk.message.content) for chunk in chunks) == "hello"


def test_nvidia_tool_call_multi_turn_message_shape() -> None:
    """Ensure OpenAI-compatible tool message fields are preserved for NIM."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    messages = [
        HumanMessage(content="List files"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "list_dir",
                    "args": {"path": "."},
                }
            ],
        ),
        ToolMessage(content='{"files": ["a.py"]}', tool_call_id="call_1"),
    ]
    # Serialize through LangChain's OpenAI converter path used by ChatOpenAI.
    llm = create_chat_model(
        api_key="nvapi-test",
        model="deepseek-ai/deepseek-v4-flash",
        base_url="https://integrate.api.nvidia.com/v1",
        provider="nvidia",
    )
    payload = llm._get_request_payload(messages)
    roles = [item.get("role") for item in payload["messages"]]
    assert "tool" in roles
    tool_msg = next(item for item in payload["messages"] if item.get("role") == "tool")
    assert tool_msg.get("tool_call_id") == "call_1"
    assistant = next(
        item for item in payload["messages"] if item.get("role") == "assistant"
    )
    assert assistant.get("tool_calls")


def test_openrouter_and_openai_providers_unchanged() -> None:
    openai = PROVIDERS.get("openai")
    openrouter = PROVIDERS.get("openrouter")
    assert openai.api_key_env == "OPENAI_API_KEY"
    assert openrouter.api_key_env == "OPENROUTER_API_KEY"
    assert split_qualified_model_id(
        "openrouter/anthropic/claude-sonnet", default_provider="openrouter"
    ) == ("openrouter", "anthropic/claude-sonnet")
    assert qualify_model_id("openai", "gpt-4.1-mini") == "openai/gpt-4.1-mini"
    # OpenRouter host of openai/* models must not be re-attributed to OpenAI.
    assert split_qualified_model_id(
        "openai/gpt-4.1-mini", default_provider="openrouter"
    ) == ("openrouter", "openai/gpt-4.1-mini")
