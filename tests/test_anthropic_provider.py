"""Unit tests for the Anthropic (Claude) inference provider integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from mana_agent.config import user_config
from mana_agent.config.inference_provider import (
    ProviderConfigurationError,
    credentials_from_mapping,
    resolve_inference_connection,
)
from mana_agent.config.model_catalog import ModelCapability, normalize_capabilities
from mana_agent.config.provider_registry import (
    PROVIDERS,
    CodexTransport,
    provider_credential_env_names,
    qualify_model_id,
    split_qualified_model_id,
)
from mana_agent.config.session import ConfigurationDraft
from mana_agent.config.settings import Settings
from mana_agent.integrations.provider_failure import (
    ProviderFailureKind,
    classify_http_status,
)
from mana_agent.multi_agent.runtime.anthropic_provider import (
    CompatibleChatAnthropic,
    DEFAULT_ANTHROPIC_MODEL,
    _convert_to_anthropic_tool,
    _format_anthropic_tool_choice,
    _map_messages_for_anthropic,
)
from mana_agent.multi_agent.runtime.compatibility import (
    create_chat_model,
    format_provider_error,
)


@pytest.fixture()
def isolated_anthropic_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_dir = tmp_path / ".mana"
    monkeypatch.setenv("MANA_HOME", str(config_dir))
    for key in (
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "NVIDIA_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_BASE_URL",
        "OPENROUTER_BASE_URL",
        "NVIDIA_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "MANA_AI_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)
    return config_dir


def test_anthropic_is_first_class_provider() -> None:
    provider = PROVIDERS.get("anthropic")
    assert provider.display_name == "Anthropic"
    assert provider.default_base_url == "https://api.anthropic.com"
    assert provider.api_key_env == "ANTHROPIC_API_KEY"
    assert provider.supports_responses_api is False
    assert provider.supports_decisions_api is False
    assert provider.codex_transport is CodexTransport.UNSUPPORTED
    assert provider_credential_env_names("anthropic") == (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
    )


def test_anthropic_qualified_ids() -> None:
    assert qualify_model_id("anthropic", "claude-sonnet-5-5") == "anthropic/claude-sonnet-5-5"
    assert split_qualified_model_id(
        "anthropic/claude-sonnet-5-5", default_provider="anthropic"
    ) == ("anthropic", "claude-sonnet-5-5")
    assert split_qualified_model_id(
        "claude-sonnet-5-5", default_provider="anthropic"
    ) == ("anthropic", "claude-sonnet-5-5")
    assert split_qualified_model_id(
        "claude-opus-5-5", default_provider="anthropic"
    ) == ("anthropic", "claude-opus-5-5")
    assert split_qualified_model_id(
        "claude-haiku-5-5", default_provider="anthropic"
    ) == ("anthropic", "claude-haiku-5-5")


def test_anthropic_model_catalog_capabilities() -> None:
    for model_id in ("claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"):
        caps = normalize_capabilities("anthropic", model_id)
        assert ModelCapability.TEXT_GENERATION in caps
        assert ModelCapability.TOOL_CALLING in caps
        assert ModelCapability.REASONING in caps
        assert ModelCapability.CODE in caps
        assert ModelCapability.IMAGE_INPUT in caps


def test_anthropic_configuration_persists_isolated_credentials(
    isolated_anthropic_config: Path,
) -> None:
    draft = ConfigurationDraft.load()
    draft.set_secret("ANTHROPIC_API_KEY", "ant-secret-test-key")
    draft.values["ANTHROPIC_BASE_URL"] = "https://api.anthropic.com"
    draft.values["MANA_AI_PROVIDER"] = "anthropic"
    draft.save()

    values = user_config.load_effective_settings(include_env=False)
    assert values["MANA_AI_PROVIDER"] == "anthropic"
    assert user_config.load_user_secrets()["ANTHROPIC_API_KEY"] == "ant-secret-test-key"
    assert "ant-secret-test-key" not in (
        isolated_anthropic_config / "config.toml"
    ).read_text()


def test_anthropic_connection_uses_anthropic_key_not_openai(
    isolated_anthropic_config: Path,
) -> None:
    user_config.save_effective_user_config(
        {
            "MANA_AI_PROVIDER": "anthropic",
            "OPENAI_API_KEY": "sk-should-not-be-used",
            "ANTHROPIC_API_KEY": "ant-valid-key",
        },
        merge=False,
    )
    connection = resolve_inference_connection(Settings())
    assert connection.provider == "anthropic"
    assert connection.api_key == "ant-valid-key"
    assert connection.base_url == "https://api.anthropic.com"
    assert "sk-should-not-be-used" not in json.dumps(
        {"key": connection.api_key, "provider": connection.provider}
    )


def test_anthropic_missing_key_is_provider_specific(
    isolated_anthropic_config: Path,
) -> None:
    user_config.save_effective_user_config(
        {
            "MANA_AI_PROVIDER": "anthropic",
            "OPENAI_API_KEY": "sk-irrelevant",
        },
        merge=False,
    )
    with pytest.raises(ProviderConfigurationError) as exc_info:
        resolve_inference_connection(Settings(), provider="anthropic")
    message = str(exc_info.value)
    assert "Anthropic" in message
    assert "ANTHROPIC_API_KEY" in message
    assert "OPENAI_API_KEY" not in message


def test_credentials_from_mapping_isolates_anthropic() -> None:
    mapping = {
        "MANA_AI_PROVIDER": "anthropic",
        "OPENAI_API_KEY": "sk-openai",
        "ANTHROPIC_API_KEY": "ant-key",
        "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
    }
    key, base = credentials_from_mapping(mapping, provider="anthropic")
    assert key == "ant-key"
    assert base == "https://api.anthropic.com"


def test_create_chat_model_instantiates_compatible_chat_anthropic() -> None:
    llm = create_chat_model(
        api_key="ant-key-123",
        model="claude-sonnet-5-5",
        provider="anthropic",
        base_url="https://api.anthropic.com",
    )
    assert isinstance(llm, CompatibleChatAnthropic)
    assert llm.model_name == "claude-sonnet-5-5"
    assert llm.selected_provider == "anthropic"
    assert llm.max_tokens == 4096


def test_anthropic_mock_basic_response() -> None:
    """Verify chat generation with mocked Anthropic SDK client."""
    mock_client = MagicMock()

    class _MockTextBlock:
        type = "text"
        text = "Hello from Claude Sonnet"

    class _MockUsage:
        input_tokens = 12
        output_tokens = 24

    class _MockResponse:
        content = [_MockTextBlock()]
        stop_reason = "end_turn"
        usage = _MockUsage()

    mock_client.messages.create.return_value = _MockResponse()

    llm = CompatibleChatAnthropic(
        anthropic_api_key="test-key",
        model_name="claude-sonnet-5-5",
        client=mock_client,
    )

    result = llm._generate([HumanMessage(content="Hello!")])
    assert isinstance(result, ChatResult)
    assert result.generations[0].message.content == "Hello from Claude Sonnet"
    assert result.llm_output["token_usage"]["prompt_tokens"] == 12
    assert result.llm_output["token_usage"]["completion_tokens"] == 24
    assert result.llm_output["token_usage"]["total_tokens"] == 36

    # Verify call parameters passed to mock client
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-5-5"
    assert call_kwargs["max_tokens"] == 4096
    assert call_kwargs["messages"] == [{"role": "user", "content": "Hello!"}]


def test_anthropic_streaming_chunks() -> None:
    """Verify streaming chunk assembly with mocked Anthropic events."""
    mock_client = MagicMock()

    class _MockMsgStartEvent:
        type = "message_start"

        class message:
            class usage:
                input_tokens = 8

    class _MockDeltaEvent1:
        type = "content_block_delta"

        class delta:
            type = "text_delta"
            text = "Hello "

    class _MockDeltaEvent2:
        type = "content_block_delta"

        class delta:
            type = "text_delta"
            text = "world!"

    class _MockMsgDeltaEvent:
        type = "message_delta"

        class usage:
            output_tokens = 6

    mock_client.messages.create.return_value = [
        _MockMsgStartEvent(),
        _MockDeltaEvent1(),
        _MockDeltaEvent2(),
        _MockMsgDeltaEvent(),
    ]

    llm = CompatibleChatAnthropic(
        anthropic_api_key="test-key",
        model_name="claude-sonnet-5-5",
        client=mock_client,
    )

    chunks = list(llm._stream([HumanMessage(content="Hi")]))
    text_chunks = [c.message.content for c in chunks if c.message.content]
    assert "".join(text_chunks) == "Hello world!"

    # Verify usage chunk is emitted
    usage_chunk = next(
        (c for c in chunks if c.message.usage_metadata), None
    )
    assert usage_chunk is not None
    assert usage_chunk.message.usage_metadata["input_tokens"] == 8
    assert usage_chunk.message.usage_metadata["output_tokens"] == 6
    assert usage_chunk.message.usage_metadata["total_tokens"] == 14


def test_anthropic_message_format_mapping_contract() -> None:
    """Verify system prompt isolation, message alternation, and tool blocks."""
    messages = [
        SystemMessage(content="You are a principal engineer."),
        SystemMessage(content="Follow TDD."),
        HumanMessage(content="Fix the bug."),
        AIMessage(
            content="I will inspect files.",
            tool_calls=[
                {
                    "id": "toolu_01",
                    "name": "read_file",
                    "args": {"path": "src/main.py"},
                }
            ],
        ),
        ToolMessage(content='{"lines": 42}', tool_call_id="toolu_01"),
        ToolMessage(content='{"status": "ok"}', tool_call_id="toolu_02"),
    ]

    system, mapped = _map_messages_for_anthropic(messages)

    # 1. System prompt is top-level string combining system messages
    assert system == "You are a principal engineer.\n\nFollow TDD."

    # 2. Alternating user/assistant turns
    roles = [m["role"] for m in mapped]
    assert roles == ["user", "assistant", "user"]

    # 3. First message is user
    assert mapped[0]["role"] == "user"
    assert mapped[0]["content"] == "Fix the bug."

    # 4. Assistant message contains text and tool_use blocks
    assistant_blocks = mapped[1]["content"]
    assert any(b.get("type") == "text" for b in assistant_blocks)
    tool_use_block = next(b for b in assistant_blocks if b.get("type") == "tool_use")
    assert tool_use_block["id"] == "toolu_01"
    assert tool_use_block["name"] == "read_file"
    assert tool_use_block["input"] == {"path": "src/main.py"}

    # 5. Consecutive ToolMessages merged into a single user message with tool_result blocks
    user_tool_blocks = mapped[2]["content"]
    assert len(user_tool_blocks) == 2
    assert user_tool_blocks[0]["type"] == "tool_result"
    assert user_tool_blocks[0]["tool_use_id"] == "toolu_01"
    assert user_tool_blocks[0]["content"] == '{"lines": 42}'
    assert user_tool_blocks[1]["type"] == "tool_result"
    assert user_tool_blocks[1]["tool_use_id"] == "toolu_02"
    assert user_tool_blocks[1]["content"] == '{"status": "ok"}'


def test_anthropic_first_message_assistant_prepends_user() -> None:
    """Anthropic rejects conversations starting with assistant; prepend user."""
    messages = [AIMessage(content="Hello there!")]
    system, mapped = _map_messages_for_anthropic(messages)
    assert mapped[0]["role"] == "user"
    assert mapped[1]["role"] == "assistant"
    assert mapped[1]["content"] == "Hello there!"


def test_anthropic_tool_binding_and_response_roundtrip() -> None:
    """Verify tool binding and parsing of assistant tool_use response."""
    tool_spec = {
        "name": "lookup_symbol",
        "description": "Find symbol in codebase",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }

    mock_client = MagicMock()

    class _MockToolUseBlock:
        type = "tool_use"
        id = "toolu_999"
        name = "lookup_symbol"
        input = {"query": "CompatibleChatAnthropic"}

    class _MockUsage:
        input_tokens = 15
        output_tokens = 10

    class _MockResponse:
        content = [_MockToolUseBlock()]
        stop_reason = "tool_use"
        usage = _MockUsage()

    mock_client.messages.create.return_value = _MockResponse()

    llm = CompatibleChatAnthropic(
        anthropic_api_key="test-key",
        model_name="claude-sonnet-5-5",
        client=mock_client,
    )
    bound_llm = llm.bind_tools([tool_spec], tool_choice="required")
    assert bound_llm.tool_choice == {"type": "any"}

    result = bound_llm._generate([HumanMessage(content="Find CompatibleChatAnthropic")])
    ai_msg = result.generations[0].message
    assert len(ai_msg.tool_calls) == 1
    assert ai_msg.tool_calls[0]["id"] == "toolu_999"
    assert ai_msg.tool_calls[0]["name"] == "lookup_symbol"
    assert ai_msg.tool_calls[0]["args"] == {"query": "CompatibleChatAnthropic"}

    call_kwargs = mock_client.messages.create.call_args[1]
    assert "tools" in call_kwargs
    assert call_kwargs["tools"][0]["name"] == "lookup_symbol"
    assert "input_schema" in call_kwargs["tools"][0]
    assert call_kwargs["tool_choice"] == {"type": "any"}


def test_anthropic_error_normalization_and_formatting() -> None:
    """Verify error status mapping and user-facing message formatting."""
    import anthropic

    # 401 Authentication
    auth_err = anthropic.AuthenticationError(
        message="Invalid API Key",
        response=MagicMock(status_code=401),
        body={"error": {"message": "Invalid API Key"}},
    )
    safe_auth = format_provider_error(
        auth_err, provider="anthropic", model="claude-sonnet-5-5"
    )
    assert "Anthropic authentication or permission failed (HTTP 401)" in safe_auth
    assert "ANTHROPIC_API_KEY" in safe_auth or "Anthropic credentials" in safe_auth
    assert "OpenAI" not in safe_auth
    assert classify_http_status(401, provider="anthropic").kind is ProviderFailureKind.AUTHENTICATION

    # 429 Rate Limit
    rate_err = anthropic.RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429),
        body={"error": {"message": "Rate limit exceeded"}},
    )
    safe_rate = format_provider_error(
        rate_err, provider="anthropic", model="claude-sonnet-5-5"
    )
    assert "Anthropic rate limit or quota exceeded (HTTP 429)" in safe_rate
    assert classify_http_status(429, provider="anthropic").kind is ProviderFailureKind.RATE_LIMITED

    # 529 Overloaded
    overloaded_err = anthropic.APIStatusError(
        message="Anthropic is overloaded",
        response=MagicMock(status_code=529),
        body={"error": {"message": "Overloaded"}},
    )
    safe_overloaded = format_provider_error(
        overloaded_err, provider="anthropic", model="claude-sonnet-5-5"
    )
    assert "overloaded (HTTP 529)" in safe_overloaded
    assert (
        classify_http_status(529, provider="anthropic").kind
        is ProviderFailureKind.PROVIDER_OVERLOADED
    )

    # 400 Invalid Request
    bad_req = anthropic.BadRequestError(
        message="max_tokens is required",
        response=MagicMock(status_code=400),
        body={"error": {"message": "max_tokens is required"}},
    )
    safe_bad_req = format_provider_error(
        bad_req, provider="anthropic", model="claude-sonnet-5-5"
    )
    assert "Anthropic request failed (HTTP 400)" in safe_bad_req
    assert classify_http_status(400, provider="anthropic").kind is ProviderFailureKind.INVALID_REQUEST


def test_other_providers_unchanged() -> None:
    openai = PROVIDERS.get("openai")
    openrouter = PROVIDERS.get("openrouter")
    nvidia = PROVIDERS.get("nvidia")
    assert openai.api_key_env == "OPENAI_API_KEY"
    assert openrouter.api_key_env == "OPENROUTER_API_KEY"
    assert nvidia.api_key_env == "NVIDIA_API_KEY"
    assert qualify_model_id("openai", "gpt-4.1-mini") == "openai/gpt-4.1-mini"
    assert qualify_model_id("nvidia", "deepseek-ai/v4") == "nvidia/deepseek-ai/v4"
