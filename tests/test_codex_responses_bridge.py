from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mana_agent.config.provider_registry import PROVIDERS, CodexTransport
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.exceptions import CodexConfigurationError
from mana_agent.integrations.codex.responses_bridge.lifecycle import ResponsesBridgeManager
from mana_agent.integrations.codex.responses_bridge.models import BridgeUpstreamConfig
from mana_agent.integrations.codex.responses_bridge.request_adapter import (
    convert_responses_request_to_chat,
    normalize_reasoning_effort,
)
from mana_agent.integrations.codex.responses_bridge.response_adapter import (
    convert_chat_completion_to_response,
)
from mana_agent.integrations.codex.responses_bridge.stream_adapter import (
    ChatToResponsesStreamAdapter,
)
from mana_agent.integrations.codex.runtime_config import CodexRuntimeConfigBuilder
from mana_agent.integrations.codex.runtime_environment import CodexRuntimeEnvironment


def test_transport_selection_openai_direct_nvidia_bridge() -> None:
    assert PROVIDERS.get("openai").codex_transport is CodexTransport.DIRECT_RESPONSES
    assert PROVIDERS.get("openai").supports_responses_api is True
    assert PROVIDERS.get("nvidia").codex_transport is CodexTransport.RESPONSES_BRIDGE
    assert PROVIDERS.get("nvidia").supports_responses_api is False
    assert PROVIDERS.get("nvidia").api_key_env == "NVIDIA_API_KEY"


def test_deepseek_v4_pro_reasoning_effort_mapping() -> None:
    assert (
        normalize_reasoning_effort(
            provider="nvidia", model="deepseek-ai/deepseek-v4-pro", effort="xhigh"
        )
        == "max"
    )
    assert (
        normalize_reasoning_effort(
            provider="nvidia", model="deepseek-ai/deepseek-v4-pro", effort="high"
        )
        == "high"
    )
    assert (
        normalize_reasoning_effort(
            provider="nvidia", model="deepseek-ai/deepseek-v4-pro", effort="low"
        )
        == "none"
    )
    # Codex xhigh must land in NIM chat_template_kwargs as max.
    chat = convert_responses_request_to_chat(
        {
            "model": "deepseek-ai/deepseek-v4-pro",
            "input": "hello",
            "reasoning": {"effort": "xhigh"},
        },
        upstream=BridgeUpstreamConfig(
            provider="nvidia",
            display_name="NVIDIA",
            api_key="nvapi",
            base_url="https://integrate.api.nvidia.com/v1",
            model="deepseek-ai/deepseek-v4-pro",
        ),
    )
    assert chat["chat_template_kwargs"]["reasoning_effort"] == "max"
    assert chat["chat_template_kwargs"]["thinking"] is True


def test_responses_to_chat_tools_and_function_outputs() -> None:
    upstream = BridgeUpstreamConfig(
        provider="nvidia",
        display_name="NVIDIA",
        api_key="nvapi-secret",
        base_url="https://integrate.api.nvidia.com/v1",
        model="deepseek-ai/deepseek-v4-pro",
    )
    body = {
        "model": "deepseek-ai/deepseek-v4-pro",
        "instructions": "You are a coding agent.",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list files"}]},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "shell",
                "arguments": '{"command":"ls"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "a.py\nb.py",
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "shell",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
            }
        ],
        "tool_choice": "auto",
        "reasoning": {"effort": "high"},
        "stream": False,
    }
    chat = convert_responses_request_to_chat(body, upstream=upstream)
    assert chat["model"] == "deepseek-ai/deepseek-v4-pro"
    # DeepSeek on NVIDIA uses chat_template_kwargs, not bare reasoning_effort.
    # Tools force thinking off so the model emits structured tool_calls rather
    # than free-form DSML/invoke pseudo-tool text (SWE-bench empty_patch).
    assert "reasoning_effort" not in chat
    assert chat["chat_template_kwargs"]["reasoning_effort"] == "none"
    assert chat["chat_template_kwargs"]["thinking"] is False
    assert chat["tools"][0]["function"]["name"] == "shell"
    roles = [message["role"] for message in chat["messages"]]
    assert roles[0] == "system"
    assert "You are a coding agent." in chat["messages"][0]["content"]
    assert "tool" in roles
    tool_message = next(message for message in chat["messages"] if message["role"] == "tool")
    assert tool_message["tool_call_id"] == "call_1"
    assistant = next(message for message in chat["messages"] if message.get("tool_calls"))
    assert assistant["tool_calls"][0]["function"]["name"] == "shell"


def test_chat_completion_to_responses_with_tool_calls() -> None:
    chat = {
        "id": "chatcmpl-1",
        "model": "deepseek-ai/deepseek-v4-pro",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": '{"command":"pytest"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }
    response = convert_chat_completion_to_response(chat, model="deepseek-ai/deepseek-v4-pro")
    assert response["status"] == "completed"
    assert response["usage"]["total_tokens"] == 14
    assert any(item["type"] == "function_call" for item in response["output"])
    call = next(item for item in response["output"] if item["type"] == "function_call")
    assert call["call_id"] == "call_abc"
    assert call["name"] == "shell"


def test_stream_adapter_fragmented_tool_arguments() -> None:
    adapter = ChatToResponsesStreamAdapter(model="deepseek-ai/deepseek-v4-pro")
    events = list(adapter.open_events())
    events.extend(
        adapter.ingest_chat_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "read_file", "arguments": '{"path":'},
                                }
                            ]
                        }
                    }
                ]
            }
        )
    )
    events.extend(
        adapter.ingest_chat_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '"/tmp/file"}'},
                                }
                            ]
                        }
                    }
                ]
            }
        )
    )
    events.extend(adapter.close_events())
    joined = "".join(events)
    assert "response.function_call_arguments.delta" in joined
    assert "response.function_call_arguments.done" in joined
    # SSE payloads are JSON-encoded, so quotes appear escaped in the raw stream.
    # The reconstructed arguments must still be the complete JSON object.
    assert adapter.tool_calls[0].arguments == '{"path":"/tmp/file"}'
    assert '\\"path\\":\\"/tmp/file\\"' in joined or '{"path":"/tmp/file"}' in joined
    # Partial fragments must appear as deltas, not rejected as incomplete JSON.
    assert "/tmp/file" in joined


def test_stream_adapter_text_deltas() -> None:
    adapter = ChatToResponsesStreamAdapter(model="deepseek-ai/deepseek-v4-pro")
    events = list(adapter.open_events())
    events.extend(adapter.ingest_chat_chunk({"choices": [{"delta": {"content": "hel"}}]}))
    events.extend(adapter.ingest_chat_chunk({"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]}))
    events.extend(adapter.close_events())
    joined = "".join(events)
    assert "response.created" in joined
    assert "response.output_text.delta" in joined
    assert "response.completed" in joined
    assert "hello" in joined


def test_nvidia_runtime_uses_bridge_and_never_exposes_upstream_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    manager = ResponsesBridgeManager()
    settings = CodexSettings(
        enabled=True,
        provider="nvidia",
        provider_display_name="NVIDIA",
        api_key="nvapi-super-secret-key",
        base_url="https://integrate.api.nvidia.com/v1",
        model="deepseek-ai/deepseek-v4-pro",
        supports_responses_api=False,
        codex_transport=CodexTransport.RESPONSES_BRIDGE,
        model_request_overrides={"reasoning_effort": "high"},
    )
    runtime = CodexRuntimeConfigBuilder.build(
        settings, sandbox_mode="workspace-write", bridge_manager=manager
    )
    try:
        assert runtime.transport is CodexTransport.RESPONSES_BRIDGE
        assert runtime.bridge is not None
        assert runtime.base_url.startswith("http://127.0.0.1:")
        assert runtime.api_key != "nvapi-super-secret-key"
        assert "nvapi-super-secret-key" not in runtime.to_toml()
        assert "nvapi-super-secret-key" not in repr(runtime)
        assert runtime.accounting_provider == "nvidia"
        assert runtime.accounting_model == "deepseek-ai/deepseek-v4-pro"
        assert runtime.bridge.healthcheck()["ok"] is True

        env = CodexRuntimeEnvironment.create(runtime)
        try:
            rendered = (env.home / "config.toml").read_text(encoding="utf-8")
            assert "wire_api = \"responses\"" in rendered
            assert "nvapi-super-secret-key" not in rendered
            assert env.environment["MANA_CODEX_API_KEY"] == runtime.api_key
            assert "NVIDIA_API_KEY" not in env.environment
            assert "nvapi-super-secret-key" not in json.dumps(
                {k: v for k, v in env.environment.items() if k != "MANA_CODEX_API_KEY"}
            )
            # Child command args are just the executable path — no secrets.
            assert "nvapi-super-secret-key" not in str(("codex", "app-server"))
        finally:
            env.close()
    finally:
        if runtime.bridge is not None and not runtime.bridge._released:
            runtime.bridge.release()
        manager.shutdown_all()


def test_bridge_multi_tool_calls_in_one_turn() -> None:
    chat = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"a"}'},
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"b"}'},
                        },
                    ]
                }
            }
        ]
    }
    response = convert_chat_completion_to_response(chat, model="m")
    calls = [item for item in response["output"] if item["type"] == "function_call"]
    assert len(calls) == 2
    assert {item["call_id"] for item in calls} == {"call_a", "call_b"}


def test_direct_responses_still_rejects_chat_completions_url() -> None:
    with pytest.raises(CodexConfigurationError):
        CodexRuntimeConfigBuilder.build(
            CodexSettings(
                enabled=True,
                provider="openai",
                provider_display_name="OpenAI",
                api_key="sk-test",
                base_url="https://api.openai.com/v1/chat/completions",
                model="gpt-4.1-mini",
                supports_responses_api=True,
                codex_transport=CodexTransport.DIRECT_RESPONSES,
            ),
            sandbox_mode="read-only",
        )


def test_deepseek_v4_pro_outgoing_payload_includes_reasoning_effort() -> None:
    upstream = BridgeUpstreamConfig(
        provider="nvidia",
        display_name="NVIDIA",
        api_key="nvapi",
        base_url="https://integrate.api.nvidia.com/v1",
        model="deepseek-ai/deepseek-v4-pro",
    )
    chat = convert_responses_request_to_chat(
        {
            "model": "deepseek-ai/deepseek-v4-pro",
            "input": "hello",
            "reasoning": {"effort": "high"},
        },
        upstream=upstream,
    )
    assert chat["model"] == "deepseek-ai/deepseek-v4-pro"
    assert chat["stream"] is False
    # NVIDIA DeepSeek requires chat_template_kwargs, not bare reasoning_effort.
    assert "reasoning_effort" not in chat
    assert chat["chat_template_kwargs"] == {
        "thinking": True,
        "reasoning_effort": "high",
    }
    assert chat["messages"][0]["role"] == "user"
    assert chat["messages"][0]["content"] == "hello"


def test_deepseek_v4_flash_default_chat_template_kwargs() -> None:
    upstream = BridgeUpstreamConfig(
        provider="nvidia",
        display_name="NVIDIA",
        api_key="nvapi",
        base_url="https://integrate.api.nvidia.com/v1",
        model="deepseek-ai/deepseek-v4-flash",
    )
    chat = convert_responses_request_to_chat(
        {"model": "deepseek-ai/deepseek-v4-flash", "input": "hi"},
        upstream=upstream,
    )
    assert chat["chat_template_kwargs"]["thinking"] is True
    assert chat["chat_template_kwargs"]["reasoning_effort"] == "high"


def test_deepseek_tools_force_thinking_off() -> None:
    """Regression: tools + thinking produced empty SWE-bench patches.

    With thinking enabled, DeepSeek V4 on NVIDIA often emits free-form
    pseudo-tool text instead of structured tool_calls. Codex then completes
    with zero worktree changes (empty_patch / status=ok).
    """
    upstream = BridgeUpstreamConfig(
        provider="nvidia",
        display_name="NVIDIA",
        api_key="nvapi",
        base_url="https://integrate.api.nvidia.com/v1",
        model="deepseek-ai/deepseek-v4-flash-0731",
    )
    chat = convert_responses_request_to_chat(
        {
            "model": "deepseek-ai/deepseek-v4-flash-0731",
            "input": "fix the bug",
            "tools": [
                {
                    "type": "function",
                    "name": "shell",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                }
            ],
            "reasoning": {"effort": "high"},
            "stream": True,
        },
        upstream=upstream,
    )
    assert chat["tools"][0]["function"]["name"] == "shell"
    assert "reasoning_effort" not in chat
    assert chat["chat_template_kwargs"] == {
        "thinking": False,
        "reasoning_effort": "none",
    }


def test_deepseek_message_sequence_and_max_tokens_clamp() -> None:
    upstream = BridgeUpstreamConfig(
        provider="nvidia",
        display_name="NVIDIA",
        api_key="nvapi",
        base_url="https://integrate.api.nvidia.com/v1",
        model="deepseek-ai/deepseek-v4-pro",
    )
    chat = convert_responses_request_to_chat(
        {
            "model": "deepseek-ai/deepseek-v4-pro",
            "instructions": "system first",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {
                    "type": "function_call_output",
                    "call_id": "call_x",
                    "output": "tool-out",
                },
            ],
            "max_output_tokens": 999_999,
            "reasoning": {"effort": "xhigh"},
            "stream": False,
        },
        upstream=upstream,
    )
    roles = [message["role"] for message in chat["messages"]]
    assert roles[0] == "system"
    assert chat["messages"][0]["content"] == "system first"
    tool = next(message for message in chat["messages"] if message["role"] == "tool")
    assert tool["tool_call_id"] == "call_x"
    assert chat["max_tokens"] <= 65_536
    assert chat["chat_template_kwargs"]["reasoning_effort"] == "max"
    assert "reasoning_effort" not in chat or chat.get("reasoning_effort") is None


def test_bridge_strips_routing_metadata_from_request_overrides() -> None:
    """Routing profile bookkeeping must not become NVIDIA Chat Completions params.

    Regression: model_configuration (source_levels, capability_source, …) was
    copied wholesale into bridge request_overrides, causing NVIDIA HTTP 400:
    Unsupported parameter(s): source_levels, capability_source.
    """
    upstream = BridgeUpstreamConfig(
        provider="nvidia",
        display_name="NVIDIA",
        api_key="nvapi",
        base_url="https://integrate.api.nvidia.com/v1",
        model="deepseek-ai/deepseek-v4-flash",
        request_overrides={
            "source_levels": ("pinned", "MODEL_LEVEL_1_FAST_TOOL"),
            "capability_source": "maintained-token-limits",
            "token_profile_confidence": "high",
            "model_kwargs": {"unused": True},
            "temperature": 0.2,
            "reasoning_effort": "high",
        },
    )
    chat = convert_responses_request_to_chat(
        {
            "model": "deepseek-ai/deepseek-v4-flash",
            "input": "hello",
            "stream": False,
        },
        upstream=upstream,
    )
    assert "source_levels" not in chat
    assert "capability_source" not in chat
    assert "token_profile_confidence" not in chat
    assert "model_kwargs" not in chat
    assert chat["temperature"] == 0.2
    # DeepSeek still shaped correctly; bare reasoning_effort is not forwarded.
    assert chat["chat_template_kwargs"]["thinking"] is True
    assert chat["chat_template_kwargs"]["reasoning_effort"] == "high"
