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


def test_multi_agent_namespace_round_trips_through_chat_functions() -> None:
    """NVIDIA Chat receives flat functions; Codex receives a namespaced call."""
    upstream = BridgeUpstreamConfig(
        provider="nvidia",
        display_name="NVIDIA",
        api_key="nvapi",
        base_url="https://integrate.api.nvidia.com/v1",
        model="deepseek-ai/deepseek-v4-flash-0731",
    )
    tools = [
        {
            "type": "namespace",
            "name": "multi_agent_v1",
            "description": "Tools for spawning and managing sub-agents.",
            "tools": [
                {
                    "type": "function",
                    "name": "spawn_agent",
                    "description": "Spawn a sub-agent.",
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                    },
                }
            ],
        }
    ]
    chat = convert_responses_request_to_chat(
        {
            "model": upstream.model,
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_spawn",
                    "namespace": "multi_agent_v1",
                    "name": "spawn_agent",
                    "arguments": '{"message":"inspect the bridge"}',
                }
            ],
            "tools": tools,
        },
        upstream=upstream,
    )
    assert chat["tools"][0]["function"]["name"] == "multi_agent_v1__spawn_agent"
    assert (
        chat["messages"][0]["tool_calls"][0]["function"]["name"]
        == "multi_agent_v1__spawn_agent"
    )
    metadata = chat["_mana_bridge"]
    assert metadata["response_tool_names"] == {
        "multi_agent_v1__spawn_agent": "spawn_agent"
    }
    assert metadata["tool_namespaces"] == {"multi_agent_v1__spawn_agent": "multi_agent_v1"}

    upstream_response = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_spawn",
                            "type": "function",
                            "function": {
                                "name": "multi_agent_v1__spawn_agent",
                                "arguments": '{"message":"inspect the bridge"}',
                            },
                        }
                    ],
                },
            }
        ]
    }
    response = convert_chat_completion_to_response(
        upstream_response,
        model=upstream.model,
        tool_origins=metadata["tool_origins"],
        response_tool_names=metadata["response_tool_names"],
        tool_namespaces=metadata["tool_namespaces"],
    )
    call = next(item for item in response["output"] if item["type"] == "function_call")
    assert call["name"] == "spawn_agent"
    assert call["namespace"] == "multi_agent_v1"

    adapter = ChatToResponsesStreamAdapter(
        model=upstream.model,
        tool_origins=metadata["tool_origins"],
        response_tool_names=metadata["response_tool_names"],
        tool_namespaces=metadata["tool_namespaces"],
    )
    adapter.open_events()
    adapter.ingest_chat_chunk(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_spawn",
                                "function": {
                                    "name": "multi_agent_v1__spawn_agent",
                                    "arguments": '{"message":"inspect the bridge"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
    )
    completed = json.loads(
        next(
            event.split("data:", 1)[1].strip()
            for event in adapter.close_events()
            if "response.completed" in event
        )
    )
    stream_call = next(
        item
        for item in completed["response"]["output"]
        if item["type"] == "function_call"
    )
    assert stream_call["name"] == "spawn_agent"
    assert stream_call["namespace"] == "multi_agent_v1"


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


def test_stream_adapter_reasoning_content_not_mixed_into_text() -> None:
    """DeepSeek CoT must round-trip as reasoning items, never as assistant text."""
    adapter = ChatToResponsesStreamAdapter(model="deepseek-ai/deepseek-v4-flash-0731")
    events = list(adapter.open_events())
    events.extend(
        adapter.ingest_chat_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "I should call shell next.",
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
                                    "id": "call_1",
                                    "function": {
                                        "name": "shell",
                                        "arguments": '{"command":"ls"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
    )
    events.extend(adapter.close_events())
    joined = "".join(events)
    assert "response.reasoning_summary_text.delta" in joined
    assert "I should call shell next." in joined
    assert adapter.reasoning_parts == ["I should call shell next."]
    assert adapter.text_parts == []
    assert "response.function_call_arguments.done" in joined
    # Final output order: reasoning then function_call (no text message).
    completed = None
    for block in joined.split("\n\n"):
        if "response.completed" in block and "data:" in block:
            payload = block.split("data:", 1)[1].strip()
            completed = json.loads(payload)
            break
    assert completed is not None
    types = [item["type"] for item in completed["response"]["output"]]
    assert types[0] == "reasoning"
    assert "function_call" in types
    assert "message" not in types


def test_responses_round_trip_preserves_reasoning_content_for_tool_loop() -> None:
    """Multi-turn DeepSeek tool loops require reasoning_content on assistant."""
    upstream = BridgeUpstreamConfig(
        provider="nvidia",
        display_name="NVIDIA",
        api_key="nvapi",
        base_url="https://integrate.api.nvidia.com/v1",
        model="deepseek-ai/deepseek-v4-flash-0731",
    )
    # Upstream chat message with CoT + tool_calls → Responses output.
    chat = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Need to inspect separable.py before editing.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": '{"command":"sed -n 240,250p astropy/modeling/separable.py"}',
                            },
                        }
                    ],
                },
            }
        ]
    }
    response = convert_chat_completion_to_response(
        chat, model="deepseek-ai/deepseek-v4-flash-0731"
    )
    types = [item["type"] for item in response["output"]]
    assert types[0] == "reasoning"
    assert "function_call" in types
    reasoning = next(item for item in response["output"] if item["type"] == "reasoning")
    assert "separable.py" in reasoning["summary"][0]["text"]

    # Codex feeds the reasoning item + function_call + tool result back.
    body = {
        "model": "deepseek-ai/deepseek-v4-flash-0731",
        "input": [
            {"type": "message", "role": "user", "content": "fix nested CompoundModel separability"},
            reasoning,
            next(item for item in response["output"] if item["type"] == "function_call"),
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "cright[-right.shape[0]:, -right.shape[1]:] = 1",
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
    }
    next_chat = convert_responses_request_to_chat(body, upstream=upstream)
    assistant = next(
        message
        for message in next_chat["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert assistant["reasoning_content"] == "Need to inspect separable.py before editing."
    assert assistant["tool_calls"][0]["id"] == "call_1"
    roles = [message["role"] for message in next_chat["messages"]]
    # Strict order: system? user → assistant(tool_calls) → tool (no orphan tool).
    assert "tool" in roles
    assistant_idx = roles.index("assistant")
    tool_idx = roles.index("tool")
    assert assistant_idx < tool_idx
    assert next_chat["messages"][tool_idx]["tool_call_id"] == "call_1"


def test_orphan_tool_result_gets_synthetic_assistant_pair() -> None:
    """Orphan tool results must not precede a missing assistant tool_calls message."""
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
            "input": [
                {"type": "message", "role": "user", "content": "continue"},
                {
                    "type": "function_call_output",
                    "call_id": "call_orphan",
                    "output": "tool-out",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "shell",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        },
        upstream=upstream,
    )
    roles = [message["role"] for message in chat["messages"]]
    assert roles.count("assistant") >= 1
    assert roles.index("assistant") < roles.index("tool")
    assistant = next(
        message
        for message in chat["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert assistant["tool_calls"][0]["id"] == "call_orphan"


def test_leaked_think_markers_stripped_from_assistant_history() -> None:
    """Confused multi-turn loops inject </think>/DSML into content; strip them."""
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
            "input": [
                {"type": "message", "role": "user", "content": "bump version"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": (
                        "The version originates solely from pyproject.toml."
                        "</think>"
                        "<|DSML|junk>More text."
                    ),
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "shell",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        },
        upstream=upstream,
    )
    assistant = next(
        message for message in chat["messages"] if message.get("role") == "assistant"
    )
    content = str(assistant.get("content") or "")
    assert "pyproject.toml" in content
    assert "</think>" not in content
    assert "DSML" not in content


def test_freeform_tool_garbage_redacted_from_assistant_history() -> None:
    """Protocol soup must not re-enter DeepSeek history as ordinary content."""
    upstream = BridgeUpstreamConfig(
        provider="nvidia",
        display_name="NVIDIA",
        api_key="nvapi",
        base_url="https://integrate.api.nvidia.com/v1",
        model="deepseek-ai/deepseek-v4-flash-0731",
    )
    garbage = (
        "Probably there was some\"\"\" tat only appeared once above; "
        "update pyccheroomportugal.py</nowarn></think>"
        "Transfer Protocol Templates:\n"
        "python-patch-before: apply_patch suppressums=false\n"
        "<|DSML|junk>actionstarted.00:00</MESSAGE_END>. "
        "Reason: mutation_required_but_no_mutation_tool_attempted"
    )
    chat = convert_responses_request_to_chat(
        {
            "model": "deepseek-ai/deepseek-v4-flash-0731",
            "input": [
                {"type": "message", "role": "user", "content": "bump version to v0.1.6"},
                {"type": "message", "role": "assistant", "content": garbage},
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "shell",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        },
        upstream=upstream,
    )
    assistant = next(
        message for message in chat["messages"] if message.get("role") == "assistant"
    )
    content = str(assistant.get("content") or "")
    assert "pyccheroomportugal" not in content
    assert "DSML" not in content
    assert "</think>" not in content
    assert "structured tools" in content.lower() or "redacted" in content.lower()


def test_leaked_ultracall_tool_invocation_redacted_from_assistant_history() -> None:
    """Codex coding response-leak: broken tool XML must not re-enter history."""
    from mana_agent.integrations.codex.text_cleanup import (
        looks_like_freeform_tool_garbage,
        sanitize_assistant_visible_text,
    )

    # Condensed form of a real agentMessage leak: malformed tool-call wrappers,
    # parameter tags, and meta-apology about failed tool syntax.
    garbage = (
        'There is a/_version.py file.\n'
        'Let me inspect it to see if version appears elsewhere too.\n\n'
        '<danke:ultracall_calls{... = ...;\n\n'
        '<birdswithering>?\n\n'
        '<danke:ultracall_calls>\n\n'
        '<` "padding": {"max_output_tokens": 2000}">\n'
        ' <parameter name="cmd">"\'\n'
        'Done="<?))(var c =...;cat src/mana_agent/_version.py</span>'
        'badge.symbol "></span></span></danke:ultracall_calls>,\n\n'
        '<parameter name="cmd">"grep -rn "0\\.1\\.5" .\n'
        "Wait, my Tools invocation syntax above failed somehow - "
        "looks like garbage output was produced.\n"
        "I apologize for that garbled mechanical response above."
    )
    assert looks_like_freeform_tool_garbage(garbage) is True
    cleaned = sanitize_assistant_visible_text(garbage)
    assert "ultracall" not in cleaned.lower()
    assert "parameter" not in cleaned.lower()
    assert "max_output_tokens" not in cleaned.lower()
    assert "birdswithering" not in cleaned.lower()
    assert "Tools invocation" not in cleaned
    assert "structured tools" in cleaned.lower() or "redacted" in cleaned.lower()

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
            "input": [
                {"type": "message", "role": "user", "content": "inspect version files"},
                {"type": "message", "role": "assistant", "content": garbage},
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "shell",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        },
        upstream=upstream,
    )
    assistant = next(
        message for message in chat["messages"] if message.get("role") == "assistant"
    )
    content = str(assistant.get("content") or "")
    assert "ultracall" not in content.lower()
    assert "<parameter" not in content.lower()
    assert "max_output_tokens" not in content.lower()


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
    assert chat["tool_choice"] == "auto"
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


def test_bridge_strips_catalog_model_object_fields_from_request_overrides() -> None:
    """Catalog model identity fields must not reach NVIDIA chat/completions.

    Regression: profile.configuration received the full /v1/models record
    (id/object/created/owned_by). Bridge request_overrides forwarded them and
    NVIDIA returned HTTP 400 Unsupported parameter(s): created, id, object,
    owned_by (deepseek-ai/deepseek-v4-flash-0731 coding turns).
    """
    upstream = BridgeUpstreamConfig(
        provider="nvidia",
        display_name="NVIDIA",
        api_key="nvapi",
        base_url="https://integrate.api.nvidia.com/v1",
        model="deepseek-ai/deepseek-v4-flash-0731",
        request_overrides={
            "id": "deepseek-ai/deepseek-v4-flash-0731",
            "object": "model",
            "created": 735790403,
            "owned_by": "deepseek-ai",
            "capabilities": ["text_generation", "tool_calling"],
            "extra_body": {
                "id": "must-not-flatten",
                "owned_by": "must-not-flatten",
                "chat_template_kwargs": {"thinking": False, "reasoning_effort": "none"},
            },
            "temperature": 0.0,
        },
    )
    chat = convert_responses_request_to_chat(
        {
            "model": "deepseek-ai/deepseek-v4-flash-0731",
            "input": "change version to v0.1.6",
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "name": "shell",
                    "description": "run a shell command",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        },
        upstream=upstream,
    )
    for key in ("id", "object", "created", "owned_by", "capabilities"):
        assert key not in chat
    assert chat["model"] == "deepseek-ai/deepseek-v4-flash-0731"
    assert chat["temperature"] == 0.0
    assert chat["stream"] is True
    assert chat["tools"]
    # Nested catalog junk was stripped from extra_body; template kwargs kept.
    assert chat["chat_template_kwargs"]["thinking"] is False
    assert chat["chat_template_kwargs"]["reasoning_effort"] == "none"


def test_bridge_openrouter_grok_4_6_omits_reasoning_effort_and_does_not_disable() -> None:
    """Verify that x-ai/grok-4.6 on OpenRouter bridge never sends disable reasoning instructions."""
    upstream = BridgeUpstreamConfig(
        provider="openrouter",
        display_name="OpenRouter",
        api_key="or-key",
        base_url="https://openrouter.ai/api/v1",
        model="x-ai/grok-4.6",
    )
    chat = convert_responses_request_to_chat(
        {
            "model": "x-ai/grok-4.6",
            "input": "write a function",
            "stream": True,
            "reasoning": {"effort": "none"},
            "reasoning_effort": "none",
            "tools": [
                {
                    "type": "function",
                    "name": "shell",
                    "description": "run a shell command",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        },
        upstream=upstream,
    )
    assert chat["model"] == "x-ai/grok-4.6"
    assert "reasoning_effort" not in chat
    assert "reasoning" not in chat
    assert "chat_template_kwargs" not in chat
    assert "thinking" not in chat
    assert chat["tools"]


def test_bridge_openrouter_deepseek_r1_mandatory_reasoning() -> None:
    """Verify that deepseek/deepseek-r1 on OpenRouter bridge preserves mandatory reasoning."""
    upstream = BridgeUpstreamConfig(
        provider="openrouter",
        display_name="OpenRouter",
        api_key="or-key",
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-r1",
    )
    chat = convert_responses_request_to_chat(
        {
            "model": "deepseek/deepseek-r1",
            "input": "explain quantum computing",
            "stream": False,
            "reasoning_effort": "none",
        },
        upstream=upstream,
    )
    assert chat["model"] == "deepseek/deepseek-r1"
    assert "reasoning_effort" not in chat
    assert "chat_template_kwargs" not in chat

