"""Anthropic (Claude) inference provider runtime adapter.

Provides full compatibility with Mana-Agent's chat model interface, supporting:
- Top-level system prompt isolation (Anthropic Messages API)
- Mandatory max_tokens parameter handling
- Alternating user/assistant message normalization
- Tool calling with tool_use and tool_result content blocks
- Streaming response chunks
- Isolated credential resolution via ANTHROPIC_API_KEY
- Error classification and bounded retry/backoff
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Callable, Iterator, Mapping, Sequence

import anthropic
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr, SecretStr

from mana_agent.config.user_config import get_setting
from mana_agent.context_cost import ContextCostGovernor
from mana_agent.context_cost.estimator import estimate_value_tokens
from mana_agent.context_cost.models import ContextSegment
from mana_agent.evals.ids import stable_hash
from mana_agent.evals.recorder import record_current
from mana_agent.multi_agent.runtime.compatibility import format_provider_error
from mana_agent.telemetry.tokens import token_usage_from_provider

logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
DEFAULT_MAX_TOKENS = 4096


def _convert_to_anthropic_tool(tool: Any) -> dict[str, Any]:
    """Convert a LangChain tool or raw dictionary into an Anthropic tool definition."""
    if isinstance(tool, dict):
        if "name" in tool and "input_schema" in tool:
            return tool
        if "type" in tool and tool["type"] == "function" and "function" in tool:
            fn = tool["function"]
            return {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": dict(fn.get("parameters") or {"type": "object", "properties": {}}),
            }
        if "name" in tool and ("parameters" in tool or "description" in tool):
            return {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": dict(tool.get("parameters") or {"type": "object", "properties": {}}),
            }
    from langchain_core.utils.function_calling import convert_to_openai_tool

    openai_tool = convert_to_openai_tool(tool)
    fn = openai_tool.get("function", openai_tool)
    return {
        "name": fn.get("name", ""),
        "description": fn.get("description", ""),
        "input_schema": dict(fn.get("parameters") or {"type": "object", "properties": {}}),
    }


def _format_anthropic_tool_choice(choice: Any) -> dict[str, Any] | None:
    """Format tool choice parameter according to Anthropic Messages API specification."""
    if not choice:
        return None
    if choice == "auto":
        return {"type": "auto"}
    if choice in ("any", "required"):
        return {"type": "any"}
    if isinstance(choice, str):
        return {"type": "tool", "name": choice}
    if isinstance(choice, dict):
        if choice.get("type") == "function" and "function" in choice:
            return {"type": "tool", "name": choice["function"].get("name", "")}
        return choice
    return None


def _map_messages_for_anthropic(
    messages: Sequence[BaseMessage],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert LangChain messages into Anthropic Messages API structure.

    Key mappings enforced:
    1. System messages extracted to top-level `system` parameter.
    2. ToolMessage converted to role `user` with `tool_result` content blocks.
    3. AIMessage with tool_calls converted to `tool_use` content blocks.
    4. Consecutive messages with identical roles merged into a single turn.
    5. First non-system message guaranteed to be role `user`.
    """
    system_prompts: list[str] = []
    raw_messages: list[dict[str, Any]] = []

    for msg in messages:
        role_type = str(getattr(msg, "type", "")).lower()

        if isinstance(msg, SystemMessage) or role_type == "system":
            if msg.content:
                system_prompts.append(str(msg.content))
            continue

        if isinstance(msg, ToolMessage) or role_type == "tool":
            content = str(msg.content or "")
            is_error = getattr(msg, "status", "") == "error"
            tool_call_id = getattr(msg, "tool_call_id", "") or ""
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": content,
            }
            if is_error:
                block["is_error"] = True
            raw_messages.append({
                "role": "user",
                "content": [block],
            })
            continue

        if isinstance(msg, AIMessage) or role_type in ("ai", "assistant"):
            blocks: list[dict[str, Any]] = []
            if msg.content:
                if isinstance(msg.content, str):
                    blocks.append({"type": "text", "text": msg.content})
                elif isinstance(msg.content, list):
                    blocks.extend(msg.content)
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                name = tc.get("name", "")
                args = tc.get("args") or {}
                blocks.append({
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": args if isinstance(args, dict) else {},
                })
            raw_messages.append({
                "role": "assistant",
                "content": blocks or [{"type": "text", "text": ""}],
            })
            continue

        # HumanMessage or fallback user message
        content = msg.content
        if isinstance(content, str):
            blocks = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            blocks = list(content)
        else:
            blocks = [{"type": "text", "text": str(content or "")}]
        raw_messages.append({
            "role": "user",
            "content": blocks,
        })

    # Merge consecutive messages with the same role to enforce alternating sequence
    merged: list[dict[str, Any]] = []
    for msg in raw_messages:
        if not merged:
            merged.append(msg)
            continue
        if merged[-1]["role"] == msg["role"]:
            prev = merged[-1]
            prev_blocks = prev["content"] if isinstance(prev["content"], list) else [{"type": "text", "text": str(prev["content"])}]
            curr_blocks = msg["content"] if isinstance(msg["content"], list) else [{"type": "text", "text": str(msg["content"])}]
            prev["content"] = prev_blocks + curr_blocks
        else:
            merged.append(msg)

    # First turn in Anthropic must be from role 'user'
    if not merged:
        merged = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    elif merged[0]["role"] == "assistant":
        merged.insert(0, {"role": "user", "content": [{"type": "text", "text": "Continue"}]})

    # Simplify single-text block content into plain string for clean payload representation
    for item in merged:
        content = item.get("content")
        if isinstance(content, list) and len(content) == 1 and content[0].get("type") == "text":
            item["content"] = content[0].get("text", "")

    system = "\n\n".join(system_prompts) if system_prompts else None
    return system, merged


class CompatibleChatAnthropic(BaseChatModel):
    """LangChain-compatible Chat Model adapter for Anthropic Claude."""

    anthropic_api_key: SecretStr | None = None
    model_name: str = DEFAULT_ANTHROPIC_MODEL
    anthropic_api_url: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float | None = None
    top_p: float | None = None
    default_headers: dict[str, str] | None = None
    selected_provider: str = "anthropic"
    context_cost_governor: ContextCostGovernor | None = None
    bound_tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: Any = None
    max_retries: int = 3

    _client: anthropic.Anthropic | None = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        client = kwargs.pop("client", None)
        super().__init__(**kwargs)
        if client is not None:
            self._client = client

    @property
    def _llm_type(self) -> str:
        return "chat-anthropic"

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            api_key = ""
            if self.anthropic_api_key:
                api_key = self.anthropic_api_key.get_secret_value() if hasattr(self.anthropic_api_key, "get_secret_value") else str(self.anthropic_api_key)
            if not api_key:
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            client_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "max_retries": self.max_retries,
            }
            if self.anthropic_api_url:
                client_kwargs["base_url"] = self.anthropic_api_url
            if self.default_headers:
                client_kwargs["default_headers"] = self.default_headers
            self._client = anthropic.Anthropic(**client_kwargs)
        return self._client

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> "CompatibleChatAnthropic":
        formatted_tools = [_convert_to_anthropic_tool(t) for t in tools]
        return self.model_copy(
            update={
                "bound_tools": formatted_tools,
                "tool_choice": _format_anthropic_tool_choice(tool_choice),
                **kwargs,
            }
        )

    def _eval_request_metadata(self, messages: list[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        roles = [str(getattr(message, "type", type(message).__name__)) for message in messages]
        prompt_material = [
            {"role": role, "content": str(getattr(message, "content", ""))}
            for role, message in zip(roles, messages)
        ]
        tool_names = [t.get("name", "") for t in (self.bound_tools or kwargs.get("tools") or []) if isinstance(t, dict)]
        return {
            "boundary": "compatible_chat_anthropic",
            "provider": self.selected_provider,
            "model": self.model_name,
            "prompt_hash": stable_hash(prompt_material),
            "safe_request_metadata": {
                "message_count": len(messages),
                "roles": roles,
                "tool_names": tool_names,
                "tool_call_count": len(tool_names),
                "streaming": bool(kwargs.get("stream")),
                "endpoint_host": self.anthropic_api_url or "https://api.anthropic.com",
            },
        }

    def _governor_preflight(
        self,
        messages: list[Any],
        kwargs: dict[str, Any],
        metadata: dict[str, Any],
    ) -> str:
        governor = self.context_cost_governor
        if governor is None or not governor.enabled:
            return ""
        segments: list[ContextSegment] = []
        human_indexes = [
            index for index, message in enumerate(messages)
            if str(getattr(message, "type", "")) in {"human", "user"}
        ]
        protected_humans = set(human_indexes[:1] + human_indexes[-1:])
        for index, message in enumerate(messages):
            role = str(getattr(message, "type", type(message).__name__)).lower()
            kind = "system" if role == "system" else "user" if index in protected_humans else "tool_result" if role == "tool" else "history"
            content = getattr(message, "content", "")
            protected_tool = role == "tool" and (
                index >= len(messages) - 3
                or any(marker in str(content).casefold() for marker in ("error", "approval", "mutation", "changed_files", "verification"))
            )
            segments.append(ContextSegment(
                kind=kind,
                content=content,
                token_estimate=estimate_value_tokens(content),
                protected=kind in {"system", "user"} or protected_tool,
                source_id=f"message:{index}",
            ))
        for index, schema in enumerate(self.bound_tools or kwargs.get("tools") or []):
            segments.append(ContextSegment(
                kind="schema",
                content=schema,
                token_estimate=estimate_value_tokens(schema),
                protected=True,
                source_id=f"schema:{index}",
            ))
        has_explicit_output_limit = any(
            kwargs.get(name) is not None
            for name in ("max_output_tokens", "max_completion_tokens", "max_tokens")
        )
        call_id, _decision = governor.before_model_call(
            segments,
            model=self.model_name,
            provider=str(metadata.get("provider") or "anthropic"),
            observation_boundary="compatible_chat",
            expected_output_tokens=(
                int(kwargs.get("max_output_tokens") or kwargs.get("max_completion_tokens") or kwargs.get("max_tokens") or self.max_tokens)
                if has_explicit_output_limit
                else self.max_tokens
            ),
            historical_prediction_enabled=not has_explicit_output_limit,
        )
        return call_id

    def _prepare_call_kwargs(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        system_prompt, anthropic_messages = _map_messages_for_anthropic(messages)
        tools = self.bound_tools or kwargs.get("tools")
        max_tokens = int(
            kwargs.get("max_tokens")
            or kwargs.get("max_output_tokens")
            or kwargs.get("max_completion_tokens")
            or self.max_tokens
            or DEFAULT_MAX_TOKENS
        )
        call_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            call_kwargs["system"] = system_prompt
        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            call_kwargs["temperature"] = float(temperature)
        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            call_kwargs["top_p"] = float(top_p)
        if stop:
            call_kwargs["stop_sequences"] = list(stop)
        if tools:
            call_kwargs["tools"] = [_convert_to_anthropic_tool(t) for t in tools]
            if self.tool_choice:
                call_kwargs["tool_choice"] = self.tool_choice
            elif "tool_choice" in kwargs:
                call_kwargs["tool_choice"] = _format_anthropic_tool_choice(kwargs["tool_choice"])
        return call_kwargs

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        started = time.perf_counter()
        metadata = self._eval_request_metadata(messages, kwargs)
        governor_call_id = self._governor_preflight(messages, kwargs, metadata)
        call_kwargs = self._prepare_call_kwargs(messages, stop=stop, **kwargs)

        client = self._get_client()
        response: Any = None
        max_attempts = max(1, self.max_retries)

        for attempt in range(1, max_attempts + 1):
            try:
                response = client.messages.create(**call_kwargs)
                break
            except Exception as exc:
                status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
                is_retryable = (
                    status in (429, 529, 500, 502, 503, 504)
                    or isinstance(exc, (anthropic.APIConnectionError, anthropic.RateLimitError))
                )
                if attempt < max_attempts and is_retryable:
                    delay = min(0.5 * (2 ** (attempt - 1)), 5.0)
                    time.sleep(delay)
                    continue

                if self.context_cost_governor is not None and governor_call_id:
                    self.context_cost_governor.release_reservation(
                        governor_call_id, reason=f"anthropic provider call failed: {type(exc).__name__}"
                    )
                safe_error = format_provider_error(
                    exc, provider=self.selected_provider, model=self.model_name
                )
                logger.error(
                    "llm.request_failed provider=%s model=%s streaming=false "
                    "retry_count=%s error_type=%s status=%s detail=%s",
                    self.selected_provider,
                    self.model_name,
                    attempt - 1,
                    type(exc).__name__,
                    status,
                    safe_error,
                )
                record_current(
                    "model.call.failed",
                    {
                        **metadata,
                        "latency_seconds": time.perf_counter() - started,
                        "error_type": type(exc).__name__,
                        "error": safe_error,
                    },
                )
                raise

        # Parse text content and tool calls from response blocks
        text_content = ""
        tool_calls: list[dict[str, Any]] = []

        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text_content += getattr(block, "text", "")
            elif block_type == "tool_use":
                raw_input = getattr(block, "input", {})
                if isinstance(raw_input, str):
                    try:
                        args = json.loads(raw_input)
                    except Exception:
                        args = {"raw": raw_input}
                elif isinstance(raw_input, dict):
                    args = raw_input
                else:
                    args = {}
                tool_calls.append({
                    "id": getattr(block, "id", "") or f"call_{uuid.uuid4().hex[:8]}",
                    "name": getattr(block, "name", ""),
                    "args": args,
                })

        input_tokens = getattr(getattr(response, "usage", None), "input_tokens", 0) or 0
        output_tokens = getattr(getattr(response, "usage", None), "output_tokens", 0) or 0
        total_tokens = input_tokens + output_tokens

        usage_dict = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        ai_message = AIMessage(
            content=text_content,
            tool_calls=tool_calls,
            response_metadata={
                "stop_reason": getattr(response, "stop_reason", None),
                "model_name": self.model_name,
            },
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        )
        if self.context_cost_governor is not None and governor_call_id:
            self.context_cost_governor.record_model_call(
                governor_call_id,
                usage=usage_dict,
                provider=self.selected_provider,
                model=self.model_name,
                estimated_input=[getattr(m, "content", "") for m in messages],
                estimated_output=ai_message,
            )
        record_current(
            "model.call",
            {
                **metadata,
                "latency_seconds": time.perf_counter() - started,
                "usage": usage_dict,
            },
        )
        return ChatResult(
            generations=[ChatGeneration(message=ai_message)],
            llm_output={"token_usage": usage_dict, "model_name": self.model_name},
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        started = time.perf_counter()
        metadata = self._eval_request_metadata(messages, kwargs)
        governor_call_id = self._governor_preflight(messages, kwargs, metadata)
        call_kwargs = self._prepare_call_kwargs(messages, stop=stop, **kwargs)

        client = self._get_client()
        input_tokens = 0
        output_tokens = 0

        try:
            stream = client.messages.create(stream=True, **call_kwargs)
            for event in stream:
                event_type = getattr(event, "type", "")

                if event_type == "message_start":
                    message_obj = getattr(event, "message", None)
                    usage_obj = getattr(message_obj, "usage", None) if message_obj else None
                    if usage_obj:
                        input_tokens = getattr(usage_obj, "input_tokens", 0) or 0

                elif event_type == "content_block_start":
                    cb = getattr(event, "content_block", None)
                    if cb and getattr(cb, "type", "") == "tool_use":
                        idx = getattr(event, "index", 0)
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(
                                content="",
                                tool_call_chunks=[
                                    ToolCallChunk(
                                        name=getattr(cb, "name", ""),
                                        args="",
                                        id=getattr(cb, "id", "") or f"call_{uuid.uuid4().hex[:8]}",
                                        index=idx,
                                    )
                                ],
                            )
                        )

                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        delta_type = getattr(delta, "type", "")
                        if delta_type == "text_delta":
                            text = getattr(delta, "text", "")
                            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                            if run_manager:
                                run_manager.on_llm_new_token(text, chunk=chunk)
                            yield chunk
                        elif delta_type == "input_json_delta":
                            partial = getattr(delta, "partial_json", "")
                            idx = getattr(event, "index", 0)
                            yield ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content="",
                                    tool_call_chunks=[
                                        ToolCallChunk(
                                            name=None,
                                            args=partial,
                                            id=None,
                                            index=idx,
                                        )
                                    ],
                                )
                            )

                elif event_type == "message_delta":
                    usage_obj = getattr(event, "usage", None)
                    if usage_obj:
                        output_tokens = getattr(usage_obj, "output_tokens", 0) or output_tokens
                        total_tokens = input_tokens + output_tokens
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(
                                content="",
                                usage_metadata={
                                    "input_tokens": input_tokens,
                                    "output_tokens": output_tokens,
                                    "total_tokens": total_tokens,
                                },
                            )
                        )

                elif hasattr(event, "message") and hasattr(event.message, "content"):
                    # Direct generation chunk fallback for mocked streams
                    chunk_text = str(getattr(event.message, "content", ""))
                    if chunk_text:
                        chunk = ChatGenerationChunk(message=AIMessageChunk(content=chunk_text))
                        if run_manager:
                            run_manager.on_llm_new_token(chunk_text, chunk=chunk)
                        yield chunk

        except Exception as exc:
            if self.context_cost_governor is not None and governor_call_id:
                self.context_cost_governor.release_reservation(
                    governor_call_id, reason=f"streaming anthropic provider call failed: {type(exc).__name__}"
                )
            safe_error = format_provider_error(
                exc, provider=self.selected_provider, model=self.model_name
            )
            status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            logger.error(
                "llm.request_failed provider=%s model=%s streaming=true "
                "error_type=%s status=%s detail=%s",
                self.selected_provider,
                self.model_name,
                type(exc).__name__,
                status,
                safe_error,
            )
            record_current(
                "model.call.failed",
                {
                    **metadata,
                    "latency_seconds": time.perf_counter() - started,
                    "error_type": type(exc).__name__,
                    "error": safe_error,
                    "streaming": True,
                },
            )
            raise

        total = input_tokens + output_tokens
        if self.context_cost_governor is not None and governor_call_id:
            self.context_cost_governor.record_model_call(
                governor_call_id,
                usage={"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": total},
                provider=self.selected_provider,
                model=self.model_name,
                estimated_input=[getattr(m, "content", "") for m in messages],
            )
        record_current(
            "model.call",
            {
                **metadata,
                "latency_seconds": time.perf_counter() - started,
                "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": total},
                "streaming": True,
            },
        )
