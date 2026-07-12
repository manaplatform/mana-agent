"""Capability-driven OpenAI-compatible request construction.

This module is the single construction point for chat models used by the
runtime. It keeps Responses and Chat Completions request shapes separate while
preserving LangChain's tool adapter and response parsing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import Any, Iterator, Literal

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

ApiMode = Literal["auto", "responses", "chat_completions"]


@dataclass(frozen=True)
class ModelCapabilities:
    """Transport capabilities for an OpenAI-compatible provider/model pair."""

    supports_responses_api: bool
    supports_chat_completions: bool = True
    supports_tools: bool = True
    supports_reasoning: bool = True
    supports_tools_with_chat_reasoning: bool = False


def _optional_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _api_mode_from_env() -> ApiMode:
    value = str(os.getenv("MANA_LLM_API_MODE") or "auto").strip().lower()
    if value not in {"auto", "responses", "chat_completions"}:
        raise ValueError("MANA_LLM_API_MODE must be auto, responses, or chat_completions")
    return value  # type: ignore[return-value]


def resolve_model_capabilities(*, base_url: str | None) -> tuple[ApiMode, ModelCapabilities]:
    """Resolve safe defaults, with explicit environment overrides for gateways.

    A custom OpenAI-compatible URL is intentionally *not* presumed to implement
    the Responses API. Operators can opt in after verifying their gateway.
    """

    normalized_url = str(base_url or "https://api.openai.com/v1").rstrip("/").lower()
    is_openai = normalized_url in {"https://api.openai.com/v1", "https://api.openai.com"}
    defaults = ModelCapabilities(
        supports_responses_api=is_openai,
        supports_tools=True,
        supports_reasoning=True,
        supports_tools_with_chat_reasoning=is_openai,
    )
    overrides = {
        "supports_responses_api": _optional_bool("MANA_LLM_SUPPORTS_RESPONSES_API"),
        "supports_chat_completions": _optional_bool("MANA_LLM_SUPPORTS_CHAT_COMPLETIONS"),
        "supports_tools": _optional_bool("MANA_LLM_SUPPORTS_TOOLS"),
        "supports_reasoning": _optional_bool("MANA_LLM_SUPPORTS_REASONING"),
        "supports_tools_with_chat_reasoning": _optional_bool(
            "MANA_LLM_SUPPORTS_TOOLS_WITH_CHAT_REASONING"
        ),
    }
    return _api_mode_from_env(), replace(
        defaults, **{key: value for key, value in overrides.items() if value is not None}
    )


def _has_tools(payload: dict[str, Any]) -> bool:
    return bool(payload.get("tools"))


def _has_reasoning(payload: dict[str, Any]) -> bool:
    effort = payload.get("reasoning_effort")
    return bool(payload.get("reasoning")) or (effort is not None and str(effort).lower() != "none")


def _is_tools_reasoning_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "reasoning_effort" in text
        and "tool" in text
        and ("not supported" in text or "unsupported" in text)
    )


class CompatibleChatOpenAI(ChatOpenAI):
    """``ChatOpenAI`` with endpoint selection and one safe compatibility retry."""

    compatibility_api_mode: ApiMode = "auto"
    compatibility_capabilities: ModelCapabilities = ModelCapabilities(False)
    compatibility_retry_attempted: bool = False
    # Set only for the bounded recovery request. This is intentionally a
    # request-construction guard rather than an inference from provider
    # metadata: the provider has already rejected the previous payload.
    compatibility_force_reasoning_none: bool = False

    def _use_responses_api(self, payload: dict) -> bool:
        if self.compatibility_api_mode == "responses":
            return True
        if self.compatibility_api_mode == "chat_completions":
            return False
        if _has_tools(payload):
            # Some OpenAI reasoning models apply reasoning by default even
            # when callers omit ``reasoning_effort``.  Their Chat
            # Completions endpoint rejects function tools unless callers
            # explicitly disable that default.  The Responses API is the
            # compatible native tool path, so use it whenever it is available
            # rather than waiting for a provider rejection and a lossy retry.
            return self.compatibility_capabilities.supports_responses_api
        # Do not make a custom gateway a Responses API client merely because
        # ``reasoning_effort`` is configured. Chat Completions still accepts it
        # when no tools are attached.
        return False

    def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        needs_chat_reasoning_normalization = (
            self.compatibility_force_reasoning_none
            or (
                _has_reasoning(payload)
                and not self._use_responses_api({**payload, "tools": payload.get("tools")})
                and not self.compatibility_capabilities.supports_tools_with_chat_reasoning
            )
        )
        if _has_tools(payload) and needs_chat_reasoning_normalization:
            # Chat Completions providers disagree on whether ``none`` is
            # accepted. The documented OpenAI-compatible form is used here;
            # explicit overrides can instead select Responses API support.
            payload.pop("reasoning", None)
            payload["reasoning_effort"] = "none"
            logger.info(
                "llm.compatibility_adjustment api_mode=chat_completions model=%s "
                "reasoning_effort=none reason=tools_with_reasoning_unsupported",
                self.model_name,
            )
        logger.debug(
            "llm.request api_mode=%s model=%s tools=%s reasoning=%s",
            "responses" if self._use_responses_api(payload) else "chat_completions",
            self.model_name,
            _has_tools(payload),
            payload.get("reasoning", {}).get("effort") if isinstance(payload.get("reasoning"), dict) else payload.get("reasoning_effort"),
        )
        return payload

    def _retry_without_chat_reasoning(self) -> "CompatibleChatOpenAI":
        return self.model_copy(
            update={
                "compatibility_api_mode": "chat_completions",
                "compatibility_capabilities": replace(
                    self.compatibility_capabilities,
                    supports_responses_api=False,
                    supports_tools_with_chat_reasoning=False,
                ),
                "compatibility_retry_attempted": True,
                "compatibility_force_reasoning_none": True,
            }
        )

    def _generate(self, messages: list[Any], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> Any:
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as exc:
            if self.compatibility_retry_attempted or not _has_tools(kwargs) or not _is_tools_reasoning_error(exc):
                raise
            logger.warning(
                "llm.compatibility_adjustment api_mode=chat_completions model=%s "
                "reason=tools_with_reasoning_unsupported retry=1",
                self.model_name,
            )
            return self._retry_without_chat_reasoning()._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        yielded = False
        try:
            for chunk in super()._stream(*args, **kwargs):
                yielded = True
                yield chunk
        except Exception as exc:
            if yielded or self.compatibility_retry_attempted or not _has_tools(kwargs) or not _is_tools_reasoning_error(exc):
                raise
            logger.warning(
                "llm.compatibility_adjustment api_mode=chat_completions model=%s "
                "reason=tools_with_reasoning_unsupported retry=1 streaming=true",
                self.model_name,
            )
            yield from self._retry_without_chat_reasoning()._stream(*args, **kwargs)


def create_chat_model(*, api_key: str, model: str, base_url: str | None = None, **kwargs: Any) -> CompatibleChatOpenAI:
    """Create the shared compatibility-aware LLM client used by every runtime role."""

    api_mode, capabilities = resolve_model_capabilities(base_url=base_url)
    reasoning_effort = str(os.getenv("MANA_LLM_REASONING_EFFORT") or "").strip() or None
    init_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "model": model,
        "compatibility_api_mode": api_mode,
        "compatibility_capabilities": capabilities,
        **kwargs,
    }
    if base_url:
        init_kwargs["base_url"] = base_url
    if reasoning_effort and "reasoning_effort" not in init_kwargs and "reasoning" not in init_kwargs:
        init_kwargs["reasoning_effort"] = reasoning_effort
    return CompatibleChatOpenAI(**init_kwargs)
