"""Capability-driven OpenAI-compatible request construction.

This module is the single construction point for chat models used by the
runtime. It keeps Responses and Chat Completions request shapes separate while
preserving LangChain's tool adapter and response parsing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Any, Iterator, Literal

from langchain_openai import ChatOpenAI
from mana_agent.config.user_config import get_setting
from mana_agent.evals.ids import stable_hash
from mana_agent.evals.recorder import record_current
from mana_agent.telemetry.tokens import token_usage_from_provider
from mana_agent.context_cost import ContextCostGovernor
from mana_agent.context_cost.estimator import estimate_value_tokens
from mana_agent.context_cost.models import ContextSegment

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
    value = get_setting(name)
    if value is None or not str(value).strip():
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _api_mode_from_config() -> ApiMode:
    value = str(get_setting("MANA_LLM_API_MODE", "auto") or "auto").strip().lower()
    if value not in {"auto", "responses", "chat_completions"}:
        raise ValueError("MANA_LLM_API_MODE must be auto, responses, or chat_completions")
    return value  # type: ignore[return-value]


def resolve_model_capabilities(*, base_url: str | None, provider: str | None = None) -> tuple[ApiMode, ModelCapabilities]:
    """Resolve safe defaults, with explicit environment overrides for gateways.

    A custom OpenAI-compatible URL is intentionally *not* presumed to implement
    the Responses API. Operators can opt in after verifying their gateway.
    NVIDIA Build / NIM is Chat Completions first; Responses API is not assumed.
    """

    provider_id = str(provider or "").strip().lower()
    normalized_url = str(base_url or "https://api.openai.com/v1").rstrip("/").lower()
    is_openai = provider_id == "openai" or normalized_url in {
        "https://api.openai.com/v1",
        "https://api.openai.com",
    }
    # NVIDIA Build / NIM and other OpenAI-compatible hosts stay on Chat
    # Completions unless the operator opts in via MANA_LLM_SUPPORTS_RESPONSES_API.
    defaults = ModelCapabilities(
        supports_responses_api=is_openai and provider_id != "nvidia",
        supports_tools=True,
        supports_reasoning=True,
        supports_tools_with_chat_reasoning=is_openai and provider_id != "nvidia",
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
    return _api_mode_from_config(), replace(
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
    context_cost_governor: ContextCostGovernor | None = None
    selected_provider: str = ""

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
        # NVIDIA DeepSeek V4 requires chat_template_kwargs; bare OpenAI-style
        # reasoning_effort alone can hang or fail on integrate.api.nvidia.com.
        from mana_agent.config.nvidia_model_requests import apply_nvidia_chat_completion_shaping
        from mana_agent.config.user_config import get_setting as _get_setting

        # LangChain may nest extras under ``extra_body``; flatten before shaping
        # so NIM receives top-level chat_template_kwargs in the HTTP body.
        extra_body = payload.pop("extra_body", None)
        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                if key == "chat_template_kwargs" and isinstance(value, dict):
                    nested = dict(payload.get("chat_template_kwargs") or {})
                    nested.update(value)
                    payload["chat_template_kwargs"] = nested
                else:
                    payload.setdefault(key, value)

        default_effort = str(_get_setting("MANA_LLM_REASONING_EFFORT", "") or "").strip() or "high"
        if self.compatibility_force_reasoning_none:
            default_effort = "none"
        apply_nvidia_chat_completion_shaping(
            payload,
            provider=self.selected_provider,
            model=self.model_name,
            default_effort=default_effort,
        )
        template = payload.get("chat_template_kwargs") if isinstance(payload.get("chat_template_kwargs"), dict) else {}
        logger.debug(
            "llm.request api_mode=%s model=%s tools=%s reasoning=%s chat_template_kwargs=%s",
            "responses" if self._use_responses_api(payload) else "chat_completions",
            self.model_name,
            _has_tools(payload),
            payload.get("reasoning", {}).get("effort") if isinstance(payload.get("reasoning"), dict) else payload.get("reasoning_effort"),
            bool(template),
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
        started = time.perf_counter()
        metadata = self._eval_request_metadata(messages, kwargs)
        governor_call_id = self._governor_preflight(messages, kwargs, metadata)
        try:
            result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as exc:
            if self.context_cost_governor is not None and governor_call_id:
                self.context_cost_governor.release_reservation(
                    governor_call_id, reason=f"provider call failed: {type(exc).__name__}"
                )
            safe_error = format_provider_error(
                exc, provider=self.selected_provider, model=self.model_name
            )
            status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            if status is None:
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
            logger.error(
                "llm.request_failed provider=%s model=%s streaming=false "
                "retry_count=%s error_type=%s status=%s detail=%s",
                self.selected_provider or "unknown",
                self.model_name,
                int(self.compatibility_retry_attempted),
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
                    "retry_attempt": int(self.compatibility_retry_attempted),
                },
            )
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
        llm_output = getattr(result, "llm_output", None)
        raw_usage = llm_output.get("token_usage") if isinstance(llm_output, dict) else None
        usage = token_usage_from_provider(raw_usage)
        if self.context_cost_governor is not None and governor_call_id:
            self.context_cost_governor.record_model_call(
                governor_call_id,
                usage=raw_usage,
                provider=str(metadata.get("provider") or ""),
                model=self.model_name,
                estimated_input=[getattr(message, "content", "") for message in messages],
                estimated_output=result,
            )
        record_current(
            "model.call",
            {
                **metadata,
                "latency_seconds": time.perf_counter() - started,
                "usage": usage.as_dict(),
                "retry_attempt": int(self.compatibility_retry_attempted),
            },
        )
        return result

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        yielded = False
        started = time.perf_counter()
        messages = args[0] if args and isinstance(args[0], list) else kwargs.get("messages", [])
        metadata = self._eval_request_metadata(messages, kwargs)
        governor_call_id = self._governor_preflight(messages, kwargs, metadata)
        usage: Any = None
        try:
            for chunk in super()._stream(*args, **kwargs):
                yielded = True
                usage = getattr(chunk, "usage_metadata", None) or usage
                yield chunk
        except Exception as exc:
            if self.context_cost_governor is not None and governor_call_id:
                self.context_cost_governor.release_reservation(
                    governor_call_id, reason=f"streaming provider call failed: {type(exc).__name__}"
                )
            safe_error = format_provider_error(
                exc, provider=self.selected_provider, model=self.model_name
            )
            status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            if status is None:
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
            logger.error(
                "llm.request_failed provider=%s model=%s streaming=true "
                "retry_count=%s error_type=%s status=%s detail=%s",
                self.selected_provider or "unknown",
                self.model_name,
                int(self.compatibility_retry_attempted),
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
                    "retry_attempt": int(self.compatibility_retry_attempted),
                },
            )
            if yielded or self.compatibility_retry_attempted or not _has_tools(kwargs) or not _is_tools_reasoning_error(exc):
                raise
            logger.warning(
                "llm.compatibility_adjustment api_mode=chat_completions model=%s "
                "reason=tools_with_reasoning_unsupported retry=1 streaming=true",
                self.model_name,
            )
            yield from self._retry_without_chat_reasoning()._stream(*args, **kwargs)
            return
        normalized = token_usage_from_provider(usage)
        if self.context_cost_governor is not None and governor_call_id:
            self.context_cost_governor.record_model_call(
                governor_call_id,
                usage=usage,
                provider=str(metadata.get("provider") or ""),
                model=self.model_name,
                estimated_input=[getattr(message, "content", "") for message in messages],
            )
        record_current(
            "model.call",
            {
                **metadata,
                "latency_seconds": time.perf_counter() - started,
                "usage": normalized.as_dict(),
                "streaming": True,
                "retry_attempt": int(self.compatibility_retry_attempted),
            },
        )

    def _eval_request_metadata(self, messages: list[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        roles = [str(getattr(message, "type", type(message).__name__)) for message in messages]
        prompt_material = [
            {"role": role, "content": str(getattr(message, "content", ""))}
            for role, message in zip(roles, messages)
        ]
        tools = []
        for item in kwargs.get("tools") or []:
            if isinstance(item, dict):
                function = item.get("function") if isinstance(item.get("function"), dict) else item
                tools.append(str(function.get("name") or ""))
        host = ""
        base = str(getattr(self, "openai_api_base", None) or getattr(self, "base_url", "") or "")
        if base:
            try:
                from urllib.parse import urlparse

                host = urlparse(base).netloc
            except Exception:
                host = ""
        return {
            "boundary": "compatible_chat_model",
            "provider": self.selected_provider or "unknown",
            "model": self.model_name,
            "prompt_hash": stable_hash(prompt_material),
            "safe_request_metadata": {
                "message_count": len(messages),
                "roles": roles,
                "tool_names": [name for name in tools if name],
                "tool_call_count": len([name for name in tools if name]),
                "streaming": bool(kwargs.get("stream")),
                "endpoint_host": host,
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
        for index, schema in enumerate(kwargs.get("tools") or []):
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
            provider=str(metadata.get("provider") or ""),
            step_id="compatibility-retry" if self.compatibility_retry_attempted else "",
            expected_output_tokens=(
                int(kwargs.get("max_output_tokens") or kwargs.get("max_completion_tokens") or kwargs.get("max_tokens"))
                if has_explicit_output_limit
                else None
            ),
            historical_prediction_enabled=not has_explicit_output_limit,
        )
        return call_id


def _apply_model_request_configuration(
    init_kwargs: dict[str, Any],
    configuration: dict[str, Any] | None,
) -> None:
    """Forward optional model/provider request fields without hardcoding vendors.

    Supported configuration keys (all optional):

    * ``extra_body`` – merged into the OpenAI-compatible request body
    * ``chat_template_kwargs`` – nested under ``extra_body`` (e.g. DeepSeek
      thinking controls on NVIDIA Build)
    * ``model_kwargs`` – merged into LangChain ``model_kwargs``
    * ``temperature``, ``max_tokens``, ``max_completion_tokens``,
      ``reasoning_effort`` – top-level client init fields when present
    """
    if not configuration:
        return
    extra_body: dict[str, Any] = {}
    raw_extra = configuration.get("extra_body")
    if isinstance(raw_extra, dict):
        extra_body.update(raw_extra)
    chat_template_kwargs = configuration.get("chat_template_kwargs")
    if isinstance(chat_template_kwargs, dict) and chat_template_kwargs:
        nested = dict(extra_body.get("chat_template_kwargs") or {})
        nested.update(chat_template_kwargs)
        extra_body["chat_template_kwargs"] = nested
    if extra_body:
        merged = dict(init_kwargs.get("extra_body") or {})
        merged.update(extra_body)
        init_kwargs["extra_body"] = merged
    model_kwargs = configuration.get("model_kwargs")
    if isinstance(model_kwargs, dict) and model_kwargs:
        existing = dict(init_kwargs.get("model_kwargs") or {})
        existing.update(model_kwargs)
        init_kwargs["model_kwargs"] = existing
    for key in (
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "reasoning_effort",
        "top_p",
    ):
        if key in configuration and configuration[key] is not None and key not in init_kwargs:
            init_kwargs[key] = configuration[key]


def create_chat_model(
    *,
    api_key: str,
    model: str,
    base_url: str | None = None,
    provider: str | None = None,
    default_headers: dict[str, str] | None = None,
    model_configuration: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CompatibleChatOpenAI:
    """Create the shared compatibility-aware LLM client used by every runtime role."""

    configuration = model_configuration
    if "model_configuration" in kwargs:
        configuration = kwargs.pop("model_configuration")
    api_mode, capabilities = resolve_model_capabilities(base_url=base_url, provider=provider)
    # NVIDIA and other Chat Completions-first hosts force chat mode when auto.
    if str(provider or "").strip().lower() == "nvidia" and api_mode == "auto":
        if not capabilities.supports_responses_api:
            api_mode = "chat_completions"
    reasoning_effort = str(get_setting("MANA_LLM_REASONING_EFFORT", "") or "").strip() or None
    init_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "model": model,
        "compatibility_api_mode": api_mode,
        "compatibility_capabilities": capabilities,
        "selected_provider": str(provider or "unknown"),
        **kwargs,
    }
    if base_url:
        init_kwargs["base_url"] = base_url
    if default_headers:
        init_kwargs["default_headers"] = default_headers
    # NVIDIA DeepSeek expects chat_template_kwargs rather than top-level
    # reasoning_effort. Attach defaults here; per-request shaping in
    # _get_request_payload remains authoritative.
    from mana_agent.config.nvidia_model_requests import (
        deepseek_chat_template_kwargs,
        is_nvidia_deepseek_model,
    )

    if is_nvidia_deepseek_model(provider=provider, model=model):
        template = deepseek_chat_template_kwargs(reasoning_effort or "high")
        extra_body = dict(init_kwargs.get("extra_body") or {})
        nested = dict(extra_body.get("chat_template_kwargs") or {})
        for key, value in template.items():
            nested.setdefault(key, value)
        extra_body["chat_template_kwargs"] = nested
        init_kwargs["extra_body"] = extra_body
    elif reasoning_effort and "reasoning_effort" not in init_kwargs and "reasoning" not in init_kwargs:
        init_kwargs["reasoning_effort"] = reasoning_effort
    if isinstance(configuration, dict):
        _apply_model_request_configuration(init_kwargs, configuration)
    return CompatibleChatOpenAI(**init_kwargs)


def format_provider_error(exc: BaseException, *, provider: str | None = None, model: str | None = None) -> str:
    """Return a user-facing error that never claims the wrong provider."""
    provider_id = str(provider or "unknown").strip().lower() or "unknown"
    try:
        from mana_agent.config.provider_registry import PROVIDERS

        label = PROVIDERS.get(provider_id).display_name
    except Exception:
        label = provider_id if provider_id != "unknown" else "Inference provider"
    text = str(exc)
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    code = int(status) if status is not None else None
    model_part = f" model={model}" if model else ""
    if code in {401, 403}:
        isolation = (
            " NVIDIA credentials are isolated from OPENAI_API_KEY."
            if provider_id == "nvidia"
            else ""
        )
        return (
            f"{label} authentication or permission failed (HTTP {code}).{model_part}"
            f"{isolation}"
        )
    if code in {404, 410}:
        return (
            f"{label} endpoint or model is unavailable (HTTP {code}).{model_part} "
            "Confirm the model id is enabled for this account."
        )
    if code == 429:
        return f"{label} rate limit or quota exceeded (HTTP 429).{model_part}"
    if code is not None and code >= 500:
        return f"{label} service failure (HTTP {code}).{model_part}"
    lowered = text.lower()
    if "timeout" in lowered or "timed out" in lowered:
        return f"{label} request timed out.{model_part}"
    if "model" in lowered and any(token in lowered for token in ("not found", "does not exist", "invalid")):
        return f"{label} selected model is unavailable.{model_part}"
    # Never rebrand a multi-provider failure as an OpenAI error.
    if "openai" in text.lower() and provider_id not in {"openai", "unknown", ""}:
        return f"{label} request failed.{model_part} {type(exc).__name__}"
    # Prefer status-aware short message when the exception body is noisy.
    if code is not None:
        return f"{label} request failed (HTTP {code}).{model_part}"
    return f"{label} request failed.{model_part} {type(exc).__name__}: {text}"
