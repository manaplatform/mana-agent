from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0
    tool_result_tokens: int = 0
    estimated: bool = False
    provider: str | None = None
    model: str | None = None
    provider_raw_usage: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
            "cache_creation_tokens",
            "reasoning_tokens",
            "tool_result_tokens",
        ):
            value = max(0, int(getattr(self, name, 0) or 0))
            setattr(self, name, value)
        if self.total_tokens <= 0:
            self.total_tokens = (
                self.input_tokens
                + self.output_tokens
                + self.reasoning_tokens
                + self.tool_result_tokens
            )

    def add(self, other: "TokenUsage") -> "TokenUsage":
        raw = dict(self.provider_raw_usage)
        if other.provider_raw_usage:
            raw.setdefault("parts", [])
            if isinstance(raw["parts"], list):
                raw["parts"].append(other.provider_raw_usage)
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            tool_result_tokens=self.tool_result_tokens + other.tool_result_tokens,
            estimated=bool(self.estimated or other.estimated),
            provider=other.provider or self.provider,
            model=other.model or self.model,
            provider_raw_usage=raw,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "tool_result_tokens": self.tool_result_tokens,
            "estimated": self.estimated,
            "provider": self.provider,
            "model": self.model,
            "provider_raw_usage": dict(self.provider_raw_usage),
        }


def estimate_tokens(text: Any) -> int:
    # Import lazily because context_cost's public package exports include the
    # governor, which consumes TokenUsage during package initialization.
    from mana_agent.context_cost.tokenizers import TokenizerRegistry

    tokenizer, _ = TokenizerRegistry().resolve(None)
    return tokenizer.count(text)


def token_usage_from_provider(usage: Any) -> TokenUsage:
    """Normalize provider usage without inventing exact numbers."""
    from mana_agent.context_cost.usage import normalize_provider_usage

    normalized = normalize_provider_usage(usage)
    if normalized is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=normalized.input_tokens,
        output_tokens=normalized.output_tokens,
        total_tokens=normalized.total_tokens,
        cached_input_tokens=normalized.cached_input_tokens,
        reasoning_tokens=normalized.reasoning_tokens,
        estimated=normalized.status != "reported",
        provider_raw_usage=dict(normalized.provider_fields),
    )


@dataclass(slots=True)
class TokenUsageTracker:
    by_model_call: dict[str, TokenUsage] = field(default_factory=dict)
    by_tool_result: dict[str, TokenUsage] = field(default_factory=dict)
    by_agent: dict[str, TokenUsage] = field(default_factory=dict)
    by_subagent: dict[str, TokenUsage] = field(default_factory=dict)
    by_step: dict[str, TokenUsage] = field(default_factory=dict)
    by_turn: dict[str, TokenUsage] = field(default_factory=dict)
    by_provider_model: dict[str, TokenUsage] = field(default_factory=dict)
    session_total: TokenUsage = field(default_factory=TokenUsage)
    current_turn_id: str = ""

    def start_turn(self, turn_id: str) -> None:
        self.current_turn_id = str(turn_id or "")
        self.by_turn.setdefault(self.current_turn_id, TokenUsage())

    def record_model_call(
        self,
        call_id: str,
        *,
        usage: Any = None,
        provider: str = "",
        model: str = "",
        agent_id: str = "main",
        subagent_id: str | None = None,
        step_id: str = "",
        turn_id: str = "",
        estimated_text: str = "",
    ) -> TokenUsage:
        token_usage = token_usage_from_provider(usage)
        if token_usage.total_tokens <= 0 and estimated_text:
            token_usage = TokenUsage(input_tokens=estimate_tokens(estimated_text), estimated=True)
        token_usage.provider = provider or token_usage.provider
        token_usage.model = model or token_usage.model
        from mana_agent.evals.recorder import record_current

        record_current(
            "model.usage",
            {
                "call_id": call_id,
                "provider": token_usage.provider,
                "model": token_usage.model,
                "agent_id": agent_id,
                "subagent_id": subagent_id,
                "step_id": step_id,
                "turn_id": turn_id or self.current_turn_id,
                "usage": token_usage.as_dict(),
            },
        )
        self._record(
            token_usage,
            bucket=self.by_model_call,
            key=call_id,
            provider=provider,
            model=model,
            agent_id=agent_id,
            subagent_id=subagent_id,
            step_id=step_id,
            turn_id=turn_id,
        )
        return token_usage

    def record_tool_result(
        self,
        tool_call_id: str,
        result: Any,
        *,
        agent_id: str = "main",
        subagent_id: str | None = None,
        step_id: str = "",
        turn_id: str = "",
    ) -> TokenUsage:
        token_usage = TokenUsage(tool_result_tokens=estimate_tokens(result), estimated=True)
        self._record(
            token_usage,
            bucket=self.by_tool_result,
            key=tool_call_id,
            agent_id=agent_id,
            subagent_id=subagent_id,
            step_id=step_id,
            turn_id=turn_id,
        )
        return token_usage

    def record_step(self, step_id: str, usage: TokenUsage) -> None:
        self._record(usage, bucket=self.by_step, key=step_id, step_id=step_id)

    def _record(
        self,
        usage: TokenUsage,
        *,
        bucket: dict[str, TokenUsage],
        key: str,
        provider: str = "",
        model: str = "",
        agent_id: str = "",
        subagent_id: str | None = None,
        step_id: str = "",
        turn_id: str = "",
    ) -> None:
        key = str(key or "unknown")
        bucket[key] = bucket.get(key, TokenUsage()).add(usage)
        self.session_total = self.session_total.add(usage)
        if agent_id:
            self.by_agent[agent_id] = self.by_agent.get(agent_id, TokenUsage()).add(usage)
        if subagent_id:
            self.by_subagent[subagent_id] = self.by_subagent.get(subagent_id, TokenUsage()).add(usage)
        if step_id:
            self.by_step[step_id] = self.by_step.get(step_id, TokenUsage()).add(usage)
        effective_turn = str(turn_id or self.current_turn_id or "")
        if effective_turn:
            self.by_turn[effective_turn] = self.by_turn.get(effective_turn, TokenUsage()).add(usage)
        if provider or model:
            provider_key = f"{provider or 'provider'}:{model or 'model'}"
            self.by_provider_model[provider_key] = self.by_provider_model.get(provider_key, TokenUsage()).add(usage)

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_total": self.session_total.as_dict(),
            "current_turn_id": self.current_turn_id,
            "by_turn": {key: value.as_dict() for key, value in self.by_turn.items()},
            "by_agent": {key: value.as_dict() for key, value in self.by_agent.items()},
            "by_subagent": {key: value.as_dict() for key, value in self.by_subagent.items()},
            "by_step": {key: value.as_dict() for key, value in self.by_step.items()},
            "by_provider_model": {key: value.as_dict() for key, value in self.by_provider_model.items()},
            "by_tool_result": {key: value.as_dict() for key, value in self.by_tool_result.items()},
        }
