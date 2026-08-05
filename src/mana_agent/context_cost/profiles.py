"""Authoritative model identity and token-capability profile resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping


def _decimal(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed >= 0 else None


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    provider: str
    model: str
    alias: str | None = None
    deployment_id: str | None = None

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model.strip():
            raise ValueError("model identity requires provider and model")

    @property
    def key(self) -> str:
        return f"{self.provider.casefold()}/{self.model}"


@dataclass(frozen=True, slots=True)
class ModelTokenProfile:
    identity: ModelIdentity
    context_window: int
    max_output_tokens: int
    tokenizer: str | None = None
    input_price_per_million: Decimal | None = None
    cached_input_price_per_million: Decimal | None = None
    output_price_per_million: Decimal | None = None
    reasoning_price_per_million: Decimal | None = None
    supports_usage_reporting: bool = False
    source: str = "unknown"
    confidence: str = "low"
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.context_window < 1 or self.max_output_tokens < 1:
            raise ValueError("model token limits must be positive")
        if self.max_output_tokens > self.context_window:
            raise ValueError("max_output_tokens cannot exceed context_window")


class UnknownModelProfileError(ValueError):
    pass


class ModelTokenProfileResolver:
    """Resolve only the identity selected by routing; never select another model."""

    def __init__(
        self,
        sources: Iterable[Any] = (),
        *,
        unknown_policy: str = "require_metadata",
        unknown_context_window: int | None = None,
        unknown_max_output_tokens: int | None = None,
    ) -> None:
        self.unknown_policy = str(unknown_policy or "require_metadata")
        self.unknown_context_window = unknown_context_window
        self.unknown_max_output_tokens = unknown_max_output_tokens
        self._profiles: dict[str, ModelTokenProfile] = {}
        self.register(sources)

    def register(self, sources: Iterable[Any]) -> None:
        for source in sources:
            profile = self._coerce(source)
            if profile is not None:
                self._profiles[profile.identity.key] = profile

    def resolve(self, identity: ModelIdentity) -> ModelTokenProfile:
        direct = self._profiles.get(identity.key)
        if direct is not None:
            return direct
        matches = [item for item in self._profiles.values() if item.identity.model == identity.model]
        if len(matches) == 1 and matches[0].identity.provider.casefold() == identity.provider.casefold():
            return matches[0]
        if self.unknown_policy != "conservative" or not self.unknown_context_window or not self.unknown_max_output_tokens:
            raise UnknownModelProfileError(
                f"Model token metadata is unavailable for {identity.key}. No arbitrary context or output limit was used."
            )
        return ModelTokenProfile(
            identity=identity,
            context_window=int(self.unknown_context_window),
            max_output_tokens=int(self.unknown_max_output_tokens),
            source="configured-unknown-model-policy",
            confidence="low",
            metadata={"unknown_model": True},
        )

    @staticmethod
    def _coerce(source: Any) -> ModelTokenProfile | None:
        if isinstance(source, ModelTokenProfile):
            return source
        provider = str(getattr(source, "provider", "") or "").strip()
        model = str(getattr(source, "model_id", "") or getattr(source, "id", "") or "").strip()
        if not provider or not model:
            return None
        metadata = dict(getattr(source, "metadata", {}) or {})
        configuration = dict(getattr(source, "configuration", {}) or {})
        combined = {**metadata, **configuration}
        context = getattr(source, "context_window", None) or combined.get("context_length") or combined.get("context_window")
        output = getattr(source, "max_output_tokens", None) or combined.get("max_output_tokens") or combined.get("max_completion_tokens")
        if context in (None, 0) or output in (None, 0):
            return None
        pricing = combined.get("pricing") if isinstance(combined.get("pricing"), Mapping) else {}
        return ModelTokenProfile(
            identity=ModelIdentity(provider, model, alias=combined.get("alias"), deployment_id=combined.get("deployment_id")),
            context_window=int(context),
            max_output_tokens=int(output),
            tokenizer=str(getattr(source, "tokenizer", None) or combined.get("tokenizer") or "") or None,
            input_price_per_million=_decimal(getattr(source, "input_cost_per_million", None) or combined.get("input_price_per_million") or pricing.get("input_per_million")),
            cached_input_price_per_million=_decimal(getattr(source, "cached_input_cost_per_million", None) or combined.get("cached_input_price_per_million") or pricing.get("cached_input_per_million")),
            output_price_per_million=_decimal(getattr(source, "output_cost_per_million", None) or combined.get("output_price_per_million") or pricing.get("output_per_million")),
            reasoning_price_per_million=_decimal(getattr(source, "reasoning_cost_per_million", None) or combined.get("reasoning_price_per_million") or pricing.get("reasoning_per_million")),
            supports_usage_reporting=bool(combined.get("supports_usage_reporting", True)),
            source=str(getattr(source, "source", "") or getattr(source, "source_level", "") or "configured"),
            confidence=str(combined.get("token_profile_confidence") or "high"),
            metadata=combined,
        )


__all__ = ["ModelIdentity", "ModelTokenProfile", "ModelTokenProfileResolver", "UnknownModelProfileError"]
