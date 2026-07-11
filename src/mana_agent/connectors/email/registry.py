from __future__ import annotations
from collections.abc import Callable
from mana_agent.connectors.email.providers.base import EmailProvider

class EmailProviderRegistry:
    def __init__(self) -> None: self._factories: dict[str, Callable[..., EmailProvider]] = {}
    def register(self, provider: str, factory: Callable[..., EmailProvider]) -> None:
        if not provider or provider in self._factories: raise ValueError(f"Email provider already registered or invalid: {provider}")
        self._factories[provider] = factory
    def create(self, provider: str, **kwargs: object) -> EmailProvider:
        try: return self._factories[provider](**kwargs)
        except KeyError as exc: raise ValueError(f"Unsupported email provider: {provider}") from exc
    def providers(self) -> tuple[str, ...]: return tuple(sorted(self._factories))

registry = EmailProviderRegistry()
