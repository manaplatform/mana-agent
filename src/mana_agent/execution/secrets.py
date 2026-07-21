"""Secret references and output redaction at the provider boundary."""

from __future__ import annotations

import os
from typing import Protocol

from mana_agent.execution.errors import ProviderConfigurationError


class SecretResolver(Protocol):
    def resolve(self, reference: str) -> str: ...


class EnvironmentSecretResolver:
    """Resolve a reference from process environment without persisting values."""

    def resolve(self, reference: str) -> str:
        value = os.environ.get(reference)
        if value is None:
            raise ProviderConfigurationError(f"secret reference is unavailable: {reference}")
        return value


def redact_values(text: str, values: list[str]) -> str:
    redacted = text
    for value in sorted((item for item in values if item), key=len, reverse=True):
        redacted = redacted.replace(value, "[REDACTED]")
    return redacted
