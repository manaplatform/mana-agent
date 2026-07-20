"""Stable, user-safe protocol errors."""

from __future__ import annotations


class ProtocolError(RuntimeError):
    """Base error safe to serialize at a protocol boundary."""


class ProtocolPolicyError(ProtocolError):
    """A validated protocol request was denied by local policy."""


class ProtocolAuthenticationError(ProtocolError):
    """A caller could not be authenticated."""


class OptionalProtocolDependencyError(ProtocolError):
    """An optional SDK is unavailable."""

    @classmethod
    def for_protocol(cls, protocol: str) -> "OptionalProtocolDependencyError":
        name = str(protocol).strip().lower()
        return cls(
            f"{name.upper()} support is not installed. "
            f"Install mana-agent with the `{name}` extra."
        )
