"""Shared, frontend-neutral chat command bus."""

from .dispatcher import CommandDispatcher
from .models import CommandContext, CommandDefinition, CommandResult
from .registry import CommandRegistry, build_default_registry

__all__ = [
    "CommandContext",
    "CommandDefinition",
    "CommandDispatcher",
    "CommandRegistry",
    "CommandResult",
    "build_default_registry",
]
