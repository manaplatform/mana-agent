"""Centralized backend selection."""

from __future__ import annotations

from pathlib import Path

from mana_agent.memory.config import MemoryConfig
from mana_agent.memory.contracts import MemoryBackend


def create_memory_backend(config: MemoryConfig, *, root: str | Path) -> MemoryBackend:
    validated = config.validate()
    if validated.mode == "internal" and validated.provider == "mana":
        from mana_agent.memory.providers.internal.backend import InternalMemoryBackend

        return InternalMemoryBackend(root)
    if validated.mode == "external" and validated.provider == "mem0":
        from mana_agent.memory.providers.mem0.backend import Mem0MemoryBackend

        return Mem0MemoryBackend(validated)
    raise AssertionError("validated memory configuration was not handled")
