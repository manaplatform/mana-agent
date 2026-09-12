"""Mana-Agent Live Mode — realtime voice interaction via OpenAI Realtime API.

This package provides an optional `mana-agent live` CLI command that enables
realtime voice/audio interaction. Live mode reuses the existing Mana-Agent
gateway, sessions, identity, memory, routing, task system, permissions,
approvals, tools, specialist lanes, and Codex coding engine.

The realtime interaction model (e.g. gpt-4o-realtime-preview) is used only
for Live mode and is never added to the normal model routing candidate pool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adapter import LiveAdapter
    from .audio_transport import AudioTransport
    from .config import LiveConfig
    from .models import LiveEvent, LiveEventType, LiveState, LiveTranscript, LiveUsage
    from .runner import LiveRunner
    from .session import LiveSession

__all__ = [
    "LiveAdapter",
    "AudioTransport",
    "LiveConfig",
    "LiveEvent",
    "LiveEventType",
    "LiveRunner",
    "LiveSession",
    "LiveState",
    "LiveTranscript",
    "LiveUsage",
]
