"""Typed data models for Mana-Agent Live mode."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class LiveState(str, Enum):
    """Observable lifecycle states for the Live session."""
    INITIALIZING = "live initializing"
    CONNECTING = "live connecting"
    LISTENING = "live listening"
    THINKING = "live thinking"
    DELEGATING = "live delegating"
    SPEAKING = "live speaking"
    INTERRUPTED = "live interrupted"
    RECONNECTING = "live reconnecting"
    ERROR = "live error"
    SHUTDOWN = "live shutdown"


class LiveEventType(str, Enum):
    """Event types emitted by the Live integration layer."""
    STATE_CHANGED = "live.state_changed"
    TRANSCRIPT_USER = "live.transcript.user"
    TRANSCRIPT_ASSISTANT = "live.transcript.assistant"
    DELEGATION_STARTED = "live.delegation.started"
    DELEGATION_COMPLETED = "live.delegation.completed"
    DELEGATION_ERROR = "live.delegation.error"
    INTERRUPTION = "live.interruption"
    SESSION_CREATED = "live.session.created"
    SESSION_ENDED = "live.session.ended"
    RECONNECT_ATTEMPT = "live.reconnect.attempt"
    AUDIO_ERROR = "live.audio.error"
    USAGE_UPDATE = "live.usage.update"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class LiveEvent:
    """An event emitted by the Live integration layer."""
    event_type: LiveEventType
    timestamp: datetime = field(default_factory=_utc_now)
    data: dict[str, Any] = field(default_factory=dict)
    task_id: str = ""
    state: LiveState | None = None


@dataclass(frozen=True, slots=True)
class LiveTranscript:
    """A single transcript entry from the Live conversation."""
    role: str  # "user" or "assistant"
    text: str
    timestamp: datetime = field(default_factory=_utc_now)
    audio_duration_ms: int = 0
    is_partial: bool = False
    transcript_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass(frozen=True, slots=True)
class LiveUsage:
    """Cumulative usage metrics for the Live session."""
    input_tokens: int = 0
    output_tokens: int = 0
    audio_input_ms: int = 0
    audio_output_ms: int = 0
    session_duration_ms: int = 0


@dataclass(frozen=True, slots=True)
class DelegationRequest:
    """A work request routed from GPT-Live to the Mana gateway."""
    text: str
    intent: str = ""
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class DelegationResult:
    """The result of a Mana gateway delegation."""
    correlation_id: str
    text: str
    success: bool = True
    task_id: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
