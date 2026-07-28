"""Recorder orchestration and a functional semantic event ingestion adapter."""

from __future__ import annotations

from collections.abc import Callable

from .models import RecordedEvent, TeachError, TeachSession


class SemanticEventRecorder:
    """Receives semantic events from browser/computer-control integrations.

    Platform integrations call ``capture``; raw keystrokes and screenshots are
    intentionally not collected by this adapter.
    """

    recorder_id = "semantic-event-recorder"

    def __init__(self):
        self._session: TeachSession | None = None
        self._emit: Callable[[RecordedEvent], None] | None = None
        self._paused = False

    def capabilities(self) -> set[str]:
        return {"semantic_events", "pause_resume", "sensitive_masking"}

    def start(self, session: TeachSession, emit: Callable[[RecordedEvent], None]) -> None:
        self._session = session
        self._emit = emit
        self._paused = False

    def capture(self, event: RecordedEvent) -> None:
        if self._session is None or self._emit is None:
            raise TeachError("Recorder is not active.")
        if self._paused:
            return
        if event.session_id != self._session.id:
            raise TeachError("Recorded event belongs to a different session.")
        self._emit(event)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._session = None
        self._emit = None
        self._paused = False
