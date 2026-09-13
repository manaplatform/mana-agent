"""Adapter bridging GPT-Live and the existing Mana-Agent gateway.

The adapter owns only Live-mode concerns:

- stream microphone audio to GPT-Live;
- play GPT-Live output audio;
- accumulate Live transcript deltas;
- handle native GPT-Live client delegation;
- route delegated work through the existing AgentChatGateway;
- return authoritative backend results with session.commentary.append;
- handle local barge-in without cancelling backend tasks;
- emit Live events for the TUI.

It does not create a second routing, tool, memory, approval, or coding system.
"""

from __future__ import annotations

import asyncio
import base64
from collections import deque
from contextlib import suppress
import inspect
import logging
from typing import Any, Callable, Deque

from mana_agent.live.models import (
    DelegationRequest,
    DelegationResult,
    LiveEvent,
    LiveEventType,
    LiveState,
    LiveTranscript,
    LiveUsage,
)

logger = logging.getLogger(__name__)

# Transcript fragments are timestamped on the Live session timeline. A small
# allowance helps include final transcription fragments that arrive immediately
# around the delegation event.
_DELEGATION_TRANSCRIPT_GRACE_MS = 1_000
_MAX_TRANSCRIPT_FRAGMENTS = 1_024


class LiveAdapter:
    """Bridge GPT-Live realtime events to the normal Mana-Agent gateway."""

    def __init__(
        self,
        *,
        session: Any,  # LiveSession
        audio_transport: Any,  # AudioTransport
        gateway: Any,  # AgentChatGateway
        gateway_session_id: str,
        event_callback: Callable[[LiveEvent], None] | None = None,
    ) -> None:
        self._session = session
        self._audio = audio_transport
        self._gateway = gateway
        self._gateway_session_id = gateway_session_id
        self._event_cb = event_callback

        self._state = LiveState.INITIALIZING
        self._transcripts: list[LiveTranscript] = []
        self._usage = LiveUsage()

        self._running = False

        # Exact GPT-Live delegation ID currently being processed.
        self._active_delegation: str | None = None

        # Delegations are background tasks so voice input/output remains live
        # while Mana-Agent is working.
        self._delegation_tasks: dict[str, asyncio.Task[None]] = {}

        # AgentChatGateway is shared state. Serialize backend turns even though
        # individual delegation jobs run outside the Live receive loop.
        self._gateway_lock = asyncio.Lock()

        # GPT-Live transcript events are deltas and have no transcript-done
        # event. Keep display buffers and timestamped user fragments.
        self._user_display_buffer = ""
        self._assistant_display_buffer = ""

        self._input_fragments: Deque[tuple[int, int, str]] = deque(
            maxlen=_MAX_TRANSCRIPT_FRAGMENTS
        )

        # Request boundaries on the Live timeline.
        self._last_assistant_activity_ms = -1
        self._last_delegation_offset_ms = -1

        # Prevent repeated local barge-in handling for the same user utterance.
        self._interrupting = False

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> LiveState:
        return self._state

    @property
    def transcripts(self) -> list[LiveTranscript]:
        return list(self._transcripts)

    @property
    def usage(self) -> LiveUsage:
        return self._usage

    # ------------------------------------------------------------------
    # Main event loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Receive GPT-Live events and stream microphone audio concurrently."""
        if self._running:
            return

        self._running = True
        self._set_state(LiveState.LISTENING)

        audio_sender = asyncio.create_task(
            self._audio_send_loop(),
            name="mana-live-audio-sender",
        )

        try:
            async for event in self._session.receive_events():
                if not self._running:
                    break

                await self._dispatch_event(event)

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            logger.exception(
                "GPT-Live adapter event loop failed: %s",
                exc,
            )
            self._set_state(LiveState.ERROR)
            self._emit(
                LiveEventType.AUDIO_ERROR,
                data={"error": str(exc)},
            )

        finally:
            self._running = False

            audio_sender.cancel()
            with suppress(asyncio.CancelledError):
                await audio_sender

            self._flush_user_transcript()
            self._flush_assistant_transcript()

    async def shutdown(self) -> None:
        """Stop realtime adapter work without cancelling completed Mana tasks."""
        if not self._running and not self._delegation_tasks:
            self._set_state(LiveState.SHUTDOWN)
            return

        self._running = False
        self._set_state(LiveState.SHUTDOWN)

        # Explicit Live-mode shutdown may stop in-flight delegation wrapper
        # tasks, but ordinary user speech / barge-in never does.
        tasks = list(self._delegation_tasks.values())

        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

        self._delegation_tasks.clear()
        self._active_delegation = None

        self._flush_user_transcript()
        self._flush_assistant_transcript()

    # ------------------------------------------------------------------
    # Audio input
    # ------------------------------------------------------------------

    async def _audio_send_loop(self) -> None:
        """Continuously send captured raw audio to GPT-Live."""
        while self._running:
            chunk = self._audio.get_capture_chunk()

            if chunk:
                try:
                    await self._session.send_audio(chunk)

                except asyncio.CancelledError:
                    raise

                except Exception:
                    logger.debug(
                        "Failed to send GPT-Live audio chunk",
                        exc_info=True,
                    )
                    await asyncio.sleep(0.02)

            else:
                await asyncio.sleep(0.02)

    # ------------------------------------------------------------------
    # GPT-Live event dispatch
    # ------------------------------------------------------------------

    async def _dispatch_event(
        self,
        event: dict[str, Any],
    ) -> None:
        """Route a current GPT-Live server event."""
        event_type = str(event.get("type") or "")

        handler = {
            "session.started": self._handle_session_started,
            "session.updated": self._handle_session_updated,
            "session.input_transcript.delta": self._handle_input_transcript_delta,
            "session.output_transcript.delta": self._handle_output_transcript_delta,
            "session.output_audio.delta": self._handle_output_audio_delta,
            "session.delegation.created": self._handle_delegation_created,
            "session.closed": self._handle_session_closed,
            "input_audio_buffer.speech_started": self._handle_speech_started,
            "error": self._handle_error,
        }.get(event_type)

        if handler is not None:
            await handler(event)
            return

        # The Live API emits additional acknowledgements such as commentary,
        # thinking, instruction, mute/unmute, etc. They do not need adapter
        # behavior unless Mana starts using those features directly.
        logger.debug(
            "Ignoring GPT-Live event type: %s",
            event_type or "<missing>",
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _handle_session_started(
        self,
        event: dict[str, Any],
    ) -> None:
        session_data = event.get("session")
        if not isinstance(session_data, dict):
            session_data = {}

        session_id = str(session_data.get("id") or "")

        self._emit(
            LiveEventType.SESSION_CREATED,
            data={"session_id": session_id},
        )

        logger.info(
            "GPT-Live session started: %s",
            session_id or "<unknown>",
        )

    async def _handle_session_updated(
        self,
        event: dict[str, Any],
    ) -> None:
        """Acknowledge accepted mutable Live session settings."""
        logger.debug("GPT-Live session updated")

    async def _handle_session_closed(
        self,
        event: dict[str, Any],
    ) -> None:
        """Finalize transcript/display state from terminal Live event."""
        self._flush_user_transcript()
        self._flush_assistant_transcript()

        reason = str(event.get("reason") or "")

        usage = event.get("usage")
        seconds = 0.0

        if isinstance(usage, dict):
            try:
                seconds = float(usage.get("seconds") or 0.0)
            except (TypeError, ValueError):
                seconds = 0.0

        # GPT-Live session usage is cumulative audio duration, not separate
        # input/output token counts. Keep LiveUsage token values unchanged and
        # expose the authoritative duration in the event data.
        self._emit(
            LiveEventType.USAGE_UPDATE,
            data={
                "usage": {
                    "seconds": seconds,
                    "input_tokens": self._usage.input_tokens,
                    "output_tokens": self._usage.output_tokens,
                }
            },
        )

        logger.info(
            "GPT-Live session closed: reason=%s audio_seconds=%.3f",
            reason or "<unknown>",
            seconds,
        )

        self._running = False

    # ------------------------------------------------------------------
    # Transcript / interruption handling
    # ------------------------------------------------------------------

    async def _handle_speech_started(
        self,
        event: dict[str, Any],
    ) -> None:
        """Handle incoming user speech interruption from server VAD."""
        try:
            self._audio.interrupt_playback()
        except Exception:
            logger.debug(
                "Failed to interrupt local playback",
                exc_info=True,
            )

        try:
            await self._session.truncate_response()
        except Exception:
            logger.debug(
                "Failed to truncate session response",
                exc_info=True,
            )

        self._emit(
            LiveEventType.INTERRUPTION,
            data=event.get("data") or {"reason": "speech_started"},
        )

    async def _handle_input_transcript_delta(
        self,
        event: dict[str, Any],
    ) -> None:
        """Accumulate one user transcript fragment.

        GPT-Live does not provide an input transcript "done" event. Deltas must
        be accumulated in delivery order.
        """
        delta = event.get("delta")

        if not isinstance(delta, str) or not delta:
            return

        start_ms = self._event_ms(event.get("start_ms"))
        end_ms = self._event_ms(event.get("end_ms"), fallback=start_ms)

        # New user speech while output is playing is a local barge-in. Stop
        # queued speaker audio, but never cancel an active Mana backend task.
        if self._state == LiveState.SPEAKING and not self._interrupting:
            self._interrupting = True
            self._set_state(LiveState.INTERRUPTED)

            try:
                self._audio.interrupt_playback()
            except Exception:
                logger.debug(
                    "Failed to interrupt local playback",
                    exc_info=True,
                )

            self._emit(
                LiveEventType.INTERRUPTION,
                data={"reason": "user_speech"},
            )

            self._set_state(LiveState.LISTENING)

        # Assistant text before this point belongs to the prior response.
        self._flush_assistant_transcript()

        self._user_display_buffer += delta
        self._input_fragments.append(
            (start_ms, end_ms, delta)
        )

    async def _handle_output_transcript_delta(
        self,
        event: dict[str, Any],
    ) -> None:
        """Accumulate one assistant transcript fragment."""
        delta = event.get("delta")

        if not isinstance(delta, str) or not delta:
            return

        start_ms = self._event_ms(event.get("start_ms"))
        end_ms = self._event_ms(event.get("end_ms"), fallback=start_ms)

        # First assistant output marks the previous user utterance boundary.
        self._flush_user_transcript()

        self._assistant_display_buffer += delta
        self._interrupting = False

        self._last_assistant_activity_ms = max(
            self._last_assistant_activity_ms,
            end_ms,
        )

        if self._state != LiveState.SPEAKING:
            self._set_state(LiveState.SPEAKING)

    # ------------------------------------------------------------------
    # Audio output
    # ------------------------------------------------------------------

    async def _handle_output_audio_delta(
        self,
        event: dict[str, Any],
    ) -> None:
        """Decode and queue a GPT-Live output-audio chunk in delivery order."""
        encoded = event.get("delta")

        if not isinstance(encoded, str) or not encoded:
            return

        try:
            audio_bytes = base64.b64decode(
                encoded,
                validate=True,
            )

        except Exception as exc:
            logger.warning(
                "Invalid GPT-Live output audio delta: %s",
                exc,
            )
            self._emit(
                LiveEventType.AUDIO_ERROR,
                data={"error": str(exc)},
            )
            return

        if not audio_bytes:
            return

        if self._state != LiveState.SPEAKING:
            self._set_state(LiveState.SPEAKING)

        try:
            # AudioTransport exposes a synchronous playback queue. Preserve
            # exact server delivery order by enqueueing from this single event
            # consumer.
            self._audio.queue_playback(audio_bytes)

        except Exception as exc:
            logger.exception(
                "Failed to queue GPT-Live output audio"
            )
            self._emit(
                LiveEventType.AUDIO_ERROR,
                data={"error": str(exc)},
            )

    # ------------------------------------------------------------------
    # Native GPT-Live client delegation
    # ------------------------------------------------------------------

    async def _handle_delegation_created(
        self,
        event: dict[str, Any],
    ) -> None:
        """Start a native client-delegation job without blocking Live audio."""
        delegation = event.get("delegation")

        if not isinstance(delegation, dict):
            logger.warning(
                "GPT-Live delegation event missing delegation object"
            )
            return

        delegation_id = str(
            delegation.get("id") or ""
        )
        target = str(
            delegation.get("target") or ""
        )

        if not delegation_id:
            logger.warning(
                "GPT-Live delegation event missing delegation.id"
            )
            return

        if target != "client":
            # Mana-Agent only owns native client delegations.
            logger.debug(
                "Ignoring non-client GPT-Live delegation %s target=%s",
                delegation_id,
                target,
            )
            return

        if delegation_id in self._delegation_tasks:
            logger.debug(
                "Ignoring duplicate GPT-Live delegation: %s",
                delegation_id,
            )
            return

        offset_ms = self._event_ms(
            event.get("offset_ms")
        )

        # Flush for the TUI, but request reconstruction uses timestamped
        # fragments so display flushing does not lose backend context.
        self._flush_user_transcript()

        task = asyncio.create_task(
            self._run_client_delegation(
                delegation_id=delegation_id,
                offset_ms=offset_ms,
            ),
            name=f"mana-live-delegation-{delegation_id}",
        )

        self._delegation_tasks[delegation_id] = task

        def _done(
            completed: asyncio.Task[None],
            *,
            delegation_id: str = delegation_id,
        ) -> None:
            self._delegation_tasks.pop(
                delegation_id,
                None,
            )

            if completed.cancelled():
                return

            exc = completed.exception()
            if exc is not None:
                logger.error(
                    "Unhandled GPT-Live delegation task failure %s: %s",
                    delegation_id,
                    exc,
                )

        task.add_done_callback(_done)

    async def _run_client_delegation(
        self,
        *,
        delegation_id: str,
        offset_ms: int,
    ) -> None:
        """Reconstruct delegated user text and execute it through Mana."""
        # Transcript deltas close to delegation creation may still be arriving.
        # Yield briefly while the receive loop continues collecting them.
        await asyncio.sleep(0.15)

        request_text = self._request_text_for_delegation(
            offset_ms
        )

        if not request_text:
            message = (
                "I could not reliably recover the spoken request for this "
                "delegation. Please ask the user to repeat it."
            )

            logger.warning(
                "No transcript available for GPT-Live delegation %s",
                delegation_id,
            )

            await self._session.send_commentary(
                message,
                delegation_id=delegation_id,
            )

            self._emit(
                LiveEventType.DELEGATION_ERROR,
                data={
                    "correlation_id": delegation_id,
                    "error": "delegation transcript unavailable",
                },
            )
            return

        self._last_delegation_offset_ms = max(
            self._last_delegation_offset_ms,
            offset_ms,
        )

        request = DelegationRequest(
            text=request_text,
            intent="live_client_delegation",
        )

        self._active_delegation = delegation_id
        self._set_state(LiveState.DELEGATING)

        try:
            result = await self._delegate_to_gateway(
                request,
                delegation_id=delegation_id,
            )

            # Client delegation is completed by appending authoritative
            # commentary associated with the exact Live delegation ID.
            await self._session.send_commentary(
                result.text,
                delegation_id=delegation_id,
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            logger.exception(
                "GPT-Live client delegation failed: %s",
                delegation_id,
            )

            with suppress(Exception):
                await self._session.send_commentary(
                    (
                        "The Mana-Agent backend could not complete that "
                        f"request: {exc}"
                    ),
                    delegation_id=delegation_id,
                )

            self._emit(
                LiveEventType.DELEGATION_ERROR,
                data={
                    "correlation_id": delegation_id,
                    "error": str(exc),
                },
            )

        finally:
            if self._active_delegation == delegation_id:
                self._active_delegation = None

            if self._running and self._state == LiveState.DELEGATING:
                self._set_state(LiveState.LISTENING)

    def _request_text_for_delegation(
        self,
        offset_ms: int,
    ) -> str:
        """Build the spoken request from input transcript fragments.

        session.delegation.created intentionally contains metadata only. The
        request text comes from transcript/application state.

        Use fragments after the most recent assistant/delegation boundary and
        at or just after the delegation timeline offset.
        """
        lower_bound = max(
            self._last_assistant_activity_ms,
            self._last_delegation_offset_ms,
        )

        upper_bound = (
            offset_ms + _DELEGATION_TRANSCRIPT_GRACE_MS
            if offset_ms >= 0
            else None
        )

        parts: list[str] = []

        for start_ms, end_ms, text in self._input_fragments:
            if end_ms <= lower_bound:
                continue

            if (
                upper_bound is not None
                and start_ms > upper_bound
            ):
                continue

            parts.append(text)

        request = "".join(parts).strip()

        if request:
            return request

        # Conservative fallback for missing/zero timestamps.
        for transcript in reversed(self._transcripts):
            if getattr(transcript, "role", "") != "user":
                continue

            text = str(
                getattr(transcript, "text", "") or ""
            ).strip()

            if text:
                return text

        return ""

    # ------------------------------------------------------------------
    # Gateway delegation
    # ------------------------------------------------------------------

    async def _delegate_to_gateway(
        self,
        request: DelegationRequest,
        *,
        delegation_id: str | None = None,
    ) -> DelegationResult:
        """Execute delegated work through the existing AgentChatGateway."""
        target_delegation_id = delegation_id or request.correlation_id
        self._emit(
            LiveEventType.DELEGATION_STARTED,
            data={
                "text": request.text,
                "intent": request.intent,
                "correlation_id": target_delegation_id,
            },
        )
        try:
            async with self._gateway_lock:
                # process_turn is the same synchronous gateway entry point used
                # elsewhere in Mana-Agent. Run it off the Live receive loop so
                # microphone/audio events keep flowing while backend work runs.
                turn_result = await asyncio.to_thread(
                    self._gateway.process_turn,
                    self._gateway_session_id,
                    request.text,
                )

                # Be tolerant if a future gateway implementation returns an
                # awaitable from the same API.
                if inspect.isawaitable(turn_result):
                    turn_result = await turn_result

            result_text = (
                getattr(turn_result, "answer", "")
                or getattr(turn_result, "text", "")
                or str(turn_result)
            )

            self._emit(
                LiveEventType.DELEGATION_COMPLETED,
                data={
                    "correlation_id": target_delegation_id,
                    "success": True,
                },
            )

            return DelegationResult(
                correlation_id=target_delegation_id,
                text=result_text,
                success=True,
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            logger.exception(
                "Mana gateway delegation failed: %s",
                exc,
            )

            self._emit(
                LiveEventType.DELEGATION_ERROR,
                data={
                    "correlation_id": target_delegation_id,
                    "error": str(exc),
                },
            )

            return DelegationResult(
                correlation_id=target_delegation_id,
                text=(
                    "I encountered an error processing that request: "
                    f"{exc}"
                ),
                success=False,
            )

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    async def _handle_error(
        self,
        event: dict[str, Any],
    ) -> None:
        error = event.get("error")

        logger.error(
            "GPT-Live API error: %s",
            error,
        )

        self._emit(
            LiveEventType.AUDIO_ERROR,
            data={"error": error or {}},
        )

        self._set_state(LiveState.ERROR)

    # ------------------------------------------------------------------
    # Transcript helpers
    # ------------------------------------------------------------------

    def _flush_user_transcript(self) -> None:
        text = self._user_display_buffer.strip()

        if not text:
            self._user_display_buffer = ""
            return

        self._user_display_buffer = ""

        transcript = LiveTranscript(
            role="user",
            text=text,
        )
        self._transcripts.append(transcript)

        self._emit(
            LiveEventType.TRANSCRIPT_USER,
            data={"text": text},
        )

    def _flush_assistant_transcript(self) -> None:
        text = self._assistant_display_buffer.strip()

        if not text:
            self._assistant_display_buffer = ""
            return

        self._assistant_display_buffer = ""

        transcript = LiveTranscript(
            role="assistant",
            text=text,
        )
        self._transcripts.append(transcript)

        self._emit(
            LiveEventType.TRANSCRIPT_ASSISTANT,
            data={"text": text},
        )

    @staticmethod
    def _event_ms(
        value: Any,
        *,
        fallback: int = -1,
    ) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    # ------------------------------------------------------------------
    # State / event helpers
    # ------------------------------------------------------------------

    def _set_state(
        self,
        state: LiveState,
    ) -> None:
        if state == self._state:
            return

        old = self._state
        self._state = state

        self._emit(
            LiveEventType.STATE_CHANGED,
            data={
                "old_state": old.value,
                "new_state": state.value,
            },
            state=state,
        )

        logger.debug(
            "Live state: %s -> %s",
            old.value,
            state.value,
        )

    def _emit(
        self,
        event_type: LiveEventType,
        *,
        data: dict[str, Any] | None = None,
        state: LiveState | None = None,
    ) -> None:
        event = LiveEvent(
            event_type=event_type,
            data=data or {},
            state=state or self._state,
        )

        if self._event_cb is None:
            return

        try:
            self._event_cb(event)
        except Exception:
            logger.debug(
                "Live event callback failed",
                exc_info=True,
            )
