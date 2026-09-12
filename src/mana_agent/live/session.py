"""OpenAI GPT-Live WebSocket session for Mana-Agent Live mode."""

from __future__ import annotations

import asyncio
import base64
from collections import deque
from contextlib import suppress
import json
import logging
from typing import Any, AsyncIterator
import uuid

logger = logging.getLogger(__name__)

_LIVE_URL = "wss://api.openai.com/v1/live/sessions"
_MAX_WS_MESSAGE_BYTES = 16 * 1024 * 1024


class LiveSessionError(Exception):
    """Raised when the Live session encounters an unrecoverable error."""


class LiveSession:
    """Manage a primary GPT-Live WebSocket connection.

    Live is only the realtime conversation layer.

    Mana-Agent continues to own:
    - AgentChatGateway
    - model routing
    - memory
    - tools
    - permissions / approvals
    - durable tasks
    - Codex
    - backend execution
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-live-1",
        voice: str = "alloy",
        vad_mode: str = "server",
        vad_threshold: float = 0.5,
        turn_detection_silence_ms: int = 500,
        audio_format: str = "pcm16",
        audio_sample_rate: int = 24_000,
        max_response_output_tokens: int | None = None,
        auto_reconnect: bool = True,
        reconnect_delay_seconds: float = 2.0,
        reconnect_max_attempts: int = 5,
        system_instructions: str = "",
        store: bool = False,
        start_timeout_seconds: float = 15.0,
        close_timeout_seconds: float = 3.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")

        if not model.strip():
            raise ValueError("model must not be empty")

        if not voice.strip():
            raise ValueError("voice must not be empty")

        if reconnect_delay_seconds < 0:
            raise ValueError(
                "reconnect_delay_seconds must be >= 0"
            )

        if reconnect_max_attempts < 0:
            raise ValueError(
                "reconnect_max_attempts must be >= 0"
            )

        if start_timeout_seconds <= 0:
            raise ValueError(
                "start_timeout_seconds must be > 0"
            )

        if close_timeout_seconds <= 0:
            raise ValueError(
                "close_timeout_seconds must be > 0"
            )

        self._api_key = api_key
        self._model = model
        self._voice = voice

        self._audio_format = audio_format
        self._audio_sample_rate = audio_sample_rate

        self._auto_reconnect = auto_reconnect
        self._reconnect_delay = reconnect_delay_seconds
        self._reconnect_max = reconnect_max_attempts

        self._system_instructions = system_instructions
        self._store = store

        self._start_timeout = start_timeout_seconds
        self._close_timeout = close_timeout_seconds

        # --------------------------------------------------------------
        # Legacy constructor compatibility
        # --------------------------------------------------------------
        #
        # These existed in the first Realtime implementation.
        #
        # GPT-Live does NOT accept the old Realtime session shape:
        #
        #   turn_detection
        #   max_response_output_tokens
        #
        # Keep the constructor arguments so existing callers don't break,
        # but don't send them to GPT-Live.
        #
        self._legacy_vad_mode = vad_mode
        self._legacy_vad_threshold = vad_threshold
        self._legacy_turn_detection_silence_ms = (
            turn_detection_silence_ms
        )
        self._legacy_max_response_output_tokens = (
            max_response_output_tokens
        )

        self._ws: Any = None

        self._connected = False
        self._closing = False
        self._receive_loop_active = False

        self._reconnect_count = 0

        self._session_id = ""

        self._final_usage: dict[str, Any] | None = None

        self._send_lock = asyncio.Lock()

        self._session_closed_event = asyncio.Event()

        # connect() consumes session.started itself so it can guarantee
        # that Live is actually ready before microphone capture begins.
        #
        # Keep consumed startup events here so the normal event loop can
        # still observe them.
        self._prefetched_events: deque[
            dict[str, Any]
        ] = deque()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Return True after session.started has been received."""
        return (
            self._connected
            and self._ws is not None
            and not self._closing
        )

    @property
    def session_id(self) -> str:
        """Return the opaque GPT-Live session ID."""
        return self._session_id

    @property
    def final_usage(self) -> dict[str, Any] | None:
        """Return final Live usage from session.closed."""
        return self._final_usage

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to GPT-Live.

        Flow:

            WebSocket open
                ↓
            session.start
                ↓
            wait for session.started
                ↓
            mark connection ready

        We do not mark the session connected merely because the WebSocket
        handshake succeeded.
        """
        if self.is_connected:
            return

        try:
            import websockets
        except ImportError as exc:
            raise LiveSessionError(
                "Live mode requires the websockets package."
            ) from exc

        self._closing = False
        self._connected = False

        self._session_id = ""
        self._final_usage = None

        self._session_closed_event.clear()
        self._prefetched_events.clear()

        try:
            self._ws = await websockets.connect(
                _LIVE_URL,
                additional_headers={
                    "Authorization": f"Bearer {self._api_key}",
                },
                max_size=_MAX_WS_MESSAGE_BYTES,
                ping_interval=20,
                ping_timeout=10,
            )

            # GPT-Live does NOT use:
            #
            #   ?model=gpt-live-1
            #
            # The model belongs inside session.start.
            await self._send_raw_event(
                {
                    "type": "session.start",
                    "event_id": self._event_id(
                        "start"
                    ),
                    "session": self._startup_config(),
                }
            )

            await asyncio.wait_for(
                self._wait_until_started(),
                timeout=self._start_timeout,
            )

            self._connected = True
            self._reconnect_count = 0

            logger.info(
                (
                    "Connected to OpenAI GPT-Live API "
                    "(model=%s, session_id=%s)"
                ),
                self._model,
                self._session_id or "unknown",
            )

        except asyncio.TimeoutError as exc:
            await self._abort_socket()

            raise LiveSessionError(
                (
                    "Timed out waiting for session.started "
                    f"after {self._start_timeout:.1f}s"
                )
            ) from exc

        except LiveSessionError:
            await self._abort_socket()
            raise

        except Exception as exc:
            await self._abort_socket()

            raise LiveSessionError(
                f"Failed to connect to GPT-Live: {exc}"
            ) from exc

    async def disconnect(self) -> None:
        """Gracefully close the GPT-Live session.

        GPT-Live should receive session.close before the underlying
        WebSocket is closed.

        If receive_events() currently owns socket reads, wait for it to
        observe session.closed. Otherwise read session.closed here.
        """
        ws = self._ws

        if ws is None:
            self._connected = False
            return

        self._closing = True

        try:
            if self._connected:
                with suppress(Exception):
                    await self._send_raw_event(
                        {
                            "type": "session.close",
                            "event_id": self._event_id(
                                "close"
                            ),
                        }
                    )

                if self._receive_loop_active:
                    with suppress(
                        asyncio.TimeoutError
                    ):
                        await asyncio.wait_for(
                            self._session_closed_event.wait(),
                            timeout=self._close_timeout,
                        )
                else:
                    with suppress(
                        asyncio.TimeoutError,
                        LiveSessionError,
                    ):
                        await asyncio.wait_for(
                            self._read_until_closed(),
                            timeout=self._close_timeout,
                        )

        finally:
            self._connected = False

            with suppress(Exception):
                await ws.close()

            self._ws = None
            self._closing = False

            logger.info(
                "Disconnected from OpenAI GPT-Live API"
            )

    async def _abort_socket(self) -> None:
        """Close a connection that didn't finish session startup."""
        self._connected = False

        ws = self._ws
        self._ws = None

        if ws is not None:
            with suppress(Exception):
                await ws.close()

    # ------------------------------------------------------------------
    # Startup configuration
    # ------------------------------------------------------------------

    def _startup_config(self) -> dict[str, Any]:
        """Build the GPT-Live session.start configuration."""
        config: dict[str, Any] = {
            "model": self._model,
            "audio": {
                "format": self._audio_config(),
                "output": {
                    "voice": self._voice,
                },
            },

            # Native client delegation.
            #
            # GPT-Live decides that backend help is required and emits:
            #
            #   session.delegation.created
            #
            # Mana-Agent then routes that work through AgentChatGateway.
            "delegation": {
                "type": "client",
            },

            "store": self._store,
        }

        if self._system_instructions.strip():
            config["instructions"] = (
                self._system_instructions
            )

        return config

    def _audio_config(self) -> dict[str, Any]:
        """Convert Mana audio names to GPT-Live audio format objects."""
        fmt = (
            self._audio_format
            .strip()
            .lower()
            .replace("-", "_")
        )

        # --------------------------------------------------------------
        # PCM16
        # --------------------------------------------------------------

        if fmt in {
            "pcm16",
            "pcm",
            "audio/pcm",
            "audio_pcm",
        }:
            if self._audio_sample_rate not in {
                16_000,
                24_000,
            }:
                raise LiveSessionError(
                    (
                        "GPT-Live PCM audio must use "
                        "16000 or 24000 Hz."
                    )
                )

            return {
                "type": "audio/pcm",
                "rate": self._audio_sample_rate,
            }

        if fmt in {
            "pcm16_16000",
            "pcm16_16k",
            "pcm_16000",
        }:
            return {
                "type": "audio/pcm",
                "rate": 16_000,
            }

        if fmt in {
            "pcm16_24000",
            "pcm16_24k",
            "pcm_24000",
        }:
            return {
                "type": "audio/pcm",
                "rate": 24_000,
            }

        # --------------------------------------------------------------
        # G.711 μ-law
        # --------------------------------------------------------------

        if fmt in {
            "pcmu",
            "audio/pcmu",
            "audio_pcmu",
            "g711_ulaw",
            "g711_mulaw",
            "ulaw",
            "mulaw",
        }:
            return {
                "type": "audio/pcmu",
                "rate": 8_000,
            }

        # --------------------------------------------------------------
        # G.711 A-law
        # --------------------------------------------------------------

        if fmt in {
            "pcma",
            "audio/pcma",
            "audio_pcma",
            "g711_alaw",
            "alaw",
        }:
            return {
                "type": "audio/pcma",
                "rate": 8_000,
            }

        raise LiveSessionError(
            (
                "Unsupported GPT-Live audio format: "
                f"{self._audio_format!r}"
            )
        )

    async def _wait_until_started(self) -> None:
        """Wait for the server's session.started event."""
        while True:
            event = await self._recv_event_direct()

            event_type = event.get("type")

            if event_type == "error":
                raise LiveSessionError(
                    self._server_error(event)
                )

            self._observe_event(event)

            # Keep startup events visible to receive_events().
            self._prefetched_events.append(event)

            if event_type == "session.started":
                return

    async def _read_until_closed(self) -> None:
        """Read directly until session.closed."""
        while self._ws is not None:
            event = await self._recv_event_direct()

            self._observe_event(event)

            if event.get("type") == "session.closed":
                return

    # ------------------------------------------------------------------
    # Sending raw events
    # ------------------------------------------------------------------

    @staticmethod
    def _delegation_tool_schema() -> dict[str, Any]:
        """Return the function tool schema for delegate_to_mana."""
        return {
            "type": "function",
            "name": "delegate_to_mana",
            "description": "Delegate a task, coding action, analysis, or query to the Mana-Agent backend.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "The user's request or command to execute.",
                    },
                    "intent": {
                        "type": "string",
                        "description": "The classified intent of the request.",
                    },
                },
                "required": ["request"],
            },
        }

    async def send_event(
        self,
        event: dict[str, Any],
    ) -> None:
        """Send a raw GPT-Live event after startup."""
        if not self.is_connected:
            raise LiveSessionError(
                "Not connected: GPT-Live session is not connected"
            )

        await self._send_raw_event(event)

    async def _send_raw_event(
        self,
        event: dict[str, Any],
    ) -> None:
        """Serialize and write an event to the WebSocket."""
        if self._ws is None:
            raise LiveSessionError(
                "GPT-Live WebSocket is not open"
            )

        try:
            payload = json.dumps(
                event,
                separators=(",", ":"),
            )

            async with self._send_lock:
                await self._ws.send(payload)

        except Exception as exc:
            self._connected = False

            raise LiveSessionError(
                f"Failed to send GPT-Live event: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    async def send_audio(
        self,
        chunk: bytes,
    ) -> None:
        """Stream microphone audio to GPT-Live."""
        if not chunk:
            return

        await self.send_event(
            {
                "type": "session.input_audio.append",
                "event_id": self._event_id(
                    "audio"
                ),
                "audio": base64.b64encode(
                    chunk
                ).decode("ascii"),
            }
        )

    async def mute_input(self) -> None:
        """Mute GPT-Live microphone input."""
        await self.send_event(
            {
                "type": "session.input_audio.mute",
                "event_id": self._event_id(
                    "mute"
                ),
            }
        )

    async def unmute_input(self) -> None:
        """Unmute GPT-Live microphone input."""
        await self.send_event(
            {
                "type": "session.input_audio.unmute",
                "event_id": self._event_id(
                    "unmute"
                ),
            }
        )

    # ------------------------------------------------------------------
    # Context / delegated results
    # ------------------------------------------------------------------

    async def send_text(
        self,
        text: str,
        *,
        delegation_id: str | None = None,
    ) -> None:
        """Send text GPT-Live should communicate aloud.

        This replaces the old conversation.item.create behavior.

        For a Mana backend result, delegation_id should be the exact
        opaque ID from:

            session.delegation.created
        """
        await self.send_commentary(
            text,
            delegation_id=delegation_id,
        )

    async def send_commentary(
        self,
        content: str,
        *,
        delegation_id: str | None = None,
    ) -> None:
        """Send information GPT-Live should communicate aloud."""
        if not content:
            return

        await self.send_event(
            {
                "type": "session.commentary.append",
                "event_id": self._event_id(
                    "commentary"
                ),
                "delegation_id": delegation_id,
                "content": content,
            }
        )

    async def send_thinking(
        self,
        content: str,
        *,
        delegation_id: str | None = None,
    ) -> None:
        """Send quiet backend context to GPT-Live."""
        if not content:
            return

        await self.send_event(
            {
                "type": "session.thinking.append",
                "event_id": self._event_id(
                    "thinking"
                ),
                "delegation_id": delegation_id,
                "content": content,
            }
        )

    async def append_instructions(
        self,
        content: str,
        *,
        delegation_id: str | None = None,
    ) -> None:
        """Append new conversation instructions."""
        if not content:
            return

        await self.send_event(
            {
                "type": "session.instructions.append",
                "event_id": self._event_id(
                    "instructions"
                ),
                "delegation_id": delegation_id,
                "content": content,
            }
        )

    # ------------------------------------------------------------------
    # Compatibility with old Realtime implementation
    # ------------------------------------------------------------------

    async def truncate_response(self) -> None:
        """Compatibility hook.

        GPT-Live is full duplex.

        Incoming user speech naturally interrupts conversation behavior.
        If Mana wants to instantly stop already-buffered speaker audio,
        that belongs in the local playback layer rather than through the
        old Realtime response.cancel event.
        """
        logger.debug(
            (
                "truncate_response is a no-op for "
                "GPT-Live full-duplex mode"
            )
        )

    async def create_response(
        self,
        instructions: str | None = None,
    ) -> None:
        """Compatibility hook for old response.create calls.

        Client-delegation GPT-Live drives spoken conversation itself.
        """
        if instructions:
            await self.append_instructions(
                instructions
            )

    async def commit_audio_buffer(self) -> None:
        """Compatibility no-op.

        GPT-Live continuously processes session.input_audio.append.
        """
        logger.debug(
            (
                "commit_audio_buffer is a no-op "
                "for GPT-Live"
            )
        )

    async def clear_audio_buffer(self) -> None:
        """Compatibility no-op.

        Stop or mute microphone capture instead.
        """
        logger.debug(
            (
                "clear_audio_buffer is a no-op "
                "for GPT-Live"
            )
        )

    async def update_session(
        self,
        config_updates: dict[str, Any],
    ) -> None:
        """Safely translate old session update calls.

        GPT-Live startup fields such as:

            model
            voice
            audio
            store
            delegation mode

        are not general mutable session.update fields.

        Instructions can be appended after startup, so preserve that
        useful compatibility behavior.
        """
        if not config_updates:
            return

        updates = dict(config_updates)

        instructions = updates.pop(
            "instructions",
            None,
        )

        if updates:
            fields = ", ".join(
                sorted(updates)
            )

            raise LiveSessionError(
                (
                    "GPT-Live client mode cannot update "
                    "these startup fields: "
                    f"{fields}. "
                    "Start a new Live session instead."
                )
            )

        if instructions:
            await self.append_instructions(
                str(instructions)
            )

    # ------------------------------------------------------------------
    # Receiving events
    # ------------------------------------------------------------------

    async def receive_events(
        self,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield GPT-Live server events.

        Important events:

            session.started

            session.input_transcript.delta
            session.output_transcript.delta

            session.output_audio.delta

            session.delegation.created

            session.commentary.appended
            session.thinking.appended
            session.instructions.appended

            session.closed
            error
        """
        if self._receive_loop_active:
            raise LiveSessionError(
                (
                    "receive_events() already has "
                    "an active reader"
                )
            )

        self._receive_loop_active = True

        try:
            while True:
                # Events consumed while waiting for session.started.
                while self._prefetched_events:
                    event = (
                        self._prefetched_events.popleft()
                    )

                    self._observe_event(event)

                    yield event

                    if event.get("type") == (
                        "session.closed"
                    ):
                        return

                if not self.is_connected:
                    if (
                        self._closing
                        or not self._auto_reconnect
                    ):
                        return

                    await self._reconnect()

                    if not self.is_connected:
                        return

                    continue

                try:
                    event = (
                        await self._recv_event_direct()
                    )

                except Exception as exc:
                    if self._closing:
                        return

                    logger.warning(
                        (
                            "GPT-Live WebSocket "
                            "receive error: %s"
                        ),
                        exc,
                    )

                    self._connected = False

                    if not self._auto_reconnect:
                        return

                    continue

                self._observe_event(event)

                yield event

                if event.get("type") == (
                    "session.closed"
                ):
                    self._connected = False
                    return

        finally:
            self._receive_loop_active = False

    async def _recv_event_direct(
        self,
    ) -> dict[str, Any]:
        """Receive one GPT-Live JSON event."""
        if self._ws is None:
            raise LiveSessionError(
                "GPT-Live WebSocket is not open"
            )

        raw = await self._ws.recv()

        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise LiveSessionError(
                    (
                        "Received non-UTF-8 data "
                        "from GPT-Live"
                    )
                ) from exc

        try:
            event = json.loads(raw)

        except json.JSONDecodeError as exc:
            raise LiveSessionError(
                (
                    "Received a non-JSON message "
                    "from GPT-Live"
                )
            ) from exc

        if not isinstance(event, dict):
            raise LiveSessionError(
                "Received an invalid GPT-Live event"
            )

        return event

    def _observe_event(
        self,
        event: dict[str, Any],
    ) -> None:
        """Update internal lifecycle state."""
        event_type = event.get("type")

        if event_type == "session.started":
            session = event.get("session")

            if isinstance(session, dict):
                session_id = session.get("id")

                if isinstance(
                    session_id,
                    str,
                ):
                    self._session_id = session_id

        elif event_type == "session.closed":
            usage = event.get("usage")

            if isinstance(usage, dict):
                self._final_usage = usage
            else:
                self._final_usage = None

            self._session_closed_event.set()

    # ------------------------------------------------------------------
    # Reconnection
    # ------------------------------------------------------------------

    async def _reconnect(self) -> None:
        """Reconnect by starting a fresh Live session.

        A dropped primary GPT-Live socket isn't silently resumed as the
        exact same voice session. Mana's higher-level transcript/session
        state should therefore remain authoritative.
        """
        while (
            self._reconnect_count
            < self._reconnect_max
        ):
            self._reconnect_count += 1

            attempt = self._reconnect_count

            delay = (
                self._reconnect_delay
                * (2 ** (attempt - 1))
            )

            logger.info(
                (
                    "GPT-Live reconnect attempt "
                    "%d/%d in %.1fs"
                ),
                attempt,
                self._reconnect_max,
                delay,
            )

            if delay:
                await asyncio.sleep(delay)

            if self._closing:
                return

            try:
                await self.connect()

                logger.warning(
                    (
                        "GPT-Live reconnected as a "
                        "fresh Live session; Mana must "
                        "restore higher-level "
                        "conversation context"
                    )
                )

                return

            except LiveSessionError as exc:
                # connect() resets reconnect_count only
                # after successful session.started.
                self._reconnect_count = attempt

                logger.debug(
                    (
                        "GPT-Live reconnect attempt "
                        "%d failed: %s"
                    ),
                    attempt,
                    exc,
                )

        self._connected = False

        logger.error(
            (
                "Max GPT-Live reconnect attempts "
                "(%d) exceeded"
            ),
            self._reconnect_max,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _event_id(
        prefix: str,
    ) -> str:
        return (
            f"mana_{prefix}_"
            f"{uuid.uuid4().hex}"
        )

    @staticmethod
    def _server_error(
        event: dict[str, Any],
    ) -> str:
        """Convert a GPT-Live error event to a readable message."""
        error = event.get("error")

        if isinstance(error, dict):
            error_type = error.get("type")
            code = error.get("code")
            message = error.get("message")

            parts = [
                str(value)
                for value in (
                    error_type,
                    code,
                    message,
                )
                if value not in (
                    None,
                    "",
                )
            ]

            if parts:
                return (
                    "GPT-Live error: "
                    + " | ".join(parts)
                )

        if (
            isinstance(error, str)
            and error
        ):
            return (
                f"GPT-Live error: {error}"
            )

        return (
            "GPT-Live error event: "
            f"{event!r}"
        )