"""Audio transport for Mana-Agent Live mode.

Manages microphone capture and speaker playback using ``sounddevice``.
Requires the ``live`` optional extra::

    pip install "mana-agent[live]"
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class AudioDeviceError(Exception):
    """Raised when an audio device cannot be opened or fails at runtime."""


class AudioTransport:
    """Bidirectional audio I/O for the Live session.

    Captures microphone input into a thread-safe ring buffer and plays back
    assistant audio through the system speaker.  All public methods are safe
    to call from an asyncio event loop.
    """

    _CHUNK_FRAMES = 2400  # 100ms at 24 kHz

    def __init__(
        self,
        *,
        input_sample_rate: int = 24000,
        output_sample_rate: int = 24000,
        input_device: int | None = None,
        output_device: int | None = None,
        channels: int = 1,
        dtype: str = "int16",
    ) -> None:
        self._input_sr = input_sample_rate
        self._output_sr = output_sample_rate
        self._input_device = input_device
        self._output_device = output_device
        self._channels = channels
        self._dtype = dtype

        # Lazy-loaded sounddevice / numpy
        self._sd: Any = None
        self._np: Any = None

        # State
        self._input_stream: Any = None
        self._output_stream: Any = None
        self._capture_buffer: collections.deque[bytes] = collections.deque(maxlen=200)
        self._playback_buffer: collections.deque[bytes] = collections.deque(maxlen=500)
        self._lock = threading.Lock()
        self._capturing = False
        self._playing = False

    # ------------------------------------------------------------------
    # Lazy dependency loading
    # ------------------------------------------------------------------

    def _ensure_deps(self) -> None:
        """Import optional audio dependencies or raise a clear error."""
        if self._sd is not None:
            return
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError as exc:
            raise AudioDeviceError(
                "Live mode requires audio dependencies.\n"
                "Install with: pip install 'mana-agent[live]'"
            ) from exc
        self._sd = sd
        self._np = np

    # ------------------------------------------------------------------
    # Capture (microphone)
    # ------------------------------------------------------------------

    async def start_capture(self) -> None:
        """Open the microphone input stream."""
        self._ensure_deps()
        if self._capturing:
            return
        try:
            self._input_stream = self._sd.InputStream(
                samplerate=self._input_sr,
                channels=self._channels,
                dtype=self._dtype,
                blocksize=self._CHUNK_FRAMES,
                device=self._input_device,
                callback=self._capture_callback,
            )
            self._input_stream.start()
            self._capturing = True
            logger.debug("Audio capture started (device=%s)", self._input_device)
        except Exception as exc:
            raise AudioDeviceError(f"Failed to open input device: {exc}") from exc

    async def stop_capture(self) -> None:
        """Close the microphone input stream."""
        self._capturing = False
        if self._input_stream is not None:
            try:
                self._input_stream.stop()
                self._input_stream.close()
            except Exception:
                logger.debug("Error closing input stream", exc_info=True)
            self._input_stream = None

    def _capture_callback(
        self, indata: Any, frames: int, time_info: Any, status: Any,
    ) -> None:
        """sounddevice callback — runs on the audio thread."""
        if status:
            logger.debug("Capture status: %s", status)
        with self._lock:
            self._capture_buffer.append(bytes(indata))

    def get_capture_chunk(self) -> bytes | None:
        """Return the next captured audio chunk, or None."""
        with self._lock:
            if self._capture_buffer:
                return self._capture_buffer.popleft()
        return None

    # ------------------------------------------------------------------
    # Playback (speaker)
    # ------------------------------------------------------------------

    async def start_playback(self) -> None:
        """Open the speaker output stream."""
        self._ensure_deps()
        if self._playing:
            return
        try:
            self._output_stream = self._sd.OutputStream(
                samplerate=self._output_sr,
                channels=self._channels,
                dtype=self._dtype,
                blocksize=self._CHUNK_FRAMES,
                device=self._output_device,
                callback=self._playback_callback,
            )
            self._output_stream.start()
            self._playing = True
            logger.debug("Audio playback started (device=%s)", self._output_device)
        except Exception as exc:
            raise AudioDeviceError(f"Failed to open output device: {exc}") from exc

    async def stop_playback(self) -> None:
        """Close the speaker output stream and drain the buffer."""
        self._playing = False
        if self._output_stream is not None:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception:
                logger.debug("Error closing output stream", exc_info=True)
            self._output_stream = None
        with self._lock:
            self._playback_buffer.clear()

    def _playback_callback(
        self, outdata: Any, frames: int, time_info: Any, status: Any,
    ) -> None:
        """sounddevice callback — runs on the audio thread."""
        if status:
            logger.debug("Playback status: %s", status)
        with self._lock:
            if self._playback_buffer:
                chunk = self._playback_buffer.popleft()
            else:
                chunk = None
        if chunk is not None:
            expected = frames * self._channels * 2  # int16 = 2 bytes
            if len(chunk) < expected:
                chunk = chunk + b"\x00" * (expected - len(chunk))
            outdata[:] = self._np.frombuffer(chunk[:expected], dtype=self._dtype).reshape(-1, self._channels)
        else:
            outdata.fill(0)

    def queue_playback(self, audio_data: bytes) -> None:
        """Enqueue raw audio bytes for playback."""
        with self._lock:
            self._playback_buffer.append(audio_data)

    def interrupt_playback(self) -> None:
        """Immediately clear the playback buffer (for user interruptions)."""
        with self._lock:
            self._playback_buffer.clear()
        logger.debug("Playback interrupted — buffer cleared")

    def clear_playback_buffer(self) -> None:
        """Clear pending playback audio."""
        with self._lock:
            self._playback_buffer.clear()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def cleanup(self) -> None:
        """Release all audio resources."""
        await self.stop_capture()
        await self.stop_playback()
        logger.debug("Audio transport cleaned up")

    @staticmethod
    def list_devices() -> list[dict[str, Any]]:
        """List available audio devices."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            if isinstance(devices, dict):
                return [devices]
            return [dict(d) for d in devices]  # type: ignore[arg-type]
        except ImportError:
            return []
        except Exception:
            logger.debug("Failed to enumerate audio devices", exc_info=True)
            return []
