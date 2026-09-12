"""Professional terminal UI for Mana-Agent Live mode.

Uses one stable ``rich.Live`` region with a single refresh loop. State changes
only mutate display data; they do not trigger independent redraws. This avoids
the flashing / repeated INITIALIZING panels that can happen when animation,
state updates, and Rich's own auto-refresh all redraw at the same time.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field

from rich import box
from rich.align import Align
from rich.console import Console, ConsoleRenderable, Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mana_agent.live.models import LiveState


@dataclass(frozen=True)
class _StateStyle:
    color: str
    amplitude: float
    speed: float
    symbol: str
    label: str


_STATE_STYLE: dict[LiveState, _StateStyle] = {
    LiveState.INITIALIZING: _StateStyle(
        "grey62", 0.10, 0.45, "○", "INITIALIZING"
    ),
    LiveState.CONNECTING: _StateStyle(
        "cyan", 0.20, 0.85, "◌", "CONNECTING"
    ),
    LiveState.LISTENING: _StateStyle(
        "green", 0.42, 1.15, "●", "LISTENING"
    ),
    LiveState.THINKING: _StateStyle(
        "yellow", 0.28, 1.80, "◆", "THINKING"
    ),
    LiveState.DELEGATING: _StateStyle(
        "magenta", 0.34, 1.50, "◇", "DELEGATING"
    ),
    LiveState.SPEAKING: _StateStyle(
        "bright_blue", 0.72, 1.35, "●", "SPEAKING"
    ),
    LiveState.INTERRUPTED: _StateStyle(
        "red", 0.18, 2.40, "■", "INTERRUPTED"
    ),
    LiveState.RECONNECTING: _StateStyle(
        "orange3", 0.24, 1.75, "↻", "RECONNECTING"
    ),
    LiveState.ERROR: _StateStyle(
        "bright_red", 0.08, 0.30, "×", "ERROR"
    ),
    LiveState.SHUTDOWN: _StateStyle(
        "grey50", 0.02, 0.20, "·", "CLOSING"
    ),
}

_DEFAULT_STATE_STYLE = _StateStyle(
    "white", 0.15, 1.0, "•", "LIVE"
)

_WAVE_CHARS = " ▁▂▃▄▅▆▇█"
_WAVE_WIDTH = 38
_TRANSCRIPT_MAX = 8
_DEFAULT_FPS = 8.0


@dataclass
class _TranscriptLine:
    role: str
    text: str


@dataclass
class _DisplayState:
    state: LiveState = LiveState.INITIALIZING

    model: str = ""
    delegation_model: str | None = None
    voice: str = ""
    vad_mode: str = ""

    transcripts: deque[_TranscriptLine] = field(
        default_factory=lambda: deque(maxlen=_TRANSCRIPT_MAX)
    )

    last_delegation: str | None = None
    last_error: str | None = None

    usage_input: int = 0
    usage_output: int = 0


class LivePulseDisplay:
    """Own one stable ``rich.Live`` region for Mana-Agent Live mode.

    The important rendering rule is:

    - state setters never redraw;
    - Rich auto-refresh is disabled;
    - exactly one animation task owns terminal refreshes.

    This prevents competing refresh sources from repeatedly printing panels.
    """

    def __init__(
        self,
        console: Console | None = None,
        fps: float = _DEFAULT_FPS,
    ) -> None:
        self._console = console or Console()

        if fps <= 0:
            raise ValueError("fps must be greater than zero")

        self._fps = fps
        self._phase = 0.0

        self._state = _DisplayState()

        self._live: Live | None = None
        self._tick_task: asyncio.Task[None] | None = None

        self._start_time = time.monotonic()

        self._dirty = True
        self._started = False

    # ------------------------------------------------------------------
    # Configuration / lifecycle
    # ------------------------------------------------------------------

    def configure(
        self,
        *,
        model: str,
        delegation_model: str | None,
        voice: str,
        vad_mode: str,
    ) -> None:
        self._state.model = model
        self._state.delegation_model = delegation_model
        self._state.voice = voice
        self._state.vad_mode = vad_mode
        self._dirty = True

    async def start(self) -> None:
        """Start the single Rich live region."""
        if self._started:
            return

        self._started = True
        self._start_time = time.monotonic()
        self._phase = 0.0
        self._dirty = True

        # auto_refresh=False is deliberate. The asyncio tick task below is the
        # only place that refreshes the terminal.
        self._live = Live(
            self._render(),
            console=self._console,
            auto_refresh=False,
            screen=False,
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
            vertical_overflow="crop",
        )

        self._live.start(refresh=True)

        self._tick_task = asyncio.create_task(
            self._tick_loop(),
            name="mana-live-tui",
        )

    async def stop(self) -> None:
        """Render one final frame and release the Live region."""
        if not self._started:
            return

        self._started = False

        if self._tick_task is not None:
            self._tick_task.cancel()

            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass

            self._tick_task = None

        if self._live is not None:
            self._live.update(
                self._render(),
                refresh=True,
            )
            self._live.stop()
            self._live = None

    async def _tick_loop(self) -> None:
        interval = 1.0 / self._fps

        try:
            while True:
                style = self._style()

                # Animate only states where motion is useful. ERROR/SHUTDOWN
                # remain visually calm.
                if self._state.state not in {
                    LiveState.ERROR,
                    LiveState.SHUTDOWN,
                }:
                    self._phase += interval * style.speed

                self._refresh()
                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            raise

    def _refresh(self) -> None:
        """Refresh from the one authoritative render path."""
        live = self._live

        if live is None:
            return

        live.update(
            self._render(),
            refresh=True,
        )
        self._dirty = False

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def set_state(self, state: LiveState) -> None:
        if state == self._state.state:
            return

        self._state.state = state

        # Keep the latest error visible until the state actually moves away
        # from ERROR. Do not clear it on every animated refresh.
        if state not in {
            LiveState.ERROR,
            LiveState.RECONNECTING,
        }:
            self._state.last_error = None

        self._dirty = True

    def add_transcript(
        self,
        role: str,
        text: str,
    ) -> None:
        cleaned = " ".join(str(text).split())

        if not cleaned:
            return

        self._state.transcripts.append(
            _TranscriptLine(
                role=role,
                text=cleaned,
            )
        )
        self._dirty = True

    def set_delegation(
        self,
        text: str | None,
    ) -> None:
        cleaned = None

        if text:
            cleaned = " ".join(str(text).split())

        self._state.last_delegation = cleaned
        self._dirty = True

    def set_error(
        self,
        message: str,
    ) -> None:
        self._state.last_error = " ".join(
            str(message).split()
        )
        self._dirty = True

    def set_usage(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self._state.usage_input = max(
            0,
            int(input_tokens or 0),
        )
        self._state.usage_output = max(
            0,
            int(output_tokens or 0),
        )
        self._dirty = True

    def print_above(
        self,
        renderable: ConsoleRenderable | str,
    ) -> None:
        """Print one durable line above the live region."""
        live = self._live

        if live is not None:
            live.console.print(renderable)
            return

        self._console.print(renderable)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _style(self) -> _StateStyle:
        return _STATE_STYLE.get(
            self._state.state,
            _DEFAULT_STATE_STYLE,
        )

    def _render_header(self) -> Table:
        """Compact product header + environment metadata."""
        style = self._style()

        table = Table.grid(
            expand=True,
            padding=(0, 1),
        )
        table.add_column(ratio=1)
        table.add_column(justify="right")

        brand = Text()
        brand.append(
            "MANA",
            style="bold bright_cyan",
        )
        brand.append(
            "  LIVE",
            style="bold white",
        )

        status = Text()
        status.append(
            f"{style.symbol} ",
            style=f"bold {style.color}",
        )
        status.append(
            style.label,
            style=f"bold {style.color}",
        )

        table.add_row(
            brand,
            status,
        )

        meta = Text()

        if self._state.model:
            meta.append(
                self._state.model,
                style="bold",
            )

        if self._state.voice:
            if len(meta):
                meta.append(
                    "  ·  ",
                    style="grey50",
                )

            meta.append(
                f"voice {self._state.voice}",
                style="grey70",
            )

        if self._state.vad_mode:
            if len(meta):
                meta.append(
                    "  ·  ",
                    style="grey50",
                )

            meta.append(
                f"vad {self._state.vad_mode}",
                style="grey70",
            )

        routing = Text()

        if self._state.delegation_model:
            routing.append(
                "backend ",
                style="grey50",
            )
            routing.append(
                self._state.delegation_model,
                style="grey78",
            )

        table.add_row(
            meta,
            routing,
        )

        return table

    def _render_wave(self) -> Text:
        """Render a subtle activity waveform for the current state."""
        style = self._style()

        text = Text(
            justify="center",
            no_wrap=True,
        )

        center = (_WAVE_WIDTH - 1) / 2

        for index in range(_WAVE_WIDTH):
            distance = abs(index - center) / max(
                center,
                1,
            )

            envelope = max(
                0.18,
                1.0 - distance * 0.58,
            )

            primary = math.sin(
                self._phase * 2.6 + index * 0.46
            )

            secondary = math.sin(
                self._phase * 4.1 + index * 0.19
            )

            wave = (
                primary * 0.68
                + secondary * 0.32
            )

            normalized = (
                0.5 + 0.5 * wave
            )

            level = (
                style.amplitude
                * envelope
                * normalized
            )

            level = max(
                0.0,
                min(1.0, level),
            )

            char_index = round(
                level
                * (len(_WAVE_CHARS) - 1)
            )

            char = _WAVE_CHARS[char_index]

            char_style = (
                style.color
                if level >= 0.11
                else "grey27"
            )

            text.append(
                char,
                style=char_style,
            )

        return text

    def _render_status_section(self) -> Group:
        style = self._style()

        waveform = Align.center(
            self._render_wave()
        )

        state_line = Text(
            justify="center",
        )
        state_line.append(
            f"{style.symbol} ",
            style=f"bold {style.color}",
        )
        state_line.append(
            style.label.title(),
            style=f"bold {style.color}",
        )

        detail = self._state_detail()

        if detail:
            state_line.append(
                f"  —  {detail}",
                style="grey62",
            )

        return Group(
            Padding(
                waveform,
                (1, 0, 0, 0),
            ),
            state_line,
        )

    def _state_detail(self) -> str:
        state = self._state.state

        if state == LiveState.INITIALIZING:
            return "Preparing session"

        if state == LiveState.CONNECTING:
            return "Opening GPT-Live"

        if state == LiveState.LISTENING:
            return "Ready for speech"

        if state == LiveState.THINKING:
            return "Understanding request"

        if state == LiveState.DELEGATING:
            return "Mana backend is working"

        if state == LiveState.SPEAKING:
            return "Responding"

        if state == LiveState.INTERRUPTED:
            return "Playback interrupted"

        if state == LiveState.RECONNECTING:
            return "Restoring connection"

        if state == LiveState.ERROR:
            return "Action required"

        if state == LiveState.SHUTDOWN:
            return "Closing session"

        return ""

    def _render_transcript(self) -> Panel:
        body = Text()

        if not self._state.transcripts:
            body.append(
                "Speak naturally — Mana is listening.",
                style="grey58 italic",
            )
        else:
            lines = list(
                self._state.transcripts
            )

            for index, line in enumerate(lines):
                is_user = line.role == "user"

                label = (
                    "YOU"
                    if is_user
                    else "MANA"
                )

                label_style = (
                    "bold cyan"
                    if is_user
                    else "bold bright_white"
                )

                body.append(
                    label,
                    style=label_style,
                )
                body.append(
                    "  ",
                    style="grey50",
                )
                body.append(
                    line.text,
                    style=(
                        "white"
                        if not is_user
                        else "grey85"
                    ),
                )

                if index < len(lines) - 1:
                    body.append("\n\n")

        callouts: list[ConsoleRenderable] = []

        if self._state.last_delegation:
            delegation = Text()
            delegation.append(
                "BACKEND  ",
                style="bold magenta",
            )
            delegation.append(
                self._state.last_delegation[:160],
                style="grey78",
            )

            callouts.append(
                Padding(
                    delegation,
                    (1, 0, 0, 0),
                )
            )

        if self._state.last_error:
            error = Text()
            error.append(
                "ERROR  ",
                style="bold bright_red",
            )
            error.append(
                self._state.last_error[:220],
                style="red",
            )

            callouts.append(
                Padding(
                    error,
                    (1, 0, 0, 0),
                )
            )

        content: ConsoleRenderable

        if callouts:
            content = Group(
                body,
                *callouts,
            )
        else:
            content = body

        return Panel(
            content,
            title=" Conversation ",
            title_align="left",
            border_style="grey35",
            box=box.ROUNDED,
            padding=(1, 2),
        )

    def _render_footer(self) -> Table:
        elapsed = max(
            0,
            int(time.monotonic() - self._start_time),
        )

        minutes, seconds = divmod(
            elapsed,
            60,
        )

        table = Table.grid(
            expand=True,
        )
        table.add_column()
        table.add_column(
            justify="center",
        )
        table.add_column(
            justify="right",
        )

        duration = Text(
            f"{minutes:02d}:{seconds:02d}",
            style="grey58",
        )

        usage = Text()

        if (
            self._state.usage_input
            or self._state.usage_output
        ):
            usage.append(
                f"in {self._state.usage_input:,}",
                style="grey58",
            )
            usage.append(
                "  ·  ",
                style="grey35",
            )
            usage.append(
                f"out {self._state.usage_output:,}",
                style="grey58",
            )
        else:
            usage.append(
                "live session",
                style="grey42",
            )

        exit_hint = Text()
        exit_hint.append(
            "Ctrl+C",
            style="bold grey70",
        )
        exit_hint.append(
            "  exit",
            style="grey48",
        )

        table.add_row(
            duration,
            usage,
            exit_hint,
        )

        return table

    def _render(self) -> Panel:
        """Render one stable outer surface instead of several moving boxes."""
        return Panel(
            Group(
                self._render_header(),
                Padding(
                    self._render_status_section(),
                    (1, 0),
                ),
                self._render_transcript(),
                Padding(
                    self._render_footer(),
                    (1, 1, 0, 1),
                ),
            ),
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
        )
