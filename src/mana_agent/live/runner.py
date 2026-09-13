"""Top-level lifecycle orchestrator for Mana-Agent Live mode.

Brings up the audio transport, GPT-Live session, adapter, and the existing
Mana-Agent gateway, then runs the Live event loop until shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path
from typing import Any

from rich.console import Console

from mana_agent.live.config import LiveConfig
from mana_agent.live.models import LiveEvent, LiveEventType, LiveState
from mana_agent.live.tui import LivePulseDisplay

logger = logging.getLogger(__name__)


class LiveRunner:
    """Orchestrate the complete lifecycle of ``mana-agent live``.

    Live mode is an additional interaction mode only. It does not replace or
    reconfigure Mana-Agent's normal gateway, routing, Codex, memory, tools,
    permissions, approvals, or task system.

    Lifecycle:
    1. Load Live configuration.
    2. Start the Live TUI.
    3. Resolve the OpenAI API key.
    4. Build the normal AgentChatGateway.
    5. Build audio transport.
    6. Start GPT-Live with native client delegation.
    7. Run the LiveAdapter.
    8. Shut everything down cleanly.
    """

    def __init__(
        self,
        *,
        settings: Any,  # mana_agent.config.settings.Settings
        workspace_path: Path | None = None,
        config_overrides: dict[str, Any] | None = None,
        console: Console | None = None,
    ) -> None:
        self._settings = settings
        self._workspace = workspace_path or Path.cwd()
        self._config_overrides = config_overrides or {}
        self._console = console or Console()

        self._config: LiveConfig | None = None
        self._gateway: Any = None
        self._session: Any = None  # LiveSession
        self._audio: Any = None  # AudioTransport
        self._adapter: Any = None  # LiveAdapter
        self._gateway_session_id: str = ""

        self._running = False
        self._stopping = False
        self._stopped = False
        self._shutdown_event = asyncio.Event()

        self._tui: LivePulseDisplay | None = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start ``mana-agent live``."""
        if self._running:
            return

        self._running = True
        self._stopping = False
        self._shutdown_event.clear()

        # 1. Load configuration.
        self._config = self._build_config()
        self._validate_audio_config(self._config)

        # 2. Start the TUI early so startup progress is visible.
        self._tui = LivePulseDisplay(console=self._console)
        self._tui.configure(
            model=self._config.realtime_model,
            # Live delegates to Mana-Agent, whose normal routing authority
            # selects the backend model. Do not advertise one forced model.
            delegation_model="Mana routing",
            voice=self._config.voice,
            vad_mode=self._config.vad_mode,
        )
        await self._tui.start()
        self._set_display_state(LiveState.INITIALIZING)

        try:
            # 3. Resolve OpenAI credentials for GPT-Live only.
            api_key = self._resolve_api_key()

            # 4. Reuse the exact normal Mana-Agent gateway/routing stack.
            self._gateway, self._gateway_session_id = self._build_gateway()

            # 5. Create the local audio transport.
            from mana_agent.live.audio_transport import AudioTransport

            self._audio = AudioTransport(
                input_sample_rate=self._config.input_sample_rate,
                output_sample_rate=self._config.output_sample_rate,
                input_device=self._config.input_device,
                output_device=self._config.output_device,
            )

            # 6. Create the GPT-Live session.
            #
            # A primary GPT-Live WebSocket has one shared audio.format for
            # input and output, so its sample rate must match the local raw
            # audio streams.
            from mana_agent.live.session import LiveSession

            self._session = LiveSession(
                api_key=api_key,
                model=self._config.realtime_model,
                voice=self._config.voice,
                vad_mode=self._config.vad_mode,
                vad_threshold=self._config.vad_threshold,
                turn_detection_silence_ms=self._config.turn_detection_silence_ms,
                audio_format=self._config.audio_format,
                audio_sample_rate=self._config.input_sample_rate,
                max_response_output_tokens=self._config.max_response_output_tokens,
                auto_reconnect=self._config.auto_reconnect,
                reconnect_delay_seconds=self._config.reconnect_delay_seconds,
                reconnect_max_attempts=self._config.reconnect_max_attempts,
                system_instructions=self._build_spirit_instructions(),
            )

            # 7. The adapter bridges GPT-Live events/audio/delegations to the
            # existing Mana-Agent gateway.
            from mana_agent.live.adapter import LiveAdapter

            self._adapter = LiveAdapter(
                session=self._session,
                audio_transport=self._audio,
                gateway=self._gateway,
                gateway_session_id=self._gateway_session_id,
                event_callback=self._on_live_event,
            )

            # 8. Connect before opening microphone/playback streams.
            self._set_display_state(LiveState.CONNECTING)
            await self._session.connect()

            await self._audio.start_capture()
            await self._audio.start_playback()

            # 9. Register process shutdown handlers after successful startup.
            self._register_signals()

            self._set_display_state(LiveState.LISTENING)

            # 10. Run until adapter/session shutdown.
            try:
                await self._adapter.run()
            except asyncio.CancelledError:
                # Cancellation is part of normal shutdown.
                pass

        except Exception as exc:
            self._set_error(str(exc))
            self._set_display_state(LiveState.ERROR)
            raise

        finally:
            await self.stop()

    async def stop(self) -> None:
        """Perform idempotent clean shutdown."""
        if self._stopping or self._stopped:
            return

        self._stopping = True
        self._running = False
        self._shutdown_event.set()

        try:
            self._set_display_state(LiveState.SHUTDOWN)

            # Ask the adapter to stop producing/consuming new work first.
            if self._adapter is not None:
                try:
                    await self._adapter.shutdown()
                except Exception:
                    logger.debug(
                        "Live adapter shutdown failed",
                        exc_info=True,
                    )

            # Stop microphone/playback so no more media is queued while the
            # Live session is closing.
            if self._audio is not None:
                try:
                    await self._audio.cleanup()
                except Exception:
                    logger.debug(
                        "Live audio cleanup failed",
                        exc_info=True,
                    )

            # Request GPT-Live session.close and let LiveSession perform its
            # graceful close/final-usage handling.
            if self._session is not None:
                try:
                    await self._session.disconnect()
                except Exception:
                    logger.debug(
                        "GPT-Live disconnect failed",
                        exc_info=True,
                    )

            # Persist/log transcript state after the adapter is no longer
            # mutating it.
            if self._adapter is not None:
                self._sync_transcripts()

            usage_input = 0
            usage_output = 0

            if self._adapter is not None:
                try:
                    usage = self._adapter.usage
                    usage_input = int(getattr(usage, "input_tokens", 0) or 0)
                    usage_output = int(getattr(usage, "output_tokens", 0) or 0)
                except Exception:
                    logger.debug(
                        "Failed to read Live adapter usage",
                        exc_info=True,
                    )

            if self._tui is not None:
                try:
                    self._tui.set_usage(usage_input, usage_output)
                    await self._tui.stop()
                finally:
                    self._tui = None

            self._console.print(
                "\n[dim]"
                f"Session usage: {usage_input} input tokens, "
                f"{usage_output} output tokens"
                "[/dim]"
            )
            self._console.print(
                "[bold cyan]Live mode ended.[/bold cyan]"
            )

            logger.info("Live mode shutdown complete")

        finally:
            self._stopping = False
            self._stopped = True

    # ------------------------------------------------------------------
    # Configuration / construction
    # ------------------------------------------------------------------

    def _build_config(self) -> LiveConfig:
        """Build LiveConfig from settings and CLI overrides."""
        config = LiveConfig.from_mana_settings(self._settings)

        if self._config_overrides:
            merged = config.model_dump()
            merged.update(self._config_overrides)
            config = LiveConfig.model_validate(merged)

        return config

    @staticmethod
    def _validate_audio_config(config: LiveConfig) -> None:
        """Validate the raw WebSocket audio contract.

        GPT-Live primary WebSocket sessions configure one audio format/rate
        shared by input and output. The current AudioTransport exposes separate
        capture/playback rates but does not provide a resampler here, so fail
        clearly instead of silently playing or sending audio at the wrong rate.
        """
        input_rate = int(config.input_sample_rate)
        output_rate = int(config.output_sample_rate)

        if input_rate != output_rate:
            raise RuntimeError(
                "GPT-Live WebSocket mode requires matching input and output "
                "sample rates unless AudioTransport performs explicit "
                f"resampling. Got input={input_rate} Hz and "
                f"output={output_rate} Hz."
            )

        audio_format = str(config.audio_format).strip().lower().replace("-", "_")

        if audio_format in {
            "pcm16",
            "pcm",
            "audio/pcm",
            "audio_pcm",
        } and input_rate not in {16_000, 24_000}:
            raise RuntimeError(
                "GPT-Live PCM WebSocket audio must use 16000 or 24000 Hz. "
                f"Got {input_rate} Hz."
            )

    def _resolve_api_key(self) -> str:
        """Resolve the OpenAI API key used by GPT-Live."""
        from mana_agent.config.inference_provider import resolve_inference_connection

        connection = resolve_inference_connection(
            self._settings,
            provider="openai",
        )

        if not connection or not connection.api_key:
            raise RuntimeError(
                "Live mode requires an OpenAI API key.\n"
                "Configure it with: mana-agent configure"
            )

        return connection.api_key

    def _build_gateway(self) -> tuple[Any, str]:
        """Build the normal Mana-Agent gateway without overriding routing.

        Live mode is only an interaction mode. Backend work delegated by
        GPT-Live must enter the same AgentChatGateway and routing authority as
        normal Mana-Agent turns.
        """
        from mana_agent.gateway import AgentChatGateway

        settings = self._settings
        delegation_model = getattr(self._config, "delegation_model", None)
        if delegation_model and settings is not None:
            if hasattr(settings, "model_copy"):
                settings = settings.model_copy(
                    update={
                        "openai_chat_model": delegation_model,
                        "mana_codex_model": delegation_model,
                        "openai_coding_planner_model": delegation_model,
                    }
                )
            elif hasattr(settings, "copy"):
                settings = settings.copy(
                    update={
                        "openai_chat_model": delegation_model,
                        "mana_codex_model": delegation_model,
                        "openai_coding_planner_model": delegation_model,
                    }
                )
            else:
                try:
                    settings.openai_chat_model = delegation_model
                    settings.mana_codex_model = delegation_model
                    settings.openai_coding_planner_model = delegation_model
                except Exception:
                    pass

        gateway = AgentChatGateway(
            root=str(self._workspace),
            settings=settings,
        )

        session_id = gateway.create_session(frontend="live")
        return gateway, session_id

    def _build_spirit_instructions(self) -> str:
        """Build GPT-Live frontend conversation instructions.

        These instructions describe voice interaction and native client
        delegation only. Backend business/tool/coding behavior remains owned by
        Mana-Agent after a delegation enters AgentChatGateway.
        """
        try:
            from mana_agent.spirit.compiler import compile_spirit_instruction
            from mana_agent.spirit.registry import resolve_configured_spirit

            spirit = resolve_configured_spirit(self._settings)
            identity = compile_spirit_instruction(spirit)
        except Exception:
            logger.debug(
                "Failed to compile configured spirit for Live mode",
                exc_info=True,
            )
            identity = "You are Mana-Agent, an AI assistant."

        return (
            f"{identity}\n\n"
            "You are running in Mana-Agent Live voice mode. The user is "
            "speaking to you through a microphone. Respond naturally and "
            "conversationally with spoken audio.\n\n"
            "For greetings, normal conversation, and simple questions that "
            "do not require external work, answer the user directly and "
            "immediately. Do not delegate simple conversation.\n\n"
            "When a request requires backend work such as coding, repository "
            "or file access, tools, web research, memory, planning, execution, "
            "or another Mana-Agent capability, use GPT-Live native client "
            "delegation so the Mana-Agent application can execute it.\n\n"
            "Client delegation is built into GPT-Live. There is no "
            "delegate_to_mana function. Never invent, request, or mention a "
            "function named delegate_to_mana.\n\n"
            "When the application supplies commentary for a delegation, "
            "communicate the result clearly and naturally to the user.\n\n"
            "If the user explicitly asks to cancel or stop an active backend "
            "task, delegate that request to the client application so "
            "Mana-Agent can use its existing validated task controls.\n\n"
            "Do not claim that backend work succeeded until the application "
            "returns an authoritative result."
        )

    # ------------------------------------------------------------------
    # Transcript / event handling
    # ------------------------------------------------------------------

    def _sync_transcripts(self) -> None:
        """Synchronize/log Live transcript entries for the Mana session.

        The actual persistence mechanism remains owned by the existing gateway
        integration. Do not invent a second session store here.
        """
        if self._adapter is None or self._gateway is None:
            return

        try:
            for transcript in self._adapter.transcripts:
                role = "human" if transcript.role == "user" else "ai"
                text = str(getattr(transcript, "text", ""))

                logger.debug(
                    "Syncing live transcript: role=%s text=%s",
                    role,
                    text[:80],
                )
        except Exception:
            logger.debug(
                "Failed to sync live transcripts",
                exc_info=True,
            )

    def _on_live_event(self, event: LiveEvent) -> None:
        """Feed LiveAdapter events into the pulse TUI."""
        if self._tui is None:
            return

        if (
            event.event_type == LiveEventType.STATE_CHANGED
            and event.state
        ):
            self._set_display_state(event.state)

        elif event.event_type == LiveEventType.TRANSCRIPT_USER:
            text = event.data.get("text", "")
            if text:
                self._tui.add_transcript("user", text)

        elif event.event_type == LiveEventType.TRANSCRIPT_ASSISTANT:
            text = event.data.get("text", "")
            if text:
                self._tui.add_transcript("assistant", text)

        elif event.event_type == LiveEventType.DELEGATION_STARTED:
            text = event.data.get("text", "")
            self._tui.set_delegation(text)

        elif event.event_type == LiveEventType.DELEGATION_COMPLETED:
            self._tui.set_delegation(None)

        elif event.event_type == LiveEventType.DELEGATION_ERROR:
            error = event.data.get("error", "unknown")
            self._set_error(f"Delegation error: {error}")

        elif event.event_type == LiveEventType.AUDIO_ERROR:
            error = event.data.get("error", {})
            self._set_error(f"Audio error: {error}")

    # ------------------------------------------------------------------
    # Display / signals
    # ------------------------------------------------------------------

    def _set_display_state(self, state: LiveState) -> None:
        """Update the animated pulse display state."""
        if self._tui is not None:
            self._tui.set_state(state)
        else:
            logger.debug(
                "Live state changed to %s (TUI not active)",
                state.value,
            )

    def _set_error(self, message: str) -> None:
        """Surface an error in the Live TUI."""
        if self._tui is not None:
            self._tui.set_error(message)
        else:
            logger.error(message)

    def _register_signals(self) -> None:
        """Register signal handlers for clean shutdown."""
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig,
                    self._signal_handler,
                )
            except (NotImplementedError, RuntimeError):
                # Windows and some embedded event loops don't support this.
                pass

    def _signal_handler(self) -> None:
        """Handle SIGINT/SIGTERM without bypassing normal cleanup."""
        logger.info("Live mode shutdown signal received")

        self._running = False
        self._shutdown_event.set()

        if self._adapter is not None:
            asyncio.create_task(
                self._adapter.shutdown()
            )
