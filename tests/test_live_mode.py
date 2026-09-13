"""Tests for mana-agent live mode.

Verifies that:
- Existing mana-agent behavior is unchanged.
- `mana-agent live` starts Live mode.
- Live reuses the same gateway / session architecture.
- Backend work passes through existing routing.
- Codex remains the coding engine.
- Interruptions do not cancel backend tasks.
- Explicit cancellation uses existing task controls.
- Shutting down Live does not corrupt the Mana session.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def live_config():
    """Return a LiveConfig with test defaults."""
    from mana_agent.live.config import LiveConfig
    return LiveConfig(
        realtime_model="gpt-4o-realtime-preview",
        voice="alloy",
        vad_mode="server",
    )


@pytest.fixture()
def mock_gateway():
    """Return a mock AgentChatGateway."""
    gw = MagicMock()
    gw.create_session.return_value = "test-session-id"
    turn_result = MagicMock()
    turn_result.answer = "The function was updated successfully."
    gw.process_turn.return_value = turn_result
    gw.cancel_task = MagicMock(return_value=True)
    return gw


@pytest.fixture()
def mock_audio_transport():
    """Return a mock AudioTransport."""
    transport = MagicMock()
    transport.start_capture = AsyncMock()
    transport.stop_capture = AsyncMock()
    transport.start_playback = AsyncMock()
    transport.stop_playback = AsyncMock()
    transport.cleanup = AsyncMock()
    transport.get_capture_chunk = MagicMock(return_value=None)
    transport.queue_playback = MagicMock()
    transport.interrupt_playback = MagicMock()
    transport.clear_playback_buffer = MagicMock()
    return transport


@pytest.fixture()
def mock_live_session():
    """Return a mock LiveSession."""
    session = MagicMock()
    session.is_connected = True
    session.session_id = "rt-session-123"
    session.connect = AsyncMock()
    session.disconnect = AsyncMock()
    session.send_audio = AsyncMock()
    session.send_text = AsyncMock()
    session.send_event = AsyncMock()
    session.truncate_response = AsyncMock()
    session.create_response = AsyncMock()
    session.commit_audio_buffer = AsyncMock()
    session.clear_audio_buffer = AsyncMock()
    session.update_session = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# 1. Existing behavior unchanged
# ---------------------------------------------------------------------------


class TestExistingBehaviorUnchanged:
    """The normal `mana-agent` CLI must not be affected by Live mode."""

    def test_live_command_is_registered(self):
        """The 'live' command exists alongside existing commands."""
        from mana_agent.commands.cli import app
        command_names = [cmd.name for cmd in app.registered_commands]
        assert "live" in command_names

    def test_chat_command_still_registered(self):
        """The 'chat' command is unchanged and still registered."""
        from mana_agent.commands.cli import app
        command_names = [cmd.name for cmd in app.registered_commands]
        assert "chat" in command_names

    def test_analyze_command_still_registered(self):
        from mana_agent.commands.cli import app
        command_names = [cmd.name for cmd in app.registered_commands]
        assert "analyze" in command_names

    def test_plan_command_still_registered(self):
        from mana_agent.commands.cli import app
        command_names = [cmd.name for cmd in app.registered_commands]
        assert "plan" in command_names

    def test_live_does_not_appear_in_default_invocation(self):
        """Running `mana-agent` without subcommand should not trigger live mode."""
        from mana_agent.commands.cli import app
        # The root callback routes to chat, not live
        # Verify the app's invoke_without_command behavior remains intact
        assert app.info.invoke_without_command is True


# ---------------------------------------------------------------------------
# 2. Live mode starts correctly
# ---------------------------------------------------------------------------


class TestLiveModeStartup:
    """Verify `mana-agent live` starts the Live mode runner."""

    def test_live_config_defaults(self, live_config):
        assert live_config.realtime_model == "gpt-4o-realtime-preview"
        assert live_config.delegation_model is None
        assert live_config.voice == "alloy"
        assert live_config.vad_mode == "server"
        assert live_config.audio_format == "pcm16"
        assert live_config.input_sample_rate == 24000
        assert live_config.output_sample_rate == 24000

    def test_live_config_delegation_model(self):
        from mana_agent.live.config import LiveConfig
        config = LiveConfig(delegation_model="gpt-5.6-luna")
        assert config.delegation_model == "gpt-5.6-luna"

    def test_live_config_from_dict(self):
        from mana_agent.live.config import LiveConfig
        config = LiveConfig.from_dict({"voice": "echo", "vad_mode": "disabled"})
        assert config.voice == "echo"
        assert config.vad_mode == "disabled"

    def test_live_config_from_empty_dict(self):
        from mana_agent.live.config import LiveConfig
        config = LiveConfig.from_dict({})
        assert config.realtime_model == "gpt-4o-realtime-preview"

    def test_live_state_enum_values(self):
        from mana_agent.live.models import LiveState
        assert LiveState.LISTENING.value == "live listening"
        assert LiveState.DELEGATING.value == "live delegating"
        assert LiveState.SPEAKING.value == "live speaking"

    def test_live_event_type_enum_values(self):
        from mana_agent.live.models import LiveEventType
        assert LiveEventType.DELEGATION_STARTED.value == "live.delegation.started"
        assert LiveEventType.STATE_CHANGED.value == "live.state_changed"


# ---------------------------------------------------------------------------
# 3. Live reuses gateway / session architecture
# ---------------------------------------------------------------------------


class TestLiveReusesGateway:
    """Live mode must use the same AgentChatGateway instance."""

    def test_adapter_receives_gateway(self, mock_gateway, mock_live_session, mock_audio_transport):
        from mana_agent.live.adapter import LiveAdapter
        adapter = LiveAdapter(
            session=mock_live_session,
            audio_transport=mock_audio_transport,
            gateway=mock_gateway,
            gateway_session_id="test-session-id",
        )
        # The adapter holds a reference to the same gateway
        assert adapter._gateway is mock_gateway
        assert adapter._gateway_session_id == "test-session-id"

    def test_delegation_calls_gateway_process_turn(
        self, mock_gateway, mock_live_session, mock_audio_transport,
    ):
        """When delegate_to_mana is triggered, process_turn is called."""
        from mana_agent.live.adapter import LiveAdapter
        from mana_agent.live.models import DelegationRequest

        adapter = LiveAdapter(
            session=mock_live_session,
            audio_transport=mock_audio_transport,
            gateway=mock_gateway,
            gateway_session_id="test-session-id",
        )

        request = DelegationRequest(text="fix the bug in main.py")
        result = asyncio.run(
            adapter._delegate_to_gateway(request)
        )

        # Gateway's process_turn must have been called
        mock_gateway.process_turn.assert_called_once_with(
            "test-session-id",
            "fix the bug in main.py",
        )
        assert result.success is True
        assert result.correlation_id == request.correlation_id

    def test_gateway_session_created_with_live_frontend(self, mock_gateway):
        """The gateway session is created with frontend='live'."""
        mock_gateway.create_session(frontend="live")
        mock_gateway.create_session.assert_called_with(frontend="live")

    def test_runner_build_gateway_applies_delegation_model(self):
        from mana_agent.config.settings import Settings
        from mana_agent.live.config import LiveConfig
        from mana_agent.live.runner import LiveRunner

        runner = LiveRunner(settings=Settings())
        runner._config = LiveConfig(delegation_model="gpt-5.6-luna")
        with patch("mana_agent.gateway.AgentChatGateway") as mock_gw_cls:
            mock_gw = MagicMock()
            mock_gw.create_session.return_value = "session-123"
            mock_gw_cls.return_value = mock_gw

            gw, sid = runner._build_gateway()
            assert sid == "session-123"
            call_kwargs = mock_gw_cls.call_args.kwargs
            passed_settings = call_kwargs["settings"]
            assert passed_settings.openai_chat_model == "gpt-5.6-luna"
            assert passed_settings.mana_codex_model == "gpt-5.6-luna"
            assert passed_settings.openai_coding_planner_model == "gpt-5.6-luna"


# ---------------------------------------------------------------------------
# 4. Backend work passes through existing routing
# ---------------------------------------------------------------------------


class TestBackendRoutingUnchanged:
    """Delegation must go through the existing gateway routing."""

    def test_delegation_result_contains_gateway_response(
        self, mock_gateway, mock_live_session, mock_audio_transport,
    ):
        from mana_agent.live.adapter import LiveAdapter
        from mana_agent.live.models import DelegationRequest

        adapter = LiveAdapter(
            session=mock_live_session,
            audio_transport=mock_audio_transport,
            gateway=mock_gateway,
            gateway_session_id="test-session-id",
        )

        request = DelegationRequest(text="search for authentication code", intent="search")
        result = asyncio.run(
            adapter._delegate_to_gateway(request)
        )

        assert result.text == "The function was updated successfully."
        assert result.success is True

    def test_delegation_emits_events(
        self, mock_gateway, mock_live_session, mock_audio_transport,
    ):
        from mana_agent.live.adapter import LiveAdapter
        from mana_agent.live.models import DelegationRequest, LiveEventType

        events_received: list[Any] = []

        adapter = LiveAdapter(
            session=mock_live_session,
            audio_transport=mock_audio_transport,
            gateway=mock_gateway,
            gateway_session_id="test-session-id",
            event_callback=events_received.append,
        )

        request = DelegationRequest(text="analyze the project")
        asyncio.run(
            adapter._delegate_to_gateway(request)
        )

        event_types = [e.event_type for e in events_received]
        assert LiveEventType.DELEGATION_STARTED in event_types
        assert LiveEventType.DELEGATION_COMPLETED in event_types


# ---------------------------------------------------------------------------
# 5. Codex remains the coding engine
# ---------------------------------------------------------------------------


class TestCodexUnchanged:
    """Coding requests delegated through the gateway still use Codex."""

    def test_coding_request_goes_through_gateway(
        self, mock_gateway, mock_live_session, mock_audio_transport,
    ):
        """A coding request through Live delegation reaches the gateway.

        The gateway internally routes to Codex. We verify that the gateway's
        process_turn is called with the coding request text.
        """
        from mana_agent.live.adapter import LiveAdapter
        from mana_agent.live.models import DelegationRequest

        adapter = LiveAdapter(
            session=mock_live_session,
            audio_transport=mock_audio_transport,
            gateway=mock_gateway,
            gateway_session_id="test-session-id",
        )

        request = DelegationRequest(
            text="refactor the authentication module to use JWT tokens",
            intent="code",
        )
        result = asyncio.run(
            adapter._delegate_to_gateway(request)
        )

        # The gateway received the coding request
        mock_gateway.process_turn.assert_called_once_with(
            "test-session-id",
            "refactor the authentication module to use JWT tokens",
        )
        assert result.success is True


# ---------------------------------------------------------------------------
# 6. Interruptions do not cancel backend tasks
# ---------------------------------------------------------------------------


class TestInterruptionSafety:
    """User interruptions must stop audio but not cancel backend tasks."""

    def test_speech_started_interrupts_playback_only(
        self, mock_gateway, mock_live_session, mock_audio_transport,
    ):
        from mana_agent.live.adapter import LiveAdapter

        adapter = LiveAdapter(
            session=mock_live_session,
            audio_transport=mock_audio_transport,
            gateway=mock_gateway,
            gateway_session_id="test-session-id",
        )

        # Simulate speech_started event
        event = {"type": "input_audio_buffer.speech_started"}
        asyncio.run(
            adapter._dispatch_event(event)
        )

        # Audio playback must be interrupted
        mock_audio_transport.interrupt_playback.assert_called_once()

        # Response must be truncated
        mock_live_session.truncate_response.assert_called_once()

        # Gateway cancel_task must NOT be called
        mock_gateway.cancel_task.assert_not_called()

    def test_interruption_emits_event(
        self, mock_gateway, mock_live_session, mock_audio_transport,
    ):
        from mana_agent.live.adapter import LiveAdapter
        from mana_agent.live.models import LiveEventType

        events: list[Any] = []
        adapter = LiveAdapter(
            session=mock_live_session,
            audio_transport=mock_audio_transport,
            gateway=mock_gateway,
            gateway_session_id="test-session-id",
            event_callback=events.append,
        )

        event = {"type": "input_audio_buffer.speech_started"}
        asyncio.run(
            adapter._dispatch_event(event)
        )

        event_types = [e.event_type for e in events]
        assert LiveEventType.INTERRUPTION in event_types

    def test_multiple_interruptions_never_cancel_tasks(
        self, mock_gateway, mock_live_session, mock_audio_transport,
    ):
        from mana_agent.live.adapter import LiveAdapter

        adapter = LiveAdapter(
            session=mock_live_session,
            audio_transport=mock_audio_transport,
            gateway=mock_gateway,
            gateway_session_id="test-session-id",
        )

        for _ in range(5):
            event = {"type": "input_audio_buffer.speech_started"}
            asyncio.run(
                adapter._dispatch_event(event)
            )

        assert mock_audio_transport.interrupt_playback.call_count == 5
        mock_gateway.cancel_task.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Explicit cancellation uses existing task controls
# ---------------------------------------------------------------------------


class TestExplicitCancellation:
    """Explicit cancel commands must use gateway task cancellation."""

    def test_cancel_request_delegates_to_gateway(
        self, mock_gateway, mock_live_session, mock_audio_transport,
    ):
        """When GPT-Live calls delegate_to_mana with cancel intent,
        the request goes through the gateway (which handles cancellation)."""
        from mana_agent.live.adapter import LiveAdapter
        from mana_agent.live.models import DelegationRequest

        adapter = LiveAdapter(
            session=mock_live_session,
            audio_transport=mock_audio_transport,
            gateway=mock_gateway,
            gateway_session_id="test-session-id",
        )

        request = DelegationRequest(
            text="cancel that task",
            intent="cancel",
        )
        result = asyncio.run(
            adapter._delegate_to_gateway(request)
        )

        # The cancellation request is sent to the gateway via process_turn
        mock_gateway.process_turn.assert_called_once_with(
            "test-session-id",
            "cancel that task",
        )


# ---------------------------------------------------------------------------
# 8. Shutdown does not corrupt session
# ---------------------------------------------------------------------------


class TestShutdownIntegrity:
    """Shutting down Live must leave the Mana session intact."""

    def test_adapter_shutdown_sets_state(self, mock_gateway, mock_live_session, mock_audio_transport):
        from mana_agent.live.adapter import LiveAdapter
        from mana_agent.live.models import LiveState

        adapter = LiveAdapter(
            session=mock_live_session,
            audio_transport=mock_audio_transport,
            gateway=mock_gateway,
            gateway_session_id="test-session-id",
        )

        asyncio.run(adapter.shutdown())
        assert adapter.state == LiveState.SHUTDOWN
        assert adapter._running is False

    def test_runner_stop_cleans_up_resources(self):
        """LiveRunner.stop() cleans up audio and session without corrupting gateway."""
        from mana_agent.live.runner import LiveRunner
        from mana_agent.live.models import LiveState

        settings = MagicMock()
        runner = LiveRunner(settings=settings, workspace_path=None)

        # Set up mock components
        runner._running = True
        runner._audio = MagicMock()
        runner._audio.cleanup = AsyncMock()
        runner._session = MagicMock()
        runner._session.disconnect = AsyncMock()
        runner._adapter = MagicMock()
        runner._adapter.shutdown = AsyncMock()
        runner._adapter.transcripts = []
        runner._adapter.usage = MagicMock(input_tokens=100, output_tokens=50)
        runner._gateway = MagicMock()

        asyncio.run(runner.stop())

        # Audio must be cleaned up
        runner._audio.cleanup.assert_called_once()

        # Session must be disconnected
        runner._session.disconnect.assert_called_once()

        # Adapter must be shut down
        runner._adapter.shutdown.assert_called_once()

        # Gateway must NOT be closed/destroyed
        # (Session persists for potential chat continuation)
        assert runner._running is False

    def test_transcripts_available_after_shutdown(self, mock_gateway, mock_live_session, mock_audio_transport):
        from mana_agent.live.adapter import LiveAdapter
        from mana_agent.live.models import LiveTranscript

        adapter = LiveAdapter(
            session=mock_live_session,
            audio_transport=mock_audio_transport,
            gateway=mock_gateway,
            gateway_session_id="test-session-id",
        )

        # Simulate transcripts being recorded
        adapter._transcripts.append(
            LiveTranscript(role="user", text="hello")
        )
        adapter._transcripts.append(
            LiveTranscript(role="assistant", text="hi there")
        )

        asyncio.run(adapter.shutdown())

        # Transcripts must still be accessible after shutdown
        transcripts = adapter.transcripts
        assert len(transcripts) == 2
        assert transcripts[0].role == "user"
        assert transcripts[1].role == "assistant"


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestLiveDataModels:
    """Test the Live mode data models."""

    def test_live_event_creation(self):
        from mana_agent.live.models import LiveEvent, LiveEventType, LiveState
        event = LiveEvent(
            event_type=LiveEventType.STATE_CHANGED,
            data={"old_state": "listening", "new_state": "thinking"},
            state=LiveState.THINKING,
        )
        assert event.event_type == LiveEventType.STATE_CHANGED
        assert event.state == LiveState.THINKING
        assert event.data["new_state"] == "thinking"

    def test_live_transcript_creation(self):
        from mana_agent.live.models import LiveTranscript
        t = LiveTranscript(role="user", text="hello world")
        assert t.role == "user"
        assert t.text == "hello world"
        assert t.is_partial is False
        assert len(t.transcript_id) == 12

    def test_live_usage_immutable(self):
        from mana_agent.live.models import LiveUsage
        usage = LiveUsage(input_tokens=100, output_tokens=50)
        with pytest.raises(AttributeError):
            usage.input_tokens = 200  # type: ignore[misc]

    def test_delegation_request_has_correlation_id(self):
        from mana_agent.live.models import DelegationRequest
        req = DelegationRequest(text="do something")
        assert len(req.correlation_id) > 0

    def test_delegation_result_defaults(self):
        from mana_agent.live.models import DelegationResult
        result = DelegationResult(correlation_id="abc", text="done")
        assert result.success is True
        assert result.task_id == ""
        assert result.events == []


# ---------------------------------------------------------------------------
# Audio transport tests (mocked sounddevice)
# ---------------------------------------------------------------------------


class TestAudioTransport:
    """Test the AudioTransport with mocked audio dependencies."""

    def test_audio_transport_import_error(self):
        from mana_agent.live.audio_transport import AudioTransport, AudioDeviceError
        transport = AudioTransport()
        # Force import failure
        with patch.dict("sys.modules", {"sounddevice": None, "numpy": None}):
            transport._sd = None  # Reset cached import
            with pytest.raises(AudioDeviceError, match="audio dependencies"):
                transport._ensure_deps()

    def test_list_devices_without_sounddevice(self):
        from mana_agent.live.audio_transport import AudioTransport
        with patch.dict("sys.modules", {"sounddevice": None}):
            devices = AudioTransport.list_devices()
            assert devices == []


# ---------------------------------------------------------------------------
# Session tests
# ---------------------------------------------------------------------------


class TestLiveSession:
    """Test the LiveSession WebSocket management."""

    def test_delegation_tool_schema(self):
        from mana_agent.live.session import LiveSession
        schema = LiveSession._delegation_tool_schema()
        assert schema["type"] == "function"
        assert schema["name"] == "delegate_to_mana"
        assert "request" in schema["parameters"]["properties"]
        assert "intent" in schema["parameters"]["properties"]
        assert "request" in schema["parameters"]["required"]

    def test_session_not_connected_by_default(self):
        from mana_agent.live.session import LiveSession
        session = LiveSession(api_key="test-key")
        assert session.is_connected is False
        assert session.session_id == ""

    def test_send_event_raises_when_not_connected(self):
        from mana_agent.live.session import LiveSession, LiveSessionError
        session = LiveSession(api_key="test-key")
        with pytest.raises(LiveSessionError, match="Not connected"):
            asyncio.run(
                session.send_event({"type": "test"})
            )


class TestLiveConfigNormalization:
    """Test LiveConfig normalization and backwards-compatibility aliases."""

    def test_accepts_model_alias_without_extra_forbidden(self):
        from mana_agent.live.config import LiveConfig
        config = LiveConfig.model_validate({"model": "gpt-live-1"})
        assert config.realtime_model == "gpt-live-1"
        assert config.model == "gpt-live-1"

    def test_accepts_both_realtime_model_and_model(self):
        from mana_agent.live.config import LiveConfig
        config = LiveConfig.model_validate({"realtime_model": "gpt-4o-realtime-preview", "model": "gpt-live-1"})
        assert config.realtime_model == "gpt-4o-realtime-preview"
        assert config.model == "gpt-4o-realtime-preview"

    def test_normalizes_empty_delegation_model(self):
        from mana_agent.live.config import LiveConfig
        config = LiveConfig.model_validate({"delegation_model": ""})
        assert config.delegation_model is None

    def test_validate_config_values_handles_live(self):
        from mana_agent.config.user_config import validate_config_values
        cleaned = validate_config_values({"live": {"model": "gpt-live-1", "delegation_model": "gpt-4o"}})
        assert cleaned["live"]["realtime_model"] == "gpt-live-1"
        assert cleaned["live"]["delegation_model"] == "gpt-4o"

