from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mana_agent.config.catalog_service import (
    ModelCatalogService,
    ModelListFetchFailedError,
    ProviderAuthenticationFailedError,
    ProviderConnectionFailedError,
    ProviderValidationError,
)
from mana_agent.config.model_catalog import (
    ModelCapability,
    ModelDescriptor,
    ModelPurpose,
    _is_model_available,
    descriptors_from_catalog,
    filter_models,
    normalize_capabilities,
    search_models,
)
from mana_agent.config.session import ConfigurationDraft
from mana_agent.media.config import MediaConfig, MediaModalityConfig
from mana_agent.media.models import MediaType
from mana_agent.tui.configuration_app import ManaConfigurationApp
from mana_agent.tui.model_picker import (
    fetch_openai_compatible_models,
    parse_openai_model_records,
)


def test_single_provider_authority_in_ui() -> None:
    """Verify Provider tab is the sole provider selector and capability tabs have no provider inputs."""
    draft = ConfigurationDraft(
        original={"MANA_AI_PROVIDER": "openai"},
        values={"MANA_AI_PROVIDER": "openai"},
    )
    app = ManaConfigurationApp(draft=draft)

    async def run() -> None:
        async with app.run_test():
            # Provider tab has provider selector and credentials
            assert app.query_one("#provider-select") is not None
            assert app.query_one("#provider-base-url") is not None
            assert app.query_one("#provider-api-key") is not None

            # Capability tabs do NOT contain their own provider/credential/base_url selectors
            for field_id in (
                "#media-image-provider",
                "#media-image-credential",
                "#media-image-base-url",
                "#media-voice-provider",
                "#media-voice-credential",
                "#media-voice-base-url",
                "#media-video-provider",
                "#media-video-credential",
                "#media-video-base-url",
                "#media-realtime-provider",
                "#media-realtime-credential",
                "#media-realtime-base-url",
                "#media-transcription-provider",
                "#media-transcription-credential",
                "#media-transcription-base-url",
            ):
                assert len(app.query(field_id)) == 0

            # Capability tabs display hint indicating active provider is used
            for hint_id in (
                "#media-image-active-provider",
                "#media-voice-active-provider",
                "#media-video-active-provider",
                "#live-active-provider",
                "#media-transcription-active-provider",
            ):
                hint = app.query_one(hint_id)
                assert "Active provider: openai" in str(hint.renderable)

    asyncio.run(run())


def test_media_config_inherits_active_provider() -> None:
    """Verify MediaConfig loads with active provider fallback when per-modality provider is absent."""
    values: dict[str, Any] = {
        "MANA_AI_PROVIDER": "openrouter",
        "media": {
            "image": {"enabled": True, "model": "gpt-image-1"},
            "voice": {"enabled": True, "model": "tts-1"},
            "video": {"enabled": False, "model": "sora-2"},
            "realtime": {"enabled": True, "model": "gpt-4o-realtime"},
            "transcription": {"enabled": True, "model": "whisper-1"},
        },
    }
    config = MediaConfig.load(values)
    assert config.image.provider == "openrouter"
    assert config.voice.provider == "openrouter"
    assert config.video.provider == "openrouter"
    assert config.realtime.provider == "openrouter"
    assert config.transcription.provider == "openrouter"

    assert config.require(MediaType.IMAGE).provider == "openrouter"
    assert config.require(MediaType.VOICE).provider == "openrouter"
    assert config.require(MediaType.REALTIME).provider == "openrouter"
    assert config.require(MediaType.TRANSCRIPTION).provider == "openrouter"


def test_dynamic_capability_detection_openai_models() -> None:
    """Verify dynamic discovery for novel models (gpt-image-2.5-sunburst, gpt-image-2.5-flare, sora-2, realtime, whisper, tts)."""
    # gpt-image-2.5-sunburst & flare
    caps_sunburst = normalize_capabilities("openai", "gpt-image-2.5-sunburst")
    assert ModelCapability.IMAGE_GENERATION in caps_sunburst
    assert ModelCapability.IMAGE_EDITING in caps_sunburst
    assert ModelCapability.TEXT_GENERATION not in caps_sunburst

    caps_flare = normalize_capabilities("openai", "gpt-image-2.5-flare")
    assert ModelCapability.IMAGE_GENERATION in caps_flare
    assert ModelCapability.IMAGE_EDITING in caps_flare

    # sora-2
    caps_sora = normalize_capabilities("openai", "sora-2")
    assert ModelCapability.VIDEO_GENERATION in caps_sora
    assert ModelCapability.IMAGE_GENERATION not in caps_sora

    # realtime
    caps_realtime = normalize_capabilities("openai", "gpt-4o-realtime-preview-2026-03")
    assert ModelCapability.REALTIME in caps_realtime

    # whisper
    caps_whisper = normalize_capabilities("openai", "whisper-large-v3")
    assert ModelCapability.SPEECH_TO_TEXT in caps_whisper
    assert ModelCapability.IMAGE_GENERATION not in caps_whisper

    # tts
    caps_tts = normalize_capabilities("openai", "tts-1-hd")
    assert ModelCapability.TEXT_TO_SPEECH in caps_tts
    assert ModelCapability.AUDIO_GENERATION in caps_tts

    # Unknown model must NOT be marked as capable of everything
    caps_unknown = normalize_capabilities("openai", "unknown-proprietary-experimental-xyz")
    assert caps_unknown == frozenset() or caps_unknown == frozenset({ModelCapability.TEXT_GENERATION})
    assert ModelCapability.IMAGE_GENERATION not in caps_unknown
    assert ModelCapability.VIDEO_GENERATION not in caps_unknown
    assert ModelCapability.EMBEDDING not in caps_unknown
    assert ModelCapability.REALTIME not in caps_unknown


def test_deprecation_and_lifecycle_metadata() -> None:
    """Verify shutdown_date and deprecated flags mark models unavailable and exclude them from capabilities."""
    past_timestamp = time.time() - 3600
    future_timestamp = time.time() + 86400 * 30

    assert _is_model_available({"shutdown_date": past_timestamp}) is False
    assert _is_model_available({"shutdown_date": future_timestamp}) is True
    assert _is_model_available({"deprecated": True}) is False
    assert _is_model_available({"status": "deprecated"}) is False
    assert _is_model_available({"status": "shutdown"}) is False
    assert _is_model_available({"shutdown_date": "2020-01-01T00:00:00Z"}) is False
    assert _is_model_available({"shutdown_date": "2020-01-01T00:00:00z"}) is False
    assert _is_model_available({"deprecation_date": "2020-01-01T00:00:00Z"}) is False
    assert _is_model_available({"shutdown_date": datetime(2020, 1, 1, tzinfo=timezone.utc)}) is False
    assert _is_model_available({"shutdown_date": "2099-01-01T00:00:00Z"}) is True
    assert _is_model_available({"shutdown_date": datetime(2099, 1, 1, tzinfo=timezone.utc)}) is True

    records = [
        {"id": "gpt-4o-active", "status": "active"},
        {"id": "gpt-4o-deprecated", "deprecated": True},
        {"id": "gpt-image-old", "shutdown_date": past_timestamp},
        {"id": "gpt-image-2.5-sunburst", "shutdown_date": future_timestamp},
    ]
    descriptors = descriptors_from_catalog("openai", records)
    desc_by_id = {d.id: d for d in descriptors}

    assert desc_by_id["gpt-4o-active"].available is True
    assert desc_by_id["gpt-4o-active"].supports(ModelPurpose.AGENT) is True

    assert desc_by_id["gpt-4o-deprecated"].available is False
    assert desc_by_id["gpt-4o-deprecated"].supports(ModelPurpose.AGENT) is False

    assert desc_by_id["gpt-image-old"].available is False
    assert desc_by_id["gpt-image-old"].supports(ModelPurpose.IMAGE) is False

    assert desc_by_id["gpt-image-2.5-sunburst"].available is True
    assert desc_by_id["gpt-image-2.5-sunburst"].supports(ModelPurpose.IMAGE) is True
    assert desc_by_id["gpt-image-2.5-sunburst"].supports(ModelPurpose.IMAGE_EDIT) is True


def test_normalized_central_catalog_filtering() -> None:
    """Verify a single normalized catalog correctly categorizes models across all capability families."""
    records = [
        {"id": "gpt-4o"},
        {"id": "o3"},
        {"id": "text-embedding-3-small"},
        {"id": "gpt-image-2.5-sunburst"},
        {"id": "sora-2"},
        {"id": "gpt-4o-realtime-preview"},
        {"id": "gpt-live-1"},
        {"id": "whisper-1"},
        {"id": "tts-1"},
    ]
    descriptors = descriptors_from_catalog("openai", records)

    agent_models = filter_models(descriptors, ModelPurpose.AGENT)
    embedding_models = filter_models(descriptors, ModelPurpose.EMBEDDING)
    image_models = filter_models(descriptors, ModelPurpose.IMAGE)
    image_edit_models = filter_models(descriptors, ModelPurpose.IMAGE_EDIT)
    video_models = filter_models(descriptors, ModelPurpose.VIDEO)
    realtime_models = filter_models(descriptors, ModelPurpose.REALTIME)
    transcription_models = filter_models(descriptors, ModelPurpose.TRANSCRIPTION)
    voice_models = filter_models(descriptors, ModelPurpose.VOICE)

    assert {m.id for m in agent_models} == {"gpt-4o", "o3"}
    assert {m.id for m in embedding_models} == {"text-embedding-3-small"}
    assert {m.id for m in image_models} == {"gpt-image-2.5-sunburst"}
    assert {m.id for m in image_edit_models} == {"gpt-image-2.5-sunburst"}
    assert {m.id for m in video_models} == {"sora-2"}
    assert {m.id for m in realtime_models} == {"gpt-4o-realtime-preview", "gpt-live-1"}
    assert {m.id for m in transcription_models} == {"whisper-1"}
    assert {m.id for m in voice_models} == {"tts-1"}


def test_gpt_live_realtime_classification() -> None:
    """Verify that gpt-live models show in REALTIME and are excluded from AGENT."""
    descriptors = descriptors_from_catalog("openai", [{"id": "gpt-live"}, {"id": "gpt-live-1"}, {"id": "gpt-4o"}])
    by_id = {d.id: d for d in descriptors}

    assert by_id["gpt-live"].supports(ModelPurpose.REALTIME) is True
    assert by_id["gpt-live"].supports(ModelPurpose.AGENT) is False
    assert by_id["gpt-live-1"].supports(ModelPurpose.REALTIME) is True
    assert by_id["gpt-live-1"].supports(ModelPurpose.AGENT) is False
    assert by_id["gpt-4o"].supports(ModelPurpose.REALTIME) is False
    assert by_id["gpt-4o"].supports(ModelPurpose.AGENT) is True


def test_model_fetch_error_differentiation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify distinct error classification: connection failed, authentication failed, model-list fetch failed."""
    import urllib.error

    # 1. Authentication Failed (HTTP 401)
    def auth_fail(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            url="https://api.openai.com/v1/models",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=MagicMock(read=lambda: b'{"error": {"message": "Invalid API key"}}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", auth_fail)
    with pytest.raises(ProviderAuthenticationFailedError) as exc_info:
        fetch_openai_compatible_models(base_url="https://api.openai.com/v1", api_key="bad-key")
    assert issubclass(ProviderAuthenticationFailedError, ProviderValidationError)
    assert "401" in str(exc_info.value) or "authentication failed" in str(exc_info.value).lower()

    # 2. Connection Failed (URLError)
    def conn_fail(*_args, **_kwargs):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr("urllib.request.urlopen", conn_fail)
    with pytest.raises(ProviderConnectionFailedError):
        fetch_openai_compatible_models(base_url="https://api.openai.com/v1", api_key="key")

    # 3. Model-List Fetch Failed (HTTP 404 or empty)
    def not_found_fail(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            url="https://api.openai.com/v1/models",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=MagicMock(read=lambda: b'{"error": {"message": "Not found"}}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", not_found_fail)
    with pytest.raises(ModelListFetchFailedError):
        fetch_openai_compatible_models(base_url="https://api.openai.com/v1", api_key="key")


def test_tui_successful_test_populates_all_capability_dropdowns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify testing the provider queries live models and updates every capability dropdown in the TUI."""
    sample_records = [
        {"id": "gpt-4o"},
        {"id": "text-embedding-3-small"},
        {"id": "gpt-image-2.5-sunburst"},
        {"id": "sora-2"},
        {"id": "gpt-4o-realtime-preview"},
        {"id": "whisper-1"},
        {"id": "tts-1"},
    ]
    mock_descriptors = descriptors_from_catalog("openai", sample_records)

    draft = ConfigurationDraft(
        original={"MANA_AI_PROVIDER": "openai"},
        values={"MANA_AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"},
    )
    app = ManaConfigurationApp(draft=draft)

    async def run() -> None:
        async with app.run_test() as pilot:
            from textual.widgets import TabbedContent

            tabs = app.query_one(TabbedContent)
            tabs.active = "providers"
            await pilot.pause()

            # Mock the catalog service refresh to return our normalized descriptors
            app.catalog_service.refresh = MagicMock(return_value=mock_descriptors)

            from textual.widgets import Button

            button = app.query_one("#test-provider", Button)
            await app.on_button_pressed(Button.Pressed(button))
            await pilot.pause()

            status = str(app.query_one("#provider-status").renderable)
            assert "Connected" in status
            assert "agent" in status
            assert "image" in status
            assert "video" in status
            assert "realtime" in status
            assert "transcription" in status

            # Check high model dropdown has gpt-4o
            high_select = app.query_one("#high-model")
            assert any(val == "gpt-4o" for _, val in high_select._options)

            # Check image model dropdown has gpt-image-2.5-sunburst
            image_select = app.query_one("#media-image-model")
            assert any(val == "gpt-image-2.5-sunburst" for _, val in image_select._options)

            # Check video model dropdown has sora-2
            video_select = app.query_one("#media-video-model")
            assert any(val == "sora-2" for _, val in video_select._options)

            # Check realtime model dropdown has gpt-4o-realtime-preview in Live tab
            realtime_select = app.query_one("#live-model")
            assert any(val == "gpt-4o-realtime-preview" for _, val in realtime_select._options)

            # Check transcription model dropdown has whisper-1
            transcription_select = app.query_one("#media-transcription-model")
            assert any(val == "whisper-1" for _, val in transcription_select._options)

            # Check voice model dropdown has tts-1
            voice_select = app.query_one("#media-voice-model")
            assert any(val == "tts-1" for _, val in voice_select._options)

    asyncio.run(run())


def test_tui_gracefully_handles_zero_models_for_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that if a provider has no models for a given capability, the dropdown gracefully shows an empty state."""
    sample_records = [
        {"id": "gpt-4o"},
    ]
    mock_descriptors = descriptors_from_catalog("openai", sample_records)

    draft = ConfigurationDraft(
        original={"MANA_AI_PROVIDER": "openai"},
        values={"MANA_AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"},
    )
    app = ManaConfigurationApp(draft=draft)

    async def run() -> None:
        async with app.run_test() as pilot:
            from textual.widgets import TabbedContent

            tabs = app.query_one(TabbedContent)
            tabs.active = "providers"
            await pilot.pause()

            app.catalog_service.refresh = MagicMock(return_value=mock_descriptors)

            from textual.widgets import Button

            button = app.query_one("#test-provider", Button)
            await app.on_button_pressed(Button.Pressed(button))
            await pilot.pause()

            # Video has no compatible models
            video_select = app.query_one("#media-video-model")
            assert any("No compatible models discovered" in label for label, _ in video_select._options)

            # Image has no compatible models
            image_select = app.query_one("#media-image-model")
            assert any("No compatible models discovered" in label for label, _ in image_select._options)

    asyncio.run(run())


def test_provider_switch_clears_models_and_invalidates_stale_selections() -> None:
    """Verify switching providers immediately clears cached models and invalidates previous selections."""
    draft = ConfigurationDraft(
        original={
            "MANA_AI_PROVIDER": "openai",
            "OPENAI_CHAT_MODEL": "gpt-4o",
            "MANA_PRIMARY_MODEL": "gpt-4o",
            "media": {
                "image": {"model": "gpt-image-2.5-sunburst", "enabled": True},
                "video": {"model": "sora-2", "enabled": True},
            },
        },
        values={
            "MANA_AI_PROVIDER": "openai",
            "OPENAI_CHAT_MODEL": "gpt-4o",
            "MANA_PRIMARY_MODEL": "gpt-4o",
            "media": {
                "image": {"model": "gpt-image-2.5-sunburst", "enabled": True},
                "video": {"model": "sora-2", "enabled": True},
            },
        },
    )
    app = ManaConfigurationApp(draft=draft)

    async def run() -> None:
        async with app.run_test() as pilot:
            from textual.widgets import Select

            # Switch provider from openai to openrouter
            provider_select = app.query_one("#provider-select", Select)
            provider_select.value = "openrouter"
            await pilot.pause()

            # In-memory models cleared
            assert app._models == []
            assert app._provider_validated is False

            # Model dropdowns reset to empty/prompt
            high_select = app.query_one("#high-model", Select)
            assert any("Test provider to discover models" in label for label, _ in high_select._options)

            image_select = app.query_one("#media-image-model", Select)
            assert any("Test provider to discover models" in label for label, _ in image_select._options)

            # Stale models in draft values purged
            assert "OPENAI_CHAT_MODEL" not in app.draft.values
            assert "MANA_PRIMARY_MODEL" not in app.draft.values
            assert app.draft.values["media"]["image"]["model"] == ""
            assert app.draft.values["media"]["video"]["model"] == ""
            assert app.draft.values["media"]["image"]["provider"] == "openrouter"

            # Hint updated across capability tabs
            hint = app.query_one("#media-image-active-provider")
            assert "Active provider: openrouter" in str(hint.renderable)

    asyncio.run(run())


def test_tab_switching_does_not_make_duplicate_api_calls() -> None:
    """Verify switching tabs does not trigger redundant API calls."""
    sample_records = [{"id": "gpt-4o"}]
    mock_descriptors = descriptors_from_catalog("openai", sample_records)

    draft = ConfigurationDraft(
        original={"MANA_AI_PROVIDER": "openai"},
        values={"MANA_AI_PROVIDER": "openai"},
    )
    app = ManaConfigurationApp(draft=draft)

    async def run() -> None:
        async with app.run_test() as pilot:
            from textual.widgets import TabbedContent

            tabs = app.query_one(TabbedContent)
            tabs.active = "providers"
            await pilot.pause()

            mock_refresh = MagicMock(return_value=mock_descriptors)
            app.catalog_service.refresh = mock_refresh

            from textual.widgets import Button

            button = app.query_one("#test-provider", Button)
            await app.on_button_pressed(Button.Pressed(button))
            await pilot.pause()
            assert mock_refresh.call_count == 1

            # Switch tabs
            from textual.widgets import TabbedContent

            tabs = app.query_one(TabbedContent)
            tabs.active = "media-image"
            await pilot.pause()
            tabs.active = "media-voice"
            await pilot.pause()
            tabs.active = "media-video"
            await pilot.pause()
            tabs.active = "live"
            await pilot.pause()
            tabs.active = "media-transcription"
            await pilot.pause()

            # No extra calls were made
            assert mock_refresh.call_count == 1

    asyncio.run(run())
