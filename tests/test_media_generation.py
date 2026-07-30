from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from mana_agent.config.model_catalog import (
    ModelPurpose,
    descriptors_from_catalog,
    filter_models,
    search_models,
)
from mana_agent.config.user_config import (
    load_effective_settings,
    masked_config_summary,
    save_effective_user_config,
    validate_config_values,
)
from mana_agent.media.artifacts import MediaArtifactStore
from mana_agent.media.config import MediaConfig
from mana_agent.media.errors import (
    MediaArtifactError,
    MediaCapabilityError,
    MediaConfigurationError,
    MediaValidationError,
    MediaProviderError,
)
from mana_agent.media.models import (
    GenerationStatus,
    ImageGenerationRequest,
    MediaType,
    VideoGenerationRequest,
    VoiceGenerationRequest,
)
from mana_agent.media.providers.base import ProviderOutput
from mana_agent.media.providers.openai import OpenAIMediaProvider
from mana_agent.media.registry import MediaProviderRegistry
from mana_agent.media.service import MediaService
from mana_agent.gateway.entry_routing import (
    EntryRouteContext,
    EntryRouteRegistry,
    EntryRouter,
    EntryRoutingDecision,
    EntryRoutingError,
    RouteAvailability,
    RouteRegistration,
)
from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.gateway.lanes import (
    LaneId,
    default_lane_contracts,
    select_lane,
    validate_tool_permission,
)


PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 32
MP3 = b"ID3" + b"x" * 32
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"x" * 32


class FakeProvider:
    provider_id = "openai"

    def __init__(self) -> None:
        self.last_image_references = ()

    def capabilities(self, model: str):
        from mana_agent.config.model_catalog import normalize_capabilities

        return normalize_capabilities("openai", model)

    def generate_image(self, request, reference_artifacts=()):
        self.last_image_references = reference_artifacts
        return ProviderOutput("req-image", GenerationStatus.COMPLETED, (PNG,), ("image/png",))

    def generate_speech(self, request):
        return ProviderOutput("req-voice", GenerationStatus.COMPLETED, (MP3,), ("audio/mpeg",))

    def generate_video(self, request, reference_artifacts=()):
        return ProviderOutput("req-video", GenerationStatus.QUEUED)

    def get_generation_status(self, provider_request_id):
        return ProviderOutput(provider_request_id, GenerationStatus.COMPLETED)

    def cancel_generation(self, provider_request_id):
        return ProviderOutput(provider_request_id, GenerationStatus.CANCELLED)

    def download_result(self, provider_request_id):
        return ProviderOutput(provider_request_id, GenerationStatus.COMPLETED, (MP4,), ("video/mp4",))


@pytest.fixture()
def media_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    return {
        "OPENAI_API_KEY": "secret",
        "media": {
            "image": {"enabled": True, "provider": "openai", "model": "gpt-image-1"},
            "voice": {"enabled": True, "provider": "openai", "model": "gpt-4o-mini-tts"},
            "video": {"enabled": True, "provider": "openai", "model": "sora-2"},
        },
    }


def service_for(values: dict, tmp_path: Path) -> MediaService:
    registry = MediaProviderRegistry()
    registry.register("openai", lambda _config, _key: FakeProvider())
    return MediaService(
        config=MediaConfig.load(values),
        settings_values=values,
        artifact_store=MediaArtifactStore(tmp_path / "artifacts"),
        provider_registry=registry,
    )


def test_workspace_root_places_only_images_in_launch_directory(
    media_values: dict, tmp_path: Path
) -> None:
    launch_root = tmp_path / "workspace"
    launch_root.mkdir()
    registry = MediaProviderRegistry()
    registry.register("openai", lambda _config, _key: FakeProvider())
    service = MediaService(
        config=MediaConfig.load(media_values),
        settings_values=media_values,
        workspace_root=launch_root,
        provider_registry=registry,
    )
    result = service.generate_image(
        ImageGenerationRequest(prompt="A lighthouse"),
        session_id="session_1",
    )
    assert result.primary_artifact is not None
    image_path = Path(result.primary_artifact.local_path)
    assert image_path.parent == launch_root
    assert image_path.name.startswith("media_")
    assert not (launch_root / ".mana").exists()
    assert (
        tmp_path
        / "artifacts"
        / "media"
        / "session_1"
        / result.generation_id
        / "generation.json"
    ).is_file()


def test_media_service_defaults_to_current_launch_directory(
    media_values: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_root = tmp_path / "workspace"
    launch_root.mkdir()
    monkeypatch.chdir(launch_root)
    registry = MediaProviderRegistry()
    registry.register("openai", lambda _config, _key: FakeProvider())
    service = MediaService(
        config=MediaConfig.load(media_values),
        settings_values=media_values,
        provider_registry=registry,
    )
    image = service.generate_image(
        ImageGenerationRequest(prompt="A lighthouse"),
        session_id="session_1",
    )
    voice = service.generate_speech(
        VoiceGenerationRequest(text="Hello"),
        session_id="session_1",
    )
    assert image.primary_artifact is not None
    assert Path(image.primary_artifact.local_path).parent == launch_root
    assert voice.primary_artifact is not None
    assert Path(voice.primary_artifact.local_path).is_relative_to(
        tmp_path / "artifacts" / "media"
    )
    assert not (launch_root / ".mana").exists()


def test_media_configuration_is_optional_and_disabled_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    config = MediaConfig.load({})
    assert not config.image.enabled
    with pytest.raises(MediaConfigurationError, match="disabled"):
        config.require(MediaType.IMAGE)


def test_media_configuration_round_trip_does_not_duplicate_secrets(
    media_values: dict, tmp_path: Path
) -> None:
    validated = validate_config_values(media_values)
    save_effective_user_config(validated, merge=False)
    effective = load_effective_settings(include_env=False)
    assert effective["media"]["image"]["model"] == "gpt-image-1"
    assert "secret" not in json.dumps(effective["media"])
    assert "secret" not in json.dumps(masked_config_summary())
    config_text = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert '= "None"' not in config_text


def test_media_configuration_repairs_legacy_string_none_duration() -> None:
    config = MediaConfig.model_validate(
        {
            "image": {"max_duration_seconds": "None"},
            "voice": {"max_duration_seconds": "null"},
        }
    )
    assert config.image.max_duration_seconds is None
    assert config.voice.max_duration_seconds is None


def test_capability_filtering_and_search_are_conservative() -> None:
    models = descriptors_from_catalog(
        "openai",
        [
            "gpt-4.1-mini",
            "gpt-image-1",
            "gpt-4o-mini-tts",
            "sora-2",
            "text-embedding-3-small",
            "unknown-model",
        ],
    )
    assert [item.id for item in filter_models(models, ModelPurpose.IMAGE)] == ["gpt-image-1"]
    assert [item.id for item in filter_models(models, ModelPurpose.VOICE)] == ["gpt-4o-mini-tts"]
    assert [item.id for item in filter_models(models, ModelPurpose.VIDEO)] == ["sora-2"]
    assert [item.id for item in search_models(models, purpose=ModelPurpose.IMAGE, query="image")] == ["gpt-image-1"]
    assert next(item for item in models if item.id == "unknown-model").capabilities == frozenset()


def test_typed_media_requests_reject_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        ImageGenerationRequest(prompt="", count=0)
    with pytest.raises(ValueError):
        VoiceGenerationRequest(text="Hello", speed=8)
    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="Hello", resolution="../bad")


def test_provider_registry_rejects_unregistered_provider() -> None:
    config = MediaConfig.model_validate(
        {
            "image": {
                "enabled": True,
                "provider": "missing",
                "model": "gpt-image-1",
            }
        }
    )
    with pytest.raises(MediaCapabilityError, match="not supported"):
        MediaProviderRegistry().create(config.image, "secret")


def test_image_and_voice_generation_persist_real_managed_artifacts(
    media_values: dict, tmp_path: Path
) -> None:
    service = service_for(media_values, tmp_path)
    image = service.generate_image(
        ImageGenerationRequest(prompt="A lighthouse"),
        session_id="session_1",
        turn_id="turn_1",
    )
    voice = service.generate_speech(
        VoiceGenerationRequest(text="Hello"),
        session_id="session_1",
        turn_id="turn_2",
    )
    assert image.primary_artifact and Path(image.primary_artifact.local_path).is_file()
    assert voice.primary_artifact and Path(voice.primary_artifact.local_path).is_file()
    assert image.primary_artifact.mime_type == "image/png"
    assert voice.primary_artifact.mime_type == "audio/mpeg"


def test_media_events_contain_safe_lifecycle_metadata_only(
    media_values: dict, tmp_path: Path
) -> None:
    events: list[tuple[str, dict]] = []
    registry = MediaProviderRegistry()
    registry.register("openai", lambda _config, _key: FakeProvider())

    def sink(event_type: str, _title: str, **kwargs) -> None:
        events.append((event_type, dict(kwargs.get("metadata") or {})))

    service = MediaService(
        config=MediaConfig.load(media_values),
        settings_values=media_values,
        artifact_store=MediaArtifactStore(tmp_path / "artifacts"),
        provider_registry=registry,
        event_sink=sink,
    )
    service.generate_image(
        ImageGenerationRequest(prompt="private prompt"),
        session_id="session_1",
        turn_id="turn_1",
    )
    assert [name for name, _ in events] == [
        "media_generation_requested",
        "media_generation_queued",
        "media_generation_started",
        "media_generation_completed",
    ]
    encoded = json.dumps(events)
    assert "private prompt" not in encoded
    assert "secret" not in encoded


def test_video_job_restores_status_and_downloads_result(
    media_values: dict, tmp_path: Path
) -> None:
    service = service_for(media_values, tmp_path)
    queued = service.generate_video(
        VideoGenerationRequest(prompt="A calm ocean"),
        session_id="session_1",
        turn_id="turn_1",
    )
    assert queued.status is GenerationStatus.QUEUED
    restored = service.get_generation_status(
        queued.generation_id,
        session_id="session_1",
        turn_id="turn_2",
    )
    assert restored.status is GenerationStatus.COMPLETED
    assert restored.primary_artifact and restored.primary_artifact.mime_type == "video/mp4"


def test_video_job_can_be_cancelled_with_exact_permission(
    media_values: dict, tmp_path: Path
) -> None:
    service = service_for(media_values, tmp_path)
    queued = service.generate_video(
        VideoGenerationRequest(prompt="A calm ocean"),
        session_id="session_1",
    )
    cancelled = service.cancel_generation(
        queued.generation_id,
        session_id="session_1",
    )
    assert cancelled.status is GenerationStatus.CANCELLED


def test_gateway_dispatches_typed_media_decision(
    media_values: dict, tmp_path: Path
) -> None:
    gateway = AgentChatGateway.__new__(AgentChatGateway)
    gateway.media_service = service_for(media_values, tmp_path)
    decision = EntryRoutingDecision(
        route="media",
        confidence=1.0,
        reason="configured image generation",
        required_sources=("media",),
        media_request={
            "operation": "image.generate",
            "prompt": "A lighthouse",
        },
    )
    result = gateway._execute_media_route(
        decision=decision,
        context=EntryRouteContext(
            session_id="session_1",
            conversation_id="conversation_1",
            turn_id="turn_1",
        ),
    )
    assert result.error is None
    assert result.payload["generation"]["status"] == "completed"
    assert result.payload["generation"]["artifacts"]


def test_explicit_export_is_confined_to_workspace(
    media_values: dict, tmp_path: Path
) -> None:
    service = service_for(media_values, tmp_path)
    generated = service.generate_image(
        ImageGenerationRequest(prompt="A lighthouse"), session_id="session_1"
    )
    artifact = generated.primary_artifact
    assert artifact is not None
    workspace = tmp_path / "workspace"
    exported = service.export_artifact(
        artifact.artifact_id,
        session_id="session_1",
        workspace_root=workspace,
        relative_destination="assets/lighthouse.png",
    )
    assert exported == workspace / "assets" / "lighthouse.png"
    with pytest.raises(MediaValidationError, match="relative workspace path"):
        service.export_artifact(
            artifact.artifact_id,
            session_id="session_1",
            workspace_root=workspace,
            relative_destination="../outside.png",
        )


def test_artifact_store_blocks_path_traversal_mime_mismatch_and_oversize(tmp_path: Path) -> None:
    store = MediaArtifactStore(tmp_path / "artifacts")
    with pytest.raises(MediaArtifactError):
        store.save_generation("../escape", {"generation_id": "job"})
    with pytest.raises(MediaArtifactError, match="not valid image"):
        store.save(
            generation_id="job",
            media_type=MediaType.IMAGE,
            data=MP3,
            declared_mime="image/png",
            max_bytes=1000,
        )
    with pytest.raises(MediaArtifactError, match="maximum size"):
        store.save(
            generation_id="job",
            media_type=MediaType.IMAGE,
            data=PNG,
            declared_mime="image/png",
            max_bytes=10,
        )


def test_unconfigured_model_and_reference_inputs_fail_without_provider_fallback(
    media_values: dict, tmp_path: Path
) -> None:
    service = service_for(media_values, tmp_path)
    with pytest.raises(MediaArtifactError, match="not found"):
        service.generate_image(
            ImageGenerationRequest(prompt="test", reference_artifact_ids=("missing",)),
            session_id="session_1",
        )


def test_denied_exact_media_scope_stops_before_provider_call(
    media_values: dict, tmp_path: Path
) -> None:
    media_values["media"]["permissions"] = {
        "media.image.generate": "deny",
        "media.artifact.write": "allow",
    }
    service = service_for(media_values, tmp_path)
    with pytest.raises(MediaValidationError, match="media.image.generate"):
        service.generate_image(
            ImageGenerationRequest(prompt="test"),
            session_id="session_1",
        )


def test_managed_reference_image_is_validated_and_passed_to_provider(
    media_values: dict, tmp_path: Path
) -> None:
    provider = FakeProvider()
    registry = MediaProviderRegistry()
    registry.register("openai", lambda _config, _key: provider)
    service = MediaService(
        config=MediaConfig.load(media_values),
        settings_values=media_values,
        artifact_store=MediaArtifactStore(tmp_path / "artifacts"),
        provider_registry=registry,
    )
    source = service.generate_image(
        ImageGenerationRequest(prompt="source"), session_id="session_1"
    )
    assert source.primary_artifact is not None
    service.generate_image(
        ImageGenerationRequest(
            prompt="make a product image",
            reference_artifact_ids=(source.primary_artifact.artifact_id,),
        ),
        session_id="session_1",
    )
    assert provider.last_image_references[0].artifact_id == source.primary_artifact.artifact_id


def test_media_entry_decision_is_typed_and_missing_decision_fails_closed() -> None:
    registry = EntryRouteRegistry()
    registry.register(
        RouteRegistration("media", "media generation", lambda: RouteAvailability(True))
    )
    router = EntryRouter(llm=object(), registry=registry)
    base = {
        "route": "media",
        "confidence": 0.99,
        "reason": "generate configured image",
        "required_sources": ["media"],
    }
    with pytest.raises(EntryRoutingError, match="valid media_request"):
        router._validate(base)
    decision = router._validate(
        {
            **base,
            "media_request": {
                "operation": "image.generate",
                "prompt": "A lighthouse",
            },
        }
    )
    assert decision.media_request["operation"] == "image.generate"
    assert select_lane(entry_route=decision.route) is LaneId.MEDIA
    assert validate_tool_permission(
        default_lane_contracts()[LaneId.MEDIA],
        "generate_image",
        task_capabilities=("media.image.generate", "media.artifact.write"),
    ) == {"media.image.generate", "media.artifact.write"}


def test_openai_provider_normalizes_timeout_without_exposing_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mana_agent.media.providers.openai.urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("secret transport detail")),
    )
    provider = OpenAIMediaProvider(api_key="sk-never-expose", timeout_seconds=1)
    with pytest.raises(MediaProviderError) as raised:
        provider.generate_speech(VoiceGenerationRequest(text="Hello", model="tts-1"))
    assert raised.value.code == "media_generation_timeout"
    assert "sk-never-expose" not in str(raised.value)


def test_openai_provider_does_not_retry_non_transient_authentication_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def reject(request, **_kwargs):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            request.full_url, 401, "Unauthorized secret body", {}, None
        )

    monkeypatch.setattr(
        "mana_agent.media.providers.openai.urllib.request.urlopen", reject
    )
    provider = OpenAIMediaProvider(api_key="sk-never-expose")
    with pytest.raises(MediaProviderError) as raised:
        provider.generate_image(
            ImageGenerationRequest(prompt="test", model="gpt-image-1")
        )
    assert raised.value.code == "media_authentication_failed"
    assert calls == 1
