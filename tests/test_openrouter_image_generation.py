from __future__ import annotations

import base64
import json
from pathlib import Path
import urllib.error
import pytest

from mana_agent.config.model_catalog import ModelCapability, normalize_capabilities
from mana_agent.context_cost.governor import ContextCostGovernor
from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.gateway.entry_routing import EntryRouteContext, EntryRoutingDecision
from mana_agent.gateway.lanes import LaneId, select_lane
from mana_agent.media.artifacts import MediaArtifactStore
from mana_agent.media.config import MediaConfig
from mana_agent.media.errors import (
    MediaCapabilityError,
    MediaConfigurationError,
    MediaProviderError,
    MediaValidationError,
)
from mana_agent.media.models import (
    GenerationStatus,
    ImageGenerationRequest,
    MediaType,
)
from mana_agent.media.providers.openrouter import OpenRouterMediaProvider
from mana_agent.media.registry import MediaProviderRegistry
from mana_agent.media.service import MediaService


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
B64_PNG = base64.b64encode(PNG_BYTES).decode("utf-8")


@pytest.fixture()
def openrouter_media_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    mana_dir = tmp_path / ".mana"
    monkeypatch.setenv("MANA_HOME", str(mana_dir))
    return {
        "OPENROUTER_API_KEY": "sk-or-v1-secret-test-key",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
        "media": {
            "image": {
                "enabled": True,
                "provider": "openrouter",
                "model": "black-forest-labs/flux-1-schnell",
                "defaults": {
                    "size": "1024x1024",
                    "aspect_ratio": "1:1",
                    "quality": "auto",
                    "output_format": "png",
                },
            }
        },
    }


# 1. OpenRouter image-model discovery (GET /api/v1/images/models)
def test_openrouter_image_model_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog_response = {
        "data": [
            {
                "id": "black-forest-labs/flux-1-schnell",
                "name": "FLUX.1 Schnell",
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["image"],
                },
                "supported_parameters": ["prompt", "aspect_ratio", "quality", "size"],
                "supports_streaming": False,
                "endpoints": [{"url": "https://openrouter.ai/api/v1/images"}],
            },
            {
                "id": "openai/dall-e-3",
                "name": "DALL-E 3",
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["image"],
                },
                "supported_parameters": ["prompt", "size", "quality", "n"],
                "supports_streaming": False,
            },
            {
                "id": "anthropic/claude-3.5-sonnet",
                "name": "Claude 3.5 Sonnet",
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
            },
        ]
    }

    class FakeResponse:
        def read(self):
            return json.dumps(catalog_response).encode("utf-8")

        @property
        def headers(self):
            return {"x-request-id": "req-models", "content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        "mana_agent.media.providers.openrouter.urllib.request.urlopen",
        lambda req, **kwargs: FakeResponse(),
    )

    provider = OpenRouterMediaProvider(api_key="sk-or-v1-test")
    models = provider.list_image_models()

    assert len(models) == 2
    ids = [m["id"] for m in models]
    assert "black-forest-labs/flux-1-schnell" in ids
    assert "openai/dall-e-3" in ids
    assert "anthropic/claude-3.5-sonnet" not in ids
    flux = next(m for m in models if m["id"] == "black-forest-labs/flux-1-schnell")
    assert flux["architecture"]["output_modalities"] == ["image"]
    assert "aspect_ratio" in flux["supported_parameters"]


# 2. Successful image generation (POST /api/v1/images)
# 3. Base64 decoding of returned images
# 4. Artifact persistence and metadata record creation
def test_successful_image_generation_and_artifact_persistence(
    openrouter_media_env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation_response = {
        "id": "gen-12345",
        "data": [
            {
                "b64_json": B64_PNG,
                "media_type": "image/png",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 0,
            "total_tokens": 12,
            "cost": 0.035,
        },
    }

    posted_urls = []
    posted_bodies = []

    class FakeResponse:
        def read(self):
            return json.dumps(generation_response).encode("utf-8")

        @property
        def headers(self):
            return {"x-request-id": "req-gen-123", "content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def fake_urlopen(req, **kwargs):
        posted_urls.append(req.full_url)
        posted_bodies.append(json.loads(req.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr(
        "mana_agent.media.providers.openrouter.urllib.request.urlopen",
        fake_urlopen,
    )

    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    artifact_store = MediaArtifactStore(tmp_path / "artifacts", image_output_root=tmp_path / "workspace")
    registry = MediaProviderRegistry()

    service = MediaService(
        config=MediaConfig.load(openrouter_media_env),
        settings_values=openrouter_media_env,
        artifact_store=artifact_store,
        provider_registry=registry,
        workspace_root=tmp_path / "workspace",
    )

    request = ImageGenerationRequest(
        prompt="A futuristic neon skyline at night",
        model="black-forest-labs/flux-1-schnell",
        size="1024x1024",
        aspect_ratio="16:9",
        quality="auto",
    )

    result = service.generate_image(request, session_id="session_test", turn_id="turn_test")

    assert posted_urls[0] == "https://openrouter.ai/api/v1/images"
    assert posted_bodies[0]["prompt"] == "A futuristic neon skyline at night"
    assert posted_bodies[0]["aspect_ratio"] == "16:9"
    assert posted_bodies[0]["response_format"] == "b64_json"

    assert result.status is GenerationStatus.COMPLETED
    assert result.provider == "openrouter"
    assert result.model == "black-forest-labs/flux-1-schnell"
    assert result.primary_artifact is not None
    assert result.usage.get("cost") == 0.035

    artifact = result.primary_artifact
    assert Path(artifact.local_path).is_file()
    assert Path(artifact.local_path).read_bytes() == PNG_BYTES
    assert artifact.mime_type == "image/png"
    assert artifact.size_bytes == len(PNG_BYTES)
    assert artifact.provider == "openrouter"
    assert artifact.model == "black-forest-labs/flux-1-schnell"


# 5. Output artifacts population and gateway execution
def test_gateway_media_execution_populates_output_artifacts(
    openrouter_media_env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation_response = {
        "id": "gen-9999",
        "data": [{"b64_json": B64_PNG, "media_type": "image/png"}],
        "usage": {"cost": 0.04},
    }

    class FakeResponse:
        def read(self):
            return json.dumps(generation_response).encode("utf-8")

        @property
        def headers(self):
            return {"x-request-id": "req-9999", "content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        "mana_agent.media.providers.openrouter.urllib.request.urlopen",
        lambda req, **kwargs: FakeResponse(),
    )

    from mana_agent.config.settings import Settings

    gateway = AgentChatGateway.__new__(AgentChatGateway)
    service = MediaService(
        config=MediaConfig.load(openrouter_media_env),
        settings_values=openrouter_media_env,
        artifact_store=MediaArtifactStore(tmp_path / "artifacts"),
    )
    gateway.media_service = service
    gateway._stack = type(
        "Stack",
        (),
        {"context_cost_governor": ContextCostGovernor(session_id="session_10", settings=Settings())},
    )()

    decision = EntryRoutingDecision(
        route="media",
        confidence=1.0,
        reason="user requested image generation",
        required_sources=("media",),
        media_request={
            "operation": "image.generate",
            "prompt": "Cyberpunk city",
            "model": "black-forest-labs/flux-1-schnell",
        },
    )

    turn_result = gateway._execute_media_route(
        decision=decision,
        context=EntryRouteContext(
            session_id="session_10",
            conversation_id="conv_10",
            turn_id="turn_10",
        ),
    )

    assert turn_result.error is None
    assert turn_result.payload["route"] == "media"
    assert turn_result.payload["provider"] == "openrouter"
    assert turn_result.payload["image_model"] == "black-forest-labs/flux-1-schnell"
    assert turn_result.payload["verification_status"] == "passed"
    assert len(turn_result.payload["output_artifacts"]) == 1
    assert len(turn_result.sources) == 1
    assert turn_result.sources[0]["type"] == "media_artifact"
    assert Path(turn_result.sources[0]["path"]).is_file()
    assert turn_result.trace[0]["tool_name"] == "media.image.generate"


# 6. Image-disabled configuration (media_image_disabled)
def test_image_disabled_configuration_fails_fast() -> None:
    config = MediaConfig.load({"media": {"image": {"enabled": False}}})
    with pytest.raises(MediaConfigurationError) as exc:
        config.require(MediaType.IMAGE)
    assert exc.value.code == "media_image_disabled"


# 7. Missing OpenRouter credentials (media_provider_auth_required)
def test_missing_openrouter_credentials_fails_closed(tmp_path: Path) -> None:
    env = {
        "media": {
            "image": {
                "enabled": True,
                "provider": "openrouter",
                "model": "black-forest-labs/flux-1-schnell",
            }
        }
    }
    service = MediaService(
        config=MediaConfig.load(env),
        settings_values=env,
        artifact_store=MediaArtifactStore(tmp_path / "artifacts"),
    )
    with pytest.raises(MediaConfigurationError) as exc:
        service.generate_image(
            ImageGenerationRequest(prompt="draw a sunset"),
            session_id="session_no_key",
        )
    assert exc.value.code == "media_provider_auth_required"


# 8. Unsupported image model / text-only model rejected (media_image_model_unsupported)
def test_text_only_model_rejected_for_image_generation(openrouter_media_env: dict, tmp_path: Path) -> None:
    openrouter_media_env["media"]["image"]["model"] = "x-ai/grok-4.6"
    service = MediaService(
        config=MediaConfig.load(openrouter_media_env),
        settings_values=openrouter_media_env,
        artifact_store=MediaArtifactStore(tmp_path / "artifacts"),
    )
    with pytest.raises(MediaCapabilityError) as exc:
        service.generate_image(
            ImageGenerationRequest(prompt="draw a logo", model="x-ai/grok-4.6"),
            session_id="session_text_model",
        )
    assert exc.value.code == "media_image_model_unsupported"


# 9. Provider HTTP error handling
def test_provider_http_error_mapped_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_reject(req, **kwargs):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request: unsupported dimension", {}, None
        )

    monkeypatch.setattr(
        "mana_agent.media.providers.openrouter.urllib.request.urlopen",
        fake_reject,
    )

    provider = OpenRouterMediaProvider(api_key="sk-or-v1-secret")
    with pytest.raises(MediaProviderError) as exc:
        provider.generate_image(
            ImageGenerationRequest(prompt="test", model="black-forest-labs/flux-1-schnell")
        )
    assert exc.value.code == "media_provider_parameter_rejected"


# 10. Malformed/missing b64_json handling
def test_malformed_b64_json_handled_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    malformed_response = {
        "id": "gen-bad",
        "data": [{"b64_json": "not-valid-base-64!!!"}],
    }

    class FakeResponse:
        def read(self):
            return json.dumps(malformed_response).encode("utf-8")

        @property
        def headers(self):
            return {"x-request-id": "req-bad", "content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        "mana_agent.media.providers.openrouter.urllib.request.urlopen",
        lambda req, **kwargs: FakeResponse(),
    )

    provider = OpenRouterMediaProvider(api_key="sk-or-v1-secret")
    with pytest.raises(MediaProviderError) as exc:
        provider.generate_image(
            ImageGenerationRequest(prompt="test", model="black-forest-labs/flux-1-schnell")
        )
    assert exc.value.code == "media_provider_invalid_output"


# 11. Accounting with OpenRouter usage.cost
def test_accounting_records_openrouter_cost(tmp_path: Path) -> None:
    from mana_agent.config.settings import Settings

    governor = ContextCostGovernor(session_id="session_acc", settings=Settings())
    governor.record_media_generation(
        call_id="media_test_call",
        cost=0.045,
        usage={"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10, "cost": 0.045},
        provider="openrouter",
        model="black-forest-labs/flux-1-schnell",
        task_id="task_image_1",
        root_task_id="root_task_1",
        attempt_id="attempt_1",
        session_id="session_acc",
    )

    task_usage = governor.task_usage("task_image_1")
    assert task_usage["actual_cost"] == 0.045
    assert task_usage["actual_cost_known"] is True
    assert governor.metrics["actual_cost"] == 0.045


# 12. API key redaction in error reporting
def test_api_key_redaction_in_error_reporting(monkeypatch: pytest.MonkeyPatch) -> None:
    secret_key = "sk-or-v1-very-secret-credential-string-here"

    def fake_reject(req, **kwargs):
        raise urllib.error.HTTPError(
            req.full_url,
            401,
            f"Unauthorized: Key {secret_key} rejected by upstream",
            {},
            None,
        )

    monkeypatch.setattr(
        "mana_agent.media.providers.openrouter.urllib.request.urlopen",
        fake_reject,
    )

    provider = OpenRouterMediaProvider(api_key=secret_key)
    with pytest.raises(MediaProviderError) as exc:
        provider.generate_image(
            ImageGenerationRequest(prompt="test", model="black-forest-labs/flux-1-schnell")
        )
    assert secret_key not in str(exc.value)
    assert secret_key not in exc.value.detail
    assert "[REDACTED]" in exc.value.detail or "Unauthorized" in exc.value.detail


# 13. Supervisor completion after artifact verification
# 14. Restart/recovery with already-persisted artifact
def test_supervisor_and_persisted_artifact_lifecycle(
    openrouter_media_env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = []

    def event_sink(event_type: str, title: str, **kwargs):
        events.append((event_type, dict(kwargs.get("metadata") or {})))

    generation_response = {
        "id": "gen-durable",
        "data": [{"b64_json": B64_PNG, "media_type": "image/png"}],
        "usage": {"cost": 0.02},
    }

    class FakeResponse:
        def read(self):
            return json.dumps(generation_response).encode("utf-8")

        @property
        def headers(self):
            return {"x-request-id": "req-durable", "content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        "mana_agent.media.providers.openrouter.urllib.request.urlopen",
        lambda req, **kwargs: FakeResponse(),
    )

    store = MediaArtifactStore(tmp_path / "artifacts")
    service = MediaService(
        config=MediaConfig.load(openrouter_media_env),
        settings_values=openrouter_media_env,
        artifact_store=store,
        event_sink=event_sink,
    )

    res = service.generate_image(
        ImageGenerationRequest(prompt="A castle in the clouds", model="black-forest-labs/flux-1-schnell"),
        session_id="session_durable",
        turn_id="turn_durable",
    )

    assert res.status is GenerationStatus.COMPLETED
    event_names = [e[0] for e in events]
    assert "media_generation_requested" in event_names
    assert "media_generation_queued" in event_names
    assert "media_generation_started" in event_names
    assert "media_generation_completed" in event_names

    # Test retrieval/recovery of persisted generation
    restored = store.load_generation("session_durable", res.generation_id)
    assert restored["status"] == "completed"
    assert len(restored["artifacts"]) == 1
    artifact_id = restored["artifacts"][0]["artifact_id"]
    loaded_art = store.load(artifact_id, session_id="session_durable")
    assert loaded_art.artifact_id == artifact_id
    assert Path(loaded_art.local_path).is_file()


# 15. Media routing remains independent from the text reasoning model
def test_media_routing_independent_from_lane_reasoning_model() -> None:
    lane_id = select_lane(entry_route="media")
    assert lane_id is LaneId.MEDIA
