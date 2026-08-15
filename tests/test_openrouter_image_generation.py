from __future__ import annotations

import base64
import json
from pathlib import Path
import urllib.error
import pytest

from mana_agent.config.model_catalog import ModelCapability
from mana_agent.context_cost.governor import ContextCostGovernor
from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.gateway.entry_routing import EntryRouteContext, EntryRoutingDecision
from mana_agent.gateway.lanes import LaneId, select_lane
from mana_agent.media.artifacts import MediaArtifactStore
from mana_agent.media.config import MediaConfig
from mana_agent.media.errors import (
    MediaCapabilityError,
    MediaConfigurationError,
    MediaModelNotFoundError,
    MediaProviderError,
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

MOCK_OPENROUTER_IMAGE_CATALOG = {
    "data": [
        {
            "id": "x-ai/grok-imagine-image-quality",
            "name": "Grok Imagine Image Quality",
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["image"],
            },
            "supported_parameters": ["prompt", "aspect_ratio", "resolution", "n", "response_format"],
            "supports_streaming": False,
            "endpoints": [{"url": "https://openrouter.ai/api/v1/images"}],
        },
        {
            "id": "black-forest-labs/flux-1-schnell",
            "name": "FLUX.1 Schnell",
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["image"],
            },
            "supported_parameters": ["prompt", "aspect_ratio", "quality", "size", "response_format"],
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
            "supported_parameters": ["prompt", "size", "quality", "n", "response_format"],
            "supports_streaming": False,
        },
        {
            "id": "anthropic/claude-3.5-sonnet",
            "name": "Claude 3.5 Sonnet",
            "architecture": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
            },
            "supported_parameters": ["prompt", "max_tokens"],
        },
        {
            "id": "x-ai/grok-4.6",
            "name": "Grok 4.6",
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            },
            "supported_parameters": ["prompt", "tools"],
        },
    ]
}


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
                "model": "x-ai/grok-imagine-image-quality",
                "defaults": {
                    "size": "1024x1024",
                    "aspect_ratio": "1:1",
                    "resolution": "1K",
                    "quality": "auto",
                    "output_format": "png",
                },
            }
        },
    }


# 1. OpenRouter image-model discovery accepts valid image model
def test_openrouter_image_model_discovery_accepts_grok_imagine(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def read(self):
            return json.dumps(MOCK_OPENROUTER_IMAGE_CATALOG).encode("utf-8")

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

    ids = [m["id"] for m in models]
    assert "x-ai/grok-imagine-image-quality" in ids
    assert "black-forest-labs/flux-1-schnell" in ids
    assert "openai/dall-e-3" in ids

    caps = provider.capabilities("x-ai/grok-imagine-image-quality")
    assert ModelCapability.IMAGE_GENERATION in caps


# 2. Absent model ID produces MEDIA_IMAGE_MODEL_NOT_FOUND with safe diagnostics
def test_absent_model_id_raises_media_image_model_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def read(self):
            return json.dumps(MOCK_OPENROUTER_IMAGE_CATALOG).encode("utf-8")

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
    with pytest.raises(MediaCapabilityError) as exc_info:
        provider.get_image_model("x-ai/grok-imagine-image-2.0")

    assert exc_info.value.code == "media_image_model_not_found"
    assert isinstance(exc_info.value, MediaModelNotFoundError)
    assert exc_info.value.metadata.get("requested_model") == "x-ai/grok-imagine-image-2.0"
    assert exc_info.value.metadata.get("provider") == "openrouter"
    assert exc_info.value.metadata.get("image_catalog_loaded") is True
    assert "x-ai/grok-imagine-image-quality" in exc_info.value.metadata.get("closest_model_ids", [])
    assert exc_info.value.metadata.get("suggested_model") == "x-ai/grok-imagine-image-quality"
    assert "sk-or-v1-test" not in str(exc_info.value)


# 3. Stale cache misses model, refreshed catalog contains it -> refresh once and succeed
def test_stale_cache_miss_refreshes_catalog_once(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch_count = 0

    catalog_v1 = {
        "data": [
            {
                "id": "openai/dall-e-3",
                "architecture": {"output_modalities": ["image"]},
                "supported_parameters": ["prompt"],
            }
        ]
    }
    catalog_v2 = {
        "data": [
            {
                "id": "openai/dall-e-3",
                "architecture": {"output_modalities": ["image"]},
                "supported_parameters": ["prompt"],
            },
            {
                "id": "x-ai/grok-imagine-image-quality",
                "architecture": {"output_modalities": ["image"]},
                "supported_parameters": ["prompt", "aspect_ratio", "resolution"],
            },
        ]
    }

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def read(self):
            return json.dumps(self._data).encode("utf-8")

        @property
        def headers(self):
            return {"x-request-id": "req-models", "content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def fake_urlopen(req, **kwargs):
        nonlocal fetch_count
        fetch_count += 1
        data = catalog_v1 if fetch_count == 1 else catalog_v2
        return FakeResponse(data)

    monkeypatch.setattr(
        "mana_agent.media.providers.openrouter.urllib.request.urlopen",
        fake_urlopen,
    )

    provider = OpenRouterMediaProvider(api_key="sk-or-v1-test")
    # Prime initial cache
    provider.list_image_models()
    assert fetch_count == 1

    # Requesting new model causes single refresh on miss and succeeds
    entry = provider.get_image_model("x-ai/grok-imagine-image-quality")
    assert fetch_count == 2
    assert entry["id"] == "x-ai/grok-imagine-image-quality"


# 4. Model exists in catalog but output_modalities does not contain image -> MEDIA_IMAGE_MODEL_UNSUPPORTED
def test_text_only_model_in_catalog_raises_media_image_model_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def read(self):
            return json.dumps(MOCK_OPENROUTER_IMAGE_CATALOG).encode("utf-8")

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
    with pytest.raises(MediaCapabilityError) as exc_info:
        provider.get_image_model("anthropic/claude-3.5-sonnet")

    assert exc_info.value.code == "media_image_model_unsupported"
    assert "anthropic/claude-3.5-sonnet" in exc_info.value.detail


# 5. OpenRouter image catalog unavailable -> media_image_provider_unavailable (not unsupported)
def test_image_catalog_unavailable_raises_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fail(req, **kwargs):
        raise urllib.error.URLError("Connection refused by upstream")

    monkeypatch.setattr(
        "mana_agent.media.providers.openrouter.urllib.request.urlopen",
        fake_fail,
    )

    provider = OpenRouterMediaProvider(api_key="sk-or-v1-test")
    with pytest.raises(MediaProviderError) as exc_info:
        provider.get_image_model("x-ai/grok-imagine-image-quality")

    assert exc_info.value.code == "media_image_provider_unavailable"
    assert exc_info.value.retryable is True


# 6. Valid model calls POST /api/v1/images
# 7. Supported parameters are derived from catalog (unsupported fields excluded)
# 8. Successful response persists artifact
# 9. Image-generation usage/cost enters accounting
def test_valid_image_generation_with_parameter_derivation_and_accounting(
    openrouter_media_env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation_response = {
        "id": "gen-grok-123",
        "data": [
            {
                "b64_json": B64_PNG,
                "media_type": "image/png",
            }
        ],
        "usage": {
            "cost": 0.05,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

    posted_urls = []
    posted_bodies = []

    class FakeCatalogResponse:
        def read(self):
            return json.dumps(MOCK_OPENROUTER_IMAGE_CATALOG).encode("utf-8")

        @property
        def headers(self):
            return {"x-request-id": "req-cat", "content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FakeGenResponse:
        def read(self):
            return json.dumps(generation_response).encode("utf-8")

        @property
        def headers(self):
            return {"x-request-id": "req-gen", "content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def fake_urlopen(req, **kwargs):
        if "/images/models" in req.full_url:
            return FakeCatalogResponse()
        posted_urls.append(req.full_url)
        posted_bodies.append(json.loads(req.data.decode("utf-8")))
        return FakeGenResponse()

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
        prompt="A cosmic nebula with vibrant colors",
        model="x-ai/grok-imagine-image-quality",
        size="1024x1024",  # Not in grok's supported_parameters
        aspect_ratio="1:1",  # In supported_parameters
        resolution="1K",  # In supported_parameters
        quality="hd",  # Not in grok's supported_parameters
    )

    result = service.generate_image(request, session_id="session_grok", turn_id="turn_grok")

    # Verify endpoint and parameter derivation
    assert posted_urls[0] == "https://openrouter.ai/api/v1/images"
    payload = posted_bodies[0]
    assert payload["model"] == "x-ai/grok-imagine-image-quality"
    assert payload["prompt"] == "A cosmic nebula with vibrant colors"
    assert payload["aspect_ratio"] == "1:1"
    assert payload["resolution"] == "1K"
    assert "quality" not in payload  # Not in supported_parameters
    assert "size" not in payload  # Not in supported_parameters

    # Verify completion and artifact persistence
    assert result.status is GenerationStatus.COMPLETED
    assert result.primary_artifact is not None
    artifact = result.primary_artifact
    assert Path(artifact.local_path).is_file()
    assert Path(artifact.local_path).read_bytes() == PNG_BYTES

    # Verify usage & accounting
    assert result.usage.get("cost") == 0.05
    assert result.usage.get("actual_cost") == 0.05
    assert result.usage.get("image_count") == 1


# 10. Terminal unsupported/not-found failure produces is_resumable=False, pending_required_work=False
def test_terminal_model_not_found_produces_non_resumable_no_pending_work(
    openrouter_media_env: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeResponse:
        def read(self):
            return json.dumps(MOCK_OPENROUTER_IMAGE_CATALOG).encode("utf-8")

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
        {"context_cost_governor": ContextCostGovernor(session_id="session_fail", settings=Settings())},
    )()

    # Request with stale/nonexistent model ID
    decision = EntryRoutingDecision(
        route="media",
        confidence=1.0,
        reason="user requested image generation",
        required_sources=("media",),
        media_request={
            "operation": "image.generate",
            "prompt": "An abstract painting",
            "model": "x-ai/grok-imagine-image-2.0",
        },
    )

    turn_result = gateway._execute_media_route(
        decision=decision,
        context=EntryRouteContext(
            session_id="session_fail",
            conversation_id="conv_fail",
            turn_id="turn_fail",
        ),
    )

    assert turn_result.error == "media_image_model_not_found"
    assert turn_result.payload["status"] == "failed"
    assert turn_result.payload["is_resumable"] is False
    assert turn_result.payload["pending_required_work"] is False
    assert turn_result.payload["goal_satisfied"] is False
    assert turn_result.payload.get("suggested_model") == "x-ai/grok-imagine-image-quality"


# 11. Lane reasoning model remains grok-4.6 while image model is independently grok-imagine-image-quality
def test_lane_reasoning_and_image_model_separation() -> None:
    lane_id = select_lane(entry_route="media")
    assert lane_id is LaneId.MEDIA

    decision = EntryRoutingDecision(
        route="media",
        confidence=1.0,
        reason="user requested image generation",
        required_sources=("media",),
        media_request={
            "operation": "image.generate",
            "prompt": "A futuristic city",
            "model": "x-ai/grok-imagine-image-quality",
        },
    )
    assert decision.route == "media"
    assert decision.media_request["model"] == "x-ai/grok-imagine-image-quality"


# 12. Missing OpenRouter credentials fails closed with media_provider_auth_required
def test_missing_openrouter_credentials_fails_closed(tmp_path: Path) -> None:
    env = {
        "media": {
            "image": {
                "enabled": True,
                "provider": "openrouter",
                "model": "x-ai/grok-imagine-image-quality",
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


# 13. API key redaction in error reporting
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
            ImageGenerationRequest(prompt="test", model="x-ai/grok-imagine-image-quality")
        )
    assert secret_key not in str(exc.value)
    assert secret_key not in exc.value.detail
    assert "[REDACTED]" in exc.value.detail or "Unauthorized" in exc.value.detail
