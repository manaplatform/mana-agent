from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch
import pytest

from mana_agent.config.model_capabilities import (
    ModelCapabilityDescriptor,
    clear_capability_cache,
    clear_model_capability_overrides,
    normalize_model_lookup_id,
    register_model_capability_override,
    resolve_model_capability,
    resolve_transport_name,
)
from mana_agent.config.provider_registry import CodexTransport
from mana_agent.evals.recorder import CURRENT_RECORDER, SimpleEvaluationRecorder
from mana_agent.gateway.routing import GatewayRoutingAuthority
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.coding_agent_shim import CodexCodingAgentShim
from mana_agent.integrations.codex.exceptions import CodexCapabilityError
from mana_agent.model_routing.models import (
    Complexity,
    LatencyClass,
    ModelProfile,
    NoWriteCapableModelAvailableError,
    RiskLevel,
    RoutingFailure,
    RoutingRequest,
)
from mana_agent.model_routing.profiles import configured_profiles
from mana_agent.model_routing.router import ModelRouter


@pytest.fixture(autouse=True)
def reset_capabilities_and_recorder():
    clear_capability_cache()
    clear_model_capability_overrides()
    recorder = SimpleEvaluationRecorder()
    token = CURRENT_RECORDER.set(recorder)
    yield recorder
    CURRENT_RECORDER.reset(token)
    clear_capability_cache()
    clear_model_capability_overrides()


def test_openrouter_model_id_normalization():
    """Rule 3: Normalize OpenRouter model IDs without stripping org namespaces."""
    assert (
        normalize_model_lookup_id("openrouter", "deepseek/deepseek-v4-flash")
        == "deepseek/deepseek-v4-flash"
    )
    assert (
        normalize_model_lookup_id("openrouter", "openrouter/deepseek/deepseek-v4-flash")
        == "deepseek/deepseek-v4-flash"
    )
    assert (
        normalize_model_lookup_id("openrouter", "anthropic/claude-3.7-sonnet")
        == "anthropic/claude-3.7-sonnet"
    )
    assert (
        normalize_model_lookup_id("openai", "openai/gpt-4.1")
        == "gpt-4.1"
    )
    assert (
        normalize_model_lookup_id("nvidia", "nvidia/deepseek-ai/deepseek-v4-flash")
        == "deepseek-ai/deepseek-v4-flash"
    )


def test_case_e_capability_keys_distinguish_provider_model_and_transport():
    """Case E: Capability resolution must distinguish provider, model, and transport."""
    # Register different capabilities for openrouter vs nvidia vs custom
    desc_openrouter = ModelCapabilityDescriptor(
        provider="openrouter",
        model="deepseek/deepseek-v4-flash",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        capability_confidence="high",
        capability_source="override",
    )
    desc_bridge = ModelCapabilityDescriptor(
        provider="openrouter",
        model="deepseek/deepseek-v4-flash",
        transport="responses_bridge",
        supports_tool_calls=False,
        supports_repository_read=False,
        supports_repository_write=False,
        capability_confidence="high",
        capability_source="override",
    )
    register_model_capability_override(desc_openrouter)
    register_model_capability_override(desc_bridge)

    resolved_direct = resolve_model_capability(
        "openrouter", "deepseek/deepseek-v4-flash", "direct_responses"
    )
    resolved_bridge = resolve_model_capability(
        "openrouter", "deepseek/deepseek-v4-flash", "responses_bridge"
    )

    assert resolved_direct.supports_tool_calls is True
    assert resolved_direct.supports_repository_write is True
    assert resolved_bridge.supports_tool_calls is False
    assert resolved_bridge.supports_repository_write is False


def test_case_a_openrouter_model_with_verified_metadata_is_admitted():
    """Case A: OpenRouter model with tool/write capability metadata is admitted."""
    openrouter_record = {
        "id": "deepseek/deepseek-v4-flash",
        "name": "DeepSeek V4 Flash",
        "supported_parameters": ["tools", "tool_choice", "response_format"],
        "context_length": 1_000_000,
    }
    desc = resolve_model_capability(
        "openrouter",
        "deepseek/deepseek-v4-flash",
        "direct_responses",
        catalog_records=[openrouter_record],
    )
    assert desc.is_known is True
    assert desc.supports_tool_calls is True
    assert desc.supports_repository_write is True
    assert desc.supports_repository_read is True
    assert desc.capability_source == "catalog"

    profiles = configured_profiles([
        {
            "provider": "openrouter",
            "model_id": "deepseek/deepseek-v4-flash",
            "supported_roles": ["coding", "tool"],
            "can_tool_call": True,
            "can_patch": True,
            "supported_tools": ["*"],
            "context_window": 1_000_000,
            "max_output_tokens": 65_536,
        }
    ])
    # Attach resolved capability descriptor
    profile = profiles[0]
    profile = ModelProfile(
        provider=profile.provider,
        model_id=profile.model_id,
        supported_roles=profile.supported_roles,
        supported_tools=profile.supported_tools,
        reasoning_settings=profile.reasoning_settings,
        context_window=profile.context_window,
        max_output_tokens=profile.max_output_tokens,
        latency_class=profile.latency_class,
        can_patch=True,
        can_tool_call=True,
        can_structured_output=True,
        capability_descriptor=desc,
    )

    router = ModelRouter([profile])
    write_request = RoutingRequest(
        role="coding",
        task_description="Implement feature with file write",
        task_type="coding",
        complexity=Complexity.MEDIUM,
        risk=RiskLevel.MEDIUM,
        required_capabilities=frozenset({"patch", "tool_calls", "repository_write"}),
        required_tools=frozenset({"repository_read", "repository_write", "test_execution"}),
    )
    decision = router.route(write_request)
    assert decision.selected_model == "deepseek/deepseek-v4-flash"
    assert decision.provider == "openrouter"


def test_case_b_unknown_preferred_candidate_skipped_and_second_verified_model_selected():
    """Case B: Preferred candidate with unknown capability is skipped, second verified model selected."""
    # Unknown candidate (no tools, confidence unknown)
    unknown_desc = ModelCapabilityDescriptor(
        provider="openrouter",
        model="deepseek/deepseek-v4-flash",
        transport="direct_responses",
        supports_tool_calls=False,
        supports_repository_read=False,
        supports_repository_write=False,
        capability_confidence="unknown",
        capability_source="unknown",
    )
    # Verified candidate
    verified_desc = ModelCapabilityDescriptor(
        provider="openrouter",
        model="anthropic/claude-3.7-sonnet",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        capability_confidence="high",
        capability_source="maintained",
    )

    p1 = ModelProfile(
        provider="openrouter",
        model_id="deepseek/deepseek-v4-flash",
        supported_roles=frozenset({"coding", "tool"}),
        supported_tools=frozenset(),
        context_window=1_000_000,
        max_output_tokens=65_536,
        can_patch=False,
        can_tool_call=False,
        logical_cost_per_1k_tokens=0.1,  # cheaper
        reliability_score=0.9,
        capability_descriptor=unknown_desc,
    )
    p2 = ModelProfile(
        provider="openrouter",
        model_id="anthropic/claude-3.7-sonnet",
        supported_roles=frozenset({"coding", "tool"}),
        supported_tools=frozenset({"*"}),
        context_window=200_000,
        max_output_tokens=16_384,
        can_patch=True,
        can_tool_call=True,
        logical_cost_per_1k_tokens=3.0,
        reliability_score=0.95,
        capability_descriptor=verified_desc,
    )

    router = ModelRouter([p1, p2])
    write_request = RoutingRequest(
        role="coding",
        task_description="Apply patch to repo",
        task_type="coding",
        complexity=Complexity.MEDIUM,
        risk=RiskLevel.MEDIUM,
        required_capabilities=frozenset({"patch", "tool_calls", "repository_write"}),
        required_tools=frozenset({"repository_read", "repository_write", "test_execution"}),
    )
    decision = router.route(write_request)

    # First candidate was skipped due to missing/unknown capability, second was selected
    assert decision.selected_model == "anthropic/claude-3.7-sonnet"
    assert decision.provider == "openrouter"
    rejected_models = [r.model for r in decision.rejected_candidates]
    assert "openrouter/deepseek/deepseek-v4-flash" in rejected_models


def test_case_c_all_candidates_unknown_raises_no_write_capable_model_available():
    """Case C: All candidate capabilities unknown/incompatible -> typed error without fallback."""
    unknown_desc = ModelCapabilityDescriptor(
        provider="openrouter",
        model="deepseek/deepseek-v4-flash",
        transport="direct_responses",
        supports_tool_calls=False,
        supports_repository_read=False,
        supports_repository_write=False,
        capability_confidence="unknown",
        capability_source="unknown",
    )
    p1 = ModelProfile(
        provider="openrouter",
        model_id="deepseek/deepseek-v4-flash",
        supported_roles=frozenset({"coding", "tool"}),
        supported_tools=frozenset(),
        context_window=1_000_000,
        max_output_tokens=65_536,
        can_patch=False,
        can_tool_call=False,
        capability_descriptor=unknown_desc,
    )

    router = ModelRouter([p1])
    write_request = RoutingRequest(
        role="coding",
        task_description="Apply mutation",
        task_type="coding",
        complexity=Complexity.MEDIUM,
        risk=RiskLevel.MEDIUM,
        required_capabilities=frozenset({"patch", "tool_calls", "repository_write"}),
        required_tools=frozenset({"repository_read", "repository_write", "test_execution"}),
    )

    with pytest.raises(NoWriteCapableModelAvailableError) as exc_info:
        router.route(write_request)

    err = exc_info.value
    assert err.error_code == "no_write_capable_model_available"
    assert "no_write_capable_model_available" in str(err)
    assert len(err.rejected) == 1
    assert err.rejected[0].model == "openrouter/deepseek/deepseek-v4-flash"


def test_case_d_read_only_coding_request_admitted_when_read_supported():
    """Case D: Read-only Coding request is admitted even if write is unsupported."""
    read_only_desc = ModelCapabilityDescriptor(
        provider="openrouter",
        model="read-only-model",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=False,
        supports_structured_output=True,
        capability_confidence="high",
        capability_source="catalog",
    )
    p1 = ModelProfile(
        provider="openrouter",
        model_id="read-only-model",
        supported_roles=frozenset({"coding", "tool", "planner"}),
        supported_tools=frozenset({"repository_read"}),
        context_window=100_000,
        max_output_tokens=4_096,
        can_patch=False,
        can_tool_call=True,
        can_structured_output=True,
        capability_descriptor=read_only_desc,
    )

    router = ModelRouter([p1])
    read_request = RoutingRequest(
        role="coding",
        task_description="Analyze repository code (read only)",
        task_type="planning",
        complexity=Complexity.MEDIUM,
        risk=RiskLevel.LOW,
        required_capabilities=frozenset({"structured_output", "repository_read"}),
        required_tools=frozenset({"repository_read"}),
    )
    decision = router.route(read_request)
    assert decision.selected_model == "read-only-model"


def test_separate_agent_permission_and_model_capability():
    """Rule 9: Agent permission (workspaceWrite) does NOT imply model transport capability."""
    shim = CodexCodingAgentShim(
        repo_root=Path("/tmp"),
        codex_settings=CodexSettings(enabled=True),
        repository_id="test_repo",
    )
    # Even if agent turn requires write, unknown model capabilities must fail closed
    with pytest.raises(CodexCapabilityError) as exc_info:
        shim._validate_write_transport_capability(
            requires_repository_write=True,
            model="unknown-unverified-model",
            provider="openrouter",
            transport=CodexTransport.DIRECT_RESPONSES,
        )
    assert "model capabilities are unknown" in str(exc_info.value)


def test_exact_deepseek_openrouter_direct_responses_reproduction():
    """Verify that deepseek/deepseek-v4-flash on OpenRouter direct_responses is validated fail-closed."""
    # When no capability metadata is provided, resolution returns unknown
    desc = resolve_model_capability("openrouter", "deepseek/deepseek-v4-flash", "direct_responses")
    assert desc.is_known is False
    assert desc.supports_tool_calls is False
    assert desc.supports_repository_write is False

    shim = CodexCodingAgentShim(
        repo_root=Path("/tmp"),
        codex_settings=CodexSettings(enabled=True),
        repository_id="test_repo",
    )
    with pytest.raises(CodexCapabilityError) as exc_info:
        shim._validate_write_transport_capability(
            requires_repository_write=True,
            model="deepseek/deepseek-v4-flash",
            provider="openrouter",
            transport="direct_responses",
        )
    assert "model capabilities are unknown" in str(exc_info.value)
    assert "provider='openrouter'" in str(exc_info.value)
    assert "model='deepseek/deepseek-v4-flash'" in str(exc_info.value)
    assert "transport='direct_responses'" in str(exc_info.value)


def test_diagnostics_recording(reset_capabilities_and_recorder):
    """Rule 11: Diagnostics recorded for resolved, unknown, rejected, and selected models."""
    recorder = reset_capabilities_and_recorder

    # Trigger unknown
    resolve_model_capability("openrouter", "unknown-model-xyz", "direct_responses")
    unknown_events = [e for e in recorder.events if e.get("type") == "model.capability.unknown"]
    assert len(unknown_events) >= 1
    assert unknown_events[-1]["payload"]["model"] == "unknown-model-xyz"

    # Trigger resolved
    resolve_model_capability("openai", "gpt-4.1", "direct_responses")
    resolved_events = [e for e in recorder.events if e.get("type") == "model.capability.resolved"]
    assert len(resolved_events) >= 1
    assert resolved_events[-1]["payload"]["model"] == "gpt-4.1"


def test_cache_invalidation():
    """Rule 10: Cache resolved capabilities and invalidate on clear_capability_cache."""
    desc1 = resolve_model_capability("openai", "gpt-4.1", "direct_responses")
    assert desc1.capability_source == "maintained"

    # Override and verify cache
    custom_desc = ModelCapabilityDescriptor(
        provider="openai",
        model="gpt-4.1",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        capability_confidence="high",
        capability_source="override",
    )
    register_model_capability_override(custom_desc)
    desc2 = resolve_model_capability("openai", "gpt-4.1", "direct_responses")
    assert desc2.capability_source == "override"

    clear_model_capability_overrides()
    clear_capability_cache()
    desc3 = resolve_model_capability("openai", "gpt-4.1", "direct_responses")
    assert desc3.capability_source == "maintained"
