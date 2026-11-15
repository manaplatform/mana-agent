"""Dynamic model capability discovery, multi-strategy AutoRouter, and chat bootstrap tests."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import httpx
import pytest
from pydantic import BaseModel

from mana_agent.config.settings import Settings
from mana_agent.llm.capabilities import (
    CapabilityFact,
    CapabilityState,
    ModelProfile as NativeProfile,
    metadata_facts,
    observation,
)
from mana_agent.llm.discovery import ModelDiscoveryService
from mana_agent.llm.engine import NativeLLMEngine
from mana_agent.llm.errors import CapabilityError, InvalidResponseError
from mana_agent.llm.messages import ToolCall
from mana_agent.llm.probes import is_probe_authorized, run_capability_probe
from mana_agent.llm.registry import ModelCapabilityRegistry
from mana_agent.llm.types import ModelResponse, Usage
from mana_agent.model_routing.models import (
    Complexity,
    LatencyClass,
    ModelProfile,
    RiskLevel,
    RoutingBudgets,
    RoutingFailure,
    RoutingRequest,
)
from mana_agent.model_routing.profiles import configured_profiles, profiles_from_legacy_configuration
from mana_agent.model_routing.router import ModelRouter
from mana_agent.multi_agent.core.types import AgentRole
from mana_agent.multi_agent.runtime.model_levels import resolve_model_for_role


class SampleRoutingDecision(BaseModel):
    candidate: str
    reason: str
    confidence: float


# 1. Missing provider metadata triggers dynamic discovery
def test_missing_provider_metadata_triggers_dynamic_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_path = tmp_path / "registry.json"
    registry = ModelCapabilityRegistry(cache_path)
    discovery_service = ModelDiscoveryService(registry=registry)

    # Candidate starts with missing capability descriptor
    native_desc = NativeProfile(provider="openrouter", model_id="test/model")
    profile = ModelProfile(
        provider="openrouter",
        model_id="test/model",
        context_window=0,
        max_output_tokens=0,
        supported_roles=frozenset({"*"}),
        supported_tools=frozenset({"*"}),
        capability_descriptor=native_desc,
        configuration={},
    )

    # Mock discovery to return fresh metadata
    refreshed_desc = NativeProfile(
        provider="openrouter",
        model_id="test/model",
        facts={
            "input_text": observation(True, source="provider"),
            "output_text": observation(True, source="provider"),
            "context_window": observation(128000, source="provider"),
            "max_output_tokens": observation(4096, source="provider"),
        },
    )
    monkeypatch.setattr(
        "mana_agent.llm.discovery.ModelDiscoveryService.refresh_model_sync",
        lambda self, **kwargs: refreshed_desc,
    )

    router = ModelRouter([profile])
    req = RoutingRequest(
        role="main",
        task_description="Start chat",
        task_type="routine",
        complexity=Complexity.LOW,
        risk=RiskLevel.LOW,
        latency_requirement=LatencyClass.STANDARD,
        assignment_only=True,
    )
    decision = router.route(req)
    assert decision.selected_model == "test/model"
    assert decision.provider == "openrouter"


# 2. Expired capability evidence transitions to STALE and triggers refresh
def test_expired_capability_evidence_transitions_to_stale_and_refreshes(tmp_path: Path) -> None:
    cache_path = tmp_path / "registry.json"
    registry = ModelCapabilityRegistry(cache_path)
    expired_time = datetime.now(timezone.utc) - timedelta(days=2)
    fact = CapabilityFact(
        state=CapabilityState.SUPPORTED,
        value=True,
        source="provider",
        observed_at=expired_time - timedelta(days=1),
        expires_at=expired_time,
    )
    profile = NativeProfile(
        provider="test-provider",
        model_id="test-model",
        facts={"output_text": fact},
        observations={"output_text": [fact]},
    )
    registry.register(profile)

    # In-memory retrieval marks expired fact as STALE
    retrieved = registry.get("test-provider", "test-model")
    assert retrieved is not None
    assert retrieved.fact("output_text").state is CapabilityState.STALE
    assert retrieved.is_stale("output_text")
    assert not retrieved.supports("output_text")

    # Refresh with fresh fact updates to SUPPORTED
    fresh_fact = observation(True, source="provider")
    updated = registry.merge(retrieved, {"output_text": fresh_fact})
    assert updated.supports("output_text")
    assert updated.fact("output_text").state is CapabilityState.SUPPORTED


# 3. Confirmed unsupported structured output allows validated routing
def test_confirmed_unsupported_structured_output_allows_validated_routing() -> None:
    native_desc = NativeProfile(
        provider="test-provider",
        model_id="test-model",
        facts={
            "input_text": observation(True, source="provider"),
            "output_text": observation(True, source="provider"),
            "native_structured_output": observation(False, source="provider"),
            "context_window": observation(32000, source="provider"),
            "max_output_tokens": observation(4096, source="provider"),
        },
    )
    profile = ModelProfile(
        provider="test-provider",
        model_id="test-model",
        context_window=32000,
        max_output_tokens=4096,
        supported_roles=frozenset({"*"}),
        supported_tools=frozenset({"*"}),
        can_structured_output=False,
        capability_descriptor=native_desc,
        configuration={"pricing_known": True},
    )
    router = ModelRouter([profile])
    # Generic "structured_output" requirement is satisfied by validated text routing
    req = RoutingRequest(
        role="head_decision",
        task_description="Classify route",
        task_type="routing",
        complexity=Complexity.LOW,
        risk=RiskLevel.LOW,
        required_capabilities=frozenset({"structured_output"}),
        latency_requirement=LatencyClass.STANDARD,
        assignment_only=True,
    )
    decision = router.route(req)
    assert decision.selected_model == "test-model"


# 4. Text-only models used for validated routing
def test_text_only_models_used_for_validated_routing() -> None:
    native_profile = NativeProfile(
        provider="test-provider",
        model_id="text-only-model",
        facts={
            "input_text": observation(True, source="provider"),
            "output_text": observation(True, source="provider"),
            "native_structured_output": observation(False, source="provider"),
            "tool_calling": observation(False, source="provider"),
        },
    )

    class TextAdapter:
        calls = 0

        async def generate(self, request, **kwargs):
            self.calls += 1
            # Model returns JSON text conforming to SampleRoutingDecision
            payload = {"candidate": "model-a", "reason": "Text routing choice", "confidence": 0.95}
            return ModelResponse(content=json.dumps(payload), usage=Usage(total_tokens=10))

    adapter = TextAdapter()
    engine = NativeLLMEngine(
        provider="test-provider",
        model="text-only-model",
        api_key="key",
        base_url="https://example.org",
        adapter=adapter,
    )
    engine.profile = lambda *args, **kwargs: native_profile

    result = engine.decide_sync("Route this task", SampleRoutingDecision)
    assert isinstance(result, SampleRoutingDecision)
    assert result.candidate == "model-a"
    assert result.confidence == 0.95
    assert adapter.calls == 1


# 5. Missing and unknown context limits are not invented
def test_missing_and_unknown_context_limits_not_invented() -> None:
    raw_record = {"id": "custom-model", "type": "chat"}
    facts = metadata_facts(raw_record, source="provider")
    assert "context_window" not in facts
    assert "max_output_tokens" not in facts

    # profiles_from_legacy_configuration does not default to 16384/4096
    profiles = profiles_from_legacy_configuration(global_model="custom-model", default_provider="openai")
    candidate = profiles[0]
    assert candidate.context_window == 0
    assert candidate.max_output_tokens == 0


# 6. Runtime verification success and failure
def test_runtime_verification_success_and_failure(tmp_path: Path) -> None:
    cache_path = tmp_path / "registry.json"
    registry = ModelCapabilityRegistry(cache_path)
    profile = NativeProfile(
        provider="test-p",
        model_id="test-m",
        facts={},
    )
    governor = SimpleNamespace(
        enabled=True,
        record_model_call=lambda *args, **kwargs: None,
        release_reservation=lambda *args, **kwargs: None,
    )

    # Success case: probe output_text
    class SuccessAdapter:
        async def generate(self, request, **kwargs):
            return ModelResponse(content="mana-capability-ok", usage=Usage(total_tokens=5))

    engine_success = SimpleNamespace(
        provider="test-p",
        model="test-m",
        parameters={},
        timeout=5,
        adapter=SuccessAdapter(),
        context_cost_governor=governor,
        _adapter=lambda: SuccessAdapter(),
        _reserve=lambda req: "res-1",
        profile=lambda *args: profile,
    )
    result_success = asyncio.run(run_capability_probe(engine_success, "output_text", authorized=True))
    assert result_success.state is CapabilityState.SUPPORTED

    # Failure case: unsupported parameter
    class FailAdapter:
        async def generate(self, request, **kwargs):
            err = RuntimeError("unsupported")
            err.status_code = 400
            err.body = {"error": {"code": "unsupported_parameter", "param": "tools"}}
            raise err

    engine_fail = SimpleNamespace(
        provider="test-p",
        model="test-m",
        parameters={},
        timeout=5,
        adapter=FailAdapter(),
        context_cost_governor=governor,
        _adapter=lambda: FailAdapter(),
        _reserve=lambda req: "res-2",
        profile=lambda *args: profile,
    )
    result_fail = asyncio.run(run_capability_probe(engine_fail, "tool_calling", authorized=True))
    assert result_fail.state is CapabilityState.UNSUPPORTED


# 7. Unauthorized or over-budget probes blocked
def test_unauthorized_or_over_budget_probes_blocked() -> None:
    settings = Settings()
    assert not is_probe_authorized(settings, "provider", "model", "tool_calling")

    engine = SimpleNamespace(
        provider="provider",
        model="model",
        parameters={},
        timeout=5,
        context_cost_governor=None,
    )
    with pytest.raises(PermissionError, match="authorized probe policy"):
        asyncio.run(run_capability_probe(engine, "tool_calling", authorized=False))


# 8. Model catalog changes
def test_model_catalog_changes_populate_registry() -> None:
    raw_catalog_item = {
        "id": "openrouter/~deepseek/deepseek-flash-latest",
        "context_length": 65536,
        "supported_parameters": ["tools", "reasoning"],
        "architecture": {"modality": "text->text"},
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
    }
    facts = metadata_facts(raw_catalog_item, source="provider")
    assert facts["input_text"].state is CapabilityState.SUPPORTED
    assert facts["output_text"].state is CapabilityState.SUPPORTED
    assert facts["tool_calling"].state is CapabilityState.SUPPORTED
    assert facts["context_window"].value == 65536


# 9. Concurrent capability discovery no probe storms
def test_concurrent_capability_discovery_no_probe_storms() -> None:
    call_count = 0

    class CountingAdapter:
        async def generate(self, request, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return ModelResponse(content="mana-capability-ok", usage=Usage(total_tokens=2))

    governor = SimpleNamespace(
        enabled=True,
        record_model_call=lambda *args, **kwargs: None,
        release_reservation=lambda *args, **kwargs: None,
    )
    profile = NativeProfile(provider="p-storm", model_id="m-storm")
    engine = SimpleNamespace(
        provider="p-storm",
        model="m-storm",
        parameters={},
        timeout=5,
        adapter=CountingAdapter(),
        context_cost_governor=governor,
        _adapter=lambda: CountingAdapter(),
        _reserve=lambda req: "res",
        profile=lambda *args: profile,
    )

    async def run_concurrent():
        tasks = [
            run_capability_probe(engine, "output_text", authorized=True)
            for _ in range(3)
        ]
        return await asyncio.gather(*tasks)

    results = asyncio.run(run_concurrent())
    assert all(r.state is CapabilityState.SUPPORTED for r in results)
    # Lock serializes and executes cleanly
    assert call_count == 3


# 10. Invalid model-generated routing decisions bounded retry and fail closed
def test_invalid_model_generated_routing_decisions() -> None:
    native_profile = NativeProfile(
        provider="test-p",
        model_id="test-m",
        facts={"output_text": observation(True, source="provider")},
    )

    class MalformedAdapter:
        calls = 0

        async def generate(self, request, **kwargs):
            self.calls += 1
            return ModelResponse(content="not valid json", usage=Usage(total_tokens=10))

    adapter = MalformedAdapter()
    engine = NativeLLMEngine(
        provider="test-p",
        model="test-m",
        api_key="key",
        base_url="https://example.org",
        adapter=adapter,
    )
    engine.profile = lambda *args, **kwargs: native_profile

    with pytest.raises(InvalidResponseError, match="failed schema validation"):
        engine.decide_sync("prompt", SampleRoutingDecision)
    # Initial attempt + 1 bounded correction attempt = 2 calls
    assert adapter.calls == 2


# 11. Provider timeouts and rate limits preserve UNKNOWN
def test_provider_timeouts_and_rate_limits_preserve_unknown() -> None:
    governor = SimpleNamespace(
        enabled=True,
        record_model_call=lambda *args, **kwargs: None,
        release_reservation=lambda *args, **kwargs: None,
    )
    profile = NativeProfile(provider="p-err", model_id="m-err")

    class RateLimitAdapter:
        async def generate(self, request, **kwargs):
            err = RuntimeError("Rate limited")
            err.status_code = 429
            err.body = {"error": {"code": "rate_limit"}}
            raise err

    engine = SimpleNamespace(
        provider="p-err",
        model="m-err",
        parameters={},
        timeout=5,
        adapter=RateLimitAdapter(),
        context_cost_governor=governor,
        _adapter=lambda: RateLimitAdapter(),
        _reserve=lambda req: "res",
        profile=lambda *args: profile,
    )
    result = asyncio.run(run_capability_probe(engine, "tool_calling", authorized=True))
    assert result.state is CapabilityState.UNKNOWN
    assert "HTTP 429" in result.diagnostic


# 12. Reliability and latency evidence gaps handled
def test_reliability_and_latency_evidence_gaps_handled() -> None:
    native_desc = NativeProfile(
        provider="fixture",
        model_id="gap-model",
        facts={
            "input_text": observation(True, source="provider"),
            "output_text": observation(True, source="provider"),
            "context_window": observation(100000, source="provider"),
            "max_output_tokens": observation(4000, source="provider"),
        },
    )
    profile = ModelProfile(
        provider="fixture",
        model_id="gap-model",
        context_window=100000,
        max_output_tokens=4000,
        supported_roles=frozenset({"*"}),
        supported_tools=frozenset({"*"}),
        reliability_score=0.8,
        configuration={"pricing_known": True},
        capability_descriptor=native_desc,
    )
    router = ModelRouter([profile])
    req = RoutingRequest(
        role="main",
        task_description="Normal task",
        task_type="routine",
        complexity=Complexity.LOW,
        risk=RiskLevel.LOW,
        latency_requirement=LatencyClass.STANDARD,
        assignment_only=True,
    )
    decision = router.route(req)
    assert decision.selected_model == "gap-model"
    assert decision.routing_score > 0.0


# 13. Budget constraint enforcement fails closed when pricing cannot be established
def test_budget_constraint_enforcement_never_classifies_unknown_as_free() -> None:
    native_desc = NativeProfile(
        provider="fixture",
        model_id="unpriced-model",
        facts={
            "input_text": observation(True, source="provider"),
            "output_text": observation(True, source="provider"),
            "context_window": observation(100000, source="provider"),
            "max_output_tokens": observation(4000, source="provider"),
        },
    )
    profile = ModelProfile(
        provider="fixture",
        model_id="unpriced-model",
        context_window=100000,
        max_output_tokens=4000,
        supported_roles=frozenset({"*"}),
        supported_tools=frozenset({"*"}),
        input_cost_per_million=0,
        output_cost_per_million=0,
        logical_cost_per_1k_tokens=0,
        configuration={"pricing_known": False},
        capability_descriptor=native_desc,
    )
    router = ModelRouter([profile])
    req = RoutingRequest(
        role="main",
        task_description="Cost constrained task",
        task_type="routine",
        complexity=Complexity.LOW,
        risk=RiskLevel.LOW,
        latency_requirement=LatencyClass.STANDARD,
        budgets=RoutingBudgets(task_cost_limit=0.50),
        assignment_only=False,
    )
    with pytest.raises(RoutingFailure, match="pricing is UNKNOWN for the cost-constrained operation"):
        router.route(req)


# 14. Normal chat startup independent of optional native structured output
def test_normal_chat_startup_independent_of_native_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    native_desc = NativeProfile(
        provider="openrouter",
        model_id="~deepseek/deepseek-flash-latest",
        facts={
            "input_text": observation(True, source="provider"),
            "output_text": observation(True, source="provider"),
            "native_structured_output": observation(False, source="provider"),
            "context_window": observation(64000, source="provider"),
            "max_output_tokens": observation(4096, source="provider"),
        },
    )
    profile = ModelProfile(
        provider="openrouter",
        model_id="~deepseek/deepseek-flash-latest",
        context_window=64000,
        max_output_tokens=4096,
        supported_roles=frozenset({"*"}),
        supported_tools=frozenset({"*"}),
        can_structured_output=False,
        configuration={"pricing_known": True},
        capability_descriptor=native_desc,
    )

    class FakeAuthority:
        def route(self, request):
            router = ModelRouter([profile])
            return router.route(request)

        def binding_for(self, decision_id):
            return None

    assignment = resolve_model_for_role(
        AgentRole.MAIN,
        global_model="openrouter/~deepseek/deepseek-flash-latest",
        routing_authority=FakeAuthority(),
    )
    assert assignment.resolved_model == "~deepseek/deepseek-flash-latest"


# 15. No eligible model after verification produces actionable diagnostics
def test_no_eligible_model_after_verification_yields_actionable_diagnostics() -> None:
    native_desc = NativeProfile(
        provider="openrouter",
        model_id="~deepseek/deepseek-flash-latest",
        facts={
            "input_text": observation(True, source="provider"),
            "output_text": observation(True, source="provider"),
            "native_structured_output": observation(False, source="provider"),
        },
    )
    profile = ModelProfile(
        provider="openrouter",
        model_id="~deepseek/deepseek-flash-latest",
        context_window=0,
        max_output_tokens=0,
        supported_roles=frozenset({"coding"}),
        supported_tools=frozenset({"*"}),
        can_structured_output=False,
        configuration={"pricing_known": True},
        capability_descriptor=native_desc,
    )
    router = ModelRouter([profile])
    req = RoutingRequest(
        role="coding",
        task_description="Requires native structured output",
        task_type="coding",
        complexity=Complexity.HIGH,
        risk=RiskLevel.MEDIUM,
        required_capabilities=frozenset({"native_structured_output"}),
        latency_requirement=LatencyClass.STANDARD,
    )
    with pytest.raises(RoutingFailure) as exc_info:
        router.route(req)

    diag = exc_info.value.format_diagnostics()
    assert "openrouter/~deepseek/deepseek-flash-latest" in diag
    assert "native_structured_output: UNSUPPORTED" in diag
    assert "Action:" in diag


# 16. Persistence and invalidation of capability evidence
def test_persistence_and_invalidation_of_capability_evidence(tmp_path: Path) -> None:
    cache_file = tmp_path / "model-capabilities.json"
    registry1 = ModelCapabilityRegistry(cache_file)
    profile = NativeProfile(
        provider="persisted-p",
        model_id="persisted-m",
        facts={"tool_calling": observation(True, source="provider")},
        observations={"tool_calling": [observation(True, source="provider")]},
    )
    registry1.register(profile, persist=True)
    assert cache_file.exists()

    registry2 = ModelCapabilityRegistry(cache_file)
    loaded = registry2.get("persisted-p", "persisted-m")
    assert loaded is not None
    assert loaded.supports("tool_calling")


# 17. No silent capability downgrade or model substitution
def test_no_silent_capability_downgrade() -> None:
    native_profile = NativeProfile(
        provider="test-p",
        model_id="text-only-model",
        facts={
            "input_text": observation(True, source="provider"),
            "output_text": observation(True, source="provider"),
            "native_structured_output": observation(False, source="provider"),
        },
    )
    engine = NativeLLMEngine(
        provider="test-p",
        model="text-only-model",
        api_key="key",
        base_url="https://example.org",
        adapter=SimpleNamespace(generate=lambda req, **kwargs: None),
    )
    engine.profile = lambda *args, **kwargs: native_profile

    # Explicitly requiring native structured output must fail closed and not downgrade
    with pytest.raises(CapabilityError, match="native_structured_output=UNSUPPORTED"):
        engine.decide_sync("prompt", SampleRoutingDecision, require_native_structured_output=True)
