from __future__ import annotations

import inspect
from dataclasses import replace

from mana_agent.config.settings import Settings
from mana_agent.gateway.routing import GatewayRoutingAuthority
from mana_agent.model_routing.models import (
    Complexity,
    LatencyClass,
    ModelProfile,
    RiskLevel,
    RoutingBudgets,
    RoutingDecision,
    RoutingMode,
    RoutingRequest,
)
from mana_agent.model_routing.router import ModelRouter
from mana_agent.multi_agent.core.types import AgentRole, ExecutionContext
from mana_agent.multi_agent.registry.agent_registry import AgentRegistry
from mana_agent.spirit.compiler import (
    SPIRIT_MAX_COMPILED_TOKENS,
    compile_spirit_instruction,
    compile_spirit_semantics,
    estimate_spirit_tokens,
)
from mana_agent.spirit.routing import (
    attach_spirit_accounting,
    bind_after_route,
    bind_parallel_candidates,
    bind_runtime_self,
    estimate_pre_route_spirit_tokens,
    resolve_base_self,
    resume_runtime_self,
    route_with_spirit,
)
from mana_agent.spirit.self_model import compose_runtime_self


def _profile(model: str, *, reliability: float, cost: float, provider: str = "fixture") -> ModelProfile:
    return ModelProfile(
        provider=provider,
        model_id=model,
        supported_roles=frozenset({"coding", "planner", "reviewer", "main", "verifier", "*"}),
        context_window=1_000_000,
        max_output_tokens=500_000,
        reasoning_settings=frozenset({"high"}) if reliability > 0.9 else frozenset({"none"}),
        logical_cost_per_1k_tokens=cost,
        reliability_score=reliability,
        benchmark_scores={"routine": reliability, "coding": reliability, "verification": reliability},
    )


CHEAP = _profile("cheap", reliability=0.75, cost=0.2)
STRONG = _profile("strong", reliability=0.97, cost=10.0)


def _request(**kwargs) -> RoutingRequest:
    payload = dict(
        role="coding",
        task_description="fixture task",
        task_type="routine",
        complexity=Complexity.LOW,
        risk=RiskLevel.LOW,
        latency_requirement=LatencyClass.STANDARD,
    )
    payload.update(kwargs)
    return RoutingRequest(**payload)


def test_spirit_resolves_before_routing_without_compiled_prompt() -> None:
    seen: list[str] = []

    class _RecordingRouter:
        def route(self, request: RoutingRequest) -> RoutingDecision:
            seen.append(request.task_description)
            return ModelRouter([CHEAP]).route(request)

    base_self = resolve_base_self(role="coding", purpose="inspect routing")
    assert base_self.spirit.id == "mana"
    assert base_self.spirit.version == 1
    assert base_self.spirit.identity.name == "Mana"
    assert base_self.spirit.temperament.curious.meaning
    assert base_self.spirit.temperament.bold.meaning
    assert base_self.spirit.temperament.calm.meaning
    compiled = compile_spirit_semantics(base_self.ref())
    assert "instantiated" not in compiled.lower()
    assert "openai" not in compiled.lower()

    binding = route_with_spirit(_RecordingRouter(), _request())
    assert seen
    assert "Mana-Agent" not in seen[0]
    assert "curious" not in seen[0]
    assert binding.compiled.startswith("You are Mana-Agent, currently instantiated through")


def test_spirit_traits_are_not_router_scoring_inputs() -> None:
    source = "\n".join(
        (
            inspect.getsource(ModelRouter._score),
            inspect.getsource(ModelRouter._reject),
            inspect.getsource(ModelRouter.route),
            inspect.getsource(RoutingRequest),
        )
    )
    lowered = source.lower()
    assert "spirit" not in lowered
    assert "curious" not in lowered
    assert "bold" not in lowered
    assert "calm" not in lowered
    assert not hasattr(RoutingRequest, "curious")
    assert "spirit" not in RoutingRequest.__dataclass_fields__


def test_existing_unconstrained_winner_is_unchanged() -> None:
    router = ModelRouter([CHEAP, STRONG])
    request = _request()
    bare = router.route(request)
    binding = route_with_spirit(router, request)
    assert binding.decision.selected_model == bare.selected_model
    assert binding.runtime_self.runtime.model == bare.selected_model
    assert binding.runtime_self.runtime.provider == bare.provider


def test_selected_model_is_attached_after_routing() -> None:
    binding = route_with_spirit(ModelRouter([CHEAP, STRONG]), _request())
    assert binding.runtime_self.runtime.model == binding.decision.selected_model
    assert binding.runtime_self.runtime.provider == binding.decision.provider
    assert binding.runtime_self.spirit.id == binding.base_self.spirit.id
    assert "instantiated through" in binding.compiled
    assert binding.decision.selected_model in binding.compiled


def test_model_switch_and_retry_preserve_spirit() -> None:
    base_self = resolve_base_self(role="coding", purpose="retry task")
    first = bind_after_route(
        base_self,
        ModelRouter([CHEAP, STRONG]).route(_request()),
    )
    second_decision = ModelRouter([STRONG]).route(_request(task_type="coding", complexity=Complexity.CRITICAL, risk=RiskLevel.CRITICAL))
    second = bind_after_route(base_self, second_decision)
    assert first.runtime_self.runtime.model != second.runtime_self.runtime.model
    assert first.runtime_self.spirit == second.runtime_self.spirit
    assert first.base_self.spirit.ref() == second.base_self.spirit.ref()


def test_provider_fallback_preserves_spirit() -> None:
    base_self = resolve_base_self(role="coding")
    openai_like = bind_runtime_self(base_self, provider="openai", model="gpt-4.1-mini")
    anthropic_like = bind_runtime_self(base_self, provider="anthropic", model="claude-sonnet-4")
    assert openai_like.spirit == anthropic_like.spirit
    assert openai_like.runtime.provider != anthropic_like.runtime.provider


def test_parallel_candidates_share_spirit() -> None:
    decision = replace(
        ModelRouter([CHEAP, STRONG]).route(
            _request(complexity=Complexity.HIGH, task_type="coding", multi_candidate_permitted=True)
        ),
        competition_candidates=("fixture/cheap", "fixture/strong"),
        candidate_competition=True,
        routing_mode=RoutingMode.PARALLEL_CANDIDATES,
    )
    base_self = resolve_base_self(role="coding")
    selves = bind_parallel_candidates(base_self, decision)
    assert len(selves) == 2
    assert selves[0].spirit == selves[1].spirit == base_self.ref()
    assert {item.runtime.model for item in selves} == {"cheap", "strong"}
    compiled = {compile_spirit_instruction(item) for item in selves}
    assert len(compiled) == 2
    assert all("curious" in text and "bold" in text and "calm" in text for text in compiled)


def test_roles_can_use_different_models_with_same_spirit() -> None:
    planner = resolve_base_self(agent_id="planner-agent", role="planner", purpose="plan")
    coding = resolve_base_self(agent_id="coding-agent", role="coding", purpose="edit")
    planner_self = bind_runtime_self(planner, provider="fixture", model="strong")
    coding_self = bind_runtime_self(coding, provider="fixture", model="cheap")
    assert planner.spirit.ref() == coding.spirit.ref()
    assert planner_self.spirit == coding_self.spirit
    assert planner_self.agent.role != coding_self.agent.role
    assert planner_self.runtime.model != coding_self.runtime.model


def test_spirit_is_compiled_only_after_route_selection() -> None:
    base_self = resolve_base_self(role="coding")
    pre = compile_spirit_semantics(base_self.ref())
    assert "instantiated through" not in pre
    binding = route_with_spirit(ModelRouter([CHEAP]), _request())
    assert "currently instantiated through fixture/cheap" in binding.compiled
    assert pre in binding.compiled


def test_spirit_is_injected_once_in_effective_model_input() -> None:
    binding = route_with_spirit(ModelRouter([CHEAP]), _request())
    from mana_agent.spirit.adapter import apply_spirit_instruction

    prompt = apply_spirit_instruction("Core Identity\nUse apply_patch.", binding.runtime_self)
    doubled = apply_spirit_instruction(prompt, binding.runtime_self)
    marker = "Mana's Spirit (mana/1)"
    assert prompt.count(marker) == 1
    assert doubled.count(marker) == 1
    assert doubled.index(marker) < doubled.index("Core Identity")
    assert "apply_patch" in doubled


def test_routing_token_accounting_includes_small_spirit_payload() -> None:
    request = _request(expected_prompt_tokens=100)
    base_self = resolve_base_self(role="coding")
    reserved = estimate_pre_route_spirit_tokens(base_self)
    prepared = attach_spirit_accounting(request, base_self)
    assert 40 <= reserved <= SPIRIT_MAX_COMPILED_TOKENS
    assert prepared.expected_prompt_tokens == 100 + reserved
    assert "curious" not in prepared.task_description
    assert prepared.estimation_components == request.estimation_components
    compiled_tokens = estimate_spirit_tokens(compile_spirit_instruction(compose_runtime_self(base_self=base_self, model="cheap")))
    assert abs(compiled_tokens - reserved) <= 8


def test_checkpoint_resume_reconstructs_same_spirit() -> None:
    binding = route_with_spirit(ModelRouter([CHEAP]), _request(task_id="task-1"))
    context = ExecutionContext(
        agent_id="subagent_coding_0001",
        agent_role="coding",
        resolved_model="other-model",
        spirit_id=binding.runtime_self.spirit.id,
        spirit_version=binding.runtime_self.spirit.version,
    )
    restored = ExecutionContext.from_mapping(context.as_dict())
    resumed = resume_runtime_self(
        spirit_id=restored.spirit_id,
        spirit_version=restored.spirit_version,
        provider="fixture",
        model="resumed-model",
        role=restored.agent_role,
    )
    assert resumed.spirit == binding.runtime_self.spirit
    assert resumed.runtime.model == "resumed-model"
    assert compile_spirit_instruction(resumed) not in str(context.as_dict())


def test_legacy_configuration_and_subagents_keep_default_spirit() -> None:
    legacy = resolve_base_self(role="main")
    assert legacy.spirit.id == "mana"
    registry = AgentRegistry()
    main = registry.find_by_role(AgentRole.MAIN)
    coding = registry.create_subagent(AgentRole.CODING, main.agent_id, ["edit"])
    assert main.spirit_id == coding.spirit_id == "mana"
    assert main.spirit_version == coding.spirit_version == 1


def test_gateway_retry_and_persist_keep_spirit_ref_not_prompt(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    events: list[dict] = []

    def _sink(event, *args, **kwargs):
        metadata = kwargs.get("metadata") if "metadata" in kwargs else (args[0] if args else {})
        if not isinstance(metadata, dict):
            metadata = kwargs
        events.append({"event": event, **dict(metadata)})

    authority = GatewayRoutingAuthority(
        tmp_path,
        settings=Settings(),
        profiles=(CHEAP, _profile("second", reliability=0.74, cost=0.3, provider="other")),
        event_sink=_sink,
        decision_path=tmp_path / "decisions.jsonl",
    )
    request = _request(role="coding", task_id="retry-task", task_description="Retry a failed coding task", task_type="coding")
    first = authority.route(request)
    first_binding = authority.binding_for(first.decision_id)
    assert first_binding is not None
    assert first_binding.runtime_self.spirit.id == "mana"
    second = authority.route_retry(request, previous_decision=first, failure_kind="timeout")
    second_binding = authority.binding_for(second.decision_id)
    assert second_binding is not None
    assert second_binding.runtime_self.spirit == first_binding.runtime_self.spirit
    rows = authority.history_rows()
    assert all(row.get("spirit") == {"id": "mana", "version": 1} for row in rows)
    assert all("curious" not in str(row.get("spirit")) for row in rows)
    assert all("You are Mana-Agent" not in str(row) for row in rows)
    requested = [item for item in events if item["event"] == "routing.requested"]
    completed = [item for item in events if item["event"] == "routing.completed"]
    assert requested and requested[0]["spirit_id"] == "mana"
    assert requested[0]["spirit_version"] == 1
    assert completed[-1]["selected_model"] == second.selected_model
    assert completed[-1]["spirit_id"] == "mana"
