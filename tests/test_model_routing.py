from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import subprocess
import pytest
from mana_agent.model_routing.competition import CandidateCompetition, CandidateEvidence, CandidateWorkspace, CompetitionJudgment
from mana_agent.model_routing.history import InMemoryRoutingHistory, JsonlRoutingHistory
from mana_agent.model_routing.models import (Complexity, LatencyClass, ModelProfile, RepositoryMetadata,
    RiskLevel, RoutingBudgets, RoutingFailure, RoutingOutcome, RoutingRequest, RoutingResourceScore,
    provider_request_overrides_from_configuration)
from mana_agent.model_routing.profiles import configured_profiles, profiles_from_legacy_configuration
from mana_agent.model_routing.router import ModelRouter
from mana_agent.model_routing.repository import RepositoryMetadataInspector
from mana_agent.llm.capabilities import ModelProfile as NativeProfile, observation


def profile(model: str, *, reliability: float = .9, cost: float = 1, **kwargs) -> ModelProfile:
    evidence = NativeProfile(provider="fixture", model_id=model, facts={key: observation(value, source="provider") for key, value in {
        "input_text": True, "output_text": True, "tool_calling": True, "native_structured_output": True,
        "context_window": 1000000, "max_output_tokens": 500000}.items()})
    return ModelProfile(provider="fixture", model_id=model, context_window=1000000, max_output_tokens=500000,
        supported_roles=frozenset({"coding", "tool", "verifier"}), supported_tools=frozenset({"*"}),
        can_tool_call=True, can_patch=True, can_structured_output=True, can_verify=True,
        input_cost_per_million=cost, output_cost_per_million=cost, reliability_score=reliability,
        configuration={"pricing_known": True}, capability_descriptor=evidence)


CHEAP, STRONG, VERIFIER = profile("cheap", cost=.2), profile("strong", cost=10), profile("verifier")


def request(**kwargs) -> RoutingRequest:
    return RoutingRequest(role="coding", task_description="fixture task", task_type="coding",
        complexity=Complexity.HIGH, risk=RiskLevel.LOW, latency_requirement=LatencyClass.STANDARD, **kwargs)


class SelectionModel:
    def __init__(self, candidate, **kwargs):
        self.choice = {"candidate": candidate, "reason": "Explicit model selection", "confidence": .99, "mode": "single", **kwargs}
        self.calls = []
        self.schemas = []
    def decide_sync(self, messages, schema):
        self.calls.append(json.loads(messages[-1].content))
        self.schemas.append(schema.model_json_schema())
        return schema.model_validate(self.choice)


def router(candidate="fixture/strong", **choice):
    return ModelRouter([CHEAP, STRONG, VERIFIER], policy=replace(ModelRouter([]).policy, parallel_execution_enabled=True), decision_model=SelectionModel(candidate, **choice))


def test_main_model_selects_even_when_candidate_is_more_expensive():
    selected = router()
    assert selected.route(request()).selected_model == "strong"
    assert len(selected.decision_model.calls) == 1
    assert {item["id"] for item in selected.decision_model.calls[0]["candidates"]} == {"fixture/cheap", "fixture/strong", "fixture/verifier"}


def test_main_model_can_choose_a_verifier_with_no_execution_role():
    verifier = replace(VERIFIER, supported_roles=frozenset({"verifier"}))
    decision_model = SelectionModel("fixture/strong", mode="single_with_verification", verifier=verifier.key)
    decision = ModelRouter([STRONG, verifier], decision_model=decision_model).route(request())
    payload, = decision_model.calls
    assert {item["id"] for item in payload["candidates"]} == {STRONG.key}
    assert verifier.key in payload["verifier_candidates"]
    assert decision.verifier_model == verifier.key


def test_startup_role_assignment_omits_competition_candidates_from_wire_schema():
    from mana_agent.llm.providers.openai import strict_schema

    selected = router()
    selected.decision_model.choice.pop("mode")
    decision = selected.route(request(assignment_only=True))
    payload, = selected.decision_model.calls
    schema, = selected.decision_model.schemas
    assert payload["permitted_modes"] == ["single"]
    wire_schema = strict_schema(schema)
    assert set(wire_schema["properties"]) == {"candidate", "reason", "confidence"}
    assert set(wire_schema["required"]) == {"candidate", "reason", "confidence"}
    assert wire_schema["additionalProperties"] is False
    assert decision.selected_model == "strong"
    assert decision.competition_candidates == ()
    assert not decision.parallel_execution_permitted


def test_startup_role_assignment_rejects_unoffered_competition_fields():
    selected = router(competition_candidates=[CHEAP.key, STRONG.key])
    selected.decision_model.choice.pop("mode")
    with pytest.raises(RoutingFailure, match="Main model selection failed"):
        selected.route(request(assignment_only=True))
    assert len(selected.decision_model.calls) == 1


def test_routing_rejects_unoffered_parallel_mode():
    selected = router(mode="parallel_candidates")
    with pytest.raises(RoutingFailure, match="outside the admitted authority"):
        selected.route(request())
    assert len(selected.decision_model.calls) == 1


def test_role_resolver_requests_assignment_only():
    from mana_agent.multi_agent.core.types import AgentRole
    from mana_agent.multi_agent.runtime.model_levels import resolve_model_for_role

    selected = router()
    selected.decision_model.choice.pop("mode")

    class Authority:
        def route(self, routing_request):
            assert routing_request.assignment_only
            return selected.route(routing_request)

        def binding_for(self, decision_id):
            return None

    result = resolve_model_for_role(AgentRole.CODING, global_model=CHEAP.model_id, routing_authority=Authority())
    assert result.resolved_model == STRONG.model_id
    assert result.routing_decision.routing_mode.value == "single"


def test_routing_reports_known_schema_fields_without_response_values():
    selected = router(confidence="fixture-sensitive-output")
    selected.decision_model.choice.pop("mode")
    with pytest.raises(RoutingFailure, match=r"role coding.*selection schema \(confidence\)") as error:
        selected.route(request(assignment_only=True))
    assert "fixture-sensitive-output" not in str(error.value)
    assert len(selected.decision_model.calls) == 1


def test_routing_reports_safe_structured_output_failure():
    from mana_agent.llm.errors import StructuredOutputError

    class Broken:
        def decide_sync(self, *_args):
            raise StructuredOutputError("The provider returned JSON that did not match the requested structured-output schema.")

    selected = ModelRouter([STRONG], decision_model=Broken())
    with pytest.raises(RoutingFailure, match="role coding.*did not match the requested structured-output schema"):
        selected.route(request(assignment_only=True))


def test_routing_rejects_competition_candidates_when_main_chooses_single_mode():
    selected = router(competition_candidates=[CHEAP.key, STRONG.key])
    with pytest.raises(RoutingFailure, match="Competition candidates were supplied for a non-competition mode"):
        selected.route(request(multi_candidate_permitted=True, parallel_execution_allowed=True,
            main_model_requested_parallel=True, independent_verifier_available=True, maximum_concurrency=2, isolation_available=True))
    assert "parallel_candidates" in selected.decision_model.calls[0]["permitted_modes"]
    assert len(selected.decision_model.calls) == 1


@pytest.mark.parametrize("choice", [
    {"candidate": "unoffered"}, {"candidate": "fixture/strong", "confidence": .01},
    {"candidate": "fixture/strong", "verifier": "fixture/strong"},
    {"candidate": "fixture/strong", "mode": "single_with_verification"},
    {"candidate": "fixture/strong", "mode": "multi_agent"},
    {"candidate": "fixture/strong", "mode": "parallel_candidates", "competition_candidates": ["fixture/cheap", "fixture/strong"]},
])
def test_invalid_or_unauthorized_model_choice_never_falls_back(choice):
    model = SelectionModel("fixture/strong")
    model.choice.update(choice)
    with pytest.raises(RoutingFailure):
        ModelRouter([CHEAP, STRONG], decision_model=model).route(request())
    assert len(model.calls) == 1


def test_unknown_limits_are_rejected_before_main_call():
    model = SelectionModel("fixture/cheap")
    with pytest.raises(RoutingFailure, match="No configured model"):
        ModelRouter([replace(CHEAP, context_window=0)], decision_model=model).route(request())
    assert model.calls == []


def test_resource_capacity_is_gathered_once_and_unavailable_candidates_are_not_offered():
    calls = []
    def resource(profile, request):
        calls.append(profile.key)
        return RoutingResourceScore(profile.model_id != "cheap", .8, 1, 1)
    model = SelectionModel("fixture/strong")
    decision = ModelRouter([CHEAP, STRONG], decision_model=model, resource_scoring=resource).route(request())
    assert calls == ["fixture/cheap", "fixture/strong"]
    assert len(model.calls[0]["candidates"]) == 1
    assert decision.resource_score.provider_capacity == .8


def test_cost_and_verification_reserve_are_enforced_before_selection():
    model = SelectionModel("fixture/strong")
    with pytest.raises(RoutingFailure):
        ModelRouter([STRONG], decision_model=model).route(request(expected_prompt_tokens=450000,
            expected_response_tokens=450000, budgets=RoutingBudgets(task_cost_limit=1, verification_reserve_ratio=.15)))
    assert model.calls == []


def test_repository_metadata_cache_refreshes_when_files_change(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "app.py").write_text("print('ok')")
    inspector = RepositoryMetadataInspector()
    first = inspector.inspect(tmp_path)
    assert inspector.inspect(tmp_path) is first
    (tmp_path / "security.py").write_text("TOKEN = None")
    second = inspector.inspect(tmp_path)
    assert second is not first
    assert first.fingerprint != second.fingerprint


def test_missing_repository_metadata_is_empty(tmp_path):
    metadata = RepositoryMetadataInspector().inspect(tmp_path / "absent")
    assert not metadata.languages and metadata.file_count == 0


def test_unknown_names_do_not_get_maintained_capabilities(monkeypatch):
    monkeypatch.setattr("mana_agent.model_routing.profiles.resolve_model_capability", lambda provider, model: NativeProfile(provider=provider, model_id=model))
    candidate = configured_profiles([{"provider": "openai", "model_id": "gpt-6-luna"}])[0]
    assert candidate.context_window == 0
    assert not candidate.can_tool_call


def test_model_failure_stops_selection_without_second_choice():
    class Broken:
        calls = 0
        def decide_sync(self, *args):
            self.calls += 1
            raise TimeoutError("controlled")
    model = Broken()
    with pytest.raises(RoutingFailure, match="no fallback"):
        ModelRouter([CHEAP], decision_model=model).route(request())
    assert model.calls == 1


def outcome(model, **kwargs):
    return RoutingOutcome(provider="fixture", model_id=model, model_configuration={}, task_category="coding",
        repository_languages=(), repository_frameworks=(), complexity="high", risk="high", routing_score=.99,
        selection_reason="controlled", occurred_at=datetime.now(timezone.utc), **kwargs)


class FakeExecutor:
    def __init__(self, root: Path) -> None:
        self.active_repository_root = root / "active"
        self.active_repository_root.mkdir()
        self.cleaned: list[str] = []
        self.promoted = ""

    def create_isolated(self, *, candidate_id: str, model: str) -> CandidateWorkspace:
        root = self.active_repository_root.parent / candidate_id
        root.mkdir()
        return CandidateWorkspace(candidate_id, root, model)

    def execute(self, workspace: CandidateWorkspace) -> CandidateEvidence:
        passed = workspace.candidate_id == "candidate-2"
        return CandidateEvidence(workspace.candidate_id, workspace.model, f"diff --git a/{workspace.candidate_id} b/{workspace.candidate_id}", ({"name": "pytest", "passed": passed, "exit_code": 0 if passed else 1},), changed_files=(workspace.candidate_id,))

    def promote(self, workspace: CandidateWorkspace, evidence: CandidateEvidence) -> None:
        assert workspace.root != self.active_repository_root
        self.promoted = evidence.candidate_id

    def cleanup(self, workspace: CandidateWorkspace) -> None:
        self.cleaned.append(workspace.candidate_id)


class FakeJudge:
    def __init__(self) -> None:
        self.received: tuple[dict, ...] = ()

    def judge(self, evidence: tuple[dict, ...], *, verifier_model: str | None) -> CompetitionJudgment:
        self.received = evidence
        assert verifier_model
        criteria = {
            "correctness": 1.0, "test_results": 1.0, "regression_risk": 1.0, "security": 1.0,
            "scope_discipline": 1.0, "maintainability": 1.0, "repository_conventions": 1.0,
            "patch_size": 1.0, "verification_completeness": 1.0, "cost_latency": 1.0,
        }
        return CompetitionJudgment("candidate-2", {"candidate-1": {**criteria, "correctness": 0.2}, "candidate-2": criteria}, ("tests passed",))


def test_candidates_are_isolated_judged_from_normalized_evidence_and_loser_cleaned(tmp_path: Path) -> None:
    executor, judge = FakeExecutor(tmp_path), FakeJudge()
    selected = router(CHEAP.key, mode="parallel_candidates", verifier=VERIFIER.key, competition_candidates=[CHEAP.key, STRONG.key])
    decision = selected.route(request(multi_candidate_permitted=True, parallel_execution_allowed=True,
        main_model_requested_parallel=True, independent_verifier_available=True, maximum_concurrency=2, isolation_available=True))
    assert "parallel_candidates" in selected.decision_model.calls[0]["permitted_modes"]
    assert "competition_candidates" in selected.decision_model.schemas[0]["properties"]
    result = CandidateCompetition(executor, judge).run(decision)
    assert result.winner.candidate_id == executor.promoted == "candidate-2"
    assert result.cleaned_candidates == ("candidate-1",)
    assert judge.received[0]["diff"].startswith("diff --git")
    assert "patch_bytes" in judge.received[0]


def test_provider_request_overrides_drop_routing_metadata() -> None:
    overrides = provider_request_overrides_from_configuration(
        {
            "source_levels": ("pinned", "MODEL_LEVEL_1_FAST_TOOL"),
            "capability_source": "maintained-token-limits",
            "token_profile_confidence": "high",
            "temperature": 0.1,
            "extra_body": {"foo": 1},
            "model_kwargs": {"bar": 2},
            "api_key": "must-not-forward",
        },
        for_http_body=True,
    )
    assert overrides == {"temperature": 0.1, "extra_body": {"foo": 1}}
    # Retired client-only settings must not reach an HTTP body.
    client_side = provider_request_overrides_from_configuration(
        {"model_kwargs": {"bar": 2}, "source_levels": ("pinned",)},
        for_http_body=False,
    )
    assert client_side == {"model_kwargs": {"bar": 2}}


def test_provider_request_overrides_drop_catalog_model_object_fields() -> None:
    """Catalog /v1/models records must not become chat/completions body params.

    Regression: gateway routing merged full NVIDIA model objects into
    profile.configuration; the Codex bridge then forwarded id/object/created/
    owned_by and NVIDIA returned HTTP 400 Unsupported parameter(s).
    """
    overrides = provider_request_overrides_from_configuration(
        {
            "id": "deepseek-ai/deepseek-v4-flash-0731",
            "object": "model",
            "created": 735790403,
            "owned_by": "deepseek-ai",
            "capabilities": ["text_generation"],
            "pricing": {"prompt": "0.1"},
            "temperature": 0.2,
            "extra_body": {
                "id": "nested-must-drop",
                "owned_by": "nested-must-drop",
                "chat_template_kwargs": {"thinking": False},
            },
        },
        for_http_body=True,
    )
    assert overrides == {
        "temperature": 0.2,
        "extra_body": {"chat_template_kwargs": {"thinking": False}},
    }
    for key in ("id", "object", "created", "owned_by", "capabilities", "pricing"):
        assert key not in overrides
    assert "id" not in overrides["extra_body"]
    assert "owned_by" not in overrides["extra_body"]


def test_history_persistence_redacts_secrets(tmp_path: Path) -> None:
    path = tmp_path / "routing.jsonl"
    store = JsonlRoutingHistory(path)
    row = replace(outcome("safe", accepted=True), model_configuration={"temperature": 0, "api_key": "never-write", "secret_ref": "hidden", "headers": {"Authorization": "Bearer nested-secret"}, "label": "sk_abcdefghijk"})
    ModelRouter([replace(CHEAP, model_id="safe")], history=store).record_outcome(row)
    text = path.read_text()
    assert "never-write" not in text and "hidden" not in text and "nested-secret" not in text and "sk_abcdefghijk" not in text
    assert "temperature" in text
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text("not-json\n")
    assert JsonlRoutingHistory(corrupt).healthy() is False
