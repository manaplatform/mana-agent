from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess

import pytest

from mana_agent.model_routing.competition import (
    CandidateCompetition, CandidateEvidence, CandidateWorkspace, CompetitionJudgment,
)
from mana_agent.model_routing.history import InMemoryRoutingHistory, JsonlRoutingHistory
from mana_agent.model_routing.models import (
    Complexity, LatencyClass, ModelProfile, RepositoryMetadata, RiskLevel,
    RoutingBudgets, RoutingFailure, RoutingOutcome, RoutingRequest,
    RoutingResourceScore,
    provider_request_overrides_from_configuration,
)
from mana_agent.model_routing.profiles import configured_profiles, profiles_from_legacy_configuration
from mana_agent.model_routing.router import ModelRouter
from mana_agent.model_routing.repository import RepositoryMetadataInspector


def profile(
    model: str,
    *,
    reliability: float,
    cost: float,
    languages: frozenset[str] = frozenset(),
    roles: frozenset[str] = frozenset({"coding", "tool", "verifier"}),
    benchmarks: dict[str, float] | None = None,
) -> ModelProfile:
    return ModelProfile(
        provider="fixture", model_id=model, supported_roles=roles,
        context_window=1_000_000, max_output_tokens=500_000,
        reasoning_settings=frozenset({"high"}) if reliability > 0.9 else frozenset({"none"}),
        logical_cost_per_1k_tokens=cost, reliability_score=reliability,
        supported_languages=languages, benchmark_scores=benchmarks or {},
    )


CHEAP = profile("cheap", reliability=0.75, cost=0.2, benchmarks={"routine": 0.9, "coding": 0.45})
STRONG = profile("strong", reliability=0.97, cost=10, benchmarks={"routine": 0.97, "coding": 0.99, "verification": 0.99})


def request(*, complexity: Complexity = Complexity.LOW, risk: RiskLevel = RiskLevel.LOW, task_type: str = "routine", **kwargs) -> RoutingRequest:
    return RoutingRequest(
        role="coding", task_description="fixture task", task_type=task_type,
        complexity=complexity, risk=risk, latency_requirement=LatencyClass.STANDARD,
        **kwargs,
    )


def test_routine_task_chooses_cheaper_qualified_model_and_high_risk_chooses_stronger() -> None:
    router = ModelRouter([CHEAP, STRONG])
    assert router.route(request()).selected_model == "cheap"
    assert router.route(request(complexity=Complexity.CRITICAL, risk=RiskLevel.CRITICAL, task_type="coding")).selected_model == "strong"


def test_resource_score_is_consumed_once_and_persisted_on_decision() -> None:
    calls: list[str] = []

    def resources(model: ModelProfile, _request: RoutingRequest) -> RoutingResourceScore:
        calls.append(model.model_id)
        return RoutingResourceScore(
            available=True,
            provider_capacity=1.0 if model.model_id == "cheap" else 0.0,
            quota_health=1.0,
            resource_confidence=1.0,
        )

    policy = replace(ModelRouter([CHEAP, STRONG]).policy, weights={
        "capability": 0.0, "quality": 0.0, "history": 0.0, "language": 0.0,
        "cost": 0.0, "latency": 0.0, "provider_capacity": 1.0,
        "quota_health": 0.0, "resource_confidence": 0.0,
    })
    decision = ModelRouter([CHEAP, STRONG], policy=policy, resource_scoring=resources).route(request())

    assert decision.selected_model == "cheap"
    assert decision.resource_score.provider_capacity == 1.0
    assert calls == ["cheap", "strong"]


def test_repository_language_changes_ranking() -> None:
    python_model = replace(STRONG, model_id="python", supported_languages=frozenset({"python"}), reliability_score=0.9)
    rust_model = replace(STRONG, model_id="rust", supported_languages=frozenset({"rust"}), reliability_score=0.9)
    decision = ModelRouter([rust_model, python_model]).route(request(repository=RepositoryMetadata(languages=("python",))))
    assert decision.selected_model == "python"


def test_repository_metadata_is_cached_by_fingerprint_and_sensitive_changes_raise_demand(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "app.py").write_text("print('ok')\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "app.py"], check=True)
    inspector = RepositoryMetadataInspector()
    first = inspector.inspect(tmp_path)
    assert first.languages == ("python",)
    assert inspector.inspect(tmp_path) is first
    auth = tmp_path / "auth"
    auth.mkdir()
    (auth / "security.py").write_text("TOKEN = None\n")
    second = inspector.inspect(tmp_path)
    assert second is not first
    assert "auth" in second.sensitive_areas or "security" in second.sensitive_areas
    decision = ModelRouter([CHEAP, STRONG]).route(request(repository=second, multi_candidate_permitted=True))
    assert decision.candidate_competition is True


def test_repository_metadata_inspector_nonexistent_or_non_directory_root(tmp_path: Path) -> None:
    inspector = RepositoryMetadataInspector()
    nonexistent = tmp_path / "does_not_exist_xyz"
    metadata = inspector.inspect(nonexistent)
    assert metadata.file_count == 0
    assert metadata.languages == ()
    assert metadata.changed_files == ()
    assert len(metadata.fingerprint) == 20

    file_path = tmp_path / "regular_file.txt"
    file_path.write_text("not a dir")
    metadata_file = inspector.inspect(file_path)
    assert metadata_file.file_count == 0
    assert metadata_file.languages == ()


def test_budget_rejects_expensive_candidate_and_reserves_verification() -> None:
    priced = replace(STRONG, input_cost_per_million=1.0, output_cost_per_million=1.0, context_window=1_000_000)
    req = request(
        expected_prompt_tokens=450_000, expected_response_tokens=450_000,
        budgets=RoutingBudgets(task_token_limit=1_000_000, task_cost_limit=1.0, verification_reserve_ratio=0.15),
    )
    with pytest.raises(RoutingFailure) as caught:
        ModelRouter([priced]).route(req)
    assert "verification reserve" in caught.value.rejected[0].reasons[0]
    cheaper = replace(priced, model_id="within-budget", input_cost_per_million=0.8, output_cost_per_million=0.8)
    assert ModelRouter([cheaper]).route(req).selected_model == "within-budget"


def outcome(model: str, *, accepted: bool = False, failure: str = "", age_seconds: int = 0) -> RoutingOutcome:
    return RoutingOutcome(
        provider="fixture", model_id=model, model_configuration={}, task_category="coding",
        repository_languages=(), repository_frameworks=(), complexity="high", risk="high",
        routing_score=0.8, selection_reason="fixture", accepted=accepted,
        verification_passed=accepted, failure_kind=failure,
        occurred_at=datetime.now(timezone.utc) - timedelta(seconds=age_seconds),
    )


def test_history_affects_ranking_recent_failure_penalty_decays_and_circuit_breaker_excludes() -> None:
    left = replace(STRONG, model_id="left", reliability_score=0.9)
    right = replace(STRONG, model_id="right", reliability_score=0.9)
    successful = InMemoryRoutingHistory(tuple(outcome("right", accepted=True) for _ in range(4)))
    assert ModelRouter([left, right], history=successful).route(request(task_type="coding", complexity=Complexity.HIGH)).selected_model == "right"

    recent = InMemoryRoutingHistory((outcome("left", failure="rate_limit"),))
    assert ModelRouter([left, right], history=recent).route(request()).selected_model == "right"
    recent_score = ModelRouter([left], history=recent).route(request()).routing_score
    old = InMemoryRoutingHistory((outcome("left", failure="rate_limit", age_seconds=100_000),))
    old_score = ModelRouter([left], history=old).route(request()).routing_score
    assert old_score > recent_score

    breaker = InMemoryRoutingHistory(tuple(outcome("left", failure="timeout") for _ in range(3)))
    independent_provider = replace(right, provider="fixture-2")
    decision = ModelRouter([left, independent_provider], history=breaker).route(request())
    assert decision.selected_model == "right"
    assert any(item.model == "fixture/left" and "circuit breaker" in item.reasons[0] for item in decision.rejected_candidates)


def test_competition_policy_is_threshold_budget_and_latency_controlled() -> None:
    base = request(multi_candidate_permitted=True)
    assert ModelRouter([CHEAP, STRONG]).route(base).candidate_competition is False
    difficult = replace(base, complexity=Complexity.HIGH, task_type="coding")
    assert ModelRouter([CHEAP, STRONG]).route(difficult).candidate_competition is True
    interactive = replace(difficult, latency_requirement=LatencyClass.INTERACTIVE)
    assert ModelRouter([replace(CHEAP, latency_class=LatencyClass.INTERACTIVE), replace(STRONG, latency_class=LatencyClass.INTERACTIVE)]).route(interactive).candidate_competition is False
    constrained = replace(difficult, budgets=RoutingBudgets(competition_cost_limit=0.01))
    assert ModelRouter([CHEAP, STRONG]).route(constrained).candidate_competition is False


def test_identical_inputs_and_evidence_produce_identical_decisions() -> None:
    router = ModelRouter([STRONG, CHEAP])
    assert router.route(request()) == router.route(request())


def test_configured_and_legacy_profiles_migrate_without_role_lock(monkeypatch) -> None:
    explicit = configured_profiles([{"provider": "custom", "model_id": "m", "supported_roles": ["coding"], "context_window": 4096, "max_output_tokens": 1024}])
    assert explicit[0].key == "custom/m"
    monkeypatch.setattr("mana_agent.model_routing.profiles.get_setting", lambda name, default="": {"MODEL_LEVEL_1_FAST_TOOL": "openai/fast"}.get(name, default))
    migrated = profiles_from_legacy_configuration(
        global_model="openai/current", context_window=16_384, max_output_tokens=4_096
    )
    assert {item.key for item in migrated} == {"openai/fast", "openai/current"}
    assert "coding" in migrated[0].supported_roles


def test_pinned_profiles_ignore_operator_model_levels(monkeypatch) -> None:
    from mana_agent.model_routing.profiles import profiles_for_pinned_models

    monkeypatch.setattr(
        "mana_agent.model_routing.profiles.get_setting",
        lambda name, default="": {"MODEL_LEVEL_3_HIGH_REASONING": "openai/gpt-5.6-luna"}.get(name, default),
    )
    pinned = profiles_for_pinned_models(["gpt-4.1-mini", "gpt-4.1-mini"])
    assert len(pinned) == 1
    assert pinned[0].model_id == "gpt-4.1-mini"
    assert pinned[0].context_window >= 100_000
    assert pinned[0].source_level == "pinned"
    # Pinned models must satisfy interactive entry routing (head_decision).
    assert pinned[0].latency_class is LatencyClass.INTERACTIVE
    router = ModelRouter(pinned)
    decision = router.route(
        RoutingRequest(
            role="head_decision",
            task_description="Classify the gateway entry route for: fixture",
            task_type="routing",
            complexity=Complexity.MEDIUM,
            risk=RiskLevel.MEDIUM,
            required_capabilities=frozenset({"structured_output"}),
            latency_requirement=LatencyClass.INTERACTIVE,
        )
    )
    assert decision.selected_model == "gpt-4.1-mini"


def test_legacy_profiles_apply_maintained_token_limits_for_known_models(monkeypatch) -> None:
    monkeypatch.setattr(
        "mana_agent.model_routing.profiles.get_setting",
        lambda name, default="": {"MODEL_LEVEL_3_HIGH_REASONING": "openai/gpt-5.6-luna"}.get(name, default),
    )
    migrated = profiles_from_legacy_configuration(context_window=16_384, max_output_tokens=4_096)
    assert len(migrated) == 1
    assert migrated[0].model_id == "gpt-5.6-luna"
    assert migrated[0].context_window == 400_000
    assert migrated[0].max_output_tokens == 128_000


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
    decision = ModelRouter([CHEAP, STRONG]).route(request(complexity=Complexity.HIGH, task_type="coding", multi_candidate_permitted=True))
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
    # LangChain path may keep model_kwargs; HTTP body path must not.
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
