from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mana_agent.config.settings import Settings
from mana_agent.gateway.lane_coordinator import LaneCoordinator, LaneCoordinatorError
from mana_agent.gateway.lanes import LaneId, LanePriority, LaneTaskState
from mana_agent.gateway.routing import GatewayRoutingAuthority
from mana_agent.model_routing.models import (
    Complexity,
    ModelProfile,
    RiskLevel,
    RoutingMode,
    RoutingRequest,
)


def _profile(provider: str, model: str, *, reliability: float, cost: float) -> ModelProfile:
    return ModelProfile(
        provider=provider,
        model_id=model,
        supported_roles=frozenset({"*"}),
        supported_tools=frozenset({"*"}),
        reasoning_settings=frozenset({"high"}),
        logical_cost_per_1k_tokens=cost,
        reliability_score=reliability,
        benchmark_scores={"coding": reliability, "verification": reliability},
    )


def _settings(**values) -> Settings:
    defaults = {
        "MANA_ROUTING_PARALLEL_ENABLED": True,
        "MANA_ROUTING_MULTI_AGENT_ENABLED": True,
        "MANA_ROUTING_MIN_PARALLEL_EVIDENCE": 0.6,
        "MANA_ROUTING_MAX_CONCURRENT_TASKS": 4,
    }
    defaults.update(values)
    return Settings(**defaults)


def test_gateway_authority_persists_every_invocation_and_emits_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    events: list[str] = []
    authority = GatewayRoutingAuthority(
        tmp_path,
        settings=_settings(),
        profiles=(_profile("one", "small", reliability=0.8, cost=0.1),),
        event_sink=lambda event, *args, **kwargs: events.append(event),
        decision_path=tmp_path / "decisions.jsonl",
    )
    request = RoutingRequest(
        role="main",
        task_description="Answer a routine question",
        task_type="routine",
        complexity=Complexity.LOW,
        risk=RiskLevel.LOW,
        task_id="task-1",
        session_id="session-1",
    )

    first = authority.route(request)
    second = authority.route(request)

    assert first.routing_mode is RoutingMode.SINGLE
    assert first.request_id != second.request_id
    assert first.decision_id != second.decision_id
    assert len(authority.history_rows()) == 2
    assert authority.latest(session_id="session-1")["decision"]["task_id"] == "task-1"
    assert events == ["routing.requested", "routing.completed"] * 2


def test_parallel_and_multi_agent_require_main_request_and_gateway_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    authority = GatewayRoutingAuthority(
        tmp_path,
        settings=_settings(),
        profiles=(
            _profile("one", "author", reliability=0.94, cost=1.0),
            _profile("two", "candidate", reliability=0.92, cost=1.1),
            _profile("three", "verifier", reliability=0.97, cost=1.2),
        ),
        decision_path=tmp_path / "decisions.jsonl",
    )
    base = RoutingRequest(
        role="coding",
        task_description="Implement a security-sensitive architecture change",
        task_type="coding",
        complexity=Complexity.CRITICAL,
        risk=RiskLevel.CRITICAL,
        task_id="task-parallel",
        session_id="session-1",
        multi_candidate_permitted=True,
        parallel_execution_allowed=True,
        subagents_allowed=True,
        main_model_requested_parallel=True,
        main_model_requested_multi_agent=True,
        isolation_available=True,
        independent_verifier_available=True,
        historical_parallel_benefit=1.0,
        historical_result_variance=1.0,
        similar_task_failures=3,
        plausible_strategy_count=3,
        maximum_concurrency=4,
        explicit_competition=True,
    )

    approved = authority.route(base)
    rejected = authority.route(replace(base, task_id="task-simple", main_model_requested_parallel=False))

    assert approved.routing_mode is RoutingMode.MULTI_AGENT_WITH_PARALLEL_CANDIDATES
    assert approved.parallel_execution_permitted is True
    assert approved.multi_agent_execution_permitted is True
    assert rejected.routing_mode is RoutingMode.MULTI_AGENT
    assert rejected.parallel_execution_permitted is False
    assert any("did not request parallel" in reason for reason in rejected.orchestration_reasons)


def test_retry_records_failure_and_creates_a_new_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    authority = GatewayRoutingAuthority(
        tmp_path,
        settings=_settings(),
        profiles=(
            _profile("one", "first", reliability=0.9, cost=0.2),
            _profile("two", "second", reliability=0.89, cost=0.3),
        ),
        decision_path=tmp_path / "decisions.jsonl",
    )
    request = RoutingRequest(
        role="coding",
        task_description="Retry a failed coding task",
        task_type="coding",
        complexity=Complexity.MEDIUM,
        risk=RiskLevel.MEDIUM,
        task_id="retry-task",
    )
    first = authority.route(request)

    second = authority.route_retry(
        request,
        previous_decision=first,
        failure_kind="timeout",
    )

    assert second.decision_id != first.decision_id
    assert second.request_id != first.request_id
    assert second.selected_model == "second"
    assert len(authority.history_rows()) == 2


def test_live_task_control_validates_transitions_and_releases_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    coordinator = LaneCoordinator(tmp_path)
    reservation = coordinator.reserve(
        normalized_intent="controlled task",
        lane_id=LaneId.CODING,
        session_id="session-1",
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
        target_files=("module.py",),
        requested_input_tokens=100,
        requested_output_tokens=200,
        routing_decision_id="decision-1",
        provider="fixture",
        model="fixture/model",
    )

    coordinator.pause(reservation.execution.task_id)
    coordinator.reprioritize(reservation.execution.task_id, LanePriority.CRITICAL)
    coordinator.resume(reservation.execution.task_id)
    coordinator.start(reservation)
    assert coordinator.inspect_task(reservation.execution.task_id).state is LaneTaskState.RUNNING
    assert coordinator._locks

    coordinator.request_verification(reservation.execution.task_id, level="independent")
    coordinator.transition(reservation.execution.task_id, LaneTaskState.COMPLETED)

    assert not coordinator._locks
    assert coordinator.budget_usage(session_id="session-1")["task_count"] == 1
    with pytest.raises(LaneCoordinatorError, match="Invalid task-state transition"):
        coordinator.transition(reservation.execution.task_id, LaneTaskState.RUNNING)
