from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from typer.testing import CliRunner

from mana_agent.commands.cli import app
from mana_agent.evals.baseline import baseline_as_runs, create_baseline, load_baseline, write_baseline
from mana_agent.evals.comparison import compare_runs
from mana_agent.evals.config import EvalConfigurationError, EvalSuite, GateThresholds, ScoreWeights, load_suite
from mana_agent.evals.ids import stable_hash, stable_id
from mana_agent.evals.inspector import inspect_run
from mana_agent.evals.leaderboard import build_leaderboard
from mana_agent.evals.matrix import ExperimentRunner, expand_matrix
from mana_agent.evals.models import (
    EnvironmentSnapshot,
    EvalRun,
    EvalTask,
    EvalVariant,
    EvaluationResult,
    ExpectedOutcome,
    RouteRecord,
    RunStatus,
    TestResult as EvalTestResult,
)
from mana_agent.evals.pricing import Price, PricingRegistry
from mana_agent.evals.recorder import ArtifactEvalRecorder, EvalExecutionContext, NullEvalRecorder, current_eval_context, record_current, use_eval_context
from mana_agent.evals.redaction import redact, redact_text
from mana_agent.evals.regression import evaluate_gate
from mana_agent.evals.replay import trajectory_replay
from mana_agent.evals.reports import write_leaderboard, write_regression
from mana_agent.evals.runner import EvalRunner
from mana_agent.evals.scoring import score_run
from mana_agent.evals.storage import CompletedRunImmutableError, EvalStorage, EvalStorageError
from mana_agent.evals.workspace import EvalWorkspaceError, LocalWorktreeBackend, workspace_backend
from mana_agent.multi_agent.runtime.compatibility import CompatibleChatOpenAI, ModelCapabilities


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def native_shell_command(*args: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(args)
    return shlex.join(args)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.email", "eval@example.test")
    git(root, "config", "user.name", "Eval")
    (root / "sample.txt").write_text("before\n", encoding="utf-8")
    git(root, "add", "sample.txt")
    git(root, "commit", "-m", "initial")
    return root


def environment(commit: str = "abc") -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        repository_commit=commit,
        dirty=False,
        starting_diff_hash="diff",
        python_version="3.12",
        mana_agent_version="0.0.17",
        operating_system="test",
        architecture="test",
        workspace_backend="local-worktree",
        git_version="git version test",
        reproducible=True,
    )


def variant(identifier: str = "candidate") -> EvalVariant:
    return EvalVariant(
        variant_id=identifier,
        display_name=identifier.title(),
        main_model="fixture-model",
        router_model="fixture-model",
        coding_model="fixture-model",
        reviewer_model="fixture-model",
        verifier_model="fixture-model",
    )


def task(repository: Path, *, identifier: str = "task", test_commands: list[str] | None = None) -> EvalTask:
    return EvalTask(
        task_id=identifier,
        suite_name="fixture",
        suite_version="1",
        description="Inspect sample.txt without changing it.",
        repository=str(repository),
        repository_commit="HEAD",
        test_commands=test_commands or [],
        expected=ExpectedOutcome(route="repository", no_repository_mutation=not bool(test_commands)),
    )


def suite(repository: Path) -> EvalSuite:
    return EvalSuite(name="fixture", version="1", tasks=[task(repository)], variants=[variant()])


def completed_run(identifier: str, task_id: str, *, score: float = 100, success: bool = True, route: str = "repository") -> EvalRun:
    now = datetime.now(timezone.utc)
    return EvalRun(
        run_id=identifier,
        run_fingerprint=stable_hash(identifier),
        experiment_id="experiment",
        task_id=task_id,
        variant_id="variant",
        started_at=now,
        completed_at=now,
        status=RunStatus.COMPLETED,
        repository_commit="abc",
        environment=environment(),
        routes=[RouteRecord(route_input_hash="input", router_model="fixture", intent=route)],
        final_answer="done",
        outcome=EvaluationResult(
            setup_success=True,
            task_success=success,
            normalized_score=score,
            score_dimensions={"routing_correctness": 1.0},
            first_attempt_success=success,
            final_success_after_retries=success,
        ),
    )


class FakeGateway:
    def __init__(self, root: Path, _variant: EvalVariant, *, mutate: bool = False) -> None:
        self.root = root
        self.mutate = mutate

    def create_session(self, *, frontend: str) -> str:
        assert frontend == "eval"
        return "session"

    def process_turn(self, session_id: str, prompt: str):
        assert session_id == "session"
        record_current("model.call", {"usage": {"input_tokens": 4, "output_tokens": 2}, "latency_seconds": 0.01})
        if self.mutate:
            (self.root / "sample.txt").write_text("after\n", encoding="utf-8")
        decision = SimpleNamespace(to_dict=lambda: {"intent": "repository", "confidence": 1.0, "selected_tools": ["read_file"]})
        return SimpleNamespace(
            answer="inspected",
            error=None,
            warnings=[],
            decision=decision,
            payload={"entry_route": "repository", "lane_id": "analysis"},
            trace=[{"tool_name": "read_file", "status": "success", "result_summary": "read sample"}],
        )


def test_models_round_trip_and_stable_fingerprint(repository: Path) -> None:
    first = task(repository)
    second = EvalTask.model_validate_json(first.model_dump_json())
    assert first == second
    assert first.fingerprint == second.fingerprint
    assert stable_id("task", {"a": 1}) == stable_id("task", {"a": 1})


def test_configuration_loads_protected_suite_and_rejects_missing_environment(tmp_path: Path) -> None:
    protected = load_suite(Path("evals/suites/routing-smoke.yaml"))
    assert len(protected.tasks) >= 10
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\nname: bad\nversion: '1'\ntasks: []\nvariants:\n- id: x\n  models: {main: '${MISSING_EVAL_MODEL}'}\n", encoding="utf-8")
    with pytest.raises(EvalConfigurationError, match="MISSING_EVAL_MODEL"):
        load_suite(path)


def test_invalid_weights_and_duplicate_ids_fail(repository: Path) -> None:
    with pytest.raises(ValueError):
        ScoreWeights(task_completion=-1)
    with pytest.raises(ValueError, match="duplicate task"):
        EvalSuite(name="x", version="1", tasks=[task(repository), task(repository)], variants=[variant()])


@pytest.mark.parametrize("secret", [
    "sk-abcdefghijklmnopqrstuvwxyz",
    "ghp_abcdefghijklmnopqrstuvwxyz123456",
    "Bearer abc.def.ghi",
    "https://user:password@example.test/path",
    "-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----",
])
def test_redaction_removes_secret_formats(secret: str) -> None:
    assert secret not in redact_text(f"prefix {secret} suffix")
    assert secret not in json.dumps(redact({"authorization": secret, "nested": [secret]}))


def test_pricing_preserves_unknown_and_calculates_known() -> None:
    registry = PricingRegistry([Price("provider", "known", date(2026, 1, 1), Decimal("1"), Decimal("0.5"), Decimal("2"))])
    assert registry.calculate(provider="provider", model="missing", input_tokens=1, cached_input_tokens=0, output_tokens=0, reasoning_tokens=0) is None
    assert registry.calculate(provider="provider", model="known", input_tokens=1_000_000, cached_input_tokens=0, output_tokens=1_000_000, reasoning_tokens=0) == 3.0


def test_score_objective_failure_cannot_be_overridden(repository: Path) -> None:
    run = completed_run("run", "task")
    run = run.model_copy(update={"reviewer_results": [{"score": 1.0}]})
    result = score_run(
        task=task(repository, test_commands=["false"]),
        run=run,
        tests=[EvalTestResult(command="false", passed=False, exit_code=1)],
        weights=ScoreWeights(),
        setup_success=True,
        first_attempt_success=True,
    )
    assert not result.task_success
    assert result.failure_category == "tests"


def test_recorder_context_sequence_and_null_behavior(tmp_path: Path) -> None:
    null = NullEvalRecorder()
    null.record_tool_started({"secret": "sk-abcdefghijklmnopqrstuvwxyz"})
    assert current_eval_context() is None
    recorder = ArtifactEvalRecorder(tmp_path, run_id="run", task_id="task", variant_id="variant")
    context = EvalExecutionContext("run", "task", "variant", recorder)
    with use_eval_context(context):
        record_current("first", {"token": "ghp_abcdefghijklmnopqrstuvwxyz123456"})
        record_current("second", {})
    events = recorder.events()
    assert [item.sequence for item in events] == [1, 2]
    assert "ghp_" not in (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert current_eval_context() is None


def test_common_model_boundary_records_safe_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ChatOpenAI,
        "_generate",
        lambda self, messages, stop=None, run_manager=None, **kwargs: SimpleNamespace(
            llm_output={"token_usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
        ),
    )
    model = CompatibleChatOpenAI(
        api_key="test", model="gpt-4.1-mini",
        compatibility_capabilities=ModelCapabilities(supports_responses_api=False),
    )
    recorder = ArtifactEvalRecorder(tmp_path, run_id="run", task_id="task", variant_id="variant")
    with use_eval_context(EvalExecutionContext("run", "task", "variant", recorder)):
        model._generate([HumanMessage(content="sk-abcdefghijklmnopqrstuvwxyz")])
    event = recorder.events()[0]
    assert event.event_type == "model.call"
    assert event.payload["usage"]["total_tokens"] == 5
    persisted = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in persisted
    assert "prompt_hash" in persisted


def test_storage_indexes_finalizes_and_rejects_mutation(tmp_path: Path) -> None:
    storage = EvalStorage(tmp_path / "evals")
    run = EvalRun(run_id="run", run_fingerprint="fp", experiment_id="exp", task_id="task", variant_id="variant", repository_commit="abc")
    storage.create_run(run, config={}, environment={})
    with pytest.raises(EvalStorageError, match="incomplete"):
        storage.load_run("run")
    done = run.model_copy(update={"status": RunStatus.COMPLETED, "completed_at": datetime.now(timezone.utc), "outcome": EvaluationResult(task_success=True)})
    done = done.model_copy(update={"final_answer": "sk-abcdefghijklmnopqrstuvwxyz"})
    storage.finalize_run(done, {"stdout.log": "sk-abcdefghijklmnopqrstuvwxyz"})
    assert storage.load_run("run").status == RunStatus.COMPLETED
    assert "sk-" not in (storage.run_dir("run") / "stdout.log").read_text(encoding="utf-8")
    assert "sk-" not in (storage.run_dir("run") / "run.json").read_text(encoding="utf-8")
    assert b"sk-abcdefghijklmnopqrstuvwxyz" not in storage.index_path.read_bytes()
    assert storage.list_runs(experiment_id="exp")[0]["run_id"] == "run"
    with pytest.raises(CompletedRunImmutableError):
        storage.finalize_run(done)


def test_incomplete_recovery(tmp_path: Path) -> None:
    storage = EvalStorage(tmp_path / "evals")
    run = EvalRun(run_id="run", run_fingerprint="fp", experiment_id="exp", task_id="task", variant_id="variant", repository_commit="abc")
    storage.create_run(run, config={}, environment={})
    assert storage.recover_incomplete() == ["run"]


def test_local_worktree_is_clean_unique_and_cleanup(repository: Path, tmp_path: Path) -> None:
    backend = LocalWorktreeBackend(tmp_path / "workspaces")
    first = backend.create(repository, "HEAD", run_id="one")
    second = backend.create(repository, "HEAD", run_id="two")
    assert first.path != second.path
    assert git(first.path, "status", "--porcelain") == ""
    backend.cleanup(first)
    backend.cleanup(second)
    assert not first.path.exists() and not second.path.exists()


def test_unsupported_workspace_backend_fails(tmp_path: Path) -> None:
    backend = workspace_backend("docker", tmp_path)
    with pytest.raises(EvalWorkspaceError, match="not implemented"):
        backend.create(tmp_path, "HEAD", run_id="x")


def test_runner_creates_complete_artifacts_without_mutating_source(repository: Path, tmp_path: Path) -> None:
    storage = EvalStorage(tmp_path / "evals")
    runner = EvalRunner(storage=storage, gateway_factory=lambda root, selected: FakeGateway(root, selected))
    run = runner.run(suite=suite(repository), task=task(repository), variant=variant(), trial_number=1, experiment_id="exp")
    assert run.status == RunStatus.COMPLETED
    assert run.outcome and run.outcome.task_success
    assert run.tool_calls[0].tool_name == "read_file"
    assert run.usage[0].calculated_cost is None
    assert (storage.run_dir(run.run_id) / ".complete").exists()
    assert (repository / "sample.txt").read_text(encoding="utf-8") == "before\n"


def test_runner_captures_patch_and_test(repository: Path, tmp_path: Path) -> None:
    storage = EvalStorage(tmp_path / "evals")
    test_command = native_shell_command(
        sys.executable,
        "-c",
        "from pathlib import Path; assert Path('sample.txt').read_text(encoding='utf-8') == 'after\\n'",
    )
    selected_task = task(repository, test_commands=[test_command])
    selected_task = selected_task.model_copy(update={"expected": ExpectedOutcome(route="repository", required_changed_files=["sample.txt"])})
    selected_suite = EvalSuite(name="fixture", version="1", tasks=[selected_task], variants=[variant()])
    runner = EvalRunner(storage=storage, gateway_factory=lambda root, selected: FakeGateway(root, selected, mutate=True))
    run = runner.run(suite=selected_suite, task=selected_task, variant=variant(), trial_number=1, experiment_id="exp")
    assert run.outcome and run.outcome.task_success
    assert run.patches[0].changed_files == ["sample.txt"]
    assert run.tests[0].passed
    assert (repository / "sample.txt").read_text(encoding="utf-8") == "before\n"


def test_matrix_expansion_and_resume_cache(repository: Path, tmp_path: Path) -> None:
    base_suite = suite(repository)
    selected_suite = base_suite.model_copy(update={"defaults": base_suite.defaults.model_copy(update={"trials": 2})})
    selections = expand_matrix(selected_suite)
    assert [item.trial_number for item in selections] == [1, 2]
    storage = EvalStorage(tmp_path / "evals")
    calls = []

    def factory(root: Path, selected: EvalVariant):
        calls.append(root)
        return FakeGateway(root, selected)

    experiment = ExperimentRunner(EvalRunner(storage=storage, gateway_factory=factory), concurrency=2).execute(selected_suite, experiment_id="exp")
    assert len(experiment.runs) == 2
    cached = ExperimentRunner(EvalRunner(storage=storage, gateway_factory=factory), concurrency=2).execute(selected_suite, experiment_id="exp2")
    assert len(cached.runs) == 2
    assert len(calls) == 2


def test_leaderboard_comparison_reports_and_gate(tmp_path: Path) -> None:
    baseline = [completed_run("b1", "one"), completed_run("b2", "two")]
    candidate = [completed_run("c1", "one", score=80, success=False), completed_run("c2", "two")]
    rows = build_leaderboard(candidate, suite_name="suite", suite_version="1")
    assert rows[0]["completed_runs"] == 2
    paths = write_leaderboard(tmp_path / "leaderboard", rows)
    assert all(path.exists() for path in paths.values())
    comparison = compare_runs(baseline, candidate)
    assert comparison["classifications"]["regressed"] == 1
    comparison["tasks"][0]["reasons"] = ["sk-abcdefghijklmnopqrstuvwxyz"]
    gate = evaluate_gate(comparison, GateThresholds(minimum_task_count=2))
    assert not gate.passed
    report_paths = write_regression(tmp_path / "regression", comparison, gate)
    assert "failure" in report_paths["junit"].read_text(encoding="utf-8")
    assert all("sk-abcdefghijklmnopqrstuvwxyz" not in path.read_text(encoding="utf-8") for path in report_paths.values())


def test_gate_fails_unknown_required_cost_and_missing_candidate() -> None:
    comparison = compare_runs([completed_run("b", "one")], [])
    gate = evaluate_gate(comparison, GateThresholds(minimum_task_count=1, require_known_cost=True))
    assert not gate.passed
    assert any("missing" in reason for reason in gate.reasons)
    assert any("unknown" in reason for reason in gate.reasons)


def test_baseline_round_trip_and_rejects_non_reproducible(tmp_path: Path) -> None:
    run = completed_run("run", "task")
    baseline = create_baseline(name="base", suite_name="suite", suite_version="1", variant_fingerprint="fp", runs=[run])
    path = write_baseline(tmp_path / "base.json", baseline)
    assert load_baseline(path) == baseline
    assert baseline_as_runs(baseline)[0].task_id == "task"
    bad = run.model_copy(update={"environment": environment().model_copy(update={"reproducible": False, "non_reproducible_reasons": ["dirty"]})})
    with pytest.raises(EvalStorageError, match="non-reproducible"):
        create_baseline(name="bad", suite_name="suite", suite_version="1", variant_fingerprint="fp", runs=[bad])


def test_inspector_reads_incremental_events(tmp_path: Path) -> None:
    storage = EvalStorage(tmp_path / "evals")
    run = EvalRun(run_id="run", run_fingerprint="fp", experiment_id="exp", task_id="task", variant_id="variant", repository_commit="abc")
    run_dir = storage.create_run(run, config={}, environment={})
    recorder = ArtifactEvalRecorder(run_dir, run_id="run", task_id="task", variant_id="variant")
    recorder.record("route.decision", {"route": "repository"})
    payload = inspect_run(storage, "run", event_type="route.decision")
    assert payload["incomplete"]
    assert len(payload["timeline"]) == 1


def test_trajectory_replay_does_not_call_model(repository: Path, tmp_path: Path) -> None:
    storage = EvalStorage(tmp_path / "evals")
    now = datetime.now(timezone.utc)
    command = subprocess.run("printf stable", cwd=repository, shell=True, capture_output=True, text=True)
    run = EvalRun(
        run_id="run", run_fingerprint="fp", experiment_id="exp", task_id="task", variant_id="variant",
        repository_commit=git(repository, "rev-parse", "HEAD"), started_at=now, completed_at=now,
        status=RunStatus.COMPLETED,
        commands=[{"command": "printf stable", "working_directory": ".", "exit_code": 0, "stdout": command.stdout, "stderr": command.stderr}],
        outcome=EvaluationResult(task_success=True),
    )
    storage.create_run(run.model_copy(update={"status": RunStatus.RUNNING, "completed_at": None}), config={"task": task(repository).model_dump(mode="json")}, environment={})
    storage.finalize_run(run)
    result = trajectory_replay(storage=storage, run_id="run")
    assert not result["diverged"]
    assert not result["model_called"]


def test_cli_help_doctor_and_missing_run_exit_codes(tmp_path: Path) -> None:
    runner = CliRunner()
    assert runner.invoke(app, ["eval", "--help"]).exit_code == 0
    doctor = runner.invoke(app, ["eval", "doctor", "--json", "--eval-root", str(tmp_path / "evals")])
    assert doctor.exit_code == 0
    missing = runner.invoke(app, ["eval", "inspect", "missing", "--eval-root", str(tmp_path / "evals")])
    assert missing.exit_code == 3


def test_cli_list_redacts_secret_identifiers(tmp_path: Path) -> None:
    storage = EvalStorage(tmp_path / "evals")
    run = EvalRun(
        run_id="run", run_fingerprint="fp", experiment_id="exp",
        task_id="sk-abcdefghijklmnopqrstuvwxyz", variant_id="variant", repository_commit="abc",
    )
    storage.create_run(run, config={}, environment={})
    result = CliRunner().invoke(app, ["eval", "list", "--eval-root", str(storage.root)])
    assert result.exit_code == 0
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in result.stdout


def test_normal_runtime_record_call_is_noop() -> None:
    assert current_eval_context() is None
    record_current("gateway.turn.started", {"secret": "value"})
