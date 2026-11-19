"""Investigation task identity, independent readiness, and honest completion."""

from types import SimpleNamespace

import pytest

from mana_agent.multi_agent.core.types import QueueJob, QueueJobStatus, QueueJobType, TaskStatus
from mana_agent.multi_agent.queue.scheduler import next_job
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.repository_evidence.models import EvidenceRequirement
from mana_agent.repository_evidence.tasks import ensure_investigation


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    repo = tmp_path / "repo"
    repo.mkdir()
    return TaskBoard(repo)


@pytest.mark.parametrize("kind", ["investigation", "understanding"])
def test_empty_knowledge_creates_a_deduplicated_real_task(board, kind):
    root = board.create_task(title="Feature", user_request="Implement feature")
    requirement = EvidenceRequirement(requirement_id="behavior", objective="Understand handler",
        acceptance_criteria=("Cite handler evidence",))
    kwargs = dict(parent_task_id=root.task_id, objective="Understand handler", scope=["src"],
                  requirements=[requirement], owner_agent_id="coding", decision_id="model-decision", task_kind=kind)
    first = ensure_investigation(board, **kwargs)
    second = ensure_investigation(board, **kwargs)
    assert first.task_id == second.task_id
    assert first.parent_task_id == root.task_id
    assert first.task_kind == kind
    assert first.evidence_requirements
    assert root.mode_workflow == {}
    assert len(board.list_tasks(parent_task_id=root.task_id)) == 1


def test_child_limit_allows_reselection_of_an_admitted_investigation(board, tmp_path):
    from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
    from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor

    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig(root=tmp_path / "execution", max_children_per_task=1))
    root = board.create_task(title="Goal", user_request="Inspect the handler")
    supervisor.create_task(task_id=root.task_id)
    requirement = EvidenceRequirement(requirement_id="behavior", objective="Inspect handler",
                                      acceptance_criteria=("Cite handler evidence",))
    kwargs = dict(parent_task_id=root.task_id, objective=requirement.objective, scope=["src"],
                  requirements=[requirement], owner_agent_id="coding", decision_id="selected",
                  supervisor=supervisor)
    first = ensure_investigation(board, **kwargs)
    supervisor.create_task(task_id=first.task_id, parent_task_id=root.task_id)

    assert supervisor.remaining_child_tasks(root.task_id) == 0
    assert ensure_investigation(board, **kwargs).task_id == first.task_id
    with pytest.raises(PermissionError, match="no remaining slots"):
        ensure_investigation(board, **{**kwargs, "objective": "Inspect another component"})
    assert [child.task_id for child in board.list_tasks(parent_task_id=root.task_id)] == [first.task_id]


def job(identifier, *, dependencies=(), worker="runner"):
    return QueueJob(job_id=identifier, task_id="task", requested_by_agent_id="coding", job_type=QueueJobType.REPO_READ,
        payload={"path": "src/app.py"}, assigned_worker_agent_id=worker, depends_on=list(dependencies))


def test_missing_evidence_blocks_only_dependent_operations():
    investigation = job("investigate")
    investigation.status = QueueJobStatus.RUNNING
    dependent = job("dependent", dependencies=("investigate",))
    independent = job("independent")
    assert next_job([investigation, dependent, independent]).job_id == "independent"
    investigation.status = QueueJobStatus.DONE
    assert next_job([investigation, dependent]).job_id == "dependent"


def test_worker_filter_retains_completed_cross_worker_dependencies():
    source = job("source", worker="first")
    source.status = QueueJobStatus.DONE
    dependent = job("dependent", dependencies=("source",), worker="second")
    assert next_job([source, dependent], worker_id="second").job_id == "dependent"
    assert next_job([source, dependent], worker_id="first") is None


def test_taskboard_rejects_dependency_cycles_without_marking_completion(board):
    root = board.create_task(title="Goal", user_request="Goal")
    first = board.create_child_task(root.task_id, title="First")
    second = board.create_child_task(root.task_id, title="Second", depends_on=[first.task_id])
    with pytest.raises(ValueError, match="cycle"):
        board.update_task(first.task_id, depends_on=[second.task_id])
    assert first.status == TaskStatus.NEW


def test_checked_off_task_cannot_be_completed_without_supervisor_evidence(board):
    task = board.create_task(title="Implement", user_request="Implement")
    board.update_status(task.task_id, TaskStatus.IN_PROGRESS)
    board.mark_crossed_off(task.task_id)
    with pytest.raises(Exception, match="supervisor"):
        board.mark_completed(task.task_id, result_summary="Agent says done")
    assert board.get_task(task.task_id).status != TaskStatus.DONE
    with pytest.raises(Exception, match="supervisor"):
        board.verify_task(task.task_id, passed=True, commands=["pytest"], summary="Planned check")


def test_saving_one_task_preserves_another_boards_update(board):
    first = board.create_task(title="First", user_request="First")
    second = board.create_task(title="Second", user_request="Second")
    other = TaskBoard(board.store.root)
    other.update_task(second.task_id, progress="Observed progress")
    board.update_task(first.task_id, progress="Independent progress")
    assert board.store.repository.get_task(second.task_id).aggregate_progress == "Observed progress"


def test_queue_admission_and_claims_survive_restart_and_reject_stale_writers(board):
    from mana_agent.repository_evidence.service import EvidenceConflict
    from mana_agent.repository_evidence.tool_queue import ToolQueueStore
    store = ToolQueueStore(board.store.db)
    first = job("first")
    first.fingerprint = "read-source"
    store.admit(first)
    duplicate = job("duplicate")
    duplicate.fingerprint = first.fingerprint
    assert store.admit(duplicate).job_id == first.job_id
    restarted = ToolQueueStore(board.store.db)
    with pytest.raises(PermissionError, match="another worker"):
        restarted.claim(first.job_id, "wrong-worker")
    claimed = restarted.claim(first.job_id, "runner")
    assert claimed.status == QueueJobStatus.CLAIMED
    assert restarted.claim(first.job_id, "runner") is None
    with pytest.raises(EvidenceConflict):
        store.save(first)


def test_missing_queue_dependencies_cannot_be_claimed_after_restart(board):
    from mana_agent.repository_evidence.tool_queue import ToolQueueStore
    store = ToolQueueStore(board.store.db)
    dependent = job("dependent", dependencies=("missing",))
    dependent.fingerprint = "dependent-read"
    store.admit(dependent)
    assert ToolQueueStore(board.store.db).claim(dependent.job_id, "runner") is None


def test_assignment_rejects_changed_objective_but_accepts_progress_updates(board):
    from mana_agent.repository_evidence.tasks import assignment_for, validate_assignment
    task = board.create_task(title="Implement", user_request="Implement the approved change")
    board.update_task(task.task_id, task_kind="implementation", investigation_scope=["src"])
    assignment = assignment_for(board, task.task_id, backend="internal", decision_id="approved-decision")
    board.update_task(task.task_id, progress="Inspected a required source")
    validate_assignment(board, task.task_id, expected=assignment.model_dump(mode="json"))
    board.update_task(task.task_id, description="A different objective")
    with pytest.raises(PermissionError, match="scope, objective"):
        validate_assignment(board, task.task_id, expected=assignment.model_dump(mode="json"))


def test_failed_task_retains_evidence_requirements_across_taskboard_reload(board):
    root = board.create_task(title="Goal", user_request="Goal")
    requirement = EvidenceRequirement(requirement_id="behavior", objective="Inspect handler",
        acceptance_criteria=("Cite handler",))
    child = ensure_investigation(board, parent_task_id=root.task_id, objective="Inspect handler", scope=["src"],
        requirements=[requirement], owner_agent_id="coding", decision_id="selected")
    board.update_status(child.task_id, TaskStatus.IN_PROGRESS)
    board.update_status(child.task_id, TaskStatus.BLOCKED, reason="A required dependency is unavailable")
    restored = TaskBoard(board.store.root).get_task(child.task_id)
    assert restored.task_id == child.task_id
    assert restored.task_kind == "investigation"
    assert restored.evidence_requirements == child.evidence_requirements
    assert restored.parent_task_id == root.task_id
    reused = ensure_investigation(board, parent_task_id=root.task_id, objective="Inspect handler", scope=["src"],
        requirements=[requirement], owner_agent_id="coding", decision_id="retry-selected")
    assert reused.task_id == child.task_id


def test_equivalent_investigations_for_different_parents_keep_separate_execution_ids(board):
    root = board.create_task(title="Goal", user_request="Independent changes")
    first = board.create_child_task(root.task_id, title="First", user_request="Implement the first change")
    second = board.create_child_task(root.task_id, title="Second", user_request="Implement the second change")
    requirement = EvidenceRequirement(requirement_id="behavior", objective="Inspect handler",
        acceptance_criteria=("Cite handler",))
    children = [ensure_investigation(board, parent_task_id=parent.task_id, objective="Inspect handler", scope=["src"],
        requirements=[requirement], owner_agent_id="coding", decision_id="selected") for parent in (first, second)]
    assert children[0].task_id != children[1].task_id
    assert [child.parent_task_id for child in children] == [first.task_id, second.task_id]


@pytest.mark.parametrize("case", ["paths", "scope", "checkpoint", "other_parent", "resume", "retry"])
def test_investigation_recovery_checks_saved_assignment_at_dispatch(board, monkeypatch, case):
    from mana_agent.agent.plan_service import InvestigateDecision
    from mana_agent.config.autonomous_config import AutonomousExecutionConfig
    from mana_agent.execution_supervisor.models import ExecutionState, RecoveryAction, RecoveryDecision
    from mana_agent.gateway.mode_workflow import ModeWorkflow
    from mana_agent.repository_evidence.models import EvidenceProvenance, EvidenceQuery, EvidenceSource
    from mana_agent.repository_evidence.observations import read_source
    from mana_agent.services.activity_context import ActivityContext

    root = board.create_task(title="Goal", user_request="Investigate handler")
    requirement = EvidenceRequirement(requirement_id="behavior", objective="Inspect handler",
        acceptance_criteria=("Cite handler",))
    child = ensure_investigation(board, parent_task_id=root.task_id, objective=requirement.objective, scope=["src"],
        requirements=[requirement], owner_agent_id="coding", decision_id="selected")
    repo = board.store.root
    (repo / "src").mkdir()
    knowledge = board.evidence_board(root=repo)
    observations = []
    for name in ("handler.py", "stale.py"):
        path = f"src/{name}"
        (repo / path).write_text("observed = True\n")
        read = read_source(repo, path)
        source = EvidenceSource.model_validate(read["source_evidence"])
        observations.append(knowledge.record_observation(
            provenance=EvidenceProvenance(task_id=child.task_id, attempt_id="saved-attempt", attempt_generation=1,
                decision_id="selected", invocation_id=name, agent_id="coding", backend="internal",
                result_digest=source.sha256),
            category="code_behavior", statement="The source sets observed", observation=read, sources=(source,),
        ))
    (repo / "src/stale.py").write_text("observed = False\n")
    current, stale = observations
    board.update_task(child.task_id, evidence_refs=[record.ref.model_dump(mode="json") for record in observations])
    child = board.get_task(child.task_id)
    child.mode_workflow["investigation_context"] = "Inspect the handler"
    board.save()
    restored = TaskBoard(repo)
    durable = SimpleNamespace(state=ExecutionState.WAITING if case == "resume" else ExecutionState.FAILED,
                              checkpoint_id="checkpoint-saved")
    retired = []
    lane = SimpleNamespace(taskboard=restored,
        execution_supervisor=SimpleNamespace(store=SimpleNamespace(get_task_or_none=lambda _: durable)),
        ensure_recoverable_execution=lambda task_id: restored.get_task(task_id),
        cancel_task=lambda task_id, **kwargs: retired.append((task_id, kwargs["source"])),
        resume_checkpoint=lambda *args, **kwargs: pytest.fail("Recovery restarted the investigation backend"),
        retry_task=lambda *args, **kwargs: pytest.fail("Recovery restarted the investigation backend"))
    workflow = ModeWorkflow(SimpleNamespace(_lane_coordinator=lane,
        _autonomous_controller=SimpleNamespace(config=AutonomousExecutionConfig()),
        _coding_agent=SimpleNamespace(for_assignment=lambda **_: pytest.fail("Requested unauthorized dispatch"))))
    activity = ActivityContext(conversation_id="fixture", execution_id="turn", task_id=root.task_id, attempt_generation=2)
    monkeypatch.setattr("mana_agent.gateway.mode_workflow.current_activity_context", lambda: activity)
    monkeypatch.setattr(workflow, "_workspace_root", lambda _: repo)
    monkeypatch.setattr(workflow, "_check_live", lambda task_id: restored.get_task(task_id))
    recovery = RecoveryDecision(decision_id="recover", task_id=child.task_id,
        action=RecoveryAction.RETRY if case == "retry" else RecoveryAction.RESUME_CHECKPOINT,
        resume_checkpoint_id={"retry": "", "checkpoint": "checkpoint-old"}.get(case, "checkpoint-saved"),
        same_task_retry_authorized=case == "retry", reason="Use saved evidence", safe_to_continue=True)
    decision = InvestigateDecision(action="investigate", reason="Recover the saved task", objective=requirement.objective,
        scope="Change the saved question" if case == "scope" else "Inspect the handler",
        paths=("other",) if case == "paths" else ("src",), recovery=recovery)

    parent_id = (restored.create_task(title="Other task", user_request="Read different code").task_id
                 if case == "other_parent" else root.task_id)
    state = {"mode": "auto", "phase": "active", "revision": 1, "decision_count": 1, "evidence": []}
    query = EvidenceQuery(task_id=parent_id, include_stale=True)
    if case in {"resume", "retry"}:
        workflow._investigate(parent_id, state, None, decision)
        assert retired == [(child.task_id, "coding_evidence_migration")]
        saved = TaskBoard(repo).get_task(parent_id)
        assert saved.evidence_refs == [current.ref.model_dump(mode="json")]
        assert saved.mode_workflow["investigation"]["report"]["evidence_ids"] == [current.evidence_id]
        assert saved.mode_workflow["evidence_accounting"]["reused_evidence"] == 1
        assert saved.mode_workflow.get("tool_operations", 0) == 0
        assert [record["evidence_id"] for record in knowledge.query(query)["records"]] == [current.evidence_id]
        assert knowledge.get(current.ref).provenance.task_id == child.task_id
        assert knowledge.get(stale.ref).freshness == "stale"
    else:
        expected = {"checkpoint": r"exact nonempty saved_investigation\.checkpoint_id",
                    "other_parent": "No saved child investigation belongs to this Coding task"}
        with pytest.raises(PermissionError, match=expected.get(case, "match the saved child, objective, and scope")):
            workflow._investigate(parent_id, state, None, decision)
        assert not retired
        assert not knowledge.query(query)["records"]
        assert not restored.get_task(parent_id).evidence_refs

    assert len(restored.list_tasks(parent_task_id=root.task_id)) == 1
    assert restored.get_task(child.task_id).investigation_scope == ["src"]
    assert restored.get_task(child.task_id).mode_workflow["investigation_context"] == "Inspect the handler"


def test_executor_adopts_new_tasks_without_an_investigation_barrier(board, complete_supervised_task):
    from threading import Event
    from mana_agent.multi_agent.runtime.multi_task_orchestrator import MultiTaskChildResult, MultiTaskItem, MultiTaskOrchestrator, MultiTaskPlan
    root = board.create_task(title="Goal", user_request="Implement while investigating an independent question")
    requirement = EvidenceRequirement(requirement_id="independent", objective="Inspect independent behavior",
        acceptance_criteria=("Cite independent evidence",))
    plan = MultiTaskPlan(goal=root.user_request, reason="Independent work", final_acceptance_criteria=["Implement"], tasks=[
        MultiTaskItem(local_id="code", title="Code", request="Implement", acceptance_criteria=["Change implemented"], reason="Ready"),
        MultiTaskItem(local_id="inspect", title="Inspect", request="Independent question", task_kind="investigation",
            investigation_scope=["src"], evidence_requirements=[requirement.model_dump(mode="json")],
            acceptance_criteria=["Cite independent evidence"], reason="Independent")])
    investigation_started, late_finished = Event(), Event()
    executor = MultiTaskOrchestrator(llm=None, taskboard=board, maximum_concurrency=2)
    def execute(item, task_id):
        if item.local_id == "inspect":
            investigation_started.set()
            assert late_finished.wait(3), "New ready work should execute while this investigation is active"
            board.update_status(task_id, TaskStatus.BLOCKED, reason="Independent evidence remains unavailable")
            return MultiTaskChildResult(item.local_id, task_id, item.title, "repository", "blocked")
        if item.local_id == "code":
            assert investigation_started.wait(3)
            board.create_child_task(root.task_id, title="Late ready work", user_request="Verify implementation",
                decomposition_local_id="late", acceptance_criteria=["Return observable result"])
        complete_supervised_task(board, task_id)
        if item.local_id == "late":
            late_finished.set()
        return MultiTaskChildResult(item.local_id, task_id, item.title, "repository", "completed")
    results = {result.local_id: result for result in executor.execute(root_task_id=root.task_id, plan=plan, execute_child=execute)}
    assert results["code"].status == results["late"].status == "completed"
    assert results["inspect"].status == "blocked"


def test_executor_rejects_worker_completion_without_durable_verification(board):
    from mana_agent.multi_agent.runtime.multi_task_orchestrator import MultiTaskChildResult, MultiTaskItem, MultiTaskOrchestrator, MultiTaskPlan
    root = board.create_task(title="Goal", user_request="Implement")
    plan = MultiTaskPlan(goal="Implement", reason="Selected work", final_acceptance_criteria=["Verified"], tasks=[
        MultiTaskItem(local_id=key, title=key, request=key, acceptance_criteria=["Verified"], reason="Selected") for key in ("a", "b")])
    results = MultiTaskOrchestrator(llm=None, taskboard=board).execute(root_task_id=root.task_id, plan=plan,
        execute_child=lambda item, task_id: MultiTaskChildResult(item.local_id, task_id, item.title, "repository", "completed"))
    assert all(result.status == "blocked" for result in results)
    assert all(board.get_task(result.task_id).status != TaskStatus.DONE for result in results)


def test_controller_lane_projection_reuses_the_existing_attempt(board, tmp_path):
    from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
    from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
    from mana_agent.gateway.lane_coordinator import LaneCoordinator
    from mana_agent.gateway.lanes import LaneId
    from mana_agent.services.activity_context import ActivityContext, activity_scope
    root = board.create_task(title="Goal", user_request="Investigate the selected scope")
    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig(root=tmp_path / "supervisor", startup_recovery=False))
    supervisor.create_task(task_id=root.task_id, routing_decision_id="selected", workspace_path=board.store.root)
    supervisor.queue(root.task_id)
    task, lease = supervisor.acquire_lease(root.task_id, owner="main")
    supervisor.start(root.task_id, attempt_id=task.attempt_id, lease_token=lease)
    coordinator = LaneCoordinator(board.store.root, taskboard=board, execution_supervisor=supervisor)
    context = ActivityContext(conversation_id="chat", execution_id="turn", task_id=root.task_id,
        attempt_id=task.attempt_id, attempt_generation=task.attempt_generation,
        lease_token=lease, supervisor=supervisor)
    with activity_scope(context):
        first = coordinator.project_active_controller(root.task_id, lane_id=LaneId.RESEARCH)
        again = coordinator.project_active_controller(root.task_id, lane_id=LaneId.RESEARCH)
        assert first is again
        assert first.supervisor_attempt_id == task.attempt_id
        assert first.workflow_coordinator
    assert supervisor.get_task(root.task_id).attempt_id == task.attempt_id
