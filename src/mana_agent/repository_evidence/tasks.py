"""TaskBoard adapters, not a second orchestration loop or task state store."""

from hashlib import sha256
import json
from pathlib import Path

from mana_agent.multi_agent.core.types import TaskBoardItem, TaskStatus
from mana_agent.repository_evidence.models import CodingAssignment, EvidenceQuery, EvidenceRef, EvidenceRequirement


def is_shallow_investigation(task: TaskBoardItem) -> bool:
    return (list(getattr(task, "investigation_scope", [])) == ["."]
            and not getattr(task, "mode_workflow", {}).get("initial_repository_scope"))


def reusable_evidence(board, task, requirement):
    evidence = board.evidence_board(root=Path(task.execution_repo_root or board.store.root), repository_id=task.primary_repository_id)
    if requirement.evidence and evidence.satisfies(requirement):
        return requirement.evidence
    return ()


def ensure_investigation(board, *, parent_task_id, objective, scope, requirements, owner_agent_id,
                         decision_id, task_kind="investigation", supervisor=None):
    """Materialize only a model-selected requirement, deduplicated in the workspace transaction."""
    if task_kind not in {"investigation", "understanding"} or not decision_id:
        raise ValueError("Investigation requires an explicit kind and decision identity")
    parent = board.get_task(parent_task_id)
    requirements = [EvidenceRequirement.model_validate(item) for item in requirements]
    if not requirements or not scope:
        raise ValueError("Investigation tasks need bounded scope and evidence requirements")
    signature = sha256(json.dumps({"objective": objective, "scope": scope, "kind": task_kind,
                                   "requirements": [r.model_dump(mode="json", exclude={"evidence"}) for r in requirements]}, sort_keys=True).encode()).hexdigest()
    with board.transaction():
        board.refresh()
        child = None
        for task in board.list_tasks(parent_task_id=parent.task_id):
            if task.decomposition_local_id == f"evidence:{signature}" and task.status != TaskStatus.CANCELLED:
                if task.status == TaskStatus.DONE:
                    knowledge = board.evidence_board(root=Path(task.execution_repo_root or board.store.root), repository_id=task.primary_repository_id)
                    if not task.evidence_requirements or not all(knowledge.satisfies(EvidenceRequirement.model_validate(r)) for r in task.evidence_requirements):
                        continue
                if supervisor is None or supervisor.store.get_task_or_none(task.task_id) is not None:
                    return task
                child = task
                break
        if supervisor is not None and not supervisor.remaining_child_tasks(parent_task_id):
            raise PermissionError("This workflow has no remaining slots for new investigation tasks. "
                                  "Use saved evidence or select scoped reads in AUTO or DEBUG.")
        if child is not None:
            return child
        child = board.create_child_task(parent_task_id, title=objective[:200], user_request=objective,
            owner_agent_id=owner_agent_id, acceptance_criteria=[criterion for r in requirements for criterion in r.acceptance_criteria],
            decomposition_local_id=f"evidence:{signature}")
        board.update_task(child.task_id, task_kind=task_kind, investigation_scope=scope,
                          evidence_requirements=[r.model_dump(mode="json") for r in requirements])
        board._record("task.evidence_requirement_created", {"task_id": child.task_id, "decision_id": decision_id})
    return child


def assignment_for(board, task_id, *, backend, decision_id, plan_digest="", constraints=()):
    task = board.get_task(task_id)
    if backend not in {"internal", "external"}:
        raise ValueError("Coding assignment requires its configured engine")
    if any(board.get_task(dep).status != TaskStatus.DONE for dep in task.depends_on):
        raise PermissionError("Coding assignment is waiting for its dependencies")
    evidence_board = board.evidence_board(root=Path(task.execution_repo_root or board.store.root), repository_id=task.primary_repository_id)
    requirements = tuple(EvidenceRequirement.model_validate(item) for item in task.evidence_requirements)
    refs = tuple(dict.fromkeys([
        *(EvidenceRef.model_validate(ref) for ref in task.evidence_refs),
        *(ref for requirement in requirements for ref in requirement.evidence),
    ]))
    if task.task_kind in {"investigation", "understanding"}:
        refs = tuple(ref for ref in refs if evidence_board.get(ref).freshness == "current"
                     and evidence_board.get(ref).validation == "supported")
    else:
        for ref in refs:
            record = evidence_board.get(ref)
            if record.freshness != "current" or record.validation != "supported":
                raise PermissionError("Coding assignment contains stale or contradictory repository evidence")
    if task.missing_evidence and task.task_kind not in {"investigation", "understanding"}:
        from mana_agent.repository_evidence.models import MissingEvidenceRequest
        indexed = {requirement.requirement_id: requirement for requirement in requirements}
        for item in task.missing_evidence:
            gap = MissingEvidenceRequest.model_validate(item)
            required = indexed.get(gap.requirement.requirement_id, gap.requirement)
            if not evidence_board.satisfies(required):
                raise PermissionError("The coding task still requires its reported missing repository evidence")
    if task.task_kind not in {"investigation", "understanding"} and any(not evidence_board.satisfies(r) for r in requirements):
        raise PermissionError("Coding assignment has unsatisfied evidence requirements")
    assignment = CodingAssignment(task_id=task_id, task_version=task.row_version + 1,
        objective=task.user_request, task_kind=task.task_kind, backend=backend,
        source_view=str(evidence_board.root), evidence=refs, requirements=requirements,
        layout_only=task.task_kind in {"investigation", "understanding"} and is_shallow_investigation(task),
        allowed_scope=tuple(task.investigation_scope or task.files_to_inspect), dependencies=tuple(task.depends_on),
        constraints=tuple(constraints) + tuple(str(value) for value in (
            task.mode_workflow.get("investigation_context"), task.mode_workflow.get("initial_repository_scope")) if value),
        expected_changes=tuple(task.implementation_targets),
        integration_requirements=tuple(task.integration_verification), verification_criteria=tuple(task.acceptance_criteria),
        plan_digest=plan_digest, decision_id=decision_id)
    board.update_task(task_id, coding_assignment=assignment.model_dump(mode="json"), expected_version=task.row_version)
    return assignment


def validate_assignment(board, task_id, *, expected=None):
    """Reject semantic steering changes without treating progress as new authority."""
    if not hasattr(board, "store"):
        return
    task = board.store.repository.get_task(task_id)
    if not task.coding_assignment:
        if expected:
            raise PermissionError("The admitted coding assignment was revoked")
        return
    assignment = CodingAssignment.model_validate(task.coding_assignment)
    if expected and assignment.model_dump(mode="json") != expected:
        raise PermissionError("The admitted coding assignment was replaced")
    requirements = [EvidenceRequirement.model_validate(item).model_dump(mode="json", exclude={"evidence"})
                    for item in task.evidence_requirements]
    if (assignment.task_id != task_id or assignment.objective != task.user_request
            or assignment.task_kind != task.task_kind
            or assignment.allowed_scope != tuple(task.investigation_scope or task.files_to_inspect)
            or assignment.dependencies != tuple(task.depends_on)
            or [r.model_dump(mode="json", exclude={"evidence"}) for r in assignment.requirements] != requirements
            or any(board.store.repository.get_task(dep).status != TaskStatus.DONE for dep in task.depends_on)):
        raise PermissionError("Task scope, objective, requirements, or dependencies changed; obtain a new coding assignment")


def evidence_context(board, task, *, query=""):
    return board.evidence_board(root=Path(task.execution_repo_root or board.store.root),
        repository_id=task.primary_repository_id).query(EvidenceQuery(
            paths=tuple(task.investigation_scope or task.files_to_inspect), query=query))


def investigation_completion_contract(task):
    from mana_agent.execution_supervisor.models import CompletionContract, CompletionContractType

    return [CompletionContract(contract_type=CompletionContractType.CUSTOM_VERIFIER,
        verifier_name="repository_evidence_requirements", metadata={"task_id": task.task_id, "workspace_id": task.workspace_id})]


def verify_requirements(contract, root, payload):
    """Registered supervisor verifier: read the authoritative task and actual source versions."""
    from mana_agent.persistence.workspace_db import get_workspace_db
    from mana_agent.persistence.workspace_repository import WorkspaceRepository
    from mana_agent.repository_evidence.service import RepositoryEvidenceBoard
    metadata = contract.metadata
    db = get_workspace_db(metadata["workspace_id"])
    task = WorkspaceRepository(metadata["workspace_id"], db=db).get_task(metadata["task_id"])
    evidence = RepositoryEvidenceBoard(db=db, repository_id=task.primary_repository_id, root=root)
    requirements = [EvidenceRequirement.model_validate(r) for r in task.evidence_requirements]
    passed = bool(requirements) and all(evidence.satisfies(r) for r in requirements)
    return passed, {"task_id": task.task_id, "evidence_requirements_satisfied": passed,
                    "evidence": task.evidence_refs, "reason": "" if passed else "Required current evidence is missing"}
