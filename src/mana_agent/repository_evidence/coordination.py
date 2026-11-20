"""Typed MainAgent decisions consumed by the existing autonomous controller."""

from typing import Literal

from pydantic import Field, field_validator

from mana_agent.repository_evidence.models import Contract, EvidenceQuery, EvidenceRef, EvidenceRequirement, relative_source
from mana_agent.execution_supervisor.models import RecoveryDecision


class RepositoryTaskDraft(Contract):
    local_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    task_kind: Literal["investigation", "understanding", "implementation", "integration", "verification"]
    scope: tuple[str, ...] = Field(min_length=1)
    requirements: tuple[EvidenceRequirement, ...] = ()
    depends_on: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = Field(min_length=1)
    priority: int = Field(default=100, ge=1)
    verification_commands: tuple[str, ...] = ()


    @field_validator("scope")
    @classmethod
    def bounded_scope(cls, values):
        return tuple(relative_source(value) for value in values)


class RepositoryCoordinationDecision(Contract):
    action: Literal["create_tasks", "dispatch", "query_evidence", "use_evidence", "update_task", "wait", "finalize"]
    reason: str = Field(min_length=1)
    tasks: tuple[RepositoryTaskDraft, ...] = ()
    task_id: str = ""
    expected_version: int | None = None
    query: EvidenceQuery | None = None
    evidence: tuple[EvidenceRef, ...] | None = None
    requirements: tuple[EvidenceRequirement, ...] | None = None
    depends_on: tuple[str, ...] | None = None
    priority: int | None = Field(default=None, ge=1)
    recovery: RecoveryDecision | None = None


def coordinate_step(main, root_id, request, route, ctx):
    """One decision/step; continuation, budgets, and terminal authority stay in the loop."""
    board = main.taskboard
    board.refresh()
    root = board.get_task(root_id)
    knowledge = board.evidence_board(root=root.execution_repo_root or main.root, repository_id=root.primary_repository_id)
    query = EvidenceQuery.model_validate(ctx.get("repository_evidence_query") or {"task_id": root_id})
    evidence = knowledge.query(query)
    children = board.list_tasks(root_task_id=root_id)
    saved_tasks = {
        task.task_id: main.execution_supervisor.store.get_task_or_none(task.task_id)
        for task in children[:100]
    }
    llm = main.routing_llm
    if llm is None:
        raise RuntimeError("Repository coordination requires its selected decision model")
    from mana_agent.llm.messages import SystemMessage, UserMessage
    from mana_agent.multi_agent.taskboard.store import serialize
    import json
    if callable(ctx.get("charge_model")):
        ctx["charge_model"]()
    selected = llm.decide_sync([
        SystemMessage("Coordinate the authorized repository objective using TaskBoard and the Repository Evidence Board. "
            "Investigation and understanding are independently scheduled task kinds, never mandatory stages. "
            "Query and reuse sufficient current evidence before creating investigations. Keep unrelated work independent. "
            "Every task needs explicit scope, dependencies and acceptance criteria. Investigation tasks need evidence requirements. "
            "Task drafts depend on local IDs in the same decision or existing durable task IDs. "
            "Dispatch only one ready task per decision; the existing loop owns continuation. "
            "Use update_task to bind evidence requirements or revise dependencies and priorities with expected_version. "
            "Before finalization, explicitly set root-task dependencies to the work required for this goal. "
            "Never replace the objective or broaden authorization. Finalize only when independent verification supports the goal."),
        UserMessage(json.dumps({"objective": request, "route": serialize(route), "evidence": evidence,
            "tasks": [{"task_id": t.task_id, "objective": t.user_request, "task_kind": t.task_kind,
                       "version": t.row_version, "status": t.status.value, "depends_on": t.depends_on,
                       "evidence_requirements": t.evidence_requirements, "evidence_refs": t.evidence_refs,
                       "blockers": t.blockers, "missing_evidence": t.missing_evidence,
                       "checkpoint_id": saved_tasks[t.task_id].checkpoint_id if saved_tasks[t.task_id] else ""}
                      for t in children[:100]]}, default=str))], RepositoryCoordinationDecision)
    decision = RepositoryCoordinationDecision.model_validate(selected)
    from mana_agent.multi_agent.core.types import AgentRole, TaskStatus
    from mana_agent.human_inbox.models import canonical_digest
    decision_id = f"{root_id}:coordination:{canonical_digest(decision.model_dump(mode='json'))}"
    board._record("repository.coordination.decision", {"task_id": root_id, "decision_id": decision_id, "decision": decision.model_dump(mode="json")})
    if decision.action == "query_evidence":
        if decision.query is None:
            raise ValueError("Evidence query decision omitted its query")
        ctx["repository_evidence_query"] = decision.query.model_dump(mode="json")
    elif decision.action == "use_evidence":
        if not decision.evidence:
            raise ValueError("Evidence reuse needs explicit references")
        for ref in decision.evidence:
            record = knowledge.get(ref)
            if record.freshness != "current" or record.validation != "supported":
                raise ValueError("Repository coordination cannot reuse stale or unsupported evidence")
        knowledge.link(root_id, decision.evidence)
        refs = root.evidence_refs + [ref.model_dump(mode="json") for ref in decision.evidence if ref.model_dump(mode="json") not in root.evidence_refs]
        board.update_task(root_id, evidence_refs=refs)
        ctx["repository_evidence_query"] = {"task_id": root_id}
    elif decision.action == "create_tasks":
        if not decision.tasks or len(decision.tasks) > 12:
            raise ValueError("Select between one and twelve task drafts")
        mapping = {t.decomposition_local_id: t.task_id for t in children if t.decomposition_local_id}
        known = {t.task_id for t in children}
        identifiers = [draft.local_id for draft in decision.tasks]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Task draft IDs must be unique")
        graph = {draft.local_id: draft.depends_on for draft in decision.tasks}
        def validate(key, ancestors):
            if key in ancestors:
                raise ValueError("Task draft dependency cycle")
            for dep in graph.get(key, ()):
                if dep not in graph and dep not in known and dep not in mapping:
                    raise ValueError("Unknown task draft dependency")
                if dep in graph:
                    validate(dep, ancestors | {key})
        for key in graph:
            validate(key, set())
        with board.transaction():
            for draft in decision.tasks:
                if draft.local_id in mapping:
                    existing = board.get_task(mapping[draft.local_id])
                    if existing.user_request != draft.objective or existing.task_kind != draft.task_kind:
                        raise ValueError("An existing task ID cannot silently acquire a new objective")
                    continue
                if draft.task_kind in {"investigation", "understanding"}:
                    if not draft.requirements:
                        raise ValueError("Investigation drafts require evidence completion criteria")
                    if all(knowledge.satisfies(r) for r in draft.requirements):
                        raise ValueError("Investigation is unnecessary; select use_evidence")
                child = board.create_child_task(root_id, title=draft.objective[:200], user_request=draft.objective,
                    owner_agent_id=main.registry.find_by_role(AgentRole.CODING).agent_id,
                    decomposition_local_id=draft.local_id, acceptance_criteria=list(draft.acceptance_criteria))
                mapping[draft.local_id] = child.task_id
                child.verification_commands = list(draft.verification_commands)
                board.update_task(child.task_id, task_kind=draft.task_kind, investigation_scope=list(draft.scope),
                    evidence_requirements=[r.model_dump(mode="json") for r in draft.requirements], priority=draft.priority)
            for draft in decision.tasks:
                board.update_task(mapping[draft.local_id], depends_on=[mapping.get(dep, dep) for dep in draft.depends_on])
    elif decision.action == "update_task":
        if decision.task_id not in {task.task_id for task in children} or decision.expected_version is None:
            raise ValueError("Task update requires an owned task and expected version")
        task_ids = {task.task_id for task in children}
        if decision.depends_on is not None and not set(decision.depends_on).issubset(task_ids - {root_id}):
            raise ValueError("Task dependencies must be selected from the admitted graph")
        if decision.evidence is not None:
            for ref in decision.evidence:
                record = knowledge.get(ref)
                if record.freshness != "current" or record.validation != "supported":
                    raise ValueError("Task updates require current and supported evidence")
        with board.transaction():
            board.update_task(decision.task_id, expected_version=decision.expected_version,
                evidence_requirements=[r.model_dump(mode="json") for r in decision.requirements] if decision.requirements is not None else None,
                evidence_refs=[ref.model_dump(mode="json") for ref in decision.evidence] if decision.evidence is not None else None,
                depends_on=list(decision.depends_on) if decision.depends_on is not None else None, priority=decision.priority)
            if decision.evidence:
                knowledge.link(decision.task_id, decision.evidence)
    elif decision.action == "dispatch":
        if decision.task_id == root_id or decision.task_id not in {task.task_id for task in children}:
            raise ValueError("Dispatch must select an existing child task")
        return main._execute_evidence_assignment(root_id, decision.task_id, decision_id, route, recovery=decision.recovery,
            runtime_budget={key: ctx[key] for key in ("charge_model", "charge_tool", "remaining_model_turns", "remaining_tool_calls") if key in ctx})
    elif decision.action == "finalize":
        required_tasks = [t for t in children if t.task_id in root.depends_on]
        if len(required_tasks) != len(root.depends_on) or any(
                t.status != TaskStatus.DONE or not main.execution_supervisor.get_verified_execution_result(t.task_id).is_verified
                for t in required_tasks):
            raise PermissionError("The goal has unresolved required tasks")
        if route.requested_effect == "write":
            implementations = [t for t in required_tasks if t.task_kind in {"implementation", "integration"}]
            if not implementations or not any(t.task_kind == "verification" for t in required_tasks):
                raise PermissionError("The implementation objective needs independent verification of its changes")
            if route.runtime_capability_change and not any(
                    t.integration_verified and t.runtime_reachability_verified for t in implementations):
                raise PermissionError("The feature objective has no verified production integration")
        else:
            requirements = [EvidenceRequirement.model_validate(row) for row in root.evidence_requirements]
            for task in required_tasks:
                requirements.extend(EvidenceRequirement.model_validate(row) for row in task.evidence_requirements)
            if not requirements or not all(knowledge.satisfies(requirement) for requirement in requirements):
                raise PermissionError("The read-only objective needs current supporting evidence")
        return {"answer": decision.reason, "goal_satisfied": True, "verification_passed": True}
    return {"answer": decision.reason, "goal_satisfied": False, "verification_passed": False,
            "waiting_for_user": decision.action == "wait"}
