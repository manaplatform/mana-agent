"""Execute an assigned investigation through the selected coding engine."""

from hashlib import sha256
import json
from pathlib import Path

from mana_agent.repository_evidence.models import EvidenceProvenance, EvidenceQuery, EvidenceRef, EvidenceSource, InvestigationResult
from mana_agent.repository_evidence.tasks import assignment_for


def execute_investigation(*, taskboard, task_id, coding_agent, decision_id, validate_source, runtime=None):
    task = taskboard.get_task(task_id)
    if task.task_kind not in {"investigation", "understanding"}:
        raise ValueError("The assigned task is not an investigation or understanding task")
    backend = coding_agent.active_engine_mode
    assignment = assignment_for(taskboard, task_id, backend=backend, decision_id=decision_id)
    root = Path(assignment.source_view)
    board = taskboard.evidence_board(root=root, repository_id=task.primary_repository_id)
    context = board.query(EvidenceQuery(paths=tuple(path for path in assignment.allowed_scope if path != "."),
                                      task_id=task_id if assignment.allowed_scope == (".",) else ""))
    prompt = (
        "Execute only this assigned read-only repository task. Investigation and understanding are tasks, "
        "never workflow stages. Reuse the supplied evidence. Inspect only the assigned scope; request "
        "missing evidence if another scope is required. Do not edit files, run verification commands, "
        "or access the network. For a dot-only scope without an explicit audit, list immediate entries only. "
        "Return exactly an InvestigationResult JSON object matching the schema. Findings are inferences or "
        "hypotheses, never verified observations. Cite existing evidence IDs with revisions, or name exact "
        "source paths, SHA-256 hashes, symbols, ranges, and source excerpts. Bind each satisfied requirement "
        "to the zero-based indexes of its supporting findings, or use evidence_bindings for existing "
        "evidence references. Report remaining gaps explicitly.\n"
        + json.dumps({"assignment": assignment.model_dump(mode="json"), "evidence": context,
                      "result_schema": InvestigationResult.model_json_schema()}, ensure_ascii=False)
    )
    if taskboard.get_task(task_id).row_version != assignment.task_version:
        raise PermissionError("The investigation assignment changed before dispatch")
    worker = coding_agent.for_assignment(root=root)
    try:
        outcome = worker.generate(prompt, execution_mode="read_only", gateway_task_id=task_id,
            dispatch_source="repository_evidence_task", repository_investigation=assignment.model_dump(mode="json"),
            coding_original_goal=task.user_request, execution_policy=task.execution_policy,
            execution_runtime=runtime or {})
    finally:
        worker.close()
    if outcome.get("status") not in {"completed", "success", "waiting"}:
        raise RuntimeError("The selected investigation backend did not complete: " + str(outcome.get("errors") or outcome.get("error") or outcome.get("status")))
    raw = outcome.get("investigation_result") or (outcome.get("result") or {}).get("investigation_result")
    if raw is None:
        raise ValueError("The selected engine returned no typed investigation result")
    report = InvestigationResult.model_validate(raw)
    from mana_agent.services.activity_context import current_activity_context
    activity = current_activity_context()
    if activity is None or activity.task_id != task_id:
        raise PermissionError("Investigation publication requires its active supervisor attempt")
    current = taskboard.get_task(task_id)
    if current.user_request != assignment.objective or tuple(current.investigation_scope) != assignment.allowed_scope:
        raise PermissionError("The investigation objective or scope changed during execution")
    provenance = EvidenceProvenance(task_id=task_id, attempt_id=activity.attempt_id,
        attempt_generation=activity.attempt_generation, decision_id=decision_id,
        invocation_id=str(outcome.get("turn_id") or decision_id), agent_id=outcome.get("producing_agent_id") or task.owner_agent_id or "coding_agent", backend=backend,
        result_digest=sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest())
    refs = []
    observations_by_finding = []
    for finding in report.findings:
        supporting = list(finding.supporting_evidence)
        for source in finding.sources:
            covered = [board.get(ref) for ref in supporting]
            if any(record.freshness == "current" and record.validation == "supported" and any(
                    saved.path == source.path and saved.sha256 == source.sha256
                    and (saved.start_line is None or (source.start_line is not None and saved.start_line <= source.start_line
                         and saved.end_line >= source.end_line)) for saved in record.sources) for record in covered):
                continue
            # The selected model supplied the exact source. This validates its
            # bytes through the registered boundary; it does not select a search.
            result = validate_source(source)
            captured = EvidenceSource.model_validate(result["source_evidence"])
            if captured.sha256 != source.sha256 or (source.excerpt and source.excerpt not in result["content"]):
                raise ValueError("Investigation source citation does not match the observed source")
            source_refs = result.get("repository_evidence_refs", [])
            if not source_refs:
                raise ValueError("Source validation returned no durable observation identity")
            supporting.extend(EvidenceRef.model_validate(item) for item in source_refs)
        submitted = board.submit_findings(finding.model_copy(update={"supporting_evidence": tuple(dict.fromkeys(supporting))}), provenance=provenance)
        refs.append(submitted.ref)
        observations_by_finding.append(tuple(supporting))
    task = taskboard.get_task(task_id)
    requirements = []
    for requirement in assignment.requirements:
        indexes = report.requirement_bindings.get(requirement.requirement_id, ())
        if any(index < 0 or index >= len(refs) for index in indexes):
            raise ValueError("Investigation requirement references an unknown finding")
        selected = list(report.evidence_bindings.get(requirement.requirement_id, ()))
        for index in indexes:
            candidates = [refs[index], *observations_by_finding[index]]
            selected.extend(ref for ref in candidates if board.get(ref).classification in requirement.acceptable_classifications
                            and (not requirement.categories or board.get(ref).category in requirement.categories))
        bound = requirement.model_copy(update={"evidence": tuple(dict.fromkeys(selected)) or requirement.evidence})
        requirements.append(bound)
    all_refs = tuple(dict.fromkeys([*(EvidenceRef.model_validate(item) for item in task.evidence_refs), *refs, *report.evidence,
                                   *(ref for requirement in requirements for ref in requirement.evidence)]))
    with activity.publication():
        taskboard.update_task(task_id, evidence_requirements=[r.model_dump(mode="json") for r in requirements],
            evidence_refs=[ref.model_dump(mode="json") for ref in all_refs],
            missing_evidence=[item.model_dump(mode="json") for item in report.missing_evidence])
        board.link(task_id, all_refs, role="supports")
    satisfied = bool(requirements) and all(board.satisfies(requirement) for requirement in requirements)
    return {"goal_satisfied": satisfied and not report.missing_evidence, "evidence_refs": [r.model_dump(mode="json") for r in refs],
            "limitations": list(report.limitations), "missing_evidence": [item.model_dump(mode="json") for item in report.missing_evidence],
            "summary": "\n".join(f.statement for f in report.findings)}
