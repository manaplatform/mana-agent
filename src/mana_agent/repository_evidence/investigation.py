"""Execute an assigned investigation through the selected coding engine."""

from hashlib import sha256
import json
from pathlib import Path

from mana_agent.repository_evidence.handoff import investigation_result_schema
from mana_agent.repository_evidence.models import (
    EvidenceCandidate, EvidenceProvenance, EvidenceQuery, EvidenceRef, EvidenceSource,
    InvestigationCitation, InvestigationResult,
)
from mana_agent.repository_evidence.tasks import assignment_for, is_shallow_investigation


def _citation_excerpt(
    citation: InvestigationCitation, source: EvidenceSource, content: str,
) -> tuple[str | None, str]:
    """Resolve the citation against a host read and report safe mismatch reasons."""
    if citation.path != source.path or citation.source_kind != source.source_kind:
        return None, "source path or kind differs from the read"
    if citation.sha256 is not None and citation.sha256 != source.sha256:
        return None, "source digest differs from the read"

    excerpt = citation.excerpt.replace("\r\n", "\n").replace("\r", "\n").strip()

    if citation.source_kind == "directory":
        if not excerpt or excerpt in content:
            return citation.excerpt, ""
        try:
            entries = json.loads(content)
        except json.JSONDecodeError:
            return None, "directory manifest is incomplete"
        names = {name for name, _ in entries}
        selected = {part.strip().removeprefix("- ").strip("`\"'")
                    for line in excerpt.splitlines() for part in line.split(",") if part.strip()}
        if selected and selected.issubset(names):
            return citation.excerpt, ""
        return None, "directory excerpt contains unobserved entries"

    # File citation
    if citation.start_line is not None:
        if source.start_line is not None and source.start_line > citation.start_line:
            return None, "line range starts before the read"
        if source.end_line is not None and source.end_line < citation.end_line:
            return None, "line range ends after the read"
        if source.start_line is not None:
            start = citation.start_line - source.start_line
            end = citation.end_line - source.start_line + 1
        else:
            start = citation.start_line - 1
            end = citation.end_line
        lines = content.splitlines(keepends=True)
        if end > len(lines):
            return None, "read content does not cover the line range"
        content = "".join(lines[start:end])

    if not excerpt:
        return content[:12000], ""

    c_norm = content.replace("\r\n", "\n").replace("\r", "\n")
    if excerpt in c_norm:
        return citation.excerpt, ""

    e_lines = [line.strip() for line in excerpt.splitlines() if line.strip()]
    c_lines = [line.strip() for line in c_norm.splitlines() if line.strip()]
    n = len(e_lines)
    for i in range(len(c_lines) - n + 1):
        if c_lines[i:i + n] == e_lines:
            return citation.excerpt, ""
        if n == 1 and e_lines[0] in c_lines[i]:
            return citation.excerpt, ""
        if (n > 1 and e_lines[0] in c_lines[i] and e_lines[-1] in c_lines[i + n - 1]
                and c_lines[i + 1:i + n - 1] == e_lines[1:-1]):
            return citation.excerpt, ""

    return None, "file excerpt does not occur in the selected lines"


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
    source_guidance = (
        'For this shallow layout assignment, cite only the immediate root listing. The only source location is '
        '{"path":".","source_kind":"directory"}. Request missing evidence if you need file content. '
        if assignment.layout_only else
        "For batch_read_files, use the evidence refs attached to individual rows. "
        "When a specific source location is needed, supply its repository-relative path and source_kind. "
        "For files supply start_line and end_line as 1-based inclusive line numbers. "
        "Use the same number for a single line. Prefer an existing evidence reference when line numbers are unavailable. "
        "For a directory use source_kind directory and omit the line fields. "
    )
    prompt = (
        "Execute only this assigned read-only repository task. Investigation and understanding are tasks, "
        "never workflow stages. Reuse the supplied evidence. Inspect only the assigned scope; request "
        "missing evidence if another scope is required. Do not edit files, run verification commands, "
        "or access the network. "
        "Complete the investigation with one InvestigationResult matching the JSON schema. Set each finding's classification "
        "to \"inferred\" for a source-backed conclusion or \"hypothetical\" for an unconfirmed explanation. "
        "The acceptable_classifications in the assignment apply to Evidence Board records. Cite existing host "
        "observations by evidence ID and revision in supporting_evidence or evidence_bindings. "
        "Include architecture_relationships inside each finding it supports. Use one object per link with "
        "source, relation, and target as strings. Omit the field when no link is relevant. "
        "Use repository_evidence_refs from successful reads as supporting_evidence. "
        + source_guidance
        + "The host captures the excerpt and source version from the read. A tool output reference identifies the tool result "
        "and cannot be used as a source hash. On each finding, list the exact requirement_id values from "
        "assignment.requirements that it supports. Use an empty requirement_ids array for an unbound finding. "
        "Use evidence_bindings to link existing evidence references under those same assigned IDs. "
        "Report each limitation as a string in the limitations array. Use [] or omit the field when there are none. "
        "Report remaining evidence gaps explicitly.\n"
        + json.dumps({"assignment": assignment.model_dump(mode="json"), "evidence": context,
                      "result_schema": investigation_result_schema(assignment)}, ensure_ascii=False)
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
    if any(gap.task_id != task_id for gap in report.missing_evidence):
        raise ValueError("Investigation gap belongs to another task")
    if assignment.layout_only and any(
        source.source_kind != "directory" or source.path != "."
        for finding in report.findings for source in finding.sources
    ):
        raise PermissionError("Shallow layout investigations can cite only the root directory. "
                              "Request a scoped investigation for file contents or nested directories.")
    assigned_ids = {requirement.requirement_id for requirement in assignment.requirements}
    binding_fields = (
        ("findings[].requirement_ids",
         (requirement_id for finding in report.findings for requirement_id in finding.requirement_ids)),
        ("evidence_bindings", report.evidence_bindings.keys()),
        ("requirement_bindings", report.requirement_bindings.keys()),
    )
    invalid_fields = [field for field, ids in binding_fields if set(ids) - assigned_ids]
    if invalid_fields:
        raise ValueError("Investigation report referenced an unknown evidence requirement in "
                         + ", ".join(invalid_fields) + ". Use only IDs from assignment.requirements.")
    bindings = {requirement_id: list(indexes) for requirement_id, indexes in report.requirement_bindings.items()}
    if any(not 0 <= index < len(report.findings) for indexes in bindings.values() for index in indexes):
        raise ValueError("Legacy investigation requirement binding has an index outside the findings array")
    for index, finding in enumerate(report.findings):
        for requirement_id in finding.requirement_ids:
            bindings.setdefault(requirement_id, []).append(index)
    from mana_agent.services.activity_context import current_activity_context
    activity = current_activity_context()
    if activity is None or activity.task_id != task_id:
        raise PermissionError("Investigation publication requires its active supervisor attempt")
    current = taskboard.get_task(task_id)
    if (current.user_request != assignment.objective or tuple(current.investigation_scope) != assignment.allowed_scope
            or is_shallow_investigation(current) != assignment.layout_only):
        raise PermissionError("The investigation objective or scope changed during execution")
    provenance = EvidenceProvenance(task_id=task_id, attempt_id=activity.attempt_id,
        attempt_generation=activity.attempt_generation, decision_id=decision_id,
        invocation_id=str(outcome.get("turn_id") or decision_id), agent_id=outcome.get("producing_agent_id") or task.owner_agent_id or "coding_agent", backend=backend,
        result_digest=sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest())
    refs = []
    observations_by_finding = []
    for finding_index, finding in enumerate(report.findings):
        supporting = list(finding.supporting_evidence)
        observed = []
        for ref in supporting:
            record = board.get(ref)
            if record.classification == "observed" and record.freshness == "current" and record.validation == "supported":
                observed.extend(record.sources)
        sources = []
        for source_index, citation in enumerate(finding.sources):
            for captured in observed:
                excerpt, _ = _citation_excerpt(citation, captured, captured.excerpt)
                if excerpt is not None:
                    break
            else:
                # Execute only the exact scoped read selected by the model.
                result = validate_source(citation)
                stamp = result.get("source_evidence")
                if not result.get("ok") or not stamp:
                    raise ValueError("The selected source read returned no complete source version")
                captured = EvidenceSource.model_validate(stamp)
                excerpt, reason = _citation_excerpt(citation, captured, result["content"])
                if excerpt is None:
                    raise ValueError(f"Investigation source citation failed at findings[{finding_index}]."
                                     f"sources[{source_index}] ({reason})")
                source_refs = result.get("repository_evidence_refs", [])
                if not source_refs:
                    raise ValueError("Source validation returned no durable observation identity")
                supporting.extend(EvidenceRef.model_validate(item) for item in source_refs)
                observed.append(captured)
            sources.append(captured.model_copy(update={
                "symbol": citation.symbol or captured.symbol,
                "start_line": citation.start_line if citation.start_line is not None else captured.start_line,
                "end_line": citation.end_line if citation.end_line is not None else captured.end_line,
                "excerpt": excerpt,
            }))
        candidate = EvidenceCandidate(**finding.model_dump(exclude={"sources", "supporting_evidence", "requirement_ids"}),
                                      sources=tuple(sources), supporting_evidence=tuple(dict.fromkeys(supporting)))
        submitted = board.submit_findings(candidate, provenance=provenance)
        refs.append(submitted.ref)
        observations_by_finding.append(tuple(supporting))
    task = taskboard.get_task(task_id)
    requirements = []
    for requirement in assignment.requirements:
        selected = list(report.evidence_bindings.get(requirement.requirement_id, ()))
        for index in bindings.get(requirement.requirement_id, ()):
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
