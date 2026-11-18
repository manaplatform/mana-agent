"""Typed missing-evidence handoff shared by both coding engines."""

import json
from typing import Any

from pydantic import ValidationError

from mana_agent.llm.errors import StructuredOutputError, describe_decision_validation
from mana_agent.repository_evidence.models import CodingAssignment, InvestigationResult, MissingEvidenceRequest


def investigation_result_schema(assignment: CodingAssignment) -> dict[str, Any]:
    """Expose the admitted assignment's requirement IDs to both engines."""
    schema = InvestigationResult.model_json_schema()
    requirement_ids = list(dict.fromkeys(requirement.requirement_id for requirement in assignment.requirements))
    finding_ids = schema["$defs"]["InvestigationFinding"]["properties"]["requirement_ids"]
    if requirement_ids:
        finding_ids["items"]["enum"] = requirement_ids
    else:
        finding_ids["maxItems"] = 0
    evidence_bindings = schema["properties"]["evidence_bindings"]
    reference_list = evidence_bindings["additionalProperties"]
    evidence_bindings["properties"] = {requirement_id: reference_list for requirement_id in requirement_ids}
    evidence_bindings["additionalProperties"] = False
    return schema


def json_response_text(text: str) -> str:
    """Accept a bare response or one enclosing Markdown JSON fence."""
    raw = text.strip()
    first, newline, rest = raw.partition("\n")
    if newline and first.strip().casefold() in {"```", "```json"}:
        body, separator, closing = rest.rpartition("\n")
        if separator and closing.strip() == "```":
            return body.strip()
    return raw


def parse_investigation_result(text: str) -> InvestigationResult:
    if len(text) > 250_000:
        raise StructuredOutputError("Investigation result exceeds the response size limit")
    try:
        return InvestigationResult.model_validate_json(json_response_text(text))
    except ValidationError as exc:
        detail = describe_decision_validation(exc, InvestigationResult.model_json_schema())
        message = f"Invalid InvestigationResult ({detail})"
        if any(
            issue["type"] == "literal_error"
            and issue["loc"][:1] == ("findings",)
            and issue["loc"][-1:] == ("classification",)
            for issue in exc.errors(include_input=False, include_context=False, include_url=False)
        ):
            message += " Use 'inferred' or 'hypothetical' for finding classification."
        raise StructuredOutputError(message) from None


def missing_evidence_from_text(text: str) -> list[dict[str, Any]]:
    if not isinstance(text, str) or len(text) > 250_000:
        return []
    try:
        payload = json.loads(json_response_text(text))
    except (ValueError, TypeError):
        return []
    if not isinstance(payload, dict) or "missing_evidence" not in payload:
        return []
    entries = payload["missing_evidence"]
    if not isinstance(entries, list) or len(entries) > 20:
        raise StructuredOutputError("missing_evidence must be an array of at most 20 requests")
    try:
        return [MissingEvidenceRequest.model_validate(item).model_dump(mode="json") for item in entries]
    except ValidationError as exc:
        detail = describe_decision_validation(exc, MissingEvidenceRequest.model_json_schema())
        raise StructuredOutputError(f"Invalid MissingEvidenceRequest ({detail})") from None


MISSING_EVIDENCE_INSTRUCTION = (
    "If required repository knowledge is missing, preserve your completed work and return a JSON object "
    "with missing_evidence containing typed requests: task_id, requirement (requirement_id, objective, "
    "paths, acceptable_classifications, acceptance_criteria, evidence), reason, continuation_reference. "
    "Use your actual assigned task_id. Do not change the objective or broaden permission. "
    "The coordinator may schedule another task and resume this assignment."
)
