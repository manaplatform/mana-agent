"""Versioned data contracts. No model-supplied field grants execution authority."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema
from pydantic_core import PydanticCustomError


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def relative_source(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or path.as_posix() != value or ".." in path.parts
            or "\\" in value or ":" in value or any(ord(c) < 32 for c in value)):
        raise ValueError("Evidence paths must be canonical repository-relative paths")
    if any(part.startswith(".env") or part in {
        ".git", ".ssh", ".mana", "secrets.toml", "credentials.json", "id_rsa", "id_ed25519",
    } for part in path.parts):
        raise ValueError("Protected sources cannot enter repository evidence")
    return value


class EvidenceRef(Contract):
    evidence_id: str = Field(min_length=1)
    revision: int = Field(ge=1)


class SourceLocation(Contract):
    path: str
    source_kind: Literal["file", "directory"] = "file"
    symbol: str = ""
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    excerpt: str = Field(default="", max_length=12000)
    revision: str = ""

    _path = field_validator("path")(relative_source)

    @model_validator(mode="after")
    def line_range(self):
        if (self.start_line is None) != (self.end_line is None):
            raise PydanticCustomError("source_citation_boundaries", "Both source line boundaries are required")
        if self.end_line is not None and self.end_line < self.start_line:
            raise PydanticCustomError("source_citation_reversed_range", "Source line range is reversed")
        return self


class EvidenceSource(SourceLocation):
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class InvestigationCitation(SourceLocation):
    """Model-selected location, resolved to a versioned source by the host."""

    sha256: SkipJsonSchema[str | None] = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    revision: SkipJsonSchema[str] = ""
    excerpt: SkipJsonSchema[str] = Field(default="", max_length=12000)

    @model_validator(mode="before")
    @classmethod
    def complete_single_line(cls, value: Any) -> Any:
        if not isinstance(value, dict) or value.get("source_kind", "file") != "file":
            return value
        start, end = value.get("start_line"), value.get("end_line")
        if (start is None) != (end is None):
            line = start if start is not None else end
            return {**value, "start_line": line, "end_line": line}
        return value

    @model_validator(mode="after")
    def identify_source(self):
        if self.source_kind == "directory" and self.start_line is not None:
            raise PydanticCustomError("source_citation_directory_range", "Directory citations cannot specify line ranges")
        if self.source_kind == "file" and self.start_line is None and self.sha256 is None and not self.excerpt.strip():
            raise PydanticCustomError("source_citation_file_range", "File citations require an exact line range")
        return self


class EvidenceRelation(Contract):
    relation: Literal["supports", "contradicts", "depends_on", "derived_from", "supersedes"]
    target: EvidenceRef


class ArchitectureRelationship(Contract):
    """One directed relationship between named repository components."""

    source: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    target: str = Field(min_length=1)


class Finding(Contract):
    category: str = Field(min_length=1, max_length=100)
    statement: str = Field(min_length=1, max_length=4000)
    classification: Literal["inferred", "hypothetical"] = Field(
        description="Use inferred for a source-backed conclusion or hypothetical for an unconfirmed explanation.",
    )
    supporting_evidence: tuple[EvidenceRef, ...] = Field(default=(), max_length=32)
    relationships: tuple[EvidenceRelation, ...] = ()
    dependencies: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    confidence: float = Field(ge=0, le=1)


class EvidenceCandidate(Finding):
    sources: tuple[EvidenceSource, ...] = Field(default=(), max_length=32)
    architecture_relationships: tuple[dict[str, str], ...] = ()


class InvestigationFinding(Finding):
    sources: tuple[InvestigationCitation, ...] = Field(default=(), max_length=32)
    requirement_ids: tuple[str, ...] = Field(
        default=(), description="IDs from the assigned requirements that this finding supports.",
    )
    # Earlier reports used arbitrary string maps. Keep decoding those without
    # advertising their shape to either coding engine.
    architecture_relationships: tuple[ArchitectureRelationship | SkipJsonSchema[dict[str, str]], ...] = Field(
        default=(), description="Architecture links described by this finding.",
    )

    @field_validator("classification", mode="before")
    @classmethod
    def normalize_classification(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip().casefold()
            # Only the host can record an observation. A model that reports one
            # is submitting a conclusion subject to the usual evidence checks.
            if value == "observed":
                return "inferred"
        return value


class EvidenceProvenance(Contract):
    task_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    attempt_generation: int = Field(ge=1)
    decision_id: str = Field(min_length=1)
    invocation_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    result_digest: str = Field(min_length=1)
    result_reference: str = ""


class EvidenceRecord(Contract):
    schema_version: Literal[1] = 1
    evidence_id: str
    revision: int = Field(ge=1)
    workspace_id: str
    repository_id: str
    source_view: str
    category: str
    statement: str
    classification: Literal["observed", "inferred", "hypothetical"]
    validation: Literal["pending", "supported", "contradicted", "rejected"]
    freshness: Literal["current", "stale", "invalidated", "unknown"]
    freshness_reason: str = ""
    confidence: float = Field(ge=0, le=1)
    sources: tuple[EvidenceSource, ...] = Field(default=(), max_length=32)
    supporting_evidence: tuple[EvidenceRef, ...] = Field(default=(), max_length=32)
    relationships: tuple[EvidenceRelation, ...] = ()
    dependencies: tuple[str, ...] = ()
    architecture_relationships: tuple[dict[str, str], ...] = ()
    limitations: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    observation: dict[str, Any] = Field(default_factory=dict)
    provenance: EvidenceProvenance
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def ref(self) -> EvidenceRef:
        return EvidenceRef(evidence_id=self.evidence_id, revision=self.revision)


class EvidenceQuery(Contract):
    task_id: str = ""
    paths: tuple[str, ...] = Field(default=(), max_length=32)
    symbols: tuple[str, ...] = Field(default=(), max_length=32)
    categories: tuple[str, ...] = Field(default=(), max_length=32)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=100)
    related_to: tuple[EvidenceRef, ...] = Field(default=(), max_length=32)
    dependencies: tuple[str, ...] = Field(default=(), max_length=32)
    query: str = Field(default="", max_length=2000)
    include_stale: bool = False
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    max_context_tokens: int = Field(default=4000, ge=1, le=4000)

    @field_validator("paths")
    @classmethod
    def paths_are_relative(cls, values):
        return tuple(relative_source(value) for value in values)


class EvidenceDetailQuery(EvidenceRef):
    offset: int = Field(default=0, ge=0)
    max_bytes: int = Field(default=4000, ge=256, le=4000)


class EvidenceRequirement(Contract):
    requirement_id: str = Field(min_length=1, max_length=100)
    objective: str = Field(min_length=1, max_length=2000)
    paths: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    acceptable_classifications: tuple[Literal["observed", "inferred"], ...] = ("observed",)
    required_tools: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = Field(min_length=1)
    evidence: tuple[EvidenceRef, ...] = Field(default=(), max_length=100)

    @field_validator("paths")
    @classmethod
    def scope_paths(cls, values):
        return tuple(relative_source(value) for value in values)


class CodingAssignment(Contract):
    task_id: str
    task_version: int = Field(ge=0)
    objective: str = Field(min_length=1)
    task_kind: Literal["general", "investigation", "understanding", "implementation", "integration", "verification"]
    backend: Literal["internal", "external"]
    source_view: str
    evidence: tuple[EvidenceRef, ...] = ()
    requirements: tuple[EvidenceRequirement, ...] = ()
    allowed_scope: tuple[str, ...] = ()
    layout_only: bool = False
    dependencies: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    expected_changes: tuple[str, ...] = ()
    integration_requirements: tuple[str, ...] = ()
    verification_criteria: tuple[str, ...] = ()
    plan_digest: str = ""
    decision_id: str = Field(min_length=1)


class MissingEvidenceRequest(Contract):
    task_id: str
    requirement: EvidenceRequirement
    reason: str = Field(min_length=1, max_length=2000)
    continuation_reference: str = ""


class InvestigationResult(Contract):
    findings: tuple[InvestigationFinding, ...] = Field(default=(), max_length=20)
    evidence: tuple[EvidenceRef, ...] = ()
    missing_evidence: tuple[MissingEvidenceRequest, ...] = ()
    limitations: tuple[str, ...] = Field(
        default=(), description="Array of text notes, one per limitation. Omit it or use [] if there are no notes.",
    )
    # Accept positional bindings from older reports without advertising them to the engines.
    requirement_bindings: SkipJsonSchema[dict[str, tuple[int, ...]]] = Field(default_factory=dict)
    evidence_bindings: dict[str, tuple[EvidenceRef, ...]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_report_envelope(cls, value: Any) -> Any:
        """Normalize optional report notes before checking for unsupported metadata."""
        if not isinstance(value, dict):
            return value
        limitations = value.get("limitations", ())
        if limitations is None or isinstance(limitations, str):
            limitations = [limitations] if limitations and limitations.strip() else []
            value = {**value, "limitations": limitations}
        extra_fields = value.keys() - cls.model_fields.keys()
        if not extra_fields:
            return value
        misplaced_finding_fields = InvestigationFinding.model_fields.keys() - {"architecture_relationships"}
        if extra_fields & misplaced_finding_fields:
            return value

        bindings = value.get("evidence_bindings")
        has_report_body = any(value.get(field) for field in ("findings", "evidence", "missing_evidence", "limitations"))
        if not has_report_body and not (isinstance(bindings, dict) and any(bindings.values())):
            return value
        if not isinstance(limitations, (list, tuple)):
            return value

        report = {key: item for key, item in value.items() if key not in extra_fields}
        report["limitations"] = [
            *limitations, f"Ignored {len(extra_fields)} unsupported top-level report fields.",
        ]
        return report


class PatchEvidenceResult(Contract):
    patch: str
    missing_evidence: tuple[MissingEvidenceRequest, ...] = ()
