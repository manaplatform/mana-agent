from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


RepositoryKind = Literal["git", "monorepo", "service", "library", "app", "docs", "infrastructure", "project", "unknown"]
RelationshipKind = Literal[
    "declared_dependency",
    "path_dependency",
    "workspace_package",
    "git_submodule",
    "shared_contract",
    "api_consumer",
    "deployment",
    "inferred",
]


class RepositoryComponent(BaseModel):
    component_id: str = Field(default_factory=lambda: _id("component"))
    name: str
    relative_path: str
    kind: RepositoryKind = "unknown"
    languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    manifests: list[str] = Field(default_factory=list)


class RepositoryStatus(BaseModel):
    available: bool = True
    dirty: bool = False
    indexed: bool = False
    index_stale: bool = True
    error: str = ""


class RepositoryRecord(BaseModel):
    schema_version: int = 1
    repository_id: str = Field(default_factory=lambda: _id("repo"))
    name: str
    canonical_path: str
    git_root: str | None = None
    remote_url: str | None = None
    branch: str | None = None
    head_sha: str | None = None
    languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    kind: RepositoryKind = "unknown"
    role: str = "unknown"
    tags: list[str] = Field(default_factory=list)
    components: list[RepositoryComponent] = Field(default_factory=list)
    status: RepositoryStatus = Field(default_factory=RepositoryStatus)
    created_at: str = Field(default_factory=utc_iso)
    updated_at: str = Field(default_factory=utc_iso)
    legacy_imported_at: str | None = None

    @field_validator("canonical_path")
    @classmethod
    def _absolute_path(cls, value: str) -> str:
        path = Path(value).expanduser().resolve()
        if not path.is_absolute():
            raise ValueError("repository path must be absolute")
        return str(path)


class WorkspaceDiscoveryConfig(BaseModel):
    roots: list[str] = Field(default_factory=list)
    max_depth: int = Field(default=6, ge=0, le=32)
    exclude: list[str] = Field(default_factory=list)


class WorkspaceRecord(BaseModel):
    schema_version: int = 1
    workspace_id: str = Field(default_factory=lambda: _id("workspace"))
    name: str
    repository_ids: list[str] = Field(default_factory=list)
    primary_repository_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    discovery: WorkspaceDiscoveryConfig = Field(default_factory=WorkspaceDiscoveryConfig)
    allowed_roots: list[str] = Field(default_factory=list)
    implicit: bool = False
    created_at: str = Field(default_factory=utc_iso)
    updated_at: str = Field(default_factory=utc_iso)

    @model_validator(mode="after")
    def _primary_is_member(self) -> "WorkspaceRecord":
        if self.primary_repository_id and self.primary_repository_id not in self.repository_ids:
            raise ValueError("primary_repository_id must be a workspace member")
        return self


class SessionRecord(BaseModel):
    schema_version: int = 2
    session_id: str = Field(default_factory=lambda: _id("session"))
    workspace_id: str
    primary_repository_id: str
    attached_repository_ids: list[str] = Field(default_factory=list)
    cwd: str
    title: str = "New chat"
    status: Literal["active", "closed", "abandoned", "archived"] = "active"
    active_flow_id: str | None = None
    created_at: str = Field(default_factory=utc_iso)
    opened_at: str = Field(default_factory=utc_iso)
    closed_at: str | None = None
    owner_pid: int | None = None
    updated_at: str = Field(default_factory=utc_iso)

    @model_validator(mode="after")
    def _include_primary(self) -> "SessionRecord":
        if self.primary_repository_id not in self.attached_repository_ids:
            self.attached_repository_ids.insert(0, self.primary_repository_id)
        return self


class RepositoryPermission(BaseModel):
    repository_id: str
    access: Literal["read", "write", "git", "verify"] = "read"
    allowed_tools: list[str] = Field(default_factory=list)
    allowed_prefixes: list[str] = Field(default_factory=list)


class RepositoryScopeDecision(BaseModel):
    workspace_id: str
    session_id: str
    primary_repository_id: str
    repository_ids: list[str]
    permissions: list[RepositoryPermission] = Field(default_factory=list)
    relationship_depth: int = Field(default=0, ge=0, le=5)
    requires_multi_repo: bool = False
    requires_verification: bool = False
    safe_to_continue: bool = False
    reason: str
    source: Literal["model", "session_binding"] = "model"


class RepositoryRelationship(BaseModel):
    relationship_id: str = Field(default_factory=lambda: _id("relationship"))
    workspace_id: str
    source_repository_id: str
    target_repository_id: str
    kind: RelationshipKind
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    detector: str = "manifest"
    review_state: Literal["confirmed", "needs_model_review", "rejected"] = "confirmed"
    created_at: str = Field(default_factory=utc_iso)
    updated_at: str = Field(default_factory=utc_iso)


class ImpactNode(BaseModel):
    repository_id: str
    qualified_path: str = ""
    symbol: str = ""
    reason: str
    depth: int = 0
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ImpactReport(BaseModel):
    workspace_id: str
    source_repository_id: str
    changed_paths: list[str]
    affected: list[ImpactNode] = Field(default_factory=list)
    verification_by_repository: dict[str, list[str]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=utc_iso)


class WorkspaceSearchRequest(BaseModel):
    workspace_id: str
    query: str
    mode: Literal["semantic", "text", "file", "symbol"] = "semantic"
    repository_ids: list[str] = Field(default_factory=list)
    limit: int = Field(default=20, ge=1, le=500)
    metadata: dict[str, Any] = Field(default_factory=dict)
