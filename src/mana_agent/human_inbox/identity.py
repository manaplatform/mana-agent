"""Reviewer identity and role resolution boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import Field

from .models import ReviewerAssignment, ReviewerType, StrictModel


class ReviewerIdentity(StrictModel):
    identity_id: str = Field(min_length=1)
    display_name: str = ""
    roles: set[str] = Field(default_factory=set)
    groups: set[str] = Field(default_factory=set)
    active: bool = True
    tenant_ids: set[str] = Field(default_factory=lambda: {"local"})
    project_ids: set[str] = Field(default_factory=set)


class IdentityDirectory(Protocol):
    def resolve(self, assignment: ReviewerAssignment) -> list[ReviewerIdentity]: ...
    def get(self, identity_id: str) -> ReviewerIdentity | None: ...


class StaticIdentityDirectory:
    def __init__(self, identities: list[ReviewerIdentity] | None = None) -> None:
        self._identities = {identity.identity_id: identity for identity in identities or []}

    def get(self, identity_id: str) -> ReviewerIdentity | None:
        identity = self._identities.get(identity_id)
        return identity if identity is not None and identity.active else None

    def resolve(self, assignment: ReviewerAssignment) -> list[ReviewerIdentity]:
        if assignment.reviewer_type is ReviewerType.PERSON:
            recorded = self._identities.get(assignment.reviewer_id)
            if recorded is not None and not recorded.active:
                return []
            configured = self.get(assignment.reviewer_id)
            # A specific identity is itself an explicit policy selection. It is
            # eligible even before optional directory metadata is configured.
            return [configured or ReviewerIdentity(identity_id=assignment.reviewer_id)]
        if assignment.reviewer_type is ReviewerType.ROLE:
            rows = [identity for identity in self._identities.values() if identity.active and assignment.reviewer_id in identity.roles]
        else:
            rows = [identity for identity in self._identities.values() if identity.active and assignment.reviewer_id in identity.groups]
        return sorted(rows, key=lambda identity: identity.identity_id)


class FileIdentityDirectory(StaticIdentityDirectory):
    """Read the explicit local reviewer directory from ``identities.json``."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        super().__init__([])
        self._reload()

    def _reload(self) -> None:
        identities: list[ReviewerIdentity] = []
        if self.path.is_file():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            rows = payload.get("identities", []) if isinstance(payload, dict) else payload
            identities = [ReviewerIdentity.model_validate(row) for row in rows]
        self._identities = {identity.identity_id: identity for identity in identities}

    def get(self, identity_id: str) -> ReviewerIdentity | None:
        self._reload()
        return StaticIdentityDirectory.get(self, identity_id)

    def resolve(self, assignment: ReviewerAssignment) -> list[ReviewerIdentity]:
        self._reload()
        return StaticIdentityDirectory.resolve(self, assignment)
