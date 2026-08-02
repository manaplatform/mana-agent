"""Deny-by-default authorization for every capsule operation."""

from __future__ import annotations

from typing import Any

from mana_agent.memory.capsules.models import (
    MERGE_CAPABILITY,
    READ_CAPABILITY,
    STAGE_CAPABILITY,
    WRITE_CAPABILITY,
    AuthorizationDecision,
    CapsuleScope,
    CapsuleTaskContext,
    MemoryCapsule,
    MemoryPrincipal,
)
from mana_agent.memory.capsules.namespaces import namespace_for


def _allow(policy: str) -> AuthorizationDecision:
    return AuthorizationDecision(True, "allowed", "Access granted by capsule policy.", policy)


def _deny(code: str, reason: str) -> AuthorizationDecision:
    return AuthorizationDecision(False, code, reason, None)


class CapsuleAuthorizationPolicy:
    """Central policy service; provider filters are never authorization."""

    def __init__(self, *, organisation_scope_enabled: bool = False, user_scope_enabled: bool = True) -> None:
        self.organisation_scope_enabled = organisation_scope_enabled
        self.user_scope_enabled = user_scope_enabled

    def authorize_read(
        self,
        principal: MemoryPrincipal,
        capsule: MemoryCapsule,
        context: CapsuleTaskContext,
    ) -> AuthorizationDecision:
        capability = READ_CAPABILITY[capsule.scope]
        if not principal.has(capability):
            return _deny("missing_capability", f"Principal lacks {capability}.")
        if capsule.deleted_at is not None:
            return _deny("not_found", "Capsule is unavailable.")
        if capsule.expired:
            return _deny("expired", "Capsule has expired.")
        if capsule.trust_state.value in {"quarantined", "rejected"}:
            return _deny("trust_state_blocked", "Capsule trust state is not readable.")
        identity = self._identity_allows(principal, capsule, context)
        if not identity.allowed:
            return identity
        try:
            if capsule.scope is CapsuleScope.PARENT_CHILD:
                expected = f"tasks/{capsule.parent_task_id}/children/{capsule.task_id}"
            else:
                expected = namespace_for(capsule.scope, context, team_id=capsule.team_id)
        except ValueError as exc:
            return _deny("trusted_context_missing", str(exc))
        if expected != capsule.namespace:
            return _deny("namespace_mismatch", "Capsule namespace does not match trusted context.")
        return _allow(f"read.{capsule.scope.value}")

    def authorize_create(
        self,
        principal: MemoryPrincipal,
        requested_scope: CapsuleScope,
        namespace: str,
        context: CapsuleTaskContext,
        *,
        team_id: str | None = None,
    ) -> AuthorizationDecision:
        if requested_scope in {CapsuleScope.TEAM, CapsuleScope.PROJECT, CapsuleScope.ORGANISATION}:
            return _deny("shared_write_requires_staging", "Shared capsules must enter the review staging path.")
        if requested_scope is CapsuleScope.USER and not self.user_scope_enabled:
            return _deny("scope_disabled", "User capsule scope is disabled.")
        capability = WRITE_CAPABILITY.get(requested_scope)
        if not capability or not principal.has(capability):
            return _deny("missing_capability", f"Principal lacks {capability or 'a write capability'}.")
        return self._authorize_namespace(principal, requested_scope, namespace, context, team_id=team_id)

    def authorize_update(
        self,
        principal: MemoryPrincipal,
        capsule: MemoryCapsule,
        patch: dict[str, Any],
        context: CapsuleTaskContext,
    ) -> AuthorizationDecision:
        if context.task_completed:
            return _deny("completed_task_immutable", "Completed task capsules require supersession, redaction, or authorized deletion.")
        if capsule.merge_state.value in {"merged", "superseded"}:
            return _deny("immutable_revision", "Shared or superseded capsule revisions are immutable.")
        if any(key in patch for key in ("scope", "namespace", "owner_user_id", "project_id", "team_id", "organisation_id")):
            return _deny("identity_patch_forbidden", "Capsule identity and scope fields cannot be patched.")
        return self.authorize_create(principal, capsule.scope, capsule.namespace, context, team_id=capsule.team_id)

    def authorize_stage(
        self,
        principal: MemoryPrincipal,
        capsule: MemoryCapsule,
        target_scope: CapsuleScope,
        context: CapsuleTaskContext,
        *,
        team_id: str | None = None,
    ) -> AuthorizationDecision:
        capability = STAGE_CAPABILITY.get(target_scope)
        if not capability or not principal.has(capability):
            return _deny("missing_capability", f"Principal lacks {capability or 'a staging capability'}.")
        if capsule.task_id != principal.task_id or capsule.scope is not CapsuleScope.PRIVATE:
            return _deny("invalid_stage_source", "Only the principal's private capsule may be staged.")
        if capsule.deleted_at is not None or capsule.expired:
            return _deny("invalid_stage_source", "Deleted or expired capsules cannot be staged.")
        if capsule.trust_state.value in {"quarantined", "rejected"}:
            return _deny("trust_state_blocked", "Quarantined or rejected capsules cannot be staged.")
        try:
            namespace = namespace_for(target_scope, context, team_id=team_id)
        except ValueError as exc:
            return _deny("trusted_context_missing", str(exc))
        return self._authorize_namespace(principal, target_scope, namespace, context, team_id=team_id)

    def authorize_merge(
        self,
        principal: MemoryPrincipal,
        staged_capsule: MemoryCapsule,
        target_capsule: MemoryCapsule | None,
        context: CapsuleTaskContext,
    ) -> AuthorizationDecision:
        target_scope = staged_capsule.proposed_scope
        capability = MERGE_CAPABILITY.get(target_scope) if target_scope else None
        if not principal.has("memory.capsule.review") or not capability or not principal.has(capability):
            return _deny("missing_capability", "Principal lacks capsule review or merge capability.")
        if target_scope is CapsuleScope.ORGANISATION and not self.organisation_scope_enabled:
            return _deny("scope_disabled", "Organisation capsule scope is disabled.")
        if staged_capsule.review_state.value != "pending" or staged_capsule.merge_state.value != "staged":
            return _deny("not_staged", "Capsule is not pending review.")
        if target_capsule is not None and target_capsule.scope is not target_scope:
            return _deny("target_scope_mismatch", "Merge target scope differs from the staged proposal.")
        return self._authorize_namespace(
            principal,
            target_scope,
            staged_capsule.proposed_namespace or "",
            context,
            team_id=staged_capsule.team_id,
        )

    def authorize_delete(
        self,
        principal: MemoryPrincipal,
        capsule: MemoryCapsule,
        context: CapsuleTaskContext,
    ) -> AuthorizationDecision:
        if not principal.has("memory.capsule.delete"):
            return _deny("missing_capability", "Principal lacks memory.capsule.delete.")
        return self.authorize_read(principal, capsule, context)

    def _identity_allows(
        self,
        principal: MemoryPrincipal,
        capsule: MemoryCapsule,
        context: CapsuleTaskContext,
    ) -> AuthorizationDecision:
        if capsule.owner_user_id and capsule.owner_user_id != principal.user_id:
            return _deny("user_mismatch", "Capsule belongs to another user.")
        if capsule.scope is CapsuleScope.PRIVATE:
            same_private_owner = (
                capsule.task_id == principal.task_id
                if principal.task_id is not None
                else bool(principal.agent_id and capsule.agent_id == principal.agent_id)
            )
            if not same_private_owner:
                return _deny("private_owner_mismatch", "Private capsule belongs to another task or agent.")
        elif capsule.scope is CapsuleScope.PARENT_CHILD:
            is_child = principal.task_id == capsule.task_id and principal.parent_task_id == capsule.parent_task_id
            is_parent = principal.task_id == capsule.parent_task_id
            if not is_child and not is_parent:
                return _deny("task_relationship_mismatch", "No direct parent-child relationship exists.")
        elif capsule.scope is CapsuleScope.TEAM and capsule.team_id not in principal.team_ids:
            return _deny("team_mismatch", "Principal is not a member of the capsule team.")
        elif capsule.scope is CapsuleScope.PROJECT and capsule.project_id != principal.project_id:
            return _deny("project_mismatch", "Principal is outside the capsule project.")
        elif capsule.scope is CapsuleScope.ORGANISATION:
            if not self.organisation_scope_enabled:
                return _deny("scope_disabled", "Organisation capsule scope is disabled.")
            if capsule.organisation_id != principal.organisation_id:
                return _deny("organisation_mismatch", "Principal is outside the capsule organisation.")
        elif capsule.scope is CapsuleScope.USER:
            if not self.user_scope_enabled or capsule.owner_user_id != principal.user_id:
                return _deny("user_mismatch", "Principal is outside the capsule user scope.")
        return _allow("identity")

    def _authorize_namespace(
        self,
        principal: MemoryPrincipal,
        scope: CapsuleScope,
        namespace: str,
        context: CapsuleTaskContext,
        *,
        team_id: str | None,
    ) -> AuthorizationDecision:
        direct_parent_delegation = (
            scope is CapsuleScope.PARENT_CHILD
            and principal.task_id == context.parent_task_id
        )
        if (principal.task_id != context.task_id and not direct_parent_delegation) or principal.user_id != context.user_id:
            return _deny("principal_context_mismatch", "Principal identity differs from trusted task context.")
        if principal.project_id != context.project_id or not principal.team_ids.issuperset(context.team_ids):
            return _deny("principal_context_mismatch", "Principal project or team identity differs from trusted task context.")
        try:
            expected = namespace_for(scope, context, team_id=team_id)
        except ValueError as exc:
            return _deny("trusted_context_missing", str(exc))
        if namespace != expected:
            return _deny("namespace_mismatch", "Requested namespace was not derived from trusted task context.")
        return _allow(f"namespace.{scope.value}")
