"""Authorization-first lifecycle service for scoped memory capsules."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

from mana_agent.memory.capsules.audit import CapsuleAuditLogger
from mana_agent.memory.capsules.models import (
    AuthorizationDecision,
    CapsuleLineage,
    CapsuleMergeRecord,
    CapsuleProjection,
    CapsuleReadRequest,
    CapsuleScope,
    CapsuleTaskContext,
    DeleteMode,
    MemoryCapsule,
    MemoryPrincipal,
    MergeState,
    MergeStrategy,
    ReviewState,
    TrustState,
    utc_now,
)
from mana_agent.memory.capsules.namespaces import namespace_for
from mana_agent.memory.capsules.policy import CapsuleAuthorizationPolicy
from mana_agent.memory.capsules.repository import CapsuleRepository, RevisionConflict
from mana_agent.memory.config import CapsuleConfig
from mana_agent.memory.errors import MemoryConfigurationError, MemoryNotFoundError
from mana_agent.workspaces.paths import repository_dir, repository_id_for_path


class CapsuleAuthorizationError(PermissionError):
    def __init__(self, decision: AuthorizationDecision) -> None:
        super().__init__(f"Capsule authorization denied: {decision.reason_code}. {decision.reason}")
        self.decision = decision


class CapsuleMergeConflict(RuntimeError):
    pass


_INJECTION_PATTERNS = (
    re.compile(r"\b(ignore|override|bypass)\b.{0,50}\b(system|developer|policy|permission|instruction)", re.I),
    re.compile(r"\b(api[_ -]?key|password|credential|private key|access token)\b", re.I),
    re.compile(r"\bgrant (me|this capsule|the agent)\b.{0,30}\b(capability|permission|access)\b", re.I),
)


def _content_hash(content: dict[str, Any]) -> str:
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _content_text(capsule: MemoryCapsule) -> str:
    return " ".join((capsule.title, capsule.summary, json.dumps(capsule.content, sort_keys=True), " ".join(capsule.tags))).lower()


class CapsuleService:
    """Single service boundary used by gateway, agents, API, CLI, and dashboard."""

    def __init__(
        self,
        root: str | Path,
        *,
        config: CapsuleConfig | None = None,
        provider: str = "mana",
        repository: CapsuleRepository | None = None,
        audit: CapsuleAuditLogger | None = None,
        retention_hook: Any | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.config = (config or CapsuleConfig()).validate()
        self.provider = provider
        storage_dir = repository_dir(repository_id_for_path(self.root))
        self.repository = repository or CapsuleRepository(storage_dir / "memory_capsules.json")
        self.audit = audit or CapsuleAuditLogger()
        self.retention_hook = retention_hook
        self.policy = CapsuleAuthorizationPolicy(
            organisation_scope_enabled=self.config.organisation_scope_enabled,
            user_scope_enabled=self.config.user_scope_enabled,
        )
        canonical_user = str(
            getattr(config, "user_id", "")
            or getattr(self.config, "user_id", "")
            or ""
        ).strip()
        if not canonical_user:
            try:
                from mana_agent.config.user_config import resolve_local_user_id

                canonical_user = resolve_local_user_id()
            except Exception:
                canonical_user = ""
        if canonical_user:
            try:
                self.repository.migrate_legacy_local_identities(canonical_user)
            except Exception:
                pass

    def migrate_legacy_local_identities(
        self,
        canonical_user_id: str,
        *,
        legacy_local_identities: set[str] | None = None,
    ) -> int:
        """Migrate locally owned legacy capsule records to the canonical Mana user identity."""
        return self.repository.migrate_legacy_local_identities(
            canonical_user_id,
            legacy_local_identities=legacy_local_identities,
        )

    def effective_settings(self) -> dict[str, Any]:
        retention = self.config.retention
        return {
            "enabled": self.config.enabled,
            "default_max_capsules": self.config.default_max_capsules,
            "default_max_tokens": self.config.default_max_tokens,
            "shared_writes_require_review": self.config.shared_writes_require_review,
            "organisation_scope_enabled": self.config.organisation_scope_enabled,
            "user_scope_enabled": self.config.user_scope_enabled,
            "record_access_events": self.config.record_access_events,
            "quarantine_prompt_injection": self.config.quarantine_prompt_injection,
            "retention": {
                "private_days": retention.private_days,
                "parent_child_days": retention.parent_child_days,
                "team_days": retention.team_days,
                "project_days": retention.project_days,
                "organisation_days": retention.organisation_days,
            },
        }

    def create_capsule(
        self,
        *,
        principal: MemoryPrincipal,
        context: CapsuleTaskContext,
        scope: CapsuleScope,
        title: str,
        summary: str,
        content: dict[str, Any],
        origin_type: str,
        origin_id: str,
        tags: list[str] | None = None,
        source_capsule_ids: list[str] | None = None,
        trust_state: TrustState = TrustState.AGENT_GENERATED,
        team_id: str | None = None,
        supporting_evidence: list[str] | None = None,
        correlation_id: str = "",
    ) -> MemoryCapsule:
        self._require_enabled()
        namespace = namespace_for(scope, context, team_id=team_id)
        decision = self.policy.authorize_create(principal, scope, namespace, context, team_id=team_id)
        self._require(decision, principal=principal, event="capsule.created", correlation_id=correlation_id)
        self._validate_content(title, summary, content)
        risk_flags = self._risk_flags(title, summary, content)
        if risk_flags and self.config.quarantine_prompt_injection:
            trust_state = TrustState.QUARANTINED
        now = utc_now()
        capsule = MemoryCapsule(
            capsule_id=uuid.uuid4().hex,
            schema_version=1,
            scope=scope,
            namespace=namespace,
            owner_user_id=context.user_id,
            organisation_id=context.organisation_id,
            project_id=context.project_id,
            team_id=team_id,
            task_id=context.task_id,
            parent_task_id=context.parent_task_id,
            agent_id=context.agent_id,
            session_id=context.session_id,
            title=title.strip()[:160],
            summary=summary.strip()[:2000],
            content=copy.deepcopy(content),
            tags=self._tags(tags),
            origin_type=origin_type.strip()[:80],
            origin_id=origin_id.strip()[:256],
            source_capsule_ids=list(dict.fromkeys(source_capsule_ids or [])),
            trust_state=trust_state,
            review_state=ReviewState.NOT_REQUIRED,
            merge_state=MergeState.NONE,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(days=self._retention_days(scope)),
            created_by=principal,
            updated_by=principal,
            content_hash=_content_hash(content),
            revision=1,
            provider=self.provider,
            supporting_evidence=[str(item)[:500] for item in (supporting_evidence or [])],
            risk_flags=risk_flags,
        )
        self.repository.put(capsule)
        event = "capsule.quarantined" if trust_state is TrustState.QUARANTINED else "capsule.created"
        self.audit.emit(event, principal=principal, capsule=capsule, correlation_id=correlation_id)
        return capsule

    def query_capsules(self, request: CapsuleReadRequest, *, correlation_id: str = "") -> list[CapsuleProjection]:
        self._require_enabled()
        max_capsules = min(max(1, request.max_capsules), self.config.default_max_capsules)
        max_tokens = min(max(1, request.max_tokens), self.config.default_max_tokens)
        if not request.allowed_scopes:
            raise ValueError("Capsule read request requires explicit allowed scopes; no fallback scope set was used.")
        scopes = request.allowed_scopes
        namespaces = request.namespaces
        terms = set(request.query.lower().split())
        candidates: list[tuple[float, MemoryCapsule]] = []
        self.audit.emit("capsule.read_requested", principal=request.principal, correlation_id=correlation_id)
        for capsule in self.repository.list():
            if capsule.scope not in scopes or (namespaces and capsule.namespace not in namespaces):
                continue
            if not request.include_staged and capsule.merge_state is MergeState.STAGED:
                continue
            decision = self.policy.authorize_read(request.principal, capsule, request.task_context)
            self._record_decision(request.principal, capsule, decision, correlation_id=correlation_id)
            if not decision.allowed:
                if decision.reason_code in {
                    "user_mismatch",
                    "private_owner_mismatch",
                    "task_relationship_mismatch",
                    "team_mismatch",
                    "project_mismatch",
                    "organisation_mismatch",
                    "namespace_mismatch",
                }:
                    self.audit.emit(
                        "capsule.access_anomaly",
                        principal=request.principal,
                        capsule=capsule,
                        decision_code=decision.reason_code,
                        correlation_id=correlation_id,
                    )
                continue
            # Reauthorization happens after the repository result has been materialized.
            rechecked = self.policy.authorize_read(request.principal, capsule, request.task_context)
            if not rechecked.allowed:
                self.audit.emit("capsule.access_anomaly", principal=request.principal, capsule=capsule, decision_code=rechecked.reason_code)
                continue
            words = set(_content_text(capsule).split())
            relevance = len(terms & words) / max(1, len(terms)) if terms else 1.0
            if terms and relevance == 0:
                continue
            candidates.append((relevance, capsule))
        candidates.sort(key=lambda item: (-item[0], -item[1].updated_at.timestamp(), item[1].capsule_id))
        result: list[CapsuleProjection] = []
        used_tokens = 0
        for _, capsule in candidates:
            projection = self._projection(capsule)
            from mana_agent.context_cost.estimator import estimate_value_tokens

            estimated = estimate_value_tokens({"content": projection.content, "summary": projection.summary})
            if used_tokens + estimated > max_tokens:
                continue
            result.append(projection)
            used_tokens += estimated
            self.repository.record_access({
                "capsule_id": capsule.capsule_id,
                "revision": capsule.revision,
                "task_id": request.principal.task_id,
                "agent_id": request.principal.agent_id,
                "timestamp": utc_now().isoformat(),
                "correlation_id": correlation_id,
            })
            if len(result) >= max_capsules:
                break
        return result

    def get_capsule(
        self,
        capsule_id: str,
        *,
        principal: MemoryPrincipal,
        context: CapsuleTaskContext,
    ) -> CapsuleProjection:
        capsule = self.repository.get(capsule_id)
        if capsule is None:
            raise MemoryNotFoundError("Capsule was not found.")
        decision = self.policy.authorize_read(principal, capsule, context)
        if not decision.allowed:
            self._record_decision(principal, capsule, decision)
            raise MemoryNotFoundError("Capsule was not found.")
        return self._projection(capsule)

    def delegate_to_child(
        self,
        capsule_ids: list[str],
        *,
        parent_principal: MemoryPrincipal,
        parent_context: CapsuleTaskContext,
        child_context: CapsuleTaskContext,
        correlation_id: str = "",
    ) -> list[MemoryCapsule]:
        """Copy only explicitly selected, parent-authorized projections to a child."""
        if child_context.parent_task_id != parent_principal.task_id:
            raise CapsuleAuthorizationError(AuthorizationDecision(False, "task_relationship_mismatch", "Child context is not directly related to the parent principal."))
        delegated: list[MemoryCapsule] = []
        for capsule_id in dict.fromkeys(capsule_ids):
            projection = self.get_capsule(capsule_id, principal=parent_principal, context=parent_context)
            delegated.append(self.create_capsule(
                principal=parent_principal,
                context=child_context,
                scope=CapsuleScope.PARENT_CHILD,
                title=projection.title,
                summary=projection.summary,
                content=projection.content,
                origin_type="capsule_delegation",
                origin_id=capsule_id,
                tags=[*projection.tags, "delegated"],
                source_capsule_ids=[capsule_id],
                trust_state=projection.trust_state,
                correlation_id=correlation_id,
            ))
        return delegated

    def create_child_return(
        self,
        *,
        child_principal: MemoryPrincipal,
        child_context: CapsuleTaskContext,
        title: str,
        summary: str,
        content: dict[str, Any],
        origin_id: str,
        source_capsule_ids: list[str] | None = None,
        correlation_id: str = "",
    ) -> MemoryCapsule:
        """Persist an immutable child-to-parent return capsule."""
        return self.create_capsule(
            principal=child_principal,
            context=child_context,
            scope=CapsuleScope.PARENT_CHILD,
            title=title,
            summary=summary,
            content=content,
            origin_type="child_task_result",
            origin_id=origin_id,
            tags=["child-return"],
            source_capsule_ids=source_capsule_ids,
            trust_state=TrustState.AGENT_GENERATED,
            correlation_id=correlation_id,
        )

    def update_capsule(
        self,
        capsule_id: str,
        *,
        principal: MemoryPrincipal,
        context: CapsuleTaskContext,
        patch: dict[str, Any],
        expected_revision: int,
    ) -> MemoryCapsule:
        capsule = self._existing(capsule_id)
        decision = self.policy.authorize_update(principal, capsule, patch, context)
        self._require(decision, principal=principal, capsule=capsule, event="capsule.update_denied")
        if capsule.revision != expected_revision:
            raise CapsuleMergeConflict("Capsule changed after it was read; no update was applied.")
        for key in ("title", "summary", "content", "tags", "expires_at"):
            if key in patch:
                setattr(capsule, key, copy.deepcopy(patch[key]))
        self._validate_content(capsule.title, capsule.summary, capsule.content)
        capsule.updated_at = utc_now()
        capsule.updated_by = principal
        capsule.content_hash = _content_hash(capsule.content)
        capsule.revision += 1
        self.repository.put(capsule, expected_revision=expected_revision)
        return capsule

    def stage_capsule(
        self,
        capsule_id: str,
        *,
        principal: MemoryPrincipal,
        context: CapsuleTaskContext,
        target_scope: CapsuleScope,
        strategy: MergeStrategy,
        team_id: str | None = None,
        correlation_id: str = "",
    ) -> MemoryCapsule:
        source = self._existing(capsule_id)
        decision = self.policy.authorize_stage(principal, source, target_scope, context, team_id=team_id)
        self._require(decision, principal=principal, capsule=source, event="capsule.stage_denied", correlation_id=correlation_id)
        staged = copy.deepcopy(source)
        staged.capsule_id = uuid.uuid4().hex
        staged.scope = CapsuleScope.PRIVATE
        staged.namespace = namespace_for(CapsuleScope.PRIVATE, context)
        staged.team_id = team_id
        staged.source_capsule_ids = list(dict.fromkeys([*source.source_capsule_ids, source.capsule_id]))
        staged.proposed_scope = target_scope
        staged.proposed_namespace = namespace_for(target_scope, context, team_id=team_id)
        staged.requested_operation = strategy
        staged.review_state = ReviewState.PENDING
        staged.merge_state = MergeState.STAGED
        staged.trust_state = TrustState.UNTRUSTED
        staged.created_at = staged.updated_at = utc_now()
        staged.created_by = staged.updated_by = principal
        staged.revision = 1
        self.repository.put(staged)
        self.audit.emit("capsule.staged", principal=principal, capsule=staged, correlation_id=correlation_id)
        return staged

    def list_staged_capsules(
        self,
        *,
        principal: MemoryPrincipal,
        context: CapsuleTaskContext,
    ) -> list[CapsuleProjection]:
        if not principal.has("memory.capsule.review"):
            raise CapsuleAuthorizationError(AuthorizationDecision(False, "missing_capability", "Principal lacks memory.capsule.review."))
        rows = [item for item in self.repository.list() if item.merge_state is MergeState.STAGED]
        visible = []
        for item in rows:
            decision = self.policy.authorize_merge(principal, item, None, context)
            if decision.allowed:
                visible.append(self._projection(item))
        return sorted(visible, key=lambda item: (item.created_at, item.capsule_id))

    def merge_capsule(
        self,
        staged_capsule_id: str,
        *,
        principal: MemoryPrincipal,
        context: CapsuleTaskContext,
        request_id: str,
        target_capsule_id: str | None = None,
        expected_target_revision: int | None = None,
        expected_target_hash: str | None = None,
        strategy: MergeStrategy | None = None,
        decision_reason: str = "",
        correlation_id: str = "",
    ) -> CapsuleMergeRecord:
        staged = self._existing(staged_capsule_id)
        target = self._existing(target_capsule_id) if target_capsule_id else None
        existing_merge = self.repository.merge_for_request(request_id)
        if existing_merge is not None:
            if staged_capsule_id not in existing_merge.source_capsule_ids:
                raise CapsuleMergeConflict("Merge request ID is already bound to another operation.")
            retry_view = copy.deepcopy(staged)
            retry_view.review_state = ReviewState.PENDING
            retry_view.merge_state = MergeState.STAGED
            retry_decision = self.policy.authorize_merge(principal, retry_view, target, context)
            self._require(retry_decision, principal=principal, capsule=staged, event="capsule.merge_requested", correlation_id=correlation_id)
            return existing_merge
        decision = self.policy.authorize_merge(principal, staged, target, context)
        self._require(decision, principal=principal, capsule=staged, event="capsule.merge_requested", correlation_id=correlation_id)
        selected = strategy or staged.requested_operation
        if selected is None:
            raise CapsuleMergeConflict("A validated merge strategy is required; no fallback strategy was selected.")
        if selected is MergeStrategy.REJECT:
            staged_revision = staged.revision
            staged.review_state = ReviewState.REJECTED
            staged.merge_state = MergeState.REJECTED
            staged.trust_state = TrustState.REJECTED
            staged.updated_at = utc_now()
            staged.updated_by = principal
            staged.revision += 1
            record = self._merge_record(staged, target, selected, expected_target_revision, expected_target_hash, None, principal, decision_reason, request_id)
            self.repository.commit_merge(
                staged=staged,
                expected_staged_revision=staged_revision,
                result=None,
                expected_target_revision=expected_target_revision,
                record=record,
            )
            self.audit.emit("capsule.rejected", principal=principal, capsule=staged, correlation_id=correlation_id)
            return record
        if target is not None and (
            target.revision != expected_target_revision
            or not expected_target_hash
            or target.content_hash != expected_target_hash
        ):
            return self._record_conflict(staged, target, selected, expected_target_revision, expected_target_hash, principal, decision_reason, request_id, correlation_id)
        try:
            resulting = self._apply_merge(staged, target, selected, principal, context)
        except RevisionConflict:
            return self._record_conflict(self._existing(staged_capsule_id), target, selected, expected_target_revision, expected_target_hash, principal, decision_reason, request_id, correlation_id)
        staged_revision = staged.revision
        staged.review_state = ReviewState.APPROVED
        staged.merge_state = MergeState.MERGED
        staged.trust_state = TrustState.REVIEWED
        staged.updated_by = principal
        staged.updated_at = utc_now()
        staged.revision += 1
        record = self._merge_record(staged, target, selected, expected_target_revision, expected_target_hash, resulting.capsule_id, principal, decision_reason, request_id)
        try:
            self.repository.commit_merge(
                staged=staged,
                expected_staged_revision=staged_revision,
                result=resulting,
                expected_target_revision=expected_target_revision,
                record=record,
            )
        except RevisionConflict:
            return self._record_conflict(self._existing(staged_capsule_id), target, selected, expected_target_revision, expected_target_hash, principal, decision_reason, request_id, correlation_id)
        self.audit.emit("capsule.merged", principal=principal, capsule=resulting, correlation_id=correlation_id)
        return record

    def resolve_conflict(
        self,
        staged_capsule_id: str,
        *,
        principal: MemoryPrincipal,
        context: CapsuleTaskContext,
        request_id: str,
        target_capsule_id: str | None,
        expected_target_revision: int | None,
        expected_target_hash: str | None,
        strategy: MergeStrategy,
        decision_reason: str,
    ) -> CapsuleMergeRecord:
        staged = self._existing(staged_capsule_id)
        if staged.merge_state is not MergeState.CONFLICT:
            raise CapsuleMergeConflict("Capsule does not have a merge conflict awaiting resolution.")
        pending = copy.deepcopy(staged)
        pending.merge_state = MergeState.STAGED
        pending.review_state = ReviewState.PENDING
        target = self._existing(target_capsule_id) if target_capsule_id else None
        decision = self.policy.authorize_merge(principal, pending, target, context)
        self._require(decision, principal=principal, capsule=staged, event="capsule.merge_requested")
        staged.merge_state = MergeState.STAGED
        staged.review_state = ReviewState.PENDING
        staged.updated_at = utc_now()
        staged.updated_by = principal
        staged.revision += 1
        self.repository.put(staged, expected_revision=staged.revision - 1)
        return self.merge_capsule(
            staged_capsule_id,
            principal=principal,
            context=context,
            request_id=request_id,
            target_capsule_id=target_capsule_id,
            expected_target_revision=expected_target_revision,
            expected_target_hash=expected_target_hash,
            strategy=strategy,
            decision_reason=decision_reason,
        )

    def delete_capsule(
        self,
        capsule_id: str,
        *,
        principal: MemoryPrincipal,
        context: CapsuleTaskContext,
        mode: DeleteMode = DeleteMode.SOFT,
    ) -> None:
        capsule = self._existing(capsule_id)
        decision = self.policy.authorize_delete(principal, capsule, context)
        self._require(decision, principal=principal, capsule=capsule, event="capsule.delete_denied")
        if callable(self.retention_hook) and not bool(self.retention_hook(capsule, mode)):
            raise CapsuleAuthorizationError(AuthorizationDecision(False, "retention_hold", "Administrative retention policy prevents deletion."))
        if mode is DeleteMode.PERMANENT:
            self.repository.remove(capsule_id)
        else:
            capsule.deleted_at = utc_now()
            if mode is DeleteMode.REDACT:
                capsule.content = {"redacted": True}
                capsule.summary = "[redacted]"
                capsule.content_hash = _content_hash(capsule.content)
            capsule.updated_at = utc_now()
            capsule.updated_by = principal
            capsule.revision += 1
            self.repository.put(capsule, expected_revision=capsule.revision - 1)
        self.audit.emit("capsule.deleted", principal=principal, capsule=capsule)

    def cleanup_expired(self, *, principal: MemoryPrincipal, permanent: bool = False) -> int:
        """Apply expiry cleanup without making expired records retrievable."""
        if not principal.has("memory.capsule.delete"):
            raise CapsuleAuthorizationError(AuthorizationDecision(False, "missing_capability", "Principal lacks memory.capsule.delete."))
        count = 0
        for capsule in self.repository.list():
            if not capsule.expired or capsule.deleted_at is not None:
                continue
            mode = DeleteMode.PERMANENT if permanent else DeleteMode.SOFT
            if callable(self.retention_hook) and not bool(self.retention_hook(capsule, mode)):
                continue
            if permanent:
                self.repository.remove(capsule.capsule_id)
            else:
                capsule.deleted_at = utc_now()
                capsule.updated_at = utc_now()
                capsule.updated_by = principal
                capsule.revision += 1
                self.repository.put(capsule, expected_revision=capsule.revision - 1)
            self.audit.emit("capsule.expired", principal=principal, capsule=capsule)
            count += 1
        return count

    def get_lineage(
        self,
        capsule_id: str,
        *,
        principal: MemoryPrincipal,
        context: CapsuleTaskContext,
    ) -> CapsuleLineage:
        capsule = self._existing(capsule_id)
        if not self.policy.authorize_read(principal, capsule, context).allowed:
            raise MemoryNotFoundError("Capsule was not found.")
        rows = self.repository.list()
        descendants = tuple(sorted(item.capsule_id for item in rows if capsule_id in item.source_capsule_ids))
        merges = tuple(record for record in self.repository.merges() if capsule_id in record.source_capsule_ids or record.target_capsule_id == capsule_id or record.resulting_capsule_id == capsule_id)
        approved = next((record.reviewed_by for record in reversed(merges) if record.resulting_capsule_id == capsule_id), None)
        consumers = tuple({key: row.get(key) for key in ("task_id", "agent_id", "revision", "timestamp")} for row in self.repository.access_records(capsule_id))
        return CapsuleLineage(
            capsule_id,
            tuple(capsule.source_capsule_ids),
            descendants,
            merges,
            consumers,
            capsule.superseded_by,
            approved,
            tuple(self.repository.revision_history(capsule_id)),
        )

    def migrate_legacy_records(self, records: list[dict[str, Any]]) -> int:
        """Quarantine legacy metadata; ownership is never inferred from text."""
        count = 0
        system = MemoryPrincipal(capabilities=frozenset({"memory.capsule.write.private"}))
        for row in records:
            now = utc_now()
            content = {"legacy_record_id": str(row.get("id") or ""), "migration_required": True}
            capsule = MemoryCapsule(
                capsule_id=uuid.uuid4().hex,
                schema_version=1,
                scope=CapsuleScope.PRIVATE,
                namespace=f"legacy/unscoped/{uuid.uuid4().hex}",
                owner_user_id=None, organisation_id=None, project_id=None, team_id=None,
                task_id="legacy-unmapped", parent_task_id=None, agent_id=None, session_id=None,
                title="Legacy memory awaiting mapping", summary="Ownership and scope were not trusted during migration.",
                content=content, tags=["legacy", "migration-required"], origin_type="legacy_memory",
                origin_id=str(row.get("id") or "unknown"), source_capsule_ids=[], trust_state=TrustState.QUARANTINED,
                review_state=ReviewState.PENDING, merge_state=MergeState.STAGED, created_at=now, updated_at=now,
                expires_at=None, created_by=system, updated_by=system, content_hash=_content_hash(content), revision=1,
                provider=str(row.get("provider") or "legacy"), risk_flags=["unverified_legacy_scope"],
            )
            self.repository.put(capsule)
            count += 1
        return count

    def _apply_merge(self, staged: MemoryCapsule, target: MemoryCapsule | None, strategy: MergeStrategy, principal: MemoryPrincipal, context: CapsuleTaskContext) -> MemoryCapsule:
        if target is None:
            result = copy.deepcopy(staged)
            result.capsule_id = uuid.uuid4().hex
            result.revision = 1
            result.created_at = utc_now()
        else:
            result = copy.deepcopy(target)
            result.revision = target.revision + 1
        if target is not None:
            if strategy is MergeStrategy.APPEND:
                result.content = {"entries": [target.content, staged.content]}
                result.summary = " ".join(filter(None, [target.summary, staged.summary]))[:2000]
            elif strategy is MergeStrategy.REPLACE:
                result.content = copy.deepcopy(staged.content)
                result.summary = staged.summary
            elif strategy is MergeStrategy.PATCH:
                result.content = {**target.content, **staged.content}
                result.summary = staged.summary or target.summary
            elif strategy is MergeStrategy.SUPERSEDE:
                result.content = copy.deepcopy(staged.content)
                result.summary = staged.summary
        result.scope = staged.proposed_scope or result.scope
        result.namespace = staged.proposed_namespace or result.namespace
        result.team_id = staged.team_id
        result.project_id = context.project_id
        result.organisation_id = context.organisation_id
        result.owner_user_id = context.user_id
        result.source_capsule_ids = list(dict.fromkeys([*result.source_capsule_ids, *staged.source_capsule_ids, staged.capsule_id]))
        result.trust_state = TrustState.APPROVED
        result.review_state = ReviewState.APPROVED
        result.merge_state = MergeState.MERGED
        result.updated_at = utc_now()
        result.updated_by = principal
        result.content_hash = _content_hash(result.content)
        return result

    def _record_conflict(self, staged: MemoryCapsule, target: MemoryCapsule | None, strategy: MergeStrategy, expected: int | None, expected_hash: str | None, principal: MemoryPrincipal, reason: str, request_id: str, correlation_id: str) -> CapsuleMergeRecord:
        staged.review_state = ReviewState.CONFLICT
        staged.merge_state = MergeState.CONFLICT
        staged.updated_at = utc_now()
        staged.updated_by = principal
        staged.revision += 1
        self.repository.put(staged, expected_revision=staged.revision - 1)
        record = replace(self._merge_record(staged, target, strategy, expected, expected_hash, None, principal, reason, request_id), conflict=True)
        self.repository.add_merge(record)
        self.audit.emit("capsule.merge_conflict", principal=principal, capsule=staged, correlation_id=correlation_id)
        return record

    @staticmethod
    def _merge_record(staged: MemoryCapsule, target: MemoryCapsule | None, strategy: MergeStrategy, expected: int | None, expected_hash: str | None, resulting: str | None, principal: MemoryPrincipal, reason: str, request_id: str) -> CapsuleMergeRecord:
        return CapsuleMergeRecord(uuid.uuid4().hex, tuple([*staged.source_capsule_ids, staged.capsule_id]), target.capsule_id if target else None, strategy, expected, expected_hash, resulting, principal, reason or None, utc_now(), request_id)

    def _record_decision(self, principal: MemoryPrincipal, capsule: MemoryCapsule, decision: AuthorizationDecision, *, correlation_id: str = "") -> None:
        if self.config.record_access_events:
            event = (
                "capsule.read_allowed" if decision.allowed
                else "capsule.expired" if decision.reason_code == "expired"
                else "capsule.read_denied"
            )
            self.audit.emit(event, principal=principal, capsule=capsule, decision_code=decision.reason_code, correlation_id=correlation_id)

    def _require(self, decision: AuthorizationDecision, *, principal: MemoryPrincipal, event: str, capsule: MemoryCapsule | None = None, correlation_id: str = "") -> None:
        if not decision.allowed:
            self.audit.emit(event, principal=principal, capsule=capsule, decision_code=decision.reason_code, correlation_id=correlation_id)
            raise CapsuleAuthorizationError(decision)

    def _existing(self, capsule_id: str | None) -> MemoryCapsule:
        capsule = self.repository.get(str(capsule_id or ""))
        if capsule is None:
            raise MemoryNotFoundError("Capsule was not found.")
        return capsule

    def _require_enabled(self) -> None:
        if not self.config.enabled:
            raise MemoryConfigurationError("Scoped memory capsules are disabled; no capsule action was executed.")

    def _retention_days(self, scope: CapsuleScope) -> int:
        retention = self.config.retention
        return {
            CapsuleScope.PRIVATE: retention.private_days,
            CapsuleScope.PARENT_CHILD: retention.parent_child_days,
            CapsuleScope.TEAM: retention.team_days,
            CapsuleScope.PROJECT: retention.project_days,
            CapsuleScope.ORGANISATION: retention.organisation_days,
            CapsuleScope.USER: retention.project_days,
        }[scope]

    @staticmethod
    def _projection(capsule: MemoryCapsule) -> CapsuleProjection:
        return CapsuleProjection(capsule.capsule_id, capsule.scope, capsule.namespace, capsule.title, capsule.summary, copy.deepcopy(capsule.content), tuple(capsule.tags), capsule.trust_state, capsule.origin_type, capsule.origin_id, tuple(capsule.source_capsule_ids), capsule.revision, capsule.content_hash, capsule.provider, capsule.created_at, capsule.expires_at)

    @staticmethod
    def _validate_content(title: str, summary: str, content: dict[str, Any]) -> None:
        if not str(title).strip() or not str(summary).strip() or not isinstance(content, dict):
            raise ValueError("Capsules require a title, compact summary, and structured object content.")
        encoded = json.dumps(content, sort_keys=True, ensure_ascii=False)
        if len(encoded.encode("utf-8")) > 64_000 or len(summary) > 2000:
            raise ValueError("Capsule content exceeds the compact storage limit.")

    @staticmethod
    def _tags(tags: list[str] | None) -> list[str]:
        return list(dict.fromkeys(str(item).strip().lower()[:64] for item in (tags or []) if str(item).strip()))[:32]

    @staticmethod
    def _risk_flags(title: str, summary: str, content: dict[str, Any]) -> list[str]:
        text = " ".join((title, summary, json.dumps(content, sort_keys=True)))
        return [f"prompt_injection_pattern_{index}" for index, pattern in enumerate(_INJECTION_PATTERNS, 1) if pattern.search(text)]
