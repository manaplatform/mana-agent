from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

from pydantic import Field

from .models import ActionIntent, ActionState, ApprovalScope, StrictModel, utc_now
from .task_scope import (
    action_matches_task_grant,
    task_scope_id_for_action,
    task_wide_operations_for,
)

if os.name == "nt":  # pragma: no cover
    import msvcrt
else:  # pragma: no cover
    import fcntl


class ApprovalGrant(StrictModel):
    approval_id: str = Field(default_factory=lambda: f"approval_{secrets.token_urlsafe(18)}")
    action_id: str
    transaction_id: str = ""
    binding_digest: str
    preview_digest: str
    policy_fingerprint: str
    scope: ApprovalScope = ApprovalScope.ACTION_ONCE
    transaction_binding_digest: str = ""
    # Task-wide multi-use fields (scope=TASK only).
    task_scope_id: str = ""
    allowed_tool_name: str = ""
    allowed_permission_scopes: list[str] = Field(default_factory=list)
    allowed_operations: list[str] = Field(default_factory=list)
    approved_by: str
    approved_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    consumed_at: datetime | None = None
    last_used_at: datetime | None = None
    use_count: int = Field(default=0, ge=0)

    def valid_for(
        self,
        action: ActionIntent,
        *,
        transaction_binding_digest: str = "",
        now: datetime | None = None,
    ) -> bool:
        current = now or utc_now()
        if self.consumed_at is not None or self.expires_at <= current:
            return False
        if action.state is not ActionState.AWAITING_APPROVAL:
            return False
        if action.policy_decision is None:
            return False
        if self.policy_fingerprint != action.policy_decision.policy_fingerprint:
            return False

        if self.scope is ApprovalScope.TASK:
            return action_matches_task_grant(self, action)

        if (
            self.action_id != action.action_id
            or self.transaction_id != action.transaction_id
            or self.binding_digest != action.binding_digest()
            or self.preview_digest != action.preview_digest
        ):
            return False

        if self.scope is ApprovalScope.ACTION_ONCE:
            return True
        if self.scope is ApprovalScope.TRANSACTION:
            return bool(
                transaction_binding_digest
                and self.transaction_binding_digest == transaction_binding_digest
            )
        return False


class ApprovalRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()
        self._lock_path = self.root / ".lock"
        self._lock_path.touch(exist_ok=True)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock, self._lock_path.open("r+b") as handle:
            if os.name == "nt":  # pragma: no cover
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":  # pragma: no cover
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def issue(
        self,
        action: ActionIntent,
        *,
        approved_by: str,
        ttl_seconds: int = 300,
        scope: ApprovalScope = ApprovalScope.ACTION_ONCE,
        transaction_binding_digest: str = "",
    ) -> ApprovalGrant:
        if action.policy_decision is None or not action.preview_digest:
            raise ValueError("action must have a policy decision and preview before approval")
        if action.state is not ActionState.AWAITING_APPROVAL:
            raise ValueError("only an action awaiting approval may receive a grant")
        if scope is ApprovalScope.TRANSACTION and not action.transaction_id:
            raise ValueError("transaction approval requires transaction membership")
        if (
            scope is ApprovalScope.TRANSACTION
            and action.policy_decision.required_approval_scope is not ApprovalScope.TRANSACTION
        ):
            raise PermissionError("policy does not allow transaction-scoped approval")
        if scope is ApprovalScope.TRANSACTION and not transaction_binding_digest:
            raise ValueError("transaction approval requires the exact transaction binding")
        task_scope_id = ""
        allowed_tool_name = ""
        allowed_permission_scopes: list[str] = []
        allowed_operations: list[str] = []
        if scope is ApprovalScope.TASK:
            task_scope_id = task_scope_id_for_action(action)
            if not task_scope_id:
                raise ValueError("task approval requires a durable task lineage identity")
            if action.policy_decision.required_approval_scope is not ApprovalScope.TASK:
                raise PermissionError("policy does not allow task-scoped approval")
            allowed_tool_name = action.tool_name
            allowed_permission_scopes = list(action.requested_capabilities)
            allowed_operations = task_wide_operations_for(action)
        grant = ApprovalGrant(
            action_id=action.action_id,
            transaction_id=action.transaction_id,
            binding_digest=action.binding_digest(),
            preview_digest=action.preview_digest,
            policy_fingerprint=action.policy_decision.policy_fingerprint,
            scope=scope,
            transaction_binding_digest=transaction_binding_digest,
            task_scope_id=task_scope_id,
            allowed_tool_name=allowed_tool_name,
            allowed_permission_scopes=allowed_permission_scopes,
            allowed_operations=allowed_operations,
            approved_by=approved_by,
            expires_at=min(action.expires_at, utc_now() + timedelta(seconds=max(1, ttl_seconds))),
        )
        self._save(grant)
        return grant

    def consume(
        self,
        approval_id: str,
        action: ActionIntent,
        *,
        transaction_binding_digest: str = "",
    ) -> ApprovalGrant:
        with self._locked():
            grant = self._get_unlocked(approval_id)
            if grant is None or not grant.valid_for(
                action, transaction_binding_digest=transaction_binding_digest
            ):
                raise PermissionError("approval is missing, expired, consumed, or not bound to this exact action")
            if grant.scope is ApprovalScope.TASK:
                # Multi-use: record usage but keep the grant valid for later
                # compatible actions under the same durable task lineage.
                grant.last_used_at = utc_now()
                grant.use_count = int(grant.use_count) + 1
            else:
                grant.consumed_at = utc_now()
            self._save_unlocked(grant)
            return grant

    def find_valid(
        self, action: ActionIntent, *, transaction_binding_digest: str = ""
    ) -> ApprovalGrant | None:
        # Prefer exact single-action grants, then multi-use task grants.
        action_once: ApprovalGrant | None = None
        task_grant: ApprovalGrant | None = None
        for path in sorted(self.root.glob("*.json")):
            grant = ApprovalGrant.model_validate_json(path.read_text(encoding="utf-8"))
            if not grant.valid_for(
                action, transaction_binding_digest=transaction_binding_digest
            ):
                continue
            if grant.scope is ApprovalScope.TASK:
                if task_grant is None:
                    task_grant = grant
            elif action_once is None:
                action_once = grant
        return action_once or task_grant

    def invalidate_for_action(self, action_id: str) -> None:
        for path in self.root.glob("*.json"):
            grant = ApprovalGrant.model_validate_json(path.read_text(encoding="utf-8"))
            if grant.action_id == action_id and grant.consumed_at is None:
                grant.consumed_at = utc_now()
                self._save(grant)

    def invalidate_for_task_scope(self, task_scope_id: str) -> None:
        """Invalidate multi-use task grants when a durable task ends or is cancelled."""
        scope = str(task_scope_id or "").strip()
        if not scope:
            return
        for path in self.root.glob("*.json"):
            grant = ApprovalGrant.model_validate_json(path.read_text(encoding="utf-8"))
            if (
                grant.scope is ApprovalScope.TASK
                and grant.task_scope_id == scope
                and grant.consumed_at is None
            ):
                grant.consumed_at = utc_now()
                self._save(grant)

    def get(self, approval_id: str) -> ApprovalGrant | None:
        with self._locked():
            return self._get_unlocked(approval_id)

    def _get_unlocked(self, approval_id: str) -> ApprovalGrant | None:
        path = self.root / f"{_safe_name(approval_id)}.json"
        return ApprovalGrant.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _save(self, grant: ApprovalGrant) -> None:
        with self._locked():
            self._save_unlocked(grant)

    def _save_unlocked(self, grant: ApprovalGrant) -> None:
        path = self.root / f"{_safe_name(grant.approval_id)}.json"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=self.root
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(grant.model_dump(mode="json"), stream, sort_keys=True, default=str)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _safe_name(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
