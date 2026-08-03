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
    approved_by: str
    approved_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    consumed_at: datetime | None = None

    def valid_for(
        self,
        action: ActionIntent,
        *,
        transaction_binding_digest: str = "",
        now: datetime | None = None,
    ) -> bool:
        current = now or utc_now()
        return (
            self.consumed_at is None
            and action.state is ActionState.AWAITING_APPROVAL
            and self.expires_at > current
            and self.action_id == action.action_id
            and self.transaction_id == action.transaction_id
            and self.binding_digest == action.binding_digest()
            and self.preview_digest == action.preview_digest
            and action.policy_decision is not None
            and self.policy_fingerprint == action.policy_decision.policy_fingerprint
            and (
                self.scope is ApprovalScope.ACTION_ONCE
                or bool(
                    transaction_binding_digest
                    and self.transaction_binding_digest == transaction_binding_digest
                )
            )
        )


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
        grant = ApprovalGrant(
            action_id=action.action_id,
            transaction_id=action.transaction_id,
            binding_digest=action.binding_digest(),
            preview_digest=action.preview_digest,
            policy_fingerprint=action.policy_decision.policy_fingerprint,
            scope=scope,
            transaction_binding_digest=transaction_binding_digest,
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
            grant.consumed_at = utc_now()
            self._save_unlocked(grant)
            return grant

    def find_valid(
        self, action: ActionIntent, *, transaction_binding_digest: str = ""
    ) -> ApprovalGrant | None:
        for path in sorted(self.root.glob("*.json")):
            grant = ApprovalGrant.model_validate_json(path.read_text(encoding="utf-8"))
            if grant.valid_for(
                action, transaction_binding_digest=transaction_binding_digest
            ):
                return grant
        return None

    def invalidate_for_action(self, action_id: str) -> None:
        for path in self.root.glob("*.json"):
            grant = ApprovalGrant.model_validate_json(path.read_text(encoding="utf-8"))
            if grant.action_id == action_id and grant.consumed_at is None:
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
