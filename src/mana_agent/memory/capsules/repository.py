"""Trusted local capsule metadata store used with every memory provider."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from mana_agent.memory.capsules.models import (
    CapsuleMergeRecord,
    CapsuleScope,
    MemoryCapsule,
    MemoryPrincipal,
    MergeState,
    MergeStrategy,
    ReviewState,
    TrustState,
)
from mana_agent.workspaces.store import atomic_write_json


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (set, frozenset, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _principal(value: dict[str, Any] | None) -> MemoryPrincipal:
    data = dict(value or {})
    data["team_ids"] = frozenset(data.get("team_ids") or [])
    data["capabilities"] = frozenset(data.get("capabilities") or [])
    return MemoryPrincipal(**data)


def _capsule(row: dict[str, Any]) -> MemoryCapsule:
    data = dict(row)
    for key in ("scope", "proposed_scope"):
        if data.get(key):
            data[key] = CapsuleScope(data[key])
    for key, enum_type in (
        ("trust_state", TrustState),
        ("review_state", ReviewState),
        ("merge_state", MergeState),
        ("requested_operation", MergeStrategy),
    ):
        if data.get(key):
            data[key] = enum_type(data[key])
    for key in ("created_at", "updated_at", "expires_at", "deleted_at"):
        if data.get(key):
            data[key] = datetime.fromisoformat(data[key])
    data["created_by"] = _principal(data.get("created_by"))
    data["updated_by"] = _principal(data.get("updated_by"))
    return MemoryCapsule(**data)


def _merge(row: dict[str, Any]) -> CapsuleMergeRecord:
    data = dict(row)
    data.setdefault("expected_target_hash", None)
    data["source_capsule_ids"] = tuple(data.get("source_capsule_ids") or [])
    data["strategy"] = MergeStrategy(data["strategy"])
    data["created_at"] = datetime.fromisoformat(data["created_at"])
    if data.get("reviewed_by"):
        data["reviewed_by"] = _principal(data["reviewed_by"])
    return CapsuleMergeRecord(**data)


class CapsuleRepository:
    """Atomic JSON persistence with revision history and idempotency records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "capsules": {}, "history": {}, "merges": [], "access": []}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or int(payload.get("version", 0)) != 1:
            raise ValueError("unsupported capsule repository schema")
        payload.setdefault("capsules", {})
        payload.setdefault("history", {})
        payload.setdefault("merges", [])
        payload.setdefault("access", [])
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, payload)

    def put(self, capsule: MemoryCapsule, *, expected_revision: int | None = None) -> None:
        with self._lock:
            payload = self._load()
            current = payload["capsules"].get(capsule.capsule_id)
            current_revision = int(current.get("revision", 0)) if current else None
            if expected_revision is not None and current_revision != expected_revision:
                raise RevisionConflict(capsule.capsule_id, expected_revision, current_revision)
            if current:
                payload["history"].setdefault(capsule.capsule_id, []).append(current)
            payload["capsules"][capsule.capsule_id] = _jsonable(capsule)
            self._save(payload)

    def get(self, capsule_id: str) -> MemoryCapsule | None:
        with self._lock:
            row = self._load()["capsules"].get(capsule_id)
            return _capsule(row) if row else None

    def list(self) -> list[MemoryCapsule]:
        with self._lock:
            return [_capsule(row) for row in self._load()["capsules"].values()]

    def remove(self, capsule_id: str) -> bool:
        with self._lock:
            payload = self._load()
            row = payload["capsules"].pop(capsule_id, None)
            if row is None:
                return False
            payload["history"].setdefault(capsule_id, []).append(row)
            self._save(payload)
            return True

    def add_merge(self, record: CapsuleMergeRecord) -> None:
        with self._lock:
            payload = self._load()
            existing = next((row for row in payload["merges"] if row.get("request_id") == record.request_id), None)
            if existing:
                return
            payload["merges"].append(_jsonable(record))
            self._save(payload)

    def commit_merge(
        self,
        *,
        staged: MemoryCapsule,
        expected_staged_revision: int,
        result: MemoryCapsule | None,
        expected_target_revision: int | None,
        record: CapsuleMergeRecord,
    ) -> None:
        """Atomically persist review state, target mutation, and idempotency record."""
        with self._lock:
            payload = self._load()
            if any(row.get("request_id") == record.request_id for row in payload["merges"]):
                return
            current_staged = payload["capsules"].get(staged.capsule_id)
            actual_staged = int(current_staged.get("revision", 0)) if current_staged else None
            if actual_staged != expected_staged_revision:
                raise RevisionConflict(staged.capsule_id, expected_staged_revision, actual_staged)
            if result is not None and record.target_capsule_id:
                current_target = payload["capsules"].get(record.target_capsule_id)
                actual_target = int(current_target.get("revision", 0)) if current_target else None
                if actual_target != expected_target_revision:
                    raise RevisionConflict(record.target_capsule_id, int(expected_target_revision or 0), actual_target)
                payload["history"].setdefault(record.target_capsule_id, []).append(current_target)
            if current_staged:
                payload["history"].setdefault(staged.capsule_id, []).append(current_staged)
            if result is not None:
                payload["capsules"][result.capsule_id] = _jsonable(result)
            payload["capsules"][staged.capsule_id] = _jsonable(staged)
            payload["merges"].append(_jsonable(record))
            self._save(payload)

    def merge_for_request(self, request_id: str) -> CapsuleMergeRecord | None:
        with self._lock:
            row = next((item for item in self._load()["merges"] if item.get("request_id") == request_id), None)
            return _merge(row) if row else None

    def merges(self) -> list[CapsuleMergeRecord]:
        with self._lock:
            return [_merge(row) for row in self._load()["merges"]]

    def record_access(self, entry: dict[str, Any]) -> None:
        with self._lock:
            payload = self._load()
            payload["access"].append(_jsonable(entry))
            payload["access"] = payload["access"][-10_000:]
            self._save(payload)

    def access_records(self, capsule_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._load()["access"] if row.get("capsule_id") == capsule_id]

    def revision_history(self, capsule_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "revision": int(row.get("revision", 0)),
                    "content_hash": str(row.get("content_hash") or ""),
                    "updated_at": row.get("updated_at"),
                    "updated_by": {
                        key: (row.get("updated_by") or {}).get(key)
                        for key in ("user_id", "organisation_id", "project_id", "team_ids", "task_id", "parent_task_id", "agent_id")
                    },
                }
                for row in self._load()["history"].get(capsule_id, [])
            ]

    # Provider-neutral names keep the service portable without allowing callers
    # to skip authorization; CapsuleRepository is intentionally not exported by
    # the application-facing memory facade.
    def create_capsule(self, capsule: MemoryCapsule) -> None:
        self.put(capsule)

    def get_capsule(self, capsule_id: str) -> MemoryCapsule | None:
        return self.get(capsule_id)

    def query_capsules(self) -> list[MemoryCapsule]:
        return self.list()

    def update_capsule(self, capsule: MemoryCapsule, *, expected_revision: int) -> None:
        self.put(capsule, expected_revision=expected_revision)

    def stage_capsule(self, capsule: MemoryCapsule) -> None:
        self.put(capsule)

    def list_staged_capsules(self) -> list[MemoryCapsule]:
        return [item for item in self.list() if item.merge_state is MergeState.STAGED]

    def merge_capsule(self, record: CapsuleMergeRecord) -> None:
        self.add_merge(record)

    def delete_capsule(self, capsule_id: str) -> bool:
        return self.remove(capsule_id)

    def get_lineage(self, capsule_id: str) -> dict[str, Any]:
        capsule = self.get(capsule_id)
        return {
            "source_capsule_ids": list(capsule.source_capsule_ids) if capsule else [],
            "merges": [record for record in self.merges() if capsule_id in record.source_capsule_ids],
            "access": self.access_records(capsule_id),
            "revision_history": self.revision_history(capsule_id),
        }

    def migrate_legacy_local_identities(
        self,
        canonical_user_id: str,
        *,
        legacy_local_identities: set[str] | None = None,
    ) -> int:
        """Migrate locally owned legacy capsule records to canonical user identity.

        Only records demonstrably owned by this local installation (e.g. root,
        getpass.getuser(), or specified local OS accounts) are migrated.
        Preserves ACL isolation and provenance without runtime fallback authorization.
        """
        import getpass
        import os

        canonical = str(canonical_user_id or "").strip()
        if not canonical:
            return 0

        legacy_identities = set(legacy_local_identities or set())
        if not legacy_identities:
            try:
                os_user = str(getpass.getuser() or "").strip()
            except Exception:
                os_user = ""
            env_user = str(os.environ.get("USER") or os.environ.get("LOGNAME") or "").strip()
            legacy_identities = {"root"}
            if os_user:
                legacy_identities.add(os_user)
            if env_user:
                legacy_identities.add(env_user)

        legacy_identities.discard(canonical)
        legacy_identities.discard("")
        if not legacy_identities:
            return 0

        migrated_count = 0
        with self._lock:
            if not self.path.exists():
                return 0
            payload = self._load()
            capsules = payload.get("capsules", {})
            now = datetime.now(timezone.utc).isoformat()
            for cid, cap in capsules.items():
                owner = str(cap.get("owner_user_id") or "").strip()
                if owner in legacy_identities:
                    old_owner = owner
                    cap["owner_user_id"] = canonical
                    if isinstance(cap.get("created_by"), dict):
                        if cap["created_by"].get("user_id") in legacy_identities:
                            cap["created_by"]["user_id"] = canonical
                    if isinstance(cap.get("updated_by"), dict):
                        if cap["updated_by"].get("user_id") in legacy_identities:
                            cap["updated_by"]["user_id"] = canonical
                    cap["revision"] = int(cap.get("revision", 1)) + 1
                    cap["updated_at"] = now
                    evidence_list = list(cap.get("supporting_evidence") or [])
                    evidence_list.append(f"migrated_identity:{old_owner}->{canonical}")
                    cap["supporting_evidence"] = evidence_list
                    payload.setdefault("history", {}).setdefault(cid, []).append(
                        {
                            "event": "legacy_identity_migration",
                            "from_user_id": old_owner,
                            "to_user_id": canonical,
                            "timestamp": now,
                        }
                    )
                    migrated_count += 1
            if migrated_count > 0:
                self._save(payload)
        return migrated_count


class RevisionConflict(RuntimeError):
    def __init__(self, capsule_id: str, expected: int, actual: int | None) -> None:
        super().__init__(f"capsule {capsule_id!r} revision conflict: expected {expected}, found {actual}")
        self.capsule_id = capsule_id
        self.expected = expected
        self.actual = actual
