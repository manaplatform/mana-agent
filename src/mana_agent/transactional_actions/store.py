from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .models import ActionIntent, ActionState, TransactionIntent

if os.name == "nt":  # pragma: no cover
    import msvcrt
else:  # pragma: no cover
    import fcntl


class ActionStore:
    """Crash-durable action, transaction, idempotency and audit persistence."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        for child in ("actions", "transactions", "idempotency", "audit"):
            (self.root / child).mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()
        self._lock_path = self.root / ".lock"
        self._lock_path.touch(exist_ok=True)

    @contextmanager
    def locked(self) -> Iterator[None]:
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

    @staticmethod
    def _write(path: Path, payload: dict) -> None:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, ensure_ascii=False, default=str)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _token(value: str) -> str:
        import hashlib
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def save_action(self, action: ActionIntent) -> None:
        with self.locked():
            self._write(self.root / "actions" / f"{action.action_id}.json", action.model_dump(mode="json"))

    def create_action(self, action: ActionIntent) -> None:
        with self.locked():
            path = self.root / "actions" / f"{action.action_id}.json"
            if path.exists():
                raise ValueError(f"action already exists: {action.action_id}")
            key_path = self.root / "idempotency" / f"{self._token(action.idempotency_key)}.json"
            if key_path.exists():
                record = json.loads(key_path.read_text(encoding="utf-8"))
                if record["intent_digest"] != action.intent_digest():
                    raise ValueError("idempotency key is already bound to a conflicting action")
                raise ValueError(f"duplicate action: {record['action_id']}")
            self._write(path, action.model_dump(mode="json"))
            self._write(key_path, {"action_id": action.action_id, "intent_digest": action.intent_digest()})

    def get_action(self, action_id: str) -> ActionIntent | None:
        path = self.root / "actions" / f"{action_id}.json"
        return ActionIntent.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else None

    def claim_execution(self, action_id: str) -> ActionIntent:
        """Atomically fence duplicate workers before any side effect occurs."""
        with self.locked():
            path = self.root / "actions" / f"{action_id}.json"
            if not path.is_file():
                raise LookupError("unknown action")
            current = ActionIntent.model_validate_json(path.read_text(encoding="utf-8"))
            if current.state is not ActionState.APPROVED:
                raise RuntimeError(f"action cannot be claimed from state {current.state.value}")
            current.transition(ActionState.EXECUTING)
            current.execution_attempts += 1
            self._write(path, current.model_dump(mode="json"))
            return current

    def action_for_idempotency_key(self, key: str) -> ActionIntent | None:
        path = self.root / "idempotency" / f"{self._token(key)}.json"
        if not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        return self.get_action(str(record["action_id"]))

    def release_idempotency(self, action: ActionIntent) -> None:
        """Release only an invalidated, non-committed binding for fresh evaluation."""
        if action.state not in {ActionState.FAILED, ActionState.EXPIRED, ActionState.CANCELLED}:
            raise ValueError("only failed, expired, or cancelled actions may release idempotency")
        path = self.root / "idempotency" / f"{self._token(action.idempotency_key)}.json"
        with self.locked():
            if not path.is_file():
                return
            record = json.loads(path.read_text(encoding="utf-8"))
            if str(record.get("action_id") or "") != action.action_id:
                raise ValueError("idempotency binding changed concurrently")
            path.unlink()

    def save_transaction(self, transaction: TransactionIntent) -> None:
        with self.locked():
            self._write(self.root / "transactions" / f"{transaction.transaction_id}.json", transaction.model_dump(mode="json"))

    def get_transaction(self, transaction_id: str) -> TransactionIntent | None:
        path = self.root / "transactions" / f"{transaction_id}.json"
        return TransactionIntent.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else None

    def append_audit(self, action: ActionIntent, event_type: str, details: dict | None = None) -> None:
        from mana_agent.utils.redaction import redact_secrets
        record = redact_secrets({
            "event_type": event_type,
            "action_id": action.action_id,
            "transaction_id": action.transaction_id,
            "parent_task_id": action.parent_task_id,
            "actor": action.actor,
            "originating_agent": action.originating_agent,
            "tool_name": action.tool_name,
            "operation_name": action.operation_name,
            "state": action.state.value,
            "state_version": action.state_version,
            "preview_digest": action.preview_digest,
            "policy_inputs": {
                "target_resources": action.target_resources,
                "normalized_arguments": action.normalized_arguments,
                "requested_capabilities": action.requested_capabilities,
                "expected_side_effects": action.expected_side_effects,
                "data_disclosure": action.data_disclosure.value,
                "blast_radius": action.blast_radius.value,
                "reversibility": action.reversibility.value,
            },
            "policy_decision": action.policy_decision.model_dump(mode="json") if action.policy_decision else None,
            "execution_attempts": action.execution_attempts,
            "created_at": action.created_at.isoformat(),
            "expires_at": action.expires_at.isoformat(),
            "verification": action.verification.model_dump(mode="json") if action.verification else None,
            "compensation": action.compensation.model_dump(mode="json") if action.compensation else None,
            "updated_at": action.updated_at.isoformat(),
            "details": details or {},
        })
        path = self.root / "audit" / "actions.jsonl"
        with self.locked(), path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
