"""Worker identity storage with Keychain support and owner-only fallback files."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


@dataclass(frozen=True)
class WorkerIdentity:
    worker_id: str
    credential: str
    private_key_pem: str

    @property
    def public_key_pem(self) -> str:
        key = serialization.load_pem_private_key(self.private_key_pem.encode(), password=None)
        return key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()


def generate_identity(worker_id: str, credential: str = "") -> WorkerIdentity:
    key = Ed25519PrivateKey.generate()
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    return WorkerIdentity(worker_id=worker_id, credential=credential, private_key_pem=private)


class CredentialStore:
    """Uses macOS Keychain when available; the fallback is deliberately narrow."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "identity.json"

    def save(self, identity: WorkerIdentity) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        payload = json.dumps({"worker_id": identity.worker_id, "credential": identity.credential,
                              "private_key_pem": identity.private_key_pem}, separators=(",", ":"))
        # Keyring is optional; a failure must not result in credentials in a plist
        # or command line.  The owner-only state file is the cross-platform backend.
        try:
            import keyring  # type: ignore
            keyring.set_password("ManaAgentWorker", identity.worker_id, base64.b64encode(payload.encode()).decode())
            self.path.unlink(missing_ok=True)
            return
        except Exception:
            pass
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.chmod(self.path, 0o600)

    def load(self, worker_id: str | None = None) -> WorkerIdentity | None:
        if worker_id:
            try:
                import keyring  # type: ignore
                raw = keyring.get_password("ManaAgentWorker", worker_id)
                if raw:
                    data = json.loads(base64.b64decode(raw).decode())
                    return WorkerIdentity(**data)
            except Exception:
                pass
        if not self.path.exists():
            return None
        if self.path.stat().st_mode & 0o077:
            raise PermissionError("worker identity file must not be group/world accessible")
        return WorkerIdentity(**json.loads(self.path.read_text(encoding="utf-8")))

    def delete(self, worker_id: str | None = None) -> None:
        if worker_id:
            try:
                import keyring  # type: ignore
                keyring.delete_password("ManaAgentWorker", worker_id)
            except Exception:
                pass
        self.path.unlink(missing_ok=True)
