"""Short-lived, single-purpose signed response tokens."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from pydantic import Field

from .models import ResponseOperation, StrictModel


if os.name == "nt":  # pragma: no cover - exercised on Windows CI
    import msvcrt
else:  # pragma: no cover - platform branch
    import fcntl


_secret_path_locks_guard = threading.Lock()
_secret_path_locks: dict[Path, Any] = {}


def _thread_lock_for_secret_path(path: Path) -> Any:
    with _secret_path_locks_guard:
        return _secret_path_locks.setdefault(path, threading.RLock())


@contextmanager
def _secret_creation_lock(secret_path: Path) -> Iterator[None]:
    """Serialize signing-key creation for all local and sibling processes."""
    lock_path = secret_path.with_name(f".{secret_path.name}.lock")
    with _thread_lock_for_secret_path(secret_path):
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "r+b") as handle:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                handle.seek(0)
                if not handle.read(1):
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:  # pragma: no cover - platform branch
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - platform branch
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class ResponseTokenClaims(StrictModel):
    inbox_item_id: str
    reviewer_scope: str
    operation: ResponseOperation
    expires_at: datetime
    nonce: str = Field(min_length=16)
    token_version: int = 1


class ResponseTokenSigner:
    def __init__(
        self,
        secret_path: Path,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.secret_path = secret_path.expanduser().resolve()
        self.secret_path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self._secret: bytes | None = None

    def _signing_secret(self) -> bytes:
        if self._secret is None:
            self._secret = self._load_or_create_secret()
        return self._secret

    def _load_or_create_secret(self) -> bytes:
        with _secret_creation_lock(self.secret_path):
            if self.secret_path.is_file():
                return self._read_secret()
            value = secrets.token_bytes(32)
            candidate_path = self.secret_path.with_name(
                f".{self.secret_path.name}.{secrets.token_hex(16)}.tmp"
            )
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            try:
                descriptor = os.open(candidate_path, flags, 0o600)
            except FileExistsError as exc:
                raise RuntimeError("unable to create human inbox signing key candidate") from exc
            try:
                try:
                    offset = 0
                    while offset < len(value):
                        written = os.write(descriptor, value[offset:])
                        if written <= 0:
                            raise RuntimeError("unable to write human inbox signing key candidate")
                        offset += written
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                # The creation lock guarantees that no sibling signer can publish
                # a key between the existence check above and this replacement.
                # ``os.replace`` is atomic on Windows and avoids relying on a
                # hard-link publication step there. Reload the published file so
                # every signer caches the exact durable secret.
                os.replace(candidate_path, self.secret_path)
                return self._read_secret()
            finally:
                candidate_path.unlink(missing_ok=True)

    def _read_secret(self) -> bytes:
        value = self.secret_path.read_bytes()
        if len(value) < 32:
            raise RuntimeError("human inbox signing key is invalid")
        return value

    def issue(
        self,
        *,
        inbox_item_id: str,
        reviewer_scope: str,
        operation: ResponseOperation,
        expires_at: datetime,
        ttl_seconds: int = 900,
    ) -> tuple[str, str]:
        deadline = min(expires_at, self.clock() + timedelta(seconds=max(1, ttl_seconds)))
        claims = ResponseTokenClaims(
            inbox_item_id=inbox_item_id,
            reviewer_scope=reviewer_scope,
            operation=operation,
            expires_at=deadline,
            nonce=secrets.token_urlsafe(24),
        )
        body = _b64encode(json.dumps(claims.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode("utf-8"))
        signature = _b64encode(hmac.new(self._signing_secret(), body.encode("ascii"), hashlib.sha256).digest())
        token = f"v1.{body}.{signature}"
        return token, self.nonce_hash(claims.nonce)

    def verify(self, token: str) -> ResponseTokenClaims:
        try:
            version, body, supplied_signature = token.split(".", 2)
        except ValueError as exc:
            raise PermissionError("response token is malformed") from exc
        if version != "v1":
            raise PermissionError("response token version is unsupported")
        expected_signature = _b64encode(hmac.new(self._signing_secret(), body.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(supplied_signature, expected_signature):
            raise PermissionError("response token signature is invalid")
        try:
            claims = ResponseTokenClaims.model_validate_json(_b64decode(body))
        except (ValueError, TypeError) as exc:
            raise PermissionError("response token claims are invalid") from exc
        if claims.expires_at <= self.clock():
            raise PermissionError("response token expired")
        return claims

    @staticmethod
    def nonce_hash(nonce: str) -> str:
        return hashlib.sha256(nonce.encode("utf-8")).hexdigest()

    def response_signature(self, token: str) -> str:
        return "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()

    def protected_digest(self, value: object) -> str:
        """Bind secret-bearing idempotency input without persisting its plaintext."""
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        return "hmac-sha256:" + hmac.new(
            self._signing_secret(),
            b"response-idempotency:" + encoded,
            hashlib.sha256,
        ).hexdigest()

    def csrf_token(self, token: str) -> str:
        return _b64encode(hmac.new(self._signing_secret(), b"csrf:" + token.encode("utf-8"), hashlib.sha256).digest())

    def verify_csrf(self, token: str, supplied: str) -> bool:
        return hmac.compare_digest(self.csrf_token(token), supplied)
