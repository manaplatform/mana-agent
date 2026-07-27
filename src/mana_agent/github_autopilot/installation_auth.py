from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from .config import GitHubAutopilotSettings


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def create_app_jwt(app_id: str, private_key: bytes, *, now: int | None = None) -> str:
    issued = int(now if now is not None else time.time())
    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64url(json.dumps({"iat": issued - 60, "exp": issued + 540, "iss": app_id}, separators=(",", ":")).encode())
    signing_input = f"{header}.{payload}".encode("ascii")
    key = serialization.load_pem_private_key(private_key, password=None)
    signature = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{header}.{payload}.{_b64url(signature)}"


@dataclass(frozen=True)
class InstallationToken:
    value: str
    expires_at: float


class InstallationAuthenticator:
    def __init__(self, settings: GitHubAutopilotSettings, client: object) -> None:
        self.settings = settings
        self.client = client
        self._cache: dict[tuple[int, int], InstallationToken] = {}
        self._lock = threading.Lock()

    def token(self, installation_id: int, repository_id: int) -> str:
        key = (installation_id, repository_id)
        with self._lock:
            cached = self._cache.get(key)
            if cached and cached.expires_at - time.time() > 300:
                return cached.value
        if self.settings.private_key_path is None:
            raise RuntimeError("GitHub App private key is not configured")
        app_jwt = create_app_jwt(self.settings.app_id, self.settings.private_key_path.read_bytes())
        data = self.client.create_installation_token(installation_id, app_jwt, repository_id)
        token = str(data.get("token") or "")
        if not token:
            raise RuntimeError("GitHub did not return an installation token")
        expires = data.get("expires_at")
        try:
            from datetime import datetime
            expires_at = datetime.fromisoformat(str(expires).replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError):
            expires_at = time.time() + 3000
        with self._lock:
            self._cache[key] = InstallationToken(token, expires_at)
        return token

    def invalidate(self, installation_id: int, repository_id: int) -> None:
        with self._lock:
            self._cache.pop((installation_id, repository_id), None)
