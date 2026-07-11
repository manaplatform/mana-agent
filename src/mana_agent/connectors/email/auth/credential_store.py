"""Secret references backed by the operating-system keyring.

No fallback writes token material to Mana configuration or repository paths.
Headless users must install/configure a keyring backend explicitly.
"""
from __future__ import annotations
import json
from uuid import uuid4
from mana_agent.connectors.email.exceptions import AuthenticationRequired

SERVICE = "mana-agent.email"

class CredentialStore:
    def _keyring(self):
        try:
            import keyring
            return keyring
        except ImportError as exc:
            raise AuthenticationRequired("Email credentials require the optional 'email' extra (keyring).") from exc
    def put(self, payload: dict[str, object], *, reference: str | None = None) -> str:
        ref = reference or f"email-{uuid4().hex}"
        self._keyring().set_password(SERVICE, ref, json.dumps(payload, sort_keys=True))
        return ref
    def get(self, reference: str) -> dict[str, object]:
        raw = self._keyring().get_password(SERVICE, reference)
        if not raw: raise AuthenticationRequired("Email credentials are unavailable; reconnect the account.")
        try:
            value = json.loads(raw)
        except ValueError as exc: raise AuthenticationRequired("Stored email credentials are invalid; reconnect the account.") from exc
        if not isinstance(value, dict): raise AuthenticationRequired("Stored email credentials are invalid; reconnect the account.")
        return value
    def delete(self, reference: str) -> None: self._keyring().delete_password(SERVICE, reference)
