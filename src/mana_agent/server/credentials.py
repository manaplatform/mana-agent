"""Credential references for server transports; secret values never enter models."""

from __future__ import annotations

from pathlib import Path

from mana_agent.config.user_config import load_user_config, save_user_config


class ServerCredentialResolver:
    """Resolve only approved credential metadata required by the transport."""

    def resolve_key_path(self, credential_ref: str) -> str:
        if not credential_ref.startswith("secret://server/"):
            raise ValueError("SSH key credentials must use secret://server/<id> references.")
        credential_id = credential_ref.removeprefix("secret://server/")
        config = load_user_config()
        server_credentials = config.get("server_credentials", {})
        payload = server_credentials.get(credential_id) if isinstance(server_credentials, dict) else None
        if not isinstance(payload, dict) or payload.get("type") != "ssh_key_path":
            raise LookupError(f"Server credential reference {credential_ref!r} is unavailable.")
        path = Path(str(payload.get("path") or "")).expanduser()
        if not path.is_file():
            raise ValueError("The authorized SSH key path does not exist or is not a regular file.")
        return str(path)

    def require_external_secret(self, credential_ref: str) -> None:
        if not credential_ref.startswith("secret://"):
            raise ValueError("Credentials must be referenced through Mana secret storage.")
        raise NotImplementedError(
            "Password and token SSH authentication require a configured secret-provider adapter; "
            "no plaintext or fallback credential was used."
        )


def register_key_path(credential_id: str, path: str) -> str:
    """Persist an authorized key path, never the private-key contents."""
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise ValueError("The authorized SSH key path does not exist or is not a regular file.")
    config = load_user_config()
    credentials = config.get("server_credentials", {})
    if not isinstance(credentials, dict):
        credentials = {}
    credentials[credential_id] = {"type": "ssh_key_path", "path": str(resolved)}
    config["server_credentials"] = credentials
    save_user_config(config, merge=False)
    return f"secret://server/{credential_id}"
