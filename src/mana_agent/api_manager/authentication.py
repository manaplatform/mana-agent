"""Credential-reference resolution and request authentication."""

from __future__ import annotations

import base64
import os
from typing import Protocol

from mana_agent.api_manager.errors import MissingCredentialError, RequestValidationError
from mana_agent.api_manager.models import AuthenticationConfig, AuthenticationType
from mana_agent.config.user_config import load_user_secrets


class CredentialResolver(Protocol):
    def resolve(self, reference: str) -> str: ...


class EnvironmentCredentialResolver:
    """Resolve an explicit env:// or mana-secret:// credential reference."""

    def resolve(self, reference: str) -> str:
        normalized = str(reference or "").strip()
        if not normalized:
            raise MissingCredentialError("A credential reference is required.")
        if normalized.startswith("env://"):
            credential_id = normalized.removeprefix("env://")
            value = os.environ.get(credential_id)
        elif normalized.startswith("mana-secret://"):
            credential_id = normalized.removeprefix("mana-secret://")
            raw = load_user_secrets().get(credential_id)
            value = str(raw) if raw is not None else None
        else:
            raise MissingCredentialError(
                "Credential references must use env://<name> or mana-secret://<id>."
            )
        if value is None:
            raise MissingCredentialError(
                f"Credential reference {normalized!r} is unavailable.",
                details={"credential_reference": normalized},
            )
        return value


def apply_authentication(
    authentication: AuthenticationConfig,
    *,
    credential_reference: str,
    headers: dict[str, str],
    query: list[tuple[str, str]],
    resolver: CredentialResolver,
) -> tuple[dict[str, str], list[tuple[str, str]], tuple[str, ...]]:
    if authentication.type is AuthenticationType.NONE:
        return headers, query, ()
    reference = credential_reference or authentication.credential_reference
    secret = resolver.resolve(reference)
    updated_headers = dict(headers)
    updated_query = list(query)
    if authentication.type is AuthenticationType.API_KEY_HEADER:
        updated_headers[authentication.parameter_name or "X-API-Key"] = secret
    elif authentication.type is AuthenticationType.API_KEY_QUERY:
        if not authentication.parameter_name:
            raise RequestValidationError("API-key query authentication is missing its parameter name.")
        updated_query.append((authentication.parameter_name, secret))
    elif authentication.type in {AuthenticationType.BEARER, AuthenticationType.OAUTH2}:
        updated_headers["Authorization"] = f"Bearer {secret}"
    elif authentication.type is AuthenticationType.BASIC:
        if ":" not in secret:
            raise MissingCredentialError(
                "Basic-auth credentials must resolve to a 'username:password' value."
            )
        encoded = base64.b64encode(secret.encode("utf-8")).decode("ascii")
        updated_headers["Authorization"] = f"Basic {encoded}"
    elif authentication.type is AuthenticationType.CUSTOM_HEADERS:
        values = secret.splitlines()
        names = authentication.custom_header_names
        if len(values) != len(names):
            raise MissingCredentialError(
                "Custom-header credential value count does not match the configured header names."
            )
        updated_headers.update(zip(names, values))
    else:
        raise RequestValidationError(f"Unsupported authentication type: {authentication.type.value}")
    return updated_headers, updated_query, (secret,)
