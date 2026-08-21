"""Resource-aware Codex provider primitives.

Credentials are references to an external secret backend (normally an
environment variable or OS keyring); this module never persists secret values.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from mana_agent.config.settings import mana_home


class CodexExecutionMode(str, Enum):
    API = "api"
    SUBSCRIPTION = "subscription"


class CredentialKind(str, Enum):
    API = "codex_api"
    SUBSCRIPTION = "codex_subscription"


@dataclass(frozen=True, slots=True)
class CodexCredential:
    kind: CredentialKind
    reference: str
    account_identity: str = ""
    expires_at: float | None = None
    refresh_state: str = "unknown"
    authenticated: bool = False
    schema_version: int = 1

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= time.time()


class CodexCredentialStore:
    """Persist metadata only; secret material stays in the referenced backend."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or mana_home() / "credentials").resolve()
        self.path = self.root / "codex.json"

    def save(self, credential: CodexCredential) -> CodexCredential:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        payload = {"schema_version": 1, "credentials": {credential.kind.value: asdict(credential)}}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        tmp.chmod(0o600)
        tmp.replace(self.path)
        return credential

    def load(self, kind: CredentialKind) -> CodexCredential | None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            value = raw.get("credentials", {}).get(kind.value)
            if not isinstance(value, dict):
                return None
            value = dict(value)
            value["kind"] = CredentialKind(value["kind"])
            return CodexCredential(**value)
        except (OSError, ValueError, TypeError):
            return None


@dataclass(frozen=True, slots=True)
class CodexUsage:
    mode: CodexExecutionMode
    authenticated: bool
    account_identity: str = ""
    tokens_consumed: int = 0
    estimated_cost_usd: float | None = None
    quota_consumed: float = 0.0
    quota_remaining: float | None = None
    quota_capacity: float | None = None
    reset_at: float | None = None
    billing_status: str = "unknown"
    expires_at: float | None = None

    def __post_init__(self) -> None:
        if self.mode is CodexExecutionMode.SUBSCRIPTION and self.estimated_cost_usd is not None:
            raise ValueError("subscription usage cannot expose a dollar cost")

    @property
    def available(self) -> bool:
        return self.authenticated and (self.expires_at is None or self.expires_at > time.time()) and (self.quota_remaining is None or self.quota_remaining > 0)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"mode": self.mode.value, "available": self.available}


class CodexUsageStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or mana_home() / "usage").resolve()
        self.path = self.root / "codex.json"

    def save(self, usage: CodexUsage) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"schema_version": 1, "usage": usage.as_dict()}, sort_keys=True), encoding="utf-8")
        self.path.chmod(0o600)

    def load(self, mode: CodexExecutionMode) -> CodexUsage | None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8")).get("usage", {})
            if raw.get("mode") != mode.value:
                return None
            if raw.get("expires_at") is not None and float(raw["expires_at"]) <= time.time():
                return None
            values = {key: value for key, value in raw.items() if key in CodexUsage.__dataclass_fields__}
            values["mode"] = CodexExecutionMode(values["mode"])
            return CodexUsage(**values)
        except (OSError, ValueError, TypeError, KeyError):
            return None


@dataclass(frozen=True, slots=True)
class CodexPolicy:
    allowed_tasks: Mapping[CodexExecutionMode, frozenset[str]] = field(default_factory=lambda: {
        CodexExecutionMode.API: frozenset({"automation", "background_jobs", "ci"}),
        CodexExecutionMode.SUBSCRIPTION: frozenset({"coding", "refactor", "review"}),
    })

    def allows(self, mode: CodexExecutionMode, task_type: str) -> bool:
        return task_type in self.allowed_tasks.get(mode, frozenset())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "CodexPolicy":
        raw = value or {}
        result: dict[CodexExecutionMode, frozenset[str]] = {}
        for mode in CodexExecutionMode:
            tasks = raw.get(mode.value, {}) if isinstance(raw, Mapping) else {}
            result[mode] = frozenset(str(item) for item in (tasks.get("allowed_tasks", ()) if isinstance(tasks, Mapping) else ()))
        defaults = cls().allowed_tasks
        return cls({mode: result[mode] or defaults[mode] for mode in CodexExecutionMode})


class CodexProvider:
    """Provider facade owning resource selection while reusing the legacy backend."""

    name = "codex"

    def __init__(self, backend: Any, *, mode: CodexExecutionMode, credentials: CodexCredentialStore | None = None, usage: CodexUsageStore | None = None) -> None:
        self.backend = backend
        self.mode = mode
        self.credentials = credentials or CodexCredentialStore()
        self.usage = usage or CodexUsageStore()

    @property
    def credential(self) -> CodexCredential | None:
        return self.credentials.load(CredentialKind.API if self.mode is CodexExecutionMode.API else CredentialKind.SUBSCRIPTION)

    @property
    def resource_usage(self) -> CodexUsage | None:
        return self.usage.load(self.mode)

    @property
    def available(self) -> bool:
        value = self.resource_usage
        return bool(value and value.available)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)


def choose_codex_mode(usages: Mapping[CodexExecutionMode, CodexUsage], task_type: str, policy: CodexPolicy = CodexPolicy()) -> CodexExecutionMode:
    """Select an available resource source; callers must supply fresh usage."""
    subscription = usages.get(CodexExecutionMode.SUBSCRIPTION)
    if subscription and subscription.available and policy.allows(CodexExecutionMode.SUBSCRIPTION, task_type):
        return CodexExecutionMode.SUBSCRIPTION
    api = usages.get(CodexExecutionMode.API)
    if api and api.available and policy.allows(CodexExecutionMode.API, task_type):
        return CodexExecutionMode.API
    raise RuntimeError("No authenticated Codex resource satisfies the task policy and quota")


def codex_resource_availability(provider: CodexProvider, task_type: str, policy: CodexPolicy = CodexPolicy()) -> tuple[bool, str]:
    """Adapter for ModelRouter's resource gate."""
    if not policy.allows(provider.mode, task_type):
        return False, f"Codex {provider.mode.value} mode is disallowed for task type {task_type!r}"
    if not provider.available:
        return False, f"Codex {provider.mode.value} authentication or quota is unavailable"
    return True, ""


__all__ = ["CodexCredential", "CodexCredentialStore", "CodexExecutionMode", "CodexPolicy", "CodexProvider", "CodexUsage", "CodexUsageStore", "CredentialKind", "choose_codex_mode", "codex_resource_availability"]
