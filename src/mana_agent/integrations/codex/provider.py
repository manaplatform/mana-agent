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
from typing import Any, Callable, Mapping, Protocol

from mana_agent.config.settings import mana_home
from mana_agent.model_routing.models import RoutingResourceScore


class CodexExecutionMode(str, Enum):
    API = "api"
    SUBSCRIPTION = "subscription"


class CredentialKind(str, Enum):
    API = "codex_api"
    SUBSCRIPTION = "codex_subscription"


class CodexAuthenticationStatus(str, Enum):
    AUTHENTICATED = "authenticated"
    EXPIRED = "expired"
    INVALID = "invalid"
    REVOKED = "revoked"
    MISSING = "missing"


class CodexIdentityError(RuntimeError):
    """Normalized error from the Luna identity boundary."""


@dataclass(frozen=True, slots=True)
class CodexAccountStatus:
    status: CodexAuthenticationStatus
    account_identity: str = ""
    expires_at: float | None = None
    refresh_state: str = "unknown"
    reason: str = ""

    @property
    def authenticated(self) -> bool:
        return self.status is CodexAuthenticationStatus.AUTHENTICATED


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
        existing: dict[str, Any] = {}
        try:
            existing = json.loads(self.path.read_text(encoding="utf-8")).get("credentials", {})
        except (OSError, ValueError, TypeError):
            pass
        payload = {"schema_version": 1, "credentials": {**existing, credential.kind.value: asdict(credential)}}
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

    def revoke(self, kind: CredentialKind) -> None:
        current = self.load(kind)
        if current is not None:
            self.save(CodexCredential(**{**asdict(current), "authenticated": False, "refresh_state": "revoked"}))


class CodexIdentityClient(Protocol):
    """Luna identity boundary; implementations own OAuth/device-flow transport."""

    def login(self, reference: str, *, account_identity: str = "") -> CodexCredential: ...
    def validate(self, credential: CodexCredential) -> CodexCredential: ...
    def refresh(self, credential: CodexCredential) -> CodexCredential: ...
    def revoke(self, credential: CodexCredential) -> None: ...


class CodexAuthenticationService:
    def __init__(self, store: CodexCredentialStore | None = None, identity: CodexIdentityClient | None = None) -> None:
        self.store = store or CodexCredentialStore()
        self.identity = identity

    def login(self, reference: str, *, account_identity: str = "") -> CodexCredential:
        if self.identity is None:
            raise RuntimeError("Codex subscription authentication requires a Luna identity client.")
        credential = self.identity.login(reference, account_identity=account_identity)
        if credential.kind is not CredentialKind.SUBSCRIPTION or not credential.authenticated:
            raise RuntimeError("Luna returned an invalid Codex subscription session.")
        return self.store.save(credential)

    def status(self) -> CodexAccountStatus:
        credential = self.store.load(CredentialKind.SUBSCRIPTION)
        if credential is None:
            return CodexAccountStatus(CodexAuthenticationStatus.MISSING)
        if credential.refresh_state == "revoked":
            return CodexAccountStatus(CodexAuthenticationStatus.REVOKED, credential.account_identity, credential.expires_at, credential.refresh_state)
        if credential.expired:
            return CodexAccountStatus(CodexAuthenticationStatus.EXPIRED, credential.account_identity, credential.expires_at, credential.refresh_state, "session expired")
        if not credential.authenticated:
            return CodexAccountStatus(CodexAuthenticationStatus.INVALID, credential.account_identity, credential.expires_at, credential.refresh_state, "session is not authenticated")
        return CodexAccountStatus(CodexAuthenticationStatus.AUTHENTICATED, credential.account_identity, credential.expires_at, credential.refresh_state)

    def validate(self) -> CodexAccountStatus:
        credential = self.store.load(CredentialKind.SUBSCRIPTION)
        if credential is None or self.identity is None:
            return self.status()
        try:
            validated = self.identity.validate(credential)
            if validated.kind is not CredentialKind.SUBSCRIPTION or not validated.authenticated:
                raise ValueError("identity rejected the subscription session")
            return CodexAccountStatus(
                CodexAuthenticationStatus.AUTHENTICATED,
                self.store.save(validated).account_identity,
                validated.expires_at,
            )
        except (OSError, RuntimeError, ValueError):
            return CodexAccountStatus(CodexAuthenticationStatus.INVALID, credential.account_identity, credential.expires_at, "invalid", "identity validation failed")

    def refresh_if_needed(self) -> CodexAccountStatus:
        credential = self.store.load(CredentialKind.SUBSCRIPTION)
        if credential is None or self.identity is None or not credential.expired:
            return self.status()
        try:
            refreshed = self.identity.refresh(credential)
            if not refreshed.authenticated:
                return CodexAccountStatus(CodexAuthenticationStatus.INVALID, credential.account_identity, credential.expires_at, "invalid", "refresh rejected")
            self.store.save(refreshed)
            return self.status()
        except (OSError, RuntimeError, ValueError):
            return CodexAccountStatus(CodexAuthenticationStatus.EXPIRED, credential.account_identity, credential.expires_at, "expired", "refresh failed")

    def recover_session(self) -> CodexAccountStatus:
        """Attempt one explicit Luna refresh and preserve AUTH_REQUIRED semantics."""
        status = self.refresh_if_needed()
        if status.status is CodexAuthenticationStatus.EXPIRED:
            return CodexAccountStatus(CodexAuthenticationStatus.EXPIRED, status.account_identity, status.expires_at, status.refresh_state, "subscription session requires authentication")
        return status

    def logout(self) -> CodexAccountStatus:
        credential = self.store.load(CredentialKind.SUBSCRIPTION)
        if credential is not None and self.identity is not None:
            self.identity.revoke(credential)
        self.store.revoke(CredentialKind.SUBSCRIPTION)
        return self.status()


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
    cache_expires_at: float | None = None
    capacity_status: str = "known"

    def __post_init__(self) -> None:
        if self.mode is CodexExecutionMode.SUBSCRIPTION and self.estimated_cost_usd is not None:
            raise ValueError("subscription usage cannot expose a dollar cost")

    @property
    def available(self) -> bool:
        return self.capacity_status == "known" and self.authenticated and (self.expires_at is None or self.expires_at > time.time()) and (self.quota_remaining is None or self.quota_remaining > 0)

    @property
    def quota_health(self) -> float:
        if self.capacity_status != "known":
            return 0.0
        if self.mode is CodexExecutionMode.API:
            return 1.0 if self.authenticated else 0.0
        if self.quota_capacity in (None, 0):
            return 0.0
        return max(0.0, min(1.0, float(self.quota_remaining or 0.0) / self.quota_capacity))

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
            if raw.get("cache_expires_at") is not None and float(raw["cache_expires_at"]) <= time.time():
                return None
            values = {key: value for key, value in raw.items() if key in CodexUsage.__dataclass_fields__}
            values["mode"] = CodexExecutionMode(values["mode"])
            return CodexUsage(**values)
        except (OSError, ValueError, TypeError, KeyError):
            return None


class CodexUsageClient(Protocol):
    def fetch_usage(self, credential: CodexCredential, mode: CodexExecutionMode) -> Mapping[str, Any]: ...


class CodexUsageProvider:
    """Fetches Luna/provider usage and persists only normalized, non-secret data."""
    def __init__(self, client: CodexUsageClient, *, store: CodexUsageStore | None = None, ttl_seconds: float = 60.0, clock: Callable[[], float] = time.time) -> None:
        self.client, self.store, self.ttl_seconds, self.clock = client, store or CodexUsageStore(), ttl_seconds, clock

    def current(self, credential: CodexCredential | None, mode: CodexExecutionMode, *, force_refresh: bool = False) -> CodexUsage:
        if credential is None or not credential.authenticated or credential.expired:
            return CodexUsage(mode, False, account_identity=credential.account_identity if credential else "", capacity_status="unknown")
        if not force_refresh:
            cached = self.store.load(mode)
            if cached is not None:
                return cached
        try:
            raw = self.client.fetch_usage(credential, mode)
            usage = self._normalize(raw, credential, mode)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError):
            usage = CodexUsage(mode, True, account_identity=credential.account_identity, cache_expires_at=self.clock() + self.ttl_seconds, capacity_status="unknown")
        self.store.save(usage)
        return usage

    def _normalize(self, raw: Mapping[str, Any], credential: CodexCredential, mode: CodexExecutionMode) -> CodexUsage:
        now = self.clock()
        if mode is CodexExecutionMode.API:
            return CodexUsage(mode, True, credential.account_identity, int(raw.get("tokens_consumed", raw.get("total_tokens", 0)) or 0), float(raw["estimated_cost_usd"]) if raw.get("estimated_cost_usd") is not None else None, cache_expires_at=now + self.ttl_seconds)
        consumed = float(raw.get("quota_consumed", raw.get("consumed", 0)) or 0)
        capacity = float(raw["quota_capacity"]) if raw.get("quota_capacity") is not None else None
        remaining = float(raw["quota_remaining"]) if raw.get("quota_remaining") is not None else (capacity - consumed if capacity is not None else None)
        return CodexUsage(mode, True, credential.account_identity, quota_consumed=consumed, quota_remaining=remaining, quota_capacity=capacity, reset_at=float(raw["reset_at"]) if raw.get("reset_at") is not None else None, cache_expires_at=now + self.ttl_seconds)


@dataclass(frozen=True, slots=True)
class CodexAccountingRecord:
    provider: str
    mode: CodexExecutionMode
    execution_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    usd_cost: float | None = None
    quota_consumed: float = 0.0
    account_identity: str = ""
    reset_at: float | None = None


class CodexAccountingStore:
    def __init__(self, root: Path | None = None) -> None:
        self.path = (root or mana_home() / "usage").resolve() / "codex-accounting.jsonl"

    def record(self, record: CodexAccountingRecord) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record) | {"mode": record.mode.value}, sort_keys=True) + "\n")


@dataclass(frozen=True, slots=True)
class CodexFallbackDecision:
    selected_provider: str
    selected_mode: CodexExecutionMode
    reason: str
    attempted_modes: tuple[CodexExecutionMode, ...] = ()


@dataclass(frozen=True, slots=True)
class CodexPolicy:
    allowed_tasks: Mapping[CodexExecutionMode, frozenset[str]] = field(default_factory=lambda: {
        CodexExecutionMode.API: frozenset({"automation", "background_jobs", "ci", "coding", "refactor", "review"}),
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

    def __init__(self, backend: Any, *, mode: CodexExecutionMode, credentials: CodexCredentialStore | None = None, usage: CodexUsageStore | None = None, usage_provider: CodexUsageProvider | None = None, accounting: CodexAccountingStore | None = None) -> None:
        self.backend = backend
        self.mode = mode
        self.credentials = credentials or CodexCredentialStore()
        self.usage = usage or CodexUsageStore()
        self.usage_provider = usage_provider
        self.accounting = accounting or CodexAccountingStore()

    @property
    def credential(self) -> CodexCredential | None:
        return self.credentials.load(CredentialKind.API if self.mode is CodexExecutionMode.API else CredentialKind.SUBSCRIPTION)

    @property
    def resource_usage(self) -> CodexUsage | None:
        if self.usage_provider is not None:
            return self.usage_provider.current(self.credential, self.mode)
        return self.usage.load(self.mode)

    @property
    def available(self) -> bool:
        value = self.resource_usage
        return bool(value and value.available)

    async def execute(self, task: Any, workspace: Any) -> Any:
        result = await self.backend.execute(task, workspace)
        usage = result.token_usage or {}
        self.accounting.record(CodexAccountingRecord(
            "codex", self.mode, task.task_id,
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            usd_cost=(float(usage.get("usd_cost", self.resource_usage.estimated_cost_usd if self.resource_usage else 0.0)) if self.mode is CodexExecutionMode.API else None),
            quota_consumed=(float(usage.get("quota_consumed", self.resource_usage.quota_consumed if self.resource_usage else 0.0) or 0) if self.mode is CodexExecutionMode.SUBSCRIPTION else 0.0),
            account_identity=self.credential.account_identity if self.credential else "",
            reset_at=self.resource_usage.reset_at if self.resource_usage else None,
        ))
        return result

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


def choose_codex_resource(usages: Mapping[CodexExecutionMode, CodexUsage], task_type: str, policy: CodexPolicy = CodexPolicy()) -> CodexFallbackDecision:
    """Choose Codex's best resource and retain why fallback occurred."""
    subscription = usages.get(CodexExecutionMode.SUBSCRIPTION)
    if subscription is not None and policy.allows(CodexExecutionMode.SUBSCRIPTION, task_type) and not subscription.authenticated:
        raise CodexIdentityError("Codex subscription authentication is required")
    candidates = [
        (usage.quota_health + (0.35 if mode is CodexExecutionMode.SUBSCRIPTION and task_type in {"coding", "refactor", "review"} else 0.0), mode, usage)
        for mode, usage in usages.items()
        if policy.allows(mode, task_type)
    ]
    candidates.sort(key=lambda item: (-item[0], item[1].value))
    attempted: list[CodexExecutionMode] = []
    for _score, mode, usage in candidates:
        attempted.append(mode)
        if usage.available:
            reason = "subscription quota healthy" if mode is CodexExecutionMode.SUBSCRIPTION else "API resource available"
            if attempted and attempted[0] is not mode:
                reason = f"{mode.value} selected after {attempted[0].value} resource unavailable"
            return CodexFallbackDecision("codex", mode, reason, tuple(attempted))
    if any(usage.authenticated is False for usage in usages.values()):
        raise CodexIdentityError("Codex resource authentication is required")
    raise RuntimeError("No authenticated Codex resource satisfies the task policy and quota")


def codex_resource_score(provider: CodexProvider, task_type: str, policy: CodexPolicy = CodexPolicy()) -> RoutingResourceScore:
    usage = provider.resource_usage
    if not policy.allows(provider.mode, task_type):
        return RoutingResourceScore(False, reason=f"Codex {provider.mode.value} mode is disallowed for task type {task_type!r}")
    if usage is None:
        return RoutingResourceScore(False, resource_confidence=0.0, reason="Codex capacity is unknown")
    auth = 1.0 if usage.authenticated else 0.0
    reason = "Codex authentication is required" if not usage.authenticated else f"Codex {provider.mode.value} capacity is {usage.capacity_status}"
    return RoutingResourceScore(usage.available, usage.quota_health, usage.quota_health, auth, reason)


def codex_resource_availability(provider: CodexProvider, task_type: str, policy: CodexPolicy = CodexPolicy()) -> tuple[bool, str]:
    """Adapter for ModelRouter's resource gate."""
    if not policy.allows(provider.mode, task_type):
        return False, f"Codex {provider.mode.value} mode is disallowed for task type {task_type!r}"
    score = codex_resource_score(provider, task_type, policy)
    if not score.available:
        return False, score.reason
    return True, ""


__all__ = [
    "CodexAccountStatus", "CodexAccountingRecord", "CodexAccountingStore", "CodexAuthenticationService", "CodexFallbackDecision", "CodexIdentityError",
    "CodexAuthenticationStatus", "CodexCredential", "CodexCredentialStore", "CodexExecutionMode",
    "CodexIdentityClient", "CodexPolicy", "CodexProvider", "CodexUsage", "CodexUsageClient",
    "CodexUsageProvider", "CodexUsageStore", "CredentialKind", "choose_codex_mode", "choose_codex_resource",
    "codex_resource_availability", "codex_resource_score",
]
